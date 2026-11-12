"""Candidate diversity generator with seed variation, schema permutation, and iterative self-correction."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlglot

from zaic_2026_4.constants import (
    DEFAULT_CANDIDATE_SEEDS,
    DEFAULT_EXECUTION_MAX_RETRIES,
    DEFAULT_SQL_MAX_RETRIES,
    DEFAULT_SQL_MAX_TOKENS,
    DEFAULT_SQL_TIMEOUT_SECONDS,
    DEFAULT_THINKING_TOKEN_BUDGET,
)
from zaic_2026_4.submission.models import (
    VALID_DATAMATE_TYPES,
    CandidateQuery,
    SQLGenerationOutput,
)
from zaic_2026_4.submission.prompt_template import (
    RepairContext,
    _build_prompt_prefix,
    build_sql_generation_prompt,
    format_permuted_schema_xml,
    format_rulebook_context,
    permute_schema,
)
from zaic_2026_4.submission.rules import ValidationIssue
from zaic_2026_4.submission.sql_validator import (
    SQLValidator,
    parse_structured_sql_output,
)
from zaic_2026_4.submission.voting_executor import normalize_cell

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Attempt:
    """Represents a single query generation or repair attempt within a candidate turn."""

    sql: str
    target_type: str
    executed: bool
    shape_ok: bool
    has_non_zero_evidence: bool
    has_evidence: bool
    issues: tuple[ValidationIssue | str, ...]
    raw_rows: list[Any] | None = None
    result_set: set[tuple[Any, ...]] | None = None
    raw_str: str = ""


def rank(a: Attempt) -> tuple[bool, bool, bool, bool, bool, int]:
    """Rank attempt by evidence quality and validity rather than raw row count.

    Priority:
    1. Executed successfully on SQLite
    2. Free of blocking issues
    3. Correct result shape
    4. Has non-zero evidence (not empty, not NULL, not single zero)
    5. Has evidence (valid non-empty rows, including e.g. legitimate (0,) from COUNT)
    6. Fewer warnings is better
    """
    has_blocking = any(getattr(i, "severity", "blocking") == "blocking" for i in a.issues)
    return (
        a.executed,
        not has_blocking,
        a.shape_ok,
        a.has_non_zero_evidence,
        a.has_evidence,
        -len(a.issues),
    )


def canonicalize_sql(sql: str) -> str:
    """Normalize SQL string for canonical stagnation detection."""
    clean = sql.strip()
    if not clean:
        return ""
    try:
        parsed = sqlglot.parse_one(clean, read="sqlite")
        return parsed.sql(dialect="sqlite", normalize=True)
    except Exception:  # noqa: BLE001
        return " ".join(clean.split())


class CandidateGenerator:
    """Generates 3 diverse candidate SQL queries with static validation and dry-run execution self-correction."""

    def __init__(
        self,
        llm_engine: Any,
        validator: SQLValidator | None = None,
        seeds: tuple[int, ...] = DEFAULT_CANDIDATE_SEEDS,
        sql_max_retries: int = DEFAULT_SQL_MAX_RETRIES,
        execution_max_retries: int = DEFAULT_EXECUTION_MAX_RETRIES,
        max_retries: int | None = None,
        rulebook_mode: str = "hybrid",
        db_path: Path | str | None = None,
        timeout_seconds: float = DEFAULT_SQL_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize candidate generator."""
        self.llm_engine = llm_engine
        self.validator = validator or SQLValidator()
        self.seeds = seeds
        self.sql_max_retries = max_retries if max_retries is not None else sql_max_retries
        self.execution_max_retries = execution_max_retries
        self.rulebook_mode = rulebook_mode
        self.db_path = Path(db_path).resolve() if db_path else None
        self.timeout_seconds = timeout_seconds

        try:
            from vllm.sampling_params import StructuredOutputsParams

            self.structured_outputs: Any | None = StructuredOutputsParams(
                json=SQLGenerationOutput.model_json_schema()
            )
        except Exception:  # noqa: BLE001
            self.structured_outputs = None

    def _dry_run_query(
        self,
        sql: str,
        db_path: Path | None,
    ) -> tuple[bool, str | None, list[Any] | None, set[tuple[Any, ...]] | None]:
        """Execute query in read-only mode against SQLite database to verify runtime execution."""
        if db_path is None or not db_path.is_file():
            return False, "Database path not configured or file not found.", None, None

        uri = f"file:{db_path.as_posix()}?mode=ro"
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=self.timeout_seconds)
            cursor = conn.cursor()
            cursor.execute(sql)
            raw_rows = cursor.fetchall()
            normalized_rows = [
                tuple(normalize_cell(cell) for cell in row)
                for row in raw_rows
            ]
            result_set = set(normalized_rows)
            return True, None, raw_rows, result_set
        except sqlite3.Error as exc:
            return False, f"SQLite error: {exc}", None, None
        except Exception as exc:  # noqa: BLE001
            return False, f"Execution error: {exc}", None, None
        finally:
            if conn is not None:
                conn.close()

    def _build_repair_prompt(
        self,
        prompt_kwargs: dict[str, Any],
        prompt_prefix: str,
        candidate_id: int,
        diagnostics: Sequence[ValidationIssue | str],
        previous_sql: str,
        previous_type: str,
        accepted_sql: str,
        last_raw_output: str,
        previous_attempts: tuple[tuple[str, tuple[str, ...]], ...] = (),
    ) -> str:
        """Build a repair prompt that shares the generation prefix, warning when they diverge."""
        repair = RepairContext(
            diagnostics=diagnostics,
            previous_sql=previous_sql,
            previous_type=previous_type,
            accepted_sql=accepted_sql,
            last_output=last_raw_output,
            previous_attempts=previous_attempts,
        )
        corr_prompt = build_sql_generation_prompt(**prompt_kwargs, repair=repair)
        if not corr_prompt.startswith(prompt_prefix):
            logger.warning(
                "Candidate %d repair prompt diverged from generation prefix; caching disabled.",
                candidate_id,
            )
        return corr_prompt

    def generate_candidate(
        self,
        candidate_id: int,
        seed: int,
        question: str,
        as_of_date: str,
        reduced_schema: set[str],
        processed_metadata: dict[str, Any],
        rulebook_text: str = "",
        vector_indexer: Any | None = None,
        db_path: Path | str | None = None,
        grounded_literals: list[dict[str, Any]] | None = None,
    ) -> CandidateQuery:
        """Generate a single candidate query using the unified self-correction state machine."""
        target_db = Path(db_path).resolve() if db_path else self.db_path

        # 1. Deterministic schema permutation
        permuted_keys = permute_schema(reduced_schema, seed=seed)
        schema_xml = format_permuted_schema_xml(permuted_keys, processed_metadata)

        # 2. Dynamic rulebook context
        rulebook_xml = format_rulebook_context(
            rulebook_text=rulebook_text,
            vector_indexer=vector_indexer,
            question=question,
            mode=self.rulebook_mode,
        )

        prompt_kwargs: dict[str, Any] = {
            "question": question,
            "as_of_date": as_of_date,
            "schema_keys": permuted_keys,
            "processed_metadata": processed_metadata,
            "rulebook_text": rulebook_text,
            "vector_indexer": vector_indexer,
            "rulebook_mode": self.rulebook_mode,
            "include_few_shot": True,
            "grounded_entities": grounded_literals,
            "schema_xml_override": schema_xml,
            "rulebook_xml_override": rulebook_xml,
        }

        # 3. Initial generation prompt
        prompt_prefix = _build_prompt_prefix(
            question=question,
            as_of_date=as_of_date,
            schema_xml=schema_xml,
            rulebook_xml=rulebook_xml,
            grounded_entities=grounded_literals,
            include_few_shot=True,
        )
        prompt = f"{prompt_prefix}\n\n<output>"

        gen_kwargs: dict[str, Any] = {
            "temperature": 0.7,
            "max_tokens": DEFAULT_SQL_MAX_TOKENS,
            "thinking_token_budget": DEFAULT_THINKING_TOKEN_BUDGET,
            "seed": seed,
        }
        if self.structured_outputs is not None:
            gen_kwargs["structured_outputs"] = self.structured_outputs

        raw_output = self.llm_engine.generate(prompt, **gen_kwargs)
        raw_str = raw_output if isinstance(raw_output, str) else raw_output[0]

        # State machine variables
        history: list[Attempt] = []
        best: Attempt | None = None
        sql_retries = 0
        execution_retries = 0
        max_total_retries = self.sql_max_retries + self.execution_max_retries
        stop_reason = "clean"
        curr_raw_str = raw_str
        initial_sql = ""
        current_phase = "sql"

        while True:
            curr_sql, curr_type, parse_issues = parse_structured_sql_output(
                curr_raw_str, default_type="number"
            )
            if not initial_sql:
                initial_sql = curr_sql

            # Check deliberate abstain
            if curr_type in ("unanswerable", "abstain") and not curr_sql.strip():
                stop_reason = "deliberate_abstain"
                best = Attempt(
                    sql="",
                    target_type=curr_type,
                    executed=True,
                    shape_ok=True,
                    has_non_zero_evidence=False,
                    has_evidence=False,
                    issues=(),
                    raw_str=curr_raw_str,
                )
                break

            # Validate static AST and schema rules
            v_res = self.validator.validate(curr_sql, target_type=curr_type, metadata=processed_metadata)
            all_issues: list[ValidationIssue | str] = list(parse_issues) + list(v_res.issues)
            has_blocking = any(getattr(i, "severity", "blocking") == "blocking" for i in all_issues)

            executed = False
            raw_rows = None
            r_set = None
            shape_ok = not has_blocking
            has_ev = False
            has_nz_ev = False

            if not has_blocking:
                if target_db is not None and target_db.is_file():
                    current_phase = "execution"
                    exec_success, exec_error, raw_rows, r_set = self._dry_run_query(curr_sql, target_db)
                    if not exec_success:
                        all_issues.append(
                            ValidationIssue(
                                code="EXECUTION_ERROR",
                                message=f"Lỗi thực thi SQLite (Runtime Error): {exec_error}",
                                severity="blocking",
                            )
                        )
                        executed = False
                        shape_ok = False
                    else:
                        executed = True
                        # Shape check on executed rows
                        if curr_type in ("number", "percent", "string", "date", "boolean"):
                            if raw_rows and (len(raw_rows) > 1 or len(raw_rows[0]) > 1):
                                all_issues.append(
                                    ValidationIssue(
                                        code="RESULT_SHAPE_MISMATCH",
                                        message=f"Target_type là '{curr_type}' nhưng kết quả trả về {len(raw_rows)} dòng và {len(raw_rows[0])} cột.",
                                        severity="blocking",
                                    )
                                )
                                shape_ok = False
                            else:
                                shape_ok = True
                        elif curr_type == "list":
                            if raw_rows and len(raw_rows[0]) > 1:
                                all_issues.append(
                                    ValidationIssue(
                                        code="RESULT_SHAPE_MISMATCH",
                                        message=f"Target_type là 'list' nhưng kết quả trả về {len(raw_rows[0])} cột.",
                                        severity="blocking",
                                    )
                                )
                                shape_ok = False
                            else:
                                shape_ok = True
                        elif curr_type == "table":
                            if raw_rows and len(raw_rows[0]) <= 1:
                                all_issues.append(
                                    ValidationIssue(
                                        code="RESULT_SHAPE_MISMATCH",
                                        message="Target_type là 'table' nhưng kết quả chỉ có 1 cột.",
                                        severity="blocking",
                                    )
                                )
                                shape_ok = False
                            else:
                                shape_ok = True

                        # Evidence calculation
                        has_ev = bool(raw_rows) and not all(all(cell is None for cell in row) for row in raw_rows)
                        if has_ev:
                            if len(raw_rows) == 1 and len(raw_rows[0]) == 1 and raw_rows[0][0] in (0, 0.0, "0", None):
                                has_nz_ev = False
                            else:
                                has_nz_ev = True
                else:
                    # No target database for dry-run
                    executed = True
                    shape_ok = True
                    has_ev = True
                    has_nz_ev = True

            curr_attempt = Attempt(
                sql=curr_sql,
                target_type=curr_type,
                executed=executed,
                shape_ok=shape_ok,
                has_non_zero_evidence=has_nz_ev,
                has_evidence=has_ev,
                issues=tuple(all_issues),
                raw_rows=raw_rows,
                result_set=r_set,
                raw_str=curr_raw_str,
            )
            history.append(curr_attempt)

            # Update best attempt: only replace if current attempt is strictly better
            if best is None or rank(curr_attempt) > rank(best):
                best = curr_attempt

            # Stop conditions
            curr_blocking = [i for i in all_issues if getattr(i, "severity", "blocking") == "blocking"]

            # 1. Clean stop with positive evidence
            if curr_attempt.executed and not curr_blocking and curr_attempt.shape_ok and curr_attempt.has_non_zero_evidence:
                stop_reason = "clean"
                break

            # 2. Zero / NULL / Empty result check
            if curr_attempt.executed and not curr_blocking and curr_attempt.shape_ok and not curr_attempt.has_non_zero_evidence:
                has_semantic_warnings = any(
                    getattr(i, "code", "") in ("LITERAL_NOT_IN_ENUM", "DATE_FORMAT_MISMATCH", "UNKNOWN_QUALIFIER", "MONEY_SCALE_RISK")
                    for i in all_issues
                )
                missing_grounded = False
                if grounded_literals:
                    for g in grounded_literals:
                        val = g.get("canonical_value", "")
                        if val and val.lower() not in curr_sql.lower():
                            missing_grounded = True
                            break

                if not has_semantic_warnings and not missing_grounded:
                    # Legitimate zero result without issues: stop immediately, don't waste LLM calls
                    stop_reason = "legitimate_zero"
                    break
                else:
                    all_issues.append(
                        ValidationIssue(
                            code="UNRESOLVED_ZERO",
                            message="Truy vấn trả về 0/NULL và có cảnh báo ngữ nghĩa/literal. Kiểm tra lại điều kiện WHERE.",
                            severity="warning",
                        )
                    )

            # 3. Stagnation check
            curr_canon = canonicalize_sql(curr_sql)
            prev_canons = [canonicalize_sql(a.sql) for a in history[:-1]]
            curr_blocking_codes = frozenset(getattr(i, "code", "UNKNOWN") for i in curr_blocking)

            is_canon_stagnation = curr_canon in prev_canons
            is_code_stagnation = (
                len(history) >= 3
                and bool(curr_blocking_codes)
                and all(
                    frozenset(getattr(i, "code", "UNKNOWN") for i in a.issues if getattr(i, "severity", "blocking") == "blocking") == curr_blocking_codes
                    for a in history[-3:]
                )
            )

            if is_canon_stagnation or is_code_stagnation:
                stop_reason = "stagnation"
                break

            # 4. Budget check
            if current_phase == "execution":
                if execution_retries >= self.execution_max_retries:
                    stop_reason = "budget_exhausted"
                    break
            else:
                if sql_retries >= self.sql_max_retries:
                    stop_reason = "budget_exhausted"
                    break

            if (sql_retries + execution_retries) >= max_total_retries:
                stop_reason = "budget_exhausted"
                break

            # 5. Build repair prompt for next turn
            previous_attempts_context = tuple(
                (canonicalize_sql(a.sql), tuple(getattr(i, "code", "UNKNOWN") for i in a.issues))
                for a in history[-2:]
            )
            accepted_sql = best.sql if best is not None else curr_sql
            prev_sql = curr_sql
            prev_type = curr_type

            corr_prompt = self._build_repair_prompt(
                prompt_kwargs=prompt_kwargs,
                prompt_prefix=prompt_prefix,
                candidate_id=candidate_id,
                diagnostics=all_issues,
                previous_sql=prev_sql,
                previous_type=prev_type,
                accepted_sql=accepted_sql,
                last_raw_output=curr_raw_str,
                previous_attempts=previous_attempts_context,
            )
            corr_kwargs: dict[str, Any] = {
                "temperature": 0.2,
                "max_tokens": DEFAULT_SQL_MAX_TOKENS,
                "thinking_token_budget": DEFAULT_THINKING_TOKEN_BUDGET,
                "seed": seed + sql_retries + execution_retries,
            }
            if self.structured_outputs is not None:
                corr_kwargs["structured_outputs"] = self.structured_outputs

            corr_output = self.llm_engine.generate(corr_prompt, **corr_kwargs)
            if current_phase == "execution":
                execution_retries += 1
            else:
                sql_retries += 1
            curr_raw_str = corr_output if isinstance(corr_output, str) else corr_output[0]

        if best is None:
            best = history[-1]

        final_sql = best.sql
        predicted_type = best.target_type
        if predicted_type not in VALID_DATAMATE_TYPES:
            predicted_type = "abstain"

        is_valid_syntax = not any(getattr(i, "severity", "blocking") == "blocking" for i in best.issues)
        is_deliberate_abstain = (predicted_type in ("unanswerable", "abstain") and not final_sql.strip())
        is_unresolved_zero = (stop_reason in ("budget_exhausted", "stagnation") and not best.has_non_zero_evidence)

        return CandidateQuery(
            candidate_id=candidate_id,
            seed=seed,
            schema_permutation=permuted_keys,
            initial_sql=initial_sql,
            final_sql=final_sql,
            predicted_type=predicted_type,
            is_valid_syntax=is_valid_syntax,
            construction_issues=[str(x) for x in best.issues],
            sql_retry_count=sql_retries,
            execution_retry_count=execution_retries,
            total_retry_count={
                "sql": sql_retries,
                "execution": execution_retries,
            },
            execution_success=best.executed,
            execution_error=next((str(i) for i in best.issues if getattr(i, "code", "") == "EXECUTION_ERROR"), None),
            raw_result=best.raw_rows,
            result_set=best.result_set,
            deliberate_abstain=is_deliberate_abstain,
            unresolved_zero=is_unresolved_zero,
            repair_stop_reason=stop_reason,
        )

    def generate_all_candidates(
        self,
        question: str,
        as_of_date: str,
        reduced_schema: set[str],
        processed_metadata: dict[str, Any],
        rulebook_text: str = "",
        vector_indexer: Any | None = None,
        db_path: Path | str | None = None,
        grounded_literals: list[dict[str, Any]] | None = None,
    ) -> list[CandidateQuery]:
        """Generate all candidate queries across configured seeds."""
        candidates: list[CandidateQuery] = []
        for i, seed in enumerate(self.seeds):
            cand = self.generate_candidate(
                candidate_id=i,
                seed=seed,
                question=question,
                as_of_date=as_of_date,
                reduced_schema=reduced_schema,
                processed_metadata=processed_metadata,
                rulebook_text=rulebook_text,
                vector_indexer=vector_indexer,
                db_path=db_path,
                grounded_literals=grounded_literals,
            )
            candidates.append(cand)
        return candidates


__all__ = [
    "Attempt",
    "CandidateGenerator",
    "canonicalize_sql",
    "rank",
]
