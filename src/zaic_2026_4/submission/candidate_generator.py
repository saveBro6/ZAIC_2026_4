"""Candidate diversity generator with seed variation, schema permutation, and iterative self-correction."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from zaic_2026_4.constants import (
    DEFAULT_CANDIDATE_SEEDS,
    DEFAULT_EXECUTION_MAX_RETRIES,
    DEFAULT_SQL_MAX_RETRIES,
    DEFAULT_SQL_TIMEOUT_SECONDS,
)
from zaic_2026_4.submission.models import (
    VALID_DATAMATE_TYPES,
    CandidateQuery,
    SQLGenerationOutput,
    ValidationResult,
)
from zaic_2026_4.submission.prompt_template import (
    build_sql_correction_prompt,
    build_sql_generation_prompt,
    format_permuted_schema_xml,
    permute_schema,
)
from zaic_2026_4.submission.sql_validator import (
    SQLValidator,
    parse_structured_sql_output,
)
from zaic_2026_4.submission.voting_executor import normalize_cell

logger = logging.getLogger(__name__)


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
        """Initialize candidate generator.

        Args:
            llm_engine: Offline inference engine supporting .generate(prompts, seed=...).
            validator: SQLValidator instance (defaults to new instance).
            seeds: Randomization seeds for candidate diversity (default: 42, 100, 2026).
            sql_max_retries: Maximum static AST correction attempts per candidate (default: 2).
            execution_max_retries: Maximum runtime execution correction attempts per candidate (default: 2).
            max_retries: Backward-compatible alias for sql_max_retries.
            rulebook_mode: Injection mode for split rulebook ('hybrid', 'full', 'retrieval').
            db_path: Optional path to business SQLite database for dry-run verification.
            timeout_seconds: Timeout for dry-run SQLite query execution.
        """
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

    @staticmethod
    def _is_suspicious_zero_result(raw_rows: list[Any] | None) -> bool:
        """Detect whether query execution returned empty rows, single NULL, or single zero."""
        if not raw_rows:
            return True
        if len(raw_rows) == 1:
            row = raw_rows[0]
            if len(row) == 1:
                val = row[0]
                if val is None or val == 0 or val == 0.0 or val == "0":
                    return True
        return False

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
        """Generate a single candidate SQL query with schema permutation, static validation, and dry-run execution."""
        target_db = Path(db_path).resolve() if db_path else self.db_path

        # 1. Permute reduced schema order using seed
        permuted_keys = permute_schema(reduced_schema, seed)
        schema_xml = format_permuted_schema_xml(permuted_keys, processed_metadata)

        # 2. Build generation prompt
        prompt = build_sql_generation_prompt(
            question=question,
            as_of_date=as_of_date,
            schema_keys=permuted_keys,
            processed_metadata=processed_metadata,
            rulebook_text=rulebook_text,
            vector_indexer=vector_indexer,
            rulebook_mode=self.rulebook_mode,
            grounded_entities=grounded_literals,
        )

        # 3. Initial generation
        gen_kwargs: dict[str, Any] = {
            "temperature": 0.7,
            "max_tokens": 512,
            "seed": seed,
        }
        if self.structured_outputs is not None:
            gen_kwargs["structured_outputs"] = self.structured_outputs

        raw_output = self.llm_engine.generate(prompt, **gen_kwargs)
        raw_str = raw_output if isinstance(raw_output, str) else raw_output[0]
        current_sql, current_type, parse_issues = parse_structured_sql_output(raw_str)
        initial_sql = current_sql

        # 4. Static Validation & Self-Correction loop (up to sql_max_retries)
        if current_type in ("unanswerable", "abstain") and not current_sql:
            v_res = ValidationResult(is_valid=True, issues=[])
        else:
            v_res = self.validator.validate(current_sql)

        all_issues = list(parse_issues) + list(v_res.issues)
        sql_retries = 0

        while (not v_res.is_valid or parse_issues) and sql_retries < self.sql_max_retries:
            sql_retries += 1
            logger.debug(
                "Candidate %d (seed %d) static retry %d due to issues: %s",
                candidate_id,
                seed,
                sql_retries,
                all_issues,
            )
            corr_prompt = build_sql_correction_prompt(
                question=question,
                as_of_date=as_of_date,
                faulty_sql=current_sql,
                validation_issues=all_issues,
                schema_xml=schema_xml,
                previous_type=current_type,
                grounded_entities=grounded_literals,
            )
            corr_kwargs: dict[str, Any] = {
                "temperature": 0.2,
                "max_tokens": 512,
                "seed": seed + sql_retries,
            }
            if self.structured_outputs is not None:
                corr_kwargs["structured_outputs"] = self.structured_outputs

            corr_output = self.llm_engine.generate(corr_prompt, **corr_kwargs)
            corr_str = corr_output if isinstance(corr_output, str) else corr_output[0]
            current_sql, current_type, parse_issues = parse_structured_sql_output(corr_str, default_type=current_type)
            if current_type in ("unanswerable", "abstain") and not current_sql:
                v_res = ValidationResult(is_valid=True, issues=[])
                all_issues = []
                break
            v_res = self.validator.validate(current_sql)
            all_issues = list(parse_issues) + list(v_res.issues)

        # 5. Dry-run Execution & Runtime Self-Correction loop (up to execution_max_retries)
        execution_retries = 0
        exec_success = False
        exec_error: str | None = None
        raw_rows: list[Any] | None = None
        r_set: set[tuple[Any, ...]] | None = None

        if current_type in ("unanswerable", "abstain") and not current_sql:
            exec_success = True
        elif v_res.is_valid and target_db is not None and target_db.is_file():
            exec_success, exec_error, raw_rows, r_set = self._dry_run_query(current_sql, target_db)
            is_suspicious = exec_success and self._is_suspicious_zero_result(raw_rows)

            while (not exec_success or is_suspicious) and execution_retries < self.execution_max_retries:
                execution_retries += 1
                if not exec_success:
                    issues = [f"Lỗi thực thi SQLite (Runtime Error): {exec_error}"]
                else:
                    issues = [
                        (
                            "Truy vấn thực thi thành công nhưng trả về 0 dòng khớp (hoặc kết quả là 0 / NULL). "
                            "Vui lòng đối chiếu schema để kiểm tra lại: định dạng ngày tháng của cột thời gian trong schema/mẫu dữ liệu, các điều kiện so sánh chuỗi trong WHERE, và điều kiện JOIN."
                        )
                    ]

                logger.debug(
                    "Candidate %d (seed %d) execution retry %d due to: %s",
                    candidate_id,
                    seed,
                    execution_retries,
                    issues,
                )
                corr_prompt = build_sql_correction_prompt(
                    question=question,
                    as_of_date=as_of_date,
                    faulty_sql=current_sql,
                    validation_issues=issues,
                    schema_xml=schema_xml,
                    previous_type=current_type,
                    grounded_entities=grounded_literals,
                )
                exec_retry_kwargs: dict[str, Any] = {
                    "temperature": 0.2,
                    "max_tokens": 512,
                    "seed": seed + self.sql_max_retries + execution_retries,
                }
                if self.structured_outputs is not None:
                    exec_retry_kwargs["structured_outputs"] = self.structured_outputs

                corr_output = self.llm_engine.generate(corr_prompt, **exec_retry_kwargs)
                new_str = corr_output if isinstance(corr_output, str) else corr_output[0]
                new_sql, new_type, new_parse_issues = parse_structured_sql_output(new_str, default_type=current_type)

                if new_type in ("unanswerable", "abstain") and not new_sql:
                    current_sql = ""
                    current_type = new_type
                    v_res = ValidationResult(is_valid=True, issues=[])
                    exec_success = True
                    exec_error = None
                    raw_rows = None
                    r_set = None
                    break

                new_v_res = self.validator.validate(new_sql)
                if new_v_res.is_valid and not new_parse_issues:
                    current_sql = new_sql
                    current_type = new_type
                    v_res = new_v_res
                    exec_success, exec_error, raw_rows, r_set = self._dry_run_query(current_sql, target_db)
                    is_suspicious = exec_success and self._is_suspicious_zero_result(raw_rows)
                    if exec_success and not is_suspicious:
                        # Successfully resolved with positive non-zero result: stop immediately
                        break
                else:
                    exec_success = False
                    comb_errs = list(new_parse_issues) + list(new_v_res.issues)
                    exec_error = f"Lỗi cú pháp/schema SQL sau khi sửa: {'; '.join(comb_errs)}"

        # Final safety check on predicted_type to guarantee 100% adherence to problem contract
        if current_type not in VALID_DATAMATE_TYPES:
            current_type = "abstain"

        return CandidateQuery(
            candidate_id=candidate_id,
            seed=seed,
            schema_permutation=permuted_keys,
            initial_sql=initial_sql,
            final_sql=current_sql,
            predicted_type=current_type,
            is_valid_syntax=v_res.is_valid,
            construction_issues=v_res.issues,
            sql_retry_count=sql_retries,
            execution_retry_count=execution_retries,
            total_retry_count={
                "sql": sql_retries,
                "execution": execution_retries,
            },
            execution_success=exec_success,
            execution_error=exec_error,
            raw_result=raw_rows,
            result_set=r_set,
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
        for idx, seed in enumerate(self.seeds):
            cand = self.generate_candidate(
                candidate_id=idx,
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


__all__ = ["CandidateGenerator"]
