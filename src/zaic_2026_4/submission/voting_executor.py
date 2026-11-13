"""SQLite query execution, set conversion, and majority voting logic."""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Any

from zaic_2026_4.constants import DEFAULT_SQL_TIMEOUT_SECONDS
from zaic_2026_4.submission.models import CandidateQuery
from zaic_2026_4.submission.result_formatter import (
    ResultFormatter,
    normalize_contract_string,
)

logger = logging.getLogger(__name__)


def normalize_cell(val: Any) -> Any:
    """Normalize SQLite cell values for stable set equality comparison."""
    if val is None:
        return None
    if isinstance(val, float):
        return round(val, 4)
    if isinstance(val, str):
        return val.strip()
    return val


def canonical_cell(val: Any) -> Any:
    """Canonicalize a scalar cell value according to scorer matching semantics."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        if math.isfinite(val):
            if abs(val - round(val)) <= 1e-6:
                return round(val)
            return round(float(val), 4)
        return str(val)
    if isinstance(val, str):
        return normalize_contract_string(val)
    if isinstance(val, (bytes, bytearray)):
        return None
    return val


def summarize_result_set(res_set: set[tuple[Any, ...]] | frozenset[tuple[Any, ...]] | None) -> str:
    """Create concise string summary of a result set for diagnostic logging."""
    if res_set is None:
        return "None"
    items = sorted([str(r) for r in res_set], key=str)
    if not items:
        return "empty set []"
    if len(items) <= 3:
        return f"{{{', '.join(items)}}}"
    return f"{{{items[0]}, {items[1]}, ... ({len(items)} items)}}"


def is_unanswerable_claim(cand: CandidateQuery) -> bool:
    """True when a candidate explicitly claims the question cannot be answered from the data."""
    return not cand.final_sql.strip() and cand.predicted_type == "unanswerable"


def is_abstain_claim(cand: CandidateQuery) -> bool:
    """True when a candidate declines to answer without asserting unanswerability."""
    return not cand.final_sql.strip() and cand.predicted_type == "abstain"


def has_evidence(res_set: set[tuple[Any, ...]] | frozenset[tuple[Any, ...]] | None) -> bool:
    """Check whether result set contains at least one non-None cell in at least one row."""
    if res_set is None or len(res_set) == 0:
        return False
    return not all(all(cell is None for cell in row) for row in res_set)


def answer_signature(cand: CandidateQuery) -> tuple[str, str] | None:
    """Canonical (answer_type, canonical_value) key of the answer this candidate would submit."""
    if not cand.execution_success or not has_evidence(cand.result_set):
        return None

    # Never synthesize arbitrary row order from an unordered set; require raw_result
    if cand.raw_result is None or not cand.raw_result:
        return None

    raw_rows = cand.raw_result

    try:
        ans_type, ans_val = ResultFormatter.format_result(
            raw_rows,
            cand.predicted_type,
            raw_columns=cand.result_columns,
            sql=cand.final_sql,
        )
    except Exception:  # noqa: BLE001
        return None

    if ans_type in ("abstain", "unanswerable") or ans_val is None:
        return None

    try:
        if ans_type == "table" and isinstance(ans_val, dict):
            raw_cols = ans_val.get("columns", [])
            norm_cols = [normalize_contract_string(c) for c in raw_cols]
            rows_data = ans_val.get("rows", [])

            # If column names are unique, align rows to alphabetically sorted normalized column names
            if len(set(norm_cols)) == len(norm_cols) and len(norm_cols) > 0:
                col_order = sorted(range(len(norm_cols)), key=lambda i: norm_cols[i])
                cols = [norm_cols[i] for i in col_order]
                aligned_rows = [
                    [canonical_cell(r[i]) if i < len(r) else None for i in col_order]
                    for r in rows_data
                    if isinstance(r, (list, tuple))
                ]
            else:
                cols = norm_cols
                aligned_rows = [
                    [canonical_cell(cell) for cell in r]
                    for r in rows_data
                    if isinstance(r, (list, tuple))
                ]

            rows = sorted(json.dumps(r, ensure_ascii=False) for r in aligned_rows)
            canon_str = json.dumps({"columns": cols, "rows": rows}, ensure_ascii=False, sort_keys=True)

        elif ans_type == "list" and isinstance(ans_val, list):
            # Preserves list element position order, normalizes each element per scorer rules
            canon_list = [canonical_cell(item) for item in ans_val]
            canon_str = json.dumps(canon_list, ensure_ascii=False)

        elif ans_type == "string":
            canon_str = json.dumps(normalize_contract_string(ans_val), ensure_ascii=False)

        elif ans_type == "percent":
            # Per scorer match_percent: rounded to 2 decimal places
            canon_str = json.dumps(round(float(ans_val), 2))

        elif ans_type == "number":
            canon_str = json.dumps(canonical_cell(ans_val))

        elif ans_type == "date":
            canon_str = json.dumps(str(ans_val).strip())

        elif ans_type == "boolean":
            canon_str = json.dumps(bool(ans_val))

        else:
            canon_str = json.dumps(ans_val, ensure_ascii=False, sort_keys=True)

        return ans_type, canon_str

    except (TypeError, ValueError, OverflowError):
        return None


class VotingExecutor:
    """Executes candidate queries against SQLite database and selects final answer via majority voting."""

    def __init__(
        self,
        db_path: Path | str,
        timeout_seconds: float = DEFAULT_SQL_TIMEOUT_SECONDS,
        min_agreement: float = 0.5,
    ) -> None:
        """Initialize VotingExecutor."""
        self.db_path = Path(db_path).resolve()
        self.timeout_seconds = timeout_seconds
        self.min_agreement = min_agreement

    def execute_query(
        self,
        sql: str,
    ) -> tuple[bool, str | None, list[Any] | None, set[tuple[Any, ...]] | None, list[str]]:
        """Execute a single query in read-only mode and convert results to a set of tuples and column names."""
        if not self.db_path.is_file():
            return False, f"Database file not found: {self.db_path}", None, None, []

        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=self.timeout_seconds)
            cursor = conn.cursor()
            cursor.execute(sql)
            raw_rows = cursor.fetchall()
            columns = [str(col[0]) for col in (cursor.description or ())]

            normalized_rows = [
                tuple(normalize_cell(cell) for cell in row)
                for row in raw_rows
            ]
            result_set = set(normalized_rows)
            return True, None, raw_rows, result_set, columns

        except sqlite3.Error as exc:
            return False, f"SQLite error: {exc}", None, None, []
        except Exception as exc:  # noqa: BLE001
            return False, f"Execution error: {exc}", None, None, []
        finally:
            if conn is not None:
                conn.close()

    def execute_candidates(self, candidates: list[CandidateQuery]) -> None:
        """Execute all valid candidates and record execution outcomes in-place."""
        for cand in candidates:
            # Handle deliberate abstain / unanswerable
            if cand.deliberate_abstain or (
                cand.predicted_type in ("unanswerable", "abstain") and not cand.final_sql.strip()
            ):
                cand.execution_success = True
                cand.raw_result = None
                cand.result_set = None
                cand.result_columns = []
                continue

            if not cand.is_valid_syntax:
                cand.execution_success = False
                cand.execution_error = "Skipped execution due to static syntax/construction errors."
                continue

            # Reuse cached outcome if already executed successfully during dry-run
            if cand.execution_success and cand.result_set is not None and cand.raw_result is not None:
                continue

            success, err, raw_rows, r_set, cols = self.execute_query(cand.final_sql)
            cand.execution_success = success
            cand.execution_error = err
            cand.raw_result = raw_rows
            cand.result_set = r_set
            cand.result_columns = cols

    def vote(
        self,
        candidates: list[CandidateQuery],
        unanswerable_proof: bool = True,
    ) -> tuple[CandidateQuery | None, str, str]:
        """Perform majority voting among executed candidates with canonical answer signatures and agreement gate.

        A single candidate claiming "unanswerable" is not evidence: two candidates may have
        failed mechanically while one gave up. The verdict is only issued on a strict
        majority of the candidates, with no competing evidence and no coverage gap that
        could have removed a required table from the linked schema.

        Args:
            candidates: Candidate queries with execution results attached.
            unanswerable_proof: False when schema coverage reported unresolved gaps, which
                makes an unanswerable verdict untrustworthy.
        """
        total_candidates = len(candidates)
        unans_cands = [c for c in candidates if is_unanswerable_claim(c)]
        abstain_cands = [c for c in candidates if is_abstain_claim(c)]
        evidence_cands = [
            c for c in candidates
            if c.execution_success and has_evidence(c.result_set)
        ]

        if unans_cands and not evidence_cands:
            has_majority = len(unans_cands) * 2 > total_candidates
            if has_majority and unanswerable_proof and not abstain_cands:
                winner = unans_cands[0]
                reason = (
                    f"unanswerable: {len(unans_cands)}/{total_candidates} candidates agree that the "
                    "released data cannot answer this question."
                )
                return winner, "unanswerable_consensus", reason

            blocking = []
            if not has_majority:
                blocking.append(f"only {len(unans_cands)}/{total_candidates} candidates claimed unanswerable")
            if not unanswerable_proof:
                blocking.append("linked schema has unresolved coverage gaps")
            if abstain_cands:
                blocking.append(f"{len(abstain_cands)} candidate(s) could not commit to an answer")
            reason = (
                "abstain_unverified_unanswerable: Refusing to submit unanswerable because "
                + "; ".join(blocking)
                + "."
            )
            return None, "abstain_unverified_unanswerable", reason

        # 2. Check insufficient evidence or complete failure
        if not evidence_cands:
            executed_any = any(c.execution_success for c in candidates)
            if executed_any:
                reason = "insufficient_evidence: Executed candidate queries produced empty set or all-NULL cells."
                return None, "insufficient_evidence", reason
            reason = "abstain: All candidate queries failed static checks or SQLite execution."
            return None, "abstain_fallback", reason

        # 3. Group candidates by canonical answer signature
        groups: dict[tuple[str, str], list[CandidateQuery]] = {}
        for c in evidence_cands:
            sig = answer_signature(c)
            if sig is not None:
                groups.setdefault(sig, []).append(c)

        if not groups:
            reason = "insufficient_evidence: Executed candidate queries produced unformattable or empty answers."
            return None, "insufficient_evidence", reason

        # Deterministic sorting within groups by candidate_id
        for grp in groups.values():
            grp.sort(key=lambda c: c.candidate_id)

        # Sort groups by decreasing size, tie-broken by lowest candidate_id of group leader
        sorted_groups = sorted(groups.values(), key=lambda g: (-len(g), g[0].candidate_id))
        best_group = sorted_groups[0]
        cluster_dist = "/".join(str(len(g)) for g in sorted_groups)
        cand_ids = [c.candidate_id for c in evidence_cands]

        agreement_ratio = len(best_group) / len(evidence_cands)

        # 4. Strict agreement gate (> min_agreement)
        if agreement_ratio > self.min_agreement:
            winner = best_group[0]
            agreeing_ids = [c.candidate_id for c in best_group]
            prefix = "unanimous" if len(best_group) == len(candidates) else "majority"
            status = f"{prefix}_{len(best_group)}_of_{len(candidates)}"
            summary = summarize_result_set(winner.result_set)
            reason = (
                f"majority: Candidate {winner.candidate_id} chosen as winning candidate because candidates {agreeing_ids} "
                f"produced identical canonical answers ({summary}, clusters: {cluster_dist})."
            )
            return winner, status, reason

        # 5. Abstain on split / low agreement (EV <= 0)
        reason = (
            f"abstain_low_agreement: Insufficient agreement ({cluster_dist}) "
            f"among {len(evidence_cands)} evidence candidates {cand_ids}."
        )
        return None, "abstain_low_agreement", reason


__all__ = [
    "VotingExecutor",
    "answer_signature",
    "canonical_cell",
    "has_evidence",
    "is_abstain_claim",
    "is_unanswerable_claim",
    "normalize_cell",
    "summarize_result_set",
]
