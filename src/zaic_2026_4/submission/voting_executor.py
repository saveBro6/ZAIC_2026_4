"""SQLite query execution, set conversion, and majority voting logic."""

from __future__ import annotations

import json
import logging
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


def has_evidence(res_set: set[tuple[Any, ...]] | frozenset[tuple[Any, ...]] | None) -> bool:
    """Check whether result set contains at least one non-None cell in at least one row."""
    if res_set is None or len(res_set) == 0:
        return False
    return not all(all(cell is None for cell in row) for row in res_set)


def answer_signature(cand: CandidateQuery) -> tuple[str, str] | None:
    """Canonical (answer_type, canonical_value) key of the answer this candidate would submit."""
    if not cand.execution_success or not has_evidence(cand.result_set):
        return None

    raw_rows = cand.raw_result
    if raw_rows is None and cand.result_set is not None:
        raw_rows = list(cand.result_set)

    if not raw_rows:
        return None

    ans_type, ans_val = ResultFormatter.format_result(
        raw_rows,
        cand.predicted_type,
        raw_columns=cand.result_columns,
        sql=cand.final_sql,
    )

    if ans_type in ("abstain", "unanswerable") or ans_val is None:
        return None

    if ans_type == "table" and isinstance(ans_val, dict):
        cols = [normalize_contract_string(c) for c in ans_val.get("columns", [])]
        rows = sorted(json.dumps(r, ensure_ascii=False) for r in ans_val.get("rows", []))
        canon_str = json.dumps({"columns": cols, "rows": rows}, ensure_ascii=False, sort_keys=True)
    elif ans_type == "string":
        canon_str = json.dumps(normalize_contract_string(ans_val), ensure_ascii=False)
    else:
        # number, percent, date, boolean, list (keeps list position order)
        canon_str = json.dumps(ans_val, ensure_ascii=False, sort_keys=True)

    return ans_type, canon_str


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
    ) -> tuple[CandidateQuery | None, str, str]:
        """Perform majority voting among executed candidates with canonical answer signatures and agreement gate."""
        # 1. Check deliberate unanswerable consensus
        unans_cands = [
            c for c in candidates
            if c.deliberate_abstain or (
                c.predicted_type in ("unanswerable", "abstain") and not c.final_sql.strip()
            )
        ]
        evidence_cands = [
            c for c in candidates
            if c.execution_success and has_evidence(c.result_set)
        ]

        if unans_cands and not evidence_cands:
            winner = unans_cands[0]
            reason = "unanswerable: Candidate consensus or deliberate abstain on unanswerable query."
            return winner, "unanswerable_consensus", reason

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
    "has_evidence",
    "normalize_cell",
    "summarize_result_set",
]
