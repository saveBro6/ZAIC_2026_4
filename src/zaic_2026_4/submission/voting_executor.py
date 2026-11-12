"""SQLite query execution, set conversion, and majority voting logic."""

from __future__ import annotations

import logging
import random
import sqlite3
from pathlib import Path
from typing import Any

from zaic_2026_4.constants import DEFAULT_SQL_TIMEOUT_SECONDS
from zaic_2026_4.submission.models import CandidateQuery

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


class VotingExecutor:
    """Executes candidate queries against SQLite database and selects final answer via majority voting."""

    def __init__(
        self,
        db_path: Path | str,
        timeout_seconds: float = DEFAULT_SQL_TIMEOUT_SECONDS,
        random_seed: int = 42,
    ) -> None:
        """Initialize VotingExecutor."""
        self.db_path = Path(db_path).resolve()
        self.timeout_seconds = timeout_seconds
        self.rng = random.Random(random_seed)

    def execute_query(
        self,
        sql: str,
    ) -> tuple[bool, str | None, list[Any] | None, set[tuple[Any, ...]] | None]:
        """Execute a single query in read-only mode and convert results to a set of tuples."""
        if not self.db_path.is_file():
            return False, f"Database file not found: {self.db_path}", None, None

        uri = f"file:{self.db_path.as_posix()}?mode=ro"
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
                continue

            if not cand.is_valid_syntax:
                cand.execution_success = False
                cand.execution_error = "Skipped execution due to static syntax/construction errors."
                continue

            # Reuse cached outcome if already executed successfully during dry-run
            if cand.execution_success and cand.result_set is not None and cand.raw_result is not None:
                continue

            success, err, raw_rows, r_set = self.execute_query(cand.final_sql)
            cand.execution_success = success
            cand.execution_error = err
            cand.raw_result = raw_rows
            cand.result_set = r_set

    def vote(
        self,
        candidates: list[CandidateQuery],
    ) -> tuple[CandidateQuery | None, str, str]:
        """Perform majority voting among executed candidates with evidence guards."""
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

        # 3. Group candidates by identical result set
        groups: dict[frozenset[tuple[Any, ...]], list[CandidateQuery]] = {}
        for c in evidence_cands:
            if c.result_set is not None:
                f_key = frozenset(c.result_set)
                groups.setdefault(f_key, []).append(c)

        # 4. Majority agreement check (>= 2 candidates produce identical result set)
        majority_group: list[CandidateQuery] | None = None
        for f_key, group in groups.items():
            if len(group) >= 2:
                majority_group = group
                break

        if majority_group is not None:
            winner = majority_group[0]
            agreeing_ids = [c.candidate_id for c in majority_group]
            prefix = "unanimous" if len(majority_group) == len(candidates) else "majority"
            status = f"{prefix}_{len(majority_group)}_of_{len(candidates)}"
            summary = summarize_result_set(winner.result_set)
            reason = (
                f"majority: Candidate {winner.candidate_id} chosen as winning candidate because candidates {agreeing_ids} "
                f"produced identical result sets ({summary})."
            )
            return winner, status, reason

        # 5. Tie-break: All valid candidates produced divergent results -> Pick one randomly from evidence candidates
        winner = self.rng.choice(evidence_cands)
        cand_ids = [c.candidate_id for c in evidence_cands]
        summary = summarize_result_set(winner.result_set)
        reason = (
            f"random: No agreement among {len(evidence_cands)} candidates {cand_ids}; "
            f"randomly selected candidate {winner.candidate_id} with result {summary}."
        )
        return winner, "random_tiebreak", reason


__all__ = [
    "VotingExecutor",
    "has_evidence",
    "normalize_cell",
    "summarize_result_set",
]
