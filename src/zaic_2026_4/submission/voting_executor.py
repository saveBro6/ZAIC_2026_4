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
        # Round to 4 decimal places to prevent floating point representation noise
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


class VotingExecutor:
    """Executes candidate queries against SQLite database and selects final answer via majority voting."""

    def __init__(
        self,
        db_path: Path | str,
        timeout_seconds: float = DEFAULT_SQL_TIMEOUT_SECONDS,
        random_seed: int = 42,
    ) -> None:
        """Initialize VotingExecutor.

        Args:
            db_path: Path to the target business SQLite database.
            timeout_seconds: Timeout for query execution.
            random_seed: Seed for reproducible tie-breaking selection.
        """
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

            # Normalize rows to immutable tuples
            normalized_rows = [
                tuple(normalize_cell(cell) for cell in row)
                for row in raw_rows
            ]
            result_set = set(normalized_rows)
            return True, None, raw_rows, result_set

        except sqlite3.Error as exc:
            return False, f"SQLite error: {exc}", None, None
        except Exception as exc:
            return False, f"Execution error: {exc}", None, None
        finally:
            if conn is not None:
                conn.close()

    def execute_candidates(self, candidates: list[CandidateQuery]) -> None:
        """Execute all valid candidates and record execution outcomes in-place."""
        for cand in candidates:
            if not cand.is_valid_syntax:
                cand.execution_success = False
                cand.execution_error = "Skipped execution due to static syntax/construction errors."
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
        """Perform majority voting among executed candidates.

        Returns:
            Tuple of (winning_candidate, voting_status, selection_reason).
        """
        valid_candidates = [
            c for c in candidates
            if c.execution_success and c.result_set is not None
        ]

        if not valid_candidates:
            reason = "abstain: All candidate queries failed static checks or SQLite execution."
            return None, "abstain_fallback", reason

        # Group valid candidates by result set (using frozenset as hashable key)
        groups: dict[frozenset[tuple[Any, ...]], list[CandidateQuery]] = {}
        for c in valid_candidates:
            if c.result_set is not None:
                f_key = frozenset(c.result_set)
                groups.setdefault(f_key, []).append(c)

        # 1. Check for Majority Agreement (>= 2 candidates agree on identical result set)
        majority_group: list[CandidateQuery] | None = None
        for f_key, group in groups.items():
            if len(group) >= 2:
                majority_group = group
                break

        if majority_group is not None:
            winner = majority_group[0]
            agreeing_ids = [c.candidate_id for c in majority_group]
            status = "unanimous_3_of_3" if len(majority_group) == len(candidates) else "majority_2_of_3"
            summary = summarize_result_set(winner.result_set)
            reason = (
                f"majority: Candidate {winner.candidate_id} chosen as winning candidate because candidates {agreeing_ids} "
                f"produced identical result sets ({summary})."
            )
            return winner, status, reason

        # 2. Tie-break: All valid candidates produced divergent results -> Pick one randomly
        winner = self.rng.choice(valid_candidates)
        cand_ids = [c.candidate_id for c in valid_candidates]
        summary = summarize_result_set(winner.result_set)
        reason = (
            f"random: No agreement among {len(valid_candidates)} candidates {cand_ids}; "
            f"randomly selected candidate {winner.candidate_id} with result {summary}."
        )
        return winner, "random_tiebreak", reason


__all__ = [
    "VotingExecutor",
    "normalize_cell",
    "summarize_result_set",
]
