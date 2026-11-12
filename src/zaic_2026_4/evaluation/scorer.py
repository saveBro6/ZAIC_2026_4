"""Ground-truth evaluation and scoring engine compliant with Rulebook Section 9."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

FLOAT_EPS = 1e-6


def normalize_string(s: Any) -> str:
    """Normalize text per Section 4/9: Unicode NFC, lowercase, remove punctuation, collapse whitespace.

    Vietnamese diacritics are strictly preserved.
    """
    if not isinstance(s, str):
        s = str(s)
    # 1. Unicode NFC
    s = unicodedata.normalize("NFC", s)
    # 2. Lowercase
    s = s.lower()
    # 3. Replace non-(letter/digit/whitespace) and underscores with space
    s = re.sub(r"[^\w\s]|_", " ", s)
    # 4. Collapse consecutive whitespace characters and trim edges
    return re.sub(r"\s+", " ", s).strip()


def is_json_number(v: Any) -> bool:
    """Check whether a value is a valid JSON number (int or finite float, excluding bool)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def match_number(pred: Any, ref: Any) -> bool:
    """Strict check for numeric value: both must be finite JSON numbers within float tolerance."""
    if not is_json_number(pred) or not is_json_number(ref):
        return False
    return abs(pred - ref) <= FLOAT_EPS


def match_percent(pred: Any, ref: Any) -> bool:
    """Check percentage value: both rounded to 2 decimal places with matching sign."""
    if not is_json_number(pred) or not is_json_number(ref):
        return False
    return abs(round(pred, 2) - round(ref, 2)) <= FLOAT_EPS


def match_date(pred: Any, ref: Any) -> bool:
    """Check ISO date string YYYY-MM-DD or YYYY-MM exact match."""
    if not isinstance(pred, str) or not isinstance(ref, str):
        return False
    return pred.strip() == ref.strip()


def match_boolean(pred: Any, ref: Any) -> bool:
    """Check boolean value: must be exact bool instance (true/false, not 1/0)."""
    if not isinstance(pred, bool) or not isinstance(ref, bool):
        return False
    return pred is ref


def match_string(pred: Any, ref: str, aliases: list[str] | None = None) -> bool:
    """Check string equivalence: normalized pred must match normalized ref or any accepted alias."""
    if not isinstance(pred, str) or not isinstance(ref, str):
        return False
    accepted = {normalize_string(ref)}
    if aliases:
        for alias in aliases:
            if isinstance(alias, str):
                accepted.add(normalize_string(alias))
    return normalize_string(pred) in accepted


def match_cell(pred: Any, ref: Any, aliases: list[str] | None = None) -> bool:
    """Match a single list item or table cell according to reference data type."""
    if isinstance(ref, str):
        return match_string(pred, ref, aliases)
    if is_json_number(ref):
        return match_number(pred, ref)
    if isinstance(ref, bool):
        return match_boolean(pred, ref)
    return pred == ref


def match_list(pred: Any, ref: list[Any], aliases: list[list[str]] | None = None) -> bool:
    """Check ordered list: identical length and position-by-position match."""
    if not isinstance(pred, list) or not isinstance(ref, list):
        return False
    if len(pred) != len(ref):
        return False
    for i, ref_item in enumerate(ref):
        item_aliases = aliases[i] if (aliases and i < len(aliases) and isinstance(aliases[i], list)) else None
        if not match_cell(pred[i], ref_item, item_aliases):
            return False
    return True


def maximum_bipartite_matching(adj: list[list[int]], num_right: int) -> int:
    """Compute maximum bipartite matching size using augmenting paths."""
    match_right = [-1] * num_right

    def dfs(u: int, visited: list[bool]) -> bool:
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                if match_right[v] < 0 or dfs(match_right[v], visited):
                    match_right[v] = u
                    return True
        return False

    matches = 0
    for u in range(len(adj)):
        visited = [False] * num_right
        if dfs(u, visited):
            matches += 1
    return matches


def match_table(pred: Any, ref: dict[str, Any], aliases: list[list[str]] | None = None) -> bool:
    """Check table: match columns by name or position, and match rows as an unordered multiset."""
    if not isinstance(pred, dict) or not isinstance(ref, dict):
        return False
    if "columns" not in pred or "rows" not in pred:
        return False
    if "columns" not in ref or "rows" not in ref:
        return False

    pred_cols = pred["columns"]
    pred_rows = pred["rows"]
    ref_cols = ref["columns"]
    ref_rows = ref["rows"]

    if not isinstance(pred_cols, list) or not isinstance(pred_rows, list):
        return False
    if not isinstance(ref_cols, list) or not isinstance(ref_rows, list):
        return False

    if len(pred_cols) != len(ref_cols) or len(pred_rows) != len(ref_rows):
        return False

    for row in pred_rows:
        if not isinstance(row, list) or len(row) != len(pred_cols):
            return False

    # Column alignment: try to match by normalized column names
    norm_ref_cols = [normalize_string(c) for c in ref_cols]
    norm_pred_cols = [normalize_string(c) for c in pred_cols]

    if len(set(norm_pred_cols)) == len(norm_pred_cols) and set(norm_pred_cols) == set(norm_ref_cols):
        col_mapping = [norm_pred_cols.index(rc) for rc in norm_ref_cols]
        aligned_pred_rows = [[row[idx] for idx in col_mapping] for row in pred_rows]
    else:
        aligned_pred_rows = pred_rows

    # Row multiset matching via bipartite graph
    num_rows = len(ref_rows)
    adj: list[list[int]] = [[] for _ in range(num_rows)]

    for i, ref_row in enumerate(ref_rows):
        row_alias = aliases[i] if (aliases and i < len(aliases) and isinstance(aliases[i], list)) else None
        for j, cand_row in enumerate(aligned_pred_rows):
            matched = True
            for col_idx in range(len(ref_cols)):
                cell_alias = row_alias if col_idx == 0 else None
                if not match_cell(cand_row[col_idx], ref_row[col_idx], cell_alias):
                    matched = False
                    break
            if matched:
                adj[i].append(j)

    return maximum_bipartite_matching(adj, num_rows) == num_rows


def grade_one(ref: dict[str, Any], pred: dict[str, Any] | None) -> tuple[str, int, str]:
    """Grade a single prediction against reference ground truth.

    Returns:
        (verdict, score, reason)
        verdict in {"correct", "wrong", "abstain"}
        score in {1, -1, 0}
    """
    ref_type = ref.get("type", "unknown")

    # 1. Missing prediction
    if pred is None:
        return "abstain", 0, "missing"

    pred_type = pred.get("type")

    # 2. Explicit abstain
    if pred_type == "abstain":
        return "abstain", 0, "explicit_abstain"

    # 3. Unanswerable prediction format validation
    if pred_type == "unanswerable":
        if "reason" in pred and pred["reason"] is not None and not isinstance(pred["reason"], str):
            # Rulebook §7: reason must be string, otherwise unreadable -> counts as empty/abstain
            return "abstain", 0, "invalid_unanswerable_reason"
        if ref_type == "unanswerable":
            return "correct", 1, "correct_unanswerable"
        return "wrong", -1, "false_unanswerable"

    # 4. Reference is unanswerable, but prediction answered something else
    if ref_type == "unanswerable":
        return "wrong", -1, "answered_unanswerable"

    # 5. Type mismatch
    if pred_type != ref_type:
        return "wrong", -1, f"type_mismatch_{pred_type}_vs_{ref_type}"

    # 6. Check presence of answer key
    if "answer" not in pred:
        return "wrong", -1, "missing_answer_key"

    pred_ans = pred["answer"]
    ref_ans = ref.get("answer")
    ref_alias = ref.get("alias", [])

    matched = False
    if ref_type == "number":
        matched = match_number(pred_ans, ref_ans)
    elif ref_type == "percent":
        matched = match_percent(pred_ans, ref_ans)
    elif ref_type == "date":
        matched = match_date(pred_ans, ref_ans)
    elif ref_type == "boolean":
        matched = match_boolean(pred_ans, ref_ans)
    elif ref_type == "string":
        matched = match_string(pred_ans, ref_ans, ref_alias)
    elif ref_type == "list":
        matched = match_list(pred_ans, ref_ans, ref_alias)
    elif ref_type == "table":
        matched = match_table(pred_ans, ref_ans, ref_alias)
    else:
        matched = pred_ans == ref_ans

    if matched:
        return "correct", 1, "correct"
    return "wrong", -1, "incorrect_value"
