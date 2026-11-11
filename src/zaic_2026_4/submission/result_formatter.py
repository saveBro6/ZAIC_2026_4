"""Result normalization and strict type conversion for problem contract compliance."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from zaic_2026_4.constants import ISO_DATE_REGEX, SLASH_DATE_REGEX


def normalize_contract_string(s: str) -> str:
    """Normalize string per Section 4 of problem contract: NFC, lower, non-alphanumeric to space, collapse whitespace."""
    if not isinstance(s, str):
        s = str(s)
    # 1. Unicode NFC normalization
    s = unicodedata.normalize("NFC", s)
    # 2. Lowercase conversion
    s = s.lower()
    # 3. Replace non-(letter/digit/whitespace) with space (preserves Vietnamese diacritics)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = s.replace("_", " ")
    # 4. Collapse runs of whitespace
    return re.sub(r"\s+", " ", s).strip()


class ResultFormatter:
    """Formats and validates raw SQLite query results into contract-compliant output JSON."""

    @classmethod
    def format_number(cls, val: Any) -> int | float:
        """Format number as integer or float rounded to 2 decimal places."""
        if val is None:
            return 0
        try:
            num = float(val)
            if num.is_integer():
                return int(num)
            return round(num, 2)
        except (ValueError, TypeError):
            return 0

    @classmethod
    def format_percent(cls, val: Any) -> float:
        """Format percentage value rounded to 2 decimal places (in percent, e.g. 12.45)."""
        if val is None:
            return 0.0
        try:
            num = float(val)
            # If query calculated ratio (e.g. 0.0527), convert to percentage if < 1.0 and > -1.0
            # Note: test ground truth values are mostly > 1.0 (e.g. 2.08, 5.27, 2.19)
            if -1.0 < num < 1.0 and num != 0.0:
                num = num * 100.0
            return round(num, 2)
        except (ValueError, TypeError):
            return 0.0

    @classmethod
    def format_date(cls, val: Any) -> str:
        """Format date into ISO YYYY-MM-DD or YYYY-MM."""
        s = str(val).strip()
        if ISO_DATE_REGEX.match(s):
            return s
        if re.match(r"^\d{4}-\d{2}$", s):
            return s

        # Try converting DD/MM/YYYY to YYYY-MM-DD
        m = SLASH_DATE_REGEX.match(s)
        if m:
            parts = s.split("/")
            return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"

        return s

    @classmethod
    def format_result(
        cls,
        raw_rows: list[Any] | None,
        target_type: str = "number",
    ) -> tuple[str, Any]:
        """Format raw SQLite rows into (answer_type, answer_value) conforming to problem contract.

        Args:
            raw_rows: Raw rows returned by SQLite query execution.
            target_type: Valid DataMate target type predicted by LLM Structured Outputs.

        Returns:
            Tuple of (type_str, answer_val).
        """
        if target_type == "unanswerable":
            return "unanswerable", None

        if target_type == "abstain":
            return "abstain", None

        # 1. Handle Empty Rows (0 rows returned)
        if not raw_rows:
            # If question is asking for count/total/percentage, 0 is the natural business default
            if target_type == "percent":
                return "percent", 0.0
            if target_type == "number":
                return "number", 0
            # For entity, date, list lookups: empty result is ambiguous -> abstain to avoid -1 penalty
            return "abstain", None

        # 2. Handle Single Row Result
        if len(raw_rows) == 1:
            row = raw_rows[0]
            # Multiple columns in single row
            if isinstance(row, (list, tuple)) and len(row) > 1:
                # Could be table or list
                if target_type == "list":
                    return "list", [cls.format_number(c) if isinstance(c, (int, float)) else str(c) for c in row]
                return "table", {
                    "columns": [f"col_{i+1}" for i in range(len(row))],
                    "rows": [list(row)],
                }

            # Single column in single row
            scalar = row[0] if isinstance(row, (list, tuple)) else row

            if scalar is None:
                if target_type in ("number", "percent"):
                    return target_type, (0 if target_type == "number" else 0.0)
                return "abstain", None

            if target_type == "percent":
                return "percent", cls.format_percent(scalar)

            if target_type == "number":
                return "number", cls.format_number(scalar)

            if target_type == "date":
                return "date", cls.format_date(scalar)

            if target_type == "boolean":
                if isinstance(scalar, bool):
                    return "boolean", scalar
                if isinstance(scalar, (int, float)):
                    return "boolean", bool(scalar)
                s_lower = str(scalar).strip().lower()
                return "boolean", (s_lower in ("true", "1", "t", "yes", "có", "đúng"))

            if target_type == "string":
                return "string", str(scalar).strip()

            if target_type == "list":
                val = cls.format_number(scalar) if isinstance(scalar, (int, float)) else str(scalar).strip()
                return "list", [val]

            # Fallback for scalar
            if isinstance(scalar, (int, float)):
                return "number", cls.format_number(scalar)
            return "string", str(scalar).strip()

        # 3. Handle Multiple Rows Result
        first_row = raw_rows[0]
        num_cols = len(first_row) if isinstance(first_row, (list, tuple)) else 1

        if num_cols == 1 or target_type == "list":
            # Single column with multiple rows -> List
            extracted_list: list[Any] = []
            for r in raw_rows:
                val = r[0] if isinstance(r, (list, tuple)) else r
                if val is not None:
                    if isinstance(val, (int, float)):
                        extracted_list.append(cls.format_number(val))
                    else:
                        extracted_list.append(str(val).strip())
            return "list", extracted_list

        # Multiple columns with multiple rows -> Table
        table_rows = [list(r) for r in raw_rows]
        return "table", {
            "columns": [f"col_{i+1}" for i in range(num_cols)],
            "rows": table_rows,
        }


__all__ = [
    "ResultFormatter",
    "normalize_contract_string",
]
