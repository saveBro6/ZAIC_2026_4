"""Result normalization and strict type conversion for problem contract compliance."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Literal

import sqlglot
from sqlglot import exp

from zaic_2026_4.constants import ISO_DATE_REGEX, SLASH_DATE_REGEX

PercentScale = Literal["percent", "ratio"]

TRUE_BOOLEAN_TOKENS: frozenset[str] = frozenset({"true", "1", "t", "yes", "có", "đúng"})
FALSE_BOOLEAN_TOKENS: frozenset[str] = frozenset({"false", "0", "f", "no", "không", "sai"})
ISO_DATETIME_PREFIX_REGEX: re.Pattern[str] = re.compile(r"^(\d{4}-\d{2}-\d{2})[T\s].+$")


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


def detect_percent_scale(sql: str | None) -> PercentScale:
    """Return whether the final projection already yields percent (0-100) or a bare ratio."""
    if not sql or not sql.strip():
        return "percent"

    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
    except Exception:  # noqa: BLE001
        return "percent"

    if parsed is None or not hasattr(parsed, "selects") or not parsed.selects:
        return "percent"

    proj = parsed.selects[0]

    # If any multiplication by numeric literal 100 exists in the projection, it's already in 0-100 scale
    for mul in proj.find_all(exp.Mul):
        for arg in (mul.this, mul.expression):
            curr = arg
            while isinstance(curr, (exp.Paren, exp.Cast, exp.TryCast)):
                curr = curr.this
            if isinstance(curr, exp.Literal) and curr.is_number:
                try:
                    if abs(float(curr.this) - 100.0) < 1e-9:
                        return "percent"
                except (ValueError, TypeError):
                    pass

    # Peel outer non-aggregating wrapper nodes
    node = proj
    while True:
        if isinstance(node, (exp.Alias, exp.Paren, exp.Cast, exp.TryCast)) and node.this is not None:
            node = node.this
            continue
        if isinstance(node, exp.Func) and not isinstance(node, exp.AggFunc) and node.this is not None:
            node = node.this
            continue
        break

    # If the unpeeled projection is a division
    if isinstance(node, exp.Div):
        denom = node.expression
        while isinstance(denom, (exp.Paren, exp.Cast, exp.TryCast, exp.Unary)):
            denom = denom.this
        # Denominator is a literal number (e.g. SUM(x) / 7) -> not a ratio calculation
        if isinstance(denom, exp.Literal) and denom.is_number:
            return "percent"
        # Denominator is an expression / column / count -> ratio needing scale
        return "ratio"

    return "percent"


def derive_table_columns(
    sql: str | None,
    description: list[str] | tuple[str, ...] | None,
    num_cols: int,
) -> list[str]:
    """Name table columns from the SQL projection, then cursor description, then positional fallback."""
    ast_names: list[str | None] = [None] * num_cols
    if sql and sql.strip():
        try:
            parsed = sqlglot.parse_one(sql, read="sqlite")
            if (
                parsed
                and hasattr(parsed, "selects")
                and len(parsed.selects) == num_cols
                and not any(isinstance(s, exp.Star) for s in parsed.selects)
            ):
                for i, proj in enumerate(parsed.selects):
                    if isinstance(proj, exp.Alias) and proj.alias and proj.alias.strip():
                        ast_names[i] = proj.alias.strip()
                    elif isinstance(proj, exp.Column) and proj.name and proj.name.strip():
                        ast_names[i] = proj.name.strip()
                    elif isinstance(proj, exp.Func):
                        if isinstance(proj.this, exp.Column) and proj.this.name and proj.this.name.strip():
                            ast_names[i] = f"{proj.sql_name().lower()}_{proj.this.name.strip()}"
                        elif proj.alias and proj.alias.strip():
                            ast_names[i] = proj.alias.strip()
        except Exception:  # noqa: BLE001, S110
            pass

    raw_names: list[str] = []
    for i in range(num_cols):
        col_name = ast_names[i]
        if not col_name and description and i < len(description) and description[i]:
            desc_str = str(description[i]).strip()
            if desc_str and "(" not in desc_str:
                col_name = desc_str
        if not col_name:
            col_name = f"col_{i + 1}"
        raw_names.append(col_name)

    # Deduplicate with _2, _3 suffixes
    seen: dict[str, int] = {}
    final_cols: list[str] = []
    for name in raw_names:
        if name not in seen:
            seen[name] = 1
            final_cols.append(name)
        else:
            seen[name] += 1
            cand = f"{name}_{seen[name]}"
            while cand in seen:
                seen[name] += 1
                cand = f"{name}_{seen[name]}"
            seen[cand] = 1
            final_cols.append(cand)

    return final_cols


class ResultFormatter:
    """Formats and validates raw SQLite query results into contract-compliant output JSON."""

    @classmethod
    def format_number(cls, val: Any) -> int | float | None:
        """Format number as integer or float rounded to 2 decimal places. Returns None if unparsable."""
        if val is None or isinstance(val, bool):
            return None
        if isinstance(val, str) and not val.strip():
            return None
        try:
            num = float(val)
            if not math.isfinite(num):
                return None
            if num.is_integer():
                return int(num)
            return round(num, 2)
        except (ValueError, TypeError):
            return None

    @classmethod
    def format_percent(cls, val: Any, scale: PercentScale = "percent") -> float | None:
        """Format percentage value rounded to 2 decimal places (in percent, e.g. 12.45). Returns None if unparsable."""
        if val is None or isinstance(val, bool):
            return None
        if isinstance(val, str) and not val.strip():
            return None
        try:
            num = float(val)
            if not math.isfinite(num):
                return None
            if scale == "ratio" and abs(num) <= 1.0:
                num *= 100.0
            res = round(num, 2)
            return 0.0 if res == 0.0 else res
        except (ValueError, TypeError):
            return None

    @classmethod
    def format_date(cls, val: Any) -> str | None:
        """Format date into ISO YYYY-MM-DD or YYYY-MM. Returns None if unparsable."""
        if val is None or isinstance(val, bool):
            return None
        s = str(val).strip()
        if not s:
            return None
        if ISO_DATE_REGEX.match(s):
            return s
        if re.match(r"^\d{4}-\d{2}$", s):
            return s

        m_dt = ISO_DATETIME_PREFIX_REGEX.match(s)
        if m_dt:
            return m_dt.group(1)

        # Try converting DD/MM/YYYY to YYYY-MM-DD
        m = SLASH_DATE_REGEX.match(s)
        if m:
            parts = s.split("/")
            return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"

        return None

    @classmethod
    def format_result(
        cls,
        raw_rows: list[Any] | None,
        target_type: str = "number",
        *,
        raw_columns: list[str] | tuple[str, ...] | None = None,
        sql: str | None = None,
    ) -> tuple[str, Any]:
        """Format raw SQLite rows into (answer_type, answer_value) conforming to problem contract.

        Args:
            raw_rows: Raw rows returned by SQLite query execution.
            target_type: Valid DataMate target type predicted by LLM Structured Outputs.
            raw_columns: Column names from SQLite cursor description.
            sql: SQL statement that produced the results for scale/column detection.

        Returns:
            Tuple of (type_str, answer_val).
        """
        if target_type == "unanswerable":
            return "unanswerable", None

        if target_type == "abstain":
            return "abstain", None

        # 1. Handle Empty Rows or all-NULL rows (never fabricate 0/0.0)
        if not raw_rows:
            return "abstain", None

        if all(
            all(c is None for c in (row if isinstance(row, (list, tuple)) else [row]))
            for row in raw_rows
        ):
            return "abstain", None

        # 2. Handle Single Row Result
        if len(raw_rows) == 1:
            row = raw_rows[0]
            # Multiple columns in single row
            if isinstance(row, (list, tuple)) and len(row) > 1:
                if target_type == "list":
                    extracted = []
                    for c in row:
                        if c is not None:
                            if isinstance(c, (int, float)) and not isinstance(c, bool):
                                f_num = cls.format_number(c)
                                if f_num is not None:
                                    extracted.append(f_num)
                            else:
                                s_c = str(c).strip()
                                if s_c:
                                    extracted.append(s_c)
                    if not extracted:
                        return "abstain", None
                    return "list", extracted

                cols = derive_table_columns(sql, raw_columns, len(row))
                return "table", {
                    "columns": cols,
                    "rows": [list(row)],
                }

            # Single column in single row
            scalar = row[0] if isinstance(row, (list, tuple)) else row

            if scalar is None:
                return "abstain", None

            if target_type == "percent":
                scale = detect_percent_scale(sql)
                pct_val = cls.format_percent(scalar, scale=scale)
                if pct_val is None:
                    return "abstain", None
                return "percent", pct_val

            if target_type == "number":
                num_val = cls.format_number(scalar)
                if num_val is None:
                    return "abstain", None
                return "number", num_val

            if target_type == "date":
                date_val = cls.format_date(scalar)
                if date_val is None:
                    return "abstain", None
                return "date", date_val

            if target_type == "boolean":
                if isinstance(scalar, bool):
                    return "boolean", scalar
                if isinstance(scalar, (int, float)):
                    if scalar == 1:
                        return "boolean", True
                    if scalar == 0:
                        return "boolean", False
                    return "abstain", None
                s_lower = str(scalar).strip().lower()
                if s_lower in TRUE_BOOLEAN_TOKENS:
                    return "boolean", True
                if s_lower in FALSE_BOOLEAN_TOKENS:
                    return "boolean", False
                return "abstain", None

            if target_type == "string":
                s_str = str(scalar).strip()
                if not s_str:
                    return "abstain", None
                return "string", s_str

            if target_type == "list":
                if isinstance(scalar, (int, float)) and not isinstance(scalar, bool):
                    l_num = cls.format_number(scalar)
                    if l_num is None:
                        return "abstain", None
                    return "list", [l_num]
                s_l = str(scalar).strip()
                if not s_l:
                    return "abstain", None
                return "list", [s_l]

            # Fallback for scalar
            if isinstance(scalar, (int, float)) and not isinstance(scalar, bool):
                fallback_num = cls.format_number(scalar)
                if fallback_num is None:
                    return "abstain", None
                return "number", fallback_num
            s_fb = str(scalar).strip()
            if not s_fb:
                return "abstain", None
            return "string", s_fb

        # 3. Handle Multiple Rows Result
        first_row = raw_rows[0]
        num_cols = len(first_row) if isinstance(first_row, (list, tuple)) else 1

        if num_cols == 1 or target_type == "list":
            # Single column with multiple rows -> List
            extracted_list: list[Any] = []
            for r in raw_rows:
                val = r[0] if isinstance(r, (list, tuple)) else r
                if val is not None:
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        item_num = cls.format_number(val)
                        if item_num is not None:
                            extracted_list.append(item_num)
                    else:
                        s_item = str(val).strip()
                        if s_item:
                            extracted_list.append(s_item)
            if not extracted_list:
                return "abstain", None
            return "list", extracted_list

        # Multiple columns with multiple rows -> Table
        cols = derive_table_columns(sql, raw_columns, num_cols)
        table_rows = [list(r) if isinstance(r, (list, tuple)) else [r] for r in raw_rows]
        return "table", {
            "columns": cols,
            "rows": table_rows,
        }


__all__ = [
    "PercentScale",
    "ResultFormatter",
    "derive_table_columns",
    "detect_percent_scale",
    "normalize_contract_string",
]
