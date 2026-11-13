"""Static SQL validation and anti-pattern detection using SQLglot for SQLite."""

from __future__ import annotations

import calendar
import json
import logging
import re
import unicodedata
from datetime import timedelta
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.optimizer.scope import traverse_scope

from zaic_2026_4.common.query_intent import QueryIntent
from zaic_2026_4.common.temporal_resolver import parse_iso_date
from zaic_2026_4.common.trace_logger import extract_thinking_trace
from zaic_2026_4.submission.models import VALID_DATAMATE_TYPES, ValidationResult
from zaic_2026_4.submission.rules import ValidationIssue

logger = logging.getLogger(__name__)


def clean_sql_string(raw_sql: str) -> str:
    """Extract clean SQL string by stripping markdown code fences and whitespace."""
    sql = raw_sql.strip()
    if sql.startswith("```"):
        lines = sql.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        sql = "\n".join(lines).strip()
    return sql


def _strip_code_fences(raw_output: str) -> str:
    """Remove leading/trailing markdown code fences from a model response."""
    text = raw_output.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def is_truncated_json_output(raw_output: str) -> bool:
    """Detect a response cut mid-JSON (unbalanced braces or unterminated string).

    A truncated generation must not be treated as an unparseable one: the repair turn has
    to shorten the reasoning and re-emit a complete object instead of guessing the SQL.
    """
    text = _strip_code_fences(raw_output)
    if not text:
        return False
    if not text.lstrip().startswith("{"):
        return False
    if text.count("{") > text.count("}"):
        return True

    # Odd number of unescaped double quotes means the last string literal was cut open.
    quote_count = 0
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            quote_count += 1
    return quote_count % 2 == 1


def parse_structured_sql_output_with_reasoning(
    raw_output: str,
    default_type: str = "number",
) -> tuple[str, str, str, str, list[ValidationIssue]]:
    """Parse JSON Structured Output into (sql, target_type, reasoning, thinking, issues).

    Emits ValidationIssue with canonical codes OUTPUT_EMPTY, OUTPUT_TRUNCATED,
    OUTPUT_UNPARSEABLE, or TYPE_INVALID. A response that clearly attempted JSON but could
    not be parsed returns an empty SQL string: echoing the raw blob as SQL used to make a
    malformed generation look like a syntactically valid query.
    """
    thinking = extract_thinking_trace(raw_output)
    text = _strip_code_fences(raw_output)
    looks_like_json = text.lstrip().startswith("{")

    parsed_dict: dict[str, Any] | None = None
    issues: list[ValidationIssue] = []

    if text.startswith("{") and text.endswith("}"):
        try:
            parsed_dict = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            parsed_dict = None

    if parsed_dict is None and looks_like_json:
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                parsed_dict = json.loads(json_match.group(0))
            except (json.JSONDecodeError, ValueError):
                parsed_dict = None

    if parsed_dict is None and looks_like_json:
        truncated = is_truncated_json_output(text)
        issues.append(
            ValidationIssue(
                code="OUTPUT_TRUNCATED" if truncated else "OUTPUT_UNPARSEABLE",
                message=(
                    "Đầu ra JSON bị cắt giữa chừng do hết ngân sách token."
                    if truncated
                    else "Không thể phân tích cú pháp JSON từ đầu ra của mô hình."
                ),
                severity="blocking",
            )
        )
        return "", default_type, "", thinking, issues

    if parsed_dict is not None and isinstance(parsed_dict, dict):
        raw_sql = str(parsed_dict.get("sql", ""))
        raw_type = str(parsed_dict.get("type", default_type)).strip().lower()
        reasoning = str(parsed_dict.get("reasoning", "") or "").strip()
        if not thinking and reasoning:
            thinking = extract_thinking_trace(reasoning)
        clean_sql = clean_sql_string(raw_sql)

        if raw_type not in VALID_DATAMATE_TYPES:
            issues.append(
                ValidationIssue(
                    code="TYPE_INVALID",
                    message=(
                        f"Trường 'type' ({raw_type!r}) không hợp lệ theo quy chế DataMate. "
                        f"Giá trị bắt buộc phải là 1 trong {VALID_DATAMATE_TYPES}."
                    ),
                    severity="blocking",
                )
            )

        return clean_sql, raw_type, reasoning, thinking, issues

    cleaned = clean_sql_string(raw_output)
    if not cleaned:
        issues.append(
            ValidationIssue(
                code="OUTPUT_EMPTY",
                message="Đầu ra của mô hình rỗng, không chứa truy vấn SQL hay đối tượng JSON hợp lệ.",
                severity="blocking",
            )
        )
    return cleaned, default_type, "", thinking, issues


def parse_structured_sql_output(
    raw_output: str,
    default_type: str = "number",
) -> tuple[str, str, list[ValidationIssue]]:
    """Parse JSON Structured Output from LLM into (sql, target_type, validation_issues)."""
    clean_sql, raw_type, _reasoning, _thinking, issues = parse_structured_sql_output_with_reasoning(
        raw_output=raw_output,
        default_type=default_type,
    )
    return clean_sql, raw_type, issues


def _extract_column_name(node: Any) -> str:
    """Extract column name identifier from an expression or node."""
    if isinstance(node, exp.Column):
        return node.name
    if isinstance(node, exp.Identifier):
        return node.name
    return str(node).strip("\"'` ")


def _normalize_text(val: str) -> str:
    """Normalize text for insensitive comparison."""
    return unicodedata.normalize("NFC", val.strip().lower())


def _is_pseudo_column_literal(name: str) -> bool:
    """Check if identifier is actually a string literal or punctuation enclosed in double quotes."""
    clean = name.strip()
    if not clean:
        return True
    if clean.isdigit():
        return True
    if re.match(r"^[-%/:\s\.,;]+$", clean) or clean.startswith("%"):
        return True
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}$|^\d{1,2}/\d{1,2}/\d{4}$", clean))


def _extract_comparison_strings(expr: exp.Expression) -> list[str]:
    """Extract string values from literals or quoted identifiers used as strings in SQLite."""
    strings: list[str] = []
    for lit in expr.find_all(exp.Literal):
        if lit.is_string or re.match(r"^\d{4}-\d{2}-\d{2}$|^\d{1,2}/\d{1,2}/\d{4}$", str(lit.this)):
            strings.append(lit.name or str(lit.this))
    for col in expr.find_all(exp.Column):
        if not col.table and (
            re.match(r"^\d{4}-\d{2}-\d{2}$|^\d{1,2}/\d{1,2}/\d{4}$", col.name)
            or col.this.args.get("quoted")
        ):
            strings.append(col.name)
    return strings


def _extract_schema_lookup(
    metadata: dict[str, Any] | None,
) -> tuple[dict[str, set[str]], dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Extract lookup tables from metadata profile.

    Returns:
        tables: dict[physical_table_name, set_of_original_column_names]
        col_info: dict[f"{table_lower}.{col_lower}", col_metadata_dict]
        all_cols: dict[col_lower, list_of_table_names]
    """
    if not metadata:
        return {}, {}, {}

    target = metadata.get("metadata_profile", metadata)
    tables: dict[str, set[str]] = {}
    col_info: dict[str, dict[str, Any]] = {}
    all_cols: dict[str, list[str]] = {}

    for key, val in target.items():
        if not isinstance(val, dict):
            continue
        if "." in key:
            tbl, col = key.split(".", 1)
        else:
            tbl, col = "", key

        tbl_clean = tbl.strip('" ')
        col_clean = col.strip('" ')

        if tbl_clean:
            tables.setdefault(tbl_clean, set()).add(col_clean)
            col_info[f"{tbl_clean.lower()}.{col_clean.lower()}"] = val
            all_cols.setdefault(col_clean.lower(), []).append(tbl_clean)
        else:
            col_info[col_clean.lower()] = val
            all_cols.setdefault(col_clean.lower(), []).append("")

    return tables, col_info, all_cols


def _is_date_normalization_node(node: exp.Expression) -> bool:
    """Check if node wraps an expression inside a date normalization function."""
    curr = node.parent
    while curr is not None:
        if isinstance(
            curr,
            (
                exp.Substring,
                exp.Date,
                exp.Datetime,
                exp.TsOrDsToDate,
                exp.TsOrDsToDatetime,
                exp.TsOrDsToTimestamp,
                exp.TimeToStr,
            ),
        ):
            return True
        if isinstance(curr, (exp.Func, exp.Anonymous)) and curr.name.lower() in (
            "substr",
            "date",
            "datetime",
            "strftime",
            "julianday",
        ):
            return True
        curr = curr.parent
    return False


def _get_column_samples(col_meta: dict[str, Any]) -> set[str]:
    """Aggregate known sample values from sample_values, top_10_samples, and raw_samples."""
    samples: set[str] = set()
    s_vals = col_meta.get("sample_values")
    if isinstance(s_vals, list):
        for s in s_vals:
            if s is not None:
                samples.add(str(s))

    top_10 = col_meta.get("top_10_samples")
    if isinstance(top_10, dict):
        for k in top_10:
            if k is not None:
                samples.add(str(k))
    elif isinstance(top_10, list):
        for s in top_10:
            if s is not None:
                samples.add(str(s))

    raw_s = col_meta.get("raw_samples")
    if isinstance(raw_s, list):
        for s in raw_s:
            if s is not None:
                samples.add(str(s))

    return samples


_CURRENCY_NOISE_REGEX = re.compile(r"[^\d.,-]")
_THOUSAND_DOT_REGEX = re.compile(r"^-?\d{1,3}(?:\.\d{3})+$")
_THOUSAND_COMMA_REGEX = re.compile(r"^-?\d{1,3}(?:,\d{3})+$")
_DECIMAL_TAIL_REGEX = re.compile(r"[.,]\d{1,2}$")


def infer_currency_separators(samples: set[str]) -> tuple[str | None, str | None]:
    """Infer the thousands and decimal separators used by a currency column.

    Returns:
        Tuple of (thousands_separator, decimal_separator); either element may be None when
        the sample values do not expose it.
    """
    thousand_votes: dict[str, int] = {}
    decimal_votes: dict[str, int] = {}
    for sample in samples:
        clean = _CURRENCY_NOISE_REGEX.sub("", str(sample)).strip()
        if not clean:
            continue
        if _THOUSAND_DOT_REGEX.match(clean):
            thousand_votes["."] = thousand_votes.get(".", 0) + 1
        elif _THOUSAND_COMMA_REGEX.match(clean):
            thousand_votes[","] = thousand_votes.get(",", 0) + 1
        elif _DECIMAL_TAIL_REGEX.search(clean) and not _THOUSAND_DOT_REGEX.match(clean) and not _THOUSAND_COMMA_REGEX.match(clean):
            tail = clean[-3:]
            if "." in tail:
                decimal_votes["."] = decimal_votes.get(".", 0) + 1
            elif "," in tail:
                decimal_votes[","] = decimal_votes.get(",", 0) + 1

    thousand = max(thousand_votes, key=lambda k: thousand_votes[k]) if thousand_votes else None
    decimal = max(decimal_votes, key=lambda k: decimal_votes[k]) if decimal_votes else None
    return thousand, decimal


def _collect_replaced_chars(col_node: exp.Expression) -> set[str] | None:
    """Collect literal characters REPLACEd around a column, or None when REPLACE is absent."""
    replaced: set[str] = set()
    found = False
    parent = col_node.parent
    while parent is not None and not isinstance(parent, exp.Select):
        if isinstance(parent, exp.Replace):
            found = True
            source_arg = parent.args.get("expression")
            if isinstance(source_arg, exp.Literal):
                replaced.add(source_arg.name)
        elif isinstance(parent, exp.Anonymous) and parent.name.lower() == "replace":
            found = True
            literal_args = [a for a in parent.expressions if isinstance(a, exp.Literal)]
            if len(literal_args) >= 2:
                replaced.add(literal_args[1].name)
        parent = parent.parent
    return replaced if found else None


class SQLValidator:
    """Static SQL analyzer enforcing SQLite syntax validity and schema-grounded domain rules."""

    @classmethod
    def has_not_null_predicate(cls, root: exp.Expression, col_name: str) -> bool:
        """Check whether root expression contains an explicit NOT NULL predicate on the column."""
        clean_target = col_name.strip("\"'` ").lower()

        for clause in root.find_all(exp.Where, exp.Having):
            for not_node in clause.find_all(exp.Not):
                is_node = not_node.find(exp.Is)
                if is_node and isinstance(is_node.expression, exp.Null):
                    ref_name = _extract_column_name(is_node.this).lower()
                    if ref_name == clean_target:
                        return True

            for is_node in clause.find_all(exp.Is):
                if isinstance(is_node.expression, exp.Not) and isinstance(is_node.expression.this, exp.Null):
                    ref_name = _extract_column_name(is_node.this).lower()
                    if ref_name == clean_target:
                        return True

        return False

    @classmethod
    def validate(
        cls,
        raw_sql: str,
        target_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        as_of_date: str | None = None,
        intent: QueryIntent | None = None,
    ) -> ValidationResult:
        """Validate candidate SQL string against SQLite dialect, AST scope, and schema metadata.

        Args:
            raw_sql: Candidate SQL statement.
            target_type: Declared DataMate answer type.
            metadata: Processed metadata mapping 'table.column' to column profiles.
            as_of_date: Anchor date of the question, used for date-window validation.
            intent: Optional QueryIntent carrying the deterministically resolved periods.
        """
        sql = clean_sql_string(raw_sql)
        if not sql:
            return ValidationResult(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        code="OUTPUT_EMPTY",
                        message="SQL query string is empty.",
                        severity="blocking",
                    )
                ],
            )

        # 1. Parse SQL
        try:
            parsed_list = sqlglot.parse(sql, read="sqlite")
        except ParseError as exc:
            return ValidationResult(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        code="SQL_SYNTAX",
                        message=f"SQL syntax error: {exc}",
                        severity="blocking",
                    )
                ],
            )
        except Exception as exc:  # noqa: BLE001
            return ValidationResult(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        code="SQL_SYNTAX",
                        message=f"Failed to parse SQL: {exc}",
                        severity="blocking",
                    )
                ],
            )

        if not parsed_list or parsed_list[0] is None:
            return ValidationResult(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        code="SQL_SYNTAX",
                        message="Failed to parse any valid SQL statement.",
                        severity="blocking",
                    )
                ],
            )

        if len(parsed_list) > 1:
            return ValidationResult(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        code="SQL_MULTIPLE_STATEMENTS",
                        message="Multiple SQL statements detected. Only a single SELECT query is permitted.",
                        severity="blocking",
                    )
                ],
            )

        parsed = parsed_list[0]
        issues: list[ValidationIssue] = []

        # 2. Safety check: must be a SELECT or UNION query
        if not isinstance(parsed, (exp.Select, exp.Union)):
            return ValidationResult(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        code="SQL_NOT_SELECT",
                        message=f"Non-SELECT statement type '{type(parsed).__name__}' is strictly forbidden.",
                        severity="blocking",
                    )
                ],
            )

        # Check for data-mutating expressions
        if any(parsed.find_all(exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter)):
            return ValidationResult(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        code="SQL_NOT_READ_ONLY",
                        message="Data-mutating statements (INSERT, UPDATE, DELETE, DROP) are strictly forbidden.",
                        severity="blocking",
                    )
                ],
            )

        # 3. Old Check 1 (demoted to warning): Ascending order on field f must have NOT NULL predicate
        aggregate_aliases: set[str] = set()
        if isinstance(parsed, exp.Select):
            for s in parsed.selects:
                if isinstance(s, exp.Alias) and any(s.this.find_all(exp.AggFunc)):
                    aggregate_aliases.add(s.alias.lower())

        tables_meta, col_info, all_cols_meta = _extract_schema_lookup(metadata)

        for order_expr in parsed.find_all(exp.Order):
            for ordered in order_expr.find_all(exp.Ordered):
                is_desc = ordered.args.get("desc")
                if not is_desc:
                    target_col_node = ordered.this
                    if any(target_col_node.find_all(exp.AggFunc)):
                        continue

                    col_name = _extract_column_name(target_col_node)
                    if (
                        col_name
                        and not col_name.isdigit()
                        and col_name.lower() not in aggregate_aliases
                        and (isinstance(target_col_node, exp.Column) or list(target_col_node.find_all(exp.Column)))
                    ):
                        target_field = target_col_node.name if isinstance(target_col_node, exp.Column) else col_name
                        # Only flag warning if metadata reports null_count > 0
                        matched_metas = [
                            v for k, v in col_info.items()
                            if k.endswith(f".{target_field.lower()}") or k == target_field.lower()
                        ] if col_info else []
                        should_warn = bool(matched_metas) and any(m.get("null_count", 0) > 0 for m in matched_metas)

                        if should_warn and not cls.has_not_null_predicate(parsed, target_field):
                            issues.append(
                                ValidationIssue(
                                    code="ORDER_ASC_NULLABLE",
                                    message=(
                                        f"Query orders by field '{target_field}' ASC without ensuring '{target_field} IS NOT NULL'. "
                                        f"In SQLite, NULL values sort first before all other values."
                                    ),
                                    severity="warning",
                                    column=target_field,
                                    hint=f"Add 'WHERE \"{target_field}\" IS NOT NULL'.",
                                )
                            )

        # 4. Old Check 2 (demoted to warning): Aggregate MIN(f) must have NOT NULL predicate
        for min_expr in parsed.find_all(exp.Min):
            col_node = min_expr.this
            col_name = _extract_column_name(col_node)
            if col_name and not col_name.isdigit() and col_name != "*":
                # Only flag warning if metadata reports null_count > 0
                matched_metas = [
                    v for k, v in col_info.items()
                    if k.endswith(f".{col_name.lower()}") or k == col_name.lower()
                ] if col_info else []
                should_warn = bool(matched_metas) and any(m.get("null_count", 0) > 0 for m in matched_metas)

                if should_warn and not cls.has_not_null_predicate(parsed, col_name):
                    issues.append(
                        ValidationIssue(
                            code="MIN_NULLABLE",
                            message=(
                                f"Aggregate MIN(\"{col_name}\") detected without a NOT NULL predicate on '{col_name}'."
                            ),
                            severity="warning",
                            column=col_name,
                            hint=f"Add 'WHERE \"{col_name}\" IS NOT NULL' to guarantee clean aggregation.",
                        )
                    )

        # 5. Old Check 4 (demoted to warning): String concatenation on fields in SELECT (skip for date)
        if target_type != "date" and isinstance(parsed, exp.Select):
            for select_expr in parsed.selects:
                concat_nodes = list(select_expr.find_all(exp.DPipe, exp.Concat))
                for anon in select_expr.find_all(exp.Anonymous):
                    if anon.name.lower() == "concat":
                        concat_nodes.append(anon)

                for c_node in concat_nodes:
                    # Ignore date formatting expressions using SUBSTR, strftime, or date
                    is_date_fmt = any(c_node.find_all(exp.Substring, exp.Date, exp.TimeToStr)) or any(
                        a.name.lower() in ("substr", "date", "datetime", "strftime")
                        for a in c_node.find_all((exp.Func, exp.Anonymous))
                    )
                    if not is_date_fmt and list(c_node.find_all(exp.Column)):
                        issues.append(
                            ValidationIssue(
                                code="CONCAT_IN_SELECT",
                                message="Query performs string concatenation ('||' or CONCAT) on fields in SELECT.",
                                severity="warning",
                                hint="Return the fields individually without concatenating them, unless formatting date.",
                            )
                        )
                        break

        # 6. Schema-Grounded AST Scope Checks (blocking when metadata present)
        if tables_meta:
            norm_tables = {t.lower(): t for t in tables_meta}
            norm_cols = {
                t.lower(): {c.lower(): c for c in cols}
                for t, cols in tables_meta.items()
            }

            try:
                for scope in traverse_scope(parsed):
                    # Collect CTE names visible in this scope
                    cte_names = {c.alias.lower() for c in scope.ctes if c.alias}
                    derived_sources = {
                        s.lower() for s, src in scope.sources.items()
                        if not isinstance(src, exp.Table)
                    }

                    # A. Check table existence
                    for tbl_node in scope.find_all(exp.Table):
                        tbl_name = tbl_node.name
                        if not tbl_name:
                            continue
                        if tbl_name.lower() in cte_names or tbl_name.lower() in derived_sources:
                            continue
                        if tbl_name.lower() not in norm_tables:
                            valid_tables = list(tables_meta.keys())
                            suggestions = [
                                t for t in valid_tables
                                if tbl_name.lower() in t.lower() or t.lower() in tbl_name.lower()
                            ][:3]
                            hint = f"Gợi ý: {suggestions}" if suggestions else f"Các bảng hợp lệ: {valid_tables}"
                            issues.append(
                                ValidationIssue(
                                    code="TABLE_NOT_FOUND",
                                    message=f"Bảng '{tbl_name}' không tồn tại trong cơ sở dữ liệu.",
                                    severity="blocking",
                                    hint=hint,
                                )
                            )

                    # B. Check column existence and qualifiers
                    for col_node in scope.columns:
                        col_name = col_node.name
                        qualifier = col_node.table
                        if not col_name or col_name == "*" or col_name.lower() in ("rowid", "_rowid_", "oid"):
                            continue
                        if _is_pseudo_column_literal(col_name):
                            continue

                        if qualifier:
                            source = scope.sources.get(qualifier)
                            if source is None:
                                # Qualifier not found in scope sources
                                if qualifier.lower() not in norm_tables:
                                    issues.append(
                                        ValidationIssue(
                                            code="UNKNOWN_QUALIFIER",
                                            message=f"Bí danh hoặc qualifier '{qualifier}' không tồn tại trong phạm vi truy vấn.",
                                            severity="blocking",
                                            hint=f"Các nguồn khả dụng: {list(scope.sources.keys())}",
                                        )
                                    )
                                else:
                                    actual_tbl = norm_tables[qualifier.lower()]
                                    tbl_cols = norm_cols.get(actual_tbl.lower(), {})
                                    if col_name.lower() not in tbl_cols:
                                        other_tbls = all_cols_meta.get(col_name.lower(), [])
                                        hint = (
                                            f"Cột '{col_name}' thuộc bảng '{other_tbls[0]}', không phải '{actual_tbl}'."
                                            if other_tbls
                                            else f"Các cột trong bảng '{actual_tbl}': {list(tables_meta[actual_tbl])[:5]}"
                                        )
                                        issues.append(
                                            ValidationIssue(
                                                code="COLUMN_NOT_FOUND",
                                                message=f"Cột '{col_name}' không tồn tại trong bảng '{actual_tbl}'.",
                                                severity="blocking",
                                                column=col_name,
                                                hint=hint,
                                            )
                                        )
                            elif isinstance(source, exp.Table):
                                real_tbl = source.name
                                if real_tbl and real_tbl.lower() in norm_tables:
                                    actual_tbl = norm_tables[real_tbl.lower()]
                                    tbl_cols = norm_cols.get(actual_tbl.lower(), {})
                                    if col_name.lower() not in tbl_cols:
                                        other_tbls = all_cols_meta.get(col_name.lower(), [])
                                        hint = (
                                            f"Cột '{col_name}' thuộc bảng '{other_tbls[0]}', không phải '{actual_tbl}'."
                                            if other_tbls
                                            else f"Các cột trong bảng '{actual_tbl}': {list(tables_meta[actual_tbl])[:5]}"
                                        )
                                        issues.append(
                                            ValidationIssue(
                                                code="COLUMN_NOT_FOUND",
                                                message=f"Cột '{col_name}' không tồn tại trong bảng '{actual_tbl}'.",
                                                severity="blocking",
                                                column=col_name,
                                                hint=hint,
                                            )
                                        )
                        else:
                            # Unqualified column
                            phys_tables = [
                                s.name for s in scope.sources.values()
                                if isinstance(s, exp.Table) and s.name
                            ]
                            if len(phys_tables) == 1 and phys_tables[0].lower() in norm_tables:
                                actual_tbl = norm_tables[phys_tables[0].lower()]
                                tbl_cols = norm_cols.get(actual_tbl.lower(), {})
                                if col_name.lower() not in tbl_cols:
                                    other_tbls = all_cols_meta.get(col_name.lower(), [])
                                    hint = (
                                        f"Cột '{col_name}' thuộc bảng '{other_tbls[0]}', không phải '{actual_tbl}'."
                                        if other_tbls
                                        else f"Các cột trong bảng '{actual_tbl}': {list(tables_meta[actual_tbl])[:5]}"
                                    )
                                    issues.append(
                                        ValidationIssue(
                                            code="COLUMN_NOT_FOUND",
                                            message=f"Cột '{col_name}' không tồn tại trong bảng '{actual_tbl}'.",
                                            severity="blocking",
                                            column=col_name,
                                            hint=hint,
                                        )
                                    )
                            elif len(phys_tables) > 1:
                                exists_any = any(
                                    col_name.lower() in norm_cols.get(pt.lower(), {})
                                    for pt in phys_tables if pt.lower() in norm_tables
                                )
                                if not exists_any:
                                    issues.append(
                                        ValidationIssue(
                                            code="COLUMN_NOT_FOUND",
                                            message=f"Cột '{col_name}' không tồn tại trong bất kỳ bảng nào trong scope ({phys_tables}).",
                                            severity="blocking",
                                            column=col_name,
                                        )
                                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug("schema grounding checks skipped: %s", exc)

        # 7. Semantic Metadata Checks: DATE_FORMAT_MISMATCH & LITERAL_NOT_IN_ENUM & MONEY_SCALE_RISK
        if col_info:
            for col_node in parsed.find_all(exp.Column):
                col_name = col_node.name
                if not col_name or col_name == "*":
                    continue

                qualifier = col_node.table
                matched_meta: dict[str, Any] | None = None
                if qualifier:
                    matched_meta = col_info.get(f"{qualifier.lower()}.{col_name.lower()}")
                if matched_meta is None:
                    candidates = [
                        v for k, v in col_info.items()
                        if k.endswith(f".{col_name.lower()}") or k == col_name.lower()
                    ]
                    if len(candidates) == 1:
                        matched_meta = candidates[0]

                if not matched_meta:
                    continue

                data_format = str(matched_meta.get("data_format", "")).upper()
                is_date = (
                    data_format in (
                        "SLASH_DATE",
                        "SLASH_DATE_SHORT",
                        "VN_DATE_ABBR",
                        "EXCEL_SERIAL",
                        "DATE_MIXED",
                        "ISO_DATE",
                        "DATETIME",
                        "DATE",
                    )
                    or "DATE" in data_format
                    or "TIME" in data_format
                )

                # A. DATE_FORMAT_MISMATCH
                if is_date and not _is_date_normalization_node(col_node):
                    parent = col_node.parent
                    while parent is not None and not isinstance(parent, (exp.Binary, exp.Between, exp.In, exp.Select)):
                        parent = parent.parent

                    # 1. Range comparison on non-ISO date string
                    is_non_iso = data_format in ("SLASH_DATE", "SLASH_DATE_SHORT", "VN_DATE_ABBR", "DATE_MIXED")
                    if isinstance(parent, (exp.Between, exp.GT, exp.GTE, exp.LT, exp.LTE)):
                        if is_non_iso:
                            issues.append(
                                ValidationIssue(
                                    code="DATE_FORMAT_MISMATCH",
                                    message=(
                                        f"Cột ngày '{col_name}' định dạng '{data_format}' được so sánh khoảng trực tiếp "
                                        f"mà không chuẩn hóa về ISO YYYY-MM-DD."
                                    ),
                                    severity="blocking",
                                    column=col_name,
                                    hint=(
                                        "Chuẩn hóa cột ngày về ISO YYYY-MM-DD bằng "
                                        "(SUBSTR(\"cột\", 7, 4) || '-' || SUBSTR(\"cột\", 4, 2) || '-' || SUBSTR(\"cột\", 1, 2)) "
                                        "trước khi so sánh khoảng."
                                    ),
                                )
                            )
                        else:
                            literals = _extract_comparison_strings(parent)
                            for lit in literals:
                                if data_format == "ISO_DATE" and re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", lit):
                                    issues.append(
                                        ValidationIssue(
                                            code="DATE_FORMAT_MISMATCH",
                                            message=f"Cột '{col_name}' là ISO_DATE nhưng so sánh với literal DD/MM/YYYY '{lit}'.",
                                            severity="blocking",
                                            column=col_name,
                                            hint="Dùng định dạng literal 'YYYY-MM-DD' khi so sánh với ISO_DATE.",
                                        )
                                    )

                    # 2. Equality / In / Binary comparison literal mismatch
                    if isinstance(parent, (exp.EQ, exp.NEQ, exp.In)):
                        literals = _extract_comparison_strings(parent)
                        for lit in literals:
                            if is_non_iso and re.match(r"^\d{4}-\d{2}-\d{2}$", lit):
                                issues.append(
                                    ValidationIssue(
                                        code="DATE_FORMAT_MISMATCH",
                                        message=(
                                            f"Cột ngày '{col_name}' định dạng '{data_format}' so sánh trực tiếp "
                                            f"với chuỗi ISO '{lit}' mà không chuẩn hóa cột."
                                        ),
                                        severity="blocking",
                                        column=col_name,
                                        hint=(
                                            "Bắt buộc chuẩn hóa cột ngày về ISO YYYY-MM-DD bằng "
                                            "(SUBSTR(\"cột\", 7, 4) || '-' || SUBSTR(\"cột\", 4, 2) || '-' || SUBSTR(\"cột\", 1, 2)) "
                                            "trước khi so sánh."
                                        ),
                                    )
                                )
                            elif data_format == "ISO_DATE" and re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", lit):
                                issues.append(
                                    ValidationIssue(
                                        code="DATE_FORMAT_MISMATCH",
                                        message=f"Cột '{col_name}' là ISO_DATE nhưng so sánh với literal DD/MM/YYYY '{lit}'.",
                                        severity="blocking",
                                        column=col_name,
                                        hint="Dùng định dạng literal 'YYYY-MM-DD' khi so sánh với ISO_DATE.",
                                    )
                                )

                # B. LITERAL_NOT_IN_ENUM (warning)
                distinct_count = matched_meta.get("distinct_count")
                if distinct_count and 0 < distinct_count <= 10:
                    samples = _get_column_samples(matched_meta)
                    if samples and len(samples) >= distinct_count:
                        parent = col_node.parent
                        while parent is not None and not isinstance(parent, (exp.EQ, exp.NEQ, exp.In, exp.Select)):
                            parent = parent.parent

                        if isinstance(parent, (exp.EQ, exp.NEQ, exp.In)):
                            norm_samples = {_normalize_text(s) for s in samples}
                            literals = _extract_comparison_strings(parent)
                            for lit in literals:
                                if _normalize_text(lit) not in norm_samples:
                                    issues.append(
                                        ValidationIssue(
                                            code="LITERAL_NOT_IN_ENUM",
                                            message=f"Giá trị so sánh '{lit}' không có trong tập giá trị mẫu của cột '{col_name}'.",
                                            severity="warning",
                                            column=col_name,
                                            hint=f"Các giá trị mẫu hợp lệ: {sorted(samples)[:5]}",
                                        )
                                    )

                # C. MONEY_SCALE_RISK (warning)
                if data_format == "CURRENCY":
                    samples = _get_column_samples(matched_meta)
                    has_decimal_risk = any(
                        re.search(r"\.\d{2,}$", str(s)) and ("," in str(s) or "." in str(s))
                        for s in samples
                    ) or "DECIMAL" in str(matched_meta.get("currency_subtype", "")).upper()

                    if has_decimal_risk:
                        col_parent = col_node.parent
                        replaces: list[exp.Anonymous] = []
                        while col_parent is not None and not isinstance(col_parent, exp.Select):
                            if isinstance(col_parent, exp.Anonymous) and col_parent.name.lower() == "replace":
                                replaces.append(col_parent)
                            col_parent = col_parent.parent

                        replaced_chars = {
                            l_node.name for r in replaces for l_node in r.find_all(exp.Literal)
                            if l_node.is_string
                        }
                        if "." in replaced_chars and "," in replaced_chars:
                            issues.append(
                                ValidationIssue(
                                    code="MONEY_SCALE_RISK",
                                    message=(
                                        f"Cột tiền tệ '{col_name}' có phần thập phân nhưng biểu thức SQL REPLACE cả '.' và ',', "
                                        f"có nguy cơ phóng đại giá trị lên 10.000 lần."
                                    ),
                                    severity="warning",
                                    column=col_name,
                                    hint="Chỉ REPLACE dấu phân cách nghìn; phần thập phân làm sạch trước khi CAST.",
                                )
                            )

        # 8. Target Type Shape Checks (blocking & warning)
        if target_type and isinstance(parsed, exp.Select):
            projections = parsed.selects
            has_star = any(
                isinstance(p, exp.Star) or (isinstance(p, exp.Column) and p.name == "*")
                for p in projections
            )
            num_proj = len(projections)

            if target_type in ("number", "percent", "string", "date", "boolean", "list"):
                if num_proj > 1 or has_star:
                    issues.append(
                        ValidationIssue(
                            code="SHAPE_MISMATCH",
                            message=(
                                f"Truy vấn trả về {num_proj} cột hoặc '*', nhưng target_type là '{target_type}' "
                                f"(chỉ cho phép đúng 1 cột)."
                            ),
                            severity="blocking",
                            hint="Chỉ SELECT 1 cột duy nhất ở mệnh đề SELECT ngoài cùng.",
                        )
                    )
            elif target_type == "table" and num_proj == 1 and not has_star:
                issues.append(
                    ValidationIssue(
                        code="SHAPE_MISMATCH",
                        message="Truy vấn chỉ trả về 1 cột đơn, nhưng target_type là 'table' (yêu cầu cấu trúc bảng nhiều cột).",
                        severity="blocking",
                        hint="Mệnh đề SELECT cho kiểu 'table' cần trả về ít nhất 2 cột hoặc SELECT *.",
                    )
                )

            if target_type in ("number", "percent"):
                has_agg = any(parsed.find_all(exp.AggFunc))
                limit_node = parsed.find(exp.Limit)
                has_limit_1 = limit_node is not None and str(limit_node.expression).strip() == "1"
                has_group = bool(parsed.find(exp.Group))
                has_from = bool(parsed.find(exp.From))

                if has_from and not has_agg and not has_group and not has_limit_1:
                    issues.append(
                        ValidationIssue(
                            code="SHAPE_SUSPECT_NO_AGGREGATE",
                            message=(
                                f"Target_type là '{target_type}' nhưng câu truy vấn không có hàm tổng hợp "
                                f"(SUM, COUNT, AVG, MIN, MAX) hay LIMIT 1."
                            ),
                            severity="warning",
                            hint="Bổ sung hàm tổng hợp thích hợp hoặc LIMIT 1 nếu tính toán giá trị đơn vị.",
                        )
                    )

            # Check rank tie-break
            order_node = parsed.find(exp.Order)
            limit_node = parsed.find(exp.Limit)
            if order_node:
                ordered_list = list(order_node.find_all(exp.Ordered))
                if (limit_node or target_type == "list") and len(ordered_list) == 1:
                    issues.append(
                        ValidationIssue(
                            code="RANK_TIEBREAK_MISSING",
                            message="Truy vấn xếp hạng chỉ có 1 khóa sắp xếp, thiếu khóa tie-break mã/ID.",
                            severity="warning",
                            hint="Thêm khóa phụ duy nhất (ví dụ: ORDER BY ... DESC, T1.\"Mã ...\" ASC) để kết quả xác định.",
                        )
                    )

        # 9. Temporal window checks against the deterministic period of the question.
        issues.extend(
            cls._check_temporal_window(
                parsed=parsed,
                as_of_date=as_of_date,
                intent=intent,
            )
        )

        # 10. Currency separator checks and categorical exact-match hints.
        if col_info:
            issues.extend(cls._check_currency_usage(parsed=parsed, col_info=col_info))
            issues.extend(
                cls._check_categorical_exact_match(
                    parsed=parsed,
                    col_info=col_info,
                    strict=bool(getattr(intent, "wants_exact_match", False)),
                )
            )

        # Final verdict: is_valid is True iff no blocking issues exist
        is_valid = not any(getattr(i, "severity", "blocking") == "blocking" for i in issues)
        return ValidationResult(is_valid=is_valid, issues=issues)

    @staticmethod
    def _natural_upper_bounds(period_label: str, period_end: Any) -> set[Any]:
        """Return the calendar boundaries a query may legitimately use around a period end.

        A correct query often bounds the resolved period by its natural calendar unit
        (whole month, whole quarter, whole year) or by the exclusive first day of the next
        unit, e.g. ``>= '2026-09-01' AND < '2026-10-01'`` for a September period. Those
        literals are legal even though they sit after ``as_of_date``.
        """
        bounds = {period_end}
        month_end = period_end.replace(
            day=calendar.monthrange(period_end.year, period_end.month)[1]
        )
        bounds.add(month_end)
        bounds.add(month_end + timedelta(days=1))

        label = (period_label or "").lower()
        if label.startswith("quarter"):
            quarter_end_month = (period_end.month - 1) // 3 * 3 + 3
            quarter_end = period_end.replace(
                month=quarter_end_month,
                day=calendar.monthrange(period_end.year, quarter_end_month)[1],
            )
            bounds.add(quarter_end)
            bounds.add(quarter_end + timedelta(days=1))
        if label.startswith("year"):
            year_end = period_end.replace(month=12, day=31)
            bounds.add(year_end)
            bounds.add(year_end + timedelta(days=1))
        return bounds

    @classmethod
    def _check_temporal_window(
        cls,
        parsed: exp.Expression,
        as_of_date: str | None,
        intent: QueryIntent | None,
    ) -> list[ValidationIssue]:
        """Flag date windows that cannot belong to the question's resolved period.

        The check only runs when the question carries a deterministically resolved period.
        Comparisons are restricted to literals sharing the anchor year, and literals on a
        natural calendar boundary of the period are accepted, so a legitimate historical
        filter or a whole-month bound is never reported as an out-of-period query.
        """
        issues: list[ValidationIssue] = []

        anchor = parse_iso_date(as_of_date)
        if anchor is None:
            return issues

        periods = list(getattr(intent, "periods", ()) or ())
        if not periods:
            return issues

        period_start = parse_iso_date(periods[0].start)
        period_end = parse_iso_date(periods[-1].end)
        if period_start is None or period_end is None:
            return issues

        candidate_dates: list[Any] = []
        for lit in parsed.find_all(exp.Literal):
            raw_val = str(lit.this)
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", raw_val):
                continue
            parsed_date = parse_iso_date(raw_val)
            if parsed_date is not None:
                candidate_dates.append(parsed_date)

        if not candidate_dates:
            return issues

        upper_bounds = cls._natural_upper_bounds(periods[-1].label, period_end)
        latest_legal = max(upper_bounds)
        beyond_bounds = sorted({d for d in candidate_dates if d > latest_legal})
        if beyond_bounds:
            issues.append(
                ValidationIssue(
                    code="DATE_RANGE_OUT_OF_PERIOD",
                    message=(
                        f"Câu SQL lọc tới ngày {beyond_bounds[-1].isoformat()}, vượt quá "
                        f"biên hợp lệ {latest_legal.isoformat()} của kỳ "
                        f"{period_start.isoformat()}..{period_end.isoformat()}."
                    ),
                    severity="blocking",
                    hint=(
                        f"Dùng đúng kỳ {period_start.isoformat()} đến {period_end.isoformat()} "
                        "theo <temporal_anchor>."
                    ),
                )
            )
            return issues

        same_year = [d for d in candidate_dates if d.year == anchor.year]
        if not same_year:
            return issues

        window_start, window_end = min(same_year), max(same_year)
        intersects = window_start <= period_end and window_end >= period_start
        if not intersects:
            issues.append(
                ValidationIssue(
                    code="DATE_RANGE_OUT_OF_PERIOD",
                    message=(
                        f"Khoảng ngày {window_start.isoformat()}..{window_end.isoformat()} "
                        f"không giao với kỳ đã xác định {period_start.isoformat()}..{period_end.isoformat()}."
                    ),
                    severity="blocking",
                    hint=(
                        f"Dùng đúng kỳ {period_start.isoformat()} đến {period_end.isoformat()} "
                        "theo <temporal_anchor>."
                    ),
                )
            )
        return issues

    @classmethod
    def _check_currency_usage(
        cls,
        parsed: exp.Expression,
        col_info: dict[str, dict[str, Any]],
    ) -> list[ValidationIssue]:
        """Detect currency cleaning expressions that keep the real thousands separator."""
        issues: list[ValidationIssue] = []
        for col_node in parsed.find_all(exp.Column):
            col_name = col_node.name
            if not col_name or col_name == "*":
                continue
            matched_meta = cls._match_column_meta(col_node, col_info)
            if not matched_meta:
                continue
            if str(matched_meta.get("data_format", "")).upper() != "CURRENCY":
                continue

            samples = _get_column_samples(matched_meta)
            thousand, decimal = infer_currency_separators(samples)
            replaced = _collect_replaced_chars(col_node)

            if replaced is None:
                if thousand is not None:
                    issues.append(
                        ValidationIssue(
                            code="CURRENCY_SEPARATOR_MISMATCH",
                            message=(
                                f"Cột tiền tệ '{col_name}' là chuỗi có dấu phân cách nghìn "
                                f"'{thousand}' nhưng không được làm sạch trước khi tính toán."
                            ),
                            severity="warning",
                            column=col_name,
                            hint=(
                                f"Bọc giá trị trong CAST(REPLACE(REPLACE(\"{col_name}\", '{thousand}', ''), "
                                "'đ', '') AS INTEGER)."
                            ),
                        )
                    )
                continue

            if thousand is not None and thousand not in replaced:
                issues.append(
                    ValidationIssue(
                        code="CURRENCY_SEPARATOR_MISMATCH",
                        message=(
                            f"Cột tiền tệ '{col_name}' dùng '{thousand}' làm dấu phân cách nghìn "
                            f"nhưng SQL không REPLACE ký tự này, nên giá trị sau CAST bị sai quy mô."
                        ),
                        severity="blocking",
                        column=col_name,
                        hint=f"Thêm REPLACE(\"{col_name}\", '{thousand}', '') trước khi CAST.",
                    )
                )
            if decimal is not None and decimal in replaced and thousand is not None:
                issues.append(
                    ValidationIssue(
                        code="MONEY_SCALE_RISK",
                        message=(
                            f"Cột tiền tệ '{col_name}' có phần thập phân với dấu '{decimal}' "
                            "nhưng SQL xóa cả phần thập phân lẫn phần nghìn."
                        ),
                        severity="warning",
                        column=col_name,
                        hint="Chỉ làm sạch dấu phân cách nghìn, xử lý phần thập phân riêng.",
                    )
                )
        return issues

    @classmethod
    def _check_categorical_exact_match(
        cls,
        parsed: exp.Expression,
        col_info: dict[str, dict[str, Any]],
        strict: bool = False,
    ) -> list[ValidationIssue]:
        """Require equality instead of LIKE when the searched value exists verbatim.

        Args:
            parsed: Parsed SQL expression.
            col_info: Column metadata lookup.
            strict: When the question asks for an exact value the finding is blocking, so
                the repair loop must rewrite the LIKE into an equality comparison.
        """
        issues: list[ValidationIssue] = []
        for node in parsed.find_all(exp.Like):
            col_node = node.this if isinstance(node.this, exp.Column) else None
            pattern_node = node.expression
            if col_node is None or not isinstance(pattern_node, exp.Literal):
                continue
            pattern = str(pattern_node.this or "")
            stripped = pattern.strip("%").strip()
            if not stripped or "%" in stripped:
                continue

            matched_meta = cls._match_column_meta(col_node, col_info)
            if not matched_meta:
                continue
            if str(matched_meta.get("data_format", "")).upper() != "STRING":
                continue

            samples = {_normalize_text(s) for s in _get_column_samples(matched_meta)}
            if _normalize_text(stripped) in samples:
                issues.append(
                    ValidationIssue(
                        code="LITERAL_PARTIAL_MATCH",
                        message=(
                            f"Giá trị '{stripped}' tồn tại nguyên vẹn trong cột '{col_node.name}' "
                            "nhưng đang được so sánh bằng LIKE."
                        ),
                        severity="blocking" if strict else "warning",
                        column=col_node.name,
                        hint=(
                            f"Dùng LOWER(\"{col_node.name}\") = LOWER('{stripped}') "
                            "để tránh khớp một phần ngoài ý muốn."
                        ),
                    )
                )
        return issues

    @staticmethod
    def _match_column_meta(
        col_node: exp.Column,
        col_info: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Resolve the metadata entry of a column node, preferring its table qualifier."""
        qualifier = col_node.table
        col_name = col_node.name
        if qualifier:
            direct = col_info.get(f"{qualifier.lower()}.{col_name.lower()}")
            if direct is not None:
                return direct
        candidates = [
            v for k, v in col_info.items()
            if k.endswith(f".{col_name.lower()}") or k == col_name.lower()
        ]
        return candidates[0] if len(candidates) == 1 else None


__all__ = [
    "SQLValidator",
    "clean_sql_string",
    "infer_currency_separators",
    "is_truncated_json_output",
    "parse_structured_sql_output",
    "parse_structured_sql_output_with_reasoning",
]
