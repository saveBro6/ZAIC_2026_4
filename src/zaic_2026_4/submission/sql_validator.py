"""Static SQL validation and anti-pattern detection using SQLglot for SQLite."""

from __future__ import annotations

import json
import re
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from zaic_2026_4.submission.models import VALID_DATAMATE_TYPES, ValidationResult


def clean_sql_string(raw_sql: str) -> str:
    """Extract clean SQL string by stripping markdown code fences and whitespace."""
    sql = raw_sql.strip()
    if sql.startswith("```"):
        # Remove opening code fence
        lines = sql.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        sql = "\n".join(lines).strip()
    return sql


def parse_structured_sql_output(
    raw_output: str,
    default_type: str = "number",
) -> tuple[str, str, list[str]]:
    """Parse JSON Structured Output from LLM into (sql, target_type, validation_issues).

    Handles direct JSON, markdown-wrapped JSON, or regex extraction fallback.
    If the parsed target_type is not in VALID_DATAMATE_TYPES, records a descriptive issue.
    """
    text = raw_output.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    parsed_dict: dict[str, Any] | None = None
    issues: list[str] = []

    if text.startswith("{") and text.endswith("}"):
        try:
            parsed_dict = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            parsed_dict = None

    if parsed_dict is None:
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                parsed_dict = json.loads(json_match.group(0))
            except (json.JSONDecodeError, ValueError):
                parsed_dict = None

    if parsed_dict is not None and isinstance(parsed_dict, dict):
        raw_sql = str(parsed_dict.get("sql", ""))
        raw_type = str(parsed_dict.get("type", default_type)).strip().lower()
        clean_sql = clean_sql_string(raw_sql)

        if raw_type not in VALID_DATAMATE_TYPES:
            issues.append(
                f"Trường 'type' ({raw_type!r}) không hợp lệ theo quy chế DataMate. "
                f"Giá trị bắt buộc phải là 1 trong {VALID_DATAMATE_TYPES}."
            )

        return clean_sql, raw_type, issues

    # Backward-compatibility fallback for tests or engines outputting raw SQL
    cleaned = clean_sql_string(raw_output)
    if not cleaned:
        issues.append("Đầu ra của mô hình rỗng, không chứa truy vấn SQL hay đối tượng JSON hợp lệ.")
    return cleaned, default_type, issues


def _extract_column_name(node: Any) -> str:
    """Extract column name identifier from an expression or node."""
    if isinstance(node, exp.Column):
        return node.name
    if isinstance(node, exp.Identifier):
        return node.name
    return str(node).strip("\"'` ")


class SQLValidator:
    """Static SQL analyzer enforcing SQLite syntax validity and domain anti-pattern checks."""

    @classmethod
    def has_not_null_predicate(cls, root: exp.Expression, col_name: str) -> bool:
        """Check whether root expression contains an explicit NOT NULL predicate on the column.

        Matches 'col IS NOT NULL' or 'NOT (col IS NULL)' across WHERE and HAVING clauses.
        """
        clean_target = col_name.strip("\"'` ").lower()

        # Check in WHERE and HAVING clauses
        for clause in root.find_all(exp.Where, exp.Having):
            for not_node in clause.find_all(exp.Not):
                is_node = not_node.find(exp.Is)
                if is_node and isinstance(is_node.expression, exp.Null):
                    ref_name = _extract_column_name(is_node.this).lower()
                    if ref_name == clean_target:
                        return True

            # Also check direct Is expressions where expression is Not(Null) if dialect converts it
            for is_node in clause.find_all(exp.Is):
                if isinstance(is_node.expression, exp.Not) and isinstance(is_node.expression.this, exp.Null):
                    ref_name = _extract_column_name(is_node.this).lower()
                    if ref_name == clean_target:
                        return True

        return False

    @classmethod
    def validate(cls, raw_sql: str) -> ValidationResult:
        """Validate candidate SQL string against SQLite dialect and specific domain rules.

        Checks:
        1. Syntax validity via sqlglot SQLite parser.
        2. Query safety (read-only SELECT / WITH ... SELECT).
        3. Check 1: Ascending order on field f must have a NOT NULL predicate on f.
        4. Check 2: Aggregate MIN(f) must have a NOT NULL predicate on f.
        5. Check 3: Nested subquery min/max instead of ORDER BY ... LIMIT 1.
        6. Check 4: String concatenation on fields in SELECT instead of individual fields.
        """
        sql = clean_sql_string(raw_sql)
        if not sql:
            return ValidationResult(is_valid=False, issues=["SQL query string is empty."])

        # 1. Parse SQL
        try:
            parsed_list = sqlglot.parse(sql, read="sqlite")
        except ParseError as exc:
            return ValidationResult(is_valid=False, issues=[f"SQL syntax error: {exc}"])
        except Exception as exc:  # noqa: BLE001
            return ValidationResult(is_valid=False, issues=[f"Failed to parse SQL: {exc}"])

        if not parsed_list or parsed_list[0] is None:
            return ValidationResult(is_valid=False, issues=["Failed to parse any valid SQL statement."])

        if len(parsed_list) > 1:
            return ValidationResult(
                is_valid=False,
                issues=["Multiple SQL statements detected. Only a single SELECT query is permitted."],
            )

        parsed = parsed_list[0]
        issues: list[str] = []

        # 2. Safety check: must be a SELECT or UNION query
        if not isinstance(parsed, (exp.Select, exp.Union)):
            return ValidationResult(
                is_valid=False,
                issues=[f"Non-SELECT statement type '{type(parsed).__name__}' is strictly forbidden."],
            )

        # Check for data-mutating expressions
        if any(parsed.find_all(exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter)):
            return ValidationResult(
                is_valid=False,
                issues=["Data-mutating statements (INSERT, UPDATE, DELETE, DROP) are strictly forbidden."],
            )

        # 3. Check 1: Ascending order on field f must have NOT NULL predicate
        # Collect aliases of aggregate expressions in SELECT (e.g. SELECT SUM(x) AS total)
        aggregate_aliases: set[str] = set()
        if isinstance(parsed, exp.Select):
            for s in parsed.selects:
                if isinstance(s, exp.Alias) and any(s.this.find_all(exp.AggFunc)):
                    aggregate_aliases.add(s.alias.lower())

        for order_expr in parsed.find_all(exp.Order):
            for ordered in order_expr.find_all(exp.Ordered):
                is_desc = ordered.args.get("desc")
                # In SQLite/SQLglot, desc=False or desc=None means ASC
                if not is_desc:
                    target_col_node = ordered.this
                    # Ignore order by expressions that contain aggregate functions (e.g. SUM, COUNT)
                    if any(target_col_node.find_all(exp.AggFunc)):
                        continue

                    col_name = _extract_column_name(target_col_node)
                    # Ignore ordering on literal numbers (e.g. ORDER BY 1 ASC) or aggregate aliases
                    if (
                        col_name
                        and not col_name.isdigit()
                        and col_name.lower() not in aggregate_aliases
                        and (isinstance(target_col_node, exp.Column) or list(target_col_node.find_all(exp.Column)))
                    ):
                        target_field = target_col_node.name if isinstance(target_col_node, exp.Column) else col_name
                        if not cls.has_not_null_predicate(parsed, target_field):
                            issues.append(
                                f"Query orders by field '{target_field}' ASC without ensuring '{target_field} IS NOT NULL'. "
                                f"In SQLite, NULL values sort first before all other values. "
                                f"Add 'WHERE \"{target_field}\" IS NOT NULL'."
                            )

        # 4. Check 2: Aggregate MIN(f) must have NOT NULL predicate
        for min_expr in parsed.find_all(exp.Min):
            col_node = min_expr.this
            col_name = _extract_column_name(col_node)
            if col_name and not col_name.isdigit() and col_name != "*" and not cls.has_not_null_predicate(parsed, col_name):
                issues.append(
                    f"Aggregate MIN(\"{col_name}\") detected without a NOT NULL predicate on '{col_name}'. "
                    f"Add 'WHERE \"{col_name}\" IS NOT NULL' to guarantee clean aggregation."
                )

        # 5. Check 3: Min/Max nested subquery instead of ORDER BY
        for clause in parsed.find_all(exp.Where, exp.Having):
            for sub_select in clause.find_all(exp.Select):
                if sub_select != parsed and any(sub_select.find_all(exp.Min, exp.Max)):
                    issues.append(
                        "Nested subquery with MIN/MAX detected in WHERE/HAVING clause. "
                        "Refactor to use 'ORDER BY ... LIMIT 1' instead of nested subqueries for min/max."
                    )
                    break

        # 6. Check 4: String concatenation on fields in SELECT
        if isinstance(parsed, exp.Select):
            for select_expr in parsed.selects:
                # Look for string concatenation expressions
                concat_nodes = list(select_expr.find_all(exp.DPipe, exp.Concat))
                # Also check exp.Anonymous where name is CONCAT
                for anon in select_expr.find_all(exp.Anonymous):
                    if anon.name.lower() == "concat":
                        concat_nodes.append(anon)

                for c_node in concat_nodes:
                    # Check if concat involves table columns
                    if list(c_node.find_all(exp.Column)):
                        issues.append(
                            "Query performs string concatenation ('||' or CONCAT) on fields in SELECT. "
                            "Return the fields individually without concatenating them."
                        )
                        break

        return ValidationResult(is_valid=(len(issues) == 0), issues=issues)


__all__ = [
    "SQLValidator",
    "clean_sql_string",
    "parse_structured_sql_output",
]
