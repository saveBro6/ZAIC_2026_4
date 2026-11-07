"""SQL AST Parser for extracting referenced fields and literals.

Uses sqlglot for SQLite dialect AST traversal with a robust regex-based fallback
to ensure fault tolerance against malformed trial SQL completions.
"""

from __future__ import annotations

import re
import unicodedata

import sqlglot
from sqlglot import exp


class SQLASTExtractor:
    """Extracts column fields and literal values from SQL queries."""

    @staticmethod
    def clean_sql(sql_text: str) -> str:
        """Strip markdown code fences, preambles, comments, and extraneous whitespace."""
        if not sql_text:
            return ""

        text = sql_text.strip()
        # Remove markdown code fences like ```sql ... ```
        if "```" in text:
            fence_match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
            if fence_match:
                text = fence_match.group(1).strip()
            else:
                text = text.replace("```sql", "").replace("```", "").strip()

        # Remove single-line comments (-- comment)
        lines = []
        for line in text.splitlines():
            line_stripped = line.strip()
            if line_stripped.startswith("--"):
                continue
            lines.append(line)
        cleaned = "\n".join(lines).strip()

        # Extract beginning from SELECT or WITH if LLM produced explanatory preamble text
        select_match = re.search(r"\b(SELECT|WITH)\b[\s\S]*", cleaned, re.IGNORECASE)
        if select_match:
            cleaned = select_match.group(0).strip()

        # Remove trailing semicolon
        if cleaned.endswith(";"):
            cleaned = cleaned[:-1].strip()

        return cleaned

    def parse_fields_and_literals(
        self,
        sql_query: str,
        known_columns: set[str] | dict[str, list[str]] | None = None,
    ) -> tuple[set[str], set[str]]:
        """Parse SQL query into referenced fields (table.column) and literal values.

        Args:
            sql_query: SQL statement string.
            known_columns: Optional collection of known 'table.column' strings
                used to resolve unqualified column names to their parent table.

        Returns:
            Tuple of (fields_Q, lits_Q).
        """
        cleaned = self.clean_sql(sql_query)
        if not cleaned:
            return set(), set()

        # Normalize known columns set for fast resolution
        known_cols_set: set[str] = set()
        table_to_cols: dict[str, set[str]] = {}
        if known_columns:
            if isinstance(known_columns, dict):
                for tbl, cols in known_columns.items():
                    table_to_cols[tbl] = set(cols)
                    for col in cols:
                        known_cols_set.add(f"{tbl}.{col}")
            else:
                known_cols_set = set(known_columns)
                for item in known_cols_set:
                    if "." in item:
                        tbl, col = item.split(".", 1)
                        table_to_cols.setdefault(tbl, set()).add(col)

        # Pre-wrap known unquoted multi-word column names in double quotes to assist SQLite parser
        quoted_sql = cleaned
        for col_key in known_cols_set:
            col_name = col_key.split(".", 1)[-1]
            # Match original, unaccented, or slugified variants
            variants_to_quote = {col_name}
            # Add variant without accents
            norm_name = unicodedata.normalize("NFD", col_name.replace("đ", "d").replace("Đ", "D"))
            unaccented = "".join(c for c in norm_name if unicodedata.category(c) != "Mn")
            variants_to_quote.add(unaccented)
            variants_to_quote.add(unaccented.replace(" ", "_"))

            for v in variants_to_quote:
                if len(v) >= 2 and v in quoted_sql:
                    pattern = re.compile(rf'(?<!["`\[]){re.escape(v)}(?!["`\]])')
                    quoted_sql = pattern.sub(f'"{col_name}"', quoted_sql)

        try:
            return self._parse_with_sqlglot(quoted_sql, table_to_cols, known_cols_set)
        except (sqlglot.errors.SqlglotError, ValueError, TypeError, AttributeError, KeyError):
            # Fall back to regex parsing on malformed queries
            return self._parse_with_fallback(cleaned, table_to_cols, known_cols_set)

    def _parse_with_sqlglot(
        self,
        sql_text: str,
        table_to_cols: dict[str, set[str]],
        known_cols_set: set[str],
    ) -> tuple[set[str], set[str]]:
        """Parse using sqlglot SQLite dialect AST traversal."""
        tree = sqlglot.parse_one(sql_text, read="sqlite")
        if tree is None:
            return set(), set()

        # 1. Map table aliases to real table names
        table_aliases: dict[str, str] = {}
        query_tables: set[str] = set()

        for table in tree.find_all(exp.Table):
            t_name = table.name.strip('"[]`') if table.name else ""
            if not t_name:
                continue
            query_tables.add(t_name)
            table_aliases[t_name] = t_name
            if table.alias:
                clean_alias = table.alias.strip('"[]`')
                table_aliases[clean_alias] = t_name

        # 2. Extract column references
        fields_q: set[str] = set()
        for col in tree.find_all(exp.Column):
            c_name = col.name.strip('"[]`') if col.name else ""
            if not c_name or c_name == "*":
                continue

            t_ref = col.table.strip('"[]`') if col.table else None
            if t_ref:
                real_table = table_aliases.get(t_ref, t_ref)
                fields_q.add(f"{real_table}.{c_name}")
            else:
                # Unqualified column: Attempt resolution
                resolved = False
                for tbl in query_tables:
                    if tbl in table_to_cols and c_name in table_to_cols[tbl]:
                        fields_q.add(f"{tbl}.{c_name}")
                        resolved = True
                        break
                    if f"{tbl}.{c_name}" in known_cols_set:
                        fields_q.add(f"{tbl}.{c_name}")
                        resolved = True
                        break

                if not resolved:
                    if len(query_tables) == 1:
                        single_tbl = next(iter(query_tables))
                        fields_q.add(f"{single_tbl}.{c_name}")
                    else:
                        fields_q.add(c_name)

        # Also detect known columns mentioned in the query text
        for col_key in known_cols_set:
            tbl_name, col_name = col_key.split(".", 1)
            if not query_tables or tbl_name in query_tables:
                col_pattern = re.compile(
                    rf'(?:\b|["`\[]){re.escape(col_name)}(?:\b|["`\]])', re.IGNORECASE
                )
                if col_pattern.search(sql_text):
                    fields_q.add(col_key)

        # 3. Extract literals (strings and numeric values)
        lits_q: set[str] = set()
        for lit in tree.find_all(exp.Literal):
            val = str(lit.this).strip()
            # If wrapped in quotes, unwrap
            if (val.startswith("'") and val.endswith("'")) or (
                val.startswith('"') and val.endswith('"')
            ):
                val = val[1:-1].strip()
            if val:
                lits_q.add(val)

        return fields_q, lits_q

    def _parse_with_fallback(
        self,
        sql_text: str,
        table_to_cols: dict[str, set[str]],
        known_cols_set: set[str] | None = None,
    ) -> tuple[set[str], set[str]]:
        """Fallback regex extractor when SQL AST generation fails."""
        fields_q: set[str] = set()
        lits_q: set[str] = set()

        # Extract table aliases: FROM <table> AS <alias> or JOIN <table> <alias>
        aliases: dict[str, str] = {}
        alias_pattern = re.compile(
            r"\b(?:FROM|JOIN)\s+([\"\`\[]?[\w\-]+[\"\`\]]?)(?:\s+(?:AS\s+)?([\"\`\[]?[\w\-]+[\"\`\]]?))?",
            re.IGNORECASE,
        )
        reserved_sql_keywords = {
            "ON", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "CROSS", "GROUP", "ORDER",
            "SELECT", "FROM", "HAVING", "LIMIT", "SET", "VALUES", "INSERT", "UPDATE", "DELETE",
        }
        for match in alias_pattern.finditer(sql_text):
            tbl = match.group(1).strip("\"`[] ")
            alias = match.group(2).strip("\"`[] ") if match.group(2) else None
            if tbl and tbl.upper() not in reserved_sql_keywords:
                aliases[tbl] = tbl
                if alias and alias.upper() not in reserved_sql_keywords:
                    aliases[alias] = tbl

        # Extract column references: <tbl_or_alias>.<col>
        # Handles both quoted and unquoted multi-word column names
        col_dot_pattern = re.compile(
            r"([\"\`\[]?[\w\-]+[\"\`\]]?)\.(?:\"([^\"]+)\"|`([^`]+)`|\[([^\]]+)\]|([a-zA-Z0-9_\u0080-\uffff]+(?:\s+[a-zA-Z0-9_\u0080-\uffff]+)*))",
            re.IGNORECASE,
        )
        for match in col_dot_pattern.finditer(sql_text):
            t_ref = match.group(1).strip("\"`[] ")
            raw_c = match.group(2) or match.group(3) or match.group(4) or match.group(5) or ""
            # Strip trailing SQL clauses
            c_name = re.split(
                r"\s+(?:LIKE|AS|AND|OR|WHERE|IN|BETWEEN|ORDER|GROUP|HAVING|LIMIT|\=|\>|\<|\!|\))",
                raw_c,
                flags=re.IGNORECASE,
            )[0].strip("\"`[](),;")
            if not c_name:
                continue
            real_tbl = aliases.get(t_ref, t_ref)
            fields_q.add(f"{real_tbl}.{c_name}")

        # Check known columns against the SQL text
        if known_cols_set:
            for col_key in known_cols_set:
                tbl_name, col_name = col_key.split(".", 1)
                if not aliases or tbl_name in aliases.values():
                    col_pattern = re.compile(
                        rf'(?:\b|["`\[]){re.escape(col_name)}(?:\b|["`\]])', re.IGNORECASE
                    )
                    if col_pattern.search(sql_text):
                        fields_q.add(col_key)

        # Extract string literals: '...'
        str_pattern = re.compile(r"'([^'\\]*(?:\\.[^'\\]*)*)'")
        for match in str_pattern.finditer(sql_text):
            lit = match.group(1).strip()
            if lit:
                lits_q.add(lit)

        # Extract numbers in comparisons: = 12345 or > 50000
        num_pattern = re.compile(r"(?:=|>|<|>=|<=)\s*(\d+(?:\.\d+)?)")
        for match in num_pattern.finditer(sql_text):
            lits_q.add(match.group(1).strip())

        return fields_q, lits_q
