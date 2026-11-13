"""Schema post-processing and canonical grounding module for Task-Aligned Schema Linking.

Cleans aliases, strips SQL syntax keywords, normalizes diacritics and naming conventions,
validates every field against ground-truth database metadata, discards hallucinations,
and wraps column names in double quotes for safe downstream SQLite execution.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

from zaic_2026_4.common.xml_sanitizer import (
    build_column_attributes,
    format_schema_table_elements,
)

logger = logging.getLogger(__name__)


def normalize_identifier(name: str) -> str:
    """Normalize string by removing accents, non-alphanumeric chars, and converting to lowercase.

    Handles Vietnamese 'Đ'/'đ' which are not decomposed by Unicode NFD.

    Examples:
        'Ngày tháng'    -> 'ngaythang'
        'Ngay_thang'     -> 'ngaythang'
        'Thanh tien'     -> 'thanhtien'
        'Quy ra ĐV lẻ'   -> 'quyradvle'
        'Quy_ra_DV_le'   -> 'quyradvle'
        'Đối tác'        -> 'doitac'
        'Doi_tac'        -> 'doitac'
    """
    s = name.replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-zA-Z0-9]", "", s).lower()


def clean_candidate_field(field_str: str) -> tuple[str, str]:
    """Extract table/alias prefix and column candidate, stripping trailing SQL keywords and punctuation.

    Args:
        field_str: Raw candidate string (e.g. 'T1.Thanh tien) AS', 'ban_hang.Ngay_thang').

    Returns:
        Tuple of (table_or_alias, cleaned_column_candidate).
    """
    cleaned = field_str.strip()
    if "." in cleaned:
        t_ref, c_ref = cleaned.split(".", 1)
    else:
        t_ref, c_ref = "", cleaned

    t_clean = t_ref.strip(' "`[]')

    # Remove trailing SQL keywords and clauses (e.g. LIKE, AS, AND, OR, WHERE, etc.)
    c_clean = re.split(
        r"\s+(?:LIKE|AS|AND|OR|WHERE|IN|BETWEEN|ORDER|GROUP|HAVING|LIMIT|\=|\>|\<|\!|\))",
        c_ref,
        flags=re.IGNORECASE,
    )[0]
    # Remove surrounding syntax brackets, quotes, and parens
    c_clean = c_clean.strip(' "`[](),;')

    return t_clean, c_clean


class SchemaPostProcessor:
    """Post-processor that grounds harvested fields against database metadata and formats them for SQL generation."""

    @staticmethod
    def build_table_columns_map(
        processed_metadata: dict[str, Any],
    ) -> dict[str, dict[str, str]]:
        """Build mapping of table_name -> {normalized_column: ground_truth_column}.

        Args:
            processed_metadata: Enriched metadata dictionary.

        Returns:
            Dict mapping table -> {norm_col: real_col}.
        """
        table_cols: dict[str, dict[str, str]] = {}
        target_dict = processed_metadata.get("metadata_profile", processed_metadata)

        if "tables" in target_dict and isinstance(target_dict["tables"], dict):
            for tbl, t_info in target_dict["tables"].items():
                for col in t_info.get("columns", {}):
                    table_cols.setdefault(tbl, {})[normalize_identifier(col)] = col
        else:
            for k in target_dict:
                if "." in k:
                    tbl, col = k.split(".", 1)
                    table_cols.setdefault(tbl, {})[normalize_identifier(col)] = col

        return table_cols

    @staticmethod
    def extract_table_aliases(sql_texts: list[str] | str) -> dict[str, str]:
        """Extract table aliases from SQL query statements (e.g. FROM ban_hang AS T1).

        Args:
            sql_texts: Single SQL string or list of SQL strings.

        Returns:
            Dict mapping alias -> real_table_name.
        """
        if isinstance(sql_texts, str):
            sql_texts = [sql_texts]

        aliases: dict[str, str] = {}
        alias_pattern = re.compile(
            r"\b(?:FROM|JOIN)\s+([\"`\[]?[\w\-]+[\"`\]]?)(?:\s+(?:AS\s+)?([\"`\[]?[\w\-]+[\"`\]]?))?",
            re.IGNORECASE,
        )

        reserved_words = {
            "ON", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "CROSS", "NATURAL", "USING", "GROUP", "ORDER",
        }

        for sql in sql_texts:
            if not sql:
                continue
            for match in alias_pattern.finditer(sql):
                tbl = match.group(1).strip('"`[] ')
                alias = match.group(2).strip('"`[] ') if match.group(2) else None
                if tbl and tbl.upper() not in reserved_words:
                    aliases[tbl] = tbl
                    if alias and alias.upper() not in reserved_words:
                        aliases[alias] = tbl

        return aliases

    @classmethod
    def resolve_field(
        cls,
        field_str: str,
        table_cols: dict[str, dict[str, str]],
        aliases: dict[str, str] | None = None,
        always_quote: bool = True,
    ) -> tuple[str, str] | None:
        """Resolve a candidate field to ground-truth database schema and produce quoted / canonical forms.

        Args:
            field_str: Raw candidate string (e.g. 'T1.Thanh tien) AS', 'ban_hang.Ngay_thang').
            table_cols: Table-to-normalized-columns mapping from build_table_columns_map.
            aliases: Optional table alias map.
            always_quote: Whether to wrap column names in double quotes for SQL generation.

        Returns:
            Tuple of (quoted_field, canonical_field) if valid, or None if candidate is hallucinated/unresolvable.
        """
        alias_map = aliases or {}
        t_clean, c_clean = clean_candidate_field(field_str)

        if not c_clean:
            return None

        # Resolve table name from alias
        real_tbl = alias_map.get(t_clean, t_clean)

        # If table name not found:
        if real_tbl not in table_cols:
            # Case A: If only one table exists in database, resolve to that table
            if len(table_cols) == 1:
                real_tbl = next(iter(table_cols.keys()))
            else:
                # Case B: Search all tables for unique column match
                norm_cand = normalize_identifier(c_clean)
                matching_tables = [tbl for tbl, cols in table_cols.items() if norm_cand in cols]
                if len(matching_tables) == 1:
                    real_tbl = matching_tables[0]
                else:
                    return None

        norm_c = normalize_identifier(c_clean)
        if norm_c in table_cols[real_tbl]:
            canonical_col = table_cols[real_tbl][norm_c]
            # Wrap column in double quotes for SQL generation
            if always_quote or " " in canonical_col or any(ord(c) > 127 for c in canonical_col):
                quoted_col = f'"{canonical_col}"'
            else:
                quoted_col = canonical_col

            quoted_field = f"{real_tbl}.{quoted_col}"
            canonical_field = f"{real_tbl}.{canonical_col}"
            return quoted_field, canonical_field

        # Hallucinated or non-existent column in database
        return None

    @classmethod
    def post_process_linked_schema(
        cls,
        raw_fields: set[str],
        processed_metadata: dict[str, Any],
        table_aliases: dict[str, str] | None = None,
        always_quote: bool = True,
    ) -> tuple[set[str], set[str]]:
        """Post-process all raw harvested fields into clean, validated, quoted schema sets.

        Args:
            raw_fields: Set of raw candidate column strings.
            processed_metadata: Enriched database metadata dictionary.
            table_aliases: Optional table alias map.
            always_quote: Whether to wrap column names in double quotes for SQL generation.

        Returns:
            Tuple of (quoted_schema_set, canonical_schema_set).
        """
        table_cols = cls.build_table_columns_map(processed_metadata)
        quoted_schema: set[str] = set()
        canonical_schema: set[str] = set()

        for field_str in raw_fields:
            resolved = cls.resolve_field(
                field_str=field_str,
                table_cols=table_cols,
                aliases=table_aliases,
                always_quote=always_quote,
            )
            if resolved is not None:
                q_field, c_field = resolved
                quoted_schema.add(q_field)
                canonical_schema.add(c_field)

        return quoted_schema, canonical_schema

    @classmethod
    def format_for_sql_generation(
        cls,
        quoted_schema: set[str],
        processed_metadata: dict[str, Any],
    ) -> str:
        """Format quoted schema into clean XML context for downstream Phase 04 SQL Generation.

        Args:
            quoted_schema: Set of 'table."column"' strings.
            processed_metadata: Enriched database metadata.

        Returns:
            Formatted XML string ready for SQL generation prompt.
        """
        target_dict = processed_metadata.get("metadata_profile", processed_metadata)
        tables: dict[str, list[dict[str, Any]]] = {}

        for q_field in sorted(quoted_schema):
            if "." not in q_field:
                continue
            tbl, q_col = q_field.split(".", 1)
            raw_col = q_col.strip('"')
            col_key = f"{tbl}.{raw_col}"

            col_meta = target_dict.get(col_key, {})
            data_type = col_meta.get("data_format", "STRING")
            desc = col_meta.get("maximal_profile", "")

            payload: dict[str, Any] = {
                "col_name": raw_col,
                "data_format": data_type,
                "desc": desc,
            }
            payload.update(build_column_attributes(col_meta))
            tables.setdefault(tbl, []).append(payload)

        lines = ["<selected_schema>"]
        for tbl, cols in sorted(tables.items()):
            lines.extend(format_schema_table_elements(tbl, cols))
        lines.append("</selected_schema>")

        return "\n".join(lines)
