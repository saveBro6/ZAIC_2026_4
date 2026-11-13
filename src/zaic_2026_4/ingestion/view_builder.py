"""Automated unified VIEW generation for tables sharing identical column schemas."""

from collections import defaultdict
import re
import polars as pl


def token_common_prefix(names: list[str]) -> str:
    """Find common prefix of whole tokens separated by standard delimiters (_, -, space, .).

    Zero pattern regex: completely generic across any language and naming scheme.
    """
    if not names:
        return ""
    token_lists = [re.split(r"([_\-\s\.]+)", name) for name in names]
    common_tokens: list[str] = []
    for tokens in zip(*token_lists, strict=False):
        if all(t == tokens[0] for t in tokens):
            common_tokens.append(tokens[0])
        else:
            break
    return "".join(common_tokens).rstrip(" _-.")


def group_tables_by_schema(tables: dict[str, pl.DataFrame]) -> list[list[str]]:
    """Group table names that share an identical column set (candidate partitions)."""
    schema_groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for tname, df in tables.items():
        if df.is_empty() and len(df.columns) == 0:
            continue
        col_key = tuple(sorted(df.columns))
        schema_groups[col_key].append(tname)
    return [members for members in schema_groups.values() if len(members) >= 2]


def generate_unified_views(
    tables: dict[str, pl.DataFrame],
) -> list[tuple[str, str]]:
    """Generate (view_name, view_sql) pairs for tables with identical column sets."""
    schema_groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for tname, df in tables.items():
        if df.is_empty() and len(df.columns) == 0:
            continue
        col_key = tuple(sorted(df.columns))
        schema_groups[col_key].append(tname)

    views: list[tuple[str, str]] = []
    used_names: set[str] = set(tables.keys())

    for _, member_tables in schema_groups.items():
        if len(member_tables) < 2:
            continue

        prefix = token_common_prefix(member_tables)
        if not prefix or len(prefix) < 2:
            base_view_name = f"view_{member_tables[0]}"
        else:
            base_view_name = prefix

        view_name = base_view_name
        counter = 1
        while view_name in used_names:
            view_name = (
                f"{base_view_name}_view"
                if counter == 1
                else f"{base_view_name}_view_{counter}"
            )
            counter += 1
        used_names.add(view_name)

        canonical_cols = tables[member_tables[0]].columns
        quoted_cols = ", ".join(f'"{c}"' for c in canonical_cols)
        select_queries = [
            f'SELECT {quoted_cols} FROM "{tbl}"' for tbl in member_tables
        ]
        view_sql = (
            f'CREATE VIEW IF NOT EXISTS "{view_name}" AS '
            + " UNION ALL ".join(select_queries)
        )
        views.append((view_name, view_sql))

    return views
