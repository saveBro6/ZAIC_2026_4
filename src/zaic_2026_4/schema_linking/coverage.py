"""Schema coverage validation: detect and repair missing tables/columns before SQL generation.

Task-aligned schema linking harvests only the columns a trial query happened to reference,
so a table that is required by the question shape but absent from every trial query (the
transaction header that owns the sale date, for example) silently disappears from the final
schema. This module applies generic, data-agnostic structural rules on the profiled metadata
to restore the minimum coverage a question implies:

* temporal questions need at least one date column reachable from the harvested tables,
* aggregate questions need at least one measurable numeric/currency column,
* any table that must be joined needs its join keys present on both sides.

All decisions are derived from profiled column formats and profiler join candidates; no
dataset-specific table, column or value is ever referenced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from zaic_2026_4.common.query_intent import QueryIntent, fold_text
from zaic_2026_4.constants import DATE_FORMATS
from zaic_2026_4.ingestion.canonical import (
    CANONICAL_COLUMNS,
    canonical_view_name,
    is_canonical_view,
)

DATE_FORMAT_TOKENS = frozenset({"DATE", "TIME", "DATETIME"})
NUMERIC_FORMAT_TOKENS = frozenset({"CURRENCY", "NUMERIC", "INTEGER", "REAL", "FLOAT", "DECIMAL", "NUMBER"})

EMPTY_PROFILE_MAX_CHARS = 0


@dataclass
class CoverageReport:
    """Outcome of a schema coverage pass."""

    required_aspects: list[str] = field(default_factory=list)
    satisfied_aspects: list[str] = field(default_factory=list)
    missing_aspects: list[str] = field(default_factory=list)
    added_columns: list[str] = field(default_factory=list)
    added_tables: list[str] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def has_unresolved_gaps(self) -> bool:
        """True when a required aspect could not be satisfied by the available metadata."""
        return bool(self.missing_aspects)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report into a JSON-compatible dictionary."""
        return {
            "required_aspects": list(self.required_aspects),
            "satisfied_aspects": list(self.satisfied_aspects),
            "missing_aspects": list(self.missing_aspects),
            "added_columns": list(self.added_columns),
            "added_tables": list(self.added_tables),
            "dropped_columns": list(self.dropped_columns),
            "notes": list(self.notes),
        }


def _metadata_dict(processed_metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Return the flat 'table.column' -> metadata mapping from a profiling payload."""
    if not processed_metadata:
        return {}
    target = processed_metadata.get("metadata_profile", processed_metadata)
    if not isinstance(target, dict):
        return {}
    if "tables" in target and isinstance(target["tables"], dict):
        flat: dict[str, Any] = {}
        for tbl, info in target["tables"].items():
            for col, meta in (info.get("columns") or {}).items():
                flat[f"{tbl}.{col}"] = meta
        return flat
    return {
        _strip_key(k): v for k, v in target.items() if "." in k and isinstance(v, dict)
    }


def _strip_key(col_key: str) -> str:
    """Normalize identifier quoting so "Table"."Column" and Table.Column are equivalent."""
    return str(col_key).replace('"', "").replace("'", "").strip()


def _table_of(col_key: str) -> str:
    return _strip_key(col_key).split(".", 1)[0]


def _is_date_meta(meta: dict[str, Any]) -> bool:
    data_format = str(meta.get("data_format", "")).upper()
    if data_format in DATE_FORMATS:
        return True
    return any(token in data_format for token in DATE_FORMAT_TOKENS)


def _is_numeric_meta(meta: dict[str, Any]) -> bool:
    data_format = str(meta.get("data_format", "")).upper()
    return any(token in data_format for token in NUMERIC_FORMAT_TOKENS)


def _is_empty_column(meta: dict[str, Any]) -> bool:
    """Detect columns with no usable content (no samples, no range, no description)."""
    if meta.get("sample_values"):
        return False
    if meta.get("top_10_samples"):
        return False
    if meta.get("min_val") is not None or meta.get("max_val") is not None:
        return False
    if int(meta.get("distinct_count") or 0) > 0:
        return False
    has_desc = bool(str(meta.get("minimal_profile") or "").strip()) or bool(
        str(meta.get("maximal_profile") or "").strip()
    )
    return not has_desc


def _join_edges(
    flat_meta: dict[str, Any],
    raw_profile: dict[str, Any] | None,
) -> list[tuple[str, str]]:
    """Collect profiler-detected join edges as ordered (source_col, target_col) pairs."""
    edges: list[tuple[str, str]] = []
    for col_key, meta in flat_meta.items():
        for cand in meta.get("join_candidates") or []:
            if not isinstance(cand, dict):
                continue
            target = cand.get("target")
            if isinstance(target, str) and "." in target:
                edges.append((col_key, target))
    if raw_profile:
        for join_item in raw_profile.get("undocumented_joins") or []:
            if not isinstance(join_item, dict):
                continue
            col1 = join_item.get("col1")
            col2 = join_item.get("col2")
            if isinstance(col1, str) and isinstance(col2, str) and "." in col1 and "." in col2:
                edges.append((col1, col2))
    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for src, tgt in edges:
        key = (src, tgt)
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def _best_column_of_table(
    table: str,
    flat_meta: dict[str, Any],
    predicate: Any,
    prefer_tokens: tuple[str, ...] = (),
) -> str | None:
    """Pick the most informative column of a table matching a format predicate."""
    candidates = [
        (col_key, meta)
        for col_key, meta in flat_meta.items()
        if _table_of(col_key) == table and isinstance(meta, dict) and predicate(meta)
    ]
    if not candidates:
        return None

    def score(item: tuple[str, dict[str, Any]]) -> tuple[int, int, int]:
        col_key, meta = item
        folded = fold_text(col_key)
        token_hit = 1 if any(tok in folded for tok in prefer_tokens) else 0
        has_samples = 1 if meta.get("sample_values") else 0
        non_empty = 0 if _is_empty_column(meta) else 1
        return (token_hit, non_empty, has_samples)

    candidates.sort(key=score, reverse=True)
    return candidates[0][0]


def _related_tables(
    referenced_tables: set[str],
    edges: list[tuple[str, str]],
) -> list[tuple[str, str, str]]:
    """Return (new_table, join_col_of_new_table, join_col_of_referenced_table) triples."""
    related: list[tuple[str, str, str]] = []
    seen_tables: set[str] = set(referenced_tables)
    for src, tgt in edges:
        src_tbl, tgt_tbl = _table_of(src), _table_of(tgt)
        if src_tbl in referenced_tables and tgt_tbl not in seen_tables:
            related.append((tgt_tbl, tgt, src))
            seen_tables.add(tgt_tbl)
        elif tgt_tbl in referenced_tables and src_tbl not in seen_tables:
            related.append((src_tbl, src, tgt))
            seen_tables.add(src_tbl)
    return related


def _table_adjacency(edges: list[tuple[str, str]]) -> dict[str, list[tuple[str, str, str]]]:
    """Build an undirected table graph: table -> [(neighbour, col_here, col_there)]."""
    graph: dict[str, list[tuple[str, str, str]]] = {}
    for src, tgt in edges:
        src_tbl, tgt_tbl = _table_of(src), _table_of(tgt)
        if src_tbl == tgt_tbl:
            continue
        graph.setdefault(src_tbl, []).append((tgt_tbl, src, tgt))
        graph.setdefault(tgt_tbl, []).append((src_tbl, tgt, src))
    return graph


def _shortest_table_path(
    graph: dict[str, list[tuple[str, str, str]]],
    start: str,
    goal: str,
    max_depth: int = 3,
) -> list[tuple[str, str, str]] | None:
    """Breadth-first path between two tables, or None when they are disconnected.

    Returns the hop list ``[(next_table, col_from_current, col_in_next), ...]``.
    """
    if start == goal:
        return []
    queue: list[tuple[str, list[tuple[str, str, str]]]] = [(start, [])]
    visited: set[str] = {start}
    while queue:
        current, path = queue.pop(0)
        if len(path) >= max_depth:
            continue
        for neighbour, col_here, col_there in graph.get(current, []):
            if neighbour in visited:
                continue
            hop = (neighbour, col_here, col_there)
            if neighbour == goal:
                return [*path, hop]
            visited.add(neighbour)
            queue.append((neighbour, [*path, hop]))
    return None


def _canonical_projection_columns(
    referenced_tables: set[str],
    flat_meta: dict[str, Any],
    already_selected: set[str],
) -> list[str]:
    """Columns of the canonical view of each referenced table, when that view exists."""
    available_tables = {_table_of(col) for col in flat_meta}
    added: list[str] = []
    for table in sorted(referenced_tables):
        if is_canonical_view(table):
            continue
        view = canonical_view_name(table)
        if view not in available_tables:
            continue
        for column in CANONICAL_COLUMNS:
            key = f"{view}.{column}"
            if key in flat_meta and key not in already_selected:
                added.append(key)
    return added


def validate_schema_coverage(
    schema_cols: set[str],
    processed_metadata: dict[str, Any] | None,
    intent: QueryIntent,
    raw_profile: dict[str, Any] | None = None,
    max_extra_tables: int = 6,
) -> tuple[set[str], CoverageReport]:
    """Expand a linked schema until every question-implied aspect is structurally covered.

    Args:
        schema_cols: Final linked schema keys ('table.column').
        processed_metadata: Profiled metadata mapping 'table.column' to column profiles.
        intent: Lexically derived query intent.
        raw_profile: Optional raw profile carrying undocumented join edges.
        max_extra_tables: Safety cap on additional tables pulled in by the closure.

    Returns:
        Tuple of (expanded_schema, CoverageReport). The input set is never mutated.
    """
    flat_meta = _metadata_dict(processed_metadata)
    report = CoverageReport()
    expanded = {_strip_key(col) for col in schema_cols}

    if not flat_meta or not expanded:
        # An empty linked schema cannot prove anything, so it is reported as a gap rather
        # than silently returning a clean report that downstream code would trust.
        report.required_aspects.append("linked_schema_present")
        report.missing_aspects.append("linked_schema_present")
        report.notes.append("coverage_skipped: missing metadata or empty schema")
        return expanded, report

    # Drop columns whose profile is completely empty: they add prompt noise without any signal.
    empty_cols = {c for c in expanded if c in flat_meta and _is_empty_column(flat_meta[c])}
    if empty_cols and len(empty_cols) < len(expanded):
        expanded -= empty_cols
        report.dropped_columns = sorted(empty_cols)

    referenced_tables = {_table_of(c) for c in expanded}
    edges = _join_edges(flat_meta, raw_profile)

    # Aspect 1: temporal questions must reach a date column.
    if intent.has_temporal_cue:
        report.required_aspects.append("temporal_date_column")
        has_date = any(
            c in flat_meta and _is_date_meta(flat_meta[c]) for c in expanded
        )
        if has_date:
            report.satisfied_aspects.append("temporal_date_column")
        else:
            date_col = None
            for table in sorted(referenced_tables):
                date_col = _best_column_of_table(
                    table,
                    flat_meta,
                    _is_date_meta,
                    prefer_tokens=("ngay", "date", "time", "thoi gian"),
                )
                if date_col:
                    break
            if date_col is None:
                added = 0
                for new_table, new_col, ref_col in _related_tables(referenced_tables, edges):
                    if added >= max_extra_tables:
                        break
                    cand_date = _best_column_of_table(
                        new_table,
                        flat_meta,
                        _is_date_meta,
                        prefer_tokens=("ngay", "date", "time", "thoi gian"),
                    )
                    if not cand_date:
                        continue
                    expanded.update({new_col, cand_date})
                    key_col = _best_column_of_table(
                        new_table,
                        flat_meta,
                        lambda m: True,
                        prefer_tokens=("ma", "id", "code", "so"),
                    )
                    if key_col:
                        expanded.add(key_col)
                    report.added_tables.append(new_table)
                    report.added_columns.extend(sorted({new_col, cand_date}))
                    date_col = cand_date
                    added += 1
                    break
            if date_col is not None:
                expanded.add(date_col)
                if date_col not in report.added_columns:
                    report.added_columns.append(date_col)
                report.satisfied_aspects.append("temporal_date_column")
            else:
                report.missing_aspects.append("temporal_date_column")

    # Aspect 2: aggregate questions must reach a measurable numeric/currency column.
    if intent.has_aggregate:
        report.required_aspects.append("aggregate_measure_column")
        has_measure = any(c in flat_meta and _is_numeric_meta(flat_meta[c]) for c in expanded)
        if has_measure:
            report.satisfied_aspects.append("aggregate_measure_column")
        else:
            measure_col = None
            for table in sorted(referenced_tables):
                measure_col = _best_column_of_table(
                    table,
                    flat_meta,
                    _is_numeric_meta,
                    prefer_tokens=("tien", "amount", "total", "gia", "price", "doanh thu", "phi"),
                )
                if measure_col:
                    break
            if measure_col is None:
                for new_table, new_col, _ref in _related_tables(referenced_tables, edges):
                    cand_measure = _best_column_of_table(
                        new_table,
                        flat_meta,
                        _is_numeric_meta,
                        prefer_tokens=("tien", "amount", "total", "gia", "price", "doanh thu", "phi"),
                    )
                    if not cand_measure:
                        continue
                    expanded.update({new_col, cand_measure})
                    report.added_tables.append(new_table)
                    report.added_columns.extend(sorted({new_col, cand_measure}))
                    measure_col = cand_measure
                    break
            if measure_col is not None:
                expanded.add(measure_col)
                if measure_col not in report.added_columns:
                    report.added_columns.append(measure_col)
                report.satisfied_aspects.append("aggregate_measure_column")
            else:
                report.missing_aspects.append("aggregate_measure_column")

    # Aspect 3: every referenced table pair must be connected by a real join path.
    # When the profiler knows a path only through an intermediate table (a bridge table),
    # that table is added instead of declaring the pair unjoinable.
    report.required_aspects.append("join_key_closure")
    closure_added: list[str] = []
    graph = _table_adjacency(edges)
    # Work on a snapshot: bridge tables are appended while the pairs are enumerated.
    initial_tables = sorted({_table_of(c) for c in expanded})
    unreachable: list[str] = []
    for idx, left in enumerate(initial_tables):
        for right in initial_tables[idx + 1:]:
            direct = any(
                _table_of(src) == left and _table_of(tgt) == right
                or _table_of(src) == right and _table_of(tgt) == left
                for src, tgt in edges
            )
            if direct:
                for src, tgt in edges:
                    if {_table_of(src), _table_of(tgt)} == {left, right}:
                        for col in (src, tgt):
                            if col in flat_meta and col not in expanded:
                                expanded.add(col)
                                closure_added.append(col)
                continue

            path = _shortest_table_path(graph, left, right)
            if path is None:
                unreachable.append(f"{left}<->{right}")
                continue
            for bridge_table, col_here, col_there in path:
                if len(path) == 1:
                    # A single hop is a direct edge; only its key columns matter.
                    for col in (col_here, col_there):
                        if col in flat_meta and col not in expanded:
                            expanded.add(col)
                            closure_added.append(col)
                    continue
                if bridge_table in report.added_tables:
                    continue
                if len(report.added_tables) >= max_extra_tables:
                    unreachable.append(f"{left}<->{right}")
                    break
                for col in (col_here, col_there):
                    if col in flat_meta and col not in expanded:
                        expanded.add(col)
                        closure_added.append(col)
                report.added_tables.append(bridge_table)
                report.notes.append(
                    f"join_bridge_added: {bridge_table} connects {left} and {right}"
                )

    if closure_added:
        report.added_columns.extend(sorted(closure_added))
        report.notes.append(f"join_key_closure_added: {sorted(closure_added)}")
    if unreachable:
        # No profiled join path means "unanswerable" cannot be proven: the linking step
        # may simply have missed the table that makes the question answerable.
        report.missing_aspects.append("join_key_closure")
    else:
        report.satisfied_aspects.append("join_key_closure")

    # Aspect 4: expose the canonical (normalized) projection of every referenced table.
    # The aspect is only declared when at least one referenced table has a canonical view,
    # so no consumer can read a phantom "required but unsatisfied" gap.
    referenced = {_table_of(c) for c in expanded}
    available_tables = {_table_of(col) for col in flat_meta}
    has_canonical_view = any(
        not is_canonical_view(table) and canonical_view_name(table) in available_tables
        for table in referenced
    )
    if has_canonical_view:
        report.required_aspects.append("canonical_projection")
        canonical_cols = _canonical_projection_columns(referenced, flat_meta, expanded)
        if canonical_cols:
            expanded.update(canonical_cols)
            report.added_columns.extend(sorted(canonical_cols))
            report.notes.append(f"canonical_projection_added: {sorted(canonical_cols)}")
            report.satisfied_aspects.append("canonical_projection")
        else:
            # The view exists but carries no derived column we can use (or all of them are
            # already selected); nothing to add, so the aspect is considered covered.
            report.satisfied_aspects.append("canonical_projection")

    report.added_columns = sorted(set(report.added_columns))
    report.added_tables = sorted(set(report.added_tables))
    return expanded, report


__all__ = [
    "CoverageReport",
    "validate_schema_coverage",
]
