"""Grouped-row normalization: fill merged header cells that repeat once per group.

The split rulebook defines a table where the header block (order id, customer, date,
status) is written once at the first row of a group and left blank on the following
detail rows. Read naively, those blank cells become NULL and every group-level
aggregate silently loses rows (a month's revenue can drop by half).

The rule is applied structurally and conservatively:

* the header block is detected as a set of columns that share the **exact same** NULL
  pattern, so item columns are never rewritten,
* a row is a continuation row only when the block is blank **and** the rest of the row
  is still populated (a real detail line), which keeps footer/total/note rows out,
* a group is the run of rows that share one header block, and a cell is filled only
  when that group carries exactly one distinct non-null value for it,
* no row is added, dropped or reordered, and nothing is written outside the block.
"""

from __future__ import annotations

import math
from typing import Any

import polars as pl

from zaic_2026_4.ingestion.canonical import fold_ascii, is_measure_column

MIN_BLOCK_COLUMNS = 2
# The identifier of a merged header sits at the left edge of the sheet.
LEADING_COLUMN_TOLERANCE = 1
# One clearly identified continuation row is enough: the block, item-coverage and
# summary-row guards already carry the safety, so a second row adds nothing.
MIN_FILLED_ROWS = 1


def _null_pattern(series: pl.Series) -> tuple[bool, ...]:
    return tuple(series.is_null().to_list())


def _header_block_columns(df: pl.DataFrame) -> list[str]:
    """Columns sharing an identical NULL pattern, i.e. one merged header block.

    Measure columns never belong to the block: carrying a quantity or a price down would
    invent numbers, and a total row would then look like a sold line item.
    """
    height = df.height
    if height == 0:
        return []
    grouped: dict[tuple[bool, ...], list[str]] = {}
    for name in df.columns:
        if is_measure_column(name, df[name]):
            continue
        pattern = _null_pattern(df[name])
        null_count = sum(pattern)
        if null_count == 0 or null_count >= height:
            continue
        grouped.setdefault(pattern, []).append(name)
    if not grouped:
        return []
    best = max(grouped.values(), key=lambda cols: (len(cols), -df.columns.index(cols[0])))
    if len(best) < MIN_BLOCK_COLUMNS:
        return []
    # A merged header always starts at the left edge of the sheet; a block that starts
    # in the middle is a sparse attribute group (missing data), not a merged header.
    if df.columns.index(best[0]) > LEADING_COLUMN_TOLERANCE:
        return []
    return best


def _item_cell_counts(
    df: pl.DataFrame,
    block: list[str],
) -> tuple[list[int], list[int]]:
    """Per row: populated item cells, and populated item cells that are not measures.

    A continuation row must still carry descriptive item data; a row whose only populated
    cell is a money amount is a total line, not a sold line.
    """
    item_columns = [name for name in df.columns if name not in block]
    if not item_columns:
        return [0] * df.height, [0] * df.height
    descriptive = [name for name in item_columns if not is_measure_column(name, df[name])]
    null_matrix = [df[name].is_null().to_list() for name in item_columns]
    descriptive_matrix = [df[name].is_null().to_list() for name in descriptive]
    populated = [
        sum(1 for flags in null_matrix if not flags[row_idx])
        for row_idx in range(df.height)
    ]
    populated_descriptive = [
        sum(1 for flags in descriptive_matrix if not flags[row_idx])
        for row_idx in range(df.height)
    ]
    return populated, populated_descriptive


# Rulebook-level summary/footer labels: such a row closes a group instead of continuing
# it, so filling it would fabricate data (for example a quantity on a "Cộng" row).
SUMMARY_ROW_LABELS: frozenset[str] = frozenset({
    "cong",
    "cong tien",
    "tong",
    "tong cong",
    "tong ket",
    "sum",
    "total",
    "subtotal",
    "note",
})
# Paragraph-style labels such as "TỔNG NGÀY 01/02" or "Ghi chú: ..." also close a group.
SUMMARY_ROW_PREFIXES: tuple[str, ...] = (
    "cong",
    "sum",
    "tong",
    "total",
    "subtotal",
    "ghi chu",
    "note",
)


def is_summary_row(row: dict[str, Any], block: list[str]) -> bool:
    """True when any populated non-block text cell is a summary/footer label.

    Every text cell is inspected, not only the first populated one: a total row often
    puts the amount in an earlier column and the label ("Tổng ngày 01/03") in a later one.
    """
    for name, value in row.items():
        if name in block or value is None:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            continue
        label = fold_ascii(str(value)).strip(" :.-")
        if label in SUMMARY_ROW_LABELS or label.startswith(SUMMARY_ROW_PREFIXES):
            return True
    return False


def _is_continuation_row(
    block_blank: bool,
    populated_items: int,
    populated_descriptive: int,
    item_width: int,
    summary_row: bool,
) -> bool:
    """A continuation row keeps descriptive item cells populated and only the block blank."""
    if not block_blank or summary_row or item_width <= 0:
        return False
    if populated_items <= 0 or populated_descriptive <= 0:
        return False
    return populated_items >= math.ceil(item_width / 2)


def has_row_level_primary_key(df: pl.DataFrame, block: list[str]) -> bool:
    """True when a complete, unique, non-block column already identifies each row.

    A customer/product/entity table has exactly one row per key: its blank cells are
    missing data, not merged header cells, so nothing may be carried down. Detail tables
    always repeat at least one value, so this guard cannot block a genuine merged layout.
    """
    if df.height < 20:
        return False
    for name in df.columns:
        if name in block:
            continue
        series = df[name]
        if series.dtype.is_numeric() or series.dtype in (pl.Date, pl.Datetime):
            continue
        if series.null_count() > 0:
            continue
        if series.n_unique() == df.height:
            return True
    return False


def _group_ids(
    block_blank: list[bool],
    continuation: list[bool],
) -> list[int]:
    """Assign group ids; continuation rows join the preceding block, junk rows stand alone."""
    group_ids: list[int] = []
    current = -1
    isolated = -10_000
    for row_idx, blank in enumerate(block_blank):
        if not blank:
            current += 1
            group_ids.append(current)
        elif continuation[row_idx]:
            group_ids.append(current)
        else:
            isolated -= 1
            group_ids.append(isolated)
    return group_ids


def forward_fill_grouped_rows(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Apply the rulebook grouped-row rule to one table.

    Returns the normalized DataFrame and a report. The report always carries
    ``applied`` so callers can log which tables were rewritten.
    """
    report: dict[str, Any] = {
        "applied": False,
        "key_column": None,
        "filled_columns": [],
        "filled_cells": 0,
    }
    if df.height < MIN_FILLED_ROWS or df.width < MIN_BLOCK_COLUMNS:
        return df, report
    if not any(df[name].null_count() > 0 for name in df.columns):
        return df, report

    block = _header_block_columns(df)
    if not block:
        return df, report

    block_blank = [bool(flag) for flag in df[block[0]].is_null().to_list()]
    populated_items, populated_descriptive = _item_cell_counts(df, block)
    item_width = df.width - len(block)
    rows = df.to_dicts()
    continuation = [
        _is_continuation_row(
            blank,
            count,
            descriptive,
            item_width,
            is_summary_row(rows[row_idx], block),
        )
        for row_idx, (blank, count, descriptive) in enumerate(
            zip(block_blank, populated_items, populated_descriptive, strict=False)
        )
    ]
    if sum(continuation) < MIN_FILLED_ROWS:
        return df, report
    if has_row_level_primary_key(df, block):
        return df, report

    group_ids = _group_ids(block_blank, continuation)

    filled_columns: list[str] = []
    filled_cells = 0
    for name in block:
        values = df[name].to_list()
        group_value: dict[int, Any] = {}
        ambiguous: set[int] = set()
        for value, group in zip(values, group_ids, strict=False):
            if value is None:
                continue
            if group not in group_value:
                group_value[group] = value
            elif group_value[group] != value:
                ambiguous.add(group)

        new_values = [
            group_value[group]
            if value is None and group in group_value and group not in ambiguous
            else value
            for value, group in zip(values, group_ids, strict=False)
        ]
        changed = sum(
            1 for old, new in zip(values, new_values, strict=False) if old is None and new is not None
        )
        if not changed:
            continue
        df = df.with_columns(pl.Series(name, new_values, dtype=df[name].dtype))
        filled_columns.append(name)
        filled_cells += changed

    if not filled_cells:
        return df, report

    return df, {
        "applied": True,
        "key_column": block[0],
        "block_columns": block,
        "continuation_rows": sum(continuation),
        "filled_columns": filled_columns,
        "filled_cells": filled_cells,
    }


__all__ = [
    "forward_fill_grouped_rows",
    "has_row_level_primary_key",
    "is_summary_row",
]
