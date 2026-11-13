"""Unit tests for the rulebook grouped-row rule (merged header cells)."""

from __future__ import annotations

import polars as pl

from zaic_2026_4.ingestion.forward_fill import (
    forward_fill_grouped_rows,
    is_summary_row,
)


def test_merged_header_cells_are_filled() -> None:
    """Only the merged header block is filled; item columns are never rewritten."""
    frame = pl.DataFrame(
        {
            "Mã đơn": ["A1", None, None, "A2", None],
            "Ngày": ["01/03/2025", None, None, "02/03/2025", None],
            "Trạng thái": ["Đã giao", None, None, "Hủy", None],
            "Tên hàng": ["Gạo", "Mắm", "Muối", "Gạo", "Đường"],
            "Số lượng": [1, 2, 3, 4, 5],
        }
    )
    filled, report = forward_fill_grouped_rows(frame)

    assert report["applied"] is True
    assert report["key_column"] == "Mã đơn"
    assert report["block_columns"] == ["Mã đơn", "Ngày", "Trạng thái"]
    assert filled["Mã đơn"].to_list() == ["A1", "A1", "A1", "A2", "A2"]
    assert filled["Ngày"].to_list() == [
        "01/03/2025",
        "01/03/2025",
        "01/03/2025",
        "02/03/2025",
        "02/03/2025",
    ]
    # Item-level columns are never rewritten.
    assert filled["Tên hàng"].to_list() == ["Gạo", "Mắm", "Muối", "Gạo", "Đường"]
    assert filled["Số lượng"].to_list() == [1, 2, 3, 4, 5]
    assert filled.height == frame.height


def test_footer_and_note_rows_are_never_filled() -> None:
    """A total row and a note row must not receive header values from the last group.

    Feeding them would inflate every aggregate computed over the table (a quantity on a
    "Cộng" row, for example).
    """
    frame = pl.DataFrame(
        {
            "Mã đơn": ["A1", None, None, None],
            "Ngày": ["01/03/2025", None, None, None],
            "Trạng thái": ["Đã giao", None, None, None],
            "Tên hàng": ["Gạo", "Mắm", "Cộng", "Ghi chú: tiền tính bằng đồng"],
            "Thành tiền": [1000, 2000, 3000, None],
        }
    )
    filled, report = forward_fill_grouped_rows(frame)

    assert report["applied"] is True
    # The real detail row inherits the header, the summary rows keep their blanks.
    assert filled["Mã đơn"].to_list() == ["A1", "A1", None, None]
    assert filled["Tên hàng"].to_list() == [
        "Gạo",
        "Mắm",
        "Cộng",
        "Ghi chú: tiền tính bằng đồng",
    ]
    assert filled["Thành tiền"].to_list() == [1000, 2000, 3000, None]


def test_is_summary_row_labels() -> None:
    """Summary detection uses generic rulebook vocabulary, not a dataset value."""
    block = ["Mã đơn", "Ngày"]
    assert is_summary_row({"Mã đơn": None, "Ngày": None, "Hàng": "Cộng"}, block) is True
    assert is_summary_row({"Mã đơn": None, "Ngày": None, "Hàng": "Tổng cộng"}, block) is True
    assert (
        is_summary_row({"Mã đơn": None, "Ngày": None, "Hàng": "Ghi chú: đơn vị VNĐ"}, block)
        is True
    )
    assert is_summary_row({"Mã đơn": None, "Ngày": None, "Hàng": "Gạo ST25"}, block) is False


def test_one_row_per_entity_table_is_untouched() -> None:
    """A customer/entity table with a single row per key must never be fabricated.

    Here the blank cells are genuinely missing data for a different customer, so carrying
    the previous customer's name/birthday down would invent facts.
    """
    rows = 25
    frame = pl.DataFrame(
        {
            "Mã KH": [f"KH{i:04d}" for i in range(rows)],
            "Tên": [f"Khách {i}" if i % 5 else None for i in range(rows)],
            "Sinh nhật": [f"0{i % 9 + 1}/01/199{i % 10}" if i % 4 else None for i in range(rows)],
        }
    )
    filled, report = forward_fill_grouped_rows(frame)
    assert report["applied"] is False
    assert filled.equals(frame)


def test_single_column_blank_is_not_enough_to_infer_a_block() -> None:
    """One sparse column is a data column, not a merged header block."""
    frame = pl.DataFrame(
        {
            "Mã đơn": ["A1", None, None],
            "Kênh": ["Web", "Shop", None],
            "Số lượng": [1, 2, 3],
        }
    )
    filled, report = forward_fill_grouped_rows(frame)
    assert report["applied"] is False
    assert filled.equals(frame)


def test_group_with_conflicting_values_is_not_filled() -> None:
    """A block column that differs inside a run keeps its blanks."""
    frame = pl.DataFrame(
        {
            "Mã đơn": ["A1", None, None],
            "Ngày": ["01/03/2025", "02/03/2025", None],
            "Tên hàng": ["Gạo", "Mắm", "Muối"],
            "Số lượng": [1, 2, 3],
        }
    )
    filled, _report = forward_fill_grouped_rows(frame)
    assert filled["Ngày"].to_list() == ["01/03/2025", "02/03/2025", None]
