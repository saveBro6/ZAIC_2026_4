"""Unit tests for the canonical date/amount/status normalization layer."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import polars as pl

from zaic_2026_4.ingestion.canonical import (
    build_amount_column,
    build_date_column,
    canonical_status,
    classify_date_value,
    classify_money_value,
    describe_canonical_column,
    detect_day_first,
    normalize_date_value,
    parse_money_value,
    plan_canonical_tables,
    scan_canonical_specs,
)
from zaic_2026_4.ingestion.sqlite_sink import write_tables_to_sqlite


def test_date_formats_normalize_to_iso() -> None:
    """Every representation observed in the split normalizes to YYYY-MM-DD."""
    assert normalize_date_value("01/03/2025") == "2025-03-01"
    assert normalize_date_value("1/3/2025") == "2025-03-01"
    assert normalize_date_value("2025-03-01") == "2025-03-01"
    assert normalize_date_value("2025-03-01 14:30:00") == "2025-03-01"
    assert normalize_date_value("1 thg 10 2025") == "2025-10-01"
    assert normalize_date_value("01-03-2025") == "2025-03-01"
    assert normalize_date_value("45658") == "2025-01-01"
    assert normalize_date_value("not a date") is None


def test_day_first_detection_and_month_first_override() -> None:
    """Ambiguous d/m columns default to day-first but yield to clear evidence."""
    assert detect_day_first(["01/03/2025", "25/03/2025"]) is True
    assert detect_day_first(["03/25/2025", "03/27/2025"]) is False
    assert normalize_date_value("03/25/2025", day_first=False) == "2025-03-25"


def test_partial_day_month_uses_column_year() -> None:
    """Day/month-only values inherit the narrative year instead of becoming NULL."""
    series = pl.Series("Ngày lập", ["1/1", "31/10/2025", "2/10"])
    values, report = build_date_column(series)
    assert values == ["2025-01-01", "2025-10-31", "2025-10-02"]
    assert report["inferred_year_rows"] == 2
    assert "SLASH_YEAR" in report["formats"]


def test_money_formats_parse_without_scale_errors() -> None:
    """Separator conventions are detected per value, including scaled shorthand."""
    assert classify_money_value("58,000.0000") == "THOUSAND_COMMA"
    assert parse_money_value("58,000.0000") == 58000.0
    assert parse_money_value("1,600") == 1600.0
    assert parse_money_value("1.234.567") == 1234567.0
    assert parse_money_value("300.000đ") == 300000.0
    assert parse_money_value("1tr440") == 1440000.0
    assert parse_money_value("2tr") == 2000000.0
    assert parse_money_value("(1.500)") == -1500.0

    series = pl.Series("Tổng tiền", ["3200", "6,000", "58,000.0000"])
    amount, report = build_amount_column(series)
    assert amount.to_list() == [3200, 6000, 58000]
    assert report["dtype"] == "INTEGER"


def test_status_vocabulary_maps_to_rulebook_families() -> None:
    """Shop-specific status words map onto the four rulebook families."""
    assert canonical_status("Hủy") == "cancelled"
    assert canonical_status("hủy đơn") == "cancelled"
    assert canonical_status("Đã hủy") == "cancelled"
    assert canonical_status("Đã giao") == "completed"
    assert canonical_status("Đã hoàn thành") == "completed"
    assert canonical_status("xong") == "completed"
    assert canonical_status("ok") == "completed"
    assert canonical_status("Đang vận chuyển") == "shipping"
    assert canonical_status("đã lấy") == "shipping"
    assert canonical_status("Hoàn trả") == "returned"
    assert canonical_status("Không giao được") == "returned"
    assert canonical_status("Chờ xử lý") == "pending"
    assert canonical_status("Kênh Facebook") == "other"


def test_scan_specs_prefers_native_datetime_and_line_total() -> None:
    """The derived columns use the unambiguous timestamp and the total, not the unit price."""
    frame = pl.DataFrame(
        {
            "Giờ": [datetime(2025, 1, 1, 6, 13, 20), datetime(2025, 1, 1, 7, 0, 0)],
            "Ngày lập": ["1/1", "2/1"],
            "Đơn giá": ["10.000đ", "10.000đ"],
            "Tổng tiền": ["20.000đ", "30.000đ"],
            "Trạng thái": ["Đã giao", "Hủy"],
        }
    )
    spec = scan_canonical_specs(frame)
    assert spec["_date"] == "Giờ"
    assert spec["_amount"] == "Tổng tiền"
    assert spec["_status"] == "Trạng thái"

    # A column with a single distinct value cannot describe order state.
    single = pl.DataFrame({"Trạng thái": ["Đã giao", "Đã giao"]})
    assert "_status" not in scan_canonical_specs(single)


def test_scan_specs_skips_identity_iso_dates() -> None:
    """An already-ISO date column needs no derived copy."""
    frame = pl.DataFrame({"date": ["2025-01-01", "2025-01-02"]})
    assert "_date" not in scan_canonical_specs(frame)


def test_describe_canonical_column_stays_within_budget() -> None:
    """Deterministic descriptions must fit the schema description budget."""
    for column in ("_date", "_amount", "_status"):
        text = describe_canonical_column("Bang__c", column)
        assert text is not None
        assert len(text) <= 120
    assert describe_canonical_column("Bang", "_date") is None


def test_sidecar_view_exposes_canonical_columns(tmp_path: Path) -> None:
    """The canonical view keeps every base column and adds the derived ones."""
    tables = {
        "HoaDon": pl.DataFrame(
            {
                "Mã đơn": ["A1", "A2", "A3"],
                "Ngày": ["01/03/2025", "45658", "3/3/2025"],
                "Tổng tiền": ["1,200", "58,000.0000", "30,000"],
                "Trạng thái": ["Đã giao", "Hủy", "Đang vận chuyển"],
            }
        )
    }
    db_path = write_tables_to_sqlite(tables, tmp_path / "biz.db", create_views=True)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        columns = [row[1] for row in cur.execute('PRAGMA table_info("HoaDon__c")')]
        assert columns[:4] == ["Mã đơn", "Ngày", "Tổng tiền", "Trạng thái"]
        assert {"_date", "_amount", "_status"}.issubset(set(columns))

        rows = list(
            cur.execute(
                'SELECT "_date", "_amount", "_status" FROM "HoaDon__c" ORDER BY "Mã đơn"'
            )
        )
        assert rows == [
            ("2025-03-01", 1200, "completed"),
            ("2025-01-01", 58000, "cancelled"),
            ("2025-03-03", 30000, "shipping"),
        ]

        # The base table keeps its raw values untouched.
        raw = list(cur.execute('SELECT "Ngày" FROM "HoaDon" ORDER BY "Mã đơn"'))
        assert raw[0][0] == "01/03/2025"

        # The sidecar is an implementation detail, not a business entity.
        names = {row[0] for row in cur.execute("SELECT name FROM sqlite_master")}
        assert "_canon_HoaDon" in names
        specs, _reports = plan_canonical_tables(tables)
        assert list(specs) == ["HoaDon"]
    finally:
        conn.close()


def test_classify_date_value_ignores_small_integers() -> None:
    """Small integers are counts, not Excel serials."""
    assert classify_date_value("42") is None
    assert classify_date_value("45658") == "EXCEL_SERIAL"


def test_unified_canonical_view_unions_partitions(tmp_path: Path) -> None:
    """Partitions consolidated into one unified view also get one canonical view."""
    def _partition(prefix: str, month: str) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "Mã đơn": [f"{prefix}-1"],
                "Ngày": [f"01/{month}/2025"],
                "Tổng tiền": ["1,000"],
                "Trạng thái": ["Đã giao"],
            }
        )

    tables = {
        "SoBanHang_04_2025": _partition("A", "04"),
        "SoBanHang_05_2025": _partition("B", "05"),
    }
    # Give each partition a second status so the status column is derivable.
    tables["SoBanHang_04_2025"] = tables["SoBanHang_04_2025"].vstack(
        pl.DataFrame(
            {"Mã đơn": ["A-2"], "Ngày": ["02/04/2025"], "Tổng tiền": ["2,000"], "Trạng thái": ["Hủy"]}
        )
    )
    tables["SoBanHang_05_2025"] = tables["SoBanHang_05_2025"].vstack(
        pl.DataFrame(
            {"Mã đơn": ["B-2"], "Ngày": ["02/05/2025"], "Tổng tiền": ["3,000"], "Trạng thái": ["Hủy"]}
        )
    )

    db_path = write_tables_to_sqlite(tables, tmp_path / "biz.db", create_views=True)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        views = {row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='view'")}
        assert "SoBanHang" in views
        assert "SoBanHang__c" in views
        rows = list(
            cur.execute(
                'SELECT "_date", "_amount", "_status" FROM "SoBanHang__c" ORDER BY "Mã đơn"'
            )
        )
        assert rows == [
            ("2025-04-01", 1000, "completed"),
            ("2025-04-02", 2000, "cancelled"),
            ("2025-05-01", 1000, "completed"),
            ("2025-05-02", 3000, "cancelled"),
        ]
    finally:
        conn.close()


def test_money_mode_does_not_break_values_of_another_convention() -> None:
    """A PLAIN-dominated column still parses its thousands-separated values."""
    series = pl.Series("Tổng tiền", ["100000", "6,000", "3200", "1,600"])
    amount, report = build_amount_column(series)
    assert amount.to_list() == [100000, 6000, 3200, 1600]
    assert report["parsed_rows"] == 4


def test_ambiguous_money_value_follows_the_column_convention() -> None:
    """'1.234' means thousands inside a dot-thousands column, decimals inside a plain one."""
    dot_thousands = pl.Series("Giá", ["1.234.567", "1.234"])
    amount, _ = build_amount_column(dot_thousands)
    assert amount.to_list() == [1234567, 1234]

    # "1.234" next to plain fractional values is a decimal value, not thousands.
    plain = pl.Series("Giá", ["12.5", "1.234"])
    amount_plain, _ = build_amount_column(plain)
    assert amount_plain.to_list() == [12.5, 1.234]


def test_ambiguous_money_majority_wins_over_a_stray_fraction() -> None:
    """Two thousands-style rows outvote one fractional row in the same column."""
    series = pl.Series("Giá", ["1.500", "2.000", "12.5"])
    amount, report = build_amount_column(series)
    assert amount.to_list() == [1500.0, 2000.0, 12.5]
    assert report["ambiguous_as_thousands"] is True


def test_ambiguous_money_keeps_accounting_minus_sign() -> None:
    """A value written as "(1.500)" is negative, whichever convention is applied."""
    series = pl.Series("Giá", ["(1.500)", "2.000", "12.5"])
    amount, _ = build_amount_column(series)
    assert amount.to_list() == [-1500.0, 2000.0, 12.5]


def test_conflicting_conventions_degrade_to_null() -> None:
    """A column mixing both thousands conventions cannot be normalized safely."""
    series = pl.Series("Giá", ["1,234", "5,678", "9,012", "3", "3.456", "7.890", "1.234", "2.345"])
    amount, report = build_amount_column(series)
    values = amount.to_list()
    # Values from the minority convention become NULL instead of a 1000x scale error.
    assert values[0] is None
    assert values[4] == 3456
    assert report["mode"] in ("THOUSAND_COMMA", "THOUSAND_DOT")


def test_existing_custom_object_name_is_not_overwritten(tmp_path: Path) -> None:
    """A real table already named X__c must survive; the derived view is skipped."""
    tables = {
        "HoaDon": pl.DataFrame(
            {
                "Mã": ["A1", "A2"],
                "Ngày": ["01/03/2025", "02/03/2025"],
                "Tổng tiền": ["1,000", "2,000"],
                "Trạng thái": ["Đã giao", "Hủy"],
            }
        ),
        "HoaDon__c": pl.DataFrame({"Z": ["x", "y"]}),
    }
    reports: dict = {}
    db_path = write_tables_to_sqlite(
        tables, tmp_path / "biz.db", create_views=True, canonical_reports=reports
    )
    conn = sqlite3.connect(db_path)
    try:
        rows = list(conn.execute('SELECT "Z" FROM "HoaDon__c"'))
        assert rows == [("x",), ("y",)]
        assert reports["skipped_collisions"] == ["HoaDon__c"]
    finally:
        conn.close()


def test_unified_canonical_view_uses_explicit_column_order(tmp_path: Path) -> None:
    """Partitions declaring the same columns in another order must not misalign."""
    left = pl.DataFrame(
        {
            "Mã đơn": ["A1", "A2"],
            "Ngày": ["01/04/2025", "02/04/2025"],
            "Tổng tiền": ["1,000", "2,000"],
            "Trạng thái": ["Đã giao", "Hủy"],
        }
    )
    right = left.select(["Trạng thái", "Tổng tiền", "Ngày", "Mã đơn"]).with_columns(
        pl.Series("Mã đơn", ["B1", "B2"])
    )
    db_path = write_tables_to_sqlite(
        {"SoBanHang_04_2025": left, "SoBanHang_05_2025": right},
        tmp_path / "biz.db",
        create_views=True,
    )
    conn = sqlite3.connect(db_path)
    try:
        rows = list(
            conn.execute(
                'SELECT "Mã đơn", "_date", "_status" FROM "SoBanHang__c" ORDER BY "Mã đơn"'
            )
        )
        assert ("B1", "2025-04-01", "completed") in rows
        assert all(row[1] is not None for row in rows)
    finally:
        conn.close()
