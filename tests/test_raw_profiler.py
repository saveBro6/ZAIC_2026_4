"""Unit and integration tests for RawDatabaseProfiler, Shape & Pattern, and MinHash joins."""

import json
import sqlite3
from pathlib import Path

import polars as pl
import pytest

from zaic_2026_4.profiling.raw_profiler import RawDatabaseProfiler


def create_sqlite_db_from_dfs(
    db_path: Path,
    tables: dict[str, pl.DataFrame],
    views: list[tuple[str, str]] | None = None,
) -> None:
    """Helper to create an SQLite database with tables and optional views."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    for tname, df in tables.items():
        quoted_cols = ", ".join(f'"{c}" TEXT' for c in df.columns)
        cur.execute(f'CREATE TABLE "{tname}" ({quoted_cols});')
        if df.height > 0:
            placeholders = ", ".join(["?"] * len(df.columns))
            rows = df.iter_rows()
            cur.executemany(
                f'INSERT INTO "{tname}" VALUES ({placeholders});',
                [[str(v) if v is not None else None for v in r] for r in rows],
            )

    if views:
        for vname, vsql in views:
            cur.execute(vsql)

    conn.commit()
    conn.close()


def test_empty_table_profiling(tmp_path: Path) -> None:
    """Test profiling an empty table (0 rows)."""
    db_path = tmp_path / "empty.db"
    df = pl.DataFrame({"id": [], "name": []}, schema={"id": pl.String, "name": pl.String})
    create_sqlite_db_from_dfs(db_path, {"empty_table": df})

    profiler = RawDatabaseProfiler(db_path)
    profile = profiler.profile_database()

    assert "empty_table" in profile["tables"]
    tbl = profile["tables"]["empty_table"]
    assert tbl["row_count"] == 0
    assert tbl["columns"]["id"]["null_count"] == 0
    assert tbl["columns"]["id"]["null_ratio"] == 0.0
    assert tbl["columns"]["id"]["distinct_count"] == 0
    assert tbl["columns"]["id"]["sample_values"] == []


def test_null_column_profiling(tmp_path: Path) -> None:
    """Test profiling a column with 100% NULL values."""
    db_path = tmp_path / "nulls.db"
    df = pl.DataFrame(
        {
            "all_null": [None, None, None],
            "some_data": ["val1", "val2", None],
        },
        schema={"all_null": pl.String, "some_data": pl.String},
    )
    create_sqlite_db_from_dfs(db_path, {"test_nulls": df})

    profiler = RawDatabaseProfiler(db_path)
    profile = profiler.profile_database()

    tbl = profile["tables"]["test_nulls"]
    assert tbl["row_count"] == 3
    assert tbl["columns"]["all_null"]["null_count"] == 3
    assert tbl["columns"]["all_null"]["null_ratio"] == 1.0
    assert tbl["columns"]["all_null"]["distinct_count"] == 0
    assert tbl["columns"]["all_null"]["data_format"] == "EMPTY"

    assert tbl["columns"]["some_data"]["null_count"] == 1
    assert tbl["columns"]["some_data"]["distinct_count"] == 2
    assert set(tbl["columns"]["some_data"]["sample_values"]) == {"val1", "val2"}


def test_shape_and_pattern_recognition(tmp_path: Path) -> None:
    """Test recognition of JSON, ISO dates, Vietnamese currency, and text."""
    db_path = tmp_path / "patterns.db"
    df = pl.DataFrame(
        {
            "json_col": ['{"key": 1}', '{"key": 2}', '{"key": 3}'],
            "date_col": ["2025-01-01", "2025-02-15", "2025-03-30"],
            "currency_col": ["560,000", "4,000", "1,200,000"],
            "text_col": ["Sản phẩm A", "Sản phẩm B", "Sản phẩm C"],
        }
    )
    create_sqlite_db_from_dfs(db_path, {"patterns": df})

    profiler = RawDatabaseProfiler(db_path)
    profile = profiler.profile_database()
    cols = profile["tables"]["patterns"]["columns"]

    assert cols["json_col"]["is_json"] is True
    assert cols["json_col"]["data_format"] == "JSON_STRING"

    assert cols["date_col"]["is_json"] is False
    assert cols["date_col"]["data_format"] == "ISO_DATE"

    assert cols["currency_col"]["data_format"] == "CURRENCY"
    assert cols["text_col"]["data_format"] == "STRING"


def test_top_10_and_string_lengths(tmp_path: Path) -> None:
    """Test top 10 frequencies and min/max string length calculation."""
    db_path = tmp_path / "stats.db"
    values = ["A"] * 50 + ["B"] * 30 + ["C"] * 15 + ["DDDD"] * 5
    df = pl.DataFrame({"code": values})
    create_sqlite_db_from_dfs(db_path, {"stats": df})

    profiler = RawDatabaseProfiler(db_path)
    profile = profiler.profile_database()
    col = profile["tables"]["stats"]["columns"]["code"]

    assert col["min_str_len"] == 1
    assert col["max_str_len"] == 4
    assert col["top_10_samples"]["A"] == 50
    assert col["top_10_samples"]["B"] == 30
    assert col["top_10_samples"]["C"] == 15
    assert col["top_10_samples"]["DDDD"] == 5


def test_skip_partition_tables_with_unified_view(tmp_path: Path) -> None:
    """Verify that partition tables are skipped when unified view is present."""
    db_path = tmp_path / "partitions.db"
    df_t1 = pl.DataFrame({"id": ["1", "2"], "amount": ["100", "200"]})
    df_t2 = pl.DataFrame({"id": ["3", "4"], "amount": ["300", "400"]})
    df_dim = pl.DataFrame({"id": ["1", "2", "3", "4"], "desc": ["A", "B", "C", "D"]})

    view_sql = (
        'CREATE VIEW "DonHang_ChiTiet" AS '
        'SELECT "id", "amount" FROM "DonHang_ChiTiet_T01" '
        'UNION ALL SELECT "id", "amount" FROM "DonHang_ChiTiet_T02";'
    )

    create_sqlite_db_from_dfs(
        db_path,
        {
            "DonHang_ChiTiet_T01": df_t1,
            "DonHang_ChiTiet_T02": df_t2,
            "DanhMuc": df_dim,
        },
        views=[("DonHang_ChiTiet", view_sql)],
    )

    profiler = RawDatabaseProfiler(db_path, skip_partition_tables_if_view_exists=True)
    profile = profiler.profile_database()

    assert "DonHang_ChiTiet" in profile["tables"]
    assert "DanhMuc" in profile["tables"]
    assert "DonHang_ChiTiet_T01" not in profile["tables"]
    assert "DonHang_ChiTiet_T02" not in profile["tables"]

    v_prof = profile["tables"]["DonHang_ChiTiet"]
    assert v_prof["is_unified_view"] is True
    assert v_prof["row_count"] == 4
    assert "DonHang_ChiTiet_T01" in v_prof["source_tables"]
    assert "DonHang_ChiTiet_T02" in v_prof["source_tables"]


def test_undocumented_join_detection(tmp_path: Path) -> None:
    """Verify MinHash Jaccard resemblance detects FK-PK candidate joins."""
    db_path = tmp_path / "joins.db"
    shared_codes = [f"SP_{i:04d}" for i in range(100)]

    df_orders = pl.DataFrame(
        {
            "order_id": [f"ORD_{i}" for i in range(100)],
            "product_code": shared_codes,
        }
    )
    df_products = pl.DataFrame(
        {
            "sku": shared_codes,
            "product_name": [f"Product {i}" for i in range(100)],
        }
    )

    create_sqlite_db_from_dfs(
        db_path,
        {"orders": df_orders, "products": df_products},
    )

    profiler = RawDatabaseProfiler(db_path, jaccard_threshold=0.7, cross_table_only=True)
    profile = profiler.profile_database()

    joins = profile["undocumented_joins"]
    assert len(joins) >= 1

    join_keys = [(j["col1"], j["col2"]) for j in joins]
    matched = any(
        ("orders.product_code" in pair and "products.sku" in pair)
        for pair in join_keys
    )
    assert matched is True
    top_join = joins[0]
    assert top_join["resemblance"] >= 0.95


def test_downstream_contract_compatibility(tmp_path: Path) -> None:
    """Verify output dictionary contains all keys required by downstream modules."""
    db_path = tmp_path / "contract.db"
    df = pl.DataFrame({"col_a": ["1", "2", "3"]})
    create_sqlite_db_from_dfs(db_path, {"tbl": df})

    profiler = RawDatabaseProfiler(db_path)
    profile = profiler.profile_database()

    assert "tables" in profile
    assert "undocumented_joins" in profile

    tbl = profile["tables"]["tbl"]
    assert "row_count" in tbl
    assert "columns" in tbl
    assert "is_view" in tbl
    assert "is_unified_view" in tbl
    assert "source_tables" in tbl

    col = tbl["columns"]["col_a"]
    required_keys = [
        "null_count",
        "distinct_count",
        "null_ratio",
        "min_val",
        "max_val",
        "min_str_len",
        "max_str_len",
        "data_format",
        "is_json",
        "top_10_samples",
        "sample_values",
        "is_fixed_length",
        "is_free_text",
        "date_subtype",
        "currency_subtype",
        "currency_scale_multiplier",
        "llm_hint",
    ]
    for k in required_keys:
        assert k in col, f"Missing required contract key: {k}"

    serialized = json.dumps(profile, ensure_ascii=False)
    assert len(serialized) > 0


def test_column_named_count_no_crash(tmp_path: Path) -> None:
    """Verify that a column named 'count' does not cause DuplicateError."""
    db_path = tmp_path / "count_col.db"
    df = pl.DataFrame({"count": [1, 2, 2, 3], "name": ["A", "B", "B", "C"]})
    create_sqlite_db_from_dfs(db_path, {"tbl": df})

    profiler = RawDatabaseProfiler(db_path)
    profile = profiler.profile_database()

    col = profile["tables"]["tbl"]["columns"]["count"]
    assert col["null_count"] == 0
    assert col["distinct_count"] == 3
    assert col["top_10_samples"]["2"] == 2


def test_numeric_range_not_inverted(tmp_path: Path) -> None:
    """Verify that numeric columns have numerically correct min_val and max_val."""
    db_path = tmp_path / "num_range.db"
    df = pl.DataFrame({"num": [2, 10, 100, 5]})
    create_sqlite_db_from_dfs(db_path, {"tbl": df})

    profiler = RawDatabaseProfiler(db_path)
    profile = profiler.profile_database()

    col = profile["tables"]["tbl"]["columns"]["num"]
    assert float(col["min_val"]) == 2.0
    assert float(col["max_val"]) == 100.0


def test_currency_and_formatted_min_max_numeric_order(tmp_path: Path) -> None:
    """Verify that formatted currency, thousand-separated numbers, and custom dates have numerically correct min_val and max_val."""
    db_path = tmp_path / "currency_min_max.db"
    df = pl.DataFrame(
        {
            "currency_thousand": ["1,056,000", "90,000", "4,000", "1,200,000"],
            "currency_vnd": ["50,000 VNĐ", "15,000 VNĐ", "1,500,000 VNĐ", "200,000 VNĐ"],
            "currency_accounting": ["(5,000)", "2,000", "10,000", "(100)"],
            "currency_dash": ["-", "5,000", "100,000", "20,000"],
            "date_vn": ["1 thg 1 2026", "9 thg 9 2025", "15 thg 6 2025", "20 thg 11 2025"],
            "date_slash": ["25/09/2025", "01/10/2025", "15/01/2024", "10/05/2024"],
        }
    )
    create_sqlite_db_from_dfs(db_path, {"tbl": df})

    profiler = RawDatabaseProfiler(db_path)
    profile = profiler.profile_database()
    cols = profile["tables"]["tbl"]["columns"]

    # Thousand separator: 4,000 is min, 1,200,000 is max (not lexicographical 1,056,000 and 90,000)
    assert cols["currency_thousand"]["min_val"] == "4,000"
    assert cols["currency_thousand"]["max_val"] == "1,200,000"

    # VND text: 15,000 VNĐ is min, 1,500,000 VNĐ is max
    assert cols["currency_vnd"]["min_val"] == "15,000 VNĐ"
    assert cols["currency_vnd"]["max_val"] == "1,500,000 VNĐ"

    # Accounting parentheses: (5,000) is min (-5000), 10,000 is max
    assert cols["currency_accounting"]["min_val"] == "(5,000)"
    assert cols["currency_accounting"]["max_val"] == "10,000"

    # Accounting dash zero: - is min (0), 100,000 is max
    assert cols["currency_dash"]["min_val"] == "-"
    assert cols["currency_dash"]["max_val"] == "100,000"

    # Vietnamese date: 15 thg 6 2025 is min, 1 thg 1 2026 is max
    assert cols["date_vn"]["min_val"] == "15 thg 6 2025"
    assert cols["date_vn"]["max_val"] == "1 thg 1 2026"

    # Slash date: 15/01/2024 is min, 01/10/2025 is max (not lexicographical 01/10/2025)
    assert cols["date_slash"]["min_val"] == "15/01/2024"
    assert cols["date_slash"]["max_val"] == "01/10/2025"


def test_fixed_length_and_free_text(tmp_path: Path) -> None:
    """Verify fixed length code vs free-text detection."""
    db_path = tmp_path / "text_types.db"
    df = pl.DataFrame(
        {
            "code_fixed": ["SP001", "SP002", "SP003"],
            "long_note": ["A" * 300, "B" * 350, "C" * 280],
        }
    )
    create_sqlite_db_from_dfs(db_path, {"tbl": df})

    profiler = RawDatabaseProfiler(db_path)
    profile = profiler.profile_database()

    cols = profile["tables"]["tbl"]["columns"]
    assert cols["code_fixed"]["is_fixed_length"] is True
    assert cols["code_fixed"]["is_free_text"] is False
    assert cols["code_fixed"]["data_format"] == "FIXED_CODE"

    assert cols["long_note"]["is_fixed_length"] is False
    assert cols["long_note"]["is_free_text"] is True
    assert cols["long_note"]["data_format"] == "FREE_TEXT"


def test_excel_custom_date_formats(tmp_path: Path) -> None:
    """Verify recognition of all Excel custom date patterns."""
    db_path = tmp_path / "dates.db"
    df = pl.DataFrame(
        {
            "slash_full": ["05/03/2026", "15/08/2025"],
            "slash_short": ["5/3/26", "1/1/25"],
            "vn_abbr": ["5 thg 3 2026", "1 thg 1 2025"],
            "en_abbr": ["05-Mar-2026", "12-Dec-2025"],
            "vn_full": ["Ngày 05 tháng 03 năm 2026", "ngày 1 tháng 1 năm 2025"],
            "en_full": ["March 05, 2026", "December 12, 2025"],
            "vn_weekday": ["Thứ Năm, 05/03/2026", "Chủ Nhật, 01/01/2025"],
            "en_weekday": ["Thu, 05 Mar 2026", "Sun, 01 Jan 2025"],
            "excel_serial": ["45717", "45658"],
        }
    )
    create_sqlite_db_from_dfs(db_path, {"tbl": df})

    profiler = RawDatabaseProfiler(db_path)
    profile = profiler.profile_database()
    cols = profile["tables"]["tbl"]["columns"]

    assert cols["slash_full"]["date_subtype"] == "SLASH_DATE"
    assert cols["slash_short"]["date_subtype"] == "SLASH_DATE_SHORT"
    assert cols["vn_abbr"]["date_subtype"] == "VN_DATE_ABBR"
    assert cols["en_abbr"]["date_subtype"] == "EN_DATE_ABBR"
    assert cols["vn_full"]["date_subtype"] == "VN_DATE_FULL"
    assert cols["en_full"]["date_subtype"] == "EN_DATE_FULL"
    assert cols["vn_weekday"]["date_subtype"] == "VN_DATE_WEEKDAY"
    assert cols["en_weekday"]["date_subtype"] == "EN_DATE_WEEKDAY"
    assert cols["excel_serial"]["date_subtype"] == "EXCEL_SERIAL"


def test_mixed_date_formats_in_single_column(tmp_path: Path) -> None:
    """Verify mixed date column (e.g. biz_117 with serials + abbr + slash)."""
    db_path = tmp_path / "mixed_date.db"
    df = pl.DataFrame({"date_col": ["45717", "1 thg 3 2025", "01/03/2025"]})
    create_sqlite_db_from_dfs(db_path, {"tbl": df})

    profiler = RawDatabaseProfiler(db_path)
    profile = profiler.profile_database()
    col = profile["tables"]["tbl"]["columns"]["date_col"]

    assert col["data_format"] == "DATE_MIXED"
    assert "EXCEL_SERIAL" in col["date_subtype"]
    assert "VN_DATE_ABBR" in col["date_subtype"]
    assert "SLASH_DATE" in col["date_subtype"]
    assert "Mixed date representations" in col["llm_hint"]


def test_excel_custom_currency_formats(tmp_path: Path) -> None:
    """Verify recognition of all Excel custom VNĐ currency patterns."""
    db_path = tmp_path / "currencies.db"
    df = pl.DataFrame(
        {
            "vnd_text": ["1,500,000 VNĐ", "-50,000 VNĐ", "0 VNĐ"],
            "vnd_symbol": ["1,500,000 ₫", "-50,000 ₫", "0 ₫"],
            "accounting_paren": ["1,500,000 ₫", "(50,000 ₫)", "0 ₫"],
            "accounting_dash": ["1,500,000 ₫", "- ₫", "200,000 ₫"],
            "scaled_k": ["1,500 k VNĐ", "500 k VNĐ", "0 k VNĐ"],
            "scaled_tr": ["1.5 tr VNĐ", "2.0 tr VNĐ", "0.5 tr VNĐ"],
            "scaled_ty": ["1.50 tỷ", "2.30 tỷ", "0.00 tỷ"],
        }
    )
    create_sqlite_db_from_dfs(db_path, {"tbl": df})

    profiler = RawDatabaseProfiler(db_path)
    profile = profiler.profile_database()
    cols = profile["tables"]["tbl"]["columns"]

    assert cols["vnd_text"]["currency_subtype"] == "VND_TEXT"
    assert cols["vnd_text"]["currency_scale_multiplier"] == 1.0

    assert cols["vnd_symbol"]["currency_subtype"] == "VND_SYMBOL"
    assert cols["vnd_symbol"]["currency_scale_multiplier"] == 1.0

    assert cols["accounting_paren"]["currency_subtype"] == "ACCOUNTING_PARENTHESES"
    assert cols["accounting_paren"]["currency_scale_multiplier"] == -1.0
    assert "Accounting negative currency format" in cols["accounting_paren"]["llm_hint"]

    assert cols["accounting_dash"]["currency_subtype"] == "ACCOUNTING_DASH_ZERO"
    assert cols["accounting_dash"]["currency_scale_multiplier"] == 0.0

    assert cols["scaled_k"]["currency_subtype"] == "SCALED_THOUSAND_K"
    assert cols["scaled_k"]["currency_scale_multiplier"] == 1000.0

    assert cols["scaled_tr"]["currency_subtype"] == "SCALED_MILLION_TR"
    assert cols["scaled_tr"]["currency_scale_multiplier"] == 1000000.0

    assert cols["scaled_ty"]["currency_subtype"] == "SCALED_BILLION_TY"
    assert cols["scaled_ty"]["currency_scale_multiplier"] == 1000000000.0


def test_real_db_biz_101() -> None:
    """Integration test against actual SQLite database in tmp_dev/train/biz_101.db."""
    db_path = Path("tmp_dev/train/biz_101.db")
    if not db_path.exists():
        pytest.skip("tmp_dev/train/biz_101.db not present")

    profiler = RawDatabaseProfiler(db_path)
    profile = profiler.profile_database()

    assert "ban_hang" in profile["tables"]
    tbl = profile["tables"]["ban_hang"]
    assert tbl["row_count"] > 50000
    assert "Mã hóa đơn" in tbl["columns"]
    assert "Ngày tháng" in tbl["columns"]
    # Ngày tháng in biz_101 has values like "1 thg 6 2025"
    date_col = tbl["columns"]["Ngày tháng"]
    assert date_col["date_subtype"] == "VN_DATE_ABBR"
