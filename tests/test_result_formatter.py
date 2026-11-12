"""Unit tests for ResultFormatter and contract type conformity."""

from __future__ import annotations

from zaic_2026_4.submission.result_formatter import (
    ResultFormatter,
    derive_table_columns,
    detect_percent_scale,
    normalize_contract_string,
)


def test_normalize_contract_string() -> None:
    raw = "Sting dâu (330ml) - Chai!"
    normalized = normalize_contract_string(raw)
    assert normalized == "sting dâu 330ml chai"

    raw_accents = "Cà phê Sữa   ĐÁ"
    assert normalize_contract_string(raw_accents) == "cà phê sữa đá"


def test_format_number() -> None:
    assert ResultFormatter.format_number(48250000) == 48250000
    assert ResultFormatter.format_number(48250000.0) == 48250000
    assert ResultFormatter.format_number(12.3456) == 12.35
    assert ResultFormatter.format_number(0) == 0
    assert ResultFormatter.format_number(None) is None
    assert ResultFormatter.format_number("4.820.000") is None
    assert ResultFormatter.format_number("abc") is None
    assert ResultFormatter.format_number(True) is None
    assert ResultFormatter.format_number(False) is None


def test_format_percent() -> None:
    assert ResultFormatter.format_percent(5.27) == 5.27
    assert ResultFormatter.format_percent(-12.45) == -12.45
    assert ResultFormatter.format_percent(0.5) == 0.5
    assert ResultFormatter.format_percent(0.0) == 0.0

    # Percent vs Ratio scale
    assert ResultFormatter.format_percent(0.0527, scale="percent") == 0.05
    assert ResultFormatter.format_percent(0.0527, scale="ratio") == 5.27
    assert ResultFormatter.format_percent(0.0208, scale="ratio") == 2.08
    assert ResultFormatter.format_percent(0.5, scale="ratio") == 50.0
    # Guard |v| <= 1.0 keeps growth >= 100% as is
    assert ResultFormatter.format_percent(123.46, scale="ratio") == 123.46

    assert ResultFormatter.format_percent(None) is None
    assert ResultFormatter.format_percent("xyz") is None
    assert ResultFormatter.format_percent(True) is None


def test_format_date() -> None:
    assert ResultFormatter.format_date("2026-08-15") == "2026-08-15"
    assert ResultFormatter.format_date("2025-09") == "2025-09"
    # Convert DD/MM/YYYY
    assert ResultFormatter.format_date("15/08/2026") == "2026-08-15"
    # ISO datetime trimmed to YYYY-MM-DD
    assert ResultFormatter.format_date("2026-06-24 21:44:11") == "2026-06-24"
    assert ResultFormatter.format_date("2026-06-24T10:15:30Z") == "2026-06-24"

    assert ResultFormatter.format_date("n/a") is None
    assert ResultFormatter.format_date("invalid-date") is None
    assert ResultFormatter.format_date(None) is None
    assert ResultFormatter.format_date(True) is None


def test_detect_percent_scale() -> None:
    # 1. Mul with 100 inside ROUND -> percent
    sql1 = "SELECT ROUND(100.0 * SUM(x) / NULLIF(COUNT(*), 0), 2) FROM t;"
    assert detect_percent_scale(sql1) == "percent"

    # 2. Mul with 100.0 -> percent
    sql2 = "SELECT COUNT(x) * 100.0 / COUNT(*) FROM t;"
    assert detect_percent_scale(sql2) == "percent"

    # 3. Ratio with 1.0 (no 100) -> ratio
    sql3 = "SELECT COUNT(DISTINCT x) * 1.0 / COUNT(*) FROM t;"
    assert detect_percent_scale(sql3) == "ratio"

    # 4. AVG -> percent
    sql4 = "SELECT AVG(col) FROM t;"
    assert detect_percent_scale(sql4) == "percent"

    # 5. Literal denominator 7 -> percent
    sql5 = "SELECT SUM(x) / 7 FROM t;"
    assert detect_percent_scale(sql5) == "percent"

    # 6. 100.0 * a / b -> percent
    sql6 = "SELECT 100.0 * a / b FROM t;"
    assert detect_percent_scale(sql6) == "percent"

    # 7. ROUND(SUM(x)*1.0 / NULLIF(SUM(y), 0), 2) -> ratio
    sql7 = "SELECT ROUND(SUM(x) * 1.0 / NULLIF(SUM(y), 0), 2) FROM t;"
    assert detect_percent_scale(sql7) == "ratio"

    # 8. Plain column -> percent
    sql8 = "SELECT col FROM t;"
    assert detect_percent_scale(sql8) == "percent"

    # Fallback cases
    assert detect_percent_scale(None) == "percent"
    assert detect_percent_scale("") == "percent"
    assert detect_percent_scale("INVALID SQL @@") == "percent"


def test_derive_table_columns() -> None:
    # (a) AST projection with aliases
    sql_alias = 'SELECT name AS "Tên khách", debt AS "Số tiền nợ" FROM t;'
    cols = derive_table_columns(sql_alias, None, 2)
    assert cols == ["Tên khách", "Số tiền nợ"]

    # (a) AST projection with columns
    sql_col = "SELECT first_name, last_name FROM t;"
    cols = derive_table_columns(sql_col, None, 2)
    assert cols == ["first_name", "last_name"]

    # (a) AST projection with functions
    sql_func = "SELECT SUM(price), COUNT(id) FROM t;"
    cols = derive_table_columns(sql_func, None, 2)
    assert cols == ["sum_price", "count_id"]

    # (b) SELECT * fallback to description
    sql_star = "SELECT * FROM t;"
    desc = ["Tên khách", "Số tiền nợ"]
    cols = derive_table_columns(sql_star, desc, 2)
    assert cols == ["Tên khách", "Số tiền nợ"]

    # (c) Fallback to col_N
    cols_fb = derive_table_columns(None, None, 3)
    assert cols_fb == ["col_1", "col_2", "col_3"]

    # Deduplication
    sql_dup = "SELECT col, col FROM t;"
    cols_dup = derive_table_columns(sql_dup, None, 2)
    assert cols_dup == ["col", "col_2"]

    # UNION query projection extraction
    sql_union = "SELECT product_name, unit_price FROM store_a UNION SELECT product_name, unit_price FROM store_b"
    cols_union = derive_table_columns(sql_union, None, 2)
    assert cols_union == ["product_name", "unit_price"]

    # t1.* join query falls back to description
    sql_t1_star = "SELECT t1.*, t2.total FROM orders AS t1 JOIN totals AS t2 ON t1.id = t2.id"
    desc_join = ["order_id", "created_date", "total"]
    cols_join = derive_table_columns(sql_t1_star, desc_join, 3)
    assert cols_join == ["order_id", "created_date", "total"]

    # Single table star query t1.* falls back to description (not literal '*')
    sql_single_star = "SELECT t1.* FROM orders AS t1"
    desc_single = ["order_id"]
    cols_single = derive_table_columns(sql_single_star, desc_single, 1)
    assert cols_single == ["order_id"]


def test_format_result_scalar_cases() -> None:
    # Number
    t, val = ResultFormatter.format_result([(48250000,)], target_type="number")
    assert t == "number"
    assert val == 48250000

    # Percent (with SQL)
    sql_pct = "SELECT ROUND(COUNT(x) * 100.0 / COUNT(*), 2) FROM t;"
    t, val = ResultFormatter.format_result([(2.08,)], target_type="percent", sql=sql_pct)
    assert t == "percent"
    assert val == 2.08

    # Percent ratio scale
    sql_ratio = "SELECT COUNT(DISTINCT x) * 1.0 / COUNT(*) FROM t;"
    t, val = ResultFormatter.format_result([(0.0208,)], target_type="percent", sql=sql_ratio)
    assert t == "percent"
    assert val == 2.08

    # String
    t, val = ResultFormatter.format_result([("Sting dâu 330ml",)], target_type="string")
    assert t == "string"
    assert val == "Sting dâu 330ml"

    # Date
    t, val = ResultFormatter.format_result([("2026-08-15",)], target_type="date")
    assert t == "date"
    assert val == "2026-08-15"

    # Boolean
    t, val = ResultFormatter.format_result([(1,)], target_type="boolean")
    assert t == "boolean"
    assert val is True

    t_f, val_f = ResultFormatter.format_result([("không",)], target_type="boolean")
    assert t_f == "boolean"
    assert val_f is False


def test_format_result_list_and_table() -> None:
    # List (single column, multiple rows)
    rows_list = [("Cà phê sữa",), ("Bạc xỉu",), ("Trà đào",)]
    t, val = ResultFormatter.format_result(rows_list, target_type="list")
    assert t == "list"
    assert val == ["Cà phê sữa", "Bạc xỉu", "Trà đào"]

    # Table (multiple columns, multiple rows) with real columns
    rows_table = [("Chị Lan", 3200000), ("Anh Tú", 1500000)]
    t, val = ResultFormatter.format_result(
        rows_table,
        target_type="table",
        raw_columns=["Tên khách", "Số tiền nợ"],
    )
    assert t == "table"
    assert isinstance(val, dict)
    assert val["columns"] == ["Tên khách", "Số tiền nợ"]
    assert val["rows"] == [["Chị Lan", 3200000], ["Anh Tú", 1500000]]


def test_format_result_unanswerable_and_abstain() -> None:
    t_unans, val_unans = ResultFormatter.format_result(None, target_type="unanswerable")
    assert t_unans == "unanswerable"
    assert val_unans is None

    t_abs, val_abs = ResultFormatter.format_result(None, target_type="abstain")
    assert t_abs == "abstain"
    assert val_abs is None


def test_format_result_edge_cases_zero_and_none() -> None:
    # Legitimate non-zero and zero evidence
    t_pct, val_pct = ResultFormatter.format_result([(0,)], target_type="percent")
    assert t_pct == "percent"
    assert val_pct == 0.0

    t_num, val_num = ResultFormatter.format_result([(0,)], target_type="number")
    assert t_num == "number"
    assert val_num == 0

    # Metric M7: None/empty/unparsable cells MUST abstain, never fabricate 0 or 0.0
    t_none_agg, val_none_agg = ResultFormatter.format_result([(None,)], target_type="number")
    assert t_none_agg == "abstain"
    assert val_none_agg is None

    t_none_pct, val_none_pct = ResultFormatter.format_result([(None,)], target_type="percent")
    assert t_none_pct == "abstain"
    assert val_none_pct is None

    t_empty_num, val_empty_num = ResultFormatter.format_result([], target_type="number")
    assert t_empty_num == "abstain"
    assert val_empty_num is None

    t_empty_pct, val_empty_pct = ResultFormatter.format_result([], target_type="percent")
    assert t_empty_pct == "abstain"
    assert val_empty_pct is None

    t_bad_num, val_bad_num = ResultFormatter.format_result([("4.820.000",)], target_type="number")
    assert t_bad_num == "abstain"
    assert val_bad_num is None

    t_dt, val_dt = ResultFormatter.format_result([("2026-06-24 21:44:11",)], target_type="date")
    assert t_dt == "date"
    assert val_dt == "2026-06-24"

    t_bad_date, val_bad_date = ResultFormatter.format_result([("n/a",)], target_type="date")
    assert t_bad_date == "abstain"
    assert val_bad_date is None

    t_bad_bool, val_bad_bool = ResultFormatter.format_result([("maybe",)], target_type="boolean")
    assert t_bad_bool == "abstain"
    assert val_bad_bool is None


def test_format_result_table_single_row_multi_column_with_raw_columns() -> None:
    # 1. Single row, multiple columns with explicit target_type="table" and raw_columns
    raw_rows = [("Hà Nội", 1500000)]
    raw_cols = ["Thành phố", "Doanh thu"]
    sql = 'SELECT city AS "Thành phố", revenue AS "Doanh thu" FROM sales LIMIT 1;'
    t, val = ResultFormatter.format_result(
        raw_rows,
        target_type="table",
        raw_columns=raw_cols,
        sql=sql,
    )
    assert t == "table"
    assert val == {
        "columns": ["Thành phố", "Doanh thu"],
        "rows": [["Hà Nội", 1500000]],
    }

    # 2. Scalar predicted_type="number" recovering to 1-row table because raw_rows has 2 columns
    t_rec, val_rec = ResultFormatter.format_result(
        raw_rows,
        target_type="number",
        raw_columns=raw_cols,
        sql=sql,
    )
    assert t_rec == "table"
    assert val_rec == {
        "columns": ["Thành phố", "Doanh thu"],
        "rows": [["Hà Nội", 1500000]],
    }

