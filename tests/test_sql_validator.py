"""Unit tests for SQLValidator static AST checking, schema grounding, and anti-pattern detection."""

from __future__ import annotations

import time

from zaic_2026_4.submission.sql_validator import (
    SQLValidator,
    clean_sql_string,
    parse_structured_sql_output,
)


def test_clean_sql_string() -> None:
    raw_fence = "```sql\nSELECT * FROM ban_hang;\n```"
    assert clean_sql_string(raw_fence) == "SELECT * FROM ban_hang;"

    raw_plain = "  SELECT 1;  "
    assert clean_sql_string(raw_plain) == "SELECT 1;"


def test_syntax_validation() -> None:
    # Valid syntax
    res_valid = SQLValidator.validate("SELECT id, name FROM users WHERE id = 1;")
    assert res_valid.is_valid
    assert len(res_valid.blocking_issues()) == 0

    # Malformed syntax
    res_invalid = SQLValidator.validate("SELECT FROM WHERE ;")
    assert not res_invalid.is_valid
    assert "SQL_SYNTAX" in res_invalid.codes()


def test_safety_mutation_rejection() -> None:
    # Rejects INSERT
    res_insert = SQLValidator.validate("INSERT INTO users VALUES (1, 'Alice');")
    assert not res_insert.is_valid
    assert "SQL_NOT_READ_ONLY" in res_insert.codes() or "SQL_NOT_SELECT" in res_insert.codes()

    # Rejects DROP
    res_drop = SQLValidator.validate("DROP TABLE users;")
    assert not res_drop.is_valid
    assert "SQL_NOT_SELECT" in res_drop.codes()

    # Rejects Multiple statements
    res_multi = SQLValidator.validate("SELECT 1; SELECT 2;")
    assert not res_multi.is_valid
    assert "SQL_MULTIPLE_STATEMENTS" in res_multi.codes()


def test_check_1_null_ascending_order_demoted_to_warning() -> None:
    # ASC without NOT NULL check -> Now a WARNING, is_valid remains True
    sql_bad = 'SELECT id, price FROM orders ORDER BY price ASC;'
    res_bad = SQLValidator.validate(sql_bad)
    assert res_bad.is_valid
    assert "ORDER_ASC_NULLABLE" in res_bad.codes()
    assert len(res_bad.blocking_issues()) == 0
    assert len(res_bad.warning_issues()) == 1

    # ASC with NOT NULL check -> Passes cleanly
    sql_good = 'SELECT id, price FROM orders WHERE price IS NOT NULL ORDER BY price ASC;'
    res_good = SQLValidator.validate(sql_good)
    assert res_good.is_valid
    assert "ORDER_ASC_NULLABLE" not in res_good.codes()

    # If metadata specifies null_count == 0 -> No warning emitted
    meta_not_null = {
        "orders.id": {"null_count": 0, "data_format": "STRING"},
        "orders.price": {"null_count": 0, "data_format": "NUMERIC"},
    }
    res_clean_meta = SQLValidator.validate(sql_bad, metadata=meta_not_null)
    assert res_clean_meta.is_valid
    assert "ORDER_ASC_NULLABLE" not in res_clean_meta.codes()


def test_check_2_aggregate_min_null_check_demoted_to_warning() -> None:
    # MIN(f) without NOT NULL -> Now a WARNING, is_valid remains True
    sql_bad = 'SELECT MIN(don_gia) FROM hoa_don_chi_tiet;'
    res_bad = SQLValidator.validate(sql_bad)
    assert res_bad.is_valid
    assert "MIN_NULLABLE" in res_bad.codes()
    assert len(res_bad.blocking_issues()) == 0

    # MIN(f) with NOT NULL -> Passes cleanly
    sql_good = 'SELECT MIN(don_gia) FROM hoa_don_chi_tiet WHERE don_gia IS NOT NULL;'
    res_good = SQLValidator.validate(sql_good)
    assert res_good.is_valid
    assert "MIN_NULLABLE" not in res_good.codes()


def test_check_3_nested_subquery_min_max_allowed() -> None:
    # Nested subquery with MIN is now completely allowed (e.g. latest-per-group)
    sql_subquery = (
        'SELECT ten_hang, gia FROM hang_hoa '
        'WHERE gia = (SELECT MIN(gia) FROM hang_hoa WHERE gia IS NOT NULL);'
    )
    res = SQLValidator.validate(sql_subquery)
    assert res.is_valid
    assert len(res.blocking_issues()) == 0


def test_check_4_string_concatenation_in_select() -> None:
    # Using || in SELECT on columns -> WARNING
    sql_bad_pipe = 'SELECT ho || " " || ten FROM khach_hang;'
    res_bad_pipe = SQLValidator.validate(sql_bad_pipe)
    assert res_bad_pipe.is_valid
    assert "CONCAT_IN_SELECT" in res_bad_pipe.codes()

    # Using CONCAT in SELECT on columns -> WARNING
    sql_bad_concat = 'SELECT CONCAT(ho, ten) FROM khach_hang;'
    res_bad_concat = SQLValidator.validate(sql_bad_concat)
    assert res_bad_concat.is_valid
    assert "CONCAT_IN_SELECT" in res_bad_concat.codes()

    # Concatenation for target_type="date" -> Allowed completely without warning
    sql_date_concat = 'SELECT (SUBSTR(ngay, 7, 4) || "-" || SUBSTR(ngay, 4, 2)) FROM bao_cao;'
    res_date = SQLValidator.validate(sql_date_concat, target_type="date")
    assert res_date.is_valid
    assert "CONCAT_IN_SELECT" not in res_date.codes()


def test_ast_scope_schema_grounding_diagnostics() -> None:
    """Test detection of TABLE_NOT_FOUND, COLUMN_NOT_FOUND, and UNKNOWN_QUALIFIER."""
    metadata = {
        "metadata_profile": {
            "don_hang.Mã đơn": {"data_format": "STRING", "null_count": 0},
            "don_hang.Ngày đặt": {"data_format": "SLASH_DATE", "null_count": 0},
            "don_hang.Tổng tiền": {"data_format": "CURRENCY", "null_count": 0},
            "chi_tiet.Mã đơn": {"data_format": "STRING", "null_count": 0},
            "chi_tiet.Mã hàng": {"data_format": "STRING", "null_count": 0},
            "calendar.Ngày": {"data_format": "SLASH_DATE", "null_count": 0},
        }
    }

    # 1. TABLE_NOT_FOUND
    res_tbl = SQLValidator.validate('SELECT * FROM khach_hang_la;', metadata=metadata)
    assert not res_tbl.is_valid
    assert "TABLE_NOT_FOUND" in res_tbl.codes()

    # 2. COLUMN_NOT_FOUND on table (real case S9: calendar."Status")
    res_col = SQLValidator.validate('SELECT T1."Status" FROM calendar AS T1;', metadata=metadata)
    assert not res_col.is_valid
    assert "COLUMN_NOT_FOUND" in res_col.codes()

    # 3. UNKNOWN_QUALIFIER (e.g. T2 not in sources)
    res_qual = SQLValidator.validate('SELECT T2."Mã đơn" FROM don_hang AS T1;', metadata=metadata)
    assert not res_qual.is_valid
    assert "UNKNOWN_QUALIFIER" in res_qual.codes()


def test_date_format_mismatch_detection() -> None:
    """Test blocking detection of DATE_FORMAT_MISMATCH on SLASH_DATE range and literal comparisons."""
    metadata = {
        "don_hang.Thời gian": {"data_format": "SLASH_DATE", "null_count": 0},
        "don_hang.Ngày tạo": {"data_format": "ISO_DATE", "null_count": 0},
    }

    # Case 1: BETWEEN on SLASH_DATE directly (Real case Q1 & Q5) -> Blocking
    sql_slash_between = (
        'SELECT SUM(1) FROM don_hang AS T1 '
        'WHERE T1."Thời gian" BETWEEN "01/09/2026" AND "07/09/2026";'
    )
    res_slash_between = SQLValidator.validate(sql_slash_between, target_type="number", metadata=metadata)
    assert not res_slash_between.is_valid
    assert "DATE_FORMAT_MISMATCH" in res_slash_between.codes()

    # Case 2: Equality against ISO literal on SLASH_DATE -> Blocking
    sql_slash_iso = 'SELECT SUM(1) FROM don_hang AS T1 WHERE T1."Thời gian" = "2026-09-01";'
    res_slash_iso = SQLValidator.validate(sql_slash_iso, target_type="number", metadata=metadata)
    assert not res_slash_iso.is_valid
    assert "DATE_FORMAT_MISMATCH" in res_slash_iso.codes()

    # Case 3: Normalized SLASH_DATE with SUBSTR -> Passes cleanly
    sql_norm_date = (
        'SELECT SUM(1) FROM don_hang AS T1 '
        'WHERE (SUBSTR(T1."Thời gian", 7, 4) || "-" || SUBSTR(T1."Thời gian", 4, 2) || "-" || SUBSTR(T1."Thời gian", 1, 2)) '
        'BETWEEN "2026-09-01" AND "2026-09-07";'
    )
    res_norm = SQLValidator.validate(sql_norm_date, target_type="number", metadata=metadata)
    assert res_norm.is_valid
    assert "DATE_FORMAT_MISMATCH" not in res_norm.codes()


def test_shape_mismatch_and_tiebreak_rules() -> None:
    """Test SHAPE_MISMATCH, SHAPE_SUSPECT_NO_AGGREGATE, and RANK_TIEBREAK_MISSING."""
    # target_type="number" but SELECT 2 columns -> Blocking SHAPE_MISMATCH
    sql_two_cols = 'SELECT 1, 2;'
    res_shape = SQLValidator.validate(sql_two_cols, target_type="number")
    assert not res_shape.is_valid
    assert "SHAPE_MISMATCH" in res_shape.codes()

    # target_type="table" but SELECT 1 column -> Blocking SHAPE_MISMATCH
    sql_one_col = 'SELECT 1;'
    res_table_shape = SQLValidator.validate(sql_one_col, target_type="table")
    assert not res_table_shape.is_valid
    assert "SHAPE_MISMATCH" in res_table_shape.codes()

    # target_type="number" with FROM but no aggregate or LIMIT 1 -> Warning SHAPE_SUSPECT_NO_AGGREGATE
    sql_no_agg = 'SELECT don_gia FROM san_pham;'
    res_no_agg = SQLValidator.validate(sql_no_agg, target_type="number")
    assert res_no_agg.is_valid
    assert "SHAPE_SUSPECT_NO_AGGREGATE" in res_no_agg.codes()

    # ORDER BY ... LIMIT 1 with only 1 sort key -> Warning RANK_TIEBREAK_MISSING
    sql_tiebreak = 'SELECT ten_hang FROM san_pham ORDER BY don_gia DESC LIMIT 1;'
    res_tiebreak = SQLValidator.validate(sql_tiebreak, target_type="string")
    assert res_tiebreak.is_valid
    assert "RANK_TIEBREAK_MISSING" in res_tiebreak.codes()


def test_metric_m1_corpus_12_valid_queries() -> None:
    """Metric M1: Corpus of 12 valid SQL queries must all yield is_valid=True with 0 blocking issues."""
    metadata = {
        "metadata_profile": {
            "ban_hang.Tên hàng": {"data_format": "STRING", "null_count": 0},
            "ban_hang.Số lượng": {"data_format": "NUMERIC", "null_count": 0},
            "ban_hang.Mã hàng": {"data_format": "STRING", "null_count": 0},
            "ban_hang.Loại": {"data_format": "STRING", "null_count": 0},
            "don_hang.Mã đơn": {"data_format": "STRING", "null_count": 0},
            "don_hang.Tổng tiền": {"data_format": "NUMERIC", "null_count": 0},
            "don_hang.Mã khách": {"data_format": "STRING", "null_count": 0},
            "don_hang.Trạng thái": {"data_format": "STRING", "null_count": 0},
            "don_hang.Ngày đặt": {"data_format": "SLASH_DATE", "null_count": 0},
            "bao_cao.Ngày": {"data_format": "SLASH_DATE", "null_count": 0},
            "bao_cao.Doanh thu": {"data_format": "NUMERIC", "null_count": 0},
            "khach_hang.Mã KH": {"data_format": "STRING", "null_count": 0},
            "khach_hang.Tên": {"data_format": "STRING", "null_count": 0},
            "khach_hang.Ngày tạo": {"data_format": "ISO_DATE", "null_count": 0},
            "san_pham.Đơn giá": {"data_format": "NUMERIC", "null_count": 0},
            "chi_tiet.Mã đơn": {"data_format": "STRING", "null_count": 0},
            "chi_tiet.Tên SP": {"data_format": "STRING", "null_count": 0},
            "ton_kho.Mã HH": {"data_format": "STRING", "null_count": 0},
            "ton_kho.Số lượng": {"data_format": "NUMERIC", "null_count": 0},
            "doanh_thu.Tên shop": {"data_format": "STRING", "null_count": 0},
            "doanh_thu.Doanh số": {"data_format": "NUMERIC", "null_count": 0},
        }
    }

    corpus = [
        ('SELECT "Tên hàng", SUM("Số lượng") FROM "ban_hang" WHERE "Số lượng" IS NOT NULL GROUP BY "Tên hàng" ORDER BY SUM("Số lượng") DESC, "Mã hàng" ASC LIMIT 5;', "table"),
        ('SELECT "Mã đơn", "Tổng tiền" FROM "don_hang" WHERE "Tổng tiền" IS NOT NULL ORDER BY "Tổng tiền" ASC;', "table"),
        ("SELECT (SUBSTR(\"Ngày\", 7, 4) || '-' || SUBSTR(\"Ngày\", 4, 2)) AS \"Tháng\", SUM(\"Doanh thu\") FROM \"bao_cao\" GROUP BY 1;", "table"),
        ('SELECT T1."Mã KH", T1."Tên" FROM "khach_hang" AS T1 WHERE T1."Ngày tạo" = (SELECT MIN(T2."Ngày tạo") FROM "khach_hang" AS T2 WHERE T2."Ngày tạo" IS NOT NULL);', "table"),
        ("SELECT COUNT(DISTINCT \"Mã khách\") FROM \"don_hang\" WHERE \"Trạng thái\" = 'Hoàn thành';", "number"),
        ('SELECT AVG("Đơn giá") FROM "san_pham" WHERE "Đơn giá" > 0;', "number"),
        ('SELECT T1."Mã đơn", T2."Tên SP" FROM "don_hang" AS T1 JOIN "chi_tiet" AS T2 ON T1."Mã đơn" = T2."Mã đơn";', "table"),
        ("SELECT SUM(CAST(REPLACE(\"Tổng tiền\", ',', '') AS INTEGER)) FROM \"don_hang\";", "number"),
        ('SELECT "Mã HH" FROM "ton_kho" WHERE "Số lượng" <= 10 ORDER BY "Số lượng" ASC, "Mã HH" ASC;', "list"),
        ("SELECT \"Ngày đặt\" FROM \"don_hang\" WHERE (SUBSTR(\"Ngày đặt\", 7, 4) || '-' || SUBSTR(\"Ngày đặt\", 4, 2) || '-' || SUBSTR(\"Ngày đặt\", 1, 2)) BETWEEN '2026-01-01' AND '2026-01-31';", "list"),
        ('SELECT T1."Loại", COUNT(1) FROM "ban_hang" AS T1 GROUP BY T1."Loại" HAVING COUNT(1) > 5;', "table"),
        ('SELECT "Tên shop", "Doanh số" FROM "doanh_thu" ORDER BY "Doanh số" DESC, "Tên shop" ASC LIMIT 1;', "table"),
    ]

    for sql, t_type in corpus:
        res = SQLValidator.validate(sql, target_type=t_type, metadata=metadata)
        assert res.is_valid, f"Expected valid query for: {sql}, issues: {res.issue_texts}"
        assert len(res.blocking_issues()) == 0


def test_metric_m6_validator_speed() -> None:
    """Metric M6: Validator execution time must be <= 5 ms per query on average."""
    sql = (
        'SELECT T1."Tên hàng", SUM(T1."Số lượng") '
        'FROM "ban_hang" AS T1 '
        'WHERE T1."Số lượng" IS NOT NULL '
        'GROUP BY T1."Tên hàng" '
        'ORDER BY SUM(T1."Số lượng") DESC, T1."Mã hàng" ASC LIMIT 5;'
    )
    # Warm-up
    SQLValidator.validate(sql, target_type="table")

    start = time.perf_counter()
    n_iterations = 100
    for _ in range(n_iterations):
        SQLValidator.validate(sql, target_type="table")
    elapsed_ms = (time.perf_counter() - start) * 1000 / n_iterations

    assert elapsed_ms < 5.0, f"Validator too slow: {elapsed_ms:.2f} ms/query (target <= 5.0 ms)"


def test_parse_structured_sql_output() -> None:
    """Test structured JSON parsing and emission of canonical ValidationIssue."""
    # Clean JSON
    raw = '{"reasoning": "<think>test</think>", "sql": "SELECT 1;", "type": "number"}'
    sql, target_type, issues = parse_structured_sql_output(raw)
    assert sql == "SELECT 1;"
    assert target_type == "number"
    assert len(issues) == 0

    # Invalid type
    raw_bad_type = '{"sql": "SELECT 1;", "type": "invalid_type"}'
    sql, target_type, issues = parse_structured_sql_output(raw_bad_type)
    assert any(i.code == "TYPE_INVALID" for i in issues)

    # Empty string
    sql, target_type, issues = parse_structured_sql_output("")
    assert any(i.code == "OUTPUT_EMPTY" for i in issues)
