"""Unit tests for SQLValidator static AST checking with SQLglot."""

from __future__ import annotations

from zaic_2026_4.submission.sql_validator import SQLValidator, clean_sql_string


def test_clean_sql_string() -> None:
    raw_fence = "```sql\nSELECT * FROM ban_hang;\n```"
    assert clean_sql_string(raw_fence) == "SELECT * FROM ban_hang;"

    raw_plain = "  SELECT 1;  "
    assert clean_sql_string(raw_plain) == "SELECT 1;"


def test_syntax_validation() -> None:
    # Valid syntax
    res_valid = SQLValidator.validate("SELECT id, name FROM users WHERE id = 1;")
    assert res_valid.is_valid
    assert len(res_valid.issues) == 0

    # Malformed syntax
    res_invalid = SQLValidator.validate("SELECT FROM WHERE ;")
    assert not res_invalid.is_valid
    assert any("syntax error" in issue.lower() or "failed to parse" in issue.lower() for issue in res_invalid.issues)


def test_safety_mutation_rejection() -> None:
    # Rejects INSERT
    res_insert = SQLValidator.validate("INSERT INTO users VALUES (1, 'Alice');")
    assert not res_insert.is_valid
    assert any("forbidden" in issue.lower() for issue in res_insert.issues)

    # Rejects DROP
    res_drop = SQLValidator.validate("DROP TABLE users;")
    assert not res_drop.is_valid

    # Rejects Multiple statements
    res_multi = SQLValidator.validate("SELECT 1; SELECT 2;")
    assert not res_multi.is_valid
    assert any("multiple" in issue.lower() for issue in res_multi.issues)


def test_check_1_null_ascending_order() -> None:
    # ASC without NOT NULL check -> Flagged
    sql_bad = 'SELECT id, price FROM orders ORDER BY price ASC;'
    res_bad = SQLValidator.validate(sql_bad)
    assert not res_bad.is_valid
    assert any("orders by field 'price' asc without ensuring" in issue.lower() for issue in res_bad.issues)

    # ASC with NOT NULL check -> Passes
    sql_good = 'SELECT id, price FROM orders WHERE price IS NOT NULL ORDER BY price ASC;'
    res_good = SQLValidator.validate(sql_good)
    assert res_good.is_valid

    # DESC without NOT NULL -> Passes (NULLs sort last in DESC in standard SQL)
    sql_desc = 'SELECT id, price FROM orders ORDER BY price DESC;'
    res_desc = SQLValidator.validate(sql_desc)
    assert res_desc.is_valid

    # ORDER BY SUM(qty) ASC (aggregate in order by) -> Passes without false positive
    sql_agg_order = 'SELECT product_id, SUM(quantity) AS total FROM sales GROUP BY product_id ORDER BY SUM(quantity) ASC;'
    res_agg_order = SQLValidator.validate(sql_agg_order)
    assert res_agg_order.is_valid

    # ORDER BY total ASC (where total is aggregate alias) -> Passes without false positive
    sql_alias_order = 'SELECT product_id, SUM(quantity) AS total FROM sales GROUP BY product_id ORDER BY total ASC;'
    res_alias_order = SQLValidator.validate(sql_alias_order)
    assert res_alias_order.is_valid


def test_check_2_aggregate_min_null_check() -> None:
    # MIN(f) without NOT NULL -> Flagged
    sql_bad = 'SELECT MIN(don_gia) FROM hoa_don_chi_tiet;'
    res_bad = SQLValidator.validate(sql_bad)
    assert not res_bad.is_valid
    assert any("aggregate min(\"don_gia\") detected without a not null" in issue.lower() for issue in res_bad.issues)

    # MIN(f) with NOT NULL -> Passes
    sql_good = 'SELECT MIN(don_gia) FROM hoa_don_chi_tiet WHERE don_gia IS NOT NULL;'
    res_good = SQLValidator.validate(sql_good)
    assert res_good.is_valid


def test_check_3_nested_subquery_min_max() -> None:
    # Nested subquery with MIN -> Flagged
    sql_bad = (
        'SELECT ten_hang, gia FROM hang_hoa '
        'WHERE gia = (SELECT MIN(gia) FROM hang_hoa);'
    )
    res_bad = SQLValidator.validate(sql_bad)
    assert not res_bad.is_valid
    assert any("nested subquery with min/max" in issue.lower() for issue in res_bad.issues)

    # Clean query using ORDER BY ... LIMIT 1 -> Passes
    sql_good = (
        'SELECT ten_hang, gia FROM hang_hoa '
        'WHERE gia IS NOT NULL '
        'ORDER BY gia ASC LIMIT 1;'
    )
    res_good = SQLValidator.validate(sql_good)
    assert res_good.is_valid


def test_check_4_string_concatenation_in_select() -> None:
    # Using || in SELECT on columns -> Flagged
    sql_bad_pipe = 'SELECT ho || " " || ten FROM khach_hang;'
    res_bad_pipe = SQLValidator.validate(sql_bad_pipe)
    assert not res_bad_pipe.is_valid
    assert any("string concatenation" in issue.lower() for issue in res_bad_pipe.issues)

    # Using CONCAT in SELECT on columns -> Flagged
    sql_bad_concat = 'SELECT CONCAT(ho, ten) FROM khach_hang;'
    res_bad_concat = SQLValidator.validate(sql_bad_concat)
    assert not res_bad_concat.is_valid
    assert any("string concatenation" in issue.lower() for issue in res_bad_concat.issues)

    # Individual columns in SELECT -> Passes
    sql_good = 'SELECT ho, ten FROM khach_hang;'
    res_good = SQLValidator.validate(sql_good)
    assert res_good.is_valid
