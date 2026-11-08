"""Unit tests for ResultFormatter and contract type conformity."""

from __future__ import annotations

from zaic_2026_4.submission.result_formatter import (
    ResultFormatter,
    infer_question_target_type,
    normalize_contract_string,
)


def test_normalize_contract_string() -> None:
    raw = "Sting dâu (330ml) - Chai!"
    normalized = normalize_contract_string(raw)
    assert normalized == "sting dâu 330ml chai"

    raw_accents = "Cà phê Sữa   ĐÁ"
    assert normalize_contract_string(raw_accents) == "cà phê sữa đá"


def test_infer_question_target_type() -> None:
    assert infer_question_target_type("Tỉ lệ huỷ đơn tháng này là bao nhiêu %?") == "percent"
    assert infer_question_target_type("Ngày nào có doanh thu cao nhất?") == "date"
    assert infer_question_target_type("Có phải khách hàng đã thanh toán?") == "boolean"
    assert infer_question_target_type("Top 3 sản phẩm bán chạy nhất?") == "list"
    assert infer_question_target_type("Sản phẩm nào bán chạy nhất?") == "string"
    assert infer_question_target_type("Tổng doanh thu tháng này là bao nhiêu?") == "number"


def test_format_number() -> None:
    assert ResultFormatter.format_number(48250000) == 48250000
    assert ResultFormatter.format_number(48250000.0) == 48250000
    assert ResultFormatter.format_number(12.3456) == 12.35
    assert ResultFormatter.format_number(None) == 0


def test_format_percent() -> None:
    assert ResultFormatter.format_percent(5.27) == 5.27
    assert ResultFormatter.format_percent(-12.45) == -12.45
    # Ratio converted to percent
    assert ResultFormatter.format_percent(0.0527) == 5.27
    assert ResultFormatter.format_percent(0.0) == 0.0


def test_format_date() -> None:
    assert ResultFormatter.format_date("2026-08-15") == "2026-08-15"
    assert ResultFormatter.format_date("2025-09") == "2025-09"
    # Convert DD/MM/YYYY
    assert ResultFormatter.format_date("15/08/2026") == "2026-08-15"


def test_format_result_scalar_cases() -> None:
    # Number
    t, val = ResultFormatter.format_result("Tổng tiền?", [(48250000,)])
    assert t == "number"
    assert val == 48250000

    # Percent
    t, val = ResultFormatter.format_result("Tỉ lệ đơn hủy là bao nhiêu %?", [(2.08,)])
    assert t == "percent"
    assert val == 2.08

    # String
    t, val = ResultFormatter.format_result("Mặt hàng nào bán chạy nhất?", [("Sting dâu 330ml",)])
    assert t == "string"
    assert val == "Sting dâu 330ml"

    # Date
    t, val = ResultFormatter.format_result("Ngày nào bán được nhiều nhất?", [("2026-08-15",)])
    assert t == "date"
    assert val == "2026-08-15"

    # Boolean
    t, val = ResultFormatter.format_result("Có phải doanh thu tăng không?", [(1,)])
    assert t == "boolean"
    assert val is True


def test_format_result_list_and_table() -> None:
    # List (single column, multiple rows)
    rows_list = [("Cà phê sữa",), ("Bạc xỉu",), ("Trà đào",)]
    t, val = ResultFormatter.format_result("Top 3 mặt hàng?", rows_list)
    assert t == "list"
    assert val == ["Cà phê sữa", "Bạc xỉu", "Trà đào"]

    # Table (multiple columns, multiple rows)
    rows_table = [("Chị Lan", 3200000), ("Anh Tú", 1500000)]
    t, val = ResultFormatter.format_result("Bảng công nợ?", rows_table)
    assert t == "table"
    assert isinstance(val, dict)
    assert val["rows"] == [["Chị Lan", 3200000], ["Anh Tú", 1500000]]


def test_format_result_edge_cases_zero_and_none() -> None:
    # [(0,)] on percent question -> percent, 0.0
    t_pct, val_pct = ResultFormatter.format_result("Tỉ lệ hủy là %?", [(0,)])
    assert t_pct == "percent"
    assert val_pct == 0.0

    # [(0,)] on count question -> number, 0
    t_num, val_num = ResultFormatter.format_result("Tổng số đơn?", [(0,)])
    assert t_num == "number"
    assert val_num == 0

    # [(None,)] on aggregate question -> number, 0
    t_none_agg, val_none_agg = ResultFormatter.format_result("Tổng tiền?", [(None,)])
    assert t_none_agg == "number"
    assert val_none_agg == 0

    # [(None,)] on entity search question -> abstain
    t_none_ent, val_none_ent = ResultFormatter.format_result("Sản phẩm nào?", [(None,)])
    assert t_none_ent == "abstain"
    assert val_none_ent is None

    # Empty rows on aggregate question -> 0
    t_empty_agg, val_empty_agg = ResultFormatter.format_result("Tổng tiền?", [])
    assert t_empty_agg == "number"
    assert val_empty_agg == 0

    # Empty rows on entity question -> abstain
    t_empty_ent, val_empty_ent = ResultFormatter.format_result("Sản phẩm nào?", [])
    assert t_empty_ent == "abstain"
    assert val_empty_ent is None
