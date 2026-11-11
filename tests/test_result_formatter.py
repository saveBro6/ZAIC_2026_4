"""Unit tests for ResultFormatter and contract type conformity."""

from __future__ import annotations

from zaic_2026_4.submission.result_formatter import (
    ResultFormatter,
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
    t, val = ResultFormatter.format_result([(48250000,)], target_type="number")
    assert t == "number"
    assert val == 48250000

    # Percent
    t, val = ResultFormatter.format_result([(2.08,)], target_type="percent")
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


def test_format_result_list_and_table() -> None:
    # List (single column, multiple rows)
    rows_list = [("Cà phê sữa",), ("Bạc xỉu",), ("Trà đào",)]
    t, val = ResultFormatter.format_result(rows_list, target_type="list")
    assert t == "list"
    assert val == ["Cà phê sữa", "Bạc xỉu", "Trà đào"]

    # Table (multiple columns, multiple rows)
    rows_table = [("Chị Lan", 3200000), ("Anh Tú", 1500000)]
    t, val = ResultFormatter.format_result(rows_table, target_type="table")
    assert t == "table"
    assert isinstance(val, dict)
    assert val["rows"] == [["Chị Lan", 3200000], ["Anh Tú", 1500000]]


def test_format_result_unanswerable_and_abstain() -> None:
    t_unans, val_unans = ResultFormatter.format_result(None, target_type="unanswerable")
    assert t_unans == "unanswerable"
    assert val_unans is None

    t_abs, val_abs = ResultFormatter.format_result(None, target_type="abstain")
    assert t_abs == "abstain"
    assert val_abs is None


def test_format_result_edge_cases_zero_and_none() -> None:
    # [(0,)] on percent -> percent, 0.0
    t_pct, val_pct = ResultFormatter.format_result([(0,)], target_type="percent")
    assert t_pct == "percent"
    assert val_pct == 0.0

    # [(0,)] on number -> number, 0
    t_num, val_num = ResultFormatter.format_result([(0,)], target_type="number")
    assert t_num == "number"
    assert val_num == 0

    # [(None,)] on number -> number, 0
    t_none_agg, val_none_agg = ResultFormatter.format_result([(None,)], target_type="number")
    assert t_none_agg == "number"
    assert val_none_agg == 0

    # [(None,)] on entity search question -> abstain
    t_none_ent, val_none_ent = ResultFormatter.format_result([(None,)], target_type="string")
    assert t_none_ent == "abstain"
    assert val_none_ent is None

    # Empty rows on aggregate question -> 0
    t_empty_agg, val_empty_agg = ResultFormatter.format_result([], target_type="number")
    assert t_empty_agg == "number"
    assert val_empty_agg == 0

    # Empty rows on entity question -> abstain
    t_empty_ent, val_empty_ent = ResultFormatter.format_result([], target_type="string")
    assert t_empty_ent == "abstain"
    assert val_empty_ent is None
