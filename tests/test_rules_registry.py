"""Unit tests for validation rules registry, issue models, and diagnostic rendering."""

from __future__ import annotations

from zaic_2026_4.submission.models import ValidationResult
from zaic_2026_4.submission.rules import (
    RULE_REGISTRY,
    ValidationIssue,
    render_diagnostics,
)

EXPECTED_CODES = {
    "OUTPUT_EMPTY",
    "OUTPUT_UNPARSEABLE",
    "OUTPUT_TRUNCATED",
    "DATE_RANGE_OUT_OF_PERIOD",
    "CURRENCY_SEPARATOR_MISMATCH",
    "LITERAL_PARTIAL_MATCH",
    "TYPE_INVALID",
    "SQL_SYNTAX",
    "SQL_NOT_READ_ONLY",
    "SQL_NOT_SELECT",
    "SQL_MULTIPLE_STATEMENTS",
    "TABLE_NOT_FOUND",
    "COLUMN_NOT_FOUND",
    "UNKNOWN_QUALIFIER",
    "DATE_FORMAT_MISMATCH",
    "LITERAL_NOT_IN_ENUM",
    "SHAPE_MISMATCH",
    "SHAPE_SUSPECT_NO_AGGREGATE",
    "RANK_TIEBREAK_MISSING",
    "MONEY_SCALE_RISK",
    "ORDER_ASC_NULLABLE",
    "MIN_NULLABLE",
    "CONCAT_IN_SELECT",
    "RESULT_SHAPE_MISMATCH",
    "UNRESOLVED_ZERO",
    "STAGNATION",
    "DELIBERATE_ABSTAIN",
    "EXECUTION_ERROR",
}


def test_rule_registry_contains_all_canonical_codes() -> None:
    """Ensure every canonical error code is documented in RULE_REGISTRY."""
    assert len(RULE_REGISTRY) == len(EXPECTED_CODES)
    assert set(RULE_REGISTRY.keys()) == EXPECTED_CODES
    for code, hint in RULE_REGISTRY.items():
        assert len(hint.strip()) > 10, f"Hint for {code} is suspiciously short"


def test_validation_issue_rendering_and_compatibility() -> None:
    """Test ValidationIssue string conversion, hint defaulting, and backward-compatible methods."""
    # Default hint lookup from registry
    issue = ValidationIssue(code="SQL_SYNTAX", message="Cú pháp sai gần 'FROM'")
    line = issue.to_prompt_line()
    assert line.startswith("-[SQL_SYNTAX] Cú pháp sai gần 'FROM'")
    assert "Lỗi cú pháp SQLite" in line
    assert str(issue) == line
    assert "cú pháp sai" in issue.lower()
    assert "SQL_SYNTAX" in issue

    # Custom hint override
    custom = ValidationIssue(
        code="COLUMN_NOT_FOUND",
        message="Không tìm thấy 'cột_lạ'",
        severity="blocking",
        column="cột_lạ",
        hint="Dùng cột 'cột_chuẩn' thay thế",
    )
    assert "Dùng cột 'cột_chuẩn' thay thế" in custom.to_prompt_line()


def test_render_diagnostics_handles_mixed_types() -> None:
    """Ensure render_diagnostics cleanly formats both ValidationIssue instances and raw strings."""
    issues = [
        ValidationIssue(code="TABLE_NOT_FOUND", message="Bảng 'users' không tồn tại"),
        "Raw error message from external check",
        "- Already prefixed message",
    ]
    rendered = render_diagnostics(issues)
    assert len(rendered) == 3
    assert rendered[0].startswith("-[TABLE_NOT_FOUND]")
    assert rendered[1] == "- Raw error message from external check"
    assert rendered[2] == "- Already prefixed message"


def test_validation_result_methods() -> None:
    """Verify ValidationResult helper methods with ValidationIssue items."""
    b_issue = ValidationIssue(code="SQL_SYNTAX", message="Syntax err", severity="blocking")
    w_issue = ValidationIssue(code="ORDER_ASC_NULLABLE", message="Nullable ASC", severity="warning")
    res = ValidationResult(is_valid=False, issues=[b_issue, w_issue])

    assert res.codes() == {"SQL_SYNTAX", "ORDER_ASC_NULLABLE"}
    assert len(res.blocking_issues()) == 1
    assert res.blocking_issues()[0] == b_issue
    assert len(res.warning_issues()) == 1
    assert res.warning_issues()[0] == w_issue
    assert len(res.issue_texts) == 2
    assert "SQL_SYNTAX" in res.issue_texts[0]
