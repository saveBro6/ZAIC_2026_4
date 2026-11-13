"""Unit tests for schema coverage closure over profiled join edges."""

from __future__ import annotations

from zaic_2026_4.common.query_intent import QueryIntent
from zaic_2026_4.schema_linking.coverage import validate_schema_coverage


def _meta(columns: dict[tuple[str, str], dict]) -> dict:
    """Build the flat 'Table.column' -> profile mapping used by the coverage pass."""
    return {f"{table}.{column}": profile for (table, column), profile in columns.items()}


def test_bridge_table_is_added_to_close_the_join() -> None:
    """A pair of referenced tables connected only through a middle table gets that table."""
    metadata = _meta(
        {
            ("HoaDon", "Mã hóa đơn"): {"data_format": "FIXED_CODE", "sample_values": ["HD1"]},
            ("HoaDon", "Thời gian"): {"data_format": "SLASH_DATE", "sample_values": ["01/03/2025"]},
            ("ChiTiet", "Mã hóa đơn"): {
                "data_format": "FIXED_CODE",
                "sample_values": ["HD1"],
                "join_candidates": [{"target": "HoaDon.Mã hóa đơn", "resemblance": 0.9}],
            },
            ("ChiTiet", "Mã SKU"): {
                "data_format": "FIXED_CODE",
                "sample_values": ["SP1"],
                "join_candidates": [{"target": "Thuoc.Mã SKU", "resemblance": 0.9}],
            },
            ("Thuoc", "Mã SKU"): {"data_format": "FIXED_CODE", "sample_values": ["SP1"]},
            ("Thuoc", "Kê đơn"): {"data_format": "INTEGER", "sample_values": ["1"]},
        }
    )
    schema = {"HoaDon.Thời gian", "Thuoc.Kê đơn"}

    expanded, report = validate_schema_coverage(
        schema, metadata, QueryIntent(has_temporal_cue=True)
    )

    assert "ChiTiet" in report.added_tables
    assert "join_key_closure" in report.satisfied_aspects
    assert "join_key_closure" not in report.missing_aspects
    assert not report.has_unresolved_gaps
    assert "ChiTiet.Thời gian" not in expanded  # only join keys are pulled in
    assert "ChiTiet.Mã hóa đơn" in expanded or "ChiTiet.Mã SKU" in expanded


def test_unreachable_tables_report_a_gap_instead_of_proving_unanswerable() -> None:
    """No profiled path means the coverage gap must be reported, not hidden."""
    metadata = _meta(
        {
            ("HoaDon", "Thời gian"): {"data_format": "SLASH_DATE", "sample_values": ["01/03/2025"]},
            ("Kho", "Tên kho"): {"data_format": "STRING", "sample_values": ["Kho A"]},
        }
    )
    schema = {"HoaDon.Thời gian", "Kho.Tên kho"}

    _expanded, report = validate_schema_coverage(
        schema, metadata, QueryIntent(has_temporal_cue=True)
    )

    assert "join_key_closure" in report.missing_aspects
    assert report.has_unresolved_gaps


def test_canonical_projection_is_added_for_referenced_tables() -> None:
    """The normalized view columns of a referenced table are offered to the generator."""
    metadata = _meta(
        {
            ("HoaDon", "Thời gian"): {"data_format": "SLASH_DATE", "sample_values": ["01/03/2025"]},
            ("HoaDon__c", "Thời gian"): {"data_format": "SLASH_DATE", "sample_values": ["01/03/2025"]},
            ("HoaDon__c", "_date"): {"data_format": "ISO_DATE", "sample_values": ["2025-03-01"]},
            ("HoaDon__c", "_amount"): {"data_format": "INTEGER", "sample_values": ["1000"]},
            ("HoaDon__c", "_status"): {"data_format": "STRING", "sample_values": ["completed"]},
        }
    )
    schema = {"HoaDon.Thời gian"}

    expanded, report = validate_schema_coverage(
        schema, metadata, QueryIntent(has_temporal_cue=True)
    )

    assert {"HoaDon__c._date", "HoaDon__c._amount", "HoaDon__c._status"} <= expanded
    assert "canonical_projection" in report.satisfied_aspects
    assert not report.has_unresolved_gaps
