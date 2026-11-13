"""Ingestion package: extracts tabular files into SQLite databases with unified VIEWs."""

from zaic_2026_4.ingestion.canonical import (
    AMOUNT_COLUMN,
    CANONICAL_COLUMNS,
    CANONICAL_SUFFIX,
    DATE_COLUMN,
    STATUS_COLUMN,
    canonical_base_of,
    canonical_view_name,
    describe_canonical_column,
    is_canonical_view,
    is_side_table,
)
from zaic_2026_4.ingestion.extractors import (
    discover_and_extract_business,
    extract_csv,
    extract_excel,
    extract_sqlite,
)
from zaic_2026_4.ingestion.forward_fill import forward_fill_grouped_rows
from zaic_2026_4.ingestion.header_detector import detect_header_row
from zaic_2026_4.ingestion.service import (
    DEFAULT_BASE_OUTPUT_DIR,
    ingest_business,
    ingest_split,
    resolve_split_paths,
)
from zaic_2026_4.ingestion.sqlite_sink import (
    polars_type_to_sqlite,
    write_tables_to_sqlite,
)
from zaic_2026_4.ingestion.view_builder import (
    generate_unified_views,
    token_common_prefix,
)

__all__ = [
    "AMOUNT_COLUMN",
    "CANONICAL_COLUMNS",
    "CANONICAL_SUFFIX",
    "DATE_COLUMN",
    "DEFAULT_BASE_OUTPUT_DIR",
    "STATUS_COLUMN",
    "canonical_base_of",
    "canonical_view_name",
    "describe_canonical_column",
    "detect_header_row",
    "discover_and_extract_business",
    "extract_csv",
    "extract_excel",
    "extract_sqlite",
    "forward_fill_grouped_rows",
    "generate_unified_views",
    "ingest_business",
    "ingest_split",
    "is_canonical_view",
    "is_side_table",
    "polars_type_to_sqlite",
    "resolve_split_paths",
    "token_common_prefix",
    "write_tables_to_sqlite",
]
