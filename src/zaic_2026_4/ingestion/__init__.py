"""Ingestion package: extracts tabular files into SQLite databases with unified VIEWs."""

from zaic_2026_4.ingestion.extractors import (
    discover_and_extract_business,
    extract_csv,
    extract_excel,
    extract_sqlite,
)
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
    "DEFAULT_BASE_OUTPUT_DIR",
    "detect_header_row",
    "discover_and_extract_business",
    "extract_csv",
    "extract_excel",
    "extract_sqlite",
    "generate_unified_views",
    "ingest_business",
    "ingest_split",
    "polars_type_to_sqlite",
    "resolve_split_paths",
    "token_common_prefix",
    "write_tables_to_sqlite",
]
