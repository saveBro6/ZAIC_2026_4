"""High-performance SQLite persistence with strict quoting and schema retention."""

import sqlite3
from pathlib import Path

import polars as pl

from zaic_2026_4.ingestion.canonical import (
    build_canonical_sidecar,
    generate_canonical_unified_views,
    generate_canonical_views,
    plan_canonical_tables,
    side_table_name,
)
from zaic_2026_4.ingestion.view_builder import (
    generate_unified_views,
    group_tables_by_schema,
)


def polars_type_to_sqlite(dtype: pl.DataType) -> str:
    """Map Polars data type to SQLite column affinity."""
    if dtype.is_integer():
        return "INTEGER"
    if dtype.is_float():
        return "REAL"
    if dtype == pl.Boolean:
        return "INTEGER"
    return "TEXT"


def _insert_frame(cur: sqlite3.Cursor, tbl_name: str, df: pl.DataFrame) -> None:
    """Insert a DataFrame into an already created table."""
    if df.height == 0:
        return
    placeholders = ", ".join(["?"] * len(df.columns))
    cur.executemany(f'INSERT INTO "{tbl_name}" VALUES ({placeholders})', df.iter_rows())


def write_tables_to_sqlite(
    tables: dict[str, pl.DataFrame],
    output_db_path: Path | str,
    create_views: bool = True,
    create_canonical_views: bool = True,
    canonical_reports: dict[str, dict] | None = None,
) -> Path:
    """Write DataFrames to SQLite and create unified and canonical VIEWs.

    Args:
        tables: Mapping of table name to DataFrame.
        output_db_path: Destination SQLite file (overwritten when it exists).
        create_views: Whether to create unified VIEWs for identical schemas.
        create_canonical_views: Whether to add derived ``_date``/``_amount``/``_status``
            columns through ``<table>__c`` views backed by ``_canon_<table>`` side tables.
        canonical_reports: Optional sink filled with the detected normalization specs.

    Returns:
        Path to the created database file.
    """
    output_path = Path(output_db_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    conn = sqlite3.connect(output_path)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA synchronous = OFF;")
        cur.execute("PRAGMA journal_mode = MEMORY;")
        cur.execute("PRAGMA cache_size = 100000;")

        for tbl_name, df in tables.items():
            if df.is_empty() and len(df.columns) == 0:
                continue

            col_defs = [
                f'"{c}" {polars_type_to_sqlite(dt)}'
                for c, dt in zip(df.columns, df.dtypes, strict=False)
            ]
            cur.execute(f'CREATE TABLE "{tbl_name}" ({", ".join(col_defs)});')
            _insert_frame(cur, tbl_name, df)

        existing_names = {
            row[0] for row in cur.execute("SELECT name FROM sqlite_master")
        }
        skipped: list[str] = []

        def _create_view(name: str, sql: str) -> None:
            """Create a view unless its name is already taken by a real entity.

            A business may legitimately ship an object already named ``X__c`` (custom
            object naming). Overwriting it would fail and lose the whole business, so the
            derived view is skipped and recorded in the ingestion report instead.
            """
            if name in existing_names:
                skipped.append(name)
                return
            cur.execute(sql)
            existing_names.add(name)

        unified_view_groups = group_tables_by_schema(tables) if create_views else []
        unified_view_names: dict[str, str] = {}
        if create_views:
            for view_name, view_sql in generate_unified_views(tables):
                _create_view(view_name, view_sql)
                for group in unified_view_groups:
                    members = tuple(group)
                    if all(f'"{member}"' in view_sql for member in members):
                        for member in members:
                            unified_view_names.setdefault(member, view_name)
                        break

        if create_canonical_views:
            specs, _ = plan_canonical_tables(tables)
            # A table exposing its own "rowid" column would break the side-table join.
            specs = {
                name: spec
                for name, spec in specs.items()
                if not any(str(col).lower() == "rowid" for col in tables[name].columns)
            }
            detections: dict[str, dict] = {}
            created_specs: dict[str, dict[str, str]] = {}
            for base_table, spec in list(specs.items()):
                if side_table_name(base_table) in existing_names:
                    skipped.append(side_table_name(base_table))
                    continue
                sidecar, detail = build_canonical_sidecar(tables[base_table], spec)
                if sidecar.height == 0 or sidecar.width <= 1:
                    continue
                detail["source_table"] = base_table
                detections[base_table] = detail
                side_name = side_table_name(base_table)
                side_defs = [
                    f'"{c}" {polars_type_to_sqlite(dt)}'
                    for c, dt in zip(sidecar.columns, sidecar.dtypes, strict=False)
                ]
                cur.execute(f'CREATE TABLE "{side_name}" ({", ".join(side_defs)});')
                _insert_frame(cur, side_name, sidecar)
                existing_names.add(side_name)
                created_specs[base_table] = spec

            specs = created_specs
            for view_name, view_sql in generate_canonical_views(tables, specs):
                _create_view(view_name, view_sql)
            if unified_view_groups:
                for view_name, view_sql in generate_canonical_unified_views(
                    unified_view_groups, specs, unified_view_names, tables=tables
                ):
                    _create_view(view_name, view_sql)

            if canonical_reports is not None:
                canonical_reports["specs"] = dict(specs)
                canonical_reports["detections"] = detections
                canonical_reports["skipped_collisions"] = sorted(skipped)

        conn.commit()
    finally:
        conn.close()

    return output_path


__all__ = ["polars_type_to_sqlite", "write_tables_to_sqlite"]
