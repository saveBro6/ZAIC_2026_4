"""High-performance SQLite persistence with strict quoting and schema retention."""

from pathlib import Path
import sqlite3
import polars as pl
from zaic_2026_4.ingestion.view_builder import generate_unified_views


def polars_type_to_sqlite(dtype: pl.DataType) -> str:
    """Map Polars data type to SQLite column affinity."""
    if dtype.is_integer():
        return "INTEGER"
    if dtype.is_float():
        return "REAL"
    if dtype == pl.Boolean:
        return "INTEGER"
    return "TEXT"


def write_tables_to_sqlite(
    tables: dict[str, pl.DataFrame],
    output_db_path: Path | str,
    create_views: bool = True,
) -> Path:
    """Write DataFrames to SQLite and create unified VIEWs for identical schemas."""
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
            create_sql = f'CREATE TABLE "{tbl_name}" ({", ".join(col_defs)});'
            cur.execute(create_sql)

            if df.height > 0:
                placeholders = ", ".join(["?"] * len(df.columns))
                insert_sql = f'INSERT INTO "{tbl_name}" VALUES ({placeholders})'
                cur.executemany(insert_sql, df.iter_rows())

        if create_views:
            views = generate_unified_views(tables)
            for _, view_sql in views:
                cur.execute(view_sql)

        conn.commit()
    finally:
        conn.close()

    return output_path
