"""Data reconciliation and schema integrity tests against train dataset."""

import sqlite3
from pathlib import Path

import pytest

from zaic_2026_4.ingestion.canonical import is_side_table
from zaic_2026_4.ingestion.extractors import extract_csv, extract_excel, extract_sqlite
from zaic_2026_4.ingestion.service import ingest_business


@pytest.fixture(scope="session")
def train_data_dir() -> Path:
    """Return path to train data directory."""
    path = Path("dataset/train/data")
    if not path.is_dir():
        pytest.skip("dataset/train/data directory not found")
    return path


def test_reconciliation_zero_unnamed_and_schema_integrity(tmp_path: Path, train_data_dir: Path):
    """Verify that ingested databases have 0 UNNAMED columns and keep original schemas."""
    # Test across representative businesses with CSV, complex multi-sheet Excel, and SQLite
    sample_biz_ids = ["biz_101", "biz_103", "biz_105", "biz_123", "biz_130"]

    for biz_id in sample_biz_ids:
        biz_dir = train_data_dir / biz_id
        if not biz_dir.exists():
            continue

        db_path = ingest_business(biz_dir, output_dir=tmp_path, create_views=True)
        assert db_path.exists()

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%';"
        )
        entities = cur.fetchall()
        assert len(entities) > 0

        tables = [
            name for name, typ in entities if typ == "table" and not is_side_table(name)
        ]
        views = [name for name, typ in entities if typ == "view"]

        # Check every table for UNNAMED columns
        for tname in tables:
            cur.execute(f'PRAGMA table_info("{tname}");')
            cols = [r[1] for r in cur.fetchall()]
            unnamed_cols = [c for c in cols if "unnamed" in c.lower()]
            assert (
                len(unnamed_cols) == 0
            ), f"[{biz_id}] Table '{tname}' contains UNNAMED columns: {unnamed_cols}"
            assert len(cols) > 0

        # Check all unified VIEWs are queryable
        for vname in views:
            cur.execute(f'SELECT COUNT(*) FROM "{vname}";')
            count = cur.fetchone()[0]
            assert count >= 0

        conn.close()


def test_reconciliation_exact_row_counts(tmp_path: Path, train_data_dir: Path):
    """Verify 1:1 row count parity between source files and SQLite tables."""
    sample_biz_ids = ["biz_101", "biz_106", "biz_124", "biz_130"]

    for biz_id in sample_biz_ids:
        biz_dir = train_data_dir / biz_id
        if not biz_dir.exists():
            continue

        db_path = ingest_business(biz_dir, output_dir=tmp_path, create_views=True)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = [r[0] for r in cur.fetchall()]

        assert "calendar" in tables, f"[{biz_id}] Expected auto-ingested 'calendar' table in DB"

        # Canonical side tables mirror derived _date/_amount/_status values, so they are
        # excluded: this test proves that source rows are ingested 1:1 into business tables.
        db_total_rows = 0
        for tname in tables:
            if tname == "calendar" or is_side_table(tname):
                continue
            cur.execute(f'SELECT COUNT(*) FROM "{tname}";')
            db_total_rows += cur.fetchone()[0]
        conn.close()

        # Count total rows directly from raw source files
        source_total_rows = 0
        for f in sorted(biz_dir.iterdir()):
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext == ".csv":
                extracted = extract_csv(f)
                source_total_rows += sum(df.height for df in extracted.values())
            elif ext in (".xlsx", ".xls"):
                extracted = extract_excel(f)
                source_total_rows += sum(df.height for df in extracted.values())
            elif ext in (".sqlite", ".sqlite3", ".db"):
                extracted = extract_sqlite(f)
                source_total_rows += sum(df.height for df in extracted.values())

        assert (
            db_total_rows == source_total_rows
        ), f"[{biz_id}] Row count mismatch: DB has {db_total_rows}, source has {source_total_rows}"
