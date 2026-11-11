"""Tests for tabular ingestion module."""

import sqlite3
from pathlib import Path

import polars as pl
import pytest

from zaic_2026_4.ingestion.service import ingest_business
from zaic_2026_4.ingestion.sqlite_sink import write_tables_to_sqlite
from zaic_2026_4.ingestion.view_builder import (
    generate_unified_views,
    token_common_prefix,
)


def test_token_common_prefix():
    """Verify delimiter-based token common prefix extraction without pattern regex."""
    assert (
        token_common_prefix(["So_chi_tiet_ban_hang_1", "So_chi_tiet_ban_hang_2"])
        == "So_chi_tiet_ban_hang"
    )
    assert (
        token_common_prefix(["BanHang_2025", "BanHang_2026"])
        == "BanHang"
    )
    assert (
        token_common_prefix(["DonHang_ChiTiet_01", "DonHang_ChiTiet_02"])
        == "DonHang_ChiTiet"
    )
    assert (
        token_common_prefix(["SingleTable"])
        == "SingleTable"
    )
    assert token_common_prefix([]) == ""


def test_generate_unified_views():
    """Verify VIEW SQL generation for tables sharing identical columns."""
    df1 = pl.DataFrame({"Mã đơn": [1, 2], "Ngày": ["2025-01-01", "2025-01-02"]})
    df2 = pl.DataFrame({"Mã đơn": [3], "Ngày": ["2025-02-01"]})
    df_other = pl.DataFrame({"Khác": [10]})

    tables = {
        "BanHang_1": df1,
        "BanHang_2": df2,
        "DanhMuc": df_other,
    }

    views = generate_unified_views(tables)
    assert len(views) == 1
    view_name, view_sql = views[0]
    assert view_name == "BanHang"
    assert "CREATE VIEW IF NOT EXISTS \"BanHang\" AS" in view_sql
    assert "UNION ALL" in view_sql
    assert "BanHang_1" in view_sql
    assert "BanHang_2" in view_sql


def test_sqlite_sink_and_view(tmp_path: Path):
    """Test writing tables and querying generated VIEW in SQLite."""
    df1 = pl.DataFrame({"Mã SP": ["SP01", "SP02"], "Giá bán": [10000, 20000]})
    df2 = pl.DataFrame({"Mã SP": ["SP03"], "Giá bán": [30000]})

    tables = {
        "ChiTiet_2025": df1,
        "ChiTiet_2026": df2,
    }

    db_path = tmp_path / "test.db"
    write_tables_to_sqlite(tables, db_path, create_views=True)

    assert db_path.exists()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view');")
    entities = dict(cur.fetchall())
    assert "ChiTiet_2025" in entities
    assert "ChiTiet_2026" in entities
    assert "ChiTiet" in entities
    assert entities["ChiTiet"] == "view"

    cur.execute('SELECT COUNT(*) FROM "ChiTiet";')
    assert cur.fetchone()[0] == 3

    conn.close()


def test_ingest_train_business_biz_101(tmp_path: Path):
    """Verify end-to-end ingestion on a representative train business (biz_101 - CSV)."""
    biz_dir = Path("dataset/train/data/biz_101")
    if not biz_dir.exists():
        pytest.skip("biz_101 does not exist in dataset/train/data")

    out_db = ingest_business(biz_dir, output_dir=tmp_path, create_views=True)
    assert out_db.exists()

    conn = sqlite3.connect(out_db)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    assert "ban_hang" in tables

    cur.execute('PRAGMA table_info("ban_hang");')
    cols = [r[1] for r in cur.fetchall()]
    assert "Giá bán" in cols
    assert not any("unnamed" in c.lower() for c in cols)

    conn.close()


def test_ingest_train_business_biz_130_excel_header_and_views(tmp_path: Path):
    """Verify Excel header detection (avoiding UNNAMED) and VIEW generation for biz_130."""
    biz_dir = Path("dataset/train/data/biz_130")
    if not biz_dir.exists():
        pytest.skip("biz_130 does not exist in dataset/train/data")

    out_db = ingest_business(biz_dir, output_dir=tmp_path, create_views=True)
    assert out_db.exists()

    conn = sqlite3.connect(out_db)
    cur = conn.cursor()

    # Verify tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    assert "So_chi_tiet_ban_hang_1" in tables
    assert "So_chi_tiet_ban_hang_2" in tables

    # Verify zero UNNAMED columns in both tables
    for tbl in ["So_chi_tiet_ban_hang_1", "So_chi_tiet_ban_hang_2"]:
        cur.execute(f'PRAGMA table_info("{tbl}");')
        cols = [r[1] for r in cur.fetchall()]
        assert "Mã đơn hàng" in cols
        assert "Ngày" in cols
        assert not any("unnamed" in c.lower() for c in cols)

    # Verify unified VIEW creation
    cur.execute("SELECT name FROM sqlite_master WHERE type='view';")
    views = [r[0] for r in cur.fetchall()]
    assert "So_chi_tiet_ban_hang" in views

    # Verify query works on the VIEW
    cur.execute('SELECT COUNT(*) FROM "So_chi_tiet_ban_hang";')
    count = cur.fetchone()[0]
    assert count > 0

    conn.close()


def test_ingest_train_business_biz_123_sqlite(tmp_path: Path):
    """Verify ingestion of existing SQLite database (biz_123 pos.sqlite)."""
    biz_dir = Path("dataset/train/data/biz_123")
    if not biz_dir.exists():
        pytest.skip("biz_123 does not exist in dataset/train/data")

    out_db = ingest_business(biz_dir, output_dir=tmp_path, create_views=True)
    assert out_db.exists()

    conn = sqlite3.connect(out_db)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    assert "don_hang" in tables
    assert "chi_tiet_don" in tables
    assert "san_pham" in tables

    cur.execute('SELECT COUNT(*) FROM "don_hang";')
    assert cur.fetchone()[0] > 0

    conn.close()


def test_calendar_auto_discovery(tmp_path: Path):
    """Verify auto-discovery of calendar.csv without explicit calendar_path."""
    biz_dir = Path("dataset/train/data/biz_101")
    if not biz_dir.exists():
        pytest.skip("biz_101 does not exist in dataset/train/data")

    out_db = ingest_business(biz_dir, output_dir=tmp_path, create_views=True)
    assert out_db.exists()

    conn = sqlite3.connect(out_db)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    assert "calendar" in tables

    cur.execute('SELECT COUNT(*) FROM "calendar";')
    assert cur.fetchone()[0] > 0

    conn.close()
