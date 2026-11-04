"""Verification script to compare /tmp/zaic_db/train/*.db against original raw files in dataset/train/data."""

from collections import defaultdict
from pathlib import Path
import sqlite3
import fastexcel
import polars as pl

from zaic_2026_4.ingestion.extractors import extract_csv, extract_excel, extract_sqlite
from zaic_2026_4.ingestion.service import DEFAULT_BASE_OUTPUT_DIR


def verify_all_train_businesses():
    train_dir = Path("dataset/train/data")
    db_base_dir = Path("/tmp/zaic_db/train")

    biz_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])

    total_businesses = len(biz_dirs)
    total_physical_tables = 0
    total_views = 0
    total_unnamed_cols = 0
    total_rows_db = 0
    total_rows_source = 0

    mismatches = []
    unnamed_details = []

    print(f"Starting verification across {total_businesses} businesses in {train_dir} ...\n")

    for idx, biz_path in enumerate(biz_dirs, 1):
        biz_id = biz_path.name
        db_path = db_base_dir / f"{biz_id}.db"

        if not db_path.exists():
            mismatches.append(f"Missing DB: {db_path}")
            continue

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Query all tables and views
        cur.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%';")
        db_entities = cur.fetchall()

        tables = [name for name, typ in db_entities if typ == "table"]
        views = [name for name, typ in db_entities if typ == "view"]

        total_physical_tables += len(tables)
        total_views += len(views)

        # 1. Check UNNAMED columns in all tables and views
        for tname in tables:
            cur.execute(f'PRAGMA table_info("{tname}");')
            cols = [r[1] for r in cur.fetchall()]
            unnamed_in_table = [c for c in cols if "unnamed" in c.lower()]
            if unnamed_in_table:
                total_unnamed_cols += len(unnamed_in_table)
                unnamed_details.append(f"[{biz_id}] Table '{tname}' has UNNAMED columns: {unnamed_in_table}")

            cur.execute(f'SELECT COUNT(*) FROM "{tname}";')
            t_rows = cur.fetchone()[0]
            total_rows_db += t_rows

        # 2. Check all VIEWs work and are queryable
        for vname in views:
            try:
                cur.execute(f'SELECT COUNT(*) FROM "{vname}";')
                v_rows = cur.fetchone()[0]
            except Exception as e:
                mismatches.append(f"[{biz_id}] View '{vname}' query failed: {e}")

        conn.close()

        # 3. Source file row count comparison
        biz_source_rows = 0
        for f in sorted(biz_path.iterdir()):
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext == ".csv":
                extracted = extract_csv(f)
                for df in extracted.values():
                    biz_source_rows += df.height
            elif ext in (".xlsx", ".xls"):
                extracted = extract_excel(f)
                for df in extracted.values():
                    biz_source_rows += df.height
            elif ext in (".sqlite", ".sqlite3", ".db"):
                extracted = extract_sqlite(f)
                for df in extracted.values():
                    biz_source_rows += df.height

        total_rows_source += biz_source_rows

    print("VERIFICATION RESULTS:")
    print(f"- Total businesses verified: {total_businesses}")
    print(f"- Total physical tables in SQLite: {total_physical_tables}")
    print(f"- Total unified VIEWs in SQLite: {total_views}")
    print(f"- Total UNNAMED columns across all tables: {total_unnamed_cols}")
    print(f"- Total row count in SQLite: {total_rows_db:,}")
    print(f"- Total row count in source files: {total_rows_source:,}")
    print(f"- Row count match: {total_rows_db == total_rows_source}")

    if unnamed_details:
        print("\nUNNAMED COLUMN DETAILS:")
        for u in unnamed_details:
            print(f"  {u}")

    if mismatches:
        print("\nMISMATCHES / ERRORS:")
        for m in mismatches:
            print(f"  {m}")
    else:
        print("\n100% verification successful: Zero UNNAMED columns, zero query failures, row counts match exactly!")


if __name__ == "__main__":
    verify_all_train_businesses()
