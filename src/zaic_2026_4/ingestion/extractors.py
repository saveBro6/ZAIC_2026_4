"""Raw tabular data extractors with zero pattern mutations."""

from pathlib import Path
import sqlite3
import fastexcel
import polars as pl

from zaic_2026_4.ingestion.header_detector import detect_header_row


def extract_csv(path: Path | str) -> dict[str, pl.DataFrame]:
    """Extract CSV preserving exact column names and raw data."""
    p = Path(path)
    df: pl.DataFrame | None = None
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    for enc in encodings:
        try:
            df = pl.read_csv(p, encoding=enc)
            break
        except Exception:
            try:
                df = pl.read_csv(p, encoding=enc, infer_schema_length=0)
                break
            except Exception:
                continue

    if df is None:
        raise ValueError(f"Failed to read CSV: {p}")

    # Ensure unique column names if duplicates exist in raw CSV
    seen: set[str] = set()
    cleaned_cols: list[str] = []
    for idx, col in enumerate(df.columns):
        c_name = str(col).strip() if str(col).strip() else f"col_{idx + 1}"
        candidate = c_name
        counter = 1
        while candidate in seen:
            candidate = f"{c_name}_{counter}"
            counter += 1
        seen.add(candidate)
        cleaned_cols.append(candidate)
    df.columns = cleaned_cols

    return {p.stem: df}


def extract_excel(path: Path | str) -> dict[str, pl.DataFrame]:
    """Extract Excel sheets using fastexcel with automatic header row detection."""
    p = Path(path)
    reader = fastexcel.read_excel(p)
    sheet_names = reader.sheet_names
    tables: dict[str, pl.DataFrame] = {}

    for sheet in sheet_names:
        h_idx = detect_header_row(reader, sheet)
        try:
            df = reader.load_sheet(sheet, header_row=h_idx).to_polars()
        except Exception:
            try:
                df = reader.load_sheet(
                    sheet, header_row=h_idx, dtype_coercion="coerce"
                ).to_polars()
            except Exception:
                df = reader.load_sheet(
                    sheet, header_row=h_idx, dtypes="string"
                ).to_polars()

        # Rename any remaining blank header cells to avoid UNNAMED and guarantee uniqueness
        seen: set[str] = set()
        new_cols: list[str] = []
        for idx, col in enumerate(df.columns):
            c_str = str(col).strip()
            if not c_str or "unnamed" in c_str.lower():
                candidate = f"col_{idx + 1}"
            else:
                candidate = c_str
            base_cand = candidate
            counter = 1
            while candidate in seen:
                candidate = f"{base_cand}_{counter}"
                counter += 1
            seen.add(candidate)
            new_cols.append(candidate)
        df.columns = new_cols

        if len(sheet_names) == 1:
            tbl_name = p.stem
        else:
            tbl_name = f"{p.stem}_{sheet}"

        tables[tbl_name] = df

    return tables


def extract_sqlite(path: Path | str) -> dict[str, pl.DataFrame]:
    """Extract user tables from existing SQLite databases safely."""
    p = Path(path)
    tables: dict[str, pl.DataFrame] = {}
    conn = sqlite3.connect(p)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%';"
        )
        table_names = [r[0] for r in cur.fetchall()]
        for tname in table_names:
            cur.execute(f'SELECT * FROM "{tname}"')
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            try:
                df = pl.DataFrame(
                    rows, schema=cols, orient="row", infer_schema_length=None
                )
            except Exception:
                str_rows = [
                    [str(v) if v is not None else None for v in row]
                    for row in rows
                ]
                df = pl.DataFrame(str_rows, schema=cols, orient="row")
            tables[tname] = df
    finally:
        conn.close()
    return tables


def discover_and_extract_business(business_dir: Path) -> dict[str, pl.DataFrame]:
    """Discover and extract all tabular files in a business folder."""
    biz_dir = Path(business_dir)
    if not biz_dir.is_dir():
        raise NotADirectoryError(f"Business directory not found: {biz_dir}")

    all_tables: dict[str, pl.DataFrame] = {}
    candidate_files = sorted(biz_dir.iterdir(), key=lambda p: p.name.lower())

    for f in candidate_files:
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        extracted: dict[str, pl.DataFrame] = {}
        if ext == ".csv":
            extracted = extract_csv(f)
        elif ext in (".xlsx", ".xls"):
            extracted = extract_excel(f)
        elif ext in (".sqlite", ".sqlite3", ".db"):
            extracted = extract_sqlite(f)

        for tbl_name, df in extracted.items():
            final_name = tbl_name
            counter = 1
            while final_name in all_tables:
                final_name = f"{tbl_name}_{counter}"
                counter += 1
            all_tables[final_name] = df

    return all_tables
