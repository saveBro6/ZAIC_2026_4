"""Excel header row detection based on statistical cell density and uniqueness."""

from typing import Any
import fastexcel
import polars as pl


def detect_header_row(
    reader: fastexcel.ExcelReader,
    sheet_name: str | int,
    max_scan_rows: int = 20,
) -> int:
    """Detect the true header row index of an Excel sheet.

    Scores candidate rows based on non-empty string count and uniqueness,
    skipping title banners, filter notes, and blank rows without pattern regex.
    """
    try:
        sample_df = reader.load_sheet(
            sheet_name, header_row=None, n_rows=max_scan_rows
        ).to_polars()
    except Exception:
        return 0

    if sample_df.height == 0:
        return 0

    best_row_idx = 0
    best_score = -1

    for r_idx in range(sample_df.height):
        row = sample_df.row(r_idx)
        str_cells = [
            str(v).strip()
            for v in row
            if v is not None and str(v).strip() != ""
        ]
        if not str_cells:
            continue

        non_empty_count = len(str_cells)
        distinct_count = len(set(str_cells))

        score = non_empty_count * 2 + distinct_count

        if score > best_score:
            best_score = score
            best_row_idx = r_idx

    return best_row_idx
