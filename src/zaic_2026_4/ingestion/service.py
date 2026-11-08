"""Orchestration service for business-level and split-level ingestion pipelines."""

import time
from pathlib import Path

from zaic_2026_4.constants import DEFAULT_BASE_OUTPUT_DIR
from zaic_2026_4.ingestion.extractors import discover_and_extract_business
from zaic_2026_4.ingestion.sqlite_sink import write_tables_to_sqlite


def infer_split_name(path: Path) -> str | None:
    """Infer the dataset split name from a directory path (e.g. train)."""
    for part in reversed(path.parts):
        if part in ("train", "public_test", "private_test"):
            return part
    return None


def ingest_business(
    business_dir: Path | str,
    output_dir: Path | str | None = None,
    create_views: bool = True,
) -> Path:
    """Ingest all tabular data for a single business into an isolated SQLite database.

    Args:
        business_dir: Path to the business directory (e.g. dataset/train/data/biz_101).
        output_dir: Target output directory. Defaults to /tmp/zaic_db/<split>.
        create_views: Whether to generate unified VIEWs for identical schemas.

    Returns:
        Path to the created SQLite database file.
    """
    biz_path = Path(business_dir).resolve()
    if not biz_path.is_dir():
        raise NotADirectoryError(f"Provided path is not a directory: {biz_path}")

    business_id = biz_path.name
    base = Path(output_dir) if output_dir is not None else DEFAULT_BASE_OUTPUT_DIR
    split_name = infer_split_name(biz_path)

    if split_name and base.name != split_name:
        target_dir = base / split_name
    else:
        target_dir = base

    output_path = target_dir / f"{business_id}.db"

    # Extract all tabular data preserving raw types and schemas
    tables = discover_and_extract_business(biz_path)

    # Persist tables to SQLite with view generation
    write_tables_to_sqlite(tables, output_path, create_views=create_views)

    return output_path


def ingest_split(
    split_name_or_path: str | Path,
    base_dir: Path | str = "dataset",
    output_dir: Path | str | None = None,
    create_views: bool = True,
) -> dict[str, Path]:
    """Ingest all businesses within a dataset split into SQLite databases.

    Args:
        split_name_or_path: Name of split ('train') or direct path to split directory.
        base_dir: Base directory containing splits (default: 'dataset').
        output_dir: Target output directory. Defaults to /tmp/zaic_db/<split>.
        create_views: Whether to generate unified VIEWs for identical schemas.

    Returns:
        Dictionary mapping business_id -> Path to created SQLite database.
    """
    raw_path = Path(split_name_or_path)

    if str(split_name_or_path) in ("train", "public_test", "private_test"):
        split_name = str(split_name_or_path)
    else:
        split_name = raw_path.name
        if split_name == "data":
            split_name = raw_path.parent.name

    base_out = Path(output_dir) if output_dir is not None else DEFAULT_BASE_OUTPUT_DIR
    if split_name and base_out.name != split_name:
        target_output_dir = base_out / split_name
    else:
        target_output_dir = base_out

    if raw_path.is_dir() and (raw_path / "data").is_dir():
        data_dir = raw_path / "data"
    elif raw_path.is_dir() and any(p.is_dir() for p in raw_path.iterdir()):
        data_dir = raw_path
    else:
        candidate = Path(base_dir) / split_name_or_path / "data"
        if candidate.is_dir():
            data_dir = candidate
        else:
            candidate_alt = Path(base_dir) / split_name_or_path
            if candidate_alt.is_dir():
                data_dir = candidate_alt
            else:
                raise FileNotFoundError(
                    f"Could not locate data directory for split '{split_name_or_path}'."
                )

    biz_folders = sorted(
        [p for p in data_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )

    results: dict[str, Path] = {}
    total = len(biz_folders)

    for idx, biz_folder in enumerate(biz_folders, 1):
        biz_id = biz_folder.name
        t0 = time.perf_counter()
        db_path = ingest_business(
            biz_folder,
            output_dir=target_output_dir,
            create_views=create_views,
        )
        elapsed = time.perf_counter() - t0
        results[biz_id] = db_path
        print(f"[{idx}/{total}] Ingested {biz_id} -> {db_path} ({elapsed:.2f}s)")

    return results
