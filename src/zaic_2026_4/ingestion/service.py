"""Orchestration service for business-level and split-level ingestion pipelines."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from zaic_2026_4.constants import DEFAULT_BASE_OUTPUT_DIR
from zaic_2026_4.ingestion.extractors import discover_and_extract_business, extract_csv
from zaic_2026_4.ingestion.sqlite_sink import write_tables_to_sqlite


def infer_split_name(path: Path) -> str | None:
    """Infer the dataset split name from a directory path (e.g. train)."""
    for part in reversed(path.parts):
        if part in ("train", "public_test", "private_test"):
            return part
    return None


def resolve_split_paths(
    split_name_or_path: str | Path,
    base_dir: Path | str = "dataset",
    require_questions: bool = True,
) -> dict[str, Any]:
    """Resolve static competition dataset paths for a given split.

    Args:
        split_name_or_path: Split name ('train', 'public_test', 'private_test') or direct path.
        base_dir: Base directory containing splits (default: 'dataset').
        require_questions: Whether questions.jsonl is mandatory (True for submission, False for profiling/linking).

    Returns:
        Dict containing:
            - split_dir: Path to split root directory.
            - questions_path: Path to questions.jsonl (or None if not required and absent).
            - data_dir: Path to data/ directory.
            - rulebook_path: Path to rulebook.md.
            - calendar_path: Path to calendar.csv (or None if not present).
    """
    raw = Path(split_name_or_path)
    split_dir = raw.resolve() if raw.is_dir() else (Path(base_dir) / split_name_or_path).resolve()

    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Could not locate dataset split directory for '{split_name_or_path}' at '{split_dir}'."
        )

    data_dir = split_dir / "data" if (split_dir / "data").is_dir() else split_dir
    rulebook_path = split_dir / "rulebook.md"
    calendar_candidate = split_dir / "calendar.csv"
    questions_candidate = split_dir / "questions.jsonl"

    if require_questions and not questions_candidate.is_file():
        raise FileNotFoundError(f"Questions file not found: {questions_candidate}")

    return {
        "split_dir": split_dir,
        "data_dir": data_dir,
        "rulebook_path": rulebook_path,
        "calendar_path": calendar_candidate if calendar_candidate.is_file() else None,
        "questions_path": questions_candidate if questions_candidate.is_file() else None,
    }


def ingest_business(
    business_dir: Path | str,
    output_dir: Path | str | None = None,
    create_views: bool = True,
    calendar_path: Path | str | None = None,
) -> Path:
    """Ingest all tabular data for a single business into an isolated SQLite database.

    Args:
        business_dir: Path to the business directory (e.g. dataset/train/data/biz_101).
        output_dir: Target output directory. Defaults to /tmp/zaic_db/<split>.
        create_views: Whether to generate unified VIEWs for identical schemas.
        calendar_path: Optional path to calendar.csv. Auto-discovered from parent directory if omitted.

    Returns:
        Path to the created SQLite database file.
    """
    biz_path = Path(business_dir).resolve()
    if not biz_path.is_dir():
        raise NotADirectoryError(f"Provided path is not a directory: {biz_path}")

    business_id = biz_path.name
    base = Path(output_dir) if output_dir is not None else DEFAULT_BASE_OUTPUT_DIR
    split_name = infer_split_name(biz_path)

    if split_name and base.name != split_name and output_dir is None:
        target_dir = base / split_name
    else:
        target_dir = base

    output_path = target_dir / f"{business_id}.db"

    # Extract all tabular data preserving raw types and schemas
    tables = discover_and_extract_business(biz_path)

    # Static competition structure: <split>/data/<biz_id> -> <split>/calendar.csv
    cal_file = (
        Path(calendar_path).resolve()
        if calendar_path
        else (biz_path.parent.parent / "calendar.csv")
    )
    if cal_file.is_file() and "calendar" not in tables:
        cal_tables = extract_csv(cal_file)
        for df in cal_tables.values():
            tables["calendar"] = df
            break

    # Persist tables to SQLite with view generation
    write_tables_to_sqlite(tables, output_path, create_views=create_views)

    return output_path


def ingest_split(
    split_name_or_path: str | Path,
    base_dir: Path | str = "dataset",
    output_dir: Path | str | None = None,
    create_views: bool = True,
    calendar_path: Path | str | None = None,
) -> dict[str, Path]:
    """Ingest all businesses within a dataset split into SQLite databases.

    Args:
        split_name_or_path: Name of split ('train') or direct path to split directory.
        base_dir: Base directory containing splits (default: 'dataset').
        output_dir: Target output directory. Defaults to /tmp/zaic_db/<split>.
        create_views: Whether to generate unified VIEWs for identical schemas.
        calendar_path: Optional path to calendar.csv file. Auto-discovered if omitted.

    Returns:
        Dictionary mapping business_id -> Path to created SQLite database.
    """
    paths = resolve_split_paths(
        split_name_or_path, base_dir=base_dir, require_questions=False
    )
    data_dir = paths["data_dir"]
    cal_path = (
        Path(calendar_path).resolve()
        if calendar_path
        else paths["calendar_path"]
    )

    base_out = Path(output_dir) if output_dir is not None else DEFAULT_BASE_OUTPUT_DIR
    target_output_dir = (
        base_out / paths["split_dir"].name if output_dir is None else base_out
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
            calendar_path=cal_path,
        )
        elapsed = time.perf_counter() - t0
        results[biz_id] = db_path
        print(f"[{idx}/{total}] Ingested {biz_id} -> {db_path} ({elapsed:.2f}s)")

    return results
