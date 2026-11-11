"""Command-line interface for dataset split database profiling and indexing."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from zaic_2026_4.ingestion import resolve_split_paths
from zaic_2026_4.profiling.profile_manager import (
    ProfilingOrchestrator,
    profile_split_calendar,
)
from zaic_2026_4.profiling.profile_summarizer import (
    DEFAULT_LOCAL_MODEL,
    ProfileLLMSummarizer,
    create_engine,
)
from zaic_2026_4.profiling.raw_profiler import RawDatabaseProfiler
from zaic_2026_4.profiling.vector_indexer import MockEmbedder


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for zaic-profile."""
    parser = argparse.ArgumentParser(
        description="Profile SQLite databases and generate schemas, LSH, and vector indexes for a dataset split."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dataset/train",
        help="Name or path of dataset split (e.g. 'train', 'public_test', 'private_test'). Default: 'dataset/train'.",
    )
    parser.add_argument(
        "--biz-id",
        "--biz_id",
        dest="biz_id",
        type=str,
        default=None,
        help="Identifier of a single business to profile within the split (e.g. '101' or 'biz_101').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target output directory for profiles (defaults to /tmp/cache_<biz_id>).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=f"Run local test mode: use {DEFAULT_LOCAL_MODEL} with gpu_memory_utilization=0.6.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic offline MockLLMEngine and MockEmbedder without loading GPU weights.",
    )
    return parser


def _resolve_biz_dir(data_dir: Path, biz_id: str) -> Path | None:
    """Resolve business directory by biz_id, supporting raw numeric ID (e.g. '101') or formatted name ('biz_101')."""
    # 1. Exact match
    candidate = data_dir / biz_id
    if candidate.is_dir():
        return candidate.resolve()

    # 2. If without 'biz_' prefix, try adding 'biz_' and zero-padding
    if not biz_id.startswith("biz_"):
        candidate = data_dir / f"biz_{biz_id}"
        if candidate.is_dir():
            return candidate.resolve()
        if biz_id.isdigit():
            for pad in (2, 3):
                cand = data_dir / f"biz_{biz_id.zfill(pad)}"
                if cand.is_dir():
                    return cand.resolve()
    else:
        # 3. If with 'biz_' prefix, try stripping prefix
        stripped = biz_id.removeprefix("biz_")
        candidate = data_dir / stripped
        if candidate.is_dir():
            return candidate.resolve()
        if stripped.isdigit():
            cand = data_dir / str(int(stripped))
            if cand.is_dir():
                return cand.resolve()

    return None


def _find_readme_text(db_path: Path, explicit_readme: str | None = None) -> str:
    """Find and read business README.md if available."""
    if explicit_readme:
        p = Path(explicit_readme)
        if p.is_dir():
            p = p / "README.md"
        if p.is_file():
            return p.read_text(encoding="utf-8")

    parent_readme = db_path.parent / "README.md"
    if parent_readme.is_file():
        return parent_readme.read_text(encoding="utf-8")

    curr = db_path.parent
    for _ in range(4):
        if curr is None or curr == curr.parent:
            break
        candidate = curr / "data" / db_path.stem / "README.md"
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
        candidate = curr / db_path.stem / "README.md"
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
        curr = curr.parent

    return ""


def profile_single_db(
    db_path: Path,
    args: Any = None,
    output_path: Path | None = None,
    shared_summarizer: ProfileLLMSummarizer | None = None,
) -> dict[str, Any]:
    """Profile one database, optionally summarize with LLM, and persist profile to JSON."""
    num_perm = getattr(args, "num_perm", 128)
    include_views = not getattr(args, "no_views", False)
    skip_partition = not getattr(args, "all_tables", False)
    threshold = getattr(args, "threshold", 0.7)
    exclude_continuous = not getattr(args, "include_continuous_numerics", False)
    summarize = getattr(args, "summarize", True)
    is_mock = getattr(args, "mock", False) or getattr(args, "mock_llm", False)
    is_local = getattr(args, "local", False)

    profiler = RawDatabaseProfiler(
        db_path=db_path,
        num_perm=num_perm,
        include_views=include_views,
        skip_partition_tables_if_view_exists=skip_partition,
        cross_table_only=True,
        jaccard_threshold=threshold,
        exclude_continuous_numerics=exclude_continuous,
    )
    raw_profile = profiler.profile_database()
    result: dict[str, Any] = {"raw_profile": raw_profile}

    if summarize:
        if shared_summarizer is not None:
            summarizer = shared_summarizer
        else:
            engine = create_engine(local=is_local, mock=is_mock)
            summarizer = ProfileLLMSummarizer(engine)

        readme_text = _find_readme_text(db_path, getattr(args, "readme", None))
        summarized = summarizer.generate_profiles(
            db_profile=raw_profile,
            business_readme=readme_text,
        )
        result["metadata_profile"] = summarized

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content_to_save = result if summarize else raw_profile
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(content_to_save, f, ensure_ascii=False, indent=2)

    return result if summarize else raw_profile


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    t_start = time.perf_counter()
    if args.mock:
        engine = create_engine(mock=True)
        embedder = MockEmbedder()
    else:
        engine = create_engine(local=args.local)
        embedder = None

    paths = resolve_split_paths(args.split, require_questions=False)
    split_p = paths["split_dir"]
    data_p = paths["data_dir"]
    rb_path = paths["rulebook_path"]
    cal_path = paths["calendar_path"]

    if args.biz_id:
        single_biz_dir = _resolve_biz_dir(data_p, args.biz_id)
        if single_biz_dir is None or not single_biz_dir.is_dir():
            print(f"Error: Business directory not found for '{args.biz_id}' in {data_p}", file=sys.stderr)
            sys.exit(1)
        biz_dirs = [single_biz_dir]
    else:
        biz_dirs = sorted([d for d in data_p.iterdir() if d.is_dir()])
        if not biz_dirs:
            print(f"Error: No business directories found in {data_p}", file=sys.stderr)
            sys.exit(1)

    print(f"Initializing profiling for {split_p.name} ({len(biz_dirs)} business(es))...")

    if args.output_dir and cal_path and cal_path.is_file():
        cal_target = Path(args.output_dir).resolve() / "calendar_profile.json"
        cal_target.parent.mkdir(parents=True, exist_ok=True)
        cal_res = profile_split_calendar(
            calendar_path=cal_path,
            llm_engine=engine,
            embedder=embedder,
            cache_path=cal_target,
            force_rebuild=True,
        )
        if embedder is None and "vector_indexer" in cal_res and cal_res["vector_indexer"] is not None:
            embedder = cal_res["vector_indexer"].embedder

    total_meta = 0
    total_lsh = 0
    total_vector = 0

    for idx, b_dir in enumerate(biz_dirs, 1):
        t_b0 = time.time()
        b_cache = (
            Path(args.output_dir).resolve() / f"cache_{b_dir.name}"
            if args.output_dir
            else Path(f"/tmp/cache_{b_dir.name}")
        )
        print(f"[{idx}/{len(biz_dirs)}] Processing {b_dir.name}...")

        orchestrator = ProfilingOrchestrator(
            business_dir=b_dir,
            rulebook_path=rb_path,
            llm_engine=engine,
            embedder=embedder,
            cache_dir=b_cache,
            calendar_path=cal_path if (cal_path and cal_path.is_file()) else None,
        )
        pipe = orchestrator.run_pipeline(force_rebuild=True)
        if embedder is None and "vector_indexer" in pipe and pipe["vector_indexer"] is not None:
            embedder = pipe["vector_indexer"].embedder

        m_cnt = len(pipe["metadata"])
        l_cnt = len(pipe["lsh_indexer"].value_to_col_map)
        v_cnt = len(pipe["vector_indexer"].doc_map)

        total_meta += m_cnt
        total_lsh += l_cnt
        total_vector += v_cnt
        print(f"     Done in {time.time() - t_b0:.2f}s ({m_cnt} columns, {l_cnt} LSH items, {v_cnt} docs).")

    elapsed_total = time.perf_counter() - t_start
    print(f"Split profiling completed successfully for {len(biz_dirs)} business(es) in {elapsed_total:.2f}s:")
    print(f"  - Total metadata columns: {total_meta}")
    print(f"  - Total LSH items:        {total_lsh}")
    print(f"  - Total FAISS documents:  {total_vector}")
    print(f"  - Average per business:   {elapsed_total / len(biz_dirs):.2f}s")


if __name__ == "__main__":
    main()
