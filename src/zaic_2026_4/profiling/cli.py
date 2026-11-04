"""Command-line interface for database profiling, MinHash join detection, and LLM summarization."""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from zaic_2026_4.profiling.lsh_indexer import VietnameseLSHIndexer
from zaic_2026_4.profiling.profile_summarizer import (
    DEFAULT_LOCAL_GPU_UTIL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_PROD_GPU_UTIL,
    DEFAULT_PROD_MODEL,
    ProfileLLMSummarizer,
    create_engine,
)
from zaic_2026_4.profiling.raw_profiler import RawDatabaseProfiler
from zaic_2026_4.profiling.vector_indexer import (
    DEFAULT_MAX_LENGTH as DEFAULT_VECTOR_MAX_LENGTH,
    DEFAULT_MODEL_PATH as DEFAULT_VECTOR_MODEL_PATH,
    MockEmbedder,
    SemanticVectorIndexer,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Profile SQLite databases and detect undocumented FK-PK joins."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--db",
        type=str,
        help="Path to a single SQLite database file.",
    )
    group.add_argument(
        "--dir",
        type=str,
        help="Directory containing SQLite database files (*.db).",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Target output JSON path (for single database profiling).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target output directory for profiles (for batch directory profiling).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="MinHash Jaccard threshold for FK-PK candidate joins (default: 0.7).",
    )
    parser.add_argument(
        "--num-perm",
        type=int,
        default=128,
        help="Number of MinHash permutations (default: 128).",
    )
    parser.add_argument(
        "--no-views",
        action="store_true",
        help="Disable profiling of SQLite VIEWs.",
    )
    parser.add_argument(
        "--all-tables",
        action="store_true",
        help="Do not skip partition tables even if a unified VIEW exists.",
    )
    parser.add_argument(
        "--include-continuous-numerics",
        action="store_true",
        help="Include continuous numeric/float columns in MinHash Jaccard calculations.",
    )

    # Module 2 LLM Profile Summarization arguments
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Run LLM profile summarizer to generate Minimal and Maximal column profiles.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            f"Run local test mode: use {DEFAULT_LOCAL_MODEL} with "
            f"gpu_memory_utilization={DEFAULT_LOCAL_GPU_UTIL}. If omitted, defaults to "
            f"{DEFAULT_PROD_MODEL} with gpu_memory_utilization={DEFAULT_PROD_GPU_UTIL}."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Explicit LLM model name or checkpoint path override.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Explicit GPU memory utilization fraction override (e.g., 0.95 or 0.60).",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use deterministic offline MockLLMEngine without loading GPU weights.",
    )
    parser.add_argument(
        "--readme",
        type=str,
        default=None,
        help="Path to business README.md or directory containing README.md.",
    )

    # Module 3 Vietnamese LSH Indexer arguments
    parser.add_argument(
        "--build-lsh",
        action="store_true",
        help="Build Vietnamese LSH index over database column samples.",
    )
    parser.add_argument(
        "--lsh-threshold",
        type=float,
        default=0.4,
        help="LSH MinHash Jaccard threshold (default: 0.4).",
    )
    parser.add_argument(
        "--lsh-path",
        type=str,
        default=None,
        help="Path to save or load the LSH index pickle file.",
    )
    parser.add_argument(
        "--query-lsh",
        type=str,
        default=None,
        help="Literal query text to match against the LSH index (e.g. 'Sting dâu').",
    )

    # Module 4 Offline FAISS Semantic Vector Indexer arguments
    parser.add_argument(
        "--build-vector",
        action="store_true",
        help="Build offline FAISS semantic vector index over columns and rules.",
    )
    parser.add_argument(
        "--vector-model",
        type=str,
        default=DEFAULT_VECTOR_MODEL_PATH,
        help=(
            f"Path or identifier for BGE-M3 model weights (default: {DEFAULT_VECTOR_MODEL_PATH})."
        ),
    )
    parser.add_argument(
        "--vector-max-len",
        type=int,
        default=DEFAULT_VECTOR_MAX_LENGTH,
        help=(
            f"Maximum token length for FlagEmbedding BGEM3FlagModel "
            f"(default: {DEFAULT_VECTOR_MAX_LENGTH})."
        ),
    )
    parser.add_argument(
        "--vector-path",
        type=str,
        default=None,
        help="Directory path to save or load the FAISS vector index.",
    )
    parser.add_argument(
        "--query-vector",
        type=str,
        default=None,
        help="Natural language query for semantic vector search against columns and rules.",
    )
    parser.add_argument(
        "--mock-embedder",
        action="store_true",
        help="Use deterministic offline MockEmbedder without loading GPU/CPU model weights.",
    )
    parser.add_argument(
        "--rulebook",
        type=str,
        default=None,
        help="Path to rulebook.md for vector indexing.",
    )
    return parser


def _find_readme_text(db_path: Path, explicit_readme: str | None) -> str:
    """Find and read business README.md if available."""
    if explicit_readme:
        p = Path(explicit_readme)
        if p.is_dir():
            p = p / "README.md"
        if p.is_file():
            return p.read_text(encoding="utf-8")

    # Check database parent folder
    parent_readme = db_path.parent / "README.md"
    if parent_readme.is_file():
        return parent_readme.read_text(encoding="utf-8")

    # Check standard dataset layout dataset/train/data/<biz_stem>/README.md
    for root in (Path("dataset/train/data"), Path("dataset/public_test/data")):
        candidate = root / db_path.stem / "README.md"
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")

    return ""


def _find_rulebook_text(db_path: Path, explicit_rulebook: str | None) -> str:
    """Find and read rulebook.md if available."""
    if explicit_rulebook:
        p = Path(explicit_rulebook)
        if p.is_dir():
            p = p / "rulebook.md"
        if p.is_file():
            return p.read_text(encoding="utf-8")

    # Check database parent folder and ancestors
    for parent in (db_path.parent, db_path.parent.parent, db_path.parent.parent.parent):
        candidate = parent / "rulebook.md"
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")

    # Check standard dataset layout
    for root in (Path("dataset/train"), Path("dataset/public_test"), Path("dataset")):
        candidate = root / "rulebook.md"
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")

    return ""


def profile_single_db(
    db_path: Path,
    args: argparse.Namespace,
    output_path: Path | None = None,
    shared_summarizer: ProfileLLMSummarizer | None = None,
) -> dict[str, Any]:
    """Profile one database, optionally summarize with LLM, and persist profile to JSON."""
    profiler = RawDatabaseProfiler(
        db_path=db_path,
        num_perm=args.num_perm,
        include_views=not args.no_views,
        skip_partition_tables_if_view_exists=not args.all_tables,
        cross_table_only=True,
        jaccard_threshold=args.threshold,
        exclude_continuous_numerics=not args.include_continuous_numerics,
    )
    raw_profile = profiler.profile_database()

    result: dict[str, Any] = {"raw_profile": raw_profile}

    if args.summarize:
        t_sum_start = time.perf_counter()
        if shared_summarizer is not None:
            summarizer = shared_summarizer
        else:
            engine = create_engine(
                local=args.local,
                model=args.model,
                gpu_memory_utilization=args.gpu_memory_utilization,
                mock=args.mock_llm,
            )
            summarizer = ProfileLLMSummarizer(engine)

        readme_text = _find_readme_text(db_path, args.readme)

        summarized = summarizer.generate_profiles(
            db_profile=raw_profile,
            business_readme=readme_text,
        )
        result["metadata_profile"] = summarized
        sum_elapsed = time.perf_counter() - t_sum_start
        print(f"Generated LLM summaries for {len(summarized)} columns in {sum_elapsed:.3f}s.")

    # Module 3 Vietnamese LSH Indexing
    lsh_indexer: VietnameseLSHIndexer | None = None
    if args.build_lsh or args.query_lsh:
        t_lsh_start = time.perf_counter()
        lsh_indexer = VietnameseLSHIndexer(
            threshold=args.lsh_threshold,
            num_perm=args.num_perm,
        )
        meta_source = result.get("metadata_profile") or raw_profile
        indexed_items = lsh_indexer.build_index(meta_source)
        lsh_elapsed = time.perf_counter() - t_lsh_start
        print(f"Built Vietnamese LSH index with {indexed_items} literal items in {lsh_elapsed:.3f}s.")

        lsh_out = None
        if args.lsh_path:
            lsh_out = Path(args.lsh_path)
        elif output_path:
            lsh_out = output_path.with_name(f"{output_path.stem}_lsh.pkl")

        if lsh_out:
            lsh_indexer.save(lsh_out)
            print(f"LSH index saved to: {lsh_out}")

    if args.query_lsh and lsh_indexer is not None:
        matches = lsh_indexer.query_literal_with_scores(args.query_lsh, top_k=5)
        print(f"Query '{args.query_lsh}' matches:")
        for col, val, score in matches:
            print(f"  - [{score:.3f}] {col} -> '{val}'")

    # Module 4 Offline FAISS Semantic Vector Indexing
    vector_indexer: SemanticVectorIndexer | None = None
    if args.build_vector or args.query_vector:
        t_vec_start = time.perf_counter()
        embedder = MockEmbedder() if args.mock_embedder else None
        vector_indexer = SemanticVectorIndexer(
            model_path=args.vector_model,
            embedder=embedder,
            max_length=args.vector_max_len,
        )
        meta_source = result.get("metadata_profile") or raw_profile
        rulebook_text = _find_rulebook_text(db_path, args.rulebook)
        readme_text = _find_readme_text(db_path, args.readme)

        indexed_docs = vector_indexer.build_index(
            processed_metadata=meta_source,
            rulebook_text=rulebook_text,
            readme_text=readme_text,
        )
        vec_elapsed = time.perf_counter() - t_vec_start
        print(
            f"Built FAISS vector index with {indexed_docs} docs in {vec_elapsed:.3f}s."
        )

        vec_out = None
        if args.vector_path:
            vec_out = Path(args.vector_path)
        elif output_path:
            vec_out = output_path.parent / f"{output_path.stem}_faiss"

        if vec_out:
            vector_indexer.save(vec_out)
            print(f"FAISS vector index saved to: {vec_out}")

    if args.query_vector and vector_indexer is not None:
        hits = vector_indexer.search_similar_schema(args.query_vector, top_k=5)
        print(f"Query '{args.query_vector}' top semantic matches:")
        for hit in hits:
            print(f"  - [{hit['score']:.3f}] ({hit['type']}) {hit['key']}")

    # Save to JSON
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # If summarization was requested, write the complete output package
        content_to_save = result if args.summarize else raw_profile
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(content_to_save, f, ensure_ascii=False, indent=2)

    return result if args.summarize else raw_profile


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    # Pre-initialize shared LLM engine if batch profiling with summarization
    shared_summarizer: ProfileLLMSummarizer | None = None
    if args.summarize and args.dir:
        engine = create_engine(
            local=args.local,
            model=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            mock=args.mock_llm,
        )
        shared_summarizer = ProfileLLMSummarizer(engine)

    if args.db:
        db_file = Path(args.db)
        if not db_file.is_file():
            print(f"Error: Database file does not exist: {db_file}", file=sys.stderr)
            sys.exit(1)

        t0 = time.perf_counter()
        out_file = Path(args.output) if args.output else None
        res = profile_single_db(db_file, args, output_path=out_file)
        elapsed = time.perf_counter() - t0

        raw_profile = res["raw_profile"] if args.summarize else res
        tbl_count = len(raw_profile["tables"])
        join_count = len(raw_profile["undocumented_joins"])
        print(
            f"Successfully processed {db_file.name} in {elapsed:.3f}s: "
            f"{tbl_count} entities, {join_count} candidate join paths."
        )
        if out_file:
            print(f"Profile saved to: {out_file}")

    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"Error: Directory does not exist: {dir_path}", file=sys.stderr)
            sys.exit(1)

        db_files = sorted(
            [
                p
                for p in dir_path.iterdir()
                if p.is_file() and p.suffix.lower() in (".db", ".sqlite", ".sqlite3")
            ]
        )
        if not db_files:
            print(
                f"No SQLite database files (.db, .sqlite, .sqlite3) found in {dir_path}",
                file=sys.stderr,
            )
            sys.exit(1)

        out_dir = Path(args.output_dir) if args.output_dir else dir_path / "profiles"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Profiling {len(db_files)} databases in {dir_path} ...")
        t_start = time.perf_counter()
        total_joins = 0

        for db_file in db_files:
            out_file = out_dir / f"{db_file.stem}_profile.json"
            res = profile_single_db(
                db_file,
                args,
                output_path=out_file,
                shared_summarizer=shared_summarizer,
            )
            raw_prof = res["raw_profile"] if args.summarize else res
            total_joins += len(raw_prof["undocumented_joins"])

        total_elapsed = time.perf_counter() - t_start
        print(
            f"Completed processing {len(db_files)} databases in {total_elapsed:.2f}s "
            f"({total_elapsed / len(db_files):.3f}s/db). "
            f"Discovered {total_joins} total candidate joins."
        )
        print(f"Profiles written to: {out_dir}")


if __name__ == "__main__":
    main()
