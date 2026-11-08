"""CLI interface for running Task-Aligned Schema Linking on business questions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from zaic_2026_4.profiling.profile_manager import ProfilingOrchestrator
from zaic_2026_4.profiling.profile_summarizer import (
    create_engine,
)
from zaic_2026_4.profiling.vector_indexer import MockEmbedder
from zaic_2026_4.schema_linking.task_aligned_linker import TaskAlignedSchemaLinker


def main() -> None:
    """CLI entry point for Task-Aligned Schema Linking."""
    parser = argparse.ArgumentParser(
        description="Run Task-Aligned Schema Linking on natural-language questions."
    )
    parser.add_argument(
        "--business-dir",
        type=str,
        required=True,
        help="Path to business directory containing data files and README.md.",
    )
    parser.add_argument(
        "--rulebook",
        type=str,
        default="dataset/train/rulebook.md",
        help="Path to dataset split rulebook.md.",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Natural language Vietnamese business question.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for profiling artifacts (defaults to /tmp/cache_<biz_id>).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally using lightweight Qwen/Qwen2.5-1.5B-Instruct.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in deterministic mock mode without GPU/network access.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Custom LLM model name/path.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Explicit GPU memory utilization fraction override (e.g. 0.70 or 0.60).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum re-prompting retries on literal mismatch (default: 2).",
    )

    args = parser.parse_args()

    biz_dir = Path(args.business_dir).resolve()
    rulebook_path = Path(args.rulebook).resolve()

    if not biz_dir.is_dir():
        print(f"Error: Business directory not found: {biz_dir}", file=sys.stderr)
        sys.exit(1)

    # 1. Initialize LLM engine
    llm = create_engine(
        local=args.local,
        model=args.model,
        mock=args.mock,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    embedder = MockEmbedder() if args.mock else None
    vector_model_path = "mock" if args.mock else "BAAI/bge-m3"

    # 2. Load or run profiling pipeline to get metadata, LSH, FAISS
    orchestrator = ProfilingOrchestrator(
        business_dir=biz_dir,
        rulebook_path=rulebook_path,
        llm_engine=llm,
        embedder=embedder,
        vector_model_path=vector_model_path,
        cache_dir=args.cache_dir,
    )
    pipeline_res = orchestrator.run_pipeline(force_rebuild=False)
    processed_metadata = pipeline_res["metadata"]
    lsh_indexer = pipeline_res["lsh_indexer"]
    vector_indexer = pipeline_res["vector_indexer"]

    # Load raw profile if cached
    raw_profile = None
    cache_dir = Path(args.cache_dir) if args.cache_dir else orchestrator.cache_dir
    raw_profile_path = cache_dir / "raw_profile.json"
    if raw_profile_path.is_file():
        try:
            with open(raw_profile_path, "r", encoding="utf-8") as f:
                raw_profile = json.load(f)
        except (OSError, json.JSONDecodeError):
            raw_profile = None

    # 3. Execute Task-Aligned Schema Linking
    linker = TaskAlignedSchemaLinker(
        llm_engine=llm,
        max_retries=args.max_retries,
    )

    t0 = time.perf_counter()
    result = linker.link_schema(
        question=args.question,
        processed_metadata=processed_metadata,
        lsh_indexer=lsh_indexer,
        vector_indexer=vector_indexer,
        raw_profile=raw_profile,
    )
    elapsed = time.perf_counter() - t0

    print(json.dumps({
        "question": args.question,
        "elapsed_seconds": round(elapsed, 3),
        "total_retries": result.total_retries,
        "final_linked_schema": sorted(result.final_linked_schema),
        "canonical_schema": sorted(result.canonical_schema),
        "preserved_keys": sorted(result.preserved_keys),
        "variant_traces": [
            {
                "case_name": t.case_name,
                "profile_type": t.profile_type,
                "retry_count": t.retry_count,
                "fields": sorted(t.fields_harvested),
                "mismatches": t.mismatches_detected,
                "final_sql": t.final_sql,
            }
            for t in result.variant_traces
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
