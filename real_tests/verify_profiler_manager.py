"""Benchmark verification script for ProfilingOrchestrator across first 3 train businesses.

Uses real vLLM (Qwen2.5) and real BAAI/bge-m3 embedding models (no mocks).
Verifies biz_101 (CSV), biz_102 (Multi-sheet Excel), and biz_103 (Multi-sheet Excel) from dataset/train.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

from zaic_2026_4.profiling.profile_manager import ProfilingOrchestrator
from zaic_2026_4.profiling.profile_summarizer import (
    DEFAULT_LOCAL_GPU_UTIL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_PROD_GPU_UTIL,
    DEFAULT_PROD_MODEL,
    create_engine,
)

TARGET_BIZ_IDS = ["biz_101", "biz_102", "biz_103"]
TRAIN_DATA_BASE = Path("dataset/train/data")
RULEBOOK_PATH = Path("dataset/train/rulebook.md")


def verify_business_orchestration(
    biz_id: str,
    llm_engine: Any,
    base_dir: Path = TRAIN_DATA_BASE,
    rulebook_path: Path = RULEBOOK_PATH,
    cache_base_dir: Path | None = None,
    vector_model_path: str = "BAAI/bge-m3",
) -> dict[str, Any]:
    """Execute real-data cold and warm pipeline verification for a single business."""
    biz_dir = base_dir / biz_id
    if not biz_dir.is_dir():
        raise FileNotFoundError(f"Business directory not found: {biz_dir}")
    if not rulebook_path.is_file():
        raise FileNotFoundError(f"Rulebook file not found: {rulebook_path}")

    if cache_base_dir is not None:
        cache_dir = cache_base_dir / f"cache_{biz_id}"
    else:
        cache_dir = Path(f"/tmp/real_verify_cache_{biz_id}")

    # Ensure fresh cache directory for real cold run
    if cache_dir.exists():
        import shutil

        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cold Run: Real Ingestion + Real Raw Profiling + Real vLLM Summarization + Real BGE-M3 FAISS
    t0_cold = time.perf_counter()
    orchestrator = ProfilingOrchestrator(
        business_dir=biz_dir,
        rulebook_path=rulebook_path,
        llm_engine=llm_engine,
        vector_model_path=vector_model_path,
        cache_dir=cache_dir,
    )
    cold_pipeline = orchestrator.run_pipeline(force_rebuild=True)
    cold_elapsed = time.perf_counter() - t0_cold

    meta = cold_pipeline["metadata"]
    lsh = cold_pipeline["lsh_indexer"]
    vec = cold_pipeline["vector_indexer"]

    column_count = len(meta)
    lsh_literal_count = len(lsh.value_to_col_map)
    vec_doc_count = len(vec.doc_map)

    # Verify generated cache files on disk
    assert (cache_dir / "metadata_profile.json").is_file(), "metadata_profile.json missing"
    assert (cache_dir / "raw_profile.json").is_file(), "raw_profile.json missing"
    assert (cache_dir / "lsh_index.pkl").is_file(), "lsh_index.pkl missing"
    assert (cache_dir / "faiss_index" / "index.faiss").is_file(), "index.faiss missing"
    assert (cache_dir / "faiss_index" / "doc_map.json").is_file(), "doc_map.json missing"

    # 2. Warm Run: Instant disk reload verification
    t0_warm = time.perf_counter()
    warm_orchestrator = ProfilingOrchestrator(
        business_dir=biz_dir,
        rulebook_path=rulebook_path,
        llm_engine=llm_engine,
        vector_model_path=vector_model_path,
        cache_dir=cache_dir,
    )
    warm_pipeline = warm_orchestrator.run_pipeline(force_rebuild=False)
    warm_elapsed = time.perf_counter() - t0_warm

    # 3. Real semantic queries against FAISS & LSH
    col_matches = warm_orchestrator.search_columns("sản phẩm", top_k=3)
    rule_matches = warm_orchestrator.search_rules("doanh thu", top_k=2)

    sample_meta_key = next(iter(meta.keys()))
    sample_minimal = meta[sample_meta_key]["minimal_profile"]

    return {
        "biz_id": biz_id,
        "cold_sec": cold_elapsed,
        "warm_sec": warm_elapsed,
        "speedup": (cold_elapsed / warm_elapsed) if warm_elapsed > 0 else 0.0,
        "columns": column_count,
        "lsh_literals": lsh_literal_count,
        "vector_docs": vec_doc_count,
        "sample_col": col_matches[0]["key"] if col_matches else "N/A",
        "sample_minimal": sample_minimal,
        "cache_dir": str(cache_dir),
    }


def main() -> None:
    """CLI execution for verifying ProfilingOrchestrator on real train businesses."""
    parser = argparse.ArgumentParser(
        description="Verify ProfilingOrchestrator on first 3 businesses using real LLM and Embedding models."
    )
    parser.add_argument(
        "--biz-ids",
        nargs="+",
        default=TARGET_BIZ_IDS,
        help="Target business IDs to verify (default: biz_101 biz_102 biz_103).",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(TRAIN_DATA_BASE),
        help="Base directory containing train data.",
    )
    parser.add_argument(
        "--rulebook",
        type=str,
        default=str(RULEBOOK_PATH),
        help="Path to rulebook.md.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"LLM model path or repo ID (defaults to {DEFAULT_LOCAL_MODEL} when local, else {DEFAULT_PROD_MODEL}).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=True,
        help=f"Run in local mode with {DEFAULT_LOCAL_MODEL} and gpu_mem={DEFAULT_LOCAL_GPU_UTIL}.",
    )
    parser.add_argument(
        "--gpu-mem",
        type=float,
        default=None,
        help="GPU memory utilization override (e.g. 0.60 or 0.70).",
    )
    parser.add_argument(
        "--vector-model",
        type=str,
        default="BAAI/bge-m3",
        help="SentenceTransformer / BGEM3 embedding model path or repo ID.",
    )

    args = parser.parse_args()

    print("=== Real Data Verification: ProfilingOrchestrator ===")
    print(f"Target Businesses: {args.biz_ids}")
    print(f"Data Directory:    {args.base_dir}")
    print(f"Rulebook:          {args.rulebook}")
    print(f"Embedding Model:   {args.vector_model}")

    # Initialize real vLLM engine once to be reused across all businesses
    print("\nInitializing real vLLM engine...")
    t_llm = time.perf_counter()
    engine = create_engine(
        local=args.local,
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        mock=False,
    )
    print(f"vLLM engine initialized in {time.perf_counter() - t_llm:.2f}s.\n")

    summary_results = []
    total_cold_time = 0.0
    total_warm_time = 0.0

    for idx, biz_id in enumerate(args.biz_ids, 1):
        print(f"[{idx}/{len(args.biz_ids)}] Running real verification for {biz_id} ...")
        t_start = time.perf_counter()
        metrics = verify_business_orchestration(
            biz_id=biz_id,
            llm_engine=engine,
            base_dir=Path(args.base_dir),
            rulebook_path=Path(args.rulebook),
            vector_model_path=args.vector_model,
        )
        total_time = time.perf_counter() - t_start
        total_cold_time += metrics["cold_sec"]
        total_warm_time += metrics["warm_sec"]
        summary_results.append(metrics)

        print(f"  -> Cold Run:    {metrics['cold_sec']:.2f}s")
        print(f"  -> Warm Run:    {metrics['warm_sec']:.2f}s (Speedup: {metrics['speedup']:.1f}x)")
        print(f"  -> Columns:     {metrics['columns']}")
        print(f"  -> LSH Items:   {metrics['lsh_literals']}")
        print(f"  -> Vector Docs: {metrics['vector_docs']}")
        print(f"  -> Top Column:  {metrics['sample_col']}")
        print(f"  -> Real LLM Summary sample: \"{metrics['sample_minimal']}\"")
        print(f"  -> Cache Dir:   {metrics['cache_dir']}")
        print()

    print("All real business verifications finished successfully.")
    print(f"Total Cold Time: {total_cold_time:.2f}s | Total Warm Time: {total_warm_time:.2f}s")


if __name__ == "__main__":
    main()
