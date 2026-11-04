"""Benchmark verification script for generic calendar profiling on real train dataset.

Uses real BAAI/bge-m3 embedding model (no mock) and real dataset/train/calendar.csv.
Outputs all diagnostic profiles, LSH, and FAISS index artifacts into ./tmp_dev/.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from zaic_2026_4.profiling.lsh_indexer import VietnameseLSHIndexer
from zaic_2026_4.profiling.profile_manager import (
    ProfilingOrchestrator,
    profile_split_calendar,
)
from zaic_2026_4.profiling.profile_summarizer import (
    MockLLMEngine,
    ProfileLLMSummarizer,
)
from zaic_2026_4.profiling.raw_profiler import (
    RawDatabaseProfiler,
    profile_calendar_csv,
)
from zaic_2026_4.profiling.vector_indexer import SemanticVectorIndexer

TRAIN_CALENDAR_PATH = Path("dataset/train/calendar.csv")
TRAIN_BIZ_101_DIR = Path("dataset/train/data/biz_101")
TRAIN_RULEBOOK_PATH = Path("dataset/train/rulebook.md")
TMP_DEV_DIR = Path("tmp_dev/calendar_verify")


def verify_real_calendar_profiling() -> None:
    """Execute end-to-end real calendar verification using real BGE-M3."""
    print("Starting Real Calendar Profiling Verification (No Mock BGE-M3)")

    if not TRAIN_CALENDAR_PATH.is_file():
        raise FileNotFoundError(f"Missing train calendar file: {TRAIN_CALENDAR_PATH}")

    TMP_DEV_DIR.mkdir(parents=True, exist_ok=True)
    raw_profile_out = TMP_DEV_DIR / "calendar_raw_profile.json"
    metadata_out = TMP_DEV_DIR / "calendar_profile.json"
    lsh_out = TMP_DEV_DIR / "calendar_lsh.pkl"
    faiss_dir = TMP_DEV_DIR / "calendar_faiss"

    # 1. Module 1: Raw Profiler on real calendar.csv
    t0_raw = time.perf_counter()
    raw_profile = profile_calendar_csv(TRAIN_CALENDAR_PATH)
    t_raw = time.perf_counter() - t0_raw

    cal_tbl = raw_profile["tables"]["calendar"]
    assert cal_tbl["row_count"] == 1461
    assert len(cal_tbl["columns"]) == 7

    with open(raw_profile_out, "w", encoding="utf-8") as f:
        json.dump(raw_profile, f, ensure_ascii=False, indent=2)

    print(f"[Module 1] Raw profiling completed in {t_raw:.4f}s: {cal_tbl['row_count']} rows, 7 columns.")
    print(f"           Saved raw profile to {raw_profile_out}")

    # 2. Module 2: Profile Summarizer
    t0_sum = time.perf_counter()
    engine = MockLLMEngine()
    summarizer = ProfileLLMSummarizer(engine)
    metadata = summarizer.summarize_calendar(raw_profile)
    t_sum = time.perf_counter() - t0_sum

    assert len(metadata) == 7
    with open(metadata_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[Module 2] Summarization completed in {t_sum:.4f}s: 7 columns summarized.")
    print(f"           Saved metadata profile to {metadata_out}")

    # 3. Module 3: LSH Indexer
    t0_lsh = time.perf_counter()
    lsh = VietnameseLSHIndexer()
    lsh_count = lsh.build_index(metadata)
    lsh.save(lsh_out)
    t_lsh = time.perf_counter() - t0_lsh

    # Verify literal queries against real holidays
    test_holidays = ["Tết", "Giỗ tổ", "2/9", "30/4", "1/5", "Trung thu"]
    resolved_holidays: dict[str, str] = {}
    for h in test_holidays:
        matches = lsh.query_literal(h, top_k=2)
        if matches:
            resolved_holidays[h] = f"{matches[0][0]} -> {matches[0][1]}"

    print(f"[Module 3] LSH indexing completed in {t_lsh:.4f}s: {lsh_count} items indexed.")
    print(f"           Saved LSH index to {lsh_out}")
    print(f"           Sample holiday matches: {resolved_holidays}")
    assert len(resolved_holidays) >= 4, f"Failed to resolve enough holidays: {resolved_holidays}"

    # 4. Module 4: Real BGE-M3 Semantic Vector Indexer (NO MOCK)
    print("[Module 4] Initializing real BAAI/bge-m3 embedding model (FP16)...")
    t0_vec_init = time.perf_counter()
    vector_indexer = SemanticVectorIndexer(
        model_path="BAAI/bge-m3",
        use_fp16=True,
    )
    t_vec_init = time.perf_counter() - t0_vec_init
    print(f"           Model loaded in {t_vec_init:.2f}s (dim={vector_indexer.dimension}).")

    t0_vec_build = time.perf_counter()
    doc_count = vector_indexer.build_index(metadata)
    vector_indexer.save(faiss_dir)
    t_vec_build = time.perf_counter() - t0_vec_build
    print(f"           FAISS index built in {t_vec_build:.4f}s ({doc_count} documents).")
    print(f"           Saved FAISS index to {faiss_dir}")

    # Semantic vector search queries
    benchmark_queries = [
        "ngày nghỉ lễ tết âm lịch",
        "doanh thu các ngày cuối tuần thứ bảy chủ nhật",
        "ngày rằm tháng tám",
    ]

    print("[Module 4] Benchmarking real semantic vector search:")
    for q in benchmark_queries:
        t0_q = time.perf_counter()
        hits = vector_indexer.search_columns(q, top_k=3)
        t_q = (time.perf_counter() - t0_q) * 1000
        top_hit = hits[0] if hits else None
        top_desc = f"{top_hit['col_name']} (score={top_hit['score']:.4f})" if top_hit else "None"
        print(f"           Query: '{q}' [{t_q:.1f}ms] -> Top: {top_desc}")
        assert top_hit is not None
        assert top_hit["score"] > 0.4

    # 5. Module 5: Orchestrator Integration & Merging with biz_101
    print("[Module 5] Verifying ProfilingOrchestrator calendar merging on biz_101...")
    t0_orch = time.perf_counter()
    biz_cache_dir = TMP_DEV_DIR / "cache_biz_101"

    orchestrator = ProfilingOrchestrator(
        business_dir=TRAIN_BIZ_101_DIR,
        rulebook_path=TRAIN_RULEBOOK_PATH,
        llm_engine=engine,
        embedder=vector_indexer.embedder,
        cache_dir=biz_cache_dir,
        calendar_path=TRAIN_CALENDAR_PATH,
        calendar_cache_path=metadata_out,
    )

    pipeline = orchestrator.run_pipeline(force_rebuild=True)
    t_orch = time.perf_counter() - t0_orch
    merged_meta = pipeline["metadata"]

    assert any(k.startswith("calendar.") for k in merged_meta)
    assert any(not k.startswith("calendar.") for k in merged_meta)

    # Test joint LSH retrieval
    biz_match = orchestrator.query_literals("Thuốc lá")
    cal_match = orchestrator.query_literals("Tết")
    print(f"           Orchestration completed in {t_orch:.2f}s ({len(merged_meta)} merged columns).")
    print(f"           Biz literal query 'Thuốc lá' -> {biz_match[:1]}")
    print(f"           Calendar literal query 'Tết' -> {cal_match[:1]}")

    print("All Real Calendar Profiling Verifications PASSED Successfully!")


if __name__ == "__main__":
    verify_real_calendar_profiling()
