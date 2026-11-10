"""Real data verification script for Task-Aligned Schema Linking.

Evaluates schema linking on real train businesses (biz_101, biz_102) with natural
language Vietnamese SMB questions from dataset/train/questions.jsonl.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from zaic_2026_4.profiling.profile_manager import ProfilingOrchestrator
from zaic_2026_4.profiling.profile_summarizer import create_engine
from zaic_2026_4.profiling.vector_indexer import MockEmbedder
from zaic_2026_4.schema_linking.task_aligned_linker import TaskAlignedSchemaLinker

DEFAULT_BIZ_ID = "biz_101"
DEFAULT_QUESTION = "thang nay hang nao ban chay nhat"
DEFAULT_DATA_DIR = Path("dataset/train/data")
DEFAULT_RULEBOOK = Path("dataset/train/rulebook.md")


def verify_schema_linking(
    biz_id: str = DEFAULT_BIZ_ID,
    question: str = DEFAULT_QUESTION,
    data_dir: Path = DEFAULT_DATA_DIR,
    rulebook_path: Path = DEFAULT_RULEBOOK,
    local: bool = True,
    mock: bool = False,
    cache_base_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute real-data schema linking verification."""
    biz_dir = data_dir / biz_id
    if not biz_dir.is_dir():
        raise FileNotFoundError(f"Business directory not found: {biz_dir}")

    cache_dir = (
        cache_base_dir / f"cache_{biz_id}"
        if cache_base_dir
        else Path(f"tmp_dev/cache_verify_{biz_id}")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initializing LLM engine (local={local}, mock={mock})...")
    llm = create_engine(local=local, mock=mock)
    embedder = MockEmbedder() if mock else None
    vector_model_path = "mock" if mock else "BAAI/bge-m3"

    print(f"Loading/profiling business context for {biz_id}...")
    orchestrator = ProfilingOrchestrator(
        business_dir=biz_dir,
        rulebook_path=rulebook_path,
        llm_engine=llm,
        embedder=embedder,
        vector_model_path=vector_model_path,
        cache_dir=cache_dir,
    )
    pipeline_res = orchestrator.run_pipeline(force_rebuild=False)
    processed_metadata = pipeline_res["metadata"]
    lsh_indexer = pipeline_res["lsh_indexer"]
    vector_indexer = pipeline_res["vector_indexer"]

    # Load raw profile if present
    raw_profile = None
    raw_profile_path = cache_dir / "raw_profile.json"
    if raw_profile_path.is_file():
        try:
            with open(raw_profile_path, "r", encoding="utf-8") as f:
                raw_profile = json.load(f)
        except (OSError, json.JSONDecodeError):
            raw_profile = None

    print(f"\nQuestion: {question}")
    print("Running Task-Aligned Schema Linking across 5 context variants...")
    linker = TaskAlignedSchemaLinker(
        llm_engine=llm,
        max_retries=2,
    )

    t0 = time.perf_counter()
    result = linker.link_schema(
        question=question,
        processed_metadata=processed_metadata,
        lsh_indexer=lsh_indexer,
        vector_indexer=vector_indexer,
        raw_profile=raw_profile,
        batch_generate=True,
    )
    elapsed = time.perf_counter() - t0

    print(f"Elapsed: {elapsed:.3f}s | Retries: {result.total_retries}")
    print(f"Harvested Fields ({len(result.all_harvested_fields)}): {sorted(result.all_harvested_fields)}")
    print(f"Preserved Keys ({len(result.preserved_keys)}): {sorted(result.preserved_keys)}")
    print(f"Final Linked Schema ({len(result.final_linked_schema)} columns):")
    for col in sorted(result.final_linked_schema):
        print(f"  • {col}")

    assert len(result.final_linked_schema) > 0, "Final linked schema must not be empty"

    output_data = {
        "biz_id": biz_id,
        "question": question,
        "elapsed_seconds": elapsed,
        "total_retries": result.total_retries,
        "all_harvested_fields": sorted(result.all_harvested_fields),
        "preserved_keys": sorted(result.preserved_keys),
        "final_linked_schema": sorted(result.final_linked_schema),
        "canonical_schema": sorted(result.canonical_schema),
        "variant_traces": [
            {
                "case_name": t.case_name,
                "profile_type": t.profile_type,
                "retry_count": t.retry_count,
                "fields": sorted(t.fields_harvested),
                "literals": sorted(t.literals_extracted),
                "mismatches": t.mismatches_detected,
                "final_sql": t.final_sql,
            }
            for t in result.variant_traces
        ],
    }

    result_file = cache_dir / "schema_linking_result.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved schema linking result to: {result_file}")
    return output_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Task-Aligned Schema Linking on real data.")
    parser.add_argument("--biz-id", type=str, default=DEFAULT_BIZ_ID)
    parser.add_argument("--question", type=str, default=DEFAULT_QUESTION)
    parser.add_argument("--local", action="store_true", default=True)
    parser.add_argument("--mock", action="store_true", default=False)
    args = parser.parse_args()

    verify_schema_linking(
        biz_id=args.biz_id,
        question=args.question,
        local=args.local and not args.mock,
        mock=args.mock,
    )


if __name__ == "__main__":
    main()
