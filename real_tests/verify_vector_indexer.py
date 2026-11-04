"""Real data verification and benchmarking script for SemanticVectorIndexer using real BGE-M3.

Loads real business profile for biz_101, rulebook.md, and README.md.
Extracts 3 actual test questions from questions.jsonl for business_id == 'biz_101'
and performs semantic vector retrieval using vLLM (BAAI/bge-m3).
"""

import json
import os
from pathlib import Path
import time
from typing import Any

from zaic_2026_4.profiling.vector_indexer import SemanticVectorIndexer


def load_biz_101_questions(questions_path: Path, limit: int = 3) -> list[dict[str, Any]]:
    """Extract sample questions for biz_101 from questions.jsonl."""
    questions: list[dict[str, Any]] = []
    if questions_path.is_file():
        with open(questions_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("business_id") == "biz_101":
                    questions.append(item)
                    if len(questions) >= limit:
                        break

    if not questions:
        # Fallback default questions if questions.jsonl is not present
        questions = [
            {"id": "q_0c8f3e07768f", "question": "e so don hang quy 4/2025 la bao nhiu vay"},
            {"id": "q_0cb7c9b93048", "question": "Cho tôi biết số đơn hôm qua."},
            {"id": "q_15fd3e0d0363", "question": "thang 2/2026 mat hang Thuoc la Vinataba thu ve bao nhieu tien"},
        ]
    return questions


def verify_vector_indexer_real_bge_m3() -> None:
    """Benchmark vector indexing and semantic retrieval using real BGE-M3 model."""
    profiles_dir = Path("tmp_dev/profiles")
    rulebook_path = Path("dataset/train/rulebook.md")
    questions_path = Path("dataset/train/questions.jsonl")
    biz_101_readme_path = Path("dataset/train/data/biz_101/README.md")
    biz_101_profile_path = profiles_dir / "biz_101_profile.json"

    print("======================================================================")
    print("Verifying SemanticVectorIndexer with Real BGE-M3 Embedding")
    print("======================================================================")

    # 1. Load inputs
    rulebook_text = ""
    if rulebook_path.is_file():
        rulebook_text = rulebook_path.read_text(encoding="utf-8")
        print(f"Loaded rulebook: {len(rulebook_text)} characters.")

    readme_text = ""
    if biz_101_readme_path.is_file():
        readme_text = biz_101_readme_path.read_text(encoding="utf-8")
        print(f"Loaded biz_101 README: {len(readme_text)} characters.")

    biz_101_data = {}
    if biz_101_profile_path.is_file():
        with open(biz_101_profile_path, "r", encoding="utf-8") as f:
            biz_101_data = json.load(f)
        print(f"Loaded biz_101 profile: {len(biz_101_data.get('tables', {}))} tables.")
    else:
        print(f"Profile {biz_101_profile_path} not found. Using raw table schema.")

    # 2. Initialize SemanticVectorIndexer with real BGE-M3
    model_name = os.environ.get("EMBEDDING_MODEL_PATH", "BAAI/bge-m3")
    print(f"\nInitializing SemanticVectorIndexer with model '{model_name}'...")
    t_init = time.perf_counter()
    indexer = SemanticVectorIndexer(
        model_path=model_name,
        max_length=4096,
    )
    init_elapsed = time.perf_counter() - t_init
    print(f"Model initialized in {init_elapsed:.2f}s (dimension={indexer.dimension}).")

    # 3. Build index
    print("\nBuilding FAISS vector index (Hierarchical Table-Row & Buffer Coalescing)...")
    t_build = time.perf_counter()
    total_docs = indexer.build_index(
        processed_metadata=biz_101_data,
        rulebook_text=rulebook_text,
        readme_text=readme_text,
    )
    build_elapsed = time.perf_counter() - t_build
    print(
        f"Built index with {total_docs} documents in {build_elapsed:.2f}s "
        f"({(build_elapsed / max(1, total_docs)) * 1000:.2f} ms/doc)."
    )

    # 4. Extract 3 questions from questions.jsonl for biz_101
    sample_questions = load_biz_101_questions(questions_path, limit=3)
    print(f"\nExtracted {len(sample_questions)} test questions for biz_101:")

    latencies = []
    for q_item in sample_questions:
        q_id = q_item["id"]
        q_text = q_item["question"]

        t_search = time.perf_counter()
        col_hits = indexer.search_columns(q_text, top_k=3)
        rule_hits = indexer.search_rules(q_text, top_k=3)
        elapsed = time.perf_counter() - t_search
        latencies.append(elapsed)

        print(f"\n----------------------------------------------------------------------")
        print(f"Question [{q_id}]: \"{q_text}\"")
        print(f"Retrieval Latency: {elapsed * 1000:.2f} ms")

        print("  Top Matching Columns:")
        for rank, h in enumerate(col_hits, start=1):
            print(f"    {rank}. [{h['score']:.4f}] {h['key']}")

        print("  Top Matching Rules:")
        for rank, r in enumerate(rule_hits, start=1):
            snippet = r["content"][:100].replace("\n", " ")
            print(f"    {rank}. [{r['score']:.4f}] ({r['key']}) {snippet}...")

    avg_latency = (sum(latencies) / len(latencies)) * 1000
    print(f"\n======================================================================")
    print(f"Average Retrieval Latency (real BGE-M3): {avg_latency:.2f} ms/query")

    # 5. Verify index persistence
    cache_dir = Path("/tmp/cache_biz_101_bge_m3")
    print(f"\nVerifying index persistence at {cache_dir}...")
    indexer.save(cache_dir)
    loaded = SemanticVectorIndexer.load(cache_dir, model_path=model_name, embedder=indexer.embedder)
    assert loaded.index.ntotal == indexer.index.ntotal
    assert loaded.dimension == indexer.dimension
    print(f"Successfully verified save/load round-trip! Total docs preserved: {loaded.index.ntotal}")


if __name__ == "__main__":
    verify_vector_indexer_real_bge_m3()
