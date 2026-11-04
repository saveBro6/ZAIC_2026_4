"""Real data verification and benchmarking script for SemanticVectorIndexer on SMB profiles."""

import json
from pathlib import Path
import time

from zaic_2026_4.profiling.vector_indexer import MockEmbedder, SemanticVectorIndexer


def verify_vector_indexer_on_train() -> None:
    """Benchmark vector indexing and semantic retrieval on real business profiles."""
    profiles_dir = Path("tmp_dev/profiles")
    rulebook_path = Path("dataset/train/rulebook.md")

    rulebook_text = ""
    if rulebook_path.is_file():
        rulebook_text = rulebook_path.read_text(encoding="utf-8")
        print(f"Loaded train rulebook: {len(rulebook_text)} characters.")
    else:
        print(f"Rulebook {rulebook_path} not found. Continuing with empty rulebook.")

    if not profiles_dir.is_dir():
        print(f"Directory {profiles_dir} not found. Skipping real profile verification.")
        return

    biz_files = sorted(profiles_dir.glob("biz_*_profile.json"))[:5]
    if not biz_files:
        print("No profile JSON files found.")
        return

    print(f"\nBenchmarking SemanticVectorIndexer across {len(biz_files)} business profiles...")

    for profile_path in biz_files:
        biz_name = profile_path.stem.replace("_profile", "")
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)

        t_start = time.perf_counter()
        indexer = SemanticVectorIndexer(embedder=MockEmbedder(dimension=1024))
        indexed_count = indexer.build_index(
            processed_metadata=profile_data,
            rulebook_text=rulebook_text,
        )
        build_time = time.perf_counter() - t_start

        print(
            f"[{biz_name}] Indexed {indexed_count} schema & rule items in {build_time:.4f}s "
            f"({(build_time / max(1, indexed_count)) * 1000:.3f} ms/doc)"
        )

    # In-depth retrieval test on biz_101
    biz_101_path = profiles_dir / "biz_101_profile.json"
    if biz_101_path.is_file():
        with open(biz_101_path, "r", encoding="utf-8") as f:
            biz_101_data = json.load(f)

        readme_path = Path("dataset/train/data/biz_101/README.md")
        readme_text = readme_path.read_text(encoding="utf-8") if readme_path.is_file() else ""

        indexer_101 = SemanticVectorIndexer(embedder=MockEmbedder(dimension=1024))
        total_docs = indexer_101.build_index(
            processed_metadata=biz_101_data,
            rulebook_text=rulebook_text,
            readme_text=readme_text,
        )
        print(f"\n[biz_101] Total indexed documents: {total_docs}")

        test_queries = [
            "doanh thu bán hàng tháng này",
            "mặt hàng nào bán chạy nhất",
            "khách hàng còn nợ bao nhiêu tiền",
            "thời gian tạo đơn hàng",
        ]

        print("\nVerifying semantic schema retrieval accuracy & query latency on biz_101:")
        latencies = []

        for q in test_queries:
            t0 = time.perf_counter()
            col_hits = indexer_101.search_columns(q, top_k=3)
            rule_hits = indexer_101.search_rules(q, top_k=2)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)

            print(f"\nQuery: '{q}' (latency: {elapsed * 1000:.2f}ms)")
            print("  Top Columns:")
            for h in col_hits:
                print(f"    - [{h['score']:.3f}] {h['key']}")
            print("  Top Rules:")
            for r in rule_hits:
                snippet = r['content'][:60].replace('\n', ' ')
                print(f"    - [{r['score']:.3f}] ({r['key']}) {snippet}...")

        avg_lat = (sum(latencies) / len(latencies)) * 1000
        print(f"\nAverage retrieval latency: {avg_lat:.2f} ms/query")

        # Persistence verification
        cache_dir = Path("/tmp/cache_biz_101_vector")
        indexer_101.save(cache_dir)
        loaded = SemanticVectorIndexer.load(cache_dir, embedder=MockEmbedder(dimension=1024))
        assert loaded.index.ntotal == indexer_101.index.ntotal
        print(f"Successfully verified save/load round-trip at {cache_dir}")


if __name__ == "__main__":
    verify_vector_indexer_on_train()
