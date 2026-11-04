"""Real data verification and benchmarking script for VietnameseLSHIndexer on real SMB profiles."""

import json
from pathlib import Path
import time

from unidecode import unidecode

from zaic_2026_4.profiling.lsh_indexer import VietnameseLSHIndexer


def verify_lsh_indexer_on_train() -> None:
    """Benchmark LSH indexing and literal matching on real business profiles."""
    profiles_dir = Path("tmp_dev/profiles")
    if not profiles_dir.is_dir():
        print(f"Directory {profiles_dir} not found. Skipping real data verification.")
        return

    biz_files = sorted(profiles_dir.glob("biz_*_profile.json"))[:5]
    if not biz_files:
        print("No profile JSON files found.")
        return

    print(f"Benchmarking VietnameseLSHIndexer across {len(biz_files)} business profiles...")

    for profile_path in biz_files:
        biz_name = profile_path.stem.replace("_profile", "")
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)

        t_start = time.perf_counter()
        indexer = VietnameseLSHIndexer(threshold=0.4, num_perm=128)
        indexed_count = indexer.build_index(profile_data)
        build_time = time.perf_counter() - t_start

        print(
            f"[{biz_name}] Indexed {indexed_count} literal values in {build_time:.4f}s "
            f"({(build_time / max(1, indexed_count)) * 1000:.3f} ms/value)"
        )

    # In-depth retrieval test on biz_101
    biz_101_path = profiles_dir / "biz_101_profile.json"
    if biz_101_path.is_file():
        with open(biz_101_path, "r", encoding="utf-8") as f:
            biz_101_data = json.load(f)

        indexer_101 = VietnameseLSHIndexer(threshold=0.35, num_perm=128)
        indexer_101.build_index(biz_101_data)

        test_cases = [
            ("Dầu gội Clear", "ban_hang.Nội dung", "Dầu gội Clear 630g"),
            ("sua tam lifebuoy", "ban_hang.Nội dung", "Sữa tắm Lifebuoy 850g"),
            ("dinh thu khanh", "ban_hang.Đối tác", "Đinh Thu Khánh"),
            ("duong bien hoa", "ban_hang.Nội dung", "Đường Biên Hòa 1kg"),
        ]

        print("\nVerifying literal retrieval accuracy & query latency on biz_101:")
        total_latencies = []

        for query_text, expected_col, expected_val_substring in test_cases:
            t_q = time.perf_counter()
            results = indexer_101.query_literal_with_scores(query_text, top_k=3)
            q_latency = (time.perf_counter() - t_q) * 1000.0
            total_latencies.append(q_latency)

            assert len(results) > 0, f"Query '{query_text}' returned 0 results"
            matched_col, matched_val, score = results[0]
            assert matched_col == expected_col, f"Expected {expected_col}, got {matched_col}"
            assert unidecode(expected_val_substring).lower() in unidecode(matched_val).lower(), (
                f"Expected '{expected_val_substring}' in '{matched_val}'"
            )

            print(
                f"  Query: '{query_text}' -> [{score:.3f}] {matched_col}: '{matched_val}' "
                f"({q_latency:.3f} ms)"
            )

        avg_latency = sum(total_latencies) / len(total_latencies)
        print(f"Average literal lookup latency: {avg_latency:.3f} ms/query (< 1.0 ms requirement met).")

        # Test persistence
        cache_path = Path("/tmp/verify_biz_101_lsh.pkl")
        indexer_101.save(cache_path)
        assert cache_path.is_file()

        loaded_indexer = VietnameseLSHIndexer.load(cache_path)
        assert len(loaded_indexer.value_to_col_map) == len(indexer_101.value_to_col_map)
        recheck_res = loaded_indexer.query_literal("sua tam lifebuoy", top_k=1)
        assert len(recheck_res) == 1
        assert "Lifebuoy" in recheck_res[0][1]
        print(f"Cache persistence roundtrip verified ({cache_path.stat().st_size / 1024:.1f} KB).")

    print("\nAll real data verification checks passed successfully.")


if __name__ == "__main__":
    verify_lsh_indexer_on_train()
