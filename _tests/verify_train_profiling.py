"""Batch verification and benchmarking script for RawDatabaseProfiler across tmp_dev/train."""

import json
from pathlib import Path
import time

from zaic_2026_4.profiling.raw_profiler import RawDatabaseProfiler


def verify_train_profiling() -> None:
    """Execute end-to-end profiling on all 30 databases in tmp_dev/train."""
    train_dir = Path("tmp_dev/train")
    if not train_dir.is_dir():
        print(f"Error: {train_dir} does not exist.")
        return

    db_files = sorted(
        [
            p
            for p in train_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".db", ".sqlite", ".sqlite3")
        ]
    )
    if not db_files:
        print(f"No database files found in {train_dir}.")
        return

    output_dir = Path("tmp_dev/profiles")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(db_files)} SQLite databases in {train_dir}.")
    print("Beginning profiling with skip_partition_tables_if_view_exists=True, cross_table_only=True ...\n")

    t_global_start = time.perf_counter()
    total_entities_profiled = 0
    total_rows_profiled = 0
    total_joins_discovered = 0
    total_date_columns = 0
    total_currency_columns = 0

    results = []

    for idx, db_path in enumerate(db_files, 1):
        t0 = time.perf_counter()
        profiler = RawDatabaseProfiler(
            db_path=db_path,
            include_views=True,
            skip_partition_tables_if_view_exists=True,
            cross_table_only=True,
            jaccard_threshold=0.7,
            exclude_continuous_numerics=True,
        )
        profile = profiler.profile_database()
        elapsed = time.perf_counter() - t0

        entity_count = len(profile["tables"])
        db_rows = sum(t["row_count"] for t in profile["tables"].values())
        join_count = len(profile["undocumented_joins"])

        date_cols = sum(
            1
            for t in profile["tables"].values()
            for c in t["columns"].values()
            if c["date_subtype"] is not None
        )
        curr_cols = sum(
            1
            for t in profile["tables"].values()
            for c in t["columns"].values()
            if c["currency_subtype"] is not None
        )

        total_entities_profiled += entity_count
        total_rows_profiled += db_rows
        total_joins_discovered += join_count
        total_date_columns += date_cols
        total_currency_columns += curr_cols

        # Persist profile to JSON
        out_json = output_dir / f"{db_path.stem}_profile.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

        results.append({
            "biz": db_path.stem,
            "entities": entity_count,
            "rows": db_rows,
            "joins": join_count,
            "date_cols": date_cols,
            "curr_cols": curr_cols,
            "time_sec": round(elapsed, 3),
        })

        sample_join_str = ""
        if profile["undocumented_joins"]:
            top = profile["undocumented_joins"][0]
            sample_join_str = f" | Top join: {top['col1']} <-> {top['col2']} ({top['resemblance']:.2f})"

        print(
            f"[{idx:02d}/{len(db_files)}] {db_path.name:<12} | "
            f"Entities: {entity_count:2d} | Rows: {db_rows:8d} | "
            f"Dates: {date_cols:2d} | Currencies: {curr_cols:2d} | "
            f"Joins: {join_count:2d} | Time: {elapsed:.3f}s{sample_join_str}"
        )

    total_time = time.perf_counter() - t_global_start
    avg_time = total_time / len(db_files)

    print("\nBenchmark Summary:")
    print(f"Total databases profiled: {len(db_files)}")
    print(f"Total entities profiled:  {total_entities_profiled}")
    print(f"Total rows scanned:       {total_rows_profiled:,}")
    print(f"Total date columns:       {total_date_columns}")
    print(f"Total currency columns:   {total_currency_columns}")
    print(f"Total joins discovered:   {total_joins_discovered}")
    print(f"Total wall-clock time:    {total_time:.2f}s (Average: {avg_time:.3f}s/database)")
    print(f"Throughput:               {total_rows_profiled / total_time:,.0f} rows/sec")
    print(f"Profiles written to:      {output_dir}")


if __name__ == "__main__":
    verify_train_profiling()
