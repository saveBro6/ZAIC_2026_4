"""Verification and benchmarking script for ProfileLLMSummarizer."""

import argparse
import json
from pathlib import Path
import time

from zaic_2026_4.profiling.profile_summarizer import (
    DEFAULT_LOCAL_GPU_UTIL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_PROD_GPU_UTIL,
    DEFAULT_PROD_MODEL,
    ProfileLLMSummarizer,
    create_engine,
)
from zaic_2026_4.profiling.raw_profiler import RawDatabaseProfiler


def parse_args() -> argparse.Namespace:
    """Parse verification script arguments."""
    parser = argparse.ArgumentParser(description="Verify Module 2 ProfileLLMSummarizer.")
    parser.add_argument(
        "--db",
        type=str,
        default="tmp_dev/train/biz_107.db",
        help="Database path to profile and summarize (default: tmp_dev/train/biz_107.db).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=f"Run in local mode with {DEFAULT_LOCAL_MODEL} and gpu_memory_utilization={DEFAULT_LOCAL_GPU_UTIL}.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or checkpoint path override.",
    )
    parser.add_argument(
        "--gpu-mem",
        type=float,
        default=None,
        help="GPU memory utilization override (e.g. 0.60 or 0.95).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic offline MockLLMEngine for rapid testing.",
    )
    return parser.parse_args()


def main() -> None:
    """Run verification pipeline."""
    args = parse_args()
    db_path = Path(args.db)

    if not db_path.is_file():
        print(f"Target database not found: {db_path}. Attempting to locate any database in tmp_dev/train...")
        candidates = list(Path("tmp_dev/train").glob("*.db"))
        if candidates:
            db_path = candidates[0]
            print(f"Using alternative database: {db_path}")
        else:
            print("No database files found to test.")
            return

    # Step 1: Raw Profiling
    t0 = time.perf_counter()
    raw_profiler = RawDatabaseProfiler(db_path=db_path)
    raw_profile = raw_profiler.profile_database()
    t_raw = time.perf_counter() - t0
    print(f"Step 1 - Raw Profiling completed in {t_raw:.3f}s ({len(raw_profile['tables'])} tables).")

    # Step 2: Initialize LLM Engine
    t_eng_start = time.perf_counter()
    engine = create_engine(
        local=args.local,
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        mock=args.mock,
    )
    t_eng = time.perf_counter() - t_eng_start
    print(f"Step 2 - Engine initialized in {t_eng:.3f}s (Mock: {args.mock}, Local: {args.local}).")

    # Step 3: Run ProfileLLMSummarizer
    summarizer = ProfileLLMSummarizer(engine)

    # Read README.md if present
    readme_candidates = [
        db_path.parent / "README.md",
        Path("dataset/train/data") / db_path.stem / "README.md",
    ]
    readme_text = ""
    for c in readme_candidates:
        if c.is_file():
            readme_text = c.read_text(encoding="utf-8")
            break

    t_sum_start = time.perf_counter()
    metadata = summarizer.generate_profiles(
        db_profile=raw_profile,
        business_readme=readme_text,
    )
    t_sum = time.perf_counter() - t_sum_start

    print(f"Step 3 - LLM Summarization completed for {len(metadata)} columns in {t_sum:.3f}s.")
    print(f"Average time per column: {(t_sum / max(1, len(metadata))):.4f}s")

    # Step 4: Display sample column profiles
    print("\nSample Generated Profiles:")
    for col_key, prof in list(metadata.items())[:3]:
        print(f"\n--- Column: {col_key} ---")
        print(f"Data Format: {prof.get('data_format')} | Subtype: {prof.get('date_subtype') or prof.get('currency_subtype')}")
        print(f"Minimal Profile: {prof.get('minimal_profile')}")
        print(f"Maximal Profile: {prof.get('maximal_profile')}")

    # Output verification file
    out_dir = Path("tmp_dev/summaries")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{db_path.stem}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"\nVerification successful! Full summary saved to: {out_path}")


if __name__ == "__main__":
    main()
