"""CLI entry point for executing the ZAIC 2026-4 Submission Pipeline."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from zaic_2026_4.constants import (
    DEFAULT_CANDIDATE_SEEDS,
    DEFAULT_LOGS_DIR,
    DEFAULT_SQL_MAX_RETRIES,
    DEFAULT_SUBMISSION_PATH,
)
from zaic_2026_4.profiling.profile_summarizer import create_engine
from zaic_2026_4.profiling.vector_indexer import MockEmbedder
from zaic_2026_4.submission.pipeline import SubmissionPipeline


def main() -> None:
    """CLI execution function for zaic-submit."""
    parser = argparse.ArgumentParser(
        description="Run ZAIC 2026-4 Submission Pipeline on questions.jsonl."
    )
    parser.add_argument(
        "--questions",
        type=str,
        required=True,
        help="Path to questions.jsonl file.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to directory containing business subfolders (e.g. dataset/train/data).",
    )
    parser.add_argument(
        "--rulebook",
        type=str,
        default="dataset/train/rulebook.md",
        help="Path to dataset split rulebook.md.",
    )
    parser.add_argument(
        "--rulebook-mode",
        type=str,
        choices=["hybrid", "full", "retrieval"],
        default="hybrid",
        help="Rulebook context injection mode (default: hybrid).",
    )
    parser.add_argument(
        "--calendar",
        type=str,
        default=None,
        help="Path to calendar.csv file (auto-discovered if omitted).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_SUBMISSION_PATH),
        help="Output path for submission.jsonl.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(DEFAULT_LOGS_DIR),
        help="Directory to write per-question diagnostic trace logs.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/tmp",
        help="Base directory for temporary profiling and database caches.",
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
        help="Custom LLM model name or path.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Explicit GPU memory utilization fraction override (e.g. 0.70 or 0.60).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_CANDIDATE_SEEDS),
        help="Randomization seeds for candidate diversity (default: 42 100 2026).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_SQL_MAX_RETRIES,
        help="Max self-correction attempts per candidate query (default: 3).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to process (for testing).",
    )

    args = parser.parse_args()

    questions_path = Path(args.questions).resolve()
    data_dir = Path(args.data_dir).resolve()
    rulebook_path = Path(args.rulebook).resolve()

    if not questions_path.is_file():
        print(f"Error: Questions file not found: {questions_path}", file=sys.stderr)
        sys.exit(1)

    if not data_dir.is_dir():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # 1. Initialize LLM engine and embedder
    llm = create_engine(
        local=args.local,
        model=args.model,
        mock=args.mock,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    embedder = MockEmbedder() if args.mock else None
    vector_model_path = "mock" if args.mock else "BAAI/bge-m3"

    # 2. Initialize and run Submission Pipeline
    pipeline = SubmissionPipeline(
        data_dir=data_dir,
        rulebook_path=rulebook_path,
        llm_engine=llm,
        embedder=embedder,
        vector_model_path=vector_model_path,
        calendar_path=args.calendar,
        output_path=args.output,
        logs_dir=args.log_dir,
        cache_base_dir=args.cache_dir,
        seeds=tuple(args.seeds),
        max_retries=args.max_retries,
        rulebook_mode=args.rulebook_mode,
    )

    t0 = time.perf_counter()
    submissions = pipeline.run(questions_path=questions_path, limit=args.limit)
    total_time = time.perf_counter() - t0

    print(
        f"Completed processing {len(submissions)} questions in {total_time:.2f}s "
        f"(average {total_time/max(1, len(submissions)):.2f}s/q). "
        f"Output written to: {pipeline.output_path}"
    )


if __name__ == "__main__":
    main()
