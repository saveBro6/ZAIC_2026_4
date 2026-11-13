"""CLI entry point for executing the ZAIC 2026-4 Submission Pipeline."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from zaic_2026_4 import constants
from zaic_2026_4.constants import (
    DEFAULT_CANDIDATE_SEEDS,
    DEFAULT_LOGS_DIR,
    DEFAULT_SQL_MAX_RETRIES,
    DEFAULT_SUBMISSION_PATH,
)
from zaic_2026_4.ingestion import resolve_split_paths
from zaic_2026_4.profiling.profile_summarizer import create_engine
from zaic_2026_4.profiling.vector_indexer import MockEmbedder
from zaic_2026_4.submission.pipeline import SubmissionPipeline


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for zaic-submit."""
    parser = argparse.ArgumentParser(
        description="Run ZAIC 2026-4 Submission Pipeline on dataset split."
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Name or path of dataset split (e.g. 'train', 'public_test', 'private_test').",
    )
    parser.add_argument(
        "--rulebook-mode",
        type=str,
        choices=["hybrid", "full", "retrieval"],
        default="hybrid",
        help="Rulebook context injection mode (default: hybrid).",
    )
    parser.add_argument(
        "--cases",
        "--case",
        type=str,
        default="abcde",
        dest="cases",
        help="Context variant cases to run for schema linking (e.g. 'abcde', 'abce', 'abe'). Default: 'abcde'.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Target directory for submission output and diagnostic logs (default: /result or ./result).",
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
        "--gpu-embed-memory-utilization",
        type=float,
        default=None,
        help="Explicit GPU memory utilization fraction for embedding model BGE-M3 (default: 0.20).",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default=None,
        help="Model quantization format for Generative LLM Engine (e.g. 'fp8', 'awq', 'gptq'). Default: None.",
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
    parser.add_argument(
        "--no-trace",
        action="store_false",
        dest="trace_enabled",
        default=True,
        help="Disable detailed pipeline trace output in cache_dir.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI execution function for zaic-submit."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        paths = resolve_split_paths(args.split)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    questions_path = paths["questions_path"]
    data_dir = paths["data_dir"]
    rulebook_path = paths["rulebook_path"]
    calendar_path = paths["calendar_path"]

    # 1. Initialize LLM engine and embedder
    if args.gpu_embed_memory_utilization is not None:
        constants.DEFAULT_EMBED_GPU_UTIL = args.gpu_embed_memory_utilization

    llm = create_engine(
        local=args.local,
        model=args.model,
        mock=args.mock,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantize=args.quantize,
    )
    embedder = MockEmbedder() if args.mock else None
    vector_model_path = "mock" if args.mock else "BAAI/bge-m3"

    # Determine submission output and log directories
    if args.result_dir:
        res_dir = Path(args.result_dir).resolve()
        output_path = res_dir / "submission.jsonl"
        logs_dir = res_dir / "logs"
    else:
        output_path = DEFAULT_SUBMISSION_PATH
        logs_dir = DEFAULT_LOGS_DIR

    # 2. Initialize and run Submission Pipeline
    pipeline = SubmissionPipeline(
        data_dir=data_dir,
        rulebook_path=rulebook_path,
        llm_engine=llm,
        embedder=embedder,
        vector_model_path=vector_model_path,
        calendar_path=calendar_path,
        output_path=output_path,
        logs_dir=logs_dir,
        cache_base_dir="/tmp",
        seeds=DEFAULT_CANDIDATE_SEEDS,
        max_retries=args.max_retries,
        rulebook_mode=args.rulebook_mode,
        cases=args.cases,
        trace_enabled=args.trace_enabled,
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
