"""CLI interface for running Task-Aligned Schema Linking on business questions."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from zaic_2026_4.ingestion import resolve_split_paths
from zaic_2026_4.profiling.cli import _resolve_biz_dir
from zaic_2026_4.profiling.profile_manager import ProfilingOrchestrator
from zaic_2026_4.profiling.profile_summarizer import create_engine
from zaic_2026_4.profiling.vector_indexer import MockEmbedder
from zaic_2026_4.schema_linking.task_aligned_linker import TaskAlignedSchemaLinker


def _matches_biz_id(target_id: str, candidate_id: str) -> bool:
    """Check if candidate business ID matches target, accounting for 'biz_' prefix and numeric IDs."""
    if target_id == candidate_id:
        return True
    t_clean = target_id.lower().removeprefix("biz_").lstrip("0")
    c_clean = candidate_id.lower().removeprefix("biz_").lstrip("0")
    return t_clean == c_clean


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for zaic-schema-link."""
    parser = argparse.ArgumentParser(
        description="Run Task-Aligned Schema Linking on natural-language questions."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dataset/train",
        help="Name or path of dataset split (e.g. 'train', 'public_test', 'private_test'). Default: 'dataset/train'.",
    )
    parser.add_argument(
        "--biz-id",
        "--biz_id",
        dest="biz_id",
        type=str,
        required=True,
        help="Identifier of the business to link (e.g. '101' or 'biz_101').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target base directory for profiling caches and linking artifacts (defaults to /tmp/cache_<biz_id>).",
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
        "--cases",
        "--case",
        type=str,
        default="abcde",
        dest="cases",
        help="Context variant cases to run (e.g. 'abcde', 'abce', 'abe'). Default: 'abcde'.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum re-prompting retries on literal mismatch (default: 2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic question sampling (default: None).",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Anchor date (YYYY-MM-DD) used for temporal resolution when the question file omits it.",
    )
    return parser


def main() -> None:
    """CLI entry point for Task-Aligned Schema Linking."""
    parser = build_parser()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    split_paths = resolve_split_paths(args.split, require_questions=True)
    rulebook_path = split_paths["rulebook_path"]
    data_dir = split_paths["data_dir"]
    questions_path = split_paths["questions_path"]

    biz_dir = _resolve_biz_dir(data_dir, args.biz_id)
    if biz_dir is None or not biz_dir.is_dir():
        print(f"Error: Business directory not found for '{args.biz_id}' in {data_dir}", file=sys.stderr)
        sys.exit(1)

    if not questions_path or not questions_path.is_file():
        print(f"Error: questions.jsonl not found in split '{args.split}'", file=sys.stderr)
        sys.exit(1)

    matching_questions: list[dict[str, Any]] = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            q_biz = item.get("business_id", "")
            if _matches_biz_id(args.biz_id, q_biz) or _matches_biz_id(biz_dir.name, q_biz):
                matching_questions.append(item)

    if not matching_questions:
        print(
            f"Error: No questions found for business '{args.biz_id}' ({biz_dir.name}) in {questions_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    selected_q = random.choice(matching_questions)
    question_text = selected_q["question"]
    question_id = selected_q.get("id", "unknown")
    as_of_date = args.as_of_date or selected_q.get("as_of_date")
    print(
        f"Selected question [{question_id}] for {biz_dir.name}: {question_text}",
        file=sys.stderr,
    )

    # 1. Initialize LLM engine
    llm = create_engine(
        local=args.local,
        mock=args.mock,
    )
    embedder = MockEmbedder() if args.mock else None
    vector_model_path = "mock" if args.mock else "BAAI/bge-m3"

    # 2. Determine cache directory and load/run profiling pipeline
    if args.output_dir:
        out_p = Path(args.output_dir).resolve()
        if (out_p / "metadata_profile.json").is_file():
            b_cache = out_p
        elif (out_p / f"cache_{biz_dir.name}").is_dir() or (
            out_p / f"cache_{biz_dir.name}" / "metadata_profile.json"
        ).is_file():
            b_cache = out_p / f"cache_{biz_dir.name}"
        else:
            b_cache = out_p / f"cache_{biz_dir.name}"
    else:
        b_cache = Path(f"/tmp/cache_{biz_dir.name}")

    orchestrator = ProfilingOrchestrator(
        business_dir=biz_dir,
        rulebook_path=rulebook_path,
        llm_engine=llm,
        embedder=embedder,
        vector_model_path=vector_model_path,
        cache_dir=b_cache,
        calendar_path=split_paths["calendar_path"],
    )
    print(f"Using cache directory: {orchestrator.cache_dir}", file=sys.stderr)
    pipeline_res = orchestrator.run_pipeline()
    processed_metadata = pipeline_res["metadata"]
    lsh_indexer = pipeline_res["lsh_indexer"]
    vector_indexer = pipeline_res["vector_indexer"]

    # Load raw profile if cached
    raw_profile = None
    raw_profile_path = orchestrator.cache_dir / "raw_profile.json"
    if raw_profile_path.is_file():
        try:
            with open(raw_profile_path, "r", encoding="utf-8") as f:
                raw_profile = json.load(f)
        except (OSError, json.JSONDecodeError):
            raw_profile = None

    # 3. Execute Task-Aligned Schema Linking
    linker = TaskAlignedSchemaLinker(
        llm_engine=llm,
        max_retries=args.max_retries,
    )

    t0 = time.perf_counter()
    result = linker.link_schema(
        question=question_text,
        processed_metadata=processed_metadata,
        lsh_indexer=lsh_indexer,
        vector_indexer=vector_indexer,
        raw_profile=raw_profile,
        cases=args.cases,
        cache_dir=orchestrator.cache_dir,
        question_id=question_id,
        as_of_date=as_of_date,
    )
    elapsed = time.perf_counter() - t0

    out_data = result.to_dict(question=question_text)
    out_data["question_id"] = question_id
    out_data["business_id"] = biz_dir.name
    out_data["elapsed_seconds"] = round(elapsed, 3)
    print(json.dumps(out_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
