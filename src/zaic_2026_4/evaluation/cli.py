"""CLI entry point for evaluating submission against ground truth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from zaic_2026_4.evaluation.metrics import (
    build_evaluation_frame,
    compute_confusion_matrix,
    compute_overall_metrics,
    compute_per_type_metrics,
    export_details_jsonl,
    load_references,
    load_submission,
    resolve_submission_path,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate submission.jsonl against ground-truth answers per Rulebook §9."
    )
    parser.add_argument(
        "--answers",
        type=str,
        default="dataset/train/answers.jsonl",
        help="Path to ground-truth answers.jsonl (default: dataset/train/answers.jsonl)",
    )
    parser.add_argument(
        "--submission",
        type=str,
        default=None,
        help="Direct path to submission.jsonl file",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Directory containing submission.jsonl (e.g. result/ or result1/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to output detailed per-question evaluation JSONL",
    )
    parser.add_argument(
        "--only-submitted",
        action="store_true",
        help="Evaluate only questions present in submission instead of all reference questions",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for 95%% CI (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap reproducibility (default: 42)",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)

    try:
        sub_path = resolve_submission_path(
            submission=parsed.submission,
            result_dir=parsed.result_dir,
        )
    except (ValueError, FileNotFoundError, NotADirectoryError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    answers_path = Path(parsed.answers)
    if not answers_path.is_file():
        print(f"Error: Answers file not found: {answers_path}", file=sys.stderr)
        return 1

    refs = load_references(answers_path)
    preds, corrupt_lines, duplicate_ids = load_submission(sub_path)

    if corrupt_lines > 0:
        print(
            f"Warning: {corrupt_lines} unparseable lines in submission skipped (counted as abstain).",
            file=sys.stderr,
        )
    if duplicate_ids > 0:
        print(
            f"Warning: {duplicate_ids} duplicate question IDs in submission (last occurrence used).",
            file=sys.stderr,
        )

    # Check for extraneous submission IDs not in ground truth
    extra_ids = set(preds.keys()) - set(refs.keys())
    if extra_ids:
        print(
            f"Warning: {len(extra_ids)} question IDs in submission not found in reference answers.",
            file=sys.stderr,
        )

    df = build_evaluation_frame(refs, preds, only_submitted=parsed.only_submitted)
    overall = compute_overall_metrics(df, n_boot=parsed.n_boot, seed=parsed.seed)
    per_type_df = compute_per_type_metrics(df, n_boot=parsed.n_boot, seed=parsed.seed)
    confusion_df = compute_confusion_matrix(df)

    ci_low, ci_high = overall["ci_95"]
    print(
        f"N={overall['n']}  correct={overall['correct']}  wrong={overall['wrong']}  "
        f"abstain={overall['abstain']} (explicit {overall['explicit_abstain']}, missing {overall['missing']})"
    )
    print(f"PA = {overall['pa']:.4f}  95% CI [{ci_low:.4f}, {ci_high:.4f}]")
    print(
        f"Abstain rate = {overall['abstain_rate']:.2f}%  Precision on answered = {overall['precision_on_answered']:.2f}%\n"
    )

    print("PA by type:")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_hide_dataframe_shape=True):
        print(per_type_df)

    print("\nType confusion (rows = ref, cols = pred):")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_hide_dataframe_shape=True):
        print(confusion_df)

    if parsed.output:
        export_details_jsonl(df, parsed.output)
        print(f"\nDetailed evaluation written to: {parsed.output}")

    return 0
