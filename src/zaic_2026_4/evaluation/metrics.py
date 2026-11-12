"""Metrics computation, data aggregation, and submission parsing for evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from zaic_2026_4.evaluation.scorer import grade_one


def resolve_submission_path(
    submission: str | Path | None = None,
    result_dir: str | Path | None = None,
) -> Path:
    """Resolve submission file path from either --submission or --result-dir."""
    if submission is not None and str(submission).strip():
        sub_path = Path(submission)
        if not sub_path.is_file():
            raise FileNotFoundError(f"Submission file not found: {sub_path}")
        return sub_path

    if result_dir is not None and str(result_dir).strip():
        dir_path = Path(result_dir)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Result directory not found: {dir_path}")
        sub_path = dir_path / "submission.jsonl"
        if not sub_path.is_file():
            raise FileNotFoundError(f"submission.jsonl not found in result directory: {dir_path}")
        return sub_path

    raise ValueError("Either --submission or --result-dir must be provided.")


def load_references(answers_path: str | Path) -> dict[str, dict[str, Any]]:
    """Load ground-truth reference answers keyed by question id."""
    path = Path(answers_path)
    if not path.is_file():
        raise FileNotFoundError(f"Answers file not found: {path}")

    refs: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Corrupt line in answers file at line {line_no}: {err}") from err
            qid = row.get("id")
            if not qid:
                raise ValueError(f"Missing 'id' in answers file at line {line_no}")
            refs[qid] = row
    return refs


def load_submission(submission_path: str | Path) -> tuple[dict[str, dict[str, Any]], int, int]:
    """Load predictions from submission.jsonl.

    Returns:
        (preds, corrupt_lines_count, duplicate_ids_count)
    """
    path = Path(submission_path)
    if not path.is_file():
        raise FileNotFoundError(f"Submission file not found: {path}")

    preds: dict[str, dict[str, Any]] = {}
    corrupt_lines = 0
    duplicate_ids = 0

    with path.open("r", encoding="utf-8") as f:
        for _line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                corrupt_lines += 1
                continue

            if not isinstance(row, dict):
                corrupt_lines += 1
                continue

            qid = row.get("id")
            if not qid or not isinstance(qid, str):
                corrupt_lines += 1
                continue

            if qid in preds:
                duplicate_ids += 1

            preds[qid] = row

    return preds, corrupt_lines, duplicate_ids


def bootstrap_ci(
    scores: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap percentile confidence interval for the mean score."""
    if len(scores) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(scores)
    sample_indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = scores[sample_indices].mean(axis=1)
    low_pct = 100.0 * (alpha / 2.0)
    high_pct = 100.0 * (1.0 - alpha / 2.0)
    low = float(np.percentile(boot_means, low_pct))
    high = float(np.percentile(boot_means, high_pct))
    return low, high


def build_evaluation_frame(
    refs: dict[str, dict[str, Any]],
    preds: dict[str, dict[str, Any]],
    only_submitted: bool = False,
) -> pl.DataFrame:
    """Build evaluation Polars DataFrame comparing predictions against references."""
    ids = [qid for qid in refs if qid in preds] if only_submitted else list(refs.keys())

    rows: list[dict[str, Any]] = []
    for qid in ids:
        ref = refs[qid]
        pred = preds.get(qid)
        verdict, score, reason = grade_one(ref, pred)
        pred_type = pred.get("type", "unknown") if pred is not None else "missing"
        rows.append(
            {
                "id": qid,
                "ref_type": ref.get("type", "unknown"),
                "pred_type": pred_type,
                "verdict": verdict,
                "score": score,
                "reason": reason,
                "ref_answer": json.dumps(ref.get("answer"), ensure_ascii=False),
                "pred_answer": json.dumps(pred.get("answer"), ensure_ascii=False)
                if (pred is not None and "answer" in pred)
                else "null",
            }
        )

    return pl.DataFrame(
        rows,
        schema={
            "id": pl.String,
            "ref_type": pl.String,
            "pred_type": pl.String,
            "verdict": pl.String,
            "score": pl.Int32,
            "reason": pl.String,
            "ref_answer": pl.String,
            "pred_answer": pl.String,
        },
    )


def compute_overall_metrics(
    df: pl.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute overall evaluation metrics including Penalized Accuracy and CI."""
    n = len(df)
    if n == 0:
        return {
            "n": 0,
            "correct": 0,
            "wrong": 0,
            "abstain": 0,
            "explicit_abstain": 0,
            "missing": 0,
            "pa": 0.0,
            "ci_95": (0.0, 0.0),
            "abstain_rate": 0.0,
            "precision_on_answered": 0.0,
        }

    correct = int((df["verdict"] == "correct").sum())
    wrong = int((df["verdict"] == "wrong").sum())
    abstain = int((df["verdict"] == "abstain").sum())
    explicit_abstain = int((df["pred_type"] == "abstain").sum())
    missing = int((df["pred_type"] == "missing").sum())

    scores = df["score"].to_numpy()
    pa = float(scores.mean())
    ci_low, ci_high = bootstrap_ci(scores, n_boot=n_boot, seed=seed)
    abstain_rate = (abstain / n) * 100.0
    answered = correct + wrong
    precision_on_answered = (correct / answered * 100.0) if answered > 0 else 0.0

    return {
        "n": n,
        "correct": correct,
        "wrong": wrong,
        "abstain": abstain,
        "explicit_abstain": explicit_abstain,
        "missing": missing,
        "pa": round(pa, 4),
        "ci_95": (round(ci_low, 4), round(ci_high, 4)),
        "abstain_rate": round(abstain_rate, 2),
        "precision_on_answered": round(precision_on_answered, 2),
    }


def compute_per_type_metrics(
    df: pl.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
) -> pl.DataFrame:
    """Compute per-reference-type metrics (n, correct, wrong, abstain, PA, 95% CI)."""
    if len(df) == 0:
        return pl.DataFrame()

    grouped = (
        df.group_by("ref_type")
        .agg(
            [
                pl.len().alias("n"),
                (pl.col("verdict") == "correct").sum().alias("correct"),
                (pl.col("verdict") == "wrong").sum().alias("wrong"),
                (pl.col("verdict") == "abstain").sum().alias("abstain"),
                pl.col("score").mean().alias("pa"),
            ]
        )
        .sort("n", descending=True)
    )

    ci_lows: list[float] = []
    ci_highs: list[float] = []
    for row in grouped.iter_rows(named=True):
        type_name = row["ref_type"]
        type_scores = df.filter(pl.col("ref_type") == type_name)["score"].to_numpy()
        low, high = bootstrap_ci(type_scores, n_boot=n_boot, seed=seed)
        ci_lows.append(round(low, 4))
        ci_highs.append(round(high, 4))

    return (
        grouped.with_columns(
            [
                pl.Series("ci_low", ci_lows),
                pl.Series("ci_high", ci_highs),
                pl.col("pa").round(4),
            ]
        )
        .with_columns(
            (
                pl.lit("[")
                + pl.col("ci_low").cast(pl.String)
                + pl.lit(", ")
                + pl.col("ci_high").cast(pl.String)
                + pl.lit("]")
            ).alias("ci_95")
        )
        .drop(["ci_low", "ci_high"])
    )


def compute_confusion_matrix(df: pl.DataFrame) -> pl.DataFrame:
    """Compute confusion matrix between reference type and predicted type."""
    if len(df) == 0:
        return pl.DataFrame()

    standard_types = [
        "number",
        "percent",
        "string",
        "date",
        "boolean",
        "list",
        "table",
        "unanswerable",
        "abstain",
        "missing",
    ]

    grouped = df.group_by(["ref_type", "pred_type"]).len()
    pivot = grouped.pivot(on="pred_type", index="ref_type").fill_null(0)

    # Ensure all standard types are present as columns
    for col in standard_types:
        if col not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(0, dtype=pl.UInt32).alias(col))

    ordered_cols = ["ref_type"] + [c for c in standard_types if c in pivot.columns] + [
        c for c in pivot.columns if c not in standard_types and c != "ref_type"
    ]
    return pivot.select(ordered_cols).sort("ref_type")


def export_details_jsonl(df: pl.DataFrame, output_path: str | Path) -> None:
    """Export detailed question-level evaluation results to a JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in df.iter_rows(named=True):
            item = {
                "id": row["id"],
                "ref_type": row["ref_type"],
                "pred_type": row["pred_type"],
                "verdict": row["verdict"],
                "score": row["score"],
                "reason": row["reason"],
                "ref_answer": json.loads(row["ref_answer"]),
                "pred_answer": json.loads(row["pred_answer"]),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
