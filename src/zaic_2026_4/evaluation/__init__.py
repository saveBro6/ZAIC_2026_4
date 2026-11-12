"""Evaluation package for ZAIC 2026."""

from __future__ import annotations

from zaic_2026_4.evaluation.cli import main as cli_main
from zaic_2026_4.evaluation.metrics import (
    bootstrap_ci,
    build_evaluation_frame,
    compute_confusion_matrix,
    compute_overall_metrics,
    compute_per_type_metrics,
    export_details_jsonl,
    load_references,
    load_submission,
    resolve_submission_path,
)
from zaic_2026_4.evaluation.scorer import (
    FLOAT_EPS,
    grade_one,
    is_json_number,
    match_boolean,
    match_cell,
    match_date,
    match_list,
    match_number,
    match_percent,
    match_string,
    match_table,
    normalize_string,
)

__all__ = [
    "FLOAT_EPS",
    "bootstrap_ci",
    "build_evaluation_frame",
    "cli_main",
    "compute_confusion_matrix",
    "compute_overall_metrics",
    "compute_per_type_metrics",
    "export_details_jsonl",
    "grade_one",
    "is_json_number",
    "load_references",
    "load_submission",
    "match_boolean",
    "match_cell",
    "match_date",
    "match_list",
    "match_number",
    "match_percent",
    "match_string",
    "match_table",
    "normalize_string",
    "resolve_submission_path",
]
