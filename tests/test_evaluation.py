"""Unit and regression tests for evaluation logic and CLI."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np

from zaic_2026_4.evaluation import cli_main
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
    grade_one,
    is_json_number,
    match_boolean,
    match_date,
    match_list,
    match_number,
    match_percent,
    match_string,
    match_table,
    normalize_string,
)


def test_normalize_string():
    # Composed vs decomposed forms must normalize to equal strings
    nfd_text = unicodedata.normalize("NFD", "Tiếng Việt")
    assert normalize_string(nfd_text) == "tiếng việt"

    # Case, punctuation, symbols, whitespace
    assert normalize_string("  Cà   phê sữa (M)  ") == "cà phê sữa m"
    assert normalize_string("Sting dâu 330ml!") == "sting dâu 330ml"
    assert normalize_string("SP_001") == "sp 001"

    # Strict diacritics check: differing diacritics are NOT equal
    assert normalize_string("Sting dâu 330ml") != normalize_string("Sting dầu 330ml")


def test_is_json_number():
    assert is_json_number(100) is True
    assert is_json_number(3.14) is True
    assert is_json_number(0) is True
    assert is_json_number(True) is False
    assert is_json_number(False) is False
    assert is_json_number("100") is False
    assert is_json_number(float("nan")) is False
    assert is_json_number(float("inf")) is False


def test_match_number():
    assert match_number(4820000, 4820000) is True
    assert match_number(18.6, 18.6) is True
    assert match_number(0.1 + 0.2, 0.3) is True
    assert match_number("4820000", 4820000) is False
    assert match_number(True, 1) is False
    assert match_number(4820001, 4820000) is False


def test_match_percent():
    assert match_percent(12.4, 12.4) is True
    assert match_percent(12.444, 12.44) is True
    assert match_percent(12.446, 12.45) is True
    assert match_percent(0.124, 12.4) is False
    assert match_percent(-12.4, 12.4) is False
    assert match_percent("12.4", 12.4) is False


def test_match_date():
    assert match_date("2026-08-15", "2026-08-15") is True
    assert match_date("2025-09", "2025-09") is True
    assert match_date("15/08/2026", "2026-08-15") is False
    assert match_date(20260815, "2026-08-15") is False


def test_match_boolean():
    assert match_boolean(True, True) is True
    assert match_boolean(False, False) is True
    assert match_boolean(1, True) is False
    assert match_boolean("true", True) is False


def test_match_string():
    ref = "Gạch ống 8x8x18 Tuynel"
    aliases = ["SP0009", "Gach ong 8x8x18"]
    assert match_string("Gạch ống 8x8x18 Tuynel", ref, aliases) is True
    assert match_string("gạch ống 8x8x18 tuynel.", ref, aliases) is True
    assert match_string("sp0009", ref, aliases) is True
    assert match_string(["Gạch ống 8x8x18 Tuynel"], ref, aliases) is False
    assert match_string("Khác biệt", ref, aliases) is False


def test_match_list():
    ref = ["Cà phê sữa (M)", "Trà đào cam sả (M)"]
    aliases = [["SP0003"], ["SP0017"]]

    # Exact match and alias match
    assert match_list(["Cà phê sữa (M)", "Trà đào cam sả (M)"], ref, aliases) is True
    assert match_list(["sp0003", "SP0017"], ref, aliases) is True

    # Order mismatch must fail
    assert match_list(["Trà đào cam sả (M)", "Cà phê sữa (M)"], ref, aliases) is False

    # Length mismatch must fail
    assert match_list(["Cà phê sữa (M)"], ref, aliases) is False

    # Numeric list
    num_ref = [10, 20, 30]
    assert match_list([10, 20, 30], num_ref, []) is True
    assert match_list([10, "20", 30], num_ref, []) is False


def test_match_table():
    ref = {
        "columns": ["khach_hang", "so_tien_no"],
        "rows": [["Chị Lan", 3200000], ["Anh Tuấn", 1500000]],
    }
    aliases = [["KH001"], ["KH002"]]

    # Exact rows
    pred1 = {
        "columns": ["khach_hang", "so_tien_no"],
        "rows": [["Chị Lan", 3200000], ["Anh Tuấn", 1500000]],
    }
    assert match_table(pred1, ref, aliases) is True

    # Permuted rows (multiset matching)
    pred2 = {
        "columns": ["khach_hang", "so_tien_no"],
        "rows": [["Anh Tuấn", 1500000], ["kh001", 3200000]],
    }
    assert match_table(pred2, ref, aliases) is True

    # Permuted columns by name
    pred3 = {
        "columns": ["so_tien_no", "khach_hang"],
        "rows": [[1500000, "Anh Tuấn"], [3200000, "Chị Lan"]],
    }
    assert match_table(pred3, ref, aliases) is True

    # Cell format error: numeric cell as formatted string
    pred_bad = {
        "columns": ["khach_hang", "so_tien_no"],
        "rows": [["Chị Lan", "3.200.000đ"], ["Anh Tuấn", 1500000]],
    }
    assert match_table(pred_bad, ref, aliases) is False


def test_grade_one():
    ref_num = {"id": "q1", "type": "number", "answer": 100, "alias": []}
    ref_unans = {"id": "q2", "type": "unanswerable", "answer": None, "alias": []}

    # Missing prediction
    assert grade_one(ref_num, None) == ("abstain", 0, "missing")

    # Explicit abstain
    assert grade_one(ref_num, {"id": "q1", "type": "abstain"}) == ("abstain", 0, "explicit_abstain")

    # Correct number
    assert grade_one(ref_num, {"id": "q1", "type": "number", "answer": 100}) == ("correct", 1, "correct")

    # Wrong value number
    assert grade_one(ref_num, {"id": "q1", "type": "number", "answer": 101}) == ("wrong", -1, "incorrect_value")

    # Type mismatch
    assert grade_one(ref_num, {"id": "q1", "type": "string", "answer": "100"}) == (
        "wrong",
        -1,
        "type_mismatch_string_vs_number",
    )

    # Correct unanswerable
    assert grade_one(ref_unans, {"id": "q2", "type": "unanswerable", "reason": "No data"}) == (
        "correct",
        1,
        "correct_unanswerable",
    )

    # Unanswerable with non-string reason -> counted as empty/abstain
    assert grade_one(ref_unans, {"id": "q2", "type": "unanswerable", "reason": 12345}) == (
        "abstain",
        0,
        "invalid_unanswerable_reason",
    )

    # Reference is unanswerable, but prediction answered number
    assert grade_one(ref_unans, {"id": "q2", "type": "number", "answer": 0}) == ("wrong", -1, "answered_unanswerable")


def test_bootstrap_ci():
    # Constant array -> CI must be exact
    scores = np.ones(100)
    low, high = bootstrap_ci(scores, n_boot=200, seed=42)
    assert low == 1.0
    assert high == 1.0

    # Determinism
    scores_mixed = np.array([1, 1, -1, 0, 1, -1, 0, 1])
    ci1 = bootstrap_ci(scores_mixed, n_boot=200, seed=123)
    ci2 = bootstrap_ci(scores_mixed, n_boot=200, seed=123)
    assert ci1 == ci2


def test_pipeline_and_cli_end_to_end(tmp_path: Path):
    answers_file = tmp_path / "answers.jsonl"
    refs_data = [
        {"id": "q1", "type": "number", "answer": 50, "alias": []},
        {"id": "q2", "type": "string", "answer": "Sản phẩm A", "alias": ["SP01"]},
        {"id": "q3", "type": "percent", "answer": 15.5, "alias": []},
        {"id": "q4", "type": "unanswerable", "answer": None, "alias": []},
    ]
    with answers_file.open("w", encoding="utf-8") as f:
        for r in refs_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    res_dir = tmp_path / "result"
    res_dir.mkdir()
    sub_file = res_dir / "submission.jsonl"
    preds_data = [
        {"id": "q1", "type": "number", "answer": 50},  # correct (+1)
        {"id": "q2", "type": "string", "answer": "sp01"},  # correct (+1)
        {"id": "q3", "type": "percent", "answer": 20.0},  # wrong (-1)
        # q4 missing -> abstain (0)
    ]
    with sub_file.open("w", encoding="utf-8") as f:
        for p in preds_data:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    out_file = tmp_path / "details.jsonl"

    # Test path resolution
    resolved = resolve_submission_path(result_dir=res_dir)
    assert resolved == sub_file

    refs = load_references(answers_file)
    preds, corrupt, dup = load_submission(sub_file)
    assert corrupt == 0
    assert dup == 0

    df = build_evaluation_frame(refs, preds)
    overall = compute_overall_metrics(df, n_boot=100)
    assert overall["n"] == 4
    assert overall["correct"] == 2
    assert overall["wrong"] == 1
    assert overall["abstain"] == 1
    # PA = (1 + 1 - 1 + 0) / 4 = 0.25
    assert overall["pa"] == 0.25

    per_type = compute_per_type_metrics(df, n_boot=100)
    assert len(per_type) == 4

    confusion = compute_confusion_matrix(df)
    assert "ref_type" in confusion.columns

    export_details_jsonl(df, out_file)
    assert out_file.is_file()
    out_lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line]
    assert len(out_lines) == 4

    # Run CLI with --result-dir and --output
    ret = cli_main(
        [
            "--answers",
            str(answers_file),
            "--result-dir",
            str(res_dir),
            "--output",
            str(out_file),
            "--n-boot",
            "100",
        ]
    )
    assert ret == 0


def test_cli_argument_handling(tmp_path: Path):
    # Error when neither --submission nor --result-dir passed
    ret = cli_main([])
    assert ret == 1
