#!/usr/bin/env python3
"""Offline replay verification script for P4: vote and format over cached SQLite databases.

Compares predictions before and after P4 against ground-truth answers using evaluation.scorer.
Also computes a reference comparison against a hypothetical strict policy requiring >= 2 agreeing candidates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure 'src' is in sys.path
_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from zaic_2026_4.evaluation.scorer import grade_one
from zaic_2026_4.submission.models import CandidateQuery
from zaic_2026_4.submission.result_formatter import ResultFormatter
from zaic_2026_4.submission.voting_executor import VotingExecutor


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    logs_dir = repo_root / "content" / "result" / "logs"
    answers_file = repo_root / "dataset" / "train" / "answers.jsonl"

    if not logs_dir.is_dir():
        print(f"Error: logs directory not found: {logs_dir}")
        sys.exit(1)

    if not answers_file.is_file():
        print(f"Error: ground-truth answers file not found: {answers_file}")
        sys.exit(1)

    db_map = {
        "biz_105": repo_root / "tmp" / "cache_biz_105" / "biz_105.db",
        "biz_112": repo_root / "tmp" / "cache_biz_112" / "biz_112.db",
        "biz_120": repo_root / "tmp" / "cache_biz_120" / "biz_120.db",
        "biz_124": repo_root / "tmp" / "cache_biz_124" / "biz_124.db",
        "biz_128": repo_root / "tmp" / "cache_biz_128" / "biz_128.db",
    }

    gt_map: dict[str, dict] = {}
    with open(answers_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                gt_map[item["id"]] = item

    log_files = sorted(logs_dir.glob("q_*.json"))
    if not log_files:
        print(f"No diagnostic logs found in {logs_dir}")
        sys.exit(1)

    scores_before: list[int] = []
    scores_after: list[int] = []
    scores_ref: list[int] = []

    print(f"{'Question ID':<16} | {'GT':<12} | {'Before P4':<15} | {'After P4':<25} | {'Ref (>=2 agreeing)':<20}")
    print("-" * 100)

    for log_path in log_files:
        with open(log_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)

        qid = log_data["question_id"]
        biz_id = log_data["business_id"]
        ref_gt = gt_map.get(qid)
        if not ref_gt:
            continue

        db_path = db_map.get(biz_id)
        if not db_path or not db_path.is_file():
            print(f"Warning: SQLite DB for {biz_id} not found at {db_path}")
            continue

        # 1. Before P4
        old_pred = {
            "id": qid,
            "type": log_data["final_predicted_type"],
            "answer": log_data["final_predicted_answer"],
        }
        _, s_before, _ = grade_one(ref_gt, old_pred)
        scores_before.append(s_before)

        # 2. Replay with P4 VotingExecutor and ResultFormatter
        executor = VotingExecutor(db_path=db_path)
        candidates = []
        for c_dict in log_data["candidates"]:
            cand = CandidateQuery(
                candidate_id=c_dict["candidate_id"],
                seed=c_dict["seed"],
                final_sql=c_dict["final_sql"],
                predicted_type=c_dict["predicted_type"],
            )
            candidates.append(cand)

        executor.execute_candidates(candidates)
        winner, status, _reason = executor.vote(candidates)

        if winner is None or winner.raw_result is None or winner.predicted_type == "abstain":
            new_pred = {"id": qid, "type": "abstain"}
        elif winner.predicted_type == "unanswerable":
            new_pred = {"id": qid, "type": "unanswerable", "answer": None}
        else:
            ans_t, ans_v = ResultFormatter.format_result(
                winner.raw_result,
                winner.predicted_type,
                raw_columns=winner.result_columns,
                sql=winner.final_sql,
            )
            new_pred = {"id": qid, "type": ans_t, "answer": ans_v}

        _, s_after, _ = grade_one(ref_gt, new_pred)
        scores_after.append(s_after)

        # 3. Reference policy: requires >= 2 agreeing candidates
        # If status starts with majority_1_of_ or abstain, it abstains
        if status.startswith(("majority_1_of_", "abstain")) or status == "insufficient_evidence":
            ref_pred = {"id": qid, "type": "abstain"}
        else:
            ref_pred = new_pred
        _, s_ref, _ = grade_one(ref_gt, ref_pred)
        scores_ref.append(s_ref)

        def fmt_ans(pred: dict) -> str:
            if pred["type"] == "abstain":
                return "abstain"
            ans_str = str(pred.get("answer"))
            if len(ans_str) > 15:
                ans_str = ans_str[:12] + "..."
            return f"{pred['type']}:{ans_str}"

        gt_str = str(ref_gt.get("answer")) if ref_gt.get("type") != "abstain" else "abstain"
        if len(gt_str) > 12:
            gt_str = gt_str[:9] + "..."

        print(
            f"{qid:<16} | {gt_str:<12} | "
            f"{fmt_ans(old_pred):<10} ({s_before:+d}) | "
            f"{fmt_ans(new_pred):<13} [{status[:11]}] ({s_after:+d}) | "
            f"{fmt_ans(ref_pred):<14} ({s_ref:+d})"
        )

    print("-" * 100)
    pa_before = sum(scores_before) / len(scores_before) if scores_before else 0.0
    pa_after = sum(scores_after) / len(scores_after) if scores_after else 0.0
    pa_ref = sum(scores_ref) / len(scores_ref) if scores_ref else 0.0

    print(f"PA Before P4:           {pa_before:+.2f} ({sum(scores_before)}/{len(scores_before)})")
    print(f"PA After P4:            {pa_after:+.2f} ({sum(scores_after)}/{len(scores_after)})")
    print(f"PA Reference (>=2 agr): {pa_ref:+.2f} ({sum(scores_ref)}/{len(scores_ref)})")


if __name__ == "__main__":
    main()
