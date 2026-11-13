"""Tests for full pipeline trace logging across profiling, schema linking, SQL generation, and repair."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from zaic_2026_4.common.trace_logger import extract_thinking_trace
from zaic_2026_4.profiling.profile_manager import ProfilingOrchestrator
from zaic_2026_4.profiling.profile_summarizer import MockLLMEngine
from zaic_2026_4.profiling.vector_indexer import MockEmbedder
from zaic_2026_4.schema_linking.task_aligned_linker import TaskAlignedSchemaLinker
from zaic_2026_4.submission.candidate_generator import CandidateGenerator
from zaic_2026_4.submission.models import QuestionItem
from zaic_2026_4.submission.pipeline import SubmissionPipeline
from zaic_2026_4.submission.sql_validator import (
    parse_structured_sql_output_with_reasoning,
)


class _ScriptedTraceLLMEngine:
    """Deterministic scripted LLM engine with last_call_traces telemetry for repair loop tests."""

    def __init__(self, responses: list[str], finish_reason: str = "stop") -> None:
        self._responses = list(responses)
        self._finish_reason = finish_reason
        self._idx = 0
        self.last_call_traces: list[dict[str, Any]] = []

    def generate(self, prompts: list[str] | str, **kwargs: Any) -> list[str] | str:
        is_single = isinstance(prompts, str)
        prompt_list = [prompts] if is_single else list(prompts)
        outputs: list[str] = []
        traces: list[dict[str, Any]] = []
        for p in prompt_list:
            if self._idx < len(self._responses):
                resp = self._responses[self._idx]
                self._idx += 1
            else:
                resp = self._responses[-1] if self._responses else ""
            outputs.append(resp)
            traces.append({
                "prompt": p,
                "raw_output": resp,
                "prompt_tokens": len(p.split()),
                "completion_tokens": max(1, len(resp.split())),
                "finish_reason": self._finish_reason,
                "stop_reason": None,
                "latency_ms": 12.5,
                "batch_latency_ms": 12.5,
                "batch_size": len(prompt_list),
                "sampling_params": {"temperature": kwargs.get("temperature", 0.0)},
            })
        self.last_call_traces = traces
        return outputs[0] if is_single else outputs


def test_parse_structured_sql_output_with_reasoning_and_thinking() -> None:
    raw = json.dumps(
        {
            "reasoning": "<think>Kiểm tra cột doanh_thu</think>Tính tổng doanh thu.",
            "sql": "SELECT SUM(doanh_thu) FROM sales;",
            "type": "number",
        },
        ensure_ascii=False,
    )
    sql, t_type, reasoning, thinking, issues = parse_structured_sql_output_with_reasoning(raw)
    assert sql == "SELECT SUM(doanh_thu) FROM sales;"
    assert t_type == "number"
    assert "Tính tổng doanh thu." in reasoning
    assert thinking == "Kiểm tra cột doanh_thu"
    assert issues == []

    # Unclosed truncated <think> block
    assert extract_thinking_trace("<think>Partial reasoning truncated") == "Partial reasoning truncated"


def test_candidate_generator_captures_repair_attempts_and_llm_calls(tmp_path: Path) -> None:
    import sqlite3

    db_path = tmp_path / "biz.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE sales (id INTEGER, amount REAL)")
    conn.execute("INSERT INTO sales VALUES (1, 100.0), (2, 200.0)")
    conn.commit()
    conn.close()

    metadata = {
        "sales.id": {"minimal_profile": "ID", "maximal_profile": "ID", "data_format": "NUMERIC"},
        "sales.amount": {"minimal_profile": "Amount", "maximal_profile": "Amount", "data_format": "NUMERIC"},
    }

    bad_resp = json.dumps({
        "reasoning": "Try bad column first",
        "sql": "SELECT SUM(nonexistent_col) FROM sales;",
        "type": "number",
    })
    good_resp = json.dumps({
        "reasoning": "<think>Fix column name to amount</think>Use sales.amount",
        "sql": "SELECT SUM(T1.\"amount\") FROM \"sales\" AS T1;",
        "type": "number",
    })
    engine = _ScriptedTraceLLMEngine([bad_resp, good_resp])
    cache_dir = tmp_path / "cache_biz"

    gen = CandidateGenerator(
        llm_engine=engine,
        seeds=(42,),
        sql_max_retries=2,
        execution_max_retries=2,
        db_path=db_path,
    )
    cand = gen.generate_candidate(
        candidate_id=0,
        seed=42,
        question="Total amount?",
        as_of_date="2026-05-01",
        reduced_schema={'sales."id"', 'sales."amount"'},
        processed_metadata=metadata,
        db_path=db_path,
        cache_dir=cache_dir,
        question_id="q_repair_1",
        trace_enabled=True,
    )

    assert cand.execution_success is True
    assert len(cand.attempts_history) == 2
    assert cand.attempts_history[0]["phase"] == "initial"
    assert cand.attempts_history[0]["reasoning"] == "Try bad column first"
    assert cand.attempts_history[0]["is_best_selected"] is False
    assert any(i["code"] == "COLUMN_NOT_FOUND" for i in cand.attempts_history[0]["issues"])

    assert cand.attempts_history[1]["phase"] == "sql_repair"
    assert cand.attempts_history[1]["thinking"] == "Fix column name to amount"
    assert cand.attempts_history[1]["is_best_selected"] is True
    assert cand.best_attempt_idx == 1
    assert "Use sales.amount" in cand.final_reasoning
    assert cand.final_thinking == "Fix column name to amount"

    llm_calls_file = cache_dir / "llm_calls.jsonl"
    assert llm_calls_file.is_file()
    records = [json.loads(line) for line in llm_calls_file.read_text(encoding="utf-8").splitlines()]
    assert [r["stage"] for r in records] == ["sql_init", "sql_repair"]
    assert records[0]["prompt_tokens"] is not None
    assert records[1]["metadata"]["turn"] == 1


def test_schema_linker_captures_prompts_and_repair_turns(tmp_path: Path) -> None:
    metadata = {
        "sales.status": {
            "minimal_profile": "Trạng thái đơn",
            "maximal_profile": "Trạng thái đơn",
            "data_format": "STRING",
            "sample_values": ["Hoàn trả", "Thành công"],
        },
        "sales.amount": {
            "minimal_profile": "Doanh thu",
            "maximal_profile": "Doanh thu",
            "data_format": "NUMERIC",
            "sample_values": ["1000"],
        },
    }
    fake_lsh = SimpleNamespace(
        query_literal_with_scores=lambda lit, top_k=5: (
            [("sales.status", "Hoàn trả", 0.95)] if "Hoàn trả" in lit else []
        )
    )
    engine = _ScriptedTraceLLMEngine([
        'SELECT SUM("amount") FROM "sales" WHERE "wrong_col" = \'Hoàn trả\'',
        'SELECT SUM("amount") FROM "sales" WHERE "status" = \'Hoàn trả\'',
    ])
    linker = TaskAlignedSchemaLinker(llm_engine=engine, max_retries=1)
    cache_dir = tmp_path / "cache_sl"

    res = linker.link_schema(
        question="Tổng tiền đơn Hoàn trả?",
        processed_metadata=metadata,
        lsh_indexer=fake_lsh,
        cases="a",
        cache_dir=cache_dir,
        question_id="q_sl_1",
        trace_enabled=True,
    )

    assert len(res.variant_traces) == 1
    vt = res.variant_traces[0]
    assert vt.formatted_prompt != ""
    assert "wrong_col" in vt.initial_sql
    assert len(vt.repair_turns) == 1
    assert vt.repair_turns[0]["retry_idx"] == 1
    assert "Hoàn trả" in str(vt.repair_turns[0]["mismatches"])
    assert vt.repair_turns[0]["revised_prompt"] != ""

    saved = json.loads((cache_dir / "schema_linking_result_q_sl_1.json").read_text(encoding="utf-8"))
    assert saved["variant_traces"][0]["initial_sql"] == vt.initial_sql
    assert saved["variant_traces"][0]["formatted_prompt"] == vt.formatted_prompt
    assert len(saved["variant_traces"][0]["repair_turns"]) == 1


def test_end_to_end_pipeline_traces_stored_in_cache_dir_and_no_trace_flag(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    biz_dir = data_dir / "biz_101"
    biz_dir.mkdir(parents=True)
    (biz_dir / "sales.csv").write_text(
        "id,ten_sp,so_luong\n1,Tra dao,5\n2,Ca phe,3\n",
        encoding="utf-8",
    )
    (biz_dir / "README.md").write_text("# Biz 101\nQuán cà phê.\n", encoding="utf-8")
    rulebook_path = tmp_path / "rulebook.md"
    rulebook_path.write_text("# Rulebook\nSố lượng tính theo cột so_luong.\n", encoding="utf-8")
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {
                "id": "q_trace_1",
                "business_id": "biz_101",
                "question": "Tổng số lượng bán ra?",
                "as_of_date": "2026-05-25",
            },
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    # 1. Default (trace_enabled=True)
    cache_base = tmp_path / "cache_on"
    logs_dir = tmp_path / "logs_on"
    pipeline = SubmissionPipeline(
        data_dir=data_dir,
        rulebook_path=rulebook_path,
        llm_engine=MockLLMEngine(),
        embedder=MockEmbedder(),
        vector_model_path="mock",
        output_path=tmp_path / "sub_on.jsonl",
        logs_dir=logs_dir,
        cache_base_dir=cache_base,
        seeds=(42,),
        cases="a",
        trace_enabled=True,
    )
    pipeline.run(questions_path)

    biz_cache = cache_base / "cache_biz_101"
    prof_trace_file = biz_cache / "profiling_trace.json"
    q_trace_file = biz_cache / "question_trace_q_trace_1.json"
    llm_calls_file = biz_cache / "llm_calls.jsonl"
    assert prof_trace_file.is_file()
    assert q_trace_file.is_file()
    assert llm_calls_file.is_file()

    prof_trace = json.loads(prof_trace_file.read_text(encoding="utf-8"))
    assert prof_trace["business_id"] == "biz_101"
    assert prof_trace["total_columns"] == 3
    assert all("prompt" in col and "raw_output" in col for col in prof_trace["columns"])

    q_trace = json.loads(q_trace_file.read_text(encoding="utf-8"))
    assert q_trace["question_id"] == "q_trace_1"
    assert "stage_timings_sec" in q_trace
    assert "llm_usage_summary" in q_trace
    assert "candidate_signatures" in q_trace
    assert "schema_linking_trace" in q_trace
    assert len(q_trace["candidates"]) == 1
    cand_trace = q_trace["candidates"][0]
    assert "schema_permutation" in cand_trace
    assert "initial_prompt" in cand_trace
    assert "initial_reasoning" in cand_trace
    assert "attempts_history" in cand_trace
    assert len(cand_trace["attempts_history"]) >= 1

    stages = {json.loads(line)["stage"] for line in llm_calls_file.read_text(encoding="utf-8").splitlines()}
    assert {"minimal_profile", "schema_linking_init", "sql_init"} <= stages

    # Verify result/logs/<qid>.json kept unchanged (does not leak heavy prompt fields)
    orig_log = json.loads((logs_dir / "q_trace_1.json").read_text(encoding="utf-8"))
    assert "attempts_history" not in orig_log["candidates"][0]
    assert "initial_prompt" not in orig_log["candidates"][0]
    assert "stage_timings_sec" not in orig_log

    # 2. Disabled (trace_enabled=False)
    cache_off = tmp_path / "cache_off"
    logs_off = tmp_path / "logs_off"
    orchestrator_off = ProfilingOrchestrator(
        business_dir=biz_dir,
        rulebook_path=rulebook_path,
        llm_engine=MockLLMEngine(),
        embedder=MockEmbedder(),
        cache_dir=cache_off / "cache_biz_101",
        trace_enabled=False,
    )
    orchestrator_off.run_pipeline()
    assert not (cache_off / "cache_biz_101" / "profiling_trace.json").exists()

    pipeline_off = SubmissionPipeline(
        data_dir=data_dir,
        rulebook_path=rulebook_path,
        llm_engine=MockLLMEngine(),
        embedder=MockEmbedder(),
        vector_model_path="mock",
        output_path=tmp_path / "sub_off.jsonl",
        logs_dir=logs_off,
        cache_base_dir=cache_off,
        seeds=(42,),
        cases="a",
        trace_enabled=False,
    )
    pipeline_off.process_question(
        QuestionItem(
            id="q_trace_off",
            business_id="biz_101",
            question="Tổng số lượng?",
            as_of_date="2026-05-25",
        )
    )
    assert not (cache_off / "cache_biz_101" / "question_trace_q_trace_off.json").exists()
    assert not (cache_off / "cache_biz_101" / "llm_calls.jsonl").exists()
    assert (logs_off / "q_trace_off.json").is_file()


def test_submission_cli_no_trace_flag(tmp_path: Path) -> None:
    from zaic_2026_4.submission.cli import build_parser

    parser = build_parser()
    args_default = parser.parse_args(["--split", "train"])
    assert args_default.trace_enabled is True

    args_no_trace = parser.parse_args(["--split", "train", "--no-trace"])
    assert args_no_trace.trace_enabled is False
