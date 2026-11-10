"""Integration tests for SubmissionPipeline and end-to-end question processing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zaic_2026_4.profiling.profile_summarizer import MockLLMEngine
from zaic_2026_4.profiling.vector_indexer import MockEmbedder
from zaic_2026_4.submission.pipeline import SubmissionPipeline


@pytest.fixture
def mock_dataset_env(tmp_path: Path) -> dict[str, Path]:
    """Create a minimal dataset environment with biz_test, rulebook, calendar, and questions."""
    data_dir = tmp_path / "data"
    biz_dir = data_dir / "biz_test"
    biz_dir.mkdir(parents=True, exist_ok=True)

    csv_path = biz_dir / "sales.csv"
    csv_path.write_text(
        "id,ma_sp,ten_sp,so_luong,don_gia,ngay_ban\n"
        "1,SP001,Sting dâu 330ml,10,12000,2026-05-01\n"
        "2,SP002,Nước ngọt Coca 320ml,5,10000,2026-05-02\n"
        "3,SP001,Sting dâu 330ml,8,12000,2026-05-03\n",
        encoding="utf-8",
    )

    readme_path = biz_dir / "README.md"
    readme_path.write_text("# Biz Test\nTạp hóa bán nước giải khát.\n", encoding="utf-8")

    rulebook_path = tmp_path / "rulebook.md"
    rulebook_path.write_text(
        "# Rulebook\nDoanh thu bằng tổng thành tiền sau chiết khấu.",
        encoding="utf-8",
    )

    calendar_path = tmp_path / "calendar.csv"
    calendar_path.write_text("solar_date,lunar_date,is_tet\n2026-05-01,2026-03-15,0\n", encoding="utf-8")

    questions_path = tmp_path / "questions.jsonl"
    q1 = {"id": "q_001", "business_id": "biz_test", "question": "Tổng số lượng Sting dâu?", "as_of_date": "2026-05-25"}
    q2 = {"id": "q_002", "business_id": "biz_test", "question": "Tỉ lệ huỷ đơn tháng này %?", "as_of_date": "2026-05-25"}
    questions_path.write_text(
        f"{json.dumps(q1, ensure_ascii=False)}\n{json.dumps(q2, ensure_ascii=False)}\n",
        encoding="utf-8",
    )

    out_file = tmp_path / "submission.jsonl"
    logs_dir = tmp_path / "logs"

    return {
        "data_dir": data_dir,
        "rulebook_path": rulebook_path,
        "calendar_path": calendar_path,
        "questions_path": questions_path,
        "output_path": out_file,
        "logs_dir": logs_dir,
        "cache_dir": tmp_path / "cache",
    }


def test_submission_pipeline_end_to_end(mock_dataset_env: dict[str, Path]) -> None:
    llm = MockLLMEngine()
    embedder = MockEmbedder()

    pipeline = SubmissionPipeline(
        data_dir=mock_dataset_env["data_dir"],
        rulebook_path=mock_dataset_env["rulebook_path"],
        llm_engine=llm,
        embedder=embedder,
        vector_model_path="mock",
        calendar_path=mock_dataset_env["calendar_path"],
        output_path=mock_dataset_env["output_path"],
        logs_dir=mock_dataset_env["logs_dir"],
        cache_base_dir=mock_dataset_env["cache_dir"],
        seeds=(42, 100, 2026),
        max_retries=2,
        rulebook_mode="hybrid",
    )

    submissions = pipeline.run(mock_dataset_env["questions_path"])
    assert len(submissions) == 2

    # Check that output file exists and has 2 lines conforming to contract
    out_file = mock_dataset_env["output_path"]
    assert out_file.is_file()

    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    for line in lines:
        data = json.loads(line)
        assert "id" in data
        assert "type" in data
        assert data["type"] in ("number", "percent", "string", "date", "boolean", "list", "table", "unanswerable", "abstain")
        if data["type"] != "abstain":
            assert "answer" in data

    # Check that log files exist and contain selection_reason and candidates
    logs_dir = mock_dataset_env["logs_dir"]
    log_q1 = logs_dir / "q_001.json"
    assert log_q1.is_file()

    log_data = json.loads(log_q1.read_text(encoding="utf-8"))
    assert log_data["question_id"] == "q_001"
    assert "selection_reason" in log_data
    assert "voting_status" in log_data
    assert "candidates" in log_data
    assert len(log_data["candidates"]) == 3


def test_resolve_split_paths(tmp_path: Path) -> None:
    from zaic_2026_4.submission.pipeline import resolve_split_paths

    split_dir = tmp_path / "train"
    split_dir.mkdir()
    (split_dir / "data").mkdir()
    (split_dir / "questions.jsonl").write_text('{"id": "1"}\n', encoding="utf-8")
    (split_dir / "rulebook.md").write_text("# Rulebook\n", encoding="utf-8")
    (split_dir / "calendar.csv").write_text("solar_date\n2026-01-01\n", encoding="utf-8")

    # 1. Resolve with direct path
    res = resolve_split_paths(split_dir)
    assert res["split_dir"] == split_dir
    assert res["questions_path"] == split_dir / "questions.jsonl"
    assert res["data_dir"] == split_dir / "data"
    assert res["rulebook_path"] == split_dir / "rulebook.md"
    assert res["calendar_path"] == split_dir / "calendar.csv"

    # 2. Resolve with split name and base_dir
    res_named = resolve_split_paths("train", base_dir=tmp_path)
    assert res_named["split_dir"] == split_dir

    # 3. Missing questions.jsonl raises FileNotFoundError
    (split_dir / "questions.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="Questions file not found"):
        resolve_split_paths(split_dir)

    # 4. Non-existent split raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Could not locate dataset split"):
        resolve_split_paths("non_existent_split", base_dir=tmp_path)


def test_submission_pipeline_from_split(mock_dataset_env: dict[str, Path]) -> None:
    # mock_dataset_env root contains data/, questions.jsonl, rulebook.md, calendar.csv
    split_dir = mock_dataset_env["data_dir"].parent
    llm = MockLLMEngine()
    embedder = MockEmbedder()

    pipeline, q_path = SubmissionPipeline.from_split(
        split=split_dir,
        llm_engine=llm,
        embedder=embedder,
        vector_model_path="mock",
        output_path=mock_dataset_env["output_path"],
        logs_dir=mock_dataset_env["logs_dir"],
        cache_base_dir=mock_dataset_env["cache_dir"],
    )
    assert q_path == split_dir / "questions.jsonl"
    assert pipeline.rulebook_path == split_dir / "rulebook.md"
    assert pipeline.calendar_path == split_dir / "calendar.csv"


def test_submission_cli_with_split(mock_dataset_env: dict[str, Path]) -> None:
    from zaic_2026_4.submission.cli import main

    split_dir = mock_dataset_env["data_dir"].parent
    out_file = mock_dataset_env["output_path"]
    logs_dir = mock_dataset_env["logs_dir"]
    cache_dir = mock_dataset_env["cache_dir"]

    cli_args = [
        "--split", str(split_dir),
        "--mock",
        "--output", str(out_file),
        "--log-dir", str(logs_dir),
        "--cache-dir", str(cache_dir),
        "--limit", "1",
    ]
    main(cli_args)

    assert out_file.is_file()
    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_submission_cli_with_cases(mock_dataset_env: dict[str, Path]) -> None:
    from zaic_2026_4.submission.cli import main

    split_dir = mock_dataset_env["data_dir"].parent
    out_file = mock_dataset_env["output_path"]
    logs_dir = mock_dataset_env["logs_dir"]
    cache_dir = mock_dataset_env["cache_dir"]

    cli_args = [
        "--split", str(split_dir),
        "--cases", "abe",
        "--mock",
        "--output", str(out_file),
        "--log-dir", str(logs_dir),
        "--cache-dir", str(cache_dir),
        "--limit", "1",
    ]
    main(cli_args)

    assert out_file.is_file()
    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_submission_pipeline_permission_fallback(mock_dataset_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    llm = MockLLMEngine()
    embedder = MockEmbedder()

    # Pass an output_path whose parent raises PermissionError on mkdir
    fake_no_perm_path = Path("/non_existent_system_root_dir/sub.jsonl")

    pipeline = SubmissionPipeline(
        data_dir=mock_dataset_env["data_dir"],
        rulebook_path=mock_dataset_env["rulebook_path"],
        llm_engine=llm,
        embedder=embedder,
        vector_model_path="mock",
        output_path=fake_no_perm_path,
        logs_dir=Path("/non_existent_system_root_dir/logs"),
        cache_base_dir=mock_dataset_env["cache_dir"],
    )
    # Must gracefully fall back to local result/
    assert pipeline.output_path.name == "sub.jsonl"
    assert pipeline.output_path.parent == Path("result").resolve()


def test_submission_pipeline_auto_saves_schema_linking_result(mock_dataset_env: dict[str, Path]) -> None:
    from zaic_2026_4.submission.models import QuestionItem

    llm = MockLLMEngine()
    embedder = MockEmbedder()
    cache_dir = mock_dataset_env["cache_dir"]

    pipeline = SubmissionPipeline(
        data_dir=mock_dataset_env["data_dir"],
        rulebook_path=mock_dataset_env["rulebook_path"],
        llm_engine=llm,
        embedder=embedder,
        vector_model_path="mock",
        calendar_path=mock_dataset_env["calendar_path"],
        output_path=mock_dataset_env["output_path"],
        logs_dir=mock_dataset_env["logs_dir"],
        cache_base_dir=cache_dir,
    )

    q = QuestionItem(
        id="q_test_101",
        business_id="biz_test",
        question="Tổng số lượng Sting dâu?",
        as_of_date="2026-05-25",
    )
    sub = pipeline.process_question(q)
    assert sub.id == "q_test_101"

    biz_cache = cache_dir / "cache_biz_test"
    schema_link_file = biz_cache / "schema_linking_result.json"
    assert schema_link_file.is_file()

    content = json.loads(schema_link_file.read_text(encoding="utf-8"))
    assert "final_linked_schema" in content
    assert "variant_traces" in content
    assert content["question_id"] == "q_test_101"

    per_q_file = biz_cache / "schema_linking_result_q_test_101.json"
    assert per_q_file.is_file()


def test_cli_quantize_and_gpu_embed_flags(
    mock_dataset_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify zaic-submit CLI parses --quantize and --gpu-embed-memory-utilization correctly."""
    from zaic_2026_4 import constants
    from zaic_2026_4.submission.cli import build_parser, main

    original_util = constants.DEFAULT_EMBED_GPU_UTIL
    try:
        parser = build_parser()
        args = parser.parse_args([
            "--split", "train",
            "--quantize", "fp8",
            "--gpu-embed-memory-utilization", "0.35",
            "--mock",
        ])
        assert args.quantize == "fp8"
        assert args.gpu_embed_memory_utilization == 0.35

        monkeypatch.setattr(
            "zaic_2026_4.submission.cli.resolve_split_paths",
            lambda *a, **kw: {
                "questions_path": mock_dataset_env["questions_path"],
                "data_dir": mock_dataset_env["data_dir"],
                "rulebook_path": mock_dataset_env["rulebook_path"],
                "calendar_path": mock_dataset_env["calendar_path"],
            },
        )
        main([
            "--split", "train",
            "--quantize", "fp8",
            "--gpu-embed-memory-utilization", "0.35",
            "--mock",
            "--output", str(mock_dataset_env["output_path"]),
            "--log-dir", str(mock_dataset_env["logs_dir"]),
            "--cache-dir", str(mock_dataset_env["cache_dir"]),
            "--limit", "1",
        ])
        assert constants.DEFAULT_EMBED_GPU_UTIL == 0.35
    finally:
        constants.DEFAULT_EMBED_GPU_UTIL = original_util





