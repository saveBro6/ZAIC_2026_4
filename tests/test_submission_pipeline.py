"""Integration tests for SubmissionPipeline and end-to-end question processing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zaic_2026_4.profiling.profile_summarizer import MockLLMEngine
from zaic_2026_4.profiling.vector_indexer import MockEmbedder
from zaic_2026_4.submission.models import QuestionItem
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
