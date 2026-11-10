"""Unit tests for CandidateGenerator and prompt engineering with diversity."""

from __future__ import annotations

from typing import Any

from zaic_2026_4.profiling.profile_summarizer import MockLLMEngine
from zaic_2026_4.submission.candidate_generator import CandidateGenerator
from zaic_2026_4.submission.prompt_template import (
    FEW_SHOT_SQL_EXAMPLES_VI,
    build_sql_correction_prompt,
    build_sql_generation_prompt,
    format_rulebook_context,
    permute_schema,
)
from zaic_2026_4.submission.sql_validator import SQLValidator


class MockRetryLLMEngine:
    """Mock engine that simulates a faulty first response followed by a corrected response."""

    def __init__(self) -> None:
        self.call_count = 0

    def generate(
        self,
        prompts: list[str] | str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[str] | str:
        self.call_count += 1
        is_single = isinstance(prompts, str)
        prompt_text = prompts if is_single else prompts[0]

        # If it's a correction prompt, return fixed SQL
        if "validation_errors" in prompt_text:
            res = 'SELECT name FROM users WHERE price IS NOT NULL ORDER BY price ASC;'
        else:
            # First attempt: faulty SQL with ASC but no NOT NULL
            res = 'SELECT name FROM users ORDER BY price ASC;'

        return res if is_single else [res]


def test_permute_schema_different_seeds() -> None:
    schema = {
        "orders.id",
        "orders.price",
        "orders.date",
        "customers.name",
        "customers.phone",
        "items.sku",
        "items.title",
    }
    p1 = permute_schema(schema, seed=42)
    p2 = permute_schema(schema, seed=100)
    p3 = permute_schema(schema, seed=2026)

    # All columns preserved
    assert set(p1) == schema
    assert set(p2) == schema
    assert set(p3) == schema

    # Permutations are different
    assert p1 != p2 or p2 != p3


def test_rulebook_context_modes() -> None:
    raw_rulebook = "# Rulebook\n1. Doanh thu sau chiet khau.\n2. Loai tru don huy."

    class DummyVectorIndexer:
        def search_rules(self, query: str, top_k: int = 3) -> list[dict[str, str]]:
            return [{"text": "1. Doanh thu sau chiet khau."}]

    dummy_indexer = DummyVectorIndexer()

    # 1. Full mode
    ctx_full = format_rulebook_context(raw_rulebook, dummy_indexer, "doanh thu", mode="full")
    assert "<split_rulebook>" in ctx_full
    assert "<focal_rulebook_sections>" not in ctx_full

    # 2. Retrieval mode
    ctx_retrieval = format_rulebook_context(raw_rulebook, dummy_indexer, "doanh thu", mode="retrieval")
    assert "<relevant_business_rules>" in ctx_retrieval
    assert "<split_rulebook>" not in ctx_retrieval

    # 3. Hybrid mode (Default)
    ctx_hybrid = format_rulebook_context(raw_rulebook, dummy_indexer, "doanh thu", mode="hybrid")
    assert "<split_rulebook>" in ctx_hybrid
    assert "<focal_rulebook_sections>" in ctx_hybrid
    assert "1. Doanh thu sau chiet khau." in ctx_hybrid


def test_candidate_generator_with_retry() -> None:
    engine = MockRetryLLMEngine()
    validator = SQLValidator()
    generator = CandidateGenerator(
        llm_engine=engine,
        validator=validator,
        seeds=(42,),
        max_retries=3,
    )

    metadata = {
        "users.id": {"data_format": "INTEGER", "minimal_profile": "ID"},
        "users.price": {"data_format": "REAL", "minimal_profile": "Gia"},
    }
    cand = generator.generate_candidate(
        candidate_id=0,
        seed=42,
        question="Giá thấp nhất?",
        as_of_date="2026-05-25",
        reduced_schema={"users.id", "users.price"},
        processed_metadata=metadata,
    )

    # Initial SQL had issues
    assert "ORDER BY price ASC" in cand.initial_sql
    # Final SQL was corrected
    assert "WHERE price IS NOT NULL" in cand.final_sql
    assert cand.is_valid_syntax
    assert cand.retry_count == 1


def test_candidate_generator_all_candidates() -> None:
    engine = MockLLMEngine()
    generator = CandidateGenerator(
        llm_engine=engine,
        seeds=(42, 100, 2026),
    )
    metadata = {
        "users.name": {"data_format": "STRING", "minimal_profile": "Ten"},
    }
    candidates = generator.generate_all_candidates(
        question="Danh sách tên người dùng?",
        as_of_date="2026-05-25",
        reduced_schema={"users.name"},
        processed_metadata=metadata,
    )

    assert len(candidates) == 3
    for idx, c in enumerate(candidates):
        assert c.candidate_id == idx
        assert c.seed in (42, 100, 2026)


def test_few_shot_sql_examples_structure() -> None:
    """Verify that FEW_SHOT_SQL_EXAMPLES_VI contains 2 complete examples with proper formatting."""
    assert '<example id="1">' in FEW_SHOT_SQL_EXAMPLES_VI
    assert '<example id="2">' in FEW_SHOT_SQL_EXAMPLES_VI
    assert '<example id="3">' not in FEW_SHOT_SQL_EXAMPLES_VI
    assert "Sữa tươi" in FEW_SHOT_SQL_EXAMPLES_VI
    assert "IS NOT NULL" in FEW_SHOT_SQL_EXAMPLES_VI
    assert "BETWEEN" in FEW_SHOT_SQL_EXAMPLES_VI
    assert "substr" in FEW_SHOT_SQL_EXAMPLES_VI
    assert "REPLACE" in FEW_SHOT_SQL_EXAMPLES_VI


def test_format_permuted_schema_xml_with_hint() -> None:
    """Verify format_permuted_schema_xml includes llm_hint when present."""
    from zaic_2026_4.submission.prompt_template import format_permuted_schema_xml

    metadata = {
        "orders.price": {
            "data_format": "CURRENCY",
            "minimal_profile": "Cột lưu giá tiền",
            "raw_samples": ["10,000", "20,000"],
            "llm_hint": "Formatted VND currency. CAST(REPLACE(col, ',', '') AS REAL)",
        },
        "orders.date": {
            "data_format": "SLASH_DATE",
            "minimal_profile": "Cột ngày",
            "llm_hint": "Vietnamese date (D/M/YYYY).",
        },
    }
    xml_output = format_permuted_schema_xml(["orders.price", "orders.date"], metadata)
    assert '<table name="orders">' in xml_output
    assert 'type="CURRENCY"' in xml_output
    assert "Samples: ['10,000', '20,000']" in xml_output
    assert "Hint: Formatted VND currency." in xml_output
    assert "Hint: Vietnamese date (D/M/YYYY)." in xml_output


def test_build_sql_generation_prompt_vietnamese() -> None:
    """Verify Vietnamese prompt formatting and structure in build_sql_generation_prompt."""
    metadata = {
        "ban_hang.Doanh thu": {"data_format": "REAL", "minimal_profile": "Doanh thu ban hang"},
    }
    prompt = build_sql_generation_prompt(
        question="Doanh thu hôm nay là bao nhiêu?",
        as_of_date="2026-05-25",
        schema_keys=["ban_hang.Doanh thu"],
        processed_metadata=metadata,
        rulebook_text="1. Loai tru hoa don bi huy.",
        include_few_shot=True,
    )

    # Check Vietnamese system instruction presence
    assert "chuyên gia kỹ sư Text-to-SQL" in prompt
    assert "[HƯỚNG DẪN SQL VÀ QUY TẮC CHÍNH]" in prompt
    assert "Neo mốc thời gian" in prompt
    assert "Xử lý NULL & Phép toán" in prompt

    # Check XML sections
    assert "<system>" in prompt
    assert "<split_rulebook>" in prompt
    assert "<temporal_anchor>\nas_of_date: 2026-05-25\n</temporal_anchor>" in prompt
    assert "<selected_schema>" in prompt
    assert "<few_shot_examples>" in prompt
    assert "<user_question>\nDoanh thu hôm nay là bao nhiêu?\n</user_question>" in prompt
    assert "<output>" in prompt


def test_build_sql_correction_prompt_vietnamese() -> None:
    """Verify Vietnamese SQL correction prompt content and structure."""
    prompt = build_sql_correction_prompt(
        question="Tổng số lượng hàng?",
        as_of_date="2026-05-25",
        faulty_sql="SELECT name FROM users ORDER BY price ASC;",
        validation_issues=["Null values not handled in ASC ordering."],
    )

    assert "Bạn là một chuyên gia kỹ thuật SQL" in prompt
    assert "Vui lòng sửa lại câu truy vấn dựa trên các phản hồi bên dưới." in prompt
    assert "<user_question>\nTổng số lượng hàng?\n</user_question>" in prompt
    assert "<temporal_anchor>\nas_of_date: 2026-05-25\n</temporal_anchor>" in prompt
    assert "<previous_sql>\nSELECT name FROM users ORDER BY price ASC;\n</previous_sql>" in prompt
    assert "<validation_errors>\n- Null values not handled in ASC ordering.\n</validation_errors>" in prompt
    assert "<corrected_sql>" in prompt

