"""Unit tests for CandidateGenerator and prompt engineering with diversity."""

from __future__ import annotations

from typing import Any

from zaic_2026_4.profiling.profile_summarizer import MockLLMEngine
from zaic_2026_4.submission.candidate_generator import CandidateGenerator
from zaic_2026_4.submission.prompt_template import (
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
