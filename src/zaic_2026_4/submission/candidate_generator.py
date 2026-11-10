"""Candidate diversity generator with seed variation, schema permutation, and iterative self-correction."""

from __future__ import annotations

import logging
from typing import Any

from zaic_2026_4.constants import (
    DEFAULT_CANDIDATE_SEEDS,
    DEFAULT_SQL_MAX_RETRIES,
)
from zaic_2026_4.submission.models import CandidateQuery
from zaic_2026_4.submission.prompt_template import (
    build_sql_correction_prompt,
    build_sql_generation_prompt,
    permute_schema,
)
from zaic_2026_4.submission.sql_validator import SQLValidator, clean_sql_string

logger = logging.getLogger(__name__)


class CandidateGenerator:
    """Generates 3 diverse candidate SQL queries and applies self-correction on static flaws."""

    def __init__(
        self,
        llm_engine: Any,
        validator: SQLValidator | None = None,
        seeds: tuple[int, ...] = DEFAULT_CANDIDATE_SEEDS,
        max_retries: int = DEFAULT_SQL_MAX_RETRIES,
        rulebook_mode: str = "hybrid",
    ) -> None:
        """Initialize candidate generator.

        Args:
            llm_engine: Offline inference engine supporting .generate(prompts, seed=...).
            validator: SQLValidator instance (defaults to new instance).
            seeds: Randomization seeds for candidate diversity (default: 42, 100, 2026).
            max_retries: Maximum self-correction attempts per candidate (default: 3).
            rulebook_mode: Injection mode for split rulebook ('hybrid', 'full', 'retrieval').
        """
        self.llm_engine = llm_engine
        self.validator = validator or SQLValidator()
        self.seeds = seeds
        self.max_retries = max_retries
        self.rulebook_mode = rulebook_mode

    def generate_candidate(
        self,
        candidate_id: int,
        seed: int,
        question: str,
        as_of_date: str,
        reduced_schema: set[str],
        processed_metadata: dict[str, Any],
        rulebook_text: str = "",
        vector_indexer: Any | None = None,
    ) -> CandidateQuery:
        """Generate a single candidate SQL query with schema permutation and retry loop."""
        # 1. Permute reduced schema order using seed
        permuted_keys = permute_schema(reduced_schema, seed)

        # 2. Build generation prompt
        prompt = build_sql_generation_prompt(
            question=question,
            as_of_date=as_of_date,
            schema_keys=permuted_keys,
            processed_metadata=processed_metadata,
            rulebook_text=rulebook_text,
            vector_indexer=vector_indexer,
            rulebook_mode=self.rulebook_mode,
        )

        # 3. Initial generation
        raw_output = self.llm_engine.generate(
            prompt,
            temperature=0.7,
            max_tokens=512,
            seed=seed,
        )
        current_sql = clean_sql_string(raw_output if isinstance(raw_output, str) else raw_output[0])
        initial_sql = current_sql

        # 4. Validate SQL
        v_res = self.validator.validate(current_sql)
        retries = 0

        # 5. Iterative self-correction loop (up to max_retries)
        while not v_res.is_valid and retries < self.max_retries:
            retries += 1
            logger.debug(
                "Candidate %d (seed %d) retry %d due to issues: %s",
                candidate_id,
                seed,
                retries,
                v_res.issues,
            )
            corr_prompt = build_sql_correction_prompt(
                question=question,
                as_of_date=as_of_date,
                faulty_sql=current_sql,
                validation_issues=v_res.issues,
            )
            corr_output = self.llm_engine.generate(
                corr_prompt,
                temperature=0.2,
                max_tokens=512,
                seed=seed + retries,
            )
            current_sql = clean_sql_string(corr_output if isinstance(corr_output, str) else corr_output[0])
            v_res = self.validator.validate(current_sql)

        return CandidateQuery(
            candidate_id=candidate_id,
            seed=seed,
            schema_permutation=permuted_keys,
            initial_sql=initial_sql,
            final_sql=current_sql,
            is_valid_syntax=v_res.is_valid,
            construction_issues=v_res.issues,
            retry_count=retries,
        )

    def generate_all_candidates(
        self,
        question: str,
        as_of_date: str,
        reduced_schema: set[str],
        processed_metadata: dict[str, Any],
        rulebook_text: str = "",
        vector_indexer: Any | None = None,
    ) -> list[CandidateQuery]:
        """Generate all candidate queries across configured seeds."""
        candidates: list[CandidateQuery] = []
        for idx, seed in enumerate(self.seeds):
            cand = self.generate_candidate(
                candidate_id=idx,
                seed=seed,
                question=question,
                as_of_date=as_of_date,
                reduced_schema=reduced_schema,
                processed_metadata=processed_metadata,
                rulebook_text=rulebook_text,
                vector_indexer=vector_indexer,
            )
            candidates.append(cand)
        return candidates


__all__ = ["CandidateGenerator"]
