"""Submission package for generating diversity candidates, SQL validation, and final submission output."""

from __future__ import annotations

from zaic_2026_4.submission.candidate_generator import CandidateGenerator
from zaic_2026_4.submission.models import (
    VALID_DATAMATE_TYPES,
    CandidateQuery,
    DataMateAnswerType,
    QuestionItem,
    QuestionSubmission,
    SQLGenerationOutput,
    ValidationResult,
)
from zaic_2026_4.submission.pipeline import (
    BusinessContext,
    SubmissionPipeline,
    resolve_split_paths,
)
from zaic_2026_4.submission.prompt_template import (
    FEW_SHOT_SQL_EXAMPLES_VI,
    build_sql_correction_prompt,
    build_sql_generation_prompt,
    format_permuted_schema_xml,
    format_rulebook_context,
    permute_schema,
)
from zaic_2026_4.submission.result_formatter import (
    ResultFormatter,
    normalize_contract_string,
)
from zaic_2026_4.submission.sql_validator import (
    SQLValidator,
    clean_sql_string,
    parse_structured_sql_output,
)
from zaic_2026_4.submission.voting_executor import (
    VotingExecutor,
    normalize_cell,
    summarize_result_set,
)

__all__ = [
    "FEW_SHOT_SQL_EXAMPLES_VI",
    "VALID_DATAMATE_TYPES",
    "BusinessContext",
    "CandidateGenerator",
    "CandidateQuery",
    "DataMateAnswerType",
    "QuestionItem",
    "QuestionSubmission",
    "ResultFormatter",
    "SQLGenerationOutput",
    "SQLValidator",
    "SubmissionPipeline",
    "ValidationResult",
    "VotingExecutor",
    "build_sql_correction_prompt",
    "build_sql_generation_prompt",
    "clean_sql_string",
    "format_permuted_schema_xml",
    "format_rulebook_context",
    "normalize_cell",
    "normalize_contract_string",
    "parse_structured_sql_output",
    "permute_schema",
    "resolve_split_paths",
    "summarize_result_set",
]
