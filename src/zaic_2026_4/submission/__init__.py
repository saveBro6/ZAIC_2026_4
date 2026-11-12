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
)
from zaic_2026_4.submission.prompt_template import (
    FEW_SHOT_SQL_EXAMPLES_VI,
    RepairContext,
    build_sql_correction_prompt,
    build_sql_generation_prompt,
    format_grounded_entities_xml,
    format_permuted_schema_xml,
    format_rulebook_context,
    permute_schema,
)
from zaic_2026_4.submission.result_formatter import (
    PercentScale,
    ResultFormatter,
    derive_table_columns,
    detect_percent_scale,
    normalize_contract_string,
)
from zaic_2026_4.submission.rules import (
    RULE_REGISTRY,
    Severity,
    ValidationIssue,
    render_diagnostics,
)
from zaic_2026_4.submission.sql_validator import (
    SQLValidator,
    clean_sql_string,
    parse_structured_sql_output,
)
from zaic_2026_4.submission.voting_executor import (
    VotingExecutor,
    answer_signature,
    has_evidence,
    normalize_cell,
    summarize_result_set,
)

__all__ = [
    "FEW_SHOT_SQL_EXAMPLES_VI",
    "RULE_REGISTRY",
    "VALID_DATAMATE_TYPES",
    "BusinessContext",
    "CandidateGenerator",
    "CandidateQuery",
    "DataMateAnswerType",
    "PercentScale",
    "QuestionItem",
    "QuestionSubmission",
    "RepairContext",
    "ResultFormatter",
    "SQLGenerationOutput",
    "SQLValidator",
    "Severity",
    "SubmissionPipeline",
    "ValidationIssue",
    "ValidationResult",
    "VotingExecutor",
    "answer_signature",
    "build_sql_correction_prompt",
    "build_sql_generation_prompt",
    "clean_sql_string",
    "derive_table_columns",
    "detect_percent_scale",
    "format_grounded_entities_xml",
    "format_permuted_schema_xml",
    "format_rulebook_context",
    "has_evidence",
    "normalize_cell",
    "normalize_contract_string",
    "parse_structured_sql_output",
    "permute_schema",
    "render_diagnostics",
    "summarize_result_set",
]
