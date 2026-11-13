"""Data models for question representation, SQL validation, candidate tracking, and submission."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from zaic_2026_4.submission.rules import ValidationIssue

DataMateAnswerType = Literal[
    "number",
    "percent",
    "string",
    "date",
    "boolean",
    "list",
    "table",
    "unanswerable",
    "abstain",
]

VALID_DATAMATE_TYPES: tuple[str, ...] = (
    "number",
    "percent",
    "string",
    "date",
    "boolean",
    "list",
    "table",
    "unanswerable",
    "abstain",
)


class SQLGenerationOutput(BaseModel):
    """Structured JSON output schema for LLM SQL generation and correction."""

    reasoning: str = ""
    sql: str
    type: DataMateAnswerType


@dataclass
class QuestionItem:
    """Represents a single query entry parsed from questions.jsonl."""

    id: str
    business_id: str
    question: str
    as_of_date: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuestionItem:
        """Create QuestionItem from raw JSON dict."""
        return cls(
            id=str(data["id"]),
            business_id=str(data["business_id"]),
            question=str(data["question"]),
            as_of_date=str(data["as_of_date"]),
        )


@dataclass
class ValidationResult:
    """Outcome of static SQLglot AST syntax and construction checks."""

    is_valid: bool
    issues: list[ValidationIssue | str] = field(default_factory=list)

    @property
    def issue_texts(self) -> list[str]:
        """Return list of string representations for all issues."""
        return [str(i) for i in self.issues]

    def codes(self) -> set[str]:
        """Return set of issue codes present in issues."""
        return {getattr(i, "code", "UNKNOWN") for i in self.issues}

    def blocking_issues(self) -> list[ValidationIssue | str]:
        """Return only blocking severity issues."""
        return [i for i in self.issues if getattr(i, "severity", "blocking") == "blocking"]

    def warning_issues(self) -> list[ValidationIssue | str]:
        """Return only warning severity issues."""
        return [i for i in self.issues if getattr(i, "severity", "blocking") == "warning"]


@dataclass
class CandidateQuery:
    """Detailed trace for a single diversity candidate SQL query."""

    candidate_id: int
    seed: int
    schema_permutation: list[str] = field(default_factory=list)
    initial_sql: str = ""
    final_sql: str = ""
    predicted_type: str = "number"
    is_valid_syntax: bool = True
    construction_issues: list[ValidationIssue | str] = field(default_factory=list)
    sql_retry_count: int = 0
    execution_retry_count: int = 0
    total_retry_count: dict[str, int] = field(default_factory=dict)
    execution_success: bool = False
    execution_error: str | None = None
    raw_result: list[Any] | None = None
    result_set: set[tuple[Any, ...]] | None = None
    result_columns: list[str] = field(default_factory=list)
    deliberate_abstain: bool = False
    unresolved_zero: bool = False
    repair_stop_reason: str | None = None
    rulebook_xml: str = ""
    grounded_literals_injected: list[dict[str, Any]] = field(default_factory=list)
    initial_prompt: str = ""
    initial_raw_output: str = ""
    initial_reasoning: str = ""
    initial_thinking: str = ""
    final_reasoning: str = ""
    final_thinking: str = ""
    best_attempt_idx: int | None = None
    attempts_history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Ensure total_retry_count dictionary is synchronized with granular retry counts."""
        if not self.total_retry_count:
            self.total_retry_count = {
                "sql": self.sql_retry_count,
                "execution": self.execution_retry_count,
            }
        else:
            if "sql" in self.total_retry_count and self.sql_retry_count == 0:
                self.sql_retry_count = self.total_retry_count["sql"]
            if "execution" in self.total_retry_count and self.execution_retry_count == 0:
                self.execution_retry_count = self.total_retry_count["execution"]

    def to_dict(self) -> dict[str, Any]:
        """Convert candidate trace to serializable dictionary."""
        res_repr: list[str] | None = None
        if self.result_set is not None:
            res_repr = [str(r) for r in sorted(self.result_set, key=str)]

        return {
            "candidate_id": self.candidate_id,
            "seed": self.seed,
            "initial_sql": self.initial_sql,
            "final_sql": self.final_sql,
            "predicted_type": self.predicted_type,
            "total_retry_count": {
                "sql": self.sql_retry_count,
                "execution": self.execution_retry_count,
            },
            "syntax_valid": self.is_valid_syntax,
            "construction_issues": [str(x) for x in self.construction_issues],
            "execution_success": self.execution_success,
            "execution_error": self.execution_error,
            "result_set": res_repr,
            "result_columns": list(self.result_columns[:20]),
            "deliberate_abstain": self.deliberate_abstain,
            "unresolved_zero": self.unresolved_zero,
            "repair_stop_reason": self.repair_stop_reason,
        }

    def to_full_trace_dict(self) -> dict[str, Any]:
        """Convert candidate trace with full prompts, reasoning, and repair attempt history."""
        base = self.to_dict()
        base.update({
            "schema_permutation": list(self.schema_permutation),
            "rulebook_xml": self.rulebook_xml,
            "grounded_literals_injected": self.grounded_literals_injected,
            "initial_prompt": self.initial_prompt,
            "initial_raw_output": self.initial_raw_output,
            "initial_reasoning": self.initial_reasoning,
            "initial_thinking": self.initial_thinking,
            "final_reasoning": self.final_reasoning,
            "final_thinking": self.final_thinking,
            "best_attempt_idx": self.best_attempt_idx,
            "attempts_history": self.attempts_history,
        })
        return base


@dataclass
class QuestionSubmission:
    """Complete prediction and execution trace for a single business question."""

    id: str
    type: str = "abstain"
    answer: Any | None = None
    reason: str | None = None
    elapsed_seconds: float = 0.0
    candidates: list[CandidateQuery] = field(default_factory=list)
    winning_candidate_id: int | None = None
    voting_status: str = "abstain_fallback"
    selection_reason: str = ""
    formatting_trace: dict[str, Any] = field(default_factory=dict)
    schema_linking_trace: dict[str, Any] = field(default_factory=dict)
    stage_timings_sec: dict[str, float] = field(default_factory=dict)
    llm_usage_summary: dict[str, Any] = field(default_factory=dict)
    candidate_signatures: list[dict[str, Any]] = field(default_factory=list)

    def to_submission_dict(self) -> dict[str, Any]:
        """Format strictly according to problem contract specification for submission.jsonl."""
        if self.type == "abstain":
            return {"id": self.id, "type": "abstain"}

        if self.type == "unanswerable":
            out: dict[str, Any] = {"id": self.id, "type": "unanswerable", "answer": None}
            if self.reason:
                out["reason"] = self.reason
            return out

        return {
            "id": self.id,
            "type": self.type,
            "answer": self.answer,
        }

    def to_trace_log(self, question_item: QuestionItem | None = None) -> dict[str, Any]:
        """Format detailed diagnostic trace for /result/logs/<id>.json."""
        log: dict[str, Any] = {
            "question_id": self.id,
            "business_id": question_item.business_id if question_item else None,
            "as_of_date": question_item.as_of_date if question_item else None,
            "question": question_item.question if question_item else None,
            "final_predicted_type": self.type,
            "final_predicted_answer": self.answer,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "selection_reason": self.selection_reason,
            "voting_status": self.voting_status,
            "winning_candidate_id": self.winning_candidate_id,
            "formatting_trace": self.formatting_trace,
            "candidates": [c.to_dict() for c in self.candidates],
        }
        if self.reason:
            log["reason"] = self.reason
        return log

    def to_full_cache_trace(self, question_item: QuestionItem | None = None) -> dict[str, Any]:
        """Format complete pipeline trace for cache_dir/question_trace_<id>.json."""
        winner_raw_preview: list[list[Any]] | None = None
        if self.winning_candidate_id is not None:
            for cand in self.candidates:
                if cand.candidate_id == self.winning_candidate_id and cand.raw_result is not None:
                    winner_raw_preview = [
                        list(row) if isinstance(row, (list, tuple)) else [row]
                        for row in cand.raw_result[:10]
                    ]
                    break

        trace: dict[str, Any] = {
            "question_id": self.id,
            "business_id": question_item.business_id if question_item else None,
            "as_of_date": question_item.as_of_date if question_item else None,
            "question": question_item.question if question_item else None,
            "final_predicted_type": self.type,
            "final_predicted_answer": self.answer,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "stage_timings_sec": self.stage_timings_sec,
            "llm_usage_summary": self.llm_usage_summary,
            "selection_reason": self.selection_reason,
            "voting_status": self.voting_status,
            "winning_candidate_id": self.winning_candidate_id,
            "candidate_signatures": self.candidate_signatures,
            "formatting_trace": {
                **self.formatting_trace,
                "raw_result_preview": winner_raw_preview,
            },
            "schema_linking_trace": self.schema_linking_trace,
            "candidates": [c.to_full_trace_dict() for c in self.candidates],
        }
        if self.reason:
            trace["reason"] = self.reason
        return trace


__all__ = [
    "VALID_DATAMATE_TYPES",
    "CandidateQuery",
    "DataMateAnswerType",
    "QuestionItem",
    "QuestionSubmission",
    "SQLGenerationOutput",
    "ValidationResult",
]
