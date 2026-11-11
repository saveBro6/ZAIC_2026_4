"""Data models for question representation, SQL validation, candidate tracking, and submission."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

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
    issues: list[str] = field(default_factory=list)


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
    construction_issues: list[str] = field(default_factory=list)
    sql_retry_count: int = 0
    execution_retry_count: int = 0
    total_retry_count: dict[str, int] = field(default_factory=dict)
    execution_success: bool = False
    execution_error: str | None = None
    raw_result: list[Any] | None = None
    result_set: set[tuple[Any, ...]] | None = None

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
            "construction_issues": list(self.construction_issues),
            "execution_success": self.execution_success,
            "execution_error": self.execution_error,
            "result_set": res_repr,
        }


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
            "candidates": [c.to_dict() for c in self.candidates],
        }
        if self.reason:
            log["reason"] = self.reason
        return log


__all__ = [
    "VALID_DATAMATE_TYPES",
    "CandidateQuery",
    "DataMateAnswerType",
    "QuestionItem",
    "QuestionSubmission",
    "SQLGenerationOutput",
    "ValidationResult",
]
