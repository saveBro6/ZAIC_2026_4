"""Lightweight lexical intent detection shared by schema coverage and SQL validation.

The detected signals are purely linguistic (question phrasing plus the resolved temporal
periods) and never depend on a concrete dataset, table or ground-truth file. They are used
to decide which generic guardrails apply: temporal anchoring, aggregation support,
exact-match comparison and duplicate-safe counting.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from zaic_2026_4.common.temporal_resolver import (
    TemporalPeriod,
    parse_iso_date,
    resolve_temporal_periods,
)

_STROKE_D_MAP = str.maketrans({"đ": "d", "Đ": "D"})

_AGGREGATE_TERMS = (
    "tong",
    "tong cong",
    "tong so",
    "sum",
    "total",
    "trung binh",
    "average",
    "avg",
    "lon nhat",
    "nho nhat",
    "cao nhat",
    "thap nhat",
    "nhieu nhat",
    "it nhat",
    "max",
    "min",
)

_COUNT_TERMS = (
    "bao nhieu",
    "so luong",
    "dem",
    "count",
    "how many",
    "number of",
)

_EXACT_TERMS = (
    "chinh xac",
    "dung la",
    "dung ten",
    "dung loai",
    "dung ma",
    "khop chinh xac",
    "exact",
    "exactly",
)

_TEMPORAL_CUE_TERMS = (
    "ngay",
    "thang",
    "quy",
    "nam",
    "ky",
    "thoi gian",
    "tinh den",
    "hom nay",
    "date",
    "month",
    "year",
    "quarter",
    "period",
    "time",
)

_DISTINCT_TERMS = (
    "duy nhat",
    "khong trung",
    "khong lap",
    "phan biet",
    "distinct",
    "unique",
    "deduplicat",
)


def fold_text(text: str) -> str:
    """Lowercase and strip Vietnamese diacritics for robust keyword matching."""
    lowered = (text or "").lower().translate(_STROKE_D_MAP)
    decomposed = unicodedata.normalize("NFD", lowered)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", stripped).strip()


def _contains_any(folded: str, terms: tuple[str, ...]) -> bool:
    return any(term in folded for term in terms)


@dataclass(frozen=True)
class QueryIntent:
    """Lexically derived signals describing what a question asks for."""

    periods: tuple[TemporalPeriod, ...] = ()
    has_temporal: bool = False
    has_temporal_cue: bool = False
    has_aggregate: bool = False
    has_count: bool = False
    wants_exact_match: bool = False
    wants_distinct: bool = False
    folded_question: str = ""
    matched_terms: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def primary_period(self) -> TemporalPeriod | None:
        """Most specific resolved period, if any."""
        return self.periods[0] if self.periods else None

    def to_dict(self) -> dict[str, Any]:
        """Serialize intent into a JSON-compatible dictionary for tracing."""
        return {
            "has_temporal": self.has_temporal,
            "has_temporal_cue": self.has_temporal_cue,
            "has_aggregate": self.has_aggregate,
            "has_count": self.has_count,
            "wants_exact_match": self.wants_exact_match,
            "wants_distinct": self.wants_distinct,
            "periods": [p.to_dict() for p in self.periods],
            "matched_terms": {k: list(v) for k, v in self.matched_terms.items()},
        }


def detect_query_intent(
    question: str,
    as_of_date: str | Any | None = None,
) -> QueryIntent:
    """Derive generic intent signals from the question text and the anchor date."""
    folded = fold_text(question)
    periods = tuple(resolve_temporal_periods(question, as_of_date))

    matched: dict[str, tuple[str, ...]] = {}
    for group, terms in (
        ("aggregate", _AGGREGATE_TERMS),
        ("count", _COUNT_TERMS),
        ("exact", _EXACT_TERMS),
        ("distinct", _DISTINCT_TERMS),
    ):
        hits = tuple(t for t in terms if t in folded)
        if hits:
            matched[group] = hits

    has_cue = bool(periods) or _contains_any(folded, _TEMPORAL_CUE_TERMS)

    return QueryIntent(
        periods=periods,
        has_temporal=bool(periods) or parse_iso_date(as_of_date) is not None,
        has_temporal_cue=has_cue,
        has_aggregate=bool(matched.get("aggregate")) or bool(matched.get("count")),
        has_count=bool(matched.get("count")),
        wants_exact_match=bool(matched.get("exact")),
        wants_distinct=bool(matched.get("distinct")),
        folded_question=folded,
        matched_terms=matched,
    )


def format_intent_hint_xml(intent: QueryIntent) -> str:
    """Render a compact <query_intent_hint> block used by the SQL generation prompts.

    Only emits guidance for signals that were actually detected, so the prompt stays small
    for questions that carry no temporal, aggregation or exact-match cue.
    """
    hints: list[str] = []
    period = intent.primary_period
    if period is not None:
        hints.append(
            f"- Khoảng thời gian đã xác định từ câu hỏi: {period.start} đến {period.end} "
            f'(biểu thức "{period.matched_text}"). Bắt buộc dùng đúng khoảng này, '
            "không tự suy diễn mốc thời gian khác."
        )
    if intent.has_aggregate:
        hints.append(
            "- Câu hỏi yêu cầu giá trị tổng hợp: đảm bảo có hàm tổng hợp (SUM/COUNT/AVG/MIN/MAX) "
            "hoặc LIMIT 1 phù hợp với kiểu dữ liệu trả về."
        )
    if intent.wants_exact_match:
        hints.append(
            "- Câu hỏi yêu cầu so khớp chính xác: dùng phép so sánh bằng (LOWER(col) = LOWER('giá trị')) "
            "thay vì LIKE với ký tự đại diện."
        )
    if intent.wants_distinct:
        hints.append(
            "- Câu hỏi nhắc đến tính duy nhất/không trùng: cân nhắc COUNT(DISTINCT ...) hoặc loại bỏ "
            "dòng trùng lặp trước khi tổng hợp."
        )
    if not hints:
        return ""
    return "<query_intent_hint>\n" + "\n".join(hints) + "\n</query_intent_hint>"


__all__ = [
    "QueryIntent",
    "detect_query_intent",
    "fold_text",
    "format_intent_hint_xml",
]
