"""Literal grounding hygiene: keep only grounded entities that are provably question-derived.

Grounding candidates arrive from two very different sources: entity mentions found in the
question itself, and literals extracted from generated trial SQL. Trial SQL is speculative,
so its literals (typically date fragments copied from the few-shot format or fuzzy matches
against a calendar table) must never be presented to the downstream model as facts "from
the question". This module enforces that contract:

* the mention must literally occur in the question,
* the target column must survive in the final linked schema,
* calendar / date-dimension columns and date-like values are dropped, because concrete
  periods are supplied by the deterministic temporal resolver instead,
* every kept item is tagged with ``source`` so the prompt can state its provenance honestly.
"""

from __future__ import annotations

import re
from typing import Any

from zaic_2026_4.common.query_intent import fold_text
from zaic_2026_4.constants import (
    DEFAULT_GROUNDED_LITERAL_LIMIT,
    DEFAULT_GROUNDED_LITERAL_MIN_SCORE,
)

# Unambiguous date-dimension table names only: bare fragments such as "lich" also occur
# inside legitimate business table names (for example a history table).
CALENDAR_TABLE_TOKENS: tuple[str, ...] = (
    "calendar",
    "date_dim",
    "dim_date",
    "dimdate",
    "date_table",
    "date_dimension",
    "ngay_thang_dim",
)

DATE_LIKE_REGEX = re.compile(
    r"^\d{4}-\d{1,2}-\d{1,2}$"
    r"|^\d{1,2}/\d{1,2}/\d{2,4}$"
    r"|^\d{1,2}-\d{1,2}-\d{4}$"
    r"|^\d{4}-\d{1,2}$"
)


def _strip_quotes(value: str) -> str:
    """Remove identifier quoting so "Table"."Column" and Table.Column compare equal."""
    return value.replace('"', "").strip().strip("'").strip()


def _split_table(column_key: str) -> tuple[str, str]:
    clean = _strip_quotes(column_key)
    if "." in clean:
        table, col = clean.split(".", 1)
        return _strip_quotes(table), _strip_quotes(col)
    return "", clean


def _is_calendar_column(column_key: str) -> bool:
    table, _col = _split_table(column_key)
    folded_table = fold_text(table)
    return any(tok in folded_table for tok in CALENDAR_TABLE_TOKENS)


def _is_date_like(value: str) -> bool:
    return bool(DATE_LIKE_REGEX.match(value.strip()))


def filter_grounded_literals(
    candidates: list[dict[str, Any]] | None,
    final_schema: set[str],
    question: str,
    min_score: float = DEFAULT_GROUNDED_LITERAL_MIN_SCORE,
    max_items: int = DEFAULT_GROUNDED_LITERAL_LIMIT,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Keep only grounded entities backed by the question and the final linked schema.

    Args:
        candidates: Raw grounding candidates with mention/canonical_value/column/score.
        final_schema: Final linked schema keys, quoted or unquoted.
        question: User question the mentions must be traceable to.
        min_score: Minimum similarity score accepted for a grounding match.
        max_items: Upper bound on retained groundings to keep the prompt small.

    Returns:
        Tuple of (kept, dropped) where dropped entries carry a ``reason``.
    """
    if not candidates:
        return [], []

    schema_lookup: dict[str, str] = {}
    for key in final_schema:
        schema_lookup[fold_text(_strip_quotes(key))] = _strip_quotes(key)

    folded_question = fold_text(question)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    ordered = sorted(
        candidates,
        key=lambda item: float(item.get("score") or 0.0),
        reverse=True,
    )

    for item in ordered:
        mention = _strip_quotes(str(item.get("mention", "")))
        canonical = _strip_quotes(str(item.get("canonical_value", "")))
        column = _strip_quotes(str(item.get("column", "")))
        score = float(item.get("score") or 0.0)

        if not mention or not canonical or not column:
            continue
        if score < min_score:
            dropped.append({"mention": mention, "column": column, "reason": "low_score"})
            continue
        if _is_date_like(canonical):
            dropped.append({"mention": mention, "column": column, "reason": "date_value"})
            continue
        if _is_calendar_column(column):
            dropped.append({"mention": mention, "column": column, "reason": "calendar_column"})
            continue

        folded_mention = fold_text(mention)
        if folded_mention not in folded_question:
            dropped.append({"mention": mention, "column": column, "reason": "not_in_question"})
            continue

        resolved_column = schema_lookup.get(fold_text(column))
        if resolved_column is None:
            dropped.append({"mention": mention, "column": column, "reason": "column_not_in_schema"})
            continue

        dedupe_key = (folded_mention, fold_text(resolved_column))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        kept.append({
            "mention": mention,
            "canonical_value": canonical,
            "column": resolved_column,
            "score": round(score, 4),
            "source": "question",
        })

    kept = kept[:max_items]
    return kept, dropped


__all__ = [
    "CALENDAR_TABLE_TOKENS",
    "DATE_LIKE_REGEX",
    "filter_grounded_literals",
]
