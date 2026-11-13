"""Deterministic resolution of relative temporal expressions into concrete ISO date ranges.

Relative expressions such as "thang nay", "quy truoc" or "30 ngay gan day" must never be
guessed by the language model: every downstream prompt is anchored by the as_of_date
supplied with the question, and this module turns each recognised expression into an
explicit inclusive [start, end] period. The resolver is purely lexical and language
agnostic with respect to the data: it never inspects tables, columns or ground truth.
"""

from __future__ import annotations

import calendar
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta

ISO_DATE_FORMAT = "%Y-%m-%d"

_VI_MONTH_WORD = r"(?:thang|tháng)"
_VI_QUARTER_WORD = r"(?:quy|quý)"
_VI_YEAR_WORD = r"(?:nam|năm)"


@dataclass(frozen=True)
class TemporalPeriod:
    """Concrete inclusive date range derived from a relative temporal expression."""

    label: str
    start: str
    end: str
    matched_text: str

    def to_dict(self) -> dict[str, str]:
        """Serialize period into a JSON-compatible dictionary."""
        return {
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "matched_text": self.matched_text,
        }


def parse_iso_date(value: str | date | datetime | None) -> date | None:
    """Parse an ISO date string (or accept date/datetime) into a date."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in (ISO_DATE_FORMAT, "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _iso(d: date) -> str:
    return d.strftime(ISO_DATE_FORMAT)


def _month_bounds(anchor: date) -> tuple[date, date]:
    last_day = calendar.monthrange(anchor.year, anchor.month)[1]
    return date(anchor.year, anchor.month, 1), date(anchor.year, anchor.month, last_day)


def _shift_month(anchor: date, delta: int) -> date:
    month_index = anchor.year * 12 + (anchor.month - 1) + delta
    return date(month_index // 12, month_index % 12 + 1, 1)


def _quarter_start(anchor: date) -> date:
    start_month = ((anchor.month - 1) // 3) * 3 + 1
    return date(anchor.year, start_month, 1)


def _quarter_bounds(anchor: date) -> tuple[date, date]:
    start = _quarter_start(anchor)
    end_month_anchor = _shift_month(start, 2)
    _, end = _month_bounds(end_month_anchor)
    return start, end


def _year_bounds(anchor: date) -> tuple[date, date]:
    return date(anchor.year, 1, 1), date(anchor.year, 12, 31)


def _as_period(label: str, start: date, end: date, matched: str, anchor: date) -> TemporalPeriod | None:
    """Clamp a candidate period to the anchor date and build the TemporalPeriod."""
    if start > anchor:
        return None
    end = min(end, anchor)
    if end < start:
        return None
    return TemporalPeriod(label=label, start=_iso(start), end=_iso(end), matched_text=matched)


_STROKE_D_MAP = str.maketrans({"đ": "d", "Đ": "D"})


def _fold(text: str) -> str:
    """Lowercase and strip diacritics so accented and unaccented spellings both match.

    Vietnamese "đ" is a standalone codepoint rather than a combining mark, so it is
    transliterated explicitly before the NFD decomposition handles the tone marks.
    """
    lowered = text.lower().translate(_STROKE_D_MAP)
    decomposed = unicodedata.normalize("NFD", lowered)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)


@dataclass(frozen=True)
class _Rule:
    pattern: re.Pattern[str]
    builder: str
    specificity: int


def _build_rules() -> list[_Rule]:
    rules: list[_Rule] = [
        # Rolling windows: "30 ngay gan day", "2 tuan qua", "3 thang tro lai day"
        _Rule(
            re.compile(
                r"(\d{1,3})\s*(ngay|tuan|thang|quy|nam)\s*(?:gan day|tro lai day|qua|truoc|vua qua)",
            ),
            "rolling",
            9,
        ),
        # Explicit day: "ngay 05/06/2026", "05/06/2026"
        _Rule(re.compile(r"(?:ngay\s*)?(\d{1,2})/(\d{1,2})/(\d{4})"), "single_day", 8),
        # Explicit month: "thang 5/2026", "thang 05 nam 2026", "05/2026"
        _Rule(
            re.compile(
                rf"{_VI_MONTH_WORD}\s*(\d{{1,2}})\s*(?:/|{_VI_YEAR_WORD}\s*)(\d{{4}})",
            ),
            "explicit_month",
            7,
        ),
        # Explicit quarter: "quy 2/2026", "quy 2 nam 2026"
        _Rule(
            re.compile(
                rf"{_VI_QUARTER_WORD}\s*(\d{{1}})\s*(?:/|{_VI_YEAR_WORD}\s*)(\d{{4}})",
            ),
            "explicit_quarter",
            7,
        ),
        # Explicit year: "nam 2025"
        _Rule(re.compile(rf"{_VI_YEAR_WORD}\s*(\d{{4}})"), "explicit_year", 6),
        # Month to date / full previous month
        _Rule(re.compile(rf"{_VI_MONTH_WORD}\s*(?:nay|hien tai)"), "month_to_date", 5),
        _Rule(re.compile(rf"{_VI_MONTH_WORD}\s*(?:truoc|roi|vua qua)"), "previous_month", 5),
        _Rule(re.compile(r"thang\s*truoc"), "previous_month", 5),
        # Quarter to date / full previous quarter
        _Rule(re.compile(rf"{_VI_QUARTER_WORD}\s*(?:nay|hien tai)"), "quarter_to_date", 5),
        _Rule(re.compile(rf"{_VI_QUARTER_WORD}\s*(?:truoc|roi)"), "previous_quarter", 5),
        # Year to date / full previous year
        _Rule(re.compile(rf"{_VI_YEAR_WORD}\s*(?:nay|hien tai)"), "year_to_date", 5),
        _Rule(re.compile(rf"{_VI_YEAR_WORD}\s*(?:truoc|roi)"), "previous_year", 5),
        _Rule(re.compile(r"dau\s*nam|tu\s*dau\s*nam"), "year_to_date", 4),
        # Week to date / full previous week
        _Rule(re.compile(r"tuan\s*(?:nay|hien tai)"), "week_to_date", 5),
        _Rule(re.compile(r"tuan\s*(?:truoc|roi)"), "previous_week", 5),
        # Single day
        _Rule(re.compile(r"hom\s*nay"), "today", 5),
        _Rule(re.compile(r"hom\s*qua"), "yesterday", 5),
        # English variants
        _Rule(re.compile(r"\btoday\b"), "today", 5),
        _Rule(re.compile(r"\byesterday\b"), "yesterday", 5),
        _Rule(re.compile(r"\bthis\s+week\b"), "week_to_date", 5),
        _Rule(re.compile(r"\blast\s+week\b"), "previous_week", 5),
        _Rule(re.compile(r"\bthis\s+month\b"), "month_to_date", 5),
        _Rule(re.compile(r"\blast\s+month\b"), "previous_month", 5),
        _Rule(re.compile(r"\bthis\s+(?:quarter|qtr)\b"), "quarter_to_date", 5),
        _Rule(re.compile(r"\blast\s+(?:quarter|qtr)\b"), "previous_quarter", 5),
        _Rule(re.compile(r"\b(?:this\s+year|year\s+to\s+date|ytd)\b"), "year_to_date", 5),
        _Rule(re.compile(r"\blast\s+year\b"), "previous_year", 5),
    ]
    return rules


_RULES = _build_rules()


def _resolve_fixed(label_key: str, anchor: date, matched: str) -> TemporalPeriod | None:
    """Resolve a non-parametric temporal keyword against the anchor date."""
    if label_key == "today":
        return _as_period("today", anchor, anchor, matched, anchor)
    if label_key == "yesterday":
        day = anchor - timedelta(days=1)
        return _as_period("yesterday", day, day, matched, anchor)
    if label_key == "week_to_date":
        start = anchor - timedelta(days=anchor.weekday())
        return _as_period("week_to_date", start, anchor, matched, anchor)
    if label_key == "previous_week":
        this_monday = anchor - timedelta(days=anchor.weekday())
        start = this_monday - timedelta(days=7)
        end = this_monday - timedelta(days=1)
        return _as_period("previous_week", start, end, matched, anchor)
    if label_key == "month_to_date":
        start, _ = _month_bounds(anchor)
        return _as_period("month_to_date", start, anchor, matched, anchor)
    if label_key == "previous_month":
        prev = _shift_month(anchor, -1)
        start, end = _month_bounds(prev)
        return _as_period("previous_month", start, end, matched, anchor)
    if label_key == "quarter_to_date":
        start, _ = _quarter_bounds(anchor)
        return _as_period("quarter_to_date", start, anchor, matched, anchor)
    if label_key == "previous_quarter":
        start, _ = _quarter_bounds(anchor)
        prev_end = start - timedelta(days=1)
        prev_start = _quarter_start(prev_end)
        return _as_period("previous_quarter", prev_start, prev_end, matched, anchor)
    if label_key == "year_to_date":
        start, _ = _year_bounds(anchor)
        return _as_period("year_to_date", start, anchor, matched, anchor)
    if label_key == "previous_year":
        year = anchor.year - 1
        return _as_period("previous_year", date(year, 1, 1), date(year, 12, 31), matched, anchor)
    return None


def _resolve_parametric(rule: _Rule, match: re.Match[str], anchor: date) -> TemporalPeriod | None:
    """Resolve a rule carrying numeric groups (rolling window, explicit day/month/quarter/year)."""
    matched = match.group(0).strip()
    if rule.builder == "rolling":
        amount = int(match.group(1))
        unit = match.group(2)
        if amount <= 0:
            return None
        if unit == "ngay":
            start = anchor - timedelta(days=amount - 1)
        elif unit == "tuan":
            start = anchor - timedelta(days=7 * amount - 1)
        elif unit == "thang":
            start = _shift_month(anchor, -amount)
        elif unit == "quy":
            start = _shift_month(anchor, -3 * amount)
        else:
            start = date(anchor.year - amount, 1, 1)
        return _as_period(f"rolling_{amount}_{unit}", start, anchor, matched, anchor)

    if rule.builder == "single_day":
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            target = date(year, month, day)
        except ValueError:
            return None
        return _as_period("single_day", target, target, matched, anchor)

    if rule.builder == "explicit_month":
        month, year = int(match.group(1)), int(match.group(2))
        if not 1 <= month <= 12:
            return None
        start, end = _month_bounds(date(year, month, 1))
        return _as_period(f"month_{year}_{month:02d}", start, end, matched, anchor)

    if rule.builder == "explicit_quarter":
        quarter, year = int(match.group(1)), int(match.group(2))
        if not 1 <= quarter <= 4:
            return None
        start = date(year, (quarter - 1) * 3 + 1, 1)
        _, end = _quarter_bounds(start)
        return _as_period(f"quarter_{year}_Q{quarter}", start, end, matched, anchor)

    if rule.builder == "explicit_year":
        year = int(match.group(1))
        if not 1900 <= year <= 2200:
            return None
        return _as_period(f"year_{year}", date(year, 1, 1), date(year, 12, 31), matched, anchor)

    return None


def resolve_temporal_periods(
    question: str,
    as_of_date: str | date | datetime | None,
) -> list[TemporalPeriod]:
    """Resolve every recognised relative temporal expression against the anchor date.

    Args:
        question: Natural language question (Vietnamese or English).
        as_of_date: Anchor date supplied with the question; resolution is a no-op without it.

    Returns:
        Deduplicated periods ordered from most to least specific; empty when the question
        carries no explicit temporal expression.
    """
    anchor = parse_iso_date(as_of_date)
    if anchor is None or not question or not question.strip():
        return []

    folded = _fold(question)
    found: list[tuple[int, int, TemporalPeriod]] = []
    seen_spans: list[tuple[int, int]] = []

    for rule in _RULES:
        for match in rule.pattern.finditer(folded):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in seen_spans):
                continue
            period = _resolve_parametric(rule, match, anchor) if match.groups() else _resolve_fixed(
                rule.builder, anchor, match.group(0).strip()
            )
            if period is None:
                continue
            if len(folded) == len(question):
                original = question[span[0] : span[1]].strip()
                if original:
                    period = TemporalPeriod(
                        label=period.label,
                        start=period.start,
                        end=period.end,
                        matched_text=original,
                    )
            seen_spans.append(span)
            found.append((rule.specificity, span[0], period))

    found.sort(key=lambda item: (-item[0], item[1]))

    periods: list[TemporalPeriod] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for _, _, period in found:
        key = (period.label, period.start, period.end)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        periods.append(period)
    return periods


def resolve_primary_period(
    question: str,
    as_of_date: str | date | datetime | None,
) -> TemporalPeriod | None:
    """Return the most specific resolved period, or None when the question is not time-bound."""
    periods = resolve_temporal_periods(question, as_of_date)
    return periods[0] if periods else None


def format_temporal_anchor_xml(
    question: str,
    as_of_date: str | date | datetime | None,
) -> str:
    """Render the <temporal_anchor> block shared by schema-linking and SQL prompts.

    Always emits the anchor date; adds one <period> line per resolved expression so the
    model never has to infer the concrete window on its own.
    """
    anchor = parse_iso_date(as_of_date)
    if anchor is None:
        return ""
    lines = ["<temporal_anchor>", f"as_of_date: {_iso(anchor)}"]
    for period in resolve_temporal_periods(question, anchor):
        lines.append(
            f'<period label="{period.label}" expression="{period.matched_text}" '
            f'start="{period.start}" end="{period.end}">{period.start} -> {period.end}</period>'
        )
    lines.append("</temporal_anchor>")
    return "\n".join(lines)


def temporal_context_summary(
    question: str,
    as_of_date: str | date | datetime | None,
) -> dict[str, object]:
    """Build a compact, JSON-serializable summary of the resolved temporal context."""
    anchor = parse_iso_date(as_of_date)
    periods = resolve_temporal_periods(question, anchor)
    return {
        "as_of_date": _iso(anchor) if anchor else None,
        "periods": [p.to_dict() for p in periods],
    }


__all__ = [
    "ISO_DATE_FORMAT",
    "TemporalPeriod",
    "format_temporal_anchor_xml",
    "parse_iso_date",
    "resolve_primary_period",
    "resolve_temporal_periods",
    "temporal_context_summary",
]
