"""Canonical normalization layer: derived date/amount/status columns per table.

Raw SMB exports mix representations freely: the same date column can hold
``dd/mm/yyyy``, Excel serials and Vietnamese text dates; money can be stored as
``'58,000.0000'`` or ``'1.234.567đ'``; order status uses each shop's own
vocabulary. Asking the SQL generator to re-derive those rules in every query is
fragile, so this module harvests the representations **from the data itself** at
ingestion time and materialises three derived columns:

* ``_date``   -> ISO date string ``YYYY-MM-DD`` (NULL when unparseable)
* ``_amount`` -> numeric money value (NULL when unparseable)
* ``_status`` -> rulebook status family (``completed`` / ``shipping`` /
                 ``cancelled`` / ``returned`` / ``pending`` / ``other``)

The derived values live in a side table ``_canon_<table>`` and are exposed
through a view ``<table>__c`` that keeps every original column untouched, so
existing queries over the base table keep working. Nothing here is specific to a
dataset, shop or column name: every format is detected from the values at
runtime and unknown values degrade to NULL instead of a wrong guess.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, timedelta
from typing import Any

import polars as pl

CANONICAL_SUFFIX = "__c"
SIDE_TABLE_PREFIX = "_canon_"

DATE_COLUMN = "_date"
AMOUNT_COLUMN = "_amount"
STATUS_COLUMN = "_status"
CANONICAL_COLUMNS: tuple[str, str, str] = (DATE_COLUMN, AMOUNT_COLUMN, STATUS_COLUMN)

ROWID_COLUMN = "_rowid"

# Column-name tokens (diacritic-free) used only to decide which columns deserve a
# derived counterpart. They are generic business vocabulary, never shop-specific.
DATE_NAME_TOKENS: tuple[str, ...] = (
    "ngay", "date", "time", "thoi gian", "created", "tao luc", "han",
)
MONEY_NAME_TOKENS: tuple[str, ...] = (
    "tien", "amount", "total", "tong", "gia", "price", "doanh thu", "phi",
    "chi phi", "revenue", "gmv", "thanh toan", "phai tra", "cong no", "no",
    "giam gia", "chiet khau", "discount", "subtotal", "vat",
)
# Money-like column names ranked from the most to the least likely measure column.
MONEY_NAME_PRIORITY: tuple[tuple[str, ...], ...] = (
    ("tong tien", "thanh tien", "tong cong", "thanh toan", "phai tra", "total", "amount",
     "subtotal", "doanh thu", "revenue", "gmv", "cong no", "no phai tra"),
    ("tien", "phi", "chi phi", "giam gia", "chiet khau", "discount", "vat", "tien no"),
    ("gia", "price", "rate", "unit"),
)
# Quantity/measure vocabulary (diacritic-free). Used to keep measure columns out of
# structural decisions such as the merged-header block.
MEASURE_NAME_TOKENS: tuple[str, ...] = (
    "so luong", "sl", "qty", "quantity", "don gia", "gia", "price", "tien", "amount",
    "total", "tong", "thanh tien", "chiet khau", "giam gia", "vat", "quy doi",
)

STATUS_NAME_TOKENS: tuple[str, ...] = (
    "trang thai", "status", "tinh trang", "state", "kenh", "ghi chu", "payment",
    "thanh toan", "tinh hinh",
)

# Rulebook status families. Matching is diacritic/case-insensitive and ordered:
# the first family whose token occurs in the folded value wins, so stronger and
# more specific vocabulary is evaluated before the generic "delivered" tokens.
STATUS_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "cancelled",
        ("huy", "huy don", "huy bo", "cancel", "cancelled", "canceled", "tu choi",
         "da huy", "huy ngang"),
    ),
    (
        "returned",
        ("hoan tra", "tra hang", "tra lai", "hoan hang", "hoan tien", "return",
         "returned", "refund", "doi tra", "boom", "bom hang", "tra ve",
         "hang tra ve", "khong giao duoc", "giao that bai", "that lac"),
    ),
    (
        "completed",
        ("hoan thanh", "da giao", "giao thanh cong", "giao hang thanh cong",
         "thanh cong", "complete", "completed", "done", "success", "successful",
         "xong", "ok", "da nhan", "da mua", "da thanh toan", "finished"),
    ),
    (
        "shipping",
        ("dang giao", "dang van chuyen", "van chuyen", "dang ship", "shipping",
         "shipped", "in transit", "da lay", "lay hang", "dang trich", "giao hang",
         "dang xu ly giao", "delivering"),
    ),
    (
        "pending",
        ("cho xu ly", "cho xac nhan", "cho lay", "dang xu ly", "xu ly", "moi",
         "new", "pending", "processing", "xac nhan", "dang dong goi", "chua xu ly"),
    ),
)

_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T].*)?$")
_ISO_YMD_SLASH_RE = re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})$")
_DMY_SLASH_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
_DMY_SLASH_SHORT_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2})$")
_DMY_DASH_RE = re.compile(r"^(\d{1,2})-(\d{1,2})-(\d{4})$")
_VN_ABBR_RE = re.compile(r"^(\d{1,2})\s*(?:thg|thang|tháng)\.?\s*(\d{1,2})\s*,?\s*(\d{4})$", re.IGNORECASE)
_EN_ABBR_RE = re.compile(
    r"^(\d{1,2})\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*(\d{4})$",
    re.IGNORECASE,
)
_ISO_MONTH_RE = re.compile(r"^(\d{4})-(\d{1,2})$")
_DAY_MONTH_RE = re.compile(r"^(\d{1,2})/(\d{1,2})$")
_EXCEL_SERIAL_RE = re.compile(r"^(\d{5})(?:\.0+)?$")

_EN_MONTHS: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

EXCEL_SERIAL_MIN = 36526  # 2000-01-01
EXCEL_SERIAL_MAX = 55153  # 2050-12-31
EXCEL_EPOCH = date(1899, 12, 30)

_CURRENCY_NOISE_RE = re.compile(r"(đ|₫|vnd|vnđ|usd|\$|eur|€)\s*$", re.IGNORECASE)
_THOUSAND_COMMA_RE = re.compile(r"^\(?-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?$")
_THOUSAND_DOT_RE = re.compile(r"^\(?-?\d{1,3}(?:\.\d{3})+(?:,\d+)?\)?$")
_DECIMAL_COMMA_RE = re.compile(r"^\d+,\d{1,3}$")
_SCALED_MILLION_RE = re.compile(r"^(\d+(?:[.,]\d+)?)\s*(?:tr|trieu)\s*(\d{0,3})$", re.IGNORECASE)
_SCALED_THOUSAND_RE = re.compile(r"^(\d+(?:[.,]\d+)?)\s*(?:k|nghin)$", re.IGNORECASE)
_SCALED_BILLION_RE = re.compile(r"^(\d+(?:[.,]\d+)?)\s*(?:ty|ti)$", re.IGNORECASE)
_PLAIN_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def fold_ascii(value: str) -> str:
    """Lowercase and remove Vietnamese diacritics for vocabulary matching."""
    decomposed = unicodedata.normalize("NFD", str(value).lower())
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    # NFD keeps the Vietnamese letter đ (U+0111) intact, so fold it explicitly.
    stripped = stripped.replace("đ", "d")
    return re.sub(r"\s+", " ", stripped).strip()


def _contains_token(folded: str, token: str) -> bool:
    """Substring match for long tokens, whole-word match for short ambiguous ones."""
    if len(token) <= 3:
        return re.search(rf"(?:^|[^a-z0-9]){re.escape(token)}(?:$|[^a-z0-9])", folded) is not None
    return token in folded


def canonical_view_name(base_table: str) -> str:
    """Return the canonical view name of a base table."""
    return f"{base_table}{CANONICAL_SUFFIX}"


def side_table_name(base_table: str) -> str:
    """Return the sidecar table holding derived values of a base table."""
    return f"{SIDE_TABLE_PREFIX}{base_table}"


def is_canonical_view(name: str) -> bool:
    """True when the entity name denotes a canonical view."""
    return str(name).endswith(CANONICAL_SUFFIX)


def canonical_base_of(name: str) -> str | None:
    """Return the base table of a canonical view name, or None."""
    text = str(name)
    if not text.endswith(CANONICAL_SUFFIX):
        return None
    return text[: -len(CANONICAL_SUFFIX)]


def is_side_table(name: str) -> bool:
    """True when the entity name denotes an internal canonical side table."""
    return str(name).startswith(SIDE_TABLE_PREFIX)


def describe_canonical_column(table: str, column: str) -> str | None:
    """Deterministic profile text for a derived column, or None if not derived.

    The value is a fact about the ingestion step, not an inference, so it must not be
    delegated to an LLM summary. Texts stay within the schema description budget.
    """
    base = canonical_base_of(table)
    if base is None:
        return None
    if column == DATE_COLUMN:
        return (
            f"ISO YYYY-MM-DD của bảng {base} đã chuẩn hóa từ mọi định dạng ngày; "
            "dùng trực tiếp, không parse lại."
        )
    if column == AMOUNT_COLUMN:
        return (
            f"Số tiền của bảng {base} đã làm sạch và ép kiểu số; cộng trực tiếp, "
            "không REPLACE/CAST lại."
        )
    if column == STATUS_COLUMN:
        return (
            f"Nhóm trạng thái chuẩn rulebook của bảng {base} "
            "(completed/shipping/cancelled/returned/pending/other)."
        )
    return None


def _has_name_token(col_name: str, tokens: tuple[str, ...]) -> bool:
    folded = fold_ascii(col_name)
    return any(_contains_token(folded, tok) for tok in tokens)


def _valid_triple(year: int, month: int, day: int) -> bool:
    try:
        date(year, month, day)
    except ValueError:
        return False
    return True


def _iso_string(year: int, month: int, day: int) -> str | None:
    if not _valid_triple(year, month, day):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def classify_date_value(value: Any) -> str | None:
    """Classify one raw value into a date representation family, or None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if _ISO_DATE_RE.match(text):
        return "ISO_DATE"
    if _ISO_YMD_SLASH_RE.match(text):
        return "ISO_SLASH"
    if _DMY_SLASH_RE.match(text):
        return "SLASH_YEAR"
    if _DMY_SLASH_SHORT_RE.match(text):
        return "SLASH_SHORT_YEAR"
    if _DMY_DASH_RE.match(text):
        return "DASH_YEAR"
    if _VN_ABBR_RE.match(text):
        return "VN_ABBR"
    if _EN_ABBR_RE.match(text):
        return "EN_ABBR"
    if _ISO_MONTH_RE.match(text):
        return "ISO_MONTH"
    serial = _EXCEL_SERIAL_RE.match(text)
    if serial:
        number = int(serial.group(1))
        if EXCEL_SERIAL_MIN <= number <= EXCEL_SERIAL_MAX:
            return "EXCEL_SERIAL"
    return None


def detect_day_first(values: list[Any]) -> bool:
    """Infer day-first order for ambiguous d/m/yyyy columns (Vietnamese default)."""
    day_first_evidence = 0
    month_first_evidence = 0
    for value in values:
        if value is None:
            continue
        match = _DMY_SLASH_RE.match(str(value).strip()) or _DMY_DASH_RE.match(str(value).strip())
        if not match:
            continue
        first, second = int(match.group(1)), int(match.group(2))
        if first > 12 and second <= 12:
            day_first_evidence += 1
        elif second > 12 and first <= 12:
            month_first_evidence += 1
    return month_first_evidence <= day_first_evidence


def normalize_date_value(value: Any, day_first: bool = True) -> str | None:
    """Convert one raw value to an ISO ``YYYY-MM-DD`` string, or None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    match = _ISO_DATE_RE.match(text)
    if match:
        return _iso_string(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    match = _ISO_YMD_SLASH_RE.match(text)
    if match:
        return _iso_string(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    for pattern in (_DMY_SLASH_RE, _DMY_DASH_RE):
        match = pattern.match(text)
        if match:
            first, second, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
            day, month = (first, second) if day_first else (second, first)
            return _iso_string(year, month, day)

    match = _DMY_SLASH_SHORT_RE.match(text)
    if match:
        first, second, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        day, month = (first, second) if day_first else (second, first)
        return _iso_string(2000 + year, month, day)

    match = _VN_ABBR_RE.match(text)
    if match:
        return _iso_string(int(match.group(3)), int(match.group(2)), int(match.group(1)))

    match = _EN_ABBR_RE.match(text)
    if match:
        month = _EN_MONTHS.get(match.group(2)[:3].lower())
        if month is not None:
            return _iso_string(int(match.group(3)), month, int(match.group(1)))

    serial = _EXCEL_SERIAL_RE.match(text)
    if serial:
        number = int(serial.group(1))
        if EXCEL_SERIAL_MIN <= number <= EXCEL_SERIAL_MAX:
            return (EXCEL_EPOCH + timedelta(days=number)).isoformat()
    return None


def build_date_column(series: pl.Series) -> tuple[list[str | None], dict[str, Any]]:
    """Normalize a whole column to ISO dates and report the observed representation."""
    raw_values = series.to_list()
    layout_counts: dict[str, int] = {}
    for value in raw_values:
        family = classify_date_value(value)
        if family:
            layout_counts[family] = layout_counts.get(family, 0) + 1

    day_first = detect_day_first(raw_values)
    normalized = [normalize_date_value(value, day_first=day_first) for value in raw_values]

    # Values such as "1/1" carry no year. When the column also holds complete dates,
    # the narrative year is used so the partial rows stay usable instead of NULL.
    year_counts: dict[int, int] = {}
    for value in normalized:
        if value:
            year = int(value[:4])
            year_counts[year] = year_counts.get(year, 0) + 1
    inferred_rows = 0
    if year_counts:
        modal_year = max(year_counts.items(), key=lambda item: item[1])[0]
        for idx, value in enumerate(normalized):
            if value is not None:
                continue
            match = _DAY_MONTH_RE.match(str(raw_values[idx]).strip())
            if not match:
                continue
            day, month = int(match.group(1)), int(match.group(2))
            if not day_first:
                day, month = month, day
            normalized[idx] = _iso_string(modal_year, month, day)
            if normalized[idx] is not None:
                inferred_rows += 1

    filled = sum(1 for item in normalized if item)
    report = {
        "source_column": series.name,
        "formats": sorted(layout_counts),
        "day_first": day_first,
        "inferred_year_rows": inferred_rows,
        "normalized_rows": filled,
        "total_rows": len(raw_values),
    }
    return normalized, report


def classify_money_value(value: Any) -> str | None:
    """Classify one raw money value by its separator convention."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = _CURRENCY_NOISE_RE.sub("", text).strip()
    if not text:
        return None
    if _SCALED_MILLION_RE.match(text):
        return "SCALED_MILLION"
    if _SCALED_THOUSAND_RE.match(text):
        return "SCALED_THOUSAND"
    if _SCALED_BILLION_RE.match(text):
        return "SCALED_BILLION"
    if _THOUSAND_COMMA_RE.match(text):
        return "THOUSAND_COMMA"
    if _THOUSAND_DOT_RE.match(text):
        return "THOUSAND_DOT"
    if _DECIMAL_COMMA_RE.match(text):
        return "DECIMAL_COMMA"
    if _PLAIN_NUMBER_RE.match(text):
        return "PLAIN"
    return None


AMBIGUOUS_SEPARATOR_RE = re.compile(r"^\d{1,3}[.,]\d{3}$")


def is_ambiguous_money_value(value: Any) -> bool:
    """True for values like "1.234" / "1,234" that read differently per convention."""
    if value is None:
        return False
    text = _CURRENCY_NOISE_RE.sub("", str(value).strip()).strip().strip("()")
    return bool(AMBIGUOUS_SEPARATOR_RE.match(text))


def parse_money_value(value: Any, mode: str | None = None) -> float | None:
    """Parse one money value into a float using its separator convention."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = _CURRENCY_NOISE_RE.sub("", text).strip().strip("()")
    if not text:
        return None

    resolved = mode or classify_money_value(text)
    if resolved == "SCALED_MILLION":
        match = _SCALED_MILLION_RE.match(text)
        if not match:
            return None
        millions = parse_money_value(match.group(1), mode=None) or 0.0
        remainder_text = match.group(2) or ""
        remainder = float(remainder_text) if remainder_text else 0.0
        # "1tr440" means 1,000,000 + 440 * 1000: the trailing digits are thousands.
        remainder = remainder * (10 ** max(3 - len(remainder_text), 0))
        total = millions * 1_000_000 + remainder * 1000
        return -total if negative else total
    if resolved == "SCALED_THOUSAND":
        match = _SCALED_THOUSAND_RE.match(text)
        if not match:
            return None
        total = (parse_money_value(match.group(1), mode=None) or 0.0) * 1_000
        return -total if negative else total
    if resolved == "SCALED_BILLION":
        match = _SCALED_BILLION_RE.match(text)
        if not match:
            return None
        total = (parse_money_value(match.group(1), mode=None) or 0.0) * 1_000_000_000
        return -total if negative else total
    if resolved == "THOUSAND_COMMA":
        text = text.replace(",", "")
    elif resolved == "THOUSAND_DOT":
        text = text.replace(".", "").replace(",", ".")
    elif resolved == "DECIMAL_COMMA":
        text = text.replace(",", ".")
    elif resolved == "PLAIN":
        pass
    else:
        cleaned = re.sub(r"[^0-9.,-]", "", text)
        if not cleaned:
            return None
        if "," in cleaned and "." in cleaned:
            if cleaned.rfind(",") > cleaned.rfind("."):
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        elif cleaned.count(",") == 1 and len(cleaned.split(",")[-1]) <= 2:
            cleaned = cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "").replace(".", "")
        text = cleaned

    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return -number if negative else number


def build_amount_column(series: pl.Series) -> tuple[pl.Series, dict[str, Any]]:
    """Normalize a money column into a numeric polars Series.

    Values such as ``"1.234"`` carry no internal evidence of their own, so the column
    votes: a three-digit group is plausibly thousands, while a plain fractional value
    such as ``"12.5"`` can only be a decimal. Whichever reading the majority of values
    supports wins and ties favour the decimal reading. A column made *exclusively* of
    ambiguous three-digit values (``["1.500", "2.250"]``) is therefore indistinguishable
    from a decimal-comma column; it is read as thousands, consistently for every row, so
    downstream sums keep a single scale instead of mixing two.
    """
    raw_values = series.to_list()
    class_counts: dict[str, int] = {}
    for value in raw_values:
        family = classify_money_value(value)
        if family:
            class_counts[family] = class_counts.get(family, 0) + 1
    mode = (
        max(class_counts.items(), key=lambda item: item[1])[0] if class_counts else "UNKNOWN"
    )
    total_classified = sum(class_counts.values())
    # A column that genuinely mixes both thousands conventions cannot be normalized
    # safely: values outside the dominant convention become NULL instead of a wrong guess.
    conflicting_conventions = (
        class_counts.get("THOUSAND_COMMA", 0) / total_classified >= 0.1
        and class_counts.get("THOUSAND_DOT", 0) / total_classified >= 0.1
    )

    # Ambiguous values ("6,000" / "1.234") have no internal evidence of their own, so the
    # column decides: ambiguous rows plead "thousands", plain fractional rows plead
    # "decimal", and the majority wins while a tie favours the decimal reading.
    ambiguous_values = [value for value in raw_values if is_ambiguous_money_value(value)]
    fractional_plain_rows = 0
    for value in raw_values:
        if classify_money_value(value) != "PLAIN":
            continue
        text = _CURRENCY_NOISE_RE.sub("", str(value).strip()).strip()
        if "." in text or "," in text:
            fractional_plain_rows += 1
    ambiguous_as_thousands = len(ambiguous_values) > fractional_plain_rows

    ambiguous_styles: dict[str, int] = {}
    for value in ambiguous_values:
        family = classify_money_value(value)
        if family:
            ambiguous_styles[family] = ambiguous_styles.get(family, 0) + 1
    ambiguous_style = (
        max(ambiguous_styles.items(), key=lambda item: item[1])[0]
        if ambiguous_styles
        else None
    )

    def _as_decimal(text: str) -> float | None:
        """Read an ambiguous value as a decimal number, keeping an accounting minus sign."""
        cleaned = _CURRENCY_NOISE_RE.sub("", text.strip()).strip()
        negative = cleaned.startswith("(") and cleaned.endswith(")")
        cleaned = cleaned.strip("()")
        if cleaned.count(",") == 1 and "." not in cleaned:
            cleaned = cleaned.replace(",", ".")
        try:
            number = float(cleaned)
        except (TypeError, ValueError):
            return None
        return -number if negative else number

    parsed: list[float | None] = []
    for value in raw_values:
        family = classify_money_value(value)
        if conflicting_conventions and family is not None and family != mode:
            parsed.append(None)
            continue
        if is_ambiguous_money_value(value):
            if not ambiguous_as_thousands:
                parsed.append(_as_decimal(str(value)))
            elif ambiguous_style is not None and family != ambiguous_style:
                parsed.append(None)
            else:
                parsed.append(parse_money_value(value, mode=ambiguous_style))
            continue
        # Each value carries its own convention: '3200' can sit next to '6,000'.
        parsed.append(parse_money_value(value, mode=None))

    usable = [item for item in parsed if item is not None]
    all_integral = all(abs(item - round(item)) < 1e-9 for item in usable) if usable else True
    if all_integral:
        data = [None if item is None else round(item) for item in parsed]
        result = pl.Series(AMOUNT_COLUMN, data, dtype=pl.Int64)
    else:
        result = pl.Series(AMOUNT_COLUMN, parsed, dtype=pl.Float64)
    report = {
        "source_column": series.name,
        "mode": mode,
        "dtype": "INTEGER" if all_integral else "REAL",
        "parsed_rows": len(usable),
        "total_rows": len(raw_values),
        "ambiguous_rows": len(ambiguous_values),
        "ambiguous_as_thousands": ambiguous_as_thousands,
    }
    return result, report


def canonical_status(raw: Any) -> str | None:
    """Map one raw status value onto a rulebook family."""
    if raw is None:
        return None
    folded = fold_ascii(str(raw))
    if not folded:
        return None
    for family, tokens in STATUS_FAMILIES:
        if any(_contains_token(folded, token) for token in tokens):
            return family
    return "other"


def build_status_column(series: pl.Series) -> tuple[pl.Series, dict[str, Any]]:
    """Map a free-text status column onto rulebook families."""
    raw_values = series.to_list()
    families = [canonical_status(value) for value in raw_values]
    mapping: dict[str, str] = {}
    for raw, family in zip(raw_values, families, strict=False):
        if raw is None or family is None:
            continue
        key = str(raw).strip()
        if key and key not in mapping:
            mapping[key] = family
    mapped = sum(1 for family in families if family and family != "other")
    result = pl.Series(STATUS_COLUMN, families, dtype=pl.String)
    report = {
        "source_column": series.name,
        "distinct_values": len(mapping),
        "mapped_rows": mapped,
        "total_rows": len(raw_values),
        "vocabulary": dict(sorted(mapping.items())[:40]),
    }
    return result, report


def _looks_like_date_column(series: pl.Series) -> bool:
    values = [value for value in series.head(400).to_list() if value is not None and str(value).strip()]
    if not values:
        return False
    recognized = sum(1 for value in values if classify_date_value(value) is not None)
    return recognized / len(values) >= 0.6


def _money_name_rank(name: str) -> int | None:
    """Rank a column name by how strongly it denotes a money measure.

    Total-like names outrank generic ones so that a line total is preferred over a
    unit price when both exist in the same table.
    """
    folded = fold_ascii(name)
    for rank, tokens in enumerate(MONEY_NAME_PRIORITY):
        if any(_contains_token(folded, tok) for tok in tokens):
            return rank
    return None


def is_measure_column(name: str, series: pl.Series) -> bool:
    """True when a column holds a measurement rather than a descriptive attribute.

    Measures (quantity, price, amount) are never part of a merged header block: carrying
    them down would fabricate numbers, so they must stay out of structural decisions.
    """
    if series.dtype.is_numeric():
        return True
    if _has_name_token(name, MEASURE_NAME_TOKENS):
        return True
    values = [value for value in series.head(200).to_list() if value is not None and str(value).strip()]
    if not values:
        return False
    money_like = sum(
        1
        for value in values
        if classify_money_value(value) in ("THOUSAND_COMMA", "THOUSAND_DOT", "DECIMAL_COMMA", "SCALED_MILLION")
    )
    return money_like / len(values) >= 0.5


def _looks_like_money_column(name: str, series: pl.Series) -> bool:
    if series.dtype.is_numeric():
        return _has_name_token(name, MONEY_NAME_TOKENS)
    values = [value for value in series.head(400).to_list() if value is not None and str(value).strip()]
    if not values:
        return False
    decorated = sum(1 for value in values if classify_money_value(value) in ("THOUSAND_COMMA", "THOUSAND_DOT", "DECIMAL_COMMA"))
    if decorated / len(values) >= 0.5:
        return True
    return _has_name_token(name, MONEY_NAME_TOKENS) and all(
        classify_money_value(value) is not None for value in values
    )


def status_column_score(name: str, series: pl.Series) -> tuple[int, float] | None:
    """Score how well a column looks like a status column, or None when it does not.

    The decision is value driven: at least 60% of the distinct values must map onto
    rulebook status families, and at least two different families must occur. The
    column name only ranks equally valid candidates.
    """
    if series.dtype.is_numeric() or series.dtype in (pl.Date, pl.Datetime, pl.Boolean):
        return None
    values = [str(value).strip() for value in series.drop_nulls().unique().to_list()]
    if not 2 <= len(values) <= 40:
        return None
    families = [canonical_status(value) for value in values]
    mapped = [family for family in families if family and family != "other"]
    if not mapped or len(set(mapped)) < 2:
        return None
    ratio = len(mapped) / len(families)
    if ratio < 0.6:
        return None
    name_rank = 0 if _has_name_token(name, STATUS_NAME_TOKENS) else 1
    return (name_rank, -ratio)


def _looks_like_status_column(name: str, series: pl.Series) -> bool:
    return status_column_score(name, series) is not None


def scan_canonical_specs(df: pl.DataFrame) -> dict[str, dict[str, str]]:
    """Detect which columns of a table deserve derived date/amount/status columns."""
    spec: dict[str, str] = {}
    amount_candidates: list[tuple[int, float, str]] = []
    status_candidates: list[tuple[int, float, str]] = []

    # A native Date/Datetime column is unambiguous, so it wins over any string column.
    native_date_columns = [
        name for name in df.columns if df[name].dtype in (pl.Date, pl.Datetime)
    ]
    if native_date_columns:
        native_date_columns.sort(key=lambda name: 0 if _has_name_token(name, DATE_NAME_TOKENS) else 1)
        spec[DATE_COLUMN] = native_date_columns[0]

    for name in df.columns:
        series = df[name]
        if DATE_COLUMN not in spec:
            date_named = _has_name_token(name, DATE_NAME_TOKENS) or not series.dtype.is_numeric()
            if date_named and _looks_like_date_column(series):
                known = [
                    value for value in series.head(400).to_list()
                    if value is not None and str(value).strip()
                ]
                identity = all(classify_date_value(value) == "ISO_DATE" for value in known)
                # Only keep the derived column when it actually changes something: an
                # already-ISO column needs no normalization.
                if not identity:
                    spec[DATE_COLUMN] = name
        if _looks_like_money_column(name, series):
            rank = _money_name_rank(name)
            if rank is not None:
                coverage = 1.0 - (series.null_count() / max(series.len(), 1))
                amount_candidates.append((rank, -coverage, name))
        status_score = status_column_score(name, series)
        if status_score is not None:
            status_candidates.append((*status_score, name))

    if amount_candidates:
        amount_candidates.sort()
        spec[AMOUNT_COLUMN] = amount_candidates[0][2]
    if status_candidates:
        status_candidates.sort()
        spec[STATUS_COLUMN] = status_candidates[0][2]
    return spec


def build_canonical_sidecar(
    df: pl.DataFrame,
    spec: dict[str, str],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Build the derived-values table for one base table.

    Returns the sidecar DataFrame keyed by ``_rowid`` plus a report describing the
    detected representations (stored for traceability and prompt-free debugging).
    """
    columns: list[pl.Series] = [
        pl.Series(ROWID_COLUMN, list(range(1, df.height + 1)), dtype=pl.Int64)
    ]
    report: dict[str, Any] = {"source_table": None, "date": None, "amount": None, "status": None}

    date_source = spec.get(DATE_COLUMN)
    if date_source and date_source in df.columns:
        values, date_report = build_date_column(df[date_source])
        columns.append(pl.Series(DATE_COLUMN, values, dtype=pl.String))
        report["date"] = date_report

    amount_source = spec.get(AMOUNT_COLUMN)
    if amount_source and amount_source in df.columns:
        amount_series, amount_report = build_amount_column(df[amount_source])
        columns.append(amount_series)
        report["amount"] = amount_report

    status_source = spec.get(STATUS_COLUMN)
    if status_source and status_source in df.columns:
        status_series, status_report = build_status_column(df[status_source])
        columns.append(status_series)
        report["status"] = status_report

    sidecar = pl.DataFrame(columns)
    return sidecar, report


def plan_canonical_tables(
    tables: dict[str, pl.DataFrame],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    """Detect canonical specs for every table.

    Returns ``(specs, reports)`` where specs maps table name to
    ``{canonical_column: source_column}`` and reports carries the detection detail.
    Tables without any detectable representation are omitted from both mappings.
    """
    specs: dict[str, dict[str, str]] = {}
    reports: dict[str, dict[str, Any]] = {}
    for name, df in tables.items():
        if df.is_empty() or len(df.columns) == 0:
            continue
        if is_side_table(name) or is_canonical_view(name):
            continue
        spec = scan_canonical_specs(df)
        if not spec:
            continue
        specs[name] = spec
        reports[name] = {"source_table": name, "columns": dict(spec)}
    return specs, reports


def generate_canonical_views(
    tables: dict[str, pl.DataFrame],
    specs: dict[str, dict[str, str]],
) -> list[tuple[str, str]]:
    """Generate ``(view_name, view_sql)`` pairs for the detected canonical tables."""
    views: list[tuple[str, str]] = []
    for base_table, spec in specs.items():
        df = tables.get(base_table)
        if df is None:
            continue
        derived = [col for col in CANONICAL_COLUMNS if col in spec]
        if not derived:
            continue
        quoted_base_cols = ", ".join(f't."{c}"' for c in df.columns)
        derived_projection = ", ".join(f'c."{col}" AS "{col}"' for col in derived)
        view_sql = (
            f'CREATE VIEW "{canonical_view_name(base_table)}" AS '
            f"SELECT {quoted_base_cols}, {derived_projection} "
            f'FROM "{base_table}" AS t '
            f'LEFT JOIN "{side_table_name(base_table)}" AS c ON c."{ROWID_COLUMN}" = t.rowid'
        )
        views.append((canonical_view_name(base_table), view_sql))
    return views


def generate_canonical_unified_views(
    unified_groups: list[list[str]],
    specs: dict[str, dict[str, str]],
    view_names: dict[str, str],
    tables: dict[str, pl.DataFrame] | None = None,
) -> list[tuple[str, str]]:
    """Union the canonical views of consolidating partitions into one canonical view.

    A unified view hides its partition tables from profiling, so without this step the
    model would only see one canonical view per month. ``view_names`` maps a member table
    to the unified view name that contains it.
    """
    views: list[tuple[str, str]] = []
    by_view: dict[str, list[str]] = {}
    for members in unified_groups:
        if not members:
            continue
        unified_name = view_names.get(members[0])
        if not unified_name:
            continue
        by_view[unified_name] = members

    for unified_name, members in by_view.items():
        member_specs = [tuple(sorted(specs.get(name, {}))) for name in members]
        if not member_specs or not member_specs[0]:
            continue
        if any(spec != member_specs[0] for spec in member_specs):
            # Members expose different derived columns, so a UNION would be ambiguous.
            continue
        # Explicit column lists: partitions can declare the same columns in a different
        # order, and a bare SELECT * would then silently misalign the union.
        canonical_columns = [f'"{col}"' for col in CANONICAL_COLUMNS if col in member_specs[0]]
        if tables is not None:
            base_frame = tables.get(members[0])
            if base_frame is not None:
                canonical_columns = [f'"{col}"' for col in base_frame.columns] + canonical_columns
        projection = ", ".join(canonical_columns)
        selects = [
            f'SELECT {projection} FROM "{canonical_view_name(name)}"'
            for name in sorted(members)
            if name in specs
        ]
        if not selects:
            continue
        views.append(
            (
                canonical_view_name(unified_name),
                f'CREATE VIEW "{canonical_view_name(unified_name)}" AS '
                + " UNION ALL ".join(selects),
            )
        )
    return views


__all__ = [
    "AMOUNT_COLUMN",
    "CANONICAL_COLUMNS",
    "CANONICAL_SUFFIX",
    "DATE_COLUMN",
    "STATUS_COLUMN",
    "STATUS_FAMILIES",
    "build_amount_column",
    "build_canonical_sidecar",
    "build_date_column",
    "build_status_column",
    "canonical_base_of",
    "canonical_status",
    "canonical_view_name",
    "classify_date_value",
    "classify_money_value",
    "describe_canonical_column",
    "detect_day_first",
    "fold_ascii",
    "generate_canonical_unified_views",
    "generate_canonical_views",
    "is_ambiguous_money_value",
    "is_canonical_view",
    "is_measure_column",
    "is_side_table",
    "normalize_date_value",
    "parse_money_value",
    "plan_canonical_tables",
    "scan_canonical_specs",
    "side_table_name",
]
