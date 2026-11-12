"""Deterministic Maximal Profile Composer.

Combines minimal semantic profile with raw profiling statistics, data patterns,
unified view metadata, and filtered cross-table join candidates without LLM inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zaic_2026_4.constants import (
    DATE_FORMATS,
    KEY_TOKENS,
    MAXIMAL_DOMINANT_RATIO,
    MAXIMAL_ENUM_MAX_DISTINCT,
    MAXIMAL_MAX_JOINS_PER_COLUMN,
    MAXIMAL_MAX_VALUE_CHARS,
    MAXIMAL_TOP_VALUES,
    RANGE_ELIGIBLE_FORMATS,
)
from zaic_2026_4.profiling.raw_profiler import (
    ColumnProfile,
    DatabaseProfile,
    _extract_name_tokens,
)


@dataclass(frozen=True)
class TableContext:
    """Structural metadata for an individual table or view."""

    table_name: str
    row_count: int
    is_view: bool
    is_unified_view: bool
    source_tables: tuple[str, ...]
    note: str | None = None


@dataclass(frozen=True)
class JoinRef:
    """Filtered candidate join target for a column."""

    target_col: str
    resemblance: float
    is_target_unique: bool = False


def _truncate_val(val: Any, max_len: int = MAXIMAL_MAX_VALUE_CHARS) -> str:
    """Format and truncate a sample literal to prevent prompt overflow."""
    s = str(val).replace("\n", " ").strip()
    if len(s) > max_len:
        return f"{s[:max_len]}..."
    return s


def build_table_context(
    db_profile: DatabaseProfile | dict[str, Any],
) -> dict[str, TableContext]:
    """Extract structural context for all tables and views in the database profile."""
    ctx_map: dict[str, TableContext] = {}
    tables = db_profile.get("tables", {})
    for tbl_name, t_info in tables.items():
        ctx_map[tbl_name] = TableContext(
            table_name=tbl_name,
            row_count=t_info.get("row_count", 0),
            is_view=t_info.get("is_view", False),
            is_unified_view=t_info.get("is_unified_view", False),
            source_tables=tuple(t_info.get("source_tables", [])),
            note=t_info.get("note"),
        )
    return ctx_map


def build_join_index(
    db_profile: DatabaseProfile | dict[str, Any],
) -> dict[str, list[JoinRef]]:
    """Index candidate join relationships per column with noise filtering.

    Filters:
      - Discards spurious date-to-date join matches between distinct business tables
        (e.g., BanHang.order_date ~ DonNhapHang.import_date) while preserving joins
        with the dedicated 'calendar' lookup table.
      - Indexes both directions (col1 -> col2 and col2 -> col1).
      - Marks target column uniqueness (acting as PK candidate).
      - Sorts by resemblance descending and limits to MAXIMAL_MAX_JOINS_PER_COLUMN.
    """
    tables = db_profile.get("tables", {})
    raw_joins = db_profile.get("undocumented_joins", [])

    def _get_col_meta(key: str) -> tuple[str, str, dict[str, Any]]:
        parts = key.split(".", 1)
        t_name = parts[0]
        c_name = parts[1] if len(parts) > 1 else key
        tbl_info = tables.get(t_name, {})
        col_info = tbl_info.get("columns", {}).get(c_name, {})
        return t_name, c_name, col_info

    def _is_date_col(t_name: str, col_info: dict[str, Any]) -> bool:
        fmt = col_info.get("data_format", "")
        subtype = col_info.get("date_subtype")
        return bool(fmt in DATE_FORMATS or subtype)

    directed_joins: dict[str, list[JoinRef]] = {}

    for j in raw_joins:
        col1 = j.get("col1", "")
        col2 = j.get("col2", "")
        score = float(j.get("resemblance", 0.0))
        if not col1 or not col2:
            continue

        t1, _, cinfo1 = _get_col_meta(col1)
        t2, _, cinfo2 = _get_col_meta(col2)

        # Spurious date join suppression: drop business-to-business date joins
        is_date1 = _is_date_col(t1, cinfo1)
        is_date2 = _is_date_col(t2, cinfo2)
        if (
            is_date1
            and is_date2
            and t1.lower() != "calendar"
            and t2.lower() != "calendar"
        ):
            continue

        # Check target uniqueness
        t1_rows = tables.get(t1, {}).get("row_count", 0)
        t2_rows = tables.get(t2, {}).get("row_count", 0)

        c1_null = cinfo1.get("null_count", 0) or 0
        c1_dist = cinfo1.get("distinct_count", 0) or 0
        is_c1_unique = bool(c1_dist > 0 and c1_dist == (t1_rows - c1_null))

        c2_null = cinfo2.get("null_count", 0) or 0
        c2_dist = cinfo2.get("distinct_count", 0) or 0
        is_c2_unique = bool(c2_dist > 0 and c2_dist == (t2_rows - c2_null))

        # Add directed edge col1 -> col2
        directed_joins.setdefault(col1, []).append(
            JoinRef(target_col=col2, resemblance=score, is_target_unique=is_c2_unique)
        )
        # Add directed edge col2 -> col1
        directed_joins.setdefault(col2, []).append(
            JoinRef(target_col=col1, resemblance=score, is_target_unique=is_c1_unique)
        )

    # Sort and trim per column
    final_index: dict[str, list[JoinRef]] = {}
    for col_key, refs in directed_joins.items():
        # Deduplicate targets keeping highest resemblance
        dedup: dict[str, JoinRef] = {}
        for r in refs:
            if r.target_col not in dedup or r.resemblance > dedup[r.target_col].resemblance:
                dedup[r.target_col] = r
        sorted_refs = sorted(dedup.values(), key=lambda x: x.resemblance, reverse=True)
        final_index[col_key] = sorted_refs[:MAXIMAL_MAX_JOINS_PER_COLUMN]

    return final_index


def compose_maximal_profile(
    minimal_profile: str,
    table_name: str,
    col_name: str,
    col_stats: ColumnProfile | dict[str, Any],
    table_ctx: TableContext | None = None,
    joins: list[JoinRef] | None = None,
) -> str:
    """Deterministically assemble a comprehensive metadata profile string.

    Segments are combined in a canonical order:
      1. Minimal semantic summary (lead-in)
      2. Unified VIEW notice (if applicable)
      3. Format & subtype specifications
      4. Null distribution & empty status
      5. Cardinality, distinct counts, and primary-key / duplicate-code warnings
      6. Typed numerical / date range bounds (selected formats only)
      7. Full enumeration (distinct <= 10) or top frequent values
      8. Dominant value flag (if >= 90% concentration)
      9. Cross-table join candidates
    """
    segments: list[str] = []

    # 1. Minimal profile
    clean_min = minimal_profile.strip()
    if clean_min:
        segments.append(clean_min)

    # 2. Unified VIEW notice
    if table_ctx and table_ctx.is_unified_view and table_ctx.source_tables:
        sources_str = ", ".join(table_ctx.source_tables[:5])
        if len(table_ctx.source_tables) > 5:
            sources_str += "..."
        segments.append(
            f"Bảng thuộc VIEW hợp nhất {len(table_ctx.source_tables)} bảng phân vùng "
            f"({sources_str}); ưu tiên truy vấn VIEW này"
        )

    # 3. Format & pattern
    data_format = col_stats.get("data_format", "STRING") or "STRING"
    is_fixed_length = bool(col_stats.get("is_fixed_length"))
    min_str_len = col_stats.get("min_str_len")
    date_subtype = col_stats.get("date_subtype")
    currency_subtype = col_stats.get("currency_subtype")

    if is_fixed_length and min_str_len is not None:
        segments.append(f"Định dạng: {data_format}, độ dài {min_str_len} ký tự")
    elif date_subtype:
        segments.append(f"Định dạng: {data_format} ({date_subtype})")
    elif currency_subtype:
        segments.append(f"Định dạng: {data_format} ({currency_subtype})")
    else:
        segments.append(f"Định dạng: {data_format}")

    # 4. Null distribution
    null_count = int(col_stats.get("null_count", 0) or 0)
    total_rows = table_ctx.row_count if table_ctx else 0
    non_null_count = max(0, total_rows - null_count)

    is_completely_empty = bool(
        data_format == "EMPTY"
        or (total_rows > 0 and null_count >= total_rows)
        or (non_null_count == 0 and total_rows > 0)
    )

    if is_completely_empty:
        segments.append("Cột rỗng 100%, không sử dụng")
    elif total_rows > 0:
        if null_count == 0:
            segments.append(f"Không NULL ({total_rows:,} dòng)")
        else:
            pct = (null_count / total_rows) * 100
            segments.append(
                f"Chứa {null_count:,}/{total_rows:,} giá trị NULL ({pct:.1f}%)"
            )

    # 5. Cardinality & Key semantics
    distinct_count = int(col_stats.get("distinct_count", 0) or 0)

    if not is_completely_empty:
        if distinct_count == 1:
            segments.append("Cột hằng số (chỉ 1 giá trị duy nhất), không có ý nghĩa khi lọc/nhóm")
        elif distinct_count == non_null_count and non_null_count > 1:
            tokens = _extract_name_tokens(col_name)
            is_key_token = bool(tokens & KEY_TOKENS) or data_format == "FIXED_CODE"
            if is_key_token:
                segments.append(f"{distinct_count:,} giá trị duy nhất (ứng viên khóa chính)")
            else:
                segments.append(f"{distinct_count:,} giá trị khác nhau (không trùng lặp)")
        elif distinct_count > 1:
            tokens = _extract_name_tokens(col_name)
            is_key_token = bool(tokens & KEY_TOKENS) or data_format == "FIXED_CODE"
            if is_key_token and distinct_count < non_null_count:
                segments.append(
                    f"{distinct_count:,} giá trị khác nhau; mã lặp lại trên nhiều dòng "
                    "(dùng COUNT(DISTINCT ...) khi đếm)"
                )
            else:
                segments.append(f"{distinct_count:,} giá trị khác nhau")

    # 6. Range bounds (selected formats only)
    if (
        not is_completely_empty
        and distinct_count > 1
        and data_format in RANGE_ELIGIBLE_FORMATS
        and data_format != "DATE_MIXED"
    ):
        min_v = col_stats.get("min_val")
        max_v = col_stats.get("max_val")
        if (
            min_v is not None
            and max_v is not None
            and min_v != "N/A"
            and max_v != "N/A"
            and min_v != max_v
        ):
            segments.append(f"Khoảng: {_truncate_val(min_v)} → {_truncate_val(max_v)}")

    # 7. Sample values: full enumeration or top frequent
    top_samples: dict[str, int] = col_stats.get("top_10_samples") or {}

    if not is_completely_empty and distinct_count > 1 and top_samples:
        if distinct_count <= MAXIMAL_ENUM_MAX_DISTINCT:
            items_str = ", ".join(
                f"'{_truncate_val(k)}' ({v:,})" for k, v in top_samples.items()
            )
            segments.append(f"Tập giá trị đầy đủ: {items_str}")
        else:
            top_items = list(top_samples.items())[:MAXIMAL_TOP_VALUES]
            items_str = ", ".join(
                f"'{_truncate_val(k)}' ({v:,})" for k, v in top_items
            )
            segments.append(f"Giá trị phổ biến: {items_str}")

    # 8. Dominant value flag (>= 90%)
    if not is_completely_empty and distinct_count > 1 and non_null_count > 0 and top_samples:
        first_val, first_cnt = next(iter(top_samples.items()))
        ratio = first_cnt / non_null_count
        if ratio >= MAXIMAL_DOMINANT_RATIO:
            pct = round(ratio * 100)
            segments.append(f"Giá trị áp đảo: '{_truncate_val(first_val)}' (chiếm {pct}%)")

    # 9. Cross-table join candidates
    if joins:
        join_strs: list[str] = []
        for j in joins:
            parts = j.target_col.split(".", 1)
            if len(parts) == 2:
                formatted_target = f'"{parts[0]}"."{parts[1]}"'
            else:
                formatted_target = f'"{j.target_col}"'
            tag = ", khóa" if j.is_target_unique else ""
            join_strs.append(f"{formatted_target} (tương đồng {j.resemblance}{tag})")
        segments.append(f"Ứng viên Join: {'; '.join(join_strs)}")

    return " | ".join(segments)
