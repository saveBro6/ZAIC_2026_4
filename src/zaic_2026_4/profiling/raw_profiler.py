"""Raw database profiling, Shape & Pattern recognition, and MinHash undocumented join discovery."""

import json
from pathlib import Path
import re
import sqlite3
from typing import Any, TypedDict

from datasketch import MinHash
import polars as pl


class ColumnProfile(TypedDict):
    """Detailed profile metrics and Shape & Pattern features for an individual column."""

    # Core data contract keys
    null_count: int
    distinct_count: int
    null_ratio: float
    min_val: str | None
    max_val: str | None
    min_str_len: int | None
    max_str_len: int | None
    data_format: str
    is_json: bool
    top_10_samples: dict[str, int]
    sample_values: list[str]

    # Enriched Shape & Pattern attributes
    is_fixed_length: bool
    is_free_text: bool
    date_subtype: str | None
    currency_subtype: str | None
    currency_scale_multiplier: float | None
    llm_hint: str | None


class TableProfile(TypedDict):
    """Profile metrics for an individual database table or view."""

    row_count: int
    is_view: bool
    is_unified_view: bool
    source_tables: list[str]
    columns: dict[str, ColumnProfile]
    note: str | None


class UndocumentedJoin(TypedDict):
    """Candidate foreign-key/primary-key relationship discovered via MinHash Jaccard resemblance."""

    col1: str
    col2: str
    resemblance: float


class DatabaseProfile(TypedDict):
    """Aggregated profile for a full database containing tables and candidate join paths."""

    tables: dict[str, TableProfile]
    undocumented_joins: list[UndocumentedJoin]


# Regex patterns for Excel Custom Date Taxonomy
ISO_DATE_REGEX = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
ISO_DATETIME_REGEX = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}[ T]\d{1,2}:\d{2}(?::\d{2})?$")
SLASH_DATE_REGEX = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
SLASH_DATE_SHORT_REGEX = re.compile(r"^\d{1,2}/\d{1,2}/\d{2}$")
VN_DATE_ABBR_REGEX = re.compile(r"^\d{1,2}\s+thg\s+\d{1,2}(?:\s+\d{4})?$", re.IGNORECASE)
EN_DATE_ABBR_REGEX = re.compile(
    r"^\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2,4}$",
    re.IGNORECASE,
)
VN_DATE_FULL_REGEX = re.compile(
    r"^(?:ngày\s+)?\d{1,2}\s+tháng\s+\d{1,2}(?:\s+năm\s+\d{4})?$",
    re.IGNORECASE,
)
EN_DATE_FULL_REGEX = re.compile(
    r"^(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*|\s+)\d{4}$",
    re.IGNORECASE,
)
VN_DATE_WEEKDAY_REGEX = re.compile(
    r"^(?:thứ\s+(?:hai|ba|tư|bốn|năm|sáu|bảy|[2-7])|chủ\s+nhật|cn)[,\s]+\d{1,2}/\d{1,2}/\d{4}$",
    re.IGNORECASE,
)
EN_DATE_WEEKDAY_REGEX = re.compile(
    r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)[,\s]+\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$",
    re.IGNORECASE,
)
EXCEL_SERIAL_REGEX = re.compile(r"^\d{5}$")

# Regex patterns for Excel Custom VNĐ Currency Taxonomy
VND_TEXT_REGEX = re.compile(
    r"^-?\d{1,3}(?:[,\.]\d{3})*(?:\.\d+)?\s*(?:vnđ|vnd)$",
    re.IGNORECASE,
)
VND_SYMBOL_REGEX = re.compile(
    r"^-?\d{1,3}(?:[,\.]\d{3})*(?:\.\d+)?\s*(?:₫|đ)$",
    re.IGNORECASE,
)
ACCOUNTING_PARENTHESES_REGEX = re.compile(
    r"^\(\s*\d{1,3}(?:[,\.]\d{3})*(?:\.\d+)?\s*(?:₫|đ|vnđ|vnd)?\s*\)$",
    re.IGNORECASE,
)
ACCOUNTING_DASH_ZERO_REGEX = re.compile(
    r"^-\s*(?:₫|đ|vnđ|vnd)?$",
    re.IGNORECASE,
)
SCALED_THOUSAND_K_REGEX = re.compile(
    r"^-?\d{1,3}(?:[,\.]\d{3})*(?:\.\d+)?\s*(?:k\s*vnđ|k\s*vnd|k)$",
    re.IGNORECASE,
)
SCALED_MILLION_TR_REGEX = re.compile(
    r"^-?\d+(?:[,\.]\d+)?\s*(?:tr\s*vnđ|tr\s*vnd|tr|triệu)$",
    re.IGNORECASE,
)
SCALED_BILLION_TY_REGEX = re.compile(
    r"^-?\d+(?:[,\.]\d+)?\s*(?:tỷ|ty)$",
    re.IGNORECASE,
)
STANDARD_THOUSAND_SEPARATOR_REGEX = re.compile(r"^-?\d{1,3}(?:[,\.]\d{3})+(?:\.\d+)?$")
PLAIN_NUMBER_REGEX = re.compile(r"^-?\d+(?:\.\d+)?$")

# SQL FROM clause parser handling quoted identifiers with spaces/Vietnamese diacritics
VIEW_FROM_REGEX = re.compile(
    r'FROM\s+(?:\"([^\"]+)\"|\`([^\`]+)\`|\[([^\]]+)\]|([^\s,;]+))',
    re.IGNORECASE,
)

# Token set for non-regex MinHash continuous numeric filtering
KEY_TOKENS = frozenset({"id", "code", "stt", "key", "sku", "ref", "ma", "mã"})
NON_KEY_METRIC_TOKENS = frozenset({
    "tien", "tiền", "gia", "giá", "luong", "lượng", "phi", "phí",
    "vat", "thue", "thuế", "du", "dư", "discount", "amount", "price",
    "revenue", "cost", "total", "qty", "quantity", "count"
})


def _extract_name_tokens(col_name: str) -> set[str]:
    """Tokenize column name by standard whitespace and punctuation without regex."""
    cleaned = col_name.lower()
    for delimiter in ("_", "-", " ", ".", "/", "(", ")", "[", "]"):
        cleaned = cleaned.replace(delimiter, " ")
    return set(cleaned.split())


class RawDatabaseProfiler:
    """Extracts raw quantitative statistics, patterns, and MinHash sketches from SQLite databases."""

    def __init__(
        self,
        db_path: str | Path,
        num_perm: int = 128,
        include_views: bool = True,
        skip_partition_tables_if_view_exists: bool = True,
        cross_table_only: bool = True,
        jaccard_threshold: float = 0.7,
        max_sample_values: int = 10000,
        exclude_continuous_numerics: bool = True,
        skip_calendar_table: bool = True,
    ) -> None:
        """Initialize the profiler with configuration options.

        Args:
            db_path: Path to the SQLite database file.
            num_perm: Number of permutations for MinHash sketch computation.
            include_views: Whether to profile SQLite VIEWs.
            skip_partition_tables_if_view_exists: If True, skips redundant monthly partition
                tables when a unified VIEW consolidates them.
            cross_table_only: If True, only computes Jaccard similarity across distinct tables.
            jaccard_threshold: Minimum Jaccard resemblance to consider a candidate FK-PK join path.
            max_sample_values: Maximum number of unique values to sample for MinHash and LSH.
            exclude_continuous_numerics: Whether to exclude continuous floating-point amounts
                from MinHash Jaccard similarity to prevent false-positive joins.
            skip_calendar_table: Whether to skip table 'calendar' if present in the database,
                allowing it to be loaded from the split-level shared cache.
        """
        self.db_path = Path(db_path) if db_path else Path("")
        self.num_perm = num_perm
        self.include_views = include_views
        self.skip_partition_tables_if_view_exists = skip_partition_tables_if_view_exists
        self.cross_table_only = cross_table_only
        self.jaccard_threshold = jaccard_threshold
        self.max_sample_values = max_sample_values
        self.exclude_continuous_numerics = exclude_continuous_numerics
        self.skip_calendar_table = skip_calendar_table

    def profile_database(self) -> DatabaseProfile:
        """Profile all tables and candidate joins in the database.

        Returns:
            DatabaseProfile matching downstream contract for summarizers and indexers.
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        try:
            return self._profile_with_connection(conn)
        finally:
            conn.close()

    def _profile_with_connection(self, conn: sqlite3.Connection) -> DatabaseProfile:
        """Internal execution with an active SQLite connection."""
        cursor = conn.cursor()

        # Query all user-defined tables and views
        cursor.execute(
            "SELECT name, type, sql FROM sqlite_master "
            "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' "
            "ORDER BY type DESC, name ASC;"
        )
        entities = cursor.fetchall()

        table_names = [name for name, typ, _ in entities if typ == "table"]
        view_info: dict[str, list[str]] = {}

        for name, typ, sql in entities:
            if typ == "view" and sql:
                matches = VIEW_FROM_REGEX.findall(sql)
                sources = []
                for m in matches:
                    matched_name = m[0] or m[1] or m[2] or m[3]
                    if matched_name and matched_name in table_names:
                        sources.append(matched_name)
                view_info[name] = sources

        # Identify partition tables to skip if covered by unified views
        partition_tables_to_skip: set[str] = set()
        if self.skip_partition_tables_if_view_exists and self.include_views:
            for sources in view_info.values():
                if len(sources) >= 2:
                    partition_tables_to_skip.update(sources)

        # Build list of entities to profile
        targets_to_profile: list[tuple[str, str, bool, list[str]]] = []
        for name, typ, _ in entities:
            if typ == "view":
                if not self.include_views:
                    continue
                is_unified = len(view_info.get(name, [])) >= 2
                targets_to_profile.append((name, typ, is_unified, view_info.get(name, [])))
            elif typ == "table":
                if self.skip_calendar_table and name.lower() == "calendar":
                    continue
                if name in partition_tables_to_skip:
                    continue
                targets_to_profile.append((name, typ, False, []))

        db_profile: DatabaseProfile = {"tables": {}, "undocumented_joins": []}
        minhashes: dict[str, MinHash] = {}

        for entity_name, entity_type, is_unified, sources in targets_to_profile:
            table_profile, table_minhashes = self._profile_single_entity(
                conn, entity_name, entity_type, is_unified, sources
            )
            db_profile["tables"][entity_name] = table_profile
            minhashes.update(table_minhashes)

        # Detect undocumented joins via MinHash Jaccard similarity
        db_profile["undocumented_joins"] = self._detect_undocumented_joins(minhashes)
        return db_profile

    def _profile_single_entity(
        self,
        conn: sqlite3.Connection,
        entity_name: str,
        entity_type: str,
        is_unified: bool,
        sources: list[str],
    ) -> tuple[TableProfile, dict[str, MinHash]]:
        """Profile an individual table or view using Polars."""
        try:
            df = pl.read_database(
                f'SELECT * FROM "{entity_name}"', conn, infer_schema_length=None
            )
        except Exception:
            # Fallback to cursor fetch with string schema for heterogeneous SQLite columns
            cur = conn.cursor()
            cur.execute(f'SELECT * FROM "{entity_name}"')
            col_names = [d[0] for d in cur.description]
            rows = cur.fetchall()
            df = pl.DataFrame(
                [[str(v) if v is not None else None for v in r] for r in rows],
                schema={c: pl.String for c in col_names},
                orient="row",
            )
        row_count = df.height
        is_view = entity_type == "view"

        note = None
        if is_unified and sources:
            note = (
                f"Unified VIEW consolidating {len(sources)} partition tables: "
                f"{', '.join(sources[:5])}{'...' if len(sources) > 5 else ''}. "
                "Prioritize querying this consolidated view."
            )

        table_profile: TableProfile = {
            "row_count": row_count,
            "is_view": is_view,
            "is_unified_view": is_unified,
            "source_tables": sources,
            "columns": {},
            "note": note,
        }

        table_minhashes: dict[str, MinHash] = {}

        for col in df.columns:
            col_prof, minhash_obj = self._profile_single_column(
                df[col], col, row_count, entity_name
            )
            table_profile["columns"][col] = col_prof
            if minhash_obj is not None:
                table_minhashes[f"{entity_name}.{col}"] = minhash_obj

        return table_profile, table_minhashes

    def _profile_single_column(
        self,
        series_raw: pl.Series,
        col_name: str,
        total_rows: int,
        entity_name: str,
    ) -> tuple[ColumnProfile, MinHash | None]:
        """Profile a single column's statistical properties and MinHash sketch."""
        null_count = int(series_raw.null_count())
        null_ratio = round(null_count / total_rows, 4) if total_rows > 0 else 0.0

        non_null_raw = series_raw.drop_nulls()
        non_null_len = non_null_raw.len()

        if non_null_len == 0:
            empty_profile: ColumnProfile = {
                "null_count": null_count,
                "distinct_count": 0,
                "null_ratio": null_ratio,
                "min_val": None,
                "max_val": None,
                "min_str_len": None,
                "max_str_len": None,
                "is_fixed_length": False,
                "is_free_text": False,
                "date_subtype": None,
                "currency_subtype": None,
                "currency_scale_multiplier": None,
                "data_format": "EMPTY",
                "is_json": False,
                "llm_hint": "Column is completely empty (100% NULL).",
                "top_10_samples": {},
                "sample_values": [],
            }
            return empty_profile, None

        # Cast non-null series to string for universal length and sample profiling
        series_str = non_null_raw.cast(pl.String)
        distinct_count = int(series_str.n_unique())

        # Compute typed min/max to avoid lexicographical inversion for numbers/dates
        if non_null_raw.dtype.is_numeric() or non_null_raw.dtype in (
            pl.Date,
            pl.Datetime,
            pl.Time,
            pl.Duration,
        ):
            min_val = str(non_null_raw.min())
            max_val = str(non_null_raw.max())
        else:
            first_val = series_str[0]
            first_val_clean = first_val.lstrip("-").strip() if first_val else ""
            if first_val_clean and (first_val_clean.isdigit() or first_val_clean.replace(".", "", 1).isdigit()):
                try:
                    num_series = non_null_raw.cast(pl.Float64)
                    min_val = str(num_series.min())
                    max_val = str(num_series.max())
                except Exception:
                    min_val = str(series_str.min())
                    max_val = str(series_str.max())
            else:
                min_val = str(series_str.min())
                max_val = str(series_str.max())

        # String length bounds
        char_lens = series_str.str.len_chars()
        min_str_len = int(char_lens.min()) if char_lens.len() > 0 else None
        max_str_len = int(char_lens.max()) if char_lens.len() > 0 else None
        is_fixed_length = bool(min_str_len == max_str_len and min_str_len is not None and min_str_len > 0)
        is_free_text = bool(max_str_len is not None and max_str_len > 255)

        # Top 10 frequent samples using non-colliding alias to avoid Polars DuplicateError on column named 'count'
        vc_df = (
            series_str.alias("_raw_val")
            .value_counts(sort=True, name="_frequency")
            .head(10)
        )
        vc_keys = [str(k) for k in vc_df["_raw_val"].to_list()]
        vc_counts = [int(c) for c in vc_df["_frequency"].to_list()]
        top_10 = dict(zip(vc_keys, vc_counts, strict=False))

        # Sample values for LSH indexing (capped at max_sample_values)
        unique_samples = series_str.unique().head(self.max_sample_values).to_list()

        # Shape & pattern recognition on head sample
        shape_info = self._infer_format_and_pattern(
            series_raw.dtype, series_str.head(50), is_fixed_length, is_free_text
        )

        col_profile: ColumnProfile = {
            "null_count": null_count,
            "distinct_count": distinct_count,
            "null_ratio": null_ratio,
            "min_val": min_val,
            "max_val": max_val,
            "min_str_len": min_str_len,
            "max_str_len": max_str_len,
            "is_fixed_length": is_fixed_length,
            "is_free_text": is_free_text,
            "date_subtype": shape_info["date_subtype"],
            "currency_subtype": shape_info["currency_subtype"],
            "currency_scale_multiplier": shape_info["currency_scale_multiplier"],
            "data_format": shape_info["data_format"],
            "is_json": shape_info["is_json"],
            "llm_hint": shape_info["llm_hint"],
            "top_10_samples": top_10,
            "sample_values": unique_samples,
        }

        # MinHash sketch evaluation
        minhash_obj = None
        if self._is_eligible_for_minhash(
            series_raw.dtype, col_name, shape_info["data_format"], distinct_count, total_rows
        ):
            minhash_obj = MinHash(num_perm=self.num_perm)
            for val in unique_samples:
                if len(val.strip()) > 0:
                    minhash_obj.update(val.encode("utf-8"))

        return col_profile, minhash_obj

    def _infer_format_and_pattern(
        self,
        dtype: pl.DataType,
        sample_head: pl.Series,
        is_fixed_length: bool,
        is_free_text: bool,
    ) -> dict[str, Any]:
        """Recognize data format, date/currency subtypes, and actionable hints."""
        result: dict[str, Any] = {
            "data_format": "STRING",
            "is_json": False,
            "date_subtype": None,
            "currency_subtype": None,
            "currency_scale_multiplier": None,
            "llm_hint": None,
        }

        if sample_head.len() == 0:
            result["data_format"] = "EMPTY"
            return result

        samples = [s.strip() for s in sample_head.to_list() if s and s.strip()]
        if not samples:
            result["data_format"] = "EMPTY"
            return result

        # 1. Native Date/Datetime dtypes
        if dtype in (pl.Date, pl.Datetime, pl.Time, pl.Duration):
            result["data_format"] = "ISO_DATETIME" if dtype == pl.Datetime else "ISO_DATE"
            result["date_subtype"] = "ISO_DATETIME" if dtype == pl.Datetime else "ISO_DATE"
            result["llm_hint"] = "Native SQLite Date/Datetime. Query directly using standard comparison operators."
            return result

        # 2. JSON Structure Detection & Validation
        is_json_candidate = all(
            (v.startswith(("{", "[")) and v.endswith(("}", "]")))
            for v in samples
            if len(v) > 1
        )
        if is_json_candidate:
            is_valid_json = True
            for candidate in samples[:5]:
                try:
                    json.loads(candidate)
                except Exception:
                    is_valid_json = False
                    break
            if is_valid_json:
                result["is_json"] = True
                result["data_format"] = "JSON_STRING"
                result["llm_hint"] = "JSON structure detected. Extract fields using json_extract(col, '$.field') in SQLite."
                return result

        # 3. Excel Custom Date Engine
        date_subtypes_found = self._check_date_patterns(samples)
        if date_subtypes_found:
            if len(date_subtypes_found) == 1:
                single_subtype = list(date_subtypes_found)[0]
                result["date_subtype"] = single_subtype
                if single_subtype in ("ISO_DATE", "ISO_DATETIME"):
                    result["data_format"] = single_subtype
                    result["llm_hint"] = "Standard ISO Date string. Perform direct string comparisons: WHERE col >= 'YYYY-MM-DD'."
                elif single_subtype == "EXCEL_SERIAL":
                    result["data_format"] = "EXCEL_SERIAL"
                    result["llm_hint"] = "Contains Excel numeric serials (e.g. 45717 -> 2025-03-01). Convert via (julianday('1899-12-30') + CAST(col AS REAL)) or pd.to_datetime(col, unit='D', origin='1899-12-30')."
                elif single_subtype == "VN_DATE_ABBR":
                    result["data_format"] = "VN_DATE_ABBR"
                    result["llm_hint"] = "Vietnamese abbreviated date ('d thg m yyyy'). Normalize 'thg ' to month number before temporal queries."
                elif single_subtype == "SLASH_DATE":
                    result["data_format"] = "SLASH_DATE"
                    result["llm_hint"] = "Vietnamese DD/MM/YYYY date. Convert to YYYY-MM-DD for standard SQLite comparisons."
                else:
                    result["data_format"] = "DATE"
                    result["llm_hint"] = f"Custom date format ({single_subtype}). Parse or normalize before temporal comparisons."
            else:
                result["data_format"] = "DATE_MIXED"
                subtypes_str = ", ".join(sorted(date_subtypes_found))
                result["date_subtype"] = subtypes_str
                result["llm_hint"] = f"Mixed date representations detected ({subtypes_str}). Normalize using Excel epoch (1899-12-30) and date parser before querying."
            return result

        # 4. Excel Custom VNĐ Currency Engine
        curr_subtype, scale, curr_hint = self._check_currency_patterns(samples)
        if curr_subtype:
            result["data_format"] = "CURRENCY"
            result["currency_subtype"] = curr_subtype
            result["currency_scale_multiplier"] = scale
            result["llm_hint"] = curr_hint
            return result

        # 5. Native Numeric or Plain Numeric String
        if dtype.is_numeric() or all(PLAIN_NUMBER_REGEX.match(v) for v in samples):
            result["data_format"] = "NUMERIC_STRING" if not dtype.is_numeric() else "NUMERIC"
            result["llm_hint"] = "Numeric column. Safe for direct mathematical aggregation."
            return result

        # 6. Fallback Text Classification
        if is_free_text:
            result["data_format"] = "FREE_TEXT"
            result["llm_hint"] = "Long unstructured text (max length > 255). Query using LIKE '%keyword%' operators."
        elif is_fixed_length and not any(" " in v for v in samples) and len(samples[0]) <= 32:
            result["data_format"] = "FIXED_CODE"
            result["llm_hint"] = "Fixed-length identifier or code. Prioritize exact equality comparisons (= 'code')."
        else:
            result["data_format"] = "STRING"

        return result

    def _check_date_patterns(self, samples: list[str]) -> set[str]:
        """Check sample values against Excel Custom Date taxonomy."""
        subtypes: set[str] = set()
        match_count = 0

        for val in samples:
            matched = True
            if ISO_DATE_REGEX.match(val):
                subtypes.add("ISO_DATE")
            elif ISO_DATETIME_REGEX.match(val):
                subtypes.add("ISO_DATETIME")
            elif SLASH_DATE_REGEX.match(val):
                subtypes.add("SLASH_DATE")
            elif SLASH_DATE_SHORT_REGEX.match(val):
                subtypes.add("SLASH_DATE_SHORT")
            elif VN_DATE_ABBR_REGEX.match(val):
                subtypes.add("VN_DATE_ABBR")
            elif EN_DATE_ABBR_REGEX.match(val):
                subtypes.add("EN_DATE_ABBR")
            elif VN_DATE_FULL_REGEX.match(val):
                subtypes.add("VN_DATE_FULL")
            elif EN_DATE_FULL_REGEX.match(val):
                subtypes.add("EN_DATE_FULL")
            elif VN_DATE_WEEKDAY_REGEX.match(val):
                subtypes.add("VN_DATE_WEEKDAY")
            elif EN_DATE_WEEKDAY_REGEX.match(val):
                subtypes.add("EN_DATE_WEEKDAY")
            elif EXCEL_SERIAL_REGEX.match(val):
                int_v = int(val)
                # Valid Excel serial range for years 2000 to 2050 (36526 to 55153)
                if 36526 <= int_v <= 55153:
                    subtypes.add("EXCEL_SERIAL")
                else:
                    matched = False
            else:
                matched = False

            if matched:
                match_count += 1

        # Require at least 30% of sampled values to match date patterns to avoid spurious noise
        if not samples or (match_count / len(samples)) < 0.3:
            return set()

        return subtypes

    def _check_currency_patterns(
        self, samples: list[str]
    ) -> tuple[str | None, float | None, str | None]:
        """Check sample values against Excel Custom VNĐ Currency taxonomy."""
        subtypes: list[str] = []

        for val in samples:
            if ACCOUNTING_PARENTHESES_REGEX.match(val):
                subtypes.append("ACCOUNTING_PARENTHESES")
            elif ACCOUNTING_DASH_ZERO_REGEX.match(val):
                subtypes.append("ACCOUNTING_DASH_ZERO")
            elif SCALED_BILLION_TY_REGEX.match(val):
                subtypes.append("SCALED_BILLION_TY")
            elif SCALED_MILLION_TR_REGEX.match(val):
                subtypes.append("SCALED_MILLION_TR")
            elif SCALED_THOUSAND_K_REGEX.match(val):
                subtypes.append("SCALED_THOUSAND_K")
            elif VND_TEXT_REGEX.match(val):
                subtypes.append("VND_TEXT")
            elif VND_SYMBOL_REGEX.match(val):
                subtypes.append("VND_SYMBOL")
            elif STANDARD_THOUSAND_SEPARATOR_REGEX.match(val):
                subtypes.append("THOUSAND_SEPARATOR")

        # Require at least 30% of sampled values to match currency patterns
        if not samples or (len(subtypes) / len(samples)) < 0.3:
            return None, None, None

        # Prioritize specialized accounting formats and scaled units over standard symbols
        subtype_set = set(subtypes)
        if "ACCOUNTING_PARENTHESES" in subtype_set:
            selected_subtype = "ACCOUNTING_PARENTHESES"
        elif "ACCOUNTING_DASH_ZERO" in subtype_set:
            selected_subtype = "ACCOUNTING_DASH_ZERO"
        elif "SCALED_BILLION_TY" in subtype_set:
            selected_subtype = "SCALED_BILLION_TY"
        elif "SCALED_MILLION_TR" in subtype_set:
            selected_subtype = "SCALED_MILLION_TR"
        elif "SCALED_THOUSAND_K" in subtype_set:
            selected_subtype = "SCALED_THOUSAND_K"
        elif "VND_TEXT" in subtype_set:
            selected_subtype = "VND_TEXT"
        elif "VND_SYMBOL" in subtype_set:
            selected_subtype = "VND_SYMBOL"
        else:
            selected_subtype = "THOUSAND_SEPARATOR"

        if selected_subtype == "ACCOUNTING_PARENTHESES":
            return (
                "ACCOUNTING_PARENTHESES",
                -1.0,
                "Accounting negative currency format '(val)'. Must convert to negative: CASE WHEN col LIKE '(%)' THEN -CAST(REPLACE(REPLACE(REPLACE(REPLACE(col, '(', ''), ')', ''), ',', ''), ' ₫', '') AS NUMERIC) ELSE CAST(REPLACE(col, ',', '') AS NUMERIC) END.",
            )
        if selected_subtype == "ACCOUNTING_DASH_ZERO":
            return (
                "ACCOUNTING_DASH_ZERO",
                0.0,
                "Accounting zero currency format '-'. Map dash entries to 0: CASE WHEN col LIKE '-%' THEN 0 ELSE CAST(col AS NUMERIC) END.",
            )
        if selected_subtype == "SCALED_BILLION_TY":
            return (
                "SCALED_BILLION_TY",
                1000000000.0,
                "Amounts scaled in Billions (tỷ). Multiply by 1,000,000,000 for actual VND amounts: CAST(REPLACE(REPLACE(col, ' tỷ', ''), ',', '.') AS REAL) * 1000000000.",
            )
        if selected_subtype == "SCALED_MILLION_TR":
            return (
                "SCALED_MILLION_TR",
                1000000.0,
                "Amounts scaled in Millions (tr VNĐ). Multiply by 1,000,000 for actual VND amounts: CAST(REPLACE(REPLACE(col, ' tr VNĐ', ''), ',', '.') AS REAL) * 1000000.",
            )
        if selected_subtype == "SCALED_THOUSAND_K":
            return (
                "SCALED_THOUSAND_K",
                1000.0,
                "Amounts scaled in Thousands (k VNĐ). Multiply by 1,000 for actual VND amounts: CAST(REPLACE(REPLACE(col, ' k VNĐ', ''), ',', '') AS NUMERIC) * 1000.",
            )
        if selected_subtype in ("VND_TEXT", "VND_SYMBOL", "THOUSAND_SEPARATOR"):
            return (
                selected_subtype,
                1.0,
                "Formatted VND currency. Clean thousand separators before aggregations: CAST(REPLACE(REPLACE(REPLACE(col, ' VNĐ', ''), ' ₫', ''), ',', '') AS INTEGER).",
            )

        return None, None, None

    def _is_eligible_for_minhash(
        self,
        dtype: pl.DataType,
        col_name: str,
        data_format: str,
        distinct_count: int,
        total_rows: int,
    ) -> bool:
        """Determine if a column should participate in MinHash FK-PK discovery."""
        if distinct_count <= 1:
            return False

        if not self.exclude_continuous_numerics:
            return True

        # Float columns are continuous amounts
        if dtype.is_float():
            return False

        # Only filter numeric or currency columns; string/text columns are always eligible
        if dtype.is_numeric() or data_format in ("NUMERIC", "NUMERIC_STRING", "CURRENCY"):
            tokens = _extract_name_tokens(col_name)

            # Exclude metric/monetary/quantity columns (e.g. so_tien, so_luong, gia_von)
            if any(t in NON_KEY_METRIC_TOKENS for t in tokens):
                return False

            # If token explicitly matches genuine key pattern (e.g. ma, id, code), retain it
            if any(t in KEY_TOKENS for t in tokens):
                return True

            # High cardinality numeric amounts without key semantics are continuous floats/totals
            if distinct_count > 50 and (distinct_count / total_rows) > 0.05:
                return False

        return True

    def _detect_undocumented_joins(
        self, minhashes: dict[str, MinHash]
    ) -> list[UndocumentedJoin]:
        """Detect FK-PK candidate joins using MinHash Jaccard resemblance."""
        undocumented_joins: list[UndocumentedJoin] = []
        keys = list(minhashes.keys())

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]

                if self.cross_table_only:
                    t1 = k1.split(".", 1)[0]
                    t2 = k2.split(".", 1)[0]
                    if t1 == t2:
                        continue

                score = float(minhashes[k1].jaccard(minhashes[k2]))
                if score >= self.jaccard_threshold:
                    undocumented_joins.append(
                        {
                            "col1": k1,
                            "col2": k2,
                            "resemblance": round(score, 4),
                        }
                    )

        # Sort candidate joins by resemblance descending
        undocumented_joins.sort(key=lambda x: x["resemblance"], reverse=True)
        return undocumented_joins

    @classmethod
    def profile_calendar(
        cls,
        calendar_path: str | Path,
        num_perm: int = 128,
        max_sample_values: int = 10000,
    ) -> DatabaseProfile:
        """Profile split-level calendar.csv directly using Polars lazy evaluation."""
        return profile_calendar_csv(
            calendar_path=calendar_path,
            num_perm=num_perm,
            max_sample_values=max_sample_values,
        )


def profile_calendar_csv(
    calendar_path: str | Path,
    num_perm: int = 128,
    max_sample_values: int = 10000,
) -> DatabaseProfile:
    """Profile split-level calendar.csv directly using Polars lazy evaluation.

    Args:
        calendar_path: Path to calendar.csv file.
        num_perm: Number of permutations for MinHash sketches.
        max_sample_values: Maximum unique sample values to retain for LSH indexing.

    Returns:
        DatabaseProfile containing the 'calendar' table profile.
    """
    path = Path(calendar_path)
    if not path.is_file():
        raise FileNotFoundError(f"Calendar file not found: {path}")

    # Use Polars lazy evaluation with scan_csv per agent skill
    q = pl.scan_csv(str(path))
    df = q.collect()
    row_count = df.height

    profiler = RawDatabaseProfiler(
        db_path="",
        num_perm=num_perm,
        max_sample_values=max_sample_values,
    )

    table_profile: TableProfile = {
        "row_count": row_count,
        "is_view": False,
        "is_unified_view": False,
        "source_tables": [],
        "columns": {},
        "note": "Split-level shared Solar-Lunar calendar lookup table.",
    }

    for col in df.columns:
        col_prof, _ = profiler._profile_single_column(
            df[col], col, row_count, "calendar"
        )
        table_profile["columns"][col] = col_prof

    return {
        "tables": {"calendar": table_profile},
        "undocumented_joins": [],
    }
