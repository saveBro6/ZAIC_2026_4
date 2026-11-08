"""Centralized constants for the ZAIC 2026-4 agent pipeline.

Contains default paths, hyperparameters, regex patterns, token sets, and prompt templates.
Maintains zero internal project dependencies to eliminate circular import risks.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Ingestion and storage defaults
DEFAULT_BASE_OUTPUT_DIR: Path = Path(os.environ.get("ZAIC_OUTPUT_DIR", "/tmp/zaic_db"))

# Profiling: MinHash and LSH indexer defaults
DEFAULT_NUM_PERM: int = 128
DEFAULT_THRESHOLD: float = 0.4
DEFAULT_SHINGLE_K: int = 3
DEFAULT_MAX_SAMPLE_VALUES: int = 10000

# Profiling: LLM summarization defaults
DEFAULT_PROD_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LOCAL_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PROD_GPU_UTIL: float = 0.95
DEFAULT_LOCAL_GPU_UTIL: float = 0.60

MINIMAL_PROFILE_PROMPT: str = """<system>
You are an expert data analyst for Vietnamese SMB (Small and Medium Business) databases.
Your objective is to generate a concise, 1-sentence SHORT SUMMARY in Vietnamese describing the exact business meaning and basic data format of the target column.
</system>

<instructions>
1. Business Meaning: Describe what business entity, attribute, or transaction property this column represents. Decode Vietnamese SMB abbreviations if present (e.g., SP -> Sản phẩm, KH -> Khách hàng, ĐVT -> Đơn vị tính, HĐ -> Hóa đơn).
2. Data Format: Specify the underlying physical and logical format (e.g., Alphanumeric code, ISO date, VND currency with thousand separators, categorical status flag, integer count).
3. Brevity Constraint: Output EXACTLY ONE concise Vietnamese sentence. Do not add boilerplate, preambles, or conversational prefixes.
</instructions>

<business_readme>
{business_readme}
</business_readme>

<few_shot_examples>
  <example id="1">
    <target_column>
      Table: "DonHang"
      Column: "ma_kh"
      Sibling Columns: ["id", "ngay_tao", "tong_tien", "trang_thai"]
      Nulls: 0/15000 (0.00%)
      Distinct Count: 3200
      Format & Hints: STRING | Code identifier
      Sample Values: ["KH0001", "KH0002", "KH0158", "VANG_LAI"]
    </target_column>
    <output>
      Mã định danh khách hàng (KH: Khách Hàng, VANG_LAI: khách mua vãng lai không lưu hồ sơ); định dạng chuỗi mã ký tự chữ và số.
    </output>
  </example>

  <example id="2">
    <target_column>
      Table: "ChiTietDonHang"
      Column: "thanh_tien"
      Sibling Columns: ["ma_sp", "so_luong", "don_gia", "chiet_khau"]
      Nulls: 0/45000 (0.00%)
      Distinct Count: 1250
      Format & Hints: CURRENCY | Formatted VND currency. Clean thousand separators before aggregations.
      Sample Values: ["12,000", "45,000", "150,000", "0"]
    </target_column>
    <output>
      Tổng số tiền thực thu của dòng sản phẩm sau khi trừ chiết khấu dòng; định dạng số tiền VNĐ có dấu phân cách hàng nghìn cần làm sạch trước khi tính toán.
    </output>
  </example>
</few_shot_examples>

<target_column>
  Table: "{table_name}"
  Column: "{col_name}"
  Sibling Columns: {sibling_cols}
  Nulls: {null_count}/{total_rows} ({null_ratio:.2%})
  Distinct Count: {distinct_count}
  Format & Hints: {data_format} | {llm_hint}
  Sample Values: {samples}
</target_column>

<output>
"""

MAXIMAL_PROFILE_PROMPT: str = """<system>
You are an expert data architect and SQL engineer for Vietnamese SMB databases.
Your objective is to generate an in-depth, structured LONG SUMMARY in Vietnamese for the target column to guide downstream Text-to-SQL generation and semantic vector retrieval.
</system>

<instructions>
Include the following 4 structural aspects in Vietnamese:
1. Business Meaning: Operational role in the business workflow, entity relationship, and relation to sibling columns.
2. Data Cleanliness & Distribution: Completeness (null count and ratio), cardinality (distinct count), value boundaries ([min_val to max_val]).
3. Dominant Patterns: Frequent categories from Top-10 samples and percentage dominance.
4. Technical SQL / Query Guidance: Specific data cleaning, casting instructions (e.g. handling formatted VND strings, parsing DD/MM/YYYY dates to ISO YYYY-MM-DD), and business status filtering.
</instructions>

<business_readme>
{business_readme}
</business_readme>

<few_shot_examples>
  <example id="1">
    <target_column>
      Table: "SoBanHang"
      Column: "Trạng thái xử lý"
      Nulls: 526/43605 (1.21%)
      Distinct Count: 4
      Value Range: ["Không giao được" to "đang gửi"]
      Format & Pattern: STRING (Categorical status)
      Top 10 Samples: {{"Đã hoàn thành": 41248, "Không giao được": 1163, "trả hàng": 660, "đang gửi": 8}}
      Code / SQL Hint: Categorical status string. Exclude cancelled/returned orders for revenue calculation per business rules.
    </target_column>
    <output>
      Cột lưu trữ trạng thái xử lý đơn hàng trong sổ bán hàng, đóng vai trò cờ phân loại giao dịch (status flag). Dữ liệu có 4 giá trị phân loại với 1.21% giá trị NULL; giá trị chiếm đa số tuyệt đối là 'Đã hoàn thành' (94.6%), tiếp theo là 'Không giao được' và 'trả hàng'. Khi truy vấn tính toán doanh thu hoặc sản phẩm bán chạy theo quy ước nghiệp vụ, BẮT BUỘC phải lọc loại bỏ các trạng thái huỷ/hoàn ('trả hàng', 'Không giao được') bằng điều kiện: WHERE "Trạng thái xử lý" = 'Đã hoàn thành'.
    </output>
  </example>

  <example id="2">
    <target_column>
      Table: "HoaDon"
      Column: "ngay_lap"
      Nulls: 0/50000 (0.00%)
      Distinct Count: 730
      Value Range: ["01/01/2024" to "31/12/2025"]
      Format & Pattern: DATE (SLASH_DATE_DD_MM_YYYY)
      Top 10 Samples: {{"15/01/2025": 140, "20/01/2025": 135, "05/02/2025": 128}}
      Code / SQL Hint: Date in DD/MM/YYYY format. Convert to ISO YYYY-MM-DD for SQL temporal filtering.
    </target_column>
    <output>
      Cột ghi nhận mốc thời gian lập hóa đơn bán lẻ của doanh nghiệp. Cột không có giá trị NULL (100% đầy đủ) với 730 ngày giao dịch liên tục từ 01/01/2024 đến 31/12/2025. Định dạng ngày tháng lưu trữ là chuỗi DD/MM/YYYY (Việt Nam), do đó khi thực hiện truy vấn so sánh thời gian, lọc theo mốc as_of_date hoặc liên kết với calendar.csv, cần chuẩn hóa sang chuẩn ISO YYYY-MM-DD hoặc sử dụng hàm phân tích ngày: SUBSTR(ngay_lap, 7, 4) || '-' || SUBSTR(ngay_lap, 4, 2) || '-' || SUBSTR(ngay_lap, 1, 2).
    </output>
  </example>
</few_shot_examples>

<target_column>
  Table: "{table_name}"
  Column: "{col_name}"
  Nulls: {null_count}/{total_rows} ({null_ratio:.2%})
  Distinct Count: {distinct_count}
  Value Range: [{min_val} to {max_val}]
  Format & Pattern: {data_format} ({subtype_info})
  Top 10 Samples: {top_10}
  Code / SQL Hint: {llm_hint}
</target_column>

<output>
"""

# Profiling: Raw profiler date regex patterns
ISO_DATE_REGEX: re.Pattern[str] = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
ISO_DATETIME_REGEX: re.Pattern[str] = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}[ T]\d{1,2}:\d{2}(?::\d{2})?$")
SLASH_DATE_REGEX: re.Pattern[str] = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
SLASH_DATE_SHORT_REGEX: re.Pattern[str] = re.compile(r"^\d{1,2}/\d{1,2}/\d{2}$")
VN_DATE_ABBR_REGEX: re.Pattern[str] = re.compile(r"^\d{1,2}\s+thg\s+\d{1,2}(?:\s+\d{4})?$", re.IGNORECASE)
EN_DATE_ABBR_REGEX: re.Pattern[str] = re.compile(
    r"^\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2,4}$",
    re.IGNORECASE,
)
VN_DATE_FULL_REGEX: re.Pattern[str] = re.compile(
    r"^(?:ngày\s+)?\d{1,2}\s+tháng\s+\d{1,2}(?:\s+năm\s+\d{4})?$",
    re.IGNORECASE,
)
EN_DATE_FULL_REGEX: re.Pattern[str] = re.compile(
    r"^(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*|\s+)\d{4}$",
    re.IGNORECASE,
)
VN_DATE_WEEKDAY_REGEX: re.Pattern[str] = re.compile(
    r"^(?:thứ\s+(?:hai|ba|tư|bốn|năm|sáu|bảy|[2-7])|chủ\s+nhật|cn)[,\s]+\d{1,2}/\d{1,2}/\d{4}$",
    re.IGNORECASE,
)
EN_DATE_WEEKDAY_REGEX: re.Pattern[str] = re.compile(
    r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)[,\s]+\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$",
    re.IGNORECASE,
)
EXCEL_SERIAL_REGEX: re.Pattern[str] = re.compile(r"^\d{5}$")

# Profiling: Raw profiler currency and numeric regex patterns
VND_TEXT_REGEX: re.Pattern[str] = re.compile(
    r"^-?\d{1,3}(?:[,\.]\d{3})*(?:\.\d+)?\s*(?:vnđ|vnd)$",
    re.IGNORECASE,
)
VND_SYMBOL_REGEX: re.Pattern[str] = re.compile(
    r"^-?\d{1,3}(?:[,\.]\d{3})*(?:\.\d+)?\s*(?:₫|đ)$",
    re.IGNORECASE,
)
ACCOUNTING_PARENTHESES_REGEX: re.Pattern[str] = re.compile(
    r"^\(\s*\d{1,3}(?:[,\.]\d{3})*(?:\.\d+)?\s*(?:₫|đ|vnđ|vnd)?\s*\)$",
    re.IGNORECASE,
)
ACCOUNTING_DASH_ZERO_REGEX: re.Pattern[str] = re.compile(
    r"^-\s*(?:₫|đ|vnđ|vnd)?$",
    re.IGNORECASE,
)
SCALED_THOUSAND_K_REGEX: re.Pattern[str] = re.compile(
    r"^-?\d{1,3}(?:[,\.]\d{3})*(?:\.\d+)?\s*(?:k\s*vnđ|k\s*vnd|k)$",
    re.IGNORECASE,
)
SCALED_MILLION_TR_REGEX: re.Pattern[str] = re.compile(
    r"^-?\d+(?:[,\.]\d+)?\s*(?:tr\s*vnđ|tr\s*vnd|tr|triệu)$",
    re.IGNORECASE,
)
SCALED_BILLION_TY_REGEX: re.Pattern[str] = re.compile(
    r"^-?\d+(?:[,\.]\d+)?\s*(?:tỷ|ty)$",
    re.IGNORECASE,
)
STANDARD_THOUSAND_SEPARATOR_REGEX: re.Pattern[str] = re.compile(r"^-?\d{1,3}(?:[,\.]\d{3})+(?:\.\d+)?$")
PLAIN_NUMBER_REGEX: re.Pattern[str] = re.compile(r"^-?\d+(?:\.\d+)?$")

# Profiling: Raw profiler SQL regex and token sets
VIEW_FROM_REGEX: re.Pattern[str] = re.compile(
    r'FROM\s+(?:\"([^\"]+)\"|\`([^\`]+)\`|\[([^\]]+)\]|([^\s,;]+))',
    re.IGNORECASE,
)
KEY_TOKENS: frozenset[str] = frozenset({"id", "code", "stt", "key", "sku", "ref", "ma", "mã"})
NON_KEY_METRIC_TOKENS: frozenset[str] = frozenset({
    "tien", "tiền", "gia", "giá", "luong", "lượng", "phi", "phí",
    "vat", "thue", "thuế", "du", "dư", "discount", "amount", "price",
    "revenue", "cost", "total", "qty", "quantity", "count",
})

# Profiling: Semantic vector indexer defaults
DEFAULT_EMBED_MODEL_PATH: str = "/app/models/bge-m3"
DEFAULT_MAX_LENGTH: int = 4096
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_TOP_K: int = 10
DEFAULT_EMBEDDING_DIM: int = 1024
DEFAULT_MAX_CHUNK_CHARS: int = 1000
DEFAULT_MIN_CHUNK_CHARS: int = 150
DEFAULT_EMBED_GPU_UTIL: float = 0.20
MIN_TABLE_ROWS_FOR_SPLIT: int = 3

# Schema linking: Context builder few-shot prompt examples
FEW_SHOT_SQL_EXAMPLES: str = (
    "<few_shot_examples>\n"
    'SELECT SUM(T1."Thành tiền") AS "Tổng doanh thu"\n'
    'FROM "ban_hang" AS T1\n'
    "WHERE T1.\"Tên hàng\" LIKE '%Sữa tươi%'\n"
    "  AND T1.\"Ngày tháng\" >= '2026-01-01'\n"
    "  AND T1.\"Ngày tháng\" <= '2026-01-31';\n"
    "\n"
    'SELECT T2."Tên khách", SUM(T1."Số lượng") AS "Tổng số lượng"\n'
    'FROM "hoa_don_chi_tiet" AS T1\n'
    'JOIN "khach_hang" AS T2 ON T1."Mã khách" = T2."Mã khách"\n'
    'GROUP BY T2."Mã khách", T2."Tên khách"\n'
    'ORDER BY "Tổng số lượng" DESC\n'
    "LIMIT 5;\n"
    "</few_shot_examples>"
)

__all__ = [
    "ACCOUNTING_DASH_ZERO_REGEX",
    "ACCOUNTING_PARENTHESES_REGEX",
    # Ingestion
    "DEFAULT_BASE_OUTPUT_DIR",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_EMBED_GPU_UTIL",
    "DEFAULT_EMBED_MODEL_PATH",
    "DEFAULT_LOCAL_GPU_UTIL",
    "DEFAULT_LOCAL_MODEL",
    "DEFAULT_MAX_CHUNK_CHARS",
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_MAX_SAMPLE_VALUES",
    "DEFAULT_MIN_CHUNK_CHARS",
    # LSH indexer
    "DEFAULT_NUM_PERM",
    "DEFAULT_PROD_GPU_UTIL",
    # Profile summarizer
    "DEFAULT_PROD_MODEL",
    "DEFAULT_SHINGLE_K",
    "DEFAULT_THRESHOLD",
    "DEFAULT_TOP_K",
    "EN_DATE_ABBR_REGEX",
    "EN_DATE_FULL_REGEX",
    "EN_DATE_WEEKDAY_REGEX",
    "EXCEL_SERIAL_REGEX",
    # Schema linking
    "FEW_SHOT_SQL_EXAMPLES",
    "ISO_DATETIME_REGEX",
    # Raw profiler
    "ISO_DATE_REGEX",
    "KEY_TOKENS",
    "MAXIMAL_PROFILE_PROMPT",
    "MINIMAL_PROFILE_PROMPT",
    "MIN_TABLE_ROWS_FOR_SPLIT",
    "NON_KEY_METRIC_TOKENS",
    "PLAIN_NUMBER_REGEX",
    "SCALED_BILLION_TY_REGEX",
    "SCALED_MILLION_TR_REGEX",
    "SCALED_THOUSAND_K_REGEX",
    "SLASH_DATE_REGEX",
    "SLASH_DATE_SHORT_REGEX",
    "STANDARD_THOUSAND_SEPARATOR_REGEX",
    "VIEW_FROM_REGEX",
    "VND_SYMBOL_REGEX",
    "VND_TEXT_REGEX",
    "VN_DATE_ABBR_REGEX",
    "VN_DATE_FULL_REGEX",
    "VN_DATE_WEEKDAY_REGEX",
]
