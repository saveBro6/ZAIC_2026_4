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
DEFAULT_NUM_PERM: int = 64
DEFAULT_THRESHOLD: float = 0.4
DEFAULT_SHINGLE_K: int = 3
DEFAULT_MAX_SAMPLE_VALUES: int = 100

# Profiling: LSH index guardrails (keeps the pickled index small and fast to load)
DEFAULT_LSH_MAX_DISTINCT_VALUES: int = 1000
DEFAULT_LSH_MAX_VALUE_CHARS: int = 120
DEFAULT_LSH_CARDINALITY_SKIP_RATIO: float = 0.9
DEFAULT_LSH_INDEX_VERSION: int = 2

# Profiling: LLM summarization defaults
DEFAULT_PROD_MODEL: str = "Qwen/Qwen3.5-9B"
DEFAULT_LOCAL_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PROD_GPU_UTIL: float = 0.70
DEFAULT_LOCAL_GPU_UTIL: float = 0.60
DEFAULT_LLM_MAX_LENGTH: int = 24576
DEFAULT_QUANTIZATION: str | None = None

# Profiling: minimal-profile generation guardrails
DEFAULT_PROFILE_BATCH_SIZE: int = 64
DEFAULT_PROFILE_MAX_TOKENS: int = 256
DEFAULT_PROFILE_RETRY_ATTEMPTS: int = 1

MINIMAL_PROFILE_PROMPT: str = """<system>
Bạn là chuyên gia phân tích dữ liệu cho các cơ sở dữ liệu doanh nghiệp vừa và nhỏ (SMB) tại Việt Nam.
Mục tiêu của bạn là tạo ra một câu TÓM TẮT RÚT GỌN (đúng 1 câu duy nhất) bằng tiếng Việt giải nghĩa chính xác vai trò nghiệp vụ của cột, bóc tách các từ viết tắt và nêu định dạng dữ liệu cơ bản.
</system>

<instructions>
1. Quy cách mở đầu: Câu tóm tắt BẮT BUỘC bắt đầu bằng tên cột theo mẫu: Cột "{col_name}" là... hoặc Cột "{col_name}" chứa... hoặc Cột "{col_name}" lưu trữ... hoặc Cột "{col_name}" theo dõi...
2. Ý nghĩa nghiệp vụ: Mô tả thực thể, thuộc tính hoặc đặc tính giao dịch mà cột đại diện. Bóc tách và giải nghĩa các từ viết tắt tiếng Việt SMB nếu có (ví dụ: SP -> Sản phẩm, KH -> Khách hàng, ĐVT -> Đơn vị tính, HĐ -> Hóa đơn) bằng cách đối chiếu với danh sách các cột khác trong bảng và tài liệu kinh doanh.
3. Định dạng dữ liệu: Nêu rõ định dạng lưu trữ cơ bản (ví dụ: Chuỗi mã chữ và số, ngày tháng, số tiền VNĐ có dấu phân cách, cờ trạng thái phân loại, số nguyên/thực).
4. Ràng buộc độ dài: Đầu ra BẮT BUỘC ĐÚNG 1 CÂU súc tích. Không thêm lời mở đầu hay giải thích thừa ngoài câu tóm tắt.
</instructions>

<business_readme>
{business_readme}
</business_readme>

<few_shot_examples>
  <example id="1">
    <target_column>
      Bảng: "PhieuTiepNhan"
      Cột: "ma_bien_so"
      Các cột khác trong bảng: ["ma_phieu", "thoi_diem_vao", "phi_uoc_tinh", "tien_do_sua"]
      Định dạng: STRING
      Các giá trị mẫu: ["59P1-12345", "60B2-98765", "43C1-45678", "VANG_LAI"]
    </target_column>
    <output>
      Cột "ma_bien_so" lưu trữ biển số xe của khách mang vào sửa chữa (VANG_LAI: khách vãng lai không lưu hồ sơ); định dạng chuỗi mã ký tự chữ và số.
    </output>
  </example>

  <example id="2">
    <target_column>
      Bảng: "ChiTietTienCong"
      Cột: "phi_dich_vu"
      Các cột khác trong bảng: ["ma_cong", "ten_hang_muc", "so_gio_cong", "giam_tru"]
      Định dạng: CURRENCY
      Các giá trị mẫu: ["150,000", "350,000", "80,000", "0"]
    </target_column>
    <output>
      Cột "phi_dich_vu" chứa số tiền phí dịch vụ kỹ thuật sửa chữa sau khi trừ chiết khấu giảm trừ; định dạng số tiền VNĐ có dấu phân cách hàng nghìn.
    </output>
  </example>
</few_shot_examples>

<target_column>
  Bảng: "{table_name}"
  Cột: "{col_name}"
  Các cột khác trong bảng: {sibling_cols}
  Định dạng: {data_format}
  Các giá trị mẫu: {samples}
</target_column>

<output>
"""

MAXIMAL_ENUM_MAX_DISTINCT: int = 10
MAXIMAL_TOP_VALUES: int = 10
MAXIMAL_DOMINANT_RATIO: float = 0.9
MAXIMAL_MAX_JOINS_PER_COLUMN: int = 5
MAXIMAL_MAX_VALUE_CHARS: int = 40
RANGE_ELIGIBLE_FORMATS: frozenset[str] = frozenset({
    "NUMERIC",
    "NUMERIC_STRING",
    "CURRENCY",
    "ISO_DATE",
    "ISO_DATETIME",
    "SLASH_DATE",
    "VN_DATE_ABBR",
    "EXCEL_SERIAL",
    "FIXED_CODE",
})
DATE_FORMATS: frozenset[str] = frozenset({
    "DATE",
    "DATE_MIXED",
    "ISO_DATE",
    "ISO_DATETIME",
    "SLASH_DATE",
    "SLASH_DATE_SHORT",
    "VN_DATE_ABBR",
    "EN_DATE_ABBR",
    "VN_DATE_FULL",
    "EN_DATE_FULL",
    "VN_DATE_WEEKDAY",
    "EN_DATE_WEEKDAY",
    "EXCEL_SERIAL",
})

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
EXCEL_SERIAL_REGEX: re.Pattern[str] = re.compile(r"^\d{5}(?:\.\d+)?$")

# Profiling: Raw profiler currency and numeric regex patterns
VND_TEXT_REGEX: re.Pattern[str] = re.compile(
    r"^-?\d{1,3}(?:[,\.]\d{3})*(?:\.\d+)?\s*(?:vnđ|vnd)$",
    re.IGNORECASE,
)
VND_SYMBOL_REGEX: re.Pattern[str] = re.compile(
    r"^-?\d{1,3}(?:[,\.]\d{3})*(?:\.\d+)?\s*đ$",
    re.IGNORECASE,
)
ACCOUNTING_PARENTHESES_REGEX: re.Pattern[str] = re.compile(
    r"^\(\s*\d{1,3}(?:[,\.]\d{3})*(?:\.\d+)?\s*(?:đ|vnđ|vnd)?\s*\)$",
    re.IGNORECASE,
)
ACCOUNTING_DASH_ZERO_REGEX: re.Pattern[str] = re.compile(
    r"^-\s*(?:đ|vnđ|vnd)?$",
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
DEFAULT_EMBED_MAX_LENGTH: int = 4096
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_TOP_K: int = 20
DEFAULT_EMBEDDING_DIM: int = 1024
DEFAULT_MAX_CHUNK_CHARS: int = 1000
DEFAULT_MIN_CHUNK_CHARS: int = 150
DEFAULT_EMBED_GPU_UTIL: float = 0.20
MIN_TABLE_ROWS_FOR_SPLIT: int = 3

# Schema linking: Context builder few-shot prompt examples.
# The identifiers below are deliberately neutral placeholders: real table and column
# names must always come from the retrieved <database_schema>, never from these examples.
FEW_SHOT_SQL_EXAMPLES: str = (
    "<few_shot_examples>\n"
    "<!-- Vi du minh hoa DINH DANG cu phap. Ten bang/cot duoi day la gia dinh, "
    "TUYET DOI khong dung lai; chi dung bang/cot co trong <database_schema>. -->\n"
    "<example>\n"
    'SELECT SUM(CAST(REPLACE(REPLACE(T1."CotTien", \'.\', \'\'), \',\', \'\') AS INTEGER))\n'
    'FROM "BangGiaoDich" AS T1\n'
    "WHERE T1.\"CotTrangThai\" NOT IN ('GiaTriLoai1', 'GiaTriLoai2')\n"
    "  AND (SUBSTR(T1.\"CotNgay\", 7, 4) || '-' || SUBSTR(T1.\"CotNgay\", 4, 2) "
    "|| '-' || SUBSTR(T1.\"CotNgay\", 1, 2)) BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD';\n"
    "</example>\n"
    "<example>\n"
    'SELECT T2."CotTen", SUM(T1."CotSoLuong") AS "TongSoLuong"\n'
    'FROM "BangChiTiet" AS T1\n'
    'JOIN "BangDanhMuc" AS T2 ON T1."CotMa" = T2."CotMa"\n'
    'WHERE T2."CotTen" IS NOT NULL\n'
    'GROUP BY T2."CotMa", T2."CotTen"\n'
    'ORDER BY "TongSoLuong" DESC, T2."CotMa" ASC\n'
    "LIMIT 5;\n"
    "</example>\n"
    "</few_shot_examples>"
)

# Submission and SQL generation defaults
DEFAULT_CANDIDATE_SEEDS: tuple[int, ...] = (42, 100, 2026)
DEFAULT_NUM_CANDIDATES: int = 3
DEFAULT_SQL_MAX_RETRIES: int = 2
DEFAULT_EXECUTION_MAX_RETRIES: int = 2
DEFAULT_SQL_TIMEOUT_SECONDS: float = 10.0
DEFAULT_SQL_MAX_TOKENS: int = 1536
DEFAULT_THINKING_TOKEN_BUDGET: int = 384
DEFAULT_SQL_FINALIZE_MAX_TOKENS: int = 768
DEFAULT_SQL_FINALIZE_THINKING_BUDGET: int = 128
DEFAULT_RULEBOOK_MODE: str = "retrieval"
DEFAULT_RULEBOOK_TOP_K: int = 3
DEFAULT_BUSINESS_CACHE_MAX_ENTRIES: int = 2
DEFAULT_GROUNDED_LITERAL_LIMIT: int = 8
DEFAULT_GROUNDED_LITERAL_MIN_SCORE: float = 0.55
DEFAULT_SCHEMA_LINK_MAX_TOKENS: int = 320
DEFAULT_REASONING_START_STR: str = '{"reasoning": "<think>'
DEFAULT_REASONING_END_STR: str = '</think>", "sql": "'

_DEFAULT_RESULT_BASE: Path = (
    Path("/result")
    if Path("/result").is_dir()
    else Path(os.environ.get("ZAIC_RESULT_DIR", "result")).resolve()
)
DEFAULT_SUBMISSION_PATH: Path = Path(
    os.environ.get("ZAIC_SUBMISSION_PATH", str(_DEFAULT_RESULT_BASE / "submission.jsonl"))
)
DEFAULT_LOGS_DIR: Path = Path(
    os.environ.get("ZAIC_LOGS_DIR", str(_DEFAULT_RESULT_BASE / "logs"))
)

__all__ = [
    "ACCOUNTING_DASH_ZERO_REGEX",
    "ACCOUNTING_PARENTHESES_REGEX",
    "DATE_FORMATS",
    "DEFAULT_BASE_OUTPUT_DIR",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_BUSINESS_CACHE_MAX_ENTRIES",
    "DEFAULT_CANDIDATE_SEEDS",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_EMBED_GPU_UTIL",
    "DEFAULT_EMBED_MAX_LENGTH",
    "DEFAULT_EMBED_MODEL_PATH",
    "DEFAULT_EXECUTION_MAX_RETRIES",
    "DEFAULT_GROUNDED_LITERAL_LIMIT",
    "DEFAULT_GROUNDED_LITERAL_MIN_SCORE",
    "DEFAULT_LLM_MAX_LENGTH",
    "DEFAULT_LOCAL_GPU_UTIL",
    "DEFAULT_LOCAL_MODEL",
    "DEFAULT_LOGS_DIR",
    "DEFAULT_LSH_CARDINALITY_SKIP_RATIO",
    "DEFAULT_LSH_INDEX_VERSION",
    "DEFAULT_LSH_MAX_DISTINCT_VALUES",
    "DEFAULT_LSH_MAX_VALUE_CHARS",
    "DEFAULT_MAX_CHUNK_CHARS",
    "DEFAULT_MAX_SAMPLE_VALUES",
    "DEFAULT_MIN_CHUNK_CHARS",
    "DEFAULT_NUM_CANDIDATES",
    "DEFAULT_NUM_PERM",
    "DEFAULT_PROD_GPU_UTIL",
    "DEFAULT_PROD_MODEL",
    "DEFAULT_PROFILE_BATCH_SIZE",
    "DEFAULT_PROFILE_MAX_TOKENS",
    "DEFAULT_PROFILE_RETRY_ATTEMPTS",
    "DEFAULT_QUANTIZATION",
    "DEFAULT_REASONING_END_STR",
    "DEFAULT_REASONING_START_STR",
    "DEFAULT_RULEBOOK_MODE",
    "DEFAULT_RULEBOOK_TOP_K",
    "DEFAULT_SCHEMA_LINK_MAX_TOKENS",
    "DEFAULT_SHINGLE_K",
    "DEFAULT_SQL_FINALIZE_MAX_TOKENS",
    "DEFAULT_SQL_FINALIZE_THINKING_BUDGET",
    "DEFAULT_SQL_MAX_RETRIES",
    "DEFAULT_SQL_MAX_TOKENS",
    "DEFAULT_SQL_TIMEOUT_SECONDS",
    "DEFAULT_SUBMISSION_PATH",
    "DEFAULT_THINKING_TOKEN_BUDGET",
    "DEFAULT_THRESHOLD",
    "DEFAULT_TOP_K",
    "EN_DATE_ABBR_REGEX",
    "EN_DATE_FULL_REGEX",
    "EN_DATE_WEEKDAY_REGEX",
    "EXCEL_SERIAL_REGEX",
    "FEW_SHOT_SQL_EXAMPLES",
    "ISO_DATETIME_REGEX",
    "ISO_DATE_REGEX",
    "KEY_TOKENS",
    "MAXIMAL_DOMINANT_RATIO",
    "MAXIMAL_ENUM_MAX_DISTINCT",
    "MAXIMAL_MAX_JOINS_PER_COLUMN",
    "MAXIMAL_MAX_VALUE_CHARS",
    "MAXIMAL_TOP_VALUES",
    "MINIMAL_PROFILE_PROMPT",
    "MIN_TABLE_ROWS_FOR_SPLIT",
    "NON_KEY_METRIC_TOKENS",
    "PLAIN_NUMBER_REGEX",
    "RANGE_ELIGIBLE_FORMATS",
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
