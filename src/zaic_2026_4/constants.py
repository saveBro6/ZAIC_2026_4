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
DEFAULT_PROD_GPU_UTIL: float = 0.70
DEFAULT_LOCAL_GPU_UTIL: float = 0.60
DEFAULT_LLM_MAX_LENGTH: int = 16384

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

MAXIMAL_PROFILE_PROMPT: str = """<system>
Bạn là chuyên gia phân tích và kiến trúc dữ liệu cho cơ sở dữ liệu doanh nghiệp vừa và nhỏ (SMB) tại Việt Nam.
Mục tiêu của bạn là tạo ra một đoạn TÓM TẮT ĐẦY ĐỦ bằng tiếng Việt cho cột mục tiêu, hợp nhất ý nghĩa nghiệp vụ từ Tóm tắt rút gọn (Minimal Profile) với toàn bộ thông số phân bổ dữ liệu thực tế.
</system>

<instructions>
Bao gồm các nội dung có cấu trúc sau bằng tiếng Việt:
1. Ý nghĩa nghiệp vụ & Mở đầu: BẮT BUỘC dùng câu Tóm tắt rút gọn (Minimal Profile) làm câu mở đầu nguyên vẹn (bắt đầu bằng Cột "{col_name}"...), sau đó mở rộng về vai trò trong quy trình vận hành và mối liên hệ với các cột khác trong bảng.
2. Phân bổ dữ liệu thực tế:
   - Giá trị NULL: Nếu số lượng NULL là 0, khẳng định rõ: "Cột không có giá trị NULL". Nếu số lượng NULL > 0, ghi rõ số lượng (ví dụ: "Cột chứa {null_count} giá trị NULL"). TUYỆT ĐỐI KHÔNG dùng cụm từ "không có giá trị NULL" khi có giá trị NULL.
   - Độ biến thiên & Giới hạn: Nêu số lượng giá trị duy nhất (distinct_count) và khoảng giá trị thực tế từ min_val đến max_val.
</instructions>

<business_readme>
{business_readme}
</business_readme>

<few_shot_examples>
  <example id="1">
    <target_column>
      Bảng: "NhatKyBaoDuong"
      Cột: "tien_do_sua"
      Tóm tắt rút gọn: "Cột \\"tien_do_sua\\" theo dõi tiến độ xử lý bảo dưỡng xe (Chờ phụ tùng, Đang sửa, Đã bàn giao, Khách hủy); định dạng chuỗi phân loại trạng thái."
      Các cột khác trong bảng: ["ma_xe", "thoi_diem_vao", "phi_dich_vu", "so_gio_cong"]
      Giá trị NULL: 120/15000
      Số giá trị duy nhất: 4
      Khoảng giá trị: ["Chờ phụ tùng" đến "Đang sửa"]
      Định dạng: STRING (Categorical status)
    </target_column>
    <output>
      Cột "tien_do_sua" theo dõi tiến độ xử lý bảo dưỡng xe (Chờ phụ tùng, Đang sửa, Đã bàn giao, Khách hủy); định dạng chuỗi phân loại trạng thái. Cột giữ vai trò quản lý tiến trình thực hiện dịch vụ kỹ thuật trong xưởng sửa chữa, liên kết với thời điểm nhận xe và phí dịch vụ. Cột chứa 120 giá trị NULL, với 4 giá trị phân loại khác nhau từ 'Chờ phụ tùng' đến 'Đang sửa'.
    </output>
  </example>

  <example id="2">
    <target_column>
      Bảng: "LichHenKham"
      Cột: "ngay_kham_hen"
      Tóm tắt rút gọn: "Cột \\"ngay_kham_hen\\" ghi nhận mốc thời gian hẹn khám chuyên khoa; định dạng chuỗi ngày tháng DD/MM/YYYY."
      Các cột khác trong bảng: ["ma_ho_so", "bac_si_phu_trach", "tong_chi_phi"]
      Giá trị NULL: 0/8500
      Số giá trị duy nhất: 365
      Khoảng giá trị: ["01/01/2025" đến "31/12/2025"]
      Định dạng: DATE (SLASH_DATE_DD_MM_YYYY)
    </target_column>
    <output>
      Cột "ngay_kham_hen" ghi nhận mốc thời gian hẹn khám chuyên khoa; định dạng chuỗi ngày tháng DD/MM/YYYY. Cột giữ vai trò quản lý lịch hẹn khám của khách hàng tại phòng khám. Cột không có giá trị NULL với 365 mốc ngày khác nhau từ 01/01/2025 đến 31/12/2025.
    </output>
  </example>
</few_shot_examples>

<target_column>
  Bảng: "{table_name}"
  Cột: "{col_name}"
  Tóm tắt rút gọn: "{minimal_profile}"
  Các cột khác trong bảng: {sibling_cols}
  Giá trị NULL: {null_count}/{total_rows}
  Số giá trị duy nhất: {distinct_count}
  Khoảng giá trị: [{min_val} đến {max_val}]
  Định dạng: {data_format} ({subtype_info})
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

# Submission and SQL generation defaults
DEFAULT_CANDIDATE_SEEDS: tuple[int, ...] = (42, 100, 2026)
DEFAULT_NUM_CANDIDATES: int = 3
DEFAULT_SQL_MAX_RETRIES: int = 3
DEFAULT_SQL_TIMEOUT_SECONDS: float = 10.0
DEFAULT_SUBMISSION_PATH: Path = Path(os.environ.get("ZAIC_SUBMISSION_PATH", "/result/submission.jsonl"))
DEFAULT_LOGS_DIR: Path = Path(os.environ.get("ZAIC_LOGS_DIR", "/result/logs"))

__all__ = [
    "ACCOUNTING_DASH_ZERO_REGEX",
    "ACCOUNTING_PARENTHESES_REGEX",
    # Ingestion
    "DEFAULT_BASE_OUTPUT_DIR",
    "DEFAULT_BATCH_SIZE",
    # Submission
    "DEFAULT_CANDIDATE_SEEDS",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_EMBED_GPU_UTIL",
    "DEFAULT_EMBED_MAX_LENGTH",
    "DEFAULT_EMBED_MODEL_PATH",
    "DEFAULT_LLM_MAX_LENGTH",
    "DEFAULT_LOCAL_GPU_UTIL",
    "DEFAULT_LOCAL_MODEL",
    "DEFAULT_LOGS_DIR",
    "DEFAULT_MAX_CHUNK_CHARS",
    "DEFAULT_MAX_SAMPLE_VALUES",
    "DEFAULT_MIN_CHUNK_CHARS",
    "DEFAULT_NUM_CANDIDATES",
    # LSH indexer
    "DEFAULT_NUM_PERM",
    "DEFAULT_PROD_GPU_UTIL",
    # Profile summarizer
    "DEFAULT_PROD_MODEL",
    "DEFAULT_SHINGLE_K",
    "DEFAULT_SQL_MAX_RETRIES",
    "DEFAULT_SQL_TIMEOUT_SECONDS",
    "DEFAULT_SUBMISSION_PATH",
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
