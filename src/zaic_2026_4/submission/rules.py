"""Validation issue models, severity levels, rule registry, and diagnostic rendering."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Severity = Literal["blocking", "warning"]

RULE_REGISTRY: dict[str, str] = {
    "OUTPUT_EMPTY": (
        "Đầu ra trống hoặc không chứa JSON hợp lệ. "
        'Bắt buộc xuất đối tượng JSON {"reasoning": "...", "sql": "...", "type": "..."}.'
    ),
    "OUTPUT_UNPARSEABLE": (
        "Không thể phân tích cú pháp JSON từ đầu ra. Kiểm tra đóng mở ngoặc nhọn và escape chuỗi."
    ),
    "TYPE_INVALID": (
        "Trường 'type' không hợp lệ theo quy chế DataMate. "
        "Chỉ dùng 1 trong: number, percent, string, date, boolean, list, table, unanswerable, abstain."
    ),
    "SQL_SYNTAX": (
        "Lỗi cú pháp SQLite. Kiểm tra dấu ngoặc, dấu nháy đơn chuỗi, dấu nháy kép tên cột và từ khóa."
    ),
    "SQL_NOT_READ_ONLY": (
        "Chỉ cho phép truy vấn đọc dữ liệu (SELECT / WITH). Cấm tuyệt đối INSERT, UPDATE, DELETE, DROP, ALTER."
    ),
    "SQL_NOT_SELECT": (
        "Câu lệnh không phải là truy vấn SELECT hoặc UNION hợp lệ."
    ),
    "SQL_MULTIPLE_STATEMENTS": (
        "Chỉ cho phép một câu lệnh SQL duy nhất, không dùng dấu chấm phẩy ';' phân tách nhiều câu lệnh."
    ),
    "TABLE_NOT_FOUND": (
        "Tên bảng không tồn tại trong cơ sở dữ liệu. Chỉ dùng các bảng được liệt kê trong <selected_schema>."
    ),
    "COLUMN_NOT_FOUND": (
        "Tên cột không tồn tại trong bảng tương ứng. Đối chiếu kỹ <selected_schema> để lấy đúng tên cột có dấu tiếng Việt."
    ),
    "UNKNOWN_QUALIFIER": (
        "Bí danh bảng hoặc qualifier không tồn tại trong mệnh đề FROM/JOIN."
    ),
    "DATE_FORMAT_MISMATCH": (
        "Lệch định dạng ngày tháng hoặc so sánh khoảng trên cột không phải ISO. "
        "Bắt buộc dùng SUBSTR đưa cột ngày về 'YYYY-MM-DD' trước khi so sánh."
    ),
    "LITERAL_NOT_IN_ENUM": (
        "Giá trị chuỗi so sánh không có trong tập giá trị mẫu của cột. Tra cứu Samples trong schema để dùng đúng chính tả."
    ),
    "SHAPE_MISMATCH": (
        "Số lượng cột được SELECT không khớp với 'type' mong đợi "
        "(ví dụ: type là number/string/date nhưng SELECT nhiều hơn 1 cột, hoặc type là table nhưng chỉ có 1 cột)."
    ),
    "SHAPE_SUSPECT_NO_AGGREGATE": (
        "Truy vấn có type là number/percent nhưng thiếu hàm tổng hợp (SUM, COUNT, AVG) hoặc LIMIT 1."
    ),
    "RANK_TIEBREAK_MISSING": (
        "Truy vấn ORDER BY ... LIMIT hoặc list thiếu khóa phụ (tie-break ID/mã) dẫn đến kết quả không xác định."
    ),
    "MONEY_SCALE_RISK": (
        "Cột tiền tệ có phần thập phân nhưng biểu thức SQL dùng REPLACE xóa cả dấu chấm và dấu phẩy, gây phóng đại giá trị."
    ),
    "ORDER_ASC_NULLABLE": (
        "Sắp xếp ASC trên cột có thể chứa NULL; trong SQLite NULL xếp trước giá trị nhỏ nhất. Cần thêm WHERE \"cột\" IS NOT NULL."
    ),
    "MIN_NULLABLE": (
        "Hàm MIN() trên cột có thể chứa NULL cần có WHERE \"cột\" IS NOT NULL để đảm bảo kết quả sạch."
    ),
    "CONCAT_IN_SELECT": (
        "Ghép chuỗi ('||' hoặc CONCAT) trực tiếp trong SELECT trên các cột dữ liệu. Hãy xuất riêng từng cột trừ khi tạo chuỗi ngày chuẩn."
    ),
    "RESULT_SHAPE_MISMATCH": (
        "Kết quả thực thi thực tế không khớp với hình dạng của type (ví dụ: type vô hướng nhưng trả về nhiều dòng hoặc nhiều cột)."
    ),
    "UNRESOLVED_ZERO": (
        "Truy vấn trả về 0 hoặc rỗng nhưng có dấu hiệu lọc sai điều kiện. Kiểm tra lại literal, ngày tháng và điều kiện JOIN."
    ),
    "STAGNATION": (
        "Câu SQL sau khi sửa bị lặp lại câu lệnh cũ hoặc lặp lại mã lỗi tương tự, dừng vòng lặp sửa lỗi."
    ),
    "DELIBERATE_ABSTAIN": (
        "Chủ đích từ chối trả lời vì thiếu dữ liệu hoặc câu hỏi không thể trả lời."
    ),
    "EXECUTION_ERROR": (
        "Lỗi thực thi truy vấn trên SQLite. Đọc kỹ thông báo lỗi chi tiết để sửa."
    ),
}


@dataclass(frozen=True)
class ValidationIssue:
    """Represents a discrete validation issue with severity, code, and remediation hint."""

    code: str
    message: str
    severity: Severity = "blocking"
    column: str | None = None
    hint: str | None = None

    def to_prompt_line(self) -> str:
        """Render diagnostic as a structured string for prompt injection."""
        effective_hint = self.hint or RULE_REGISTRY.get(self.code, "")
        if effective_hint:
            return f"-[{self.code}] {self.message} | Gợi ý: {effective_hint}"
        return f"-[{self.code}] {self.message}"

    def __str__(self) -> str:
        """String representation matching to_prompt_line for string compatibility."""
        return self.to_prompt_line()

    def lower(self) -> str:
        """Provide lowercase string conversion for backward compatibility."""
        return str(self).lower()

    def __contains__(self, item: str) -> bool:
        """Allow 'substring in issue' checks for backward compatibility."""
        return item in str(self)


def render_diagnostics(issues: Sequence[ValidationIssue | str]) -> list[str]:
    """Convert sequence of issues (ValidationIssue or raw string) to structured prompt lines."""
    rendered: list[str] = []
    for issue in issues:
        if isinstance(issue, ValidationIssue):
            rendered.append(issue.to_prompt_line())
        else:
            issue_str = str(issue).strip()
            if not issue_str.startswith("-"):
                issue_str = f"- {issue_str}"
            rendered.append(issue_str)
    return rendered


__all__ = [
    "RULE_REGISTRY",
    "Severity",
    "ValidationIssue",
    "render_diagnostics",
]
