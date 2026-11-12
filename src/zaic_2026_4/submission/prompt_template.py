"""Prompt templates and dynamic schema permutation for diverse candidate SQL generation."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from zaic_2026_4.common.xml_sanitizer import (
    DATA_INSTRUCTION_SEPARATION_DIRECTIVE,
    format_schema_table_elements,
    sanitize_cdata,
    sanitize_text,
)
from zaic_2026_4.submission.rules import ValidationIssue, render_diagnostics

# Few-shot SQL examples in Vietnamese for SQLite generation
FEW_SHOT_SQL_EXAMPLES_VI: str = (
    "<few_shot_examples>\n"
    '  <example id="1">\n'
    '    <user_question>Tổng doanh thu bán hàng thực tế trong tháng này (tính đến hôm nay) là bao nhiêu?</user_question>\n'
    '    <temporal_anchor>as_of_date: 2026-05-25</temporal_anchor>\n'
    "    <selected_schema>\n"
    '      <table name="don_hang">\n'
    '        <column name="Mã đơn" type="STRING">Mã định danh duy nhất của hóa đơn. | Samples: [\'DH001\', \'DH002\']</column>\n'
    '        <column name="Ngày đặt" type="SLASH_DATE">Ngày ghi nhận đơn hàng định dạng DD/MM/YYYY. | Samples: [\'02/05/2026\', \'15/05/2026\'] | Hint: Vietnamese date (DD/MM/YYYY).</column>\n'
    '        <column name="Tổng tiền" type="CURRENCY">Tổng giá trị thanh toán thực tế của đơn hàng. | Samples: [\'1.200.000\', \'3.500.000 đ\'] | Hint: Formatted VND currency string.</column>\n'
    '        <column name="Trạng thái" type="STRING">Trạng thái xử lý đơn hàng. | Samples: [\'Hoàn thành\', \'Đã hủy\', \'Trả hàng\']</column>\n'
    "      </table>\n"
    "    </selected_schema>\n"
    "    <output>\n"
    '{"reasoning": "<think>1. Thời gian: \'tháng này\' tính từ ngày đầu tháng đến as_of_date: \'2026-05-01\' đến \'2026-05-25\'. Cột \'Ngày đặt\' lưu dạng DD/MM/YYYY nên chuyển về ISO bằng (SUBSTR(T1.\\"Ngày đặt\\", 7, 4) || \'-\' || SUBSTR(T1.\\"Ngày đặt\\", 4, 2) || \'-\' || SUBSTR(T1.\\"Ngày đặt\\", 1, 2)). 2. Tiền tệ: Cột \'Tổng tiền\' dạng chuỗi chứa dấu chấm và ký hiệu đ, cần làm sạch và ép kiểu INTEGER: CAST(REPLACE(REPLACE(REPLACE(T1.\\"Tổng tiền\\", \'đ\', \'\'), \'.\', \'\'), \',\', \'\') AS INTEGER). 3. Bộ lọc: Loại trừ đơn hủy và trả hàng: T1.\\"Trạng thái\\" NOT IN (\'Đã hủy\', \'Trả hàng\').</think>", "sql": "SELECT SUM(CAST(REPLACE(REPLACE(REPLACE(T1.\\"Tổng tiền\\", \'đ\', \'\'), \'.\', \'\'), \',\', \'\') AS INTEGER)) AS \\"Tổng doanh thu\\" FROM \\"don_hang\\" AS T1 WHERE (T1.\\"Trạng thái\\" NOT IN (\'Đã hủy\', \'Trả hàng\') OR T1.\\"Trạng thái\\" IS NULL) AND (SUBSTR(T1.\\"Ngày đặt\\", 7, 4) || \'-\' || SUBSTR(T1.\\"Ngày đặt\\", 4, 2) || \'-\' || SUBSTR(T1.\\"Ngày đặt\\", 1, 2)) BETWEEN \'2026-05-01\' AND \'2026-05-25\';", "type": "number"}\n'
    "    </output>\n"
    "  </example>\n\n"
    '  <example id="2">\n'
    '    <user_question>Mặt hàng nào có số lượng bán ra nhiều nhất trong kỳ?</user_question>\n'
    '    <temporal_anchor>as_of_date: 2026-05-25</temporal_anchor>\n'
    "    <selected_schema>\n"
    '      <table name="chi_tiet_ban_hang">\n'
    '        <column name="Mã hàng" type="STRING">Mã sản phẩm duy nhất. | Samples: [\'MH01\', \'MH02\']</column>\n'
    '        <column name="Tên hàng" type="STRING">Tên mặt hàng bán lẻ. | Samples: [\'Gạo thơm ST25\', \'Nước mắm Phú Quốc\']</column>\n'
    '        <column name="Số lượng" type="NUMERIC">Số lượng sản phẩm bán ra. | Samples: [5, 12]</column>\n'
    '        <column name="Trạng thái đơn" type="STRING">Trạng thái xử lý. | Samples: [\'Đã giao\', \'Hủy\']</column>\n'
    "      </table>\n"
    "    </selected_schema>\n"
    "    <output>\n"
    '{"reasoning": "<think>1. Nghiệp vụ: Tìm mặt hàng bán chạy nhất theo tổng số lượng bán ra, loại trừ đơn hủy. 2. Nhóm và tính toán: GROUP BY T1.\\"Mã hàng\\", T1.\\"Tên hàng\\" và SUM(T1.\\"Số lượng\\"). 3. Sắp xếp & Tie-breaking: Xếp giảm dần theo tổng số lượng và bắt buộc thêm mã định danh T1.\\"Mã hàng\\" ASC để phá vỡ đồng hạng. LIMIT 1 trả về tên hàng.</think>", "sql": "SELECT T1.\\"Tên hàng\\" FROM \\"chi_tiet_ban_hang\\" AS T1 WHERE (T1.\\"Trạng thái đơn\\" != \'Hủy\' OR T1.\\"Trạng thái đơn\\" IS NULL) AND T1.\\"Số lượng\\" IS NOT NULL GROUP BY T1.\\"Mã hàng\\", T1.\\"Tên hàng\\" ORDER BY SUM(T1.\\"Số lượng\\") DESC, T1.\\"Mã hàng\\" ASC LIMIT 1;", "type": "string"}\n'
    "    </output>\n"
    "  </example>\n"
    "</few_shot_examples>"
)


def permute_schema(reduced_schema: set[str], seed: int) -> list[str]:
    """Deterministically permute tables and columns of the reduced schema using a random seed.

    Preserves table-level grouping while randomizing the order of tables and
    the order of columns within each table.
    """
    rng = random.Random(seed)

    # Group columns by table
    tables: dict[str, list[str]] = {}
    for key in sorted(reduced_schema):
        if "." in key:
            tbl, col = key.split(".", 1)
        else:
            tbl, col = "", key
        tables.setdefault(tbl, []).append(col)

    # Permute table order
    table_names = list(tables.keys())
    rng.shuffle(table_names)

    permuted_keys: list[str] = []
    for tbl in table_names:
        cols = list(tables[tbl])
        rng.shuffle(cols)
        for col in cols:
            permuted_keys.append(f"{tbl}.{col}" if tbl else col)

    return permuted_keys


def format_permuted_schema_xml(
    schema_keys: list[str],
    processed_metadata: dict[str, Any],
) -> str:
    """Format permuted schema keys into structured XML for LLM SQL generation prompt."""
    target_dict = processed_metadata.get("metadata_profile", processed_metadata)

    # Group by table preserving the permuted table and column order
    table_groups: dict[str, list[dict[str, Any]]] = {}
    for col_key in schema_keys:
        if "." in col_key:
            tbl, raw_col = col_key.split(".", 1)
        else:
            tbl, raw_col = "", col_key

        col_clean = raw_col.strip('"')
        meta_key = f"{tbl}.{col_clean}" if tbl else col_clean
        col_meta = target_dict.get(meta_key, {})

        data_format = col_meta.get("data_format", "STRING")
        desc = col_meta.get("maximal_profile", "")
        llm_hint = col_meta.get("llm_hint")
        hint_str = f" | Hint: {llm_hint}" if llm_hint else ""

        table_groups.setdefault(tbl, []).append({
            "col_name": col_clean,
            "data_format": data_format,
            "desc": f"{desc}{hint_str}".strip(),
        })

    lines = ["<selected_schema>"]
    for tbl_name, cols in table_groups.items():
        lines.extend(format_schema_table_elements(tbl_name, cols))
    lines.append("</selected_schema>")

    return "\n".join(lines)


def format_rulebook_context(
    rulebook_text: str,
    vector_indexer: Any | None,
    question: str,
    mode: str = "hybrid",
    top_k: int = 3,
) -> str:
    """Dynamically format split rulebook context under full, retrieval, or hybrid mode."""
    clean_rulebook = sanitize_text(rulebook_text.strip()) if rulebook_text else ""
    rule_mode = mode.lower()

    if rule_mode == "full" or vector_indexer is None:
        if not clean_rulebook:
            return ""
        return f"<split_rulebook>\n{clean_rulebook}\n</split_rulebook>"

    # Retrieve relevant rule sections via FAISS
    retrieved_chunks: list[str] = []
    try:
        results = vector_indexer.search_rules(question, top_k=top_k)
        for r in results:
            txt = r.get("text", "").strip()
            if txt:
                retrieved_chunks.append(sanitize_text(txt))
    except Exception:  # noqa: BLE001
        retrieved_chunks = []

    retrieved_text = "\n---\n".join(retrieved_chunks)

    if rule_mode == "retrieval":
        if not retrieved_text:
            return f"<split_rulebook>\n{clean_rulebook}\n</split_rulebook>" if clean_rulebook else ""
        return f"<relevant_business_rules>\n{retrieved_text}\n</relevant_business_rules>"

    # Hybrid mode (Default): Full rulebook + highlighted focal sections
    sections = []
    if clean_rulebook:
        sections.append(f"<split_rulebook>\n{clean_rulebook}\n</split_rulebook>")
    if retrieved_text:
        sections.append(
            "<focal_rulebook_sections>\n"
            "Các điều khoản quy tắc sau đây được xác định là đặc biệt liên quan đến câu hỏi này:\n"
            f"{retrieved_text}\n"
            "</focal_rulebook_sections>"
        )

    return "\n\n".join(sections)


def format_grounded_entities_xml(
    grounded_entities: list[dict[str, Any]] | None,
) -> str:
    """Format grounded literal mappings into clean XML context for SQL generation."""
    if not grounded_entities:
        return ""

    lines = [
        "<grounded_entities>",
        "Các thực thể sau trong câu hỏi đã được đối soát chính xác với giá trị thực tế trong cơ sở dữ liệu:",
    ]
    for item in grounded_entities:
        mention = sanitize_text(item.get("mention", "").strip())
        col = item.get("column", "").strip()
        can_val = sanitize_text(item.get("canonical_value", "").strip())
        if mention and can_val:
            if "." in col:
                tbl, cname = col.split(".", 1)
                col_fmt = f'{tbl}."{cname}"'
            else:
                col_fmt = f'"{col}"'
            lines.append(
                f'- Từ trong câu hỏi: "{mention}" -> Cột: {col_fmt} | Giá trị chuẩn trong DB: "{can_val}"'
            )

    lines.extend([
        "LƯU Ý BẮT BUỘC: Khi viết mệnh đề WHERE/HAVING, BẮT BUỘC sử dụng đúng 'Giá trị chuẩn trong DB' và tên cột tương ứng ở trên.",
        "TUYỆT ĐỐI KHÔNG dùng từ ngữ viết tắt hoặc không dấu từ câu hỏi để so sánh chuỗi.",
        "</grounded_entities>",
    ])
    return "\n".join(lines)


REPAIR_CONTRACT_TEXT: str = (
    "Chỉ sửa tối thiểu để giải quyết đúng các mục trong <repair_diagnostics>.\n"
    "Giữ nguyên logic thời gian, bộ lọc, JOIN và thứ tự sắp xếp đang đúng trong <accepted_sql>.\n"
    "Chỉ dùng bảng/cột có trong <selected_schema>; không thêm cột không tồn tại.\n"
    'Trả lại đúng định dạng JSON {"reasoning": "…", "sql": "…", "type": "…"} như lượt sinh SQL.\n'
    "Nếu <accepted_sql> đã đúng, trả lại nguyên văn câu lệnh đó.\n"
    'Nếu dữ liệu phát hành không đủ để trả lời, trả {"sql": "", "type": "unanswerable"}.'
)


@dataclass(frozen=True)
class RepairContext:
    """Repair-specific inputs appended after the shared generation prefix."""

    diagnostics: Sequence[ValidationIssue | str]
    previous_sql: str = ""
    previous_type: str = "number"
    accepted_sql: str = ""
    last_output: str = ""
    max_chars: int = 4000
    previous_attempts: tuple[tuple[str, tuple[str, ...]], ...] = ()


def _build_prompt_prefix(
    question: str,
    as_of_date: str,
    schema_xml: str,
    rulebook_xml: str,
    grounded_entities: list[dict[str, Any]] | None,
    include_few_shot: bool,
) -> str:
    """Render the shared prompt prefix reused by the generation and repair turns."""
    safe_as_of_date = sanitize_text(as_of_date)
    safe_question = sanitize_text(question)

    prompt_parts = [
        "<system>",
        "Bạn là chuyên gia kỹ sư Text-to-SQL chuyên sâu về SQLite cho hệ thống dữ liệu doanh nghiệp (SMB) tại Việt Nam.",
        f"{DATA_INSTRUCTION_SEPARATION_DIRECTIVE}",
        "",
        'Nhiệm vụ: Phân tích ngữ cảnh và xuất DUY NHẤT một đối tượng JSON hợp lệ theo đúng cấu trúc: {"reasoning": "<think>...</think>", "sql": "<câu lệnh SQLite SQL>", "type": "<type>"}.',
        "",
        "[QUY TẮC ĐỊNH DẠNG ĐẦU RA JSON & KIỂU DỮ LIỆU DATAMATE]",
        '1. Trường "reasoning": BẮT BUỘC mở đầu bằng <think> và kết thúc bằng </think>. Thực hiện lập kế hoạch Chain-of-Thought ngắn gọn và có cấu trúc trước khi viết câu truy vấn SQL:',
        '   - Kiểm tra Schema Metadata & Mẫu dữ liệu (Samples): Quan sát data_format, llm_hint và Samples thực tế của các cột liên quan.',
        '   - Chuẩn hóa Ngày tháng về ISO DATE: Nếu cột ngày lưu dạng chuỗi chưa chuẩn ISO (ví dụ DD/MM/YYYY, văn bản hoặc timestamp), dựa vào mẫu thực tế để áp dụng biểu thức biến đổi phù hợp (như SUBSTR, date, strftime) đưa về chuỗi ISO YYYY-MM-DD (hoặc YYYY-MM theo tháng) trước khi lọc hoặc so sánh với as_of_date.',
        '   - Làm sạch & Ép kiểu Tiền tệ về INTEGER: Nếu cột tiền tệ là chuỗi có chứa ký tự phân cách nghìn (., ,), ký hiệu đơn vị (đ, VND), làm sạch và ép kiểu về INTEGER để đảm bảo kết quả đúng từng đồng theo quy định DataMate. Với trung bình tiền bọc ROUND(AVG(...), 0). Chỉ dùng REAL cho số lượng có phần lẻ hoặc tỷ lệ chia.',
        '   - Bảng, JOIN & Bộ lọc: Xác định bảng chính/phụ, dùng LEFT JOIN cho bảng danh mục, lọc loại trừ đơn hủy/trả theo rulebook, IS NOT NULL cho ORDER BY ... ASC.',
        '   - Sắp xếp & Tie-breaking: Luôn phá vỡ đồng hạng bằng khóa chính / mã định danh: ORDER BY ..., mã ASC.',
        '2. Trường "sql": Câu lệnh SQLite SQL thô có thể thực thi được. Tuyệt đối không thêm lời giải thích, chào hỏi hoặc văn bản mở đầu/kết thúc bên ngoài JSON.',
        '3. Trường "type": BẮT BUỘC là 1 trong các kiểu dữ liệu hợp lệ sau theo quy chuẩn cuộc thi DataMate:',
        '   - "number": Câu hỏi về giá trị số (tổng tiền VNĐ, số lượng, đếm, chi phí). Mệnh đề SELECT trả về 1 dòng, 1 cột số.',
        '   - "percent": Câu hỏi về tỷ lệ phần trăm (tính theo %, thang đo 0-100, ví dụ 12.45 tương ứng 12.45%). Mệnh đề SELECT trả về 1 dòng, 1 cột tỷ lệ.',
        '   - "string": Câu hỏi về tên hoặc mã của một đối tượng/thực thể cụ thể (sản phẩm, khách hàng, chi nhánh, thứ trong tuần). Mệnh đề SELECT trả về 1 dòng, 1 cột chuỗi.',
        '   - "date": Câu hỏi về ngày tháng theo chuẩn ISO (YYYY-MM-DD hoặc YYYY-MM). Mệnh đề SELECT trả về 1 dòng, 1 cột ngày tháng.',
        '   - "boolean": Câu hỏi dạng Có/Không, Đúng/Sai. Mệnh đề SELECT trả về 1 dòng, 1 cột logic (1/0 hoặc true/false).',
        '   - "list": Câu hỏi yêu cầu danh sách các tên hoặc số (ví dụ Top 3 sản phẩm, danh sách khách hàng). Mệnh đề SELECT BẮT BUỘC CHỈ CHỌN 1 CỘT DUY NHẤT với nhiều dòng theo đúng thứ tự yêu cầu.',
        '   - "table": Câu hỏi yêu cầu bảng dữ liệu nhiều cột. Mệnh đề SELECT chọn nhiều cột thông tin.',
        '   - "unanswerable": Khi cơ sở dữ liệu/schema hoàn toàn không có thông tin để trả lời câu hỏi. Khi đó đặt "sql": "" và "type": "unanswerable".',
        '   - "abstain": Khi không thể xác định câu truy vấn an toàn hoặc rủi ro sai sót cao. Khi đó đặt "sql": "" và "type": "abstain".',
        "",
        "[HƯỚNG DẪN SQL VÀ QUY TẮC CHÍNH]",
        '1. Bọc tên bảng và tên cột trong dấu ngoặc kép (double quotes) nếu chứa dấu tiếng Việt, khoảng trắng hoặc ký tự đặc biệt (ví dụ: SELECT T1."Tên hàng" FROM "ban_hang" AS T1).',
        "2. Luôn ưu tiên dùng LEFT JOIN khi nối từ bảng giao dịch/bán hàng sang bảng danh mục để bảo toàn dữ liệu khi khóa ngoại chứa NULL hoặc bị thiếu.",
        "",
        "[XỬ LÝ DỮ LIỆU & BẪY KỸ THUẬT SQLITE THEO PROFILING]",
        "1. Neo mốc thời gian (<temporal_anchor>):",
        "   - Mọi mốc thời gian tương đối ('tháng này', 'tuần này', 'hôm nay') BẮT BUỘC phải được tính toán dựa trên ngày mốc as_of_date được cung cấp.",
        "2. Xử lý NULL & Phép toán:",
        "   - Trong SQLite, NULL đứng đầu khi xếp tăng dần (ASC). Khi dùng ORDER BY <col> ASC hoặc MIN(<col>), BẮT BUỘC thêm điều kiện WHERE <col> IS NOT NULL (trừ khi Profile cột khẳng định 0% NULL).",
        "   - Khi chia tỷ lệ/phần trăm, luôn nhân tử số với 1.0 và bọc mẫu số bằng NULLIF(..., 0) để chống chia số nguyên và lỗi chia cho 0.",
        "3. Quy tắc Truy vấn & Tie-Breaking:",
        "   - TUYỆT ĐỐI KHÔNG dùng subquery lồng nhau để tìm min/max; ưu tiên cấu trúc phẳng: ORDER BY <metric> DESC/ASC LIMIT 1.",
        "   - Khi xếp hạng (ranking/top-k), BẮT BUỘC thêm khóa chính hoặc mã định danh vào mệnh đề sắp xếp để phá vỡ đồng hạng: ORDER BY <metric> DESC, <primary_key_or_code> ASC.",
        "   - Trả về các cột riêng biệt; KHÔNG dùng toán tử nối chuỗi '||' trong mệnh đề SELECT trừ khi có yêu cầu cụ thể.",
        "4. Ép kiểu & Khớp chuỗi theo <selected_schema> tham khảo Hint được cung cấp.",
        "   - Khi câu hỏi đề cập đến thực thể có trong <grounded_entities>, BẮT BUỘC sử dụng đúng 'Giá trị chuẩn trong DB' và tên cột tương ứng; tuyệt đối không dùng chuỗi không dấu hoặc viết tắt từ câu hỏi.",
        "   - Tìm kiếm chuỗi tiếng Việt có dấu khác: luôn dùng LOWER(\"col\") LIKE LOWER('%...%') để đảm bảo không phân biệt hoa/thường.",
        "",
        "[QUAN HỆ BẢNG & NGHIỆP VỤ]",
        "Tuân thủ nghiêm ngặt các quy tắc nghiệp vụ về trạng thái đơn hàng (loại trừ hủy, hoàn) và định nghĩa doanh thu trong <split_rulebook>.",
        "</system>",
        "",
    ]

    if rulebook_xml:
        prompt_parts.append(rulebook_xml)
        prompt_parts.append("")

    grounded_xml = format_grounded_entities_xml(grounded_entities)
    if grounded_xml:
        prompt_parts.append(grounded_xml)
        prompt_parts.append("")

    prompt_parts.extend([
        f"<temporal_anchor>\nas_of_date: {safe_as_of_date}\n</temporal_anchor>",
        "",
        schema_xml,
    ])

    if include_few_shot:
        prompt_parts.extend([
            "",
            FEW_SHOT_SQL_EXAMPLES_VI,
        ])

    prompt_parts.extend([
        "",
        f"<user_question>\n{safe_question}\n</user_question>",
    ])

    return "\n".join(prompt_parts)


def _format_repair_task(repair: RepairContext) -> str:
    """Render the <repair_task> section carrying diagnostics and the repair contract."""
    previous_output_json = json.dumps(
        {"sql": repair.previous_sql, "type": repair.previous_type},
        ensure_ascii=False,
    )
    rendered_issues = render_diagnostics(repair.diagnostics)
    diagnostics = "\n".join(rendered_issues)
    if not diagnostics:
        diagnostics = "- (no diagnostics were reported)"

    sections = [
        "<repair_task>",
        (
            "<accepted_sql><![CDATA["
            f"{sanitize_cdata(repair.accepted_sql, repair.max_chars)}"
            "]]></accepted_sql>"
        ),
        (
            "<previous_output><![CDATA["
            f"{sanitize_cdata(previous_output_json, repair.max_chars)}"
            "]]></previous_output>"
        ),
    ]
    if repair.last_output and repair.last_output.strip() != previous_output_json:
        sections.append(
            "<last_raw_output><![CDATA["
            f"{sanitize_cdata(repair.last_output, repair.max_chars)}"
            "]]></last_raw_output>"
        )

    if repair.previous_attempts:
        attempt_lines = [
            "<previous_attempts>",
            "Các câu lệnh trước đó đã thử nhưng không hợp lệ (không lặp lại):",
        ]
        for idx, (attempt_sql, issue_codes) in enumerate(repair.previous_attempts[-2:], 1):
            codes_str = ", ".join(issue_codes) if issue_codes else "None"
            attempt_lines.append(f"- Lần thử {idx}: `{attempt_sql}` -> Lỗi: [{codes_str}]")
        attempt_lines.append("</previous_attempts>")
        sections.append("\n".join(attempt_lines))

    sections.extend([
        "<repair_diagnostics>",
        diagnostics,
        "</repair_diagnostics>",
        "<repair_contract>",
        REPAIR_CONTRACT_TEXT,
        "</repair_contract>",
        "</repair_task>",
    ])
    return "\n".join(sections)


def build_sql_generation_prompt(
    question: str,
    as_of_date: str,
    schema_keys: list[str],
    processed_metadata: dict[str, Any],
    rulebook_text: str = "",
    vector_indexer: Any | None = None,
    rulebook_mode: str = "hybrid",
    include_few_shot: bool = True,
    grounded_entities: list[dict[str, Any]] | None = None,
    repair: RepairContext | None = None,
    schema_xml_override: str | None = None,
    rulebook_xml_override: str | None = None,
) -> str:
    """Build the SQL generation prompt, optionally followed by a repair task section.

    Passing a RepairContext keeps the prompt prefix byte-identical to the generation
    turn, so the SQL rules, rulebook and schema stay consistent between turns and
    vLLM prefix caching can reuse the shared prefix.
    """
    schema_xml = (
        schema_xml_override
        if schema_xml_override is not None
        else format_permuted_schema_xml(schema_keys, processed_metadata)
    )
    if rulebook_xml_override is not None:
        rulebook_xml = rulebook_xml_override
    else:
        rulebook_xml = format_rulebook_context(
            rulebook_text=rulebook_text,
            vector_indexer=vector_indexer,
            question=question,
            mode=rulebook_mode,
        )

    prefix = _build_prompt_prefix(
        question=question,
        as_of_date=as_of_date,
        schema_xml=schema_xml,
        rulebook_xml=rulebook_xml,
        grounded_entities=grounded_entities,
        include_few_shot=include_few_shot,
    )

    if repair is None:
        return f"{prefix}\n\n<output>"
    return f"{prefix}\n\n{_format_repair_task(repair)}\n\n<output>"


def build_sql_correction_prompt(
    question: str,
    as_of_date: str,
    faulty_sql: str,
    validation_issues: list[str],
    schema_xml: str | None = None,
    previous_type: str = "number",
    grounded_entities: list[dict[str, Any]] | None = None,
    last_output: str | None = None,
    accepted_sql: str | None = None,
    rulebook_text: str = "",
    vector_indexer: Any | None = None,
    rulebook_mode: str = "hybrid",
    include_few_shot: bool = True,
) -> str:
    """Build a repair prompt through the unified generation pipeline (deprecated wrapper).

    Prefer passing a RepairContext to build_sql_generation_prompt directly, which keeps
    the prompt prefix byte-identical to the generation turn and lets the caller reuse a
    pre-rendered schema and rulebook.
    """
    repair = RepairContext(
        diagnostics=list(validation_issues),
        previous_sql=faulty_sql,
        previous_type=previous_type,
        accepted_sql=accepted_sql if accepted_sql is not None else faulty_sql,
        last_output=last_output or "",
    )

    return build_sql_generation_prompt(
        question=question,
        as_of_date=as_of_date,
        schema_keys=[],
        processed_metadata={},
        rulebook_text=rulebook_text,
        vector_indexer=vector_indexer,
        rulebook_mode=rulebook_mode,
        include_few_shot=include_few_shot,
        grounded_entities=grounded_entities,
        repair=repair,
        schema_xml_override=schema_xml or "",
    )


__all__ = [
    "FEW_SHOT_SQL_EXAMPLES_VI",
    "REPAIR_CONTRACT_TEXT",
    "RepairContext",
    "build_sql_correction_prompt",
    "build_sql_generation_prompt",
    "format_grounded_entities_xml",
    "format_permuted_schema_xml",
    "format_rulebook_context",
    "permute_schema",
]
