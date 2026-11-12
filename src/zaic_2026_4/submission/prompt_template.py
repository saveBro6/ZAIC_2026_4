"""Prompt templates and dynamic schema permutation for diverse candidate SQL generation."""

from __future__ import annotations

import random
from typing import Any

# Few-shot SQL examples in Vietnamese for SQLite generation
FEW_SHOT_SQL_EXAMPLES_VI: str = (
    "<few_shot_examples>\n"
    '  <example id="1">\n'
    '    <user_question>Tổng học phí thực thu của khóa học "Lập trình Python" trong tháng này là bao nhiêu?</user_question>\n'
    '    <temporal_anchor>as_of_date: 2026-05-25</temporal_anchor>\n'
    "    <selected_schema>\n"
    '      <table name="dang_ky_khoa_hoc">\n'
    '        <column name="Ngày ghi danh" type="SLASH_DATE">Cột lưu trữ ngày ghi danh học viên định dạng DD/MM/YYYY. | Samples: [\'02/05/2026\', \'15/05/2026\'] | Hint: Vietnamese date (D/M/YYYY).</column>\n'
    '        <column name="Học phí thực thu" type="CURRENCY">Tổng số tiền học phí thực tế sau miễn giảm. | Samples: [\'1,200,000\', \'3,500,000\'] | Hint: Formatted VND currency. CAST(REPLACE(col, \',\', \'\') AS REAL)</column>\n'
    '        <column name="Tên khóa học" type="STRING">Tên học phần hoặc chuyên đề đào tạo. | Samples: [\'Lập trình Python chuyên sâu\']</column>\n'
    '        <column name="Trạng thái học vụ" type="STRING">Trạng thái hồ sơ học vụ. | Samples: [\'Đã ghi danh\', \'Bảo lưu\']</column>\n'
    "      </table>\n"
    "    </selected_schema>\n"
    "    <output>\n"
    '{"sql": "SELECT SUM(CAST(REPLACE(T1.\\"Học phí thực thu\\", \',\', \'\') AS REAL)) AS \\"Tổng học phí\\" FROM \\"dang_ky_khoa_hoc\\" AS T1 WHERE LOWER(T1.\\"Tên khóa học\\") LIKE LOWER(\'%Lập trình Python%\') AND T1.\\"Trạng thái học vụ\\" != \'Bảo lưu\' AND (substr(T1.\\"Ngày ghi danh\\", 7, 4) || \'-\' || substr(T1.\\"Ngày ghi danh\\", 4, 2) || \'-\' || substr(T1.\\"Ngày ghi danh\\", 1, 2)) BETWEEN \'2026-05-01\' AND \'2026-05-25\';", "type": "number"}\n'
    "    </output>\n"
    "  </example>\n\n"
    '  <example id="2">\n'
    '    <user_question>Thiết bị nào có kinh phí duy trì thấp nhất còn số lượng khả dụng?</user_question>\n'
    '    <temporal_anchor>as_of_date: 2026-05-25</temporal_anchor>\n'
    "    <selected_schema>\n"
    '      <table name="thiet_bi_phong_lab">\n'
    '        <column name="Mã thiết bị" type="STRING">Mã định danh duy nhất của thiết bị nghiên cứu. | Samples: [\'TB-001\', \'TB-015\']</column>\n'
    '        <column name="Tên thiết bị" type="STRING">Tên trang thiết bị phòng thí nghiệm. | Samples: [\'Kính hiển vi quang học\']</column>\n'
    '        <column name="Kinh phí duy trì" type="CURRENCY">Định mức chi phí bảo dưỡng định kỳ. | Samples: [\'350,000\', \'1,200,000\'] | Hint: Formatted VND currency. CAST(REPLACE(col, \',\', \'\') AS REAL)</column>\n'
    '        <column name="Số lượng khả dụng" type="NUMERIC">Số lượng thiết bị sẵn sàng vận hành. | Samples: [2, 10]</column>\n'
    "      </table>\n"
    "    </selected_schema>\n"
    "    <output>\n"
    '{"sql": "SELECT T1.\\"Tên thiết bị\\" FROM \\"thiet_bi_phong_lab\\" AS T1 WHERE T1.\\"Kinh phí duy trì\\" IS NOT NULL AND T1.\\"Số lượng khả dụng\\" > 0 ORDER BY CAST(REPLACE(T1.\\"Kinh phí duy trì\\", \',\', \'\') AS REAL) ASC, T1.\\"Mã thiết bị\\" ASC LIMIT 1;", "type": "string"}\n'
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
        if tbl_name:
            lines.append(f'  <table name="{tbl_name}">')
            for c in cols:
                lines.append(f'    <column name="{c["col_name"]}" type="{c["data_format"]}">{c["desc"]}</column>')
            lines.append("  </table>")
        else:
            for c in cols:
                lines.append(f'  <column name="{c["col_name"]}" type="{c["data_format"]}">{c["desc"]}</column>')
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
    clean_rulebook = rulebook_text.strip() if rulebook_text else ""
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
                retrieved_chunks.append(txt)
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
        mention = item.get("mention", "").strip()
        col = item.get("column", "").strip()
        can_val = item.get("canonical_value", "").strip()
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
) -> str:
    """Build complete prompt for LLM SQLite generation with permuted schema, Vietnamese guidelines, and few-shot examples."""
    schema_xml = format_permuted_schema_xml(schema_keys, processed_metadata)
    rulebook_xml = format_rulebook_context(
        rulebook_text=rulebook_text,
        vector_indexer=vector_indexer,
        question=question,
        mode=rulebook_mode,
    )

    prompt_parts = [
        "<system>",
        "Bạn là chuyên gia kỹ sư Text-to-SQL chuyên sâu về SQLite cho hệ thống dữ liệu doanh nghiệp (SMB) tại Việt Nam.",
        'Nhiệm vụ: Phân tích ngữ cảnh và xuất DUY NHẤT một đối tượng JSON hợp lệ theo đúng cấu trúc: {"sql": "<câu lệnh SQLite SQL>", "type": "<type>"}.',
        "",
        "[QUY TẮC ĐỊNH DẠNG ĐẦU RA JSON & KIỂU DỮ LIỆU DATAMATE]",
        '1. Trường "sql": Câu lệnh SQLite SQL thô có thể thực thi được. Tuyệt đối không thêm lời giải thích, chào hỏi hoặc văn bản mở đầu/kết thúc bên ngoài JSON.',
        '2. Trường "type": BẮT BUỘC là 1 trong các kiểu dữ liệu hợp lệ sau theo quy chuẩn cuộc thi DataMate:',
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
        f"<temporal_anchor>\nas_of_date: {as_of_date}\n</temporal_anchor>",
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
        f"<user_question>\n{question}\n</user_question>",
        "",
        "<output>",
    ])

    return "\n".join(prompt_parts)


def build_sql_correction_prompt(
    question: str,
    as_of_date: str,
    faulty_sql: str,
    validation_issues: list[str],
    schema_xml: str | None = None,
    previous_type: str = "number",
    grounded_entities: list[dict[str, Any]] | None = None,
) -> str:
    """Build prompt asking LLM to correct bad SQL constructions, runtime execution errors, or empty results."""
    issues_str = "\n".join(f"- {issue}" for issue in validation_issues)

    prompt_parts = [
        "<system>",
        "Bạn là một chuyên gia kỹ thuật SQL. Câu truy vấn SQLite hoặc định dạng JSON trước đó gặp lỗi thực thi hoặc kết quả không phù hợp.",
        (
            "Vui lòng sửa lại và xuất DUY NHẤT một đối tượng JSON hợp lệ theo cấu trúc: "
            '{"sql": "<câu lệnh SQLite SQL đã sửa>", "type": "<type>"} '
            'với type là 1 trong các giá trị: ("number", "percent", "string", "date", "boolean", "list", "table", "unanswerable", "abstain").'
        ),
        "</system>",
        "",
        f"<user_question>\n{question}\n</user_question>",
        "",
        f"<temporal_anchor>\nas_of_date: {as_of_date}\n</temporal_anchor>",
    ]

    grounded_xml = format_grounded_entities_xml(grounded_entities)
    if grounded_xml:
        prompt_parts.extend(["", grounded_xml])

    if schema_xml:
        prompt_parts.extend(["", schema_xml])

    prompt_parts.extend([
        "",
        f"<previous_output>\n{{\"sql\": {faulty_sql!r}, \"type\": {previous_type!r}}}\n</previous_output>",
        "",
        f"<validation_errors>\n{issues_str}\n</validation_errors>",
        "",
        "<output>",
    ])

    return "\n".join(prompt_parts)


__all__ = [
    "FEW_SHOT_SQL_EXAMPLES_VI",
    "build_sql_correction_prompt",
    "build_sql_generation_prompt",
    "format_grounded_entities_xml",
    "format_permuted_schema_xml",
    "format_rulebook_context",
    "permute_schema",
]
