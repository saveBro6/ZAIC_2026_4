"""Prompt templates and dynamic schema permutation for diverse candidate SQL generation."""

from __future__ import annotations

import random
from typing import Any

# Few-shot SQL examples in Vietnamese for SQLite generation
FEW_SHOT_SQL_EXAMPLES_VI: str = (
    "<few_shot_examples>\n"
    '  <example id="1">\n'
    '    <user_question>Tổng doanh thu của mặt hàng "Sữa tươi" trong tháng này là bao nhiêu?</user_question>\n'
    '    <temporal_anchor>as_of_date: 2026-05-25</temporal_anchor>\n'
    "    <selected_schema>\n"
    '      <table name="ban_hang">\n'
    '        <column name="Ngày bán" type="SLASH_DATE">Cột lưu trữ ngày lập hóa đơn bán hàng định dạng DD/MM/YYYY. | Samples: [\'02/05/2026\', \'15/05/2026\'] | Hint: Vietnamese date (D/M/YYYY).</column>\n'
    '        <column name="Thành tiền" type="CURRENCY">Tổng số tiền thanh toán của dòng hàng sau giảm giá. | Samples: [\'45,000\', \'120,000\'] | Hint: Formatted VND currency. CAST(REPLACE(col, \',\', \'\') AS REAL)</column>\n'
    '        <column name="Tên hàng" type="STRING">Tên mặt hàng bán lẻ. | Samples: [\'Sữa tươi tiệt trùng 1L\']</column>\n'
    '        <column name="Trạng thái" type="STRING">Trạng thái đơn hàng. | Samples: [\'Hoàn thành\', \'Đã hủy\']</column>\n'
    "      </table>\n"
    "    </selected_schema>\n"
    "    <sql>\n"
    'SELECT SUM(CAST(REPLACE(T1."Thành tiền", \',\', \'\') AS REAL)) AS "Tổng doanh thu"\n'
    'FROM "ban_hang" AS T1\n'
    'WHERE T1."Tên hàng" LIKE \'%Sữa tươi%\'\n'
    '  AND T1."Trạng thái" != \'Đã hủy\'\n'
    '  AND (substr(T1."Ngày bán", 7, 4) || \'-\' || substr(T1."Ngày bán", 4, 2) || \'-\' || substr(T1."Ngày bán", 1, 2)) BETWEEN \'2026-05-01\' AND \'2026-05-25\';\n'
    "    </sql>\n"
    "  </example>\n\n"
    '  <example id="2">\n'
    '    <user_question>Sản phẩm nào có đơn giá thấp nhất còn tồn kho?</user_question>\n'
    '    <temporal_anchor>as_of_date: 2026-05-25</temporal_anchor>\n'
    "    <selected_schema>\n"
    '      <table name="san_pham">\n'
    '        <column name="Tên sản phẩm" type="STRING">Tên sản phẩm trong kho. | Samples: [\'Bánh mì bơ tỏi\']</column>\n'
    '        <column name="Đơn giá" type="CURRENCY">Giá niêm yết bán lẻ của sản phẩm. | Samples: [\'15,000\', \'30,000\'] | Hint: Formatted VND currency. CAST(REPLACE(col, \',\', \'\') AS REAL)</column>\n'
    '        <column name="Số lượng tồn" type="NUMERIC">Số lượng sản phẩm còn tồn trong kho. | Samples: [10, 50]</column>\n'
    "      </table>\n"
    "    </selected_schema>\n"
    "    <sql>\n"
    'SELECT T1."Tên sản phẩm", T1."Đơn giá"\n'
    'FROM "san_pham" AS T1\n'
    'WHERE T1."Đơn giá" IS NOT NULL\n'
    '  AND T1."Số lượng tồn" > 0\n'
    'ORDER BY CAST(REPLACE(T1."Đơn giá", \',\', \'\') AS REAL) ASC\n'
    "LIMIT 1;\n"
    "    </sql>\n"
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
        desc = (
            col_meta.get("maximal_profile")
            or col_meta.get("minimal_profile")
            or "Tabular column"
        )
        samples = col_meta.get("raw_samples", [])
        sample_str = f" | Samples: {samples[:3]}" if samples else ""
        llm_hint = col_meta.get("llm_hint")
        hint_str = f" | Hint: {llm_hint}" if llm_hint else ""

        table_groups.setdefault(tbl, []).append({
            "col_name": col_clean,
            "data_format": data_format,
            "desc": f"{desc}{sample_str}{hint_str}",
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


def build_sql_generation_prompt(
    question: str,
    as_of_date: str,
    schema_keys: list[str],
    processed_metadata: dict[str, Any],
    rulebook_text: str = "",
    vector_indexer: Any | None = None,
    rulebook_mode: str = "hybrid",
    include_few_shot: bool = True,
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
        "Nhiệm vụ: Phân tích ngữ cảnh và xuất DUY NHẤT một câu truy vấn SQLite chuẩn xác, hợp lệ để trả lời câu hỏi.",
        "",
        "[HƯỚNG DẪN SQL VÀ QUY TẮC CHÍNH]",
        "1. Chỉ xuất DUY NHẤT câu lệnh SQLite SQL thô có thể thực thi được. Tuyệt đối không thêm lời giải thích, chào hỏi hoặc văn bản mở đầu/kết thúc.",
        '2. Bọc tên bảng và tên cột trong dấu ngoặc kép (double quotes) nếu chứa dấu tiếng Việt, khoảng trắng hoặc ký tự đặc biệt (ví dụ: SELECT T1."Tên hàng" FROM "ban_hang" AS T1).',
        "3. Luôn ưu tiên dùng LEFT JOIN khi nối từ bảng giao dịch/bán hàng sang bảng danh mục để bảo toàn dữ liệu khi khóa ngoại chứa NULL hoặc bị thiếu.",
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
        "   - Tìm kiếm chuỗi tiếng Việt có dấu: luôn dùng LOWER(\"col\") LIKE LOWER('%...%') để đảm bảo không phân biệt hoa/thường.",
        "",
        "[QUAN HỆ BẢNG & NGHIỆP VỤ]",
        "Tuân thủ nghiêm ngặt các quy tắc nghiệp vụ về trạng thái đơn hàng (loại trừ hủy, hoàn) và định nghĩa doanh thu trong <split_rulebook>.",
        "</system>",
        "",
    ]

    if rulebook_xml:
        prompt_parts.append(rulebook_xml)
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
) -> str:
    """Build prompt asking LLM to correct bad SQL constructions or syntax errors."""
    issues_str = "\n".join(f"- {issue}" for issue in validation_issues)

    return (
        "<system>\n"
        "Bạn là một chuyên gia kỹ thuật SQL. Câu truy vấn SQLite bạn đã sinh trước đó có lỗi kiểm tra tính hợp lệ hoặc vi phạm quy tắc nghiệp vụ/kỹ thuật.\n"
        "Vui lòng sửa lại câu truy vấn dựa trên các phản hồi bên dưới.\n"
        "Chỉ xuất DUY NHẤT câu lệnh SQLite SQL đã sửa và có thể thực thi được.\n"
        "</system>\n\n"
        f"<user_question>\n{question}\n</user_question>\n\n"
        f"<temporal_anchor>\nas_of_date: {as_of_date}\n</temporal_anchor>\n\n"
        f"<previous_sql>\n{faulty_sql}\n</previous_sql>\n\n"
        f"<validation_errors>\n{issues_str}\n</validation_errors>\n\n"
        "<corrected_sql>\n"
    )


__all__ = [
    "FEW_SHOT_SQL_EXAMPLES_VI",
    "build_sql_correction_prompt",
    "build_sql_generation_prompt",
    "format_permuted_schema_xml",
    "format_rulebook_context",
    "permute_schema",
]
