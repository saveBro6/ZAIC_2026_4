"""Prompt templates and dynamic schema permutation for diverse candidate SQL generation."""

from __future__ import annotations

import random
from typing import Any


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

        table_groups.setdefault(tbl, []).append({
            "col_name": col_clean,
            "data_format": data_format,
            "desc": f"{desc}{sample_str}",
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
    except Exception:
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
            "The following rule sections are identified as highly relevant to this specific question:\n"
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
) -> str:
    """Build complete prompt for LLM SQLite generation with permuted schema and dynamic rulebook."""
    schema_xml = format_permuted_schema_xml(schema_keys, processed_metadata)
    rulebook_xml = format_rulebook_context(
        rulebook_text=rulebook_text,
        vector_indexer=vector_indexer,
        question=question,
        mode=rulebook_mode,
    )

    prompt_parts = [
        "<system>",
        "You are an expert SQL engineer specializing in SQLite databases for Vietnamese Small and Medium Businesses (SMBs).",
        "Your task is to write a single, clean, valid SQLite query that precisely answers the user question.",
        "",
        "CRITICAL SQL & BUSINESS GUIDELINES:",
        "1. Output ONLY the raw executable SQLite SQL statement. Do not provide conversational explanations or preambles.",
        '2. Wrap table and column names with double quotes if they contain Vietnamese diacritics, spaces, or special characters (e.g. SELECT T1."Tên hàng" FROM "ban_hang" AS T1).',
        "3. Relative Temporal Grounding: All relative date expressions (e.g. 'tháng này', 'hôm nay', 'tuần này', 'cùng kỳ năm ngoái') MUST be anchored against the provided <as_of_date>.",
        "   - 'tháng này' means from the 1st of the month of as_of_date up to and including as_of_date.",
        "4. NULL Handling in Ordering and Aggregates:",
        "   - In SQLite, NULL is considered smaller than any other value. If ordering ASC, you MUST add a NOT NULL check (e.g. WHERE \"field\" IS NOT NULL).",
        "   - When using MIN(f), ensure 'f IS NOT NULL'.",
        "5. Subqueries vs Ordering: Avoid nested subqueries for min/max calculations. Use 'ORDER BY ... LIMIT 1' instead.",
        "6. Do not perform string concatenation ('||' or CONCAT) on fields in SELECT; return the columns individually.",
        "7. Adhere strictly to the business rules in <split_rulebook> (revenue definitions, discount treatments, order status exclusions).",
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
        "You are an expert SQL engineer. Your previously generated SQLite query had validation errors or violated domain guidelines.\n"
        "Please fix the query according to the feedback below.\n"
        "Output ONLY the corrected, executable SQLite statement.\n"
        "</system>\n\n"
        f"<user_question>\n{question}\n</user_question>\n\n"
        f"<temporal_anchor>\nas_of_date: {as_of_date}\n</temporal_anchor>\n\n"
        f"<previous_sql>\n{faulty_sql}\n</previous_sql>\n\n"
        f"<validation_errors>\n{issues_str}\n</validation_errors>\n\n"
        "<corrected_sql>\n"
    )


__all__ = [
    "build_sql_correction_prompt",
    "build_sql_generation_prompt",
    "format_permuted_schema_xml",
    "format_rulebook_context",
    "permute_schema",
]
