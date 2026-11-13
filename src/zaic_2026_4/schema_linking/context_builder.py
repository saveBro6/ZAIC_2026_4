"""Module for constructing focused schema and 5 context variants for Task-Aligned Schema Linking.

Implements the 5 prompt variants from 'Automatic Metadata Extraction for Text-to-SQL':
  - Case a (Focused + Minimal Profile)
  - Case b (Focused + Maximal Profile)
  - Case c (Full + Minimal Profile)
  - Case d (Full + Maximal Profile)
  - Case e (Focused + Full Profile with Prompt-Level SME & Rulebook Context)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from zaic_2026_4.common.query_intent import (
    QueryIntent,
    detect_query_intent,
    format_intent_hint_xml,
)
from zaic_2026_4.common.temporal_resolver import format_temporal_anchor_xml
from zaic_2026_4.common.xml_sanitizer import (
    build_column_attributes,
    format_schema_table_elements,
)
from zaic_2026_4.constants import FEW_SHOT_SQL_EXAMPLES


@dataclass
class SchemaVariant:
    """Represents one of the 5 schema prompt variants."""

    case_name: str
    schema_cols: set[str]
    profile_type: str
    formatted_prompt: str
    sme_context: str = ""
    rulebook_context: str = ""
    extra_metadata: dict[str, Any] = field(default_factory=dict)


def parse_case_spec(cases: str | list[str] | set[str] | None) -> list[str]:
    """Parse case specification into an ordered list of case identifiers ('a' through 'e').

    Supports inputs like 'abcde', 'abce', 'a,b,c,e', ['a', 'b', 'c', 'e'], 'case_a', etc.
    If cases is None or empty, defaults to ['a', 'b', 'c', 'd', 'e'].

    Raises:
        ValueError: If cases contains no valid case identifiers from 'a' to 'e'.
    """
    if cases is None:
        return ["a", "b", "c", "d", "e"]

    valid_keys = ("a", "b", "c", "d", "e")

    if isinstance(cases, str):
        raw = cases.strip().lower()
        if not raw:
            return ["a", "b", "c", "d", "e"]

        # Check if delimiter separated (comma, space, semicolon, pipe)
        if any(sep in raw for sep in (",", " ", ";", "|")):
            tokens = re.split(r"[,;\s|]+", raw)
            selected = []
            for t in tokens:
                c = t.replace("case_", "").strip()
                if c in valid_keys and c not in selected:
                    selected.append(c)
        else:
            selected = []
            for char in raw:
                if char in valid_keys and char not in selected:
                    selected.append(char)
    else:
        selected = []
        for item in cases:
            c = str(item).strip().lower().replace("case_", "")
            if c in valid_keys and c not in selected:
                selected.append(c)

    if not selected:
        raise ValueError(
            f"Invalid cases specification '{cases}'. Must contain at least one of 'a', 'b', 'c', 'd', 'e'."
        )

    return selected


class SchemaContextBuilder:
    """Constructs focused schema subsets and formats prompts for the 5 context variants."""

    def __init__(self, top_k_vector: int = 20, top_k_lsh: int = 10, top_k_rules: int = 3, top_k_readme: int = 2) -> None:
        """Initialize SchemaContextBuilder with retrieval parameters."""
        self.top_k_vector = top_k_vector
        self.top_k_lsh = top_k_lsh
        self.top_k_rules = top_k_rules
        self.top_k_readme = top_k_readme

    def get_focused_schema(
        self,
        question: str,
        lsh_indexer: Any,
        vector_indexer: Any,
        top_k_vector: int | None = None,
        top_k_lsh: int | None = None,
    ) -> set[str]:
        """Extract focused schema column subset via FAISS semantic search and LSH literal lookup.

        Args:
            question: Natural language question text.
            lsh_indexer: VietnameseLSHIndexer instance.
            vector_indexer: SemanticVectorIndexer instance.
            top_k_vector: Number of semantic columns to retrieve (default: self.top_k_vector).
            top_k_lsh: Number of LSH matches per query token (default: self.top_k_lsh).

        Returns:
            Set of 'table.column' keys identified as candidates.
        """
        k_vec = top_k_vector or self.top_k_vector
        k_lsh = top_k_lsh or self.top_k_lsh
        focused: set[str] = set()

        # 1. Semantic Column Retrieval via FAISS
        if vector_indexer is not None:
            col_results = vector_indexer.search_columns(question, top_k=k_vec)
            for res in col_results:
                col_key = res.get("key")
                if col_key and "." in col_key:
                    focused.add(col_key)

        # 2. Literal Matching via LSH
        if lsh_indexer is not None:
            # Query the whole question
            q_clean = question.strip()
            lsh_matches = lsh_indexer.query_literal_with_scores(q_clean, top_k=k_lsh)
            for col_key, _, score in lsh_matches:
                if score >= 0.4:
                    focused.add(col_key)

            # Extract quoted phrases ('...' or "...")
            quoted_phrases = re.findall(r"['\"]([^'\"]+)['\"]", question)
            for phrase in quoted_phrases:
                p_matches = lsh_indexer.query_literal_with_scores(phrase.strip(), top_k=k_lsh)
                for col_key, _, score in p_matches:
                    if score >= 0.4:
                        focused.add(col_key)

            # Extract n-grams (2 to 4 consecutive words) for entity mentions
            words = [w for w in re.split(r"\s+", question) if len(w) > 1]
            n = len(words)
            for size in (3, 2):
                for i in range(max(0, n - size + 1)):
                    ngram = " ".join(words[i : i + size])
                    ng_matches = lsh_indexer.query_literal_with_scores(ngram, top_k=5)
                    for col_key, _, score in ng_matches:
                        if score >= 0.5:
                            focused.add(col_key)

        return focused

    def get_all_tables_and_columns(
        self,
        processed_metadata: dict[str, Any],
    ) -> set[str]:
        """Extract all 'table.column' keys available in the database metadata."""
        cols: set[str] = set()
        target_dict = processed_metadata.get("metadata_profile", processed_metadata)
        if "tables" in target_dict and isinstance(target_dict["tables"], dict):
            for tbl, t_info in target_dict["tables"].items():
                for col in t_info.get("columns", {}):
                    cols.add(f"{tbl}.{col}")
        else:
            for k in target_dict:
                if "." in k:
                    cols.add(k)
        return cols

    def format_prompt_schema(
        self,
        schema_cols: set[str],
        profile_type: str,
        processed_metadata: dict[str, Any],
    ) -> str:
        """Format selected schema columns into a clean XML-tagged schema representation.

        Args:
            schema_cols: Set of 'table.column' keys to include.
            profile_type: 'minimal_profile', 'maximal_profile', or 'full_profile'.
            processed_metadata: Enriched database metadata dictionary.

        Returns:
            Formatted schema XML string.
        """
        target_dict = processed_metadata.get("metadata_profile", processed_metadata)

        # Group columns by table
        table_groups: dict[str, list[dict[str, Any]]] = {}
        for col_key in sorted(schema_cols):
            parts = col_key.split(".", 1)
            tbl = parts[0]
            col = parts[1] if len(parts) > 1 else col_key

            col_meta = target_dict.get(col_key, {})
            data_format = col_meta.get("data_format", "STRING")

            if profile_type == "minimal_profile":
                desc = col_meta.get("minimal_profile", "")
            else:
                # maximal_profile or full_profile
                desc = col_meta.get("maximal_profile", "")

            payload: dict[str, Any] = {
                "col_name": col,
                "data_format": data_format,
                "description": desc,
            }
            payload.update(build_column_attributes(col_meta))
            table_groups.setdefault(tbl, []).append(payload)

        lines = ["<database_schema>"]
        for tbl_name, cols in sorted(table_groups.items()):
            lines.extend(format_schema_table_elements(tbl_name, cols))
        lines.append("</database_schema>")

        return "\n".join(lines)

    def build_variant_prompt(
        self,
        question: str,
        schema_cols: set[str],
        profile_type: str,
        processed_metadata: dict[str, Any],
        sme_context: str = "",
        rulebook_context: str = "",
        reprompt_warning: str = "",
        as_of_date: str | None = None,
        intent: QueryIntent | None = None,
    ) -> str:
        """Build the complete generation prompt for a single context variant.

        The prompt is always anchored to the question's ``as_of_date`` and carries the
        deterministically resolved temporal period, so a trial query can never fall back to
        a period copied from the few-shot format examples.
        """
        schema_xml = self.format_prompt_schema(schema_cols, profile_type, processed_metadata)
        resolved_intent = intent if intent is not None else detect_query_intent(question, as_of_date)
        temporal_xml = format_temporal_anchor_xml(question, as_of_date)
        intent_xml = format_intent_hint_xml(resolved_intent)

        sections = [
            "<system>",
            "You are an expert SQL engineer. Your task is to write a single trial SQLite query",
            "to answer the user question based strictly on the provided schema and business context.",
            "Rules:",
            "1. Output ONLY the raw executable SQLite SQL statement.",
            "2. Wrap table and column names with double quotes if they contain Vietnamese accents, spaces, or special characters.",
            "3. Use appropriate table aliases (e.g. FROM Table AS T1).",
            "</system>",
            "",
            FEW_SHOT_SQL_EXAMPLES,
            "",
            "<user_question>",
            question.strip(),
            "</user_question>",
        ]

        if temporal_xml:
            sections.extend(["", temporal_xml])

        if intent_xml:
            sections.extend(["", intent_xml])

        if sme_context.strip():
            sections.extend([
                "",
                "<business_readme>",
                sme_context.strip(),
                "</business_readme>",
            ])

        if rulebook_context.strip():
            sections.extend([
                "",
                "<business_rules>",
                rulebook_context.strip(),
                "</business_rules>",
            ])

        sections.extend([
            "",
            schema_xml,
        ])

        if reprompt_warning.strip():
            sections.extend([
                "",
                "<reprompt_correction_guidance>",
                reprompt_warning.strip(),
                "</reprompt_correction_guidance>",
            ])

        sections.extend([
            "",
            "<output>",
            "-- SQLite query:",
        ])

        return "\n".join(sections)

    def build_case_e_prompt(
        self,
        question: str,
        focused_schema: set[str],
        processed_metadata: dict[str, Any],
        vector_indexer: Any,
        as_of_date: str | None = None,
        intent: QueryIntent | None = None,
    ) -> tuple[str, str, str]:
        """Construct Case e with Prompt-Level SME and Rulebook context.

        Returns:
            Tuple of (formatted_prompt, sme_context, rulebook_context).
        """
        sme_text = ""
        rule_text = ""

        if vector_indexer is not None:
            # 1. Retrieve top-2 README context chunks
            readme_matches = vector_indexer.search_similar_schema(
                question, top_k=self.top_k_readme, doc_type="readme"
            )
            if readme_matches:
                sme_text = "\n\n".join(
                    [m.get("content", "") for m in readme_matches if m.get("content")]
                )

            # 2. Retrieve top-3 Rulebook clauses
            rule_matches = vector_indexer.search_rules(question, top_k=self.top_k_rules)
            if rule_matches:
                rule_text = "\n\n".join(
                    [m.get("content", "") for m in rule_matches if m.get("content")]
                )

        prompt = self.build_variant_prompt(
            question=question,
            schema_cols=focused_schema,
            profile_type="maximal_profile",
            processed_metadata=processed_metadata,
            sme_context=sme_text,
            rulebook_context=rule_text,
            as_of_date=as_of_date,
            intent=intent,
        )

        return prompt, sme_text, rule_text

    def build_5_variants(
        self,
        question: str,
        processed_metadata: dict[str, Any],
        lsh_indexer: Any,
        vector_indexer: Any,
        cases: str | list[str] | set[str] | None = None,
        as_of_date: str | None = None,
        intent: QueryIntent | None = None,
    ) -> list[SchemaVariant]:
        """Generate context variants for Task-Aligned Schema Linking.

        Variants:
          - case_a: Focused + Minimal Profile
          - case_b: Focused + Maximal Profile
          - case_c: Full + Minimal Profile
          - case_d: Full + Maximal Profile
          - case_e: Focused + Full Profile (Maximal Profile + Prompt-Level SME & Rulebook)

        Args:
            question: Natural language question text.
            processed_metadata: Enriched database metadata dictionary.
            lsh_indexer: VietnameseLSHIndexer instance.
            vector_indexer: SemanticVectorIndexer instance.
            cases: Specific cases to run, e.g. "abcde", "abce", "abe", or None (defaults to all).

        Returns:
            List of SchemaVariant instances matching the requested cases.
        """
        case_keys = parse_case_spec(cases)
        resolved_intent = intent if intent is not None else detect_query_intent(question, as_of_date)

        # Step 1: Compute focused schema if needed by cases a, b, or e
        need_focused = any(k in case_keys for k in ("a", "b", "e"))
        need_full = any(k in case_keys for k in ("c", "d"))

        focused_schema: set[str] = set()
        if need_focused:
            focused_schema = self.get_focused_schema(
                question=question,
                lsh_indexer=lsh_indexer,
                vector_indexer=vector_indexer,
            )

        full_schema: set[str] = set()
        if need_full or (need_focused and not focused_schema):
            full_schema = self.get_all_tables_and_columns(processed_metadata)

        # Fallback: If focused schema is empty, fall back to full schema
        if need_focused and not focused_schema:
            focused_schema = set(full_schema)

        variants: list[SchemaVariant] = []
        for key in case_keys:
            if key == "a":
                prompt_a = self.build_variant_prompt(
                    question=question,
                    schema_cols=focused_schema,
                    profile_type="minimal_profile",
                    processed_metadata=processed_metadata,
                    as_of_date=as_of_date,
                    intent=resolved_intent,
                )
                variants.append(
                    SchemaVariant(
                        case_name="case_a",
                        schema_cols=set(focused_schema),
                        profile_type="minimal_profile",
                        formatted_prompt=prompt_a,
                    )
                )
            elif key == "b":
                prompt_b = self.build_variant_prompt(
                    question=question,
                    schema_cols=focused_schema,
                    profile_type="maximal_profile",
                    processed_metadata=processed_metadata,
                    as_of_date=as_of_date,
                    intent=resolved_intent,
                )
                variants.append(
                    SchemaVariant(
                        case_name="case_b",
                        schema_cols=set(focused_schema),
                        profile_type="maximal_profile",
                        formatted_prompt=prompt_b,
                    )
                )
            elif key == "c":
                prompt_c = self.build_variant_prompt(
                    question=question,
                    schema_cols=full_schema,
                    profile_type="minimal_profile",
                    processed_metadata=processed_metadata,
                    as_of_date=as_of_date,
                    intent=resolved_intent,
                )
                variants.append(
                    SchemaVariant(
                        case_name="case_c",
                        schema_cols=set(full_schema),
                        profile_type="minimal_profile",
                        formatted_prompt=prompt_c,
                    )
                )
            elif key == "d":
                prompt_d = self.build_variant_prompt(
                    question=question,
                    schema_cols=full_schema,
                    profile_type="maximal_profile",
                    processed_metadata=processed_metadata,
                    as_of_date=as_of_date,
                    intent=resolved_intent,
                )
                variants.append(
                    SchemaVariant(
                        case_name="case_d",
                        schema_cols=set(full_schema),
                        profile_type="maximal_profile",
                        formatted_prompt=prompt_d,
                    )
                )
            elif key == "e":
                prompt_e, sme_ctx, rule_ctx = self.build_case_e_prompt(
                    question=question,
                    focused_schema=focused_schema,
                    processed_metadata=processed_metadata,
                    vector_indexer=vector_indexer,
                    as_of_date=as_of_date,
                    intent=resolved_intent,
                )
                variants.append(
                    SchemaVariant(
                        case_name="case_e",
                        schema_cols=set(focused_schema),
                        profile_type="full_profile",
                        formatted_prompt=prompt_e,
                        sme_context=sme_ctx,
                        rulebook_context=rule_ctx,
                    )
                )

        return variants
