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


class SchemaContextBuilder:
    """Constructs focused schema subsets and formats prompts for the 5 context variants."""

    def __init__(self, top_k_vector: int = 10, top_k_lsh: int = 5, top_k_rules: int = 3, top_k_readme: int = 2) -> None:
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
                    ng_matches = lsh_indexer.query_literal_with_scores(ngram, top_k=2)
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
                desc = col_meta.get("minimal_profile") or "Standard tabular column"
            else:
                # maximal_profile or full_profile
                desc = (
                    col_meta.get("maximal_profile")
                    or col_meta.get("minimal_profile")
                    or "Standard tabular column"
                )

            table_groups.setdefault(tbl, []).append({
                "col_name": col,
                "data_format": data_format,
                "description": desc,
            })

        lines = ["<database_schema>"]
        for tbl_name, cols in sorted(table_groups.items()):
            lines.append(f'  <table name="{tbl_name}">')
            for c in cols:
                lines.append(
                    f'    <column name="{c["col_name"]}" type="{c["data_format"]}">'
                )
                lines.append(f'      {c["description"]}')
                lines.append("    </column>")
            lines.append("  </table>")
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
    ) -> str:
        """Build the complete generation prompt for a single context variant."""
        schema_xml = self.format_prompt_schema(schema_cols, profile_type, processed_metadata)

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
            "<user_question>",
            question.strip(),
            "</user_question>",
        ]

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
        )

        return prompt, sme_text, rule_text

    def build_5_variants(
        self,
        question: str,
        processed_metadata: dict[str, Any],
        lsh_indexer: Any,
        vector_indexer: Any,
    ) -> list[SchemaVariant]:
        """Generate all 5 context variants for Task-Aligned Schema Linking.

        Variants:
          - case_a: Focused + Minimal Profile
          - case_b: Focused + Maximal Profile
          - case_c: Full + Minimal Profile
          - case_d: Full + Maximal Profile
          - case_e: Focused + Full Profile (Maximal Profile + Prompt-Level SME & Rulebook)

        Returns:
            List of 5 SchemaVariant instances.
        """
        # Step 1: Compute focused schema and full schema
        focused_schema = self.get_focused_schema(
            question=question,
            lsh_indexer=lsh_indexer,
            vector_indexer=vector_indexer,
        )
        full_schema = self.get_all_tables_and_columns(processed_metadata)

        # Fallback: If focused schema is empty, fall back to full schema
        if not focused_schema:
            focused_schema = set(full_schema)

        # Case a: Focused + Minimal
        prompt_a = self.build_variant_prompt(
            question=question,
            schema_cols=focused_schema,
            profile_type="minimal_profile",
            processed_metadata=processed_metadata,
        )
        case_a = SchemaVariant(
            case_name="case_a",
            schema_cols=set(focused_schema),
            profile_type="minimal_profile",
            formatted_prompt=prompt_a,
        )

        # Case b: Focused + Maximal
        prompt_b = self.build_variant_prompt(
            question=question,
            schema_cols=focused_schema,
            profile_type="maximal_profile",
            processed_metadata=processed_metadata,
        )
        case_b = SchemaVariant(
            case_name="case_b",
            schema_cols=set(focused_schema),
            profile_type="maximal_profile",
            formatted_prompt=prompt_b,
        )

        # Case c: Full + Minimal
        prompt_c = self.build_variant_prompt(
            question=question,
            schema_cols=full_schema,
            profile_type="minimal_profile",
            processed_metadata=processed_metadata,
        )
        case_c = SchemaVariant(
            case_name="case_c",
            schema_cols=set(full_schema),
            profile_type="minimal_profile",
            formatted_prompt=prompt_c,
        )

        # Case d: Full + Maximal
        prompt_d = self.build_variant_prompt(
            question=question,
            schema_cols=full_schema,
            profile_type="maximal_profile",
            processed_metadata=processed_metadata,
        )
        case_d = SchemaVariant(
            case_name="case_d",
            schema_cols=set(full_schema),
            profile_type="maximal_profile",
            formatted_prompt=prompt_d,
        )

        # Case e: Focused + Full (Maximal Profile + Prompt-Level SME & Rulebook)
        prompt_e, sme_ctx, rule_ctx = self.build_case_e_prompt(
            question=question,
            focused_schema=focused_schema,
            processed_metadata=processed_metadata,
            vector_indexer=vector_indexer,
        )
        case_e = SchemaVariant(
            case_name="case_e",
            schema_cols=set(focused_schema),
            profile_type="full_profile",
            formatted_prompt=prompt_e,
            sme_context=sme_ctx,
            rulebook_context=rule_ctx,
        )

        return [case_a, case_b, case_c, case_d, case_e]
