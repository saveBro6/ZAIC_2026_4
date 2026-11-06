"""Core Task-Aligned Schema Linking implementation based on 'Automatic Metadata Extraction for Text-to-SQL'.

Iterates through 5 context variants (Cases a to e), generates trial SQLite queries,
harvests referenced fields and literals using SQL AST parsing, performs LSH literal mismatch
detection with iterative re-prompting, and fusions all harvested columns while preserving PK/FK
and undocumented join keys.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from zaic_2026_4.schema_linking.ast_parser import SQLASTExtractor
from zaic_2026_4.schema_linking.context_builder import (
    SchemaContextBuilder,
)

logger = logging.getLogger(__name__)


@dataclass
class VariantExecutionTrace:
    """Trace details for a single context variant execution."""

    case_name: str
    profile_type: str
    initial_sql: str
    final_sql: str
    retry_count: int
    fields_harvested: set[str]
    literals_extracted: set[str]
    mismatches_detected: list[tuple[str, list[str]]] = field(default_factory=list)


@dataclass
class SchemaLinkingResult:
    """Consolidated result of Task-Aligned Schema Linking."""

    final_linked_schema: set[str]
    variant_traces: list[VariantExecutionTrace] = field(default_factory=list)
    total_retries: int = 0
    all_harvested_fields: set[str] = field(default_factory=set)
    preserved_keys: set[str] = field(default_factory=set)


class TaskAlignedSchemaLinker:
    """Orchestrates Task-Aligned Schema Linking across 5 context variants."""

    def __init__(
        self,
        llm_engine: Any,
        ast_parser: SQLASTExtractor | None = None,
        context_builder: SchemaContextBuilder | None = None,
        max_retries: int = 2,
    ) -> None:
        """Initialize TaskAlignedSchemaLinker.

        Args:
            llm_engine: LLM generation engine supporting .generate(prompts).
            ast_parser: SQLASTExtractor instance (defaults to new instance).
            context_builder: SchemaContextBuilder instance (defaults to new instance).
            max_retries: Maximum number of re-prompting attempts on literal mismatch.
        """
        self.llm_engine = llm_engine
        self.ast_parser = ast_parser or SQLASTExtractor()
        self.context_builder = context_builder or SchemaContextBuilder()
        self.max_retries = max_retries

    def preserve_linking_keys(
        self,
        linked_fields: set[str],
        raw_profile: dict[str, Any] | None = None,
        processed_metadata: dict[str, Any] | None = None,
    ) -> set[str]:
        """Preserve Primary Keys, Foreign Keys, and undocumented join keys for all involved tables.

        Args:
            linked_fields: Harvested set of 'table.column' strings.
            raw_profile: Optional raw database profile containing 'undocumented_joins' and table stats.
            processed_metadata: Optional enriched metadata dictionary.

        Returns:
            Extended set of column keys with essential join keys preserved.
        """
        final_schema = set(linked_fields)
        if not final_schema:
            return final_schema

        # Identify all tables currently referenced in linked fields
        referenced_tables: set[str] = set()
        for col_key in linked_fields:
            if "." in col_key:
                referenced_tables.add(col_key.split(".", 1)[0])

        if not referenced_tables:
            return final_schema

        # 1. Preserve Undocumented Joins detected by MinHash profiler (resemblance >= 0.7)
        if raw_profile and "undocumented_joins" in raw_profile:
            for join_item in raw_profile.get("undocumented_joins", []):
                col1 = join_item.get("col1")
                col2 = join_item.get("col2")
                if col1 and col2 and "." in col1 and "." in col2:
                    t1 = col1.split(".", 1)[0]
                    t2 = col2.split(".", 1)[0]
                    if t1 in referenced_tables and t2 in referenced_tables:
                        final_schema.add(col1)
                        final_schema.add(col2)

        # 2. Preserve likely Primary Keys and IDs for referenced tables
        meta_dict = (
            processed_metadata.get("metadata_profile", processed_metadata)
            if processed_metadata
            else {}
        )
        for tbl in referenced_tables:
            # Check known columns in meta_dict or raw_profile
            candidate_cols: list[str] = []
            if raw_profile and "tables" in raw_profile and tbl in raw_profile["tables"]:
                candidate_cols = list(raw_profile["tables"][tbl].get("columns", {}).keys())
            elif meta_dict:
                candidate_cols = [
                    k.split(".", 1)[1] for k in meta_dict if k.startswith(f"{tbl}.")
                ]

            for c in candidate_cols:
                c_lower = c.lower()
                # Common primary key conventions
                if c_lower in ("id", f"{tbl.lower()}_id", "code", f"ma_{tbl.lower()}") or (
                    c_lower.startswith("ma_") and ("bill" in c_lower or "don" in c_lower or "hd" in c_lower)
                ):
                    final_schema.add(f"{tbl}.{c}")

        return final_schema

    def link_schema(
        self,
        question: str,
        processed_metadata: dict[str, Any],
        lsh_indexer: Any,
        vector_indexer: Any | None = None,
        raw_profile: dict[str, Any] | None = None,
        max_retries: int | None = None,
        batch_generate: bool = True,
    ) -> SchemaLinkingResult:
        """Execute Task-Aligned Schema Linking across 5 context variants.

        Args:
            question: User natural language query in Vietnamese or English.
            processed_metadata: Enriched database metadata mapping 'table.col' to column profiles.
            lsh_indexer: VietnameseLSHIndexer instance for literal lookups.
            vector_indexer: SemanticVectorIndexer instance for semantic search and SME retrieval.
            raw_profile: Optional raw database profile containing undocumented_joins.
            max_retries: Maximum re-prompting retries on literal mismatch (default: self.max_retries).
            batch_generate: Whether to execute initial prompts using batched LLM generation.

        Returns:
            SchemaLinkingResult with final linked schema, execution traces, and diagnostics.
        """
        retries_limit = max_retries if max_retries is not None else self.max_retries
        all_columns = self.context_builder.get_all_tables_and_columns(processed_metadata)

        # Step 1: Construct 5 context variants (Cases a to e)
        variants = self.context_builder.build_5_variants(
            question=question,
            processed_metadata=processed_metadata,
            lsh_indexer=lsh_indexer,
            vector_indexer=vector_indexer,
        )

        # Step 2: Generate initial trial SQL queries
        prompts = [v.formatted_prompt for v in variants]
        if batch_generate and hasattr(self.llm_engine, "generate"):
            try:
                initial_sqls = self.llm_engine.generate(
                    prompts,
                    max_tokens=256,
                    stop=["</output>", "\n\n\n", "<|im_end|>", "<|endoftext|>"],
                )
                if isinstance(initial_sqls, str):
                    initial_sqls = [initial_sqls]
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.warning("Batch generation failed (%s), falling back to sequential: %s", type(e).__name__, e)
                initial_sqls = [
                    self.llm_engine.generate(
                        p, max_tokens=256, stop=["</output>", "\n\n\n", "<|im_end|>"]
                    )
                    for p in prompts
                ]
        else:
            initial_sqls = [
                self.llm_engine.generate(
                    p, max_tokens=256, stop=["</output>", "\n\n\n", "<|im_end|>"]
                )
                for p in prompts
            ]

        linked_fields: set[str] = set()
        traces: list[VariantExecutionTrace] = []
        total_retries = 0

        # Step 3: Process each variant with AST extraction & LSH re-prompting loop
        for idx, variant in enumerate(variants):
            raw_sql = initial_sqls[idx] if idx < len(initial_sqls) else ""
            test_sql = self.ast_parser.clean_sql(str(raw_sql))

            initial_sql_saved = test_sql
            retry_count = 0
            current_schema = set(variant.schema_cols)
            variant_mismatches: list[tuple[str, list[str]]] = []

            while retry_count <= retries_limit:
                # b. AST parsing to extract fields and literals
                fields_q, lits_q = self.ast_parser.parse_fields_and_literals(
                    test_sql, known_columns=all_columns
                )

                lit_fields_q: set[str] = set()
                missing_lits: list[tuple[str, list[str]]] = []

                # d. Query LSH Index for each literal in lits_q
                if lsh_indexer is not None:
                    for literal in lits_q:
                        # Skip short trivial numbers or single chars
                        if len(literal.strip()) < 2:
                            continue

                        # Look up columns matching literal in LSH index
                        fields_l = lsh_indexer.lookup_value(literal, threshold=0.4)

                        # If literal exists in database, but NONE of its columns are in fields_q
                        if fields_l and not (fields_l & fields_q):
                            lit_fields_q.update(fields_l)
                            missing_lits.append((literal, sorted(fields_l)))

                # e. If literal mismatch detected and retries available
                if missing_lits and retry_count < retries_limit:
                    variant_mismatches.extend(missing_lits)
                    current_schema.update(lit_fields_q)

                    reprompt_guidance = (
                        f"Previously generated SQL: {test_sql}\n"
                        f"Literal Mismatch Warning: The literal(s) {missing_lits} were matched "
                        f"to column(s) {sorted(lit_fields_q)} in the database.\n"
                        f"Please revise the SQL query to use these correct columns for filtering."
                    )

                    revised_prompt = self.context_builder.build_variant_prompt(
                        question=question,
                        schema_cols=current_schema,
                        profile_type=variant.profile_type,
                        processed_metadata=processed_metadata,
                        sme_context=variant.sme_context,
                        rulebook_context=variant.rulebook_context,
                        reprompt_warning=reprompt_guidance,
                    )

                    revised_res = self.llm_engine.generate(
                        revised_prompt,
                        max_tokens=256,
                        stop=["</output>", "\n\n\n", "<|im_end|>", "<|endoftext|>"],
                    )
                    if isinstance(revised_res, list):
                        revised_res = revised_res[0]

                    test_sql = self.ast_parser.clean_sql(str(revised_res))

                    retry_count += 1
                    total_retries += 1
                else:
                    # No mismatches or max retries reached
                    break

            # f. Union harvested fields from this variant
            linked_fields.update(fields_q)

            traces.append(
                VariantExecutionTrace(
                    case_name=variant.case_name,
                    profile_type=variant.profile_type,
                    initial_sql=initial_sql_saved,
                    final_sql=test_sql,
                    retry_count=retry_count,
                    fields_harvested=fields_q,
                    literals_extracted=lits_q,
                    mismatches_detected=variant_mismatches,
                )
            )

        # Step 4: Key preservation (Primary Keys, Foreign Keys, Undocumented Joins)
        final_linked_schema = self.preserve_linking_keys(
            linked_fields=linked_fields,
            raw_profile=raw_profile,
            processed_metadata=processed_metadata,
        )

        preserved = final_linked_schema - linked_fields

        return SchemaLinkingResult(
            final_linked_schema=final_linked_schema,
            variant_traces=traces,
            total_retries=total_retries,
            all_harvested_fields=linked_fields,
            preserved_keys=preserved,
        )


def task_aligned_schema_linking(
    question: str,
    processed_metadata: dict[str, Any],
    lsh_indexer: Any,
    vector_indexer: Any | None = None,
    llm_engine: Any | None = None,
    raw_profile: dict[str, Any] | None = None,
    max_retries: int = 2,
) -> set[str]:
    """Functional interface for Task-Aligned Schema Linking matching the original paper's pseudo-code.

    Input:
        - question: Natural language question from user.
        - processed_metadata: Database profile containing column metadata (Minimal, Maximal, Full).
        - lsh_indexer: VietnameseLSHIndexer indexing column values (N=10,000 samples/col).
        - vector_indexer: SemanticVectorIndexer for FAISS column & rule retrieval.
        - llm_engine: LLM inference engine supporting text generation.
        - raw_profile: Raw database statistics and undocumented join paths.
        - max_retries: Maximum number of re-prompting retries on literal mismatch.

    Output:
        - final_linked_schema: Set of selected columns formatted as 'TableName.ColumnName'.
    """
    if llm_engine is None:
        raise ValueError("llm_engine must be provided to task_aligned_schema_linking")

    linker = TaskAlignedSchemaLinker(
        llm_engine=llm_engine,
        max_retries=max_retries,
    )
    result = linker.link_schema(
        question=question,
        processed_metadata=processed_metadata,
        lsh_indexer=lsh_indexer,
        vector_indexer=vector_indexer,
        raw_profile=raw_profile,
        max_retries=max_retries,
    )
    return result.final_linked_schema
