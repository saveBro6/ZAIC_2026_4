"""Core Task-Aligned Schema Linking implementation based on 'Automatic Metadata Extraction for Text-to-SQL'.

Iterates through 5 context variants (Cases a to e), generates trial SQLite queries,
harvests referenced fields and literals using SQL AST parsing, performs LSH literal mismatch
detection with iterative re-prompting, and fusions all harvested columns while preserving PK/FK
and undocumented join keys.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zaic_2026_4.common.trace_logger import (
    LLMCallRecord,
    append_llm_calls,
    extract_llm_engine_call_traces,
    make_llm_call_record,
)
from zaic_2026_4.schema_linking.ast_parser import SQLASTExtractor
from zaic_2026_4.schema_linking.context_builder import (
    SchemaContextBuilder,
)
from zaic_2026_4.schema_linking.post_processor import (
    SchemaPostProcessor,
    normalize_identifier,
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
    formatted_prompt: str = ""
    raw_initial_output: str = ""
    sme_context_used: str = ""
    rulebook_context_used: str = ""
    initial_llm_metrics: dict[str, Any] = field(default_factory=dict)
    repair_turns: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SchemaLinkingResult:
    """Consolidated result of Task-Aligned Schema Linking."""

    final_linked_schema: set[str]
    canonical_schema: set[str] = field(default_factory=set)
    variant_traces: list[VariantExecutionTrace] = field(default_factory=list)
    total_retries: int = 0
    all_harvested_fields: set[str] = field(default_factory=set)
    preserved_keys: set[str] = field(default_factory=set)
    grounded_literals: list[dict[str, Any]] = field(default_factory=list)
    llm_usage_summary: dict[str, Any] = field(default_factory=dict)

    def format_for_sql_generation(self, processed_metadata: dict[str, Any]) -> str:
        """Format quoted schema into clean XML context for downstream Phase 04 SQL Generation."""
        return SchemaPostProcessor.format_for_sql_generation(self.final_linked_schema, processed_metadata)

    def to_dict(
        self,
        question: str | None = None,
        question_id: str | None = None,
        include_detailed_traces: bool = True,
    ) -> dict[str, Any]:
        """Convert linking result into serializable JSON dictionary."""
        variant_items: list[dict[str, Any]] = []
        for t in self.variant_traces:
            entry: dict[str, Any] = {
                "case_name": t.case_name,
                "profile_type": t.profile_type,
                "retry_count": t.retry_count,
                "fields": sorted(t.fields_harvested),
                "literals": sorted(t.literals_extracted),
                "mismatches": t.mismatches_detected,
                "final_sql": t.final_sql,
            }
            if include_detailed_traces:
                entry.update({
                    "initial_sql": t.initial_sql,
                    "formatted_prompt": t.formatted_prompt,
                    "raw_initial_output": t.raw_initial_output,
                    "sme_context_used": t.sme_context_used,
                    "rulebook_context_used": t.rulebook_context_used,
                    "initial_llm_metrics": t.initial_llm_metrics,
                    "repair_turns": t.repair_turns,
                })
            variant_items.append(entry)

        data: dict[str, Any] = {
            "final_linked_schema": sorted(self.final_linked_schema),
            "canonical_schema": sorted(self.canonical_schema),
            "preserved_keys": sorted(self.preserved_keys),
            "grounded_literals": self.grounded_literals,
            "total_retries": self.total_retries,
            "variant_traces": variant_items,
        }
        if include_detailed_traces:
            data["all_harvested_fields"] = sorted(self.all_harvested_fields)
            data["llm_usage_summary"] = self.llm_usage_summary
        if question is not None:
            data["question"] = question
        if question_id is not None:
            data["question_id"] = question_id
        return data


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

    @staticmethod
    def is_candidate_key_column(col_name: str, table_name: str = "") -> bool:
        """Check if column is a primary key, foreign key, or entity identification code.

        Handles both English conventions (id, order_id, code, CDSCode) and Vietnamese
        business conventions (Mã hóa đơn, Mã KH, Mã SP, Số HĐ, Số phiếu, etc.).
        """
        c_clean = col_name.strip().lstrip("\ufeff")
        c_lower = c_clean.lower()
        c_norm = normalize_identifier(c_clean)
        tbl_norm = normalize_identifier(table_name) if table_name else ""
        tbl_base = tbl_norm.rstrip("s") if tbl_norm else ""

        # 1. Vietnamese explicit code prefix ('Mã...', 'Ma...', 'ma_...')
        # Note: 'mã ' or 'ma_' distinguishes entity codes from words like 'mẫu...'
        if c_lower.startswith(("mã ", "mã_", "ma_", "ma ", "mã-", "ma-")):
            return True

        # 2. Direct key keywords and table-specific id/code patterns
        if c_norm in (
            "id",
            "code",
            f"{tbl_norm}id",
            f"id{tbl_norm}",
            f"ma{tbl_norm}",
            f"code{tbl_norm}",
            f"{tbl_base}id",
            f"{tbl_base}code",
        ):
            return True

        # 3. Delimited or CamelCase id/code prefix or suffix (e.g. order_id, CDSCode, CustomerID, ID_SP, Product_Code)
        if (
            c_lower.endswith(("_id", "_code", "-id", "-code", " id", " code", " mã"))
            or c_lower.startswith(("id_", "code_", "id-", "code-", "id ", "code "))
            or c_clean.endswith(("ID", "Id", "Code", "CODE"))
            or c_clean.startswith(("ID", "Id", "Code", "CODE"))
        ):
            return True

        # 4. Vietnamese invoice, bill, receipt serial / document numbers
        # Explicitly excludes measures / metrics like 'soluong' or 'sodienthoai'
        return c_norm in ("sohd", "sohoadon", "soct", "sophieu", "sokh", "sochungtu")

    @staticmethod
    def extract_grounded_literals_from_question(
        question: str,
        lsh_indexer: Any,
        candidate_columns: set[str] | None = None,
        threshold: float = 0.55,
    ) -> list[dict[str, Any]]:
        """Extract entity mentions from question and ground them to canonical values in DB."""
        if not lsh_indexer or not question or not question.strip():
            return []

        clean_q = question.strip()
        mentions: set[str] = set()

        # 1. Quoted phrases ('...' or "...")
        for phrase in re.findall(r"['\"]([^'\"]+)['\"]", clean_q):
            p_strip = phrase.strip()
            if len(p_strip) >= 2 and not p_strip.isnumeric():
                mentions.add(p_strip)

        # 2. Extract n-grams (4, 3, 2 words)
        words = [w for w in re.split(r"\s+", clean_q) if len(w) > 1]
        n = len(words)
        for size in (4, 3, 2):
            for i in range(max(0, n - size + 1)):
                ngram = " ".join(words[i : i + size]).strip()
                if len(ngram) >= 2 and not ngram.isnumeric():
                    mentions.add(ngram)

        # 3. Query LSH for each mention
        results: list[dict[str, Any]] = []
        for m in mentions:
            scored = lsh_indexer.query_literal_with_scores(m, top_k=5)
            for col_key, real_val, score in scored:
                if score < threshold:
                    continue
                # If candidate_columns is given, prioritize columns in schema or allow very high score
                if candidate_columns and col_key not in candidate_columns and score < 0.75:
                    continue
                results.append({
                    "mention": m,
                    "canonical_value": real_val,
                    "column": col_key,
                    "score": round(float(score), 4),
                })

        return results

    @staticmethod
    def deduplicate_grounded_literals(
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Deduplicate grounded literals by (mention, column, canonical_value), preserving highest score."""
        best: dict[tuple[str, str, str], dict[str, Any]] = {}
        for item in candidates:
            key = (
                item["mention"].strip().lower(),
                item["column"].strip(),
                item["canonical_value"].strip(),
            )
            if key not in best or item["score"] > best[key]["score"]:
                best[key] = item

        # Sort by score descending
        return sorted(best.values(), key=lambda x: x["score"], reverse=True)

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
                if self.is_candidate_key_column(c, tbl):
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
        cases: str | list[str] | set[str] | None = None,
        cache_dir: Path | str | None = None,
        question_id: str | None = None,
        trace_enabled: bool = True,
    ) -> SchemaLinkingResult:
        """Execute Task-Aligned Schema Linking across specified context variants.

        Args:
            question: User natural language query in Vietnamese or English.
            processed_metadata: Enriched database metadata mapping 'table.col' to column profiles.
            lsh_indexer: VietnameseLSHIndexer instance for literal lookups.
            vector_indexer: SemanticVectorIndexer instance for semantic search and SME retrieval.
            raw_profile: Optional raw database profile containing undocumented_joins.
            max_retries: Maximum re-prompting retries on literal mismatch (default: self.max_retries).
            batch_generate: Whether to execute initial prompts using batched LLM generation.
            cases: Specific context variants to run, e.g. "abcde", "abce", "abe", or None (defaults to all).
            cache_dir: Optional business cache directory to automatically persist schema_linking_result.json.
            question_id: Optional question ID for logging per-question schema linking results.

        Returns:
            SchemaLinkingResult with final linked schema, execution traces, and diagnostics.
        """
        retries_limit = max_retries if max_retries is not None else self.max_retries
        all_columns = self.context_builder.get_all_tables_and_columns(processed_metadata)

        # Step 1: Construct context variants (Cases a to e as specified)
        variants = self.context_builder.build_5_variants(
            question=question,
            processed_metadata=processed_metadata,
            lsh_indexer=lsh_indexer,
            vector_indexer=vector_indexer,
            cases=cases,
        )

        # Step 2: Generate initial trial SQL queries
        prompts = [v.formatted_prompt for v in variants]
        init_engine_traces: list[dict[str, Any]] = []
        if batch_generate and hasattr(self.llm_engine, "generate"):
            try:
                initial_sqls = self.llm_engine.generate(
                    prompts,
                    max_tokens=256,
                    stop=["</output>", "\n\n\n", "<|im_end|>", "<|endoftext|>"],
                )
                if isinstance(initial_sqls, str):
                    initial_sqls = [initial_sqls]
                batch_traces = extract_llm_engine_call_traces(self.llm_engine)
                init_engine_traces = [
                    batch_traces[i] if i < len(batch_traces) else {}
                    for i in range(len(prompts))
                ]
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.warning("Batch generation failed (%s), falling back to sequential: %s", type(e).__name__, e)
                initial_sqls = []
                init_engine_traces = []
                for p in prompts:
                    out = self.llm_engine.generate(
                        p, max_tokens=256, stop=["</output>", "\n\n\n", "<|im_end|>"]
                    )
                    initial_sqls.append(out[0] if isinstance(out, list) else out)
                    seq_traces = extract_llm_engine_call_traces(self.llm_engine)
                    init_engine_traces.append(seq_traces[0] if seq_traces else {})
        else:
            initial_sqls = []
            for p in prompts:
                out = self.llm_engine.generate(
                    p, max_tokens=256, stop=["</output>", "\n\n\n", "<|im_end|>"]
                )
                initial_sqls.append(out[0] if isinstance(out, list) else out)
                seq_traces = extract_llm_engine_call_traces(self.llm_engine)
                init_engine_traces.append(seq_traces[0] if seq_traces else {})

        linked_fields: set[str] = set()
        traces: list[VariantExecutionTrace] = []
        total_retries = 0
        grounded_candidates: list[dict[str, Any]] = []
        llm_call_records: list[LLMCallRecord] = []

        # Step 3: Process each variant with AST extraction & LSH re-prompting loop
        for idx, variant in enumerate(variants):
            raw_sql = str(initial_sqls[idx]) if idx < len(initial_sqls) else ""
            init_eng_trace = init_engine_traces[idx] if idx < len(init_engine_traces) else {}
            if trace_enabled:
                llm_call_records.append(
                    make_llm_call_record(
                        stage="schema_linking_init",
                        prompt=variant.formatted_prompt,
                        raw_output=raw_sql,
                        engine_trace=init_eng_trace,
                        metadata={
                            "question_id": question_id,
                            "case_name": variant.case_name,
                            "profile_type": variant.profile_type,
                        },
                    )
                )
            test_sql = self.ast_parser.clean_sql(raw_sql)

            initial_sql_saved = test_sql
            retry_count = 0
            current_schema = set(variant.schema_cols)
            variant_mismatches: list[tuple[str, list[str]]] = []
            repair_turns: list[dict[str, Any]] = []

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
                        scored_matches = lsh_indexer.query_literal_with_scores(literal, top_k=5)
                        fields_l = {col for col, _, sc in scored_matches if sc >= 0.4}

                        for col, real_val, sc in scored_matches:
                            if sc >= 0.4:
                                grounded_candidates.append({
                                    "mention": literal,
                                    "canonical_value": real_val,
                                    "column": col,
                                    "score": round(float(sc), 4),
                                })

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
                    raw_revised = str(revised_res)
                    rev_traces = extract_llm_engine_call_traces(self.llm_engine)
                    rev_eng_trace = rev_traces[0] if rev_traces else {}

                    test_sql = self.ast_parser.clean_sql(raw_revised)

                    retry_count += 1
                    total_retries += 1

                    if trace_enabled:
                        rev_metrics = {
                            "prompt_tokens": rev_eng_trace.get("prompt_tokens"),
                            "completion_tokens": rev_eng_trace.get("completion_tokens"),
                            "finish_reason": rev_eng_trace.get("finish_reason"),
                            "stop_reason": rev_eng_trace.get("stop_reason"),
                            "latency_ms": rev_eng_trace.get("latency_ms"),
                            "sampling_params": rev_eng_trace.get("sampling_params") or {},
                        }
                        repair_turns.append({
                            "retry_idx": retry_count,
                            "mismatches": missing_lits,
                            "added_columns": sorted(lit_fields_q),
                            "reprompt_guidance": reprompt_guidance,
                            "revised_prompt": revised_prompt,
                            "raw_output": raw_revised,
                            "cleaned_sql": test_sql,
                            "llm_metrics": rev_metrics,
                        })
                        llm_call_records.append(
                            make_llm_call_record(
                                stage="schema_linking_repair",
                                prompt=revised_prompt,
                                raw_output=raw_revised,
                                engine_trace=rev_eng_trace,
                                metadata={
                                    "question_id": question_id,
                                    "case_name": variant.case_name,
                                    "profile_type": variant.profile_type,
                                    "retry_idx": retry_count,
                                },
                            )
                        )
                else:
                    # No mismatches or max retries reached
                    break

            # f. Union harvested fields from this variant
            linked_fields.update(fields_q)

            init_metrics = (
                {
                    "prompt_tokens": init_eng_trace.get("prompt_tokens"),
                    "completion_tokens": init_eng_trace.get("completion_tokens"),
                    "finish_reason": init_eng_trace.get("finish_reason"),
                    "stop_reason": init_eng_trace.get("stop_reason"),
                    "latency_ms": init_eng_trace.get("latency_ms"),
                    "sampling_params": init_eng_trace.get("sampling_params") or {},
                }
                if trace_enabled
                else {}
            )

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
                    formatted_prompt=variant.formatted_prompt if trace_enabled else "",
                    raw_initial_output=raw_sql if trace_enabled else "",
                    sme_context_used=variant.sme_context if trace_enabled else "",
                    rulebook_context_used=variant.rulebook_context if trace_enabled else "",
                    initial_llm_metrics=init_metrics,
                    repair_turns=repair_turns,
                )
            )

        # Step 4: Key preservation (Primary Keys, Foreign Keys, Undocumented Joins)
        raw_final_schema = self.preserve_linking_keys(
            linked_fields=linked_fields,
            raw_profile=raw_profile,
            processed_metadata=processed_metadata,
        )

        # Step 5: Post-processing (ground-truth validation, alias resolution, double-quoting for SQL generation)
        all_sqls = [t.final_sql for t in traces] + [t.initial_sql for t in traces]
        discovered_aliases = SchemaPostProcessor.extract_table_aliases(all_sqls)
        quoted_schema, canonical_schema = SchemaPostProcessor.post_process_linked_schema(
            raw_fields=raw_final_schema,
            processed_metadata=processed_metadata,
            table_aliases=discovered_aliases,
            always_quote=True,
        )

        preserved = canonical_schema - linked_fields

        # Step 6: Proactive Literal Grounding from question & trial queries
        if lsh_indexer is not None:
            q_grounded = self.extract_grounded_literals_from_question(
                question=question,
                lsh_indexer=lsh_indexer,
                candidate_columns=canonical_schema or linked_fields,
                threshold=0.55,
            )
            grounded_candidates.extend(q_grounded)

        final_grounded = self.deduplicate_grounded_literals(grounded_candidates)

        usage_summary = {
            "llm_calls": len(llm_call_records),
            "total_prompt_tokens": sum(
                r.prompt_tokens for r in llm_call_records if isinstance(r.prompt_tokens, int)
            ),
            "total_completion_tokens": sum(
                r.completion_tokens for r in llm_call_records if isinstance(r.completion_tokens, int)
            ),
            "total_latency_ms": round(sum(float(r.latency_ms or 0.0) for r in llm_call_records), 3),
            "truncated_calls": sum(1 for r in llm_call_records if r.finish_reason == "length"),
        }

        result = SchemaLinkingResult(
            final_linked_schema=quoted_schema,
            canonical_schema=canonical_schema,
            variant_traces=traces,
            total_retries=total_retries,
            all_harvested_fields=linked_fields,
            preserved_keys=preserved,
            grounded_literals=final_grounded,
            llm_usage_summary=usage_summary,
        )

        # Automatically persist schema_linking_result.json to cache_dir if provided
        if cache_dir is not None:
            c_dir = Path(cache_dir)
            try:
                c_dir.mkdir(parents=True, exist_ok=True)
                out_dict = result.to_dict(
                    question=question,
                    question_id=question_id,
                    include_detailed_traces=trace_enabled,
                )
                res_file = c_dir / "schema_linking_result.json"
                with open(res_file, "w", encoding="utf-8") as f:
                    json.dump(out_dict, f, ensure_ascii=False, indent=2)

                if question_id:
                    q_file = c_dir / f"schema_linking_result_{question_id}.json"
                    with open(q_file, "w", encoding="utf-8") as f:
                        json.dump(out_dict, f, ensure_ascii=False, indent=2)
            except (OSError, TypeError, ValueError) as exc:
                logger.warning("Failed to save schema_linking_result.json in %s: %s", c_dir, exc)

            if trace_enabled and llm_call_records:
                append_llm_calls(c_dir, llm_call_records)

        return result


def task_aligned_schema_linking(
    question: str,
    processed_metadata: dict[str, Any],
    lsh_indexer: Any,
    vector_indexer: Any | None = None,
    llm_engine: Any | None = None,
    raw_profile: dict[str, Any] | None = None,
    max_retries: int = 2,
    cases: str | list[str] | set[str] | None = None,
    cache_dir: Path | str | None = None,
    question_id: str | None = None,
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
        - cases: Specific context variants to run, e.g. "abcde", "abce", or None.
        - cache_dir: Optional directory to persist schema_linking_result.json.
        - question_id: Optional question ID for per-question result files.

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
        cases=cases,
        cache_dir=cache_dir,
        question_id=question_id,
    )
    return result.final_linked_schema
