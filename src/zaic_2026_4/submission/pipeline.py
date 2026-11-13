"""Submission pipeline orchestrator coordinating caching, schema linking, diversity generation, and streaming JSONL output."""

from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zaic_2026_4.common.trace_logger import build_call_id, write_json_trace
from zaic_2026_4.constants import (
    DEFAULT_BUSINESS_CACHE_MAX_ENTRIES,
    DEFAULT_CANDIDATE_SEEDS,
    DEFAULT_EXECUTION_MAX_RETRIES,
    DEFAULT_LOGS_DIR,
    DEFAULT_RULEBOOK_MODE,
    DEFAULT_SQL_MAX_RETRIES,
    DEFAULT_SUBMISSION_PATH,
)
from zaic_2026_4.ingestion import resolve_split_paths
from zaic_2026_4.profiling.profile_manager import ProfilingOrchestrator
from zaic_2026_4.schema_linking.task_aligned_linker import TaskAlignedSchemaLinker
from zaic_2026_4.submission.candidate_generator import CandidateGenerator
from zaic_2026_4.submission.models import QuestionItem, QuestionSubmission
from zaic_2026_4.submission.result_formatter import (
    ResultFormatter,
    detect_percent_scale,
)
from zaic_2026_4.submission.voting_executor import (
    VotingExecutor,
    answer_signature,
    has_evidence,
)

logger = logging.getLogger(__name__)


def _attempt_blocking_codes(candidate: Any) -> list[str]:
    """Extract blocking issue codes of the candidate's selected attempt from its trace."""
    attempts = getattr(candidate, "attempts_history", None) or []
    if not attempts:
        return []
    best_idx = getattr(candidate, "best_attempt_idx", None)
    if not isinstance(best_idx, int) or not 0 <= best_idx < len(attempts):
        best_idx = len(attempts) - 1
    codes = {
        str(issue.get("code"))
        for issue in (attempts[best_idx].get("issues") or [])
        if isinstance(issue, dict)
        and issue.get("code")
        and issue.get("severity") == "blocking"
    }
    return sorted(codes)


def compute_unanswerable_proof(coverage_report: Any | None) -> bool:
    """Return True only when the linked schema covered every question-implied aspect.

    A missing coverage report means schema linking failed, so the all-columns fallback is
    in force and an "unanswerable" verdict cannot be trusted.
    """
    return coverage_report is not None and not coverage_report.has_unresolved_gaps


@dataclass
class BusinessContext:
    """Cached per-business profiling, database connection, and indexing artifacts."""

    business_id: str
    db_path: Path
    processed_metadata: dict[str, Any]
    raw_profile: dict[str, Any] | None
    lsh_indexer: Any
    vector_indexer: Any
    voting_executor: VotingExecutor
    linker: TaskAlignedSchemaLinker
    cache_dir: Path | None = None
    load_timings: dict[str, float] = field(default_factory=dict)


class SubmissionPipeline:
    """End-to-end pipeline executing questions against business databases and streaming submission output."""

    def __init__(
        self,
        data_dir: Path | str,
        rulebook_path: Path | str,
        llm_engine: Any,
        embedder: Any | None = None,
        vector_model_path: str = "BAAI/bge-m3",
        calendar_path: Path | str | None = None,
        output_path: Path | str = DEFAULT_SUBMISSION_PATH,
        logs_dir: Path | str = DEFAULT_LOGS_DIR,
        cache_base_dir: Path | str = "/tmp",
        seeds: tuple[int, ...] = DEFAULT_CANDIDATE_SEEDS,
        sql_max_retries: int = DEFAULT_SQL_MAX_RETRIES,
        execution_max_retries: int = DEFAULT_EXECUTION_MAX_RETRIES,
        max_retries: int | None = None,
        rulebook_mode: str = DEFAULT_RULEBOOK_MODE,
        cases: str | list[str] | set[str] | None = "abcde",
        trace_enabled: bool = True,
        business_cache_max_entries: int = DEFAULT_BUSINESS_CACHE_MAX_ENTRIES,
    ) -> None:
        """Initialize submission pipeline.

        Args:
            data_dir: Root directory containing business data folders (<split>/data/).
            rulebook_path: Path to dataset split rulebook.md.
            llm_engine: Inference engine conforming to LLMEngineProtocol.
            embedder: Sentence embedder for vector indexing.
            vector_model_path: HuggingFace model path or local path for embeddings.
            calendar_path: Optional path to calendar.csv.
            output_path: Target path for submission.jsonl.
            logs_dir: Directory to record per-question diagnostic traces.
            cache_base_dir: Base directory for storing temporary profiling caches.
            seeds: Randomization seeds for candidate generation.
            sql_max_retries: Retries for static AST self-correction.
            execution_max_retries: Retries for dry-run runtime execution self-correction.
            max_retries: Backward-compatible alias for sql_max_retries.
            rulebook_mode: Mode for injecting split rulebook ('hybrid', 'full', 'retrieval').
            cases: Specific context variants to run for schema linking (default: 'abcde').
        """
        self.data_dir = Path(data_dir).resolve()
        self.rulebook_path = Path(rulebook_path).resolve()
        self.llm_engine = llm_engine
        self.embedder = embedder
        self.vector_model_path = vector_model_path
        self.calendar_path = Path(calendar_path).resolve() if calendar_path else None
        self.output_path = Path(output_path).resolve()
        self.logs_dir = Path(logs_dir).resolve()
        self.cache_base_dir = Path(cache_base_dir).resolve()
        self.seeds = seeds
        self.sql_max_retries = max_retries if max_retries is not None else sql_max_retries
        self.execution_max_retries = execution_max_retries
        self.max_retries = self.sql_max_retries
        self.rulebook_mode = rulebook_mode
        self.cases = cases
        self.trace_enabled = trace_enabled

        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.logs_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError):
            fallback_dir = Path("result").resolve()
            logger.warning(
                "Permission denied or filesystem error creating %s or %s. Falling back to local directory %s",
                self.output_path.parent,
                self.logs_dir,
                fallback_dir,
            )
            fallback_dir.mkdir(parents=True, exist_ok=True)
            self.output_path = fallback_dir / self.output_path.name
            self.logs_dir = fallback_dir / "logs"
            self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Bounded LRU: keeping every business DB and index resident does not fit in memory
        # when a split spans many businesses, so the least recently used one is evicted.
        self._business_cache: OrderedDict[str, BusinessContext] = OrderedDict()
        self._business_cache_max_entries = max(1, int(business_cache_max_entries))
        self.run_id = build_call_id()
        self._rulebook_text = self._read_text_file(self.rulebook_path)

    @classmethod
    def from_split(
        cls,
        split: str | Path,
        llm_engine: Any,
        base_dir: Path | str = "dataset",
        embedder: Any | None = None,
        vector_model_path: str = "BAAI/bge-m3",
        output_path: Path | str = DEFAULT_SUBMISSION_PATH,
        logs_dir: Path | str = DEFAULT_LOGS_DIR,
        cache_base_dir: Path | str = "/tmp",
        seeds: tuple[int, ...] = DEFAULT_CANDIDATE_SEEDS,
        sql_max_retries: int = DEFAULT_SQL_MAX_RETRIES,
        execution_max_retries: int = DEFAULT_EXECUTION_MAX_RETRIES,
        max_retries: int | None = None,
        rulebook_mode: str = DEFAULT_RULEBOOK_MODE,
        cases: str | list[str] | set[str] | None = "abcde",
        trace_enabled: bool = True,
        business_cache_max_entries: int = DEFAULT_BUSINESS_CACHE_MAX_ENTRIES,
    ) -> tuple[SubmissionPipeline, Path]:
        """Create a SubmissionPipeline from a dataset split.

        Returns:
            Tuple of (pipeline_instance, questions_path).
        """
        paths = resolve_split_paths(split, base_dir=base_dir)
        pipeline = cls(
            data_dir=paths["data_dir"],
            rulebook_path=paths["rulebook_path"],
            llm_engine=llm_engine,
            embedder=embedder,
            vector_model_path=vector_model_path,
            calendar_path=paths["calendar_path"],
            output_path=output_path,
            logs_dir=logs_dir,
            cache_base_dir=cache_base_dir,
            seeds=seeds,
            sql_max_retries=sql_max_retries,
            execution_max_retries=execution_max_retries,
            max_retries=max_retries,
            rulebook_mode=rulebook_mode,
            cases=cases,
            trace_enabled=trace_enabled,
            business_cache_max_entries=business_cache_max_entries,
        )
        return pipeline, paths["questions_path"]

    @staticmethod
    def _read_text_file(path: Path) -> str:
        """Read text file safely."""
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8").strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read %s: %s", path, exc)
        return ""

    def get_or_load_business(self, business_id: str) -> BusinessContext:
        """Load or create cached business context including DB, profiles, and indexers."""
        if business_id in self._business_cache:
            self._business_cache.move_to_end(business_id)
            return self._business_cache[business_id]

        biz_dir = self.data_dir / business_id
        if not biz_dir.is_dir():
            raise FileNotFoundError(f"Business data directory not found: {biz_dir}")

        biz_cache_dir = self.cache_base_dir / f"cache_{business_id}"
        orchestrator = ProfilingOrchestrator(
            business_dir=biz_dir,
            rulebook_path=self.rulebook_path,
            llm_engine=self.llm_engine,
            embedder=self.embedder,
            vector_model_path=self.vector_model_path,
            cache_dir=biz_cache_dir,
            calendar_path=self.calendar_path,
            trace_enabled=self.trace_enabled,
        )

        pipeline_res = orchestrator.run_pipeline()
        processed_metadata = pipeline_res["metadata"]
        lsh_indexer = pipeline_res["lsh_indexer"]
        vector_indexer = pipeline_res["vector_indexer"]
        db_path = orchestrator.db_path or (biz_cache_dir / f"{business_id}.db")

        # Load raw profile if present
        raw_profile = None
        raw_profile_path = biz_cache_dir / "raw_profile.json"
        if raw_profile_path.is_file():
            try:
                with open(raw_profile_path, "r", encoding="utf-8") as f:
                    raw_profile = json.load(f)
            except Exception:  # noqa: BLE001
                raw_profile = None

        linker = TaskAlignedSchemaLinker(
            llm_engine=self.llm_engine,
            max_retries=2,
        )
        voting_executor = VotingExecutor(db_path=db_path)

        context = BusinessContext(
            business_id=business_id,
            db_path=db_path,
            processed_metadata=processed_metadata,
            raw_profile=raw_profile,
            lsh_indexer=lsh_indexer,
            vector_indexer=vector_indexer,
            voting_executor=voting_executor,
            linker=linker,
            cache_dir=biz_cache_dir,
            load_timings=dict(getattr(orchestrator, "last_stage_timings", {}) or {}),
        )
        self._business_cache[business_id] = context
        while len(self._business_cache) > self._business_cache_max_entries:
            evicted_id, _ = self._business_cache.popitem(last=False)
            logger.info("Evicted business context %s from the bounded LRU cache", evicted_id)
        return context

    def process_question(self, item: QuestionItem) -> QuestionSubmission:
        """Process a single question through schema linking, diversity generation, voting, and formatting."""
        t0 = time.perf_counter()

        try:
            return self._process_question_impl(item, t0)
        except Exception as exc:
            logger.exception("Unexpected error processing question %s", item.id)
            return QuestionSubmission(
                id=item.id,
                type="abstain",
                answer=None,
                elapsed_seconds=time.perf_counter() - t0,
                voting_status="abstain_fallback",
                selection_reason=f"abstain: Unexpected execution error ({exc})",
            )

    def _process_question_impl(self, item: QuestionItem, t0: float) -> QuestionSubmission:
        """Internal implementation of question processing."""
        t_biz0 = time.perf_counter()
        business_was_cached = item.business_id in self._business_cache
        try:
            biz_ctx = self.get_or_load_business(item.business_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load business context for %s: %s", item.business_id, exc)
            return QuestionSubmission(
                id=item.id,
                type="abstain",
                elapsed_seconds=time.perf_counter() - t0,
                voting_status="abstain_fallback",
                selection_reason=f"abstain: Failed to load business data ({exc})",
            )
        t_biz_sec = time.perf_counter() - t_biz0

        # 1. Run Task-Aligned Schema Linking
        t_sl0 = time.perf_counter()
        schema_linking_trace: dict[str, Any] = {}
        sl_usage: dict[str, Any] = {}
        coverage_report = None
        intent_summary: dict[str, Any] = {}
        try:
            linking_res = biz_ctx.linker.link_schema(
                question=item.question,
                processed_metadata=biz_ctx.processed_metadata,
                lsh_indexer=biz_ctx.lsh_indexer,
                vector_indexer=biz_ctx.vector_indexer,
                raw_profile=biz_ctx.raw_profile,
                cases=self.cases,
                cache_dir=biz_ctx.cache_dir,
                question_id=item.id,
                trace_enabled=self.trace_enabled,
                as_of_date=item.as_of_date,
            )
            reduced_schema = linking_res.final_linked_schema
            grounded_literals = linking_res.grounded_literals
            coverage_report = linking_res.coverage
            intent_summary = linking_res.intent.to_dict() if linking_res.intent is not None else {}
            sl_usage = dict(linking_res.llm_usage_summary or {})
            if self.trace_enabled:
                schema_linking_trace = linking_res.to_dict(
                    question=item.question,
                    question_id=item.id,
                    include_detailed_traces=True,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Schema linking error on %s: %s", item.id, exc)
            reduced_schema = set()
            grounded_literals = []
            coverage_report = None
            intent_summary = {}
            if self.trace_enabled:
                schema_linking_trace = {"error": str(exc)}
        t_sl_sec = time.perf_counter() - t_sl0

        # Fallback if reduced schema is empty: collect all columns from metadata
        if not reduced_schema:
            target_dict = biz_ctx.processed_metadata.get("metadata_profile", biz_ctx.processed_metadata)
            reduced_schema = {k for k in target_dict if "." in k}

        # 2. Candidate Diversity Generation (3 candidates with seeds & schema permutation)
        t_cg0 = time.perf_counter()
        cand_gen = CandidateGenerator(
            llm_engine=self.llm_engine,
            seeds=self.seeds,
            sql_max_retries=self.sql_max_retries,
            execution_max_retries=self.execution_max_retries,
            rulebook_mode=self.rulebook_mode,
            db_path=biz_ctx.db_path,
        )
        candidates = cand_gen.generate_all_candidates(
            question=item.question,
            as_of_date=item.as_of_date,
            reduced_schema=reduced_schema,
            processed_metadata=biz_ctx.processed_metadata,
            rulebook_text=self._rulebook_text,
            vector_indexer=biz_ctx.vector_indexer,
            db_path=biz_ctx.db_path,
            grounded_literals=grounded_literals,
            cache_dir=biz_ctx.cache_dir,
            question_id=item.id,
            trace_enabled=self.trace_enabled,
        )
        t_cg_sec = time.perf_counter() - t_cg0

        # 3. Candidate Execution & Majority Voting
        t_vt0 = time.perf_counter()
        biz_ctx.voting_executor.execute_candidates(candidates)
        # A coverage gap means a table required by the question shape never reached the
        # linked schema, so "unanswerable" could simply be a schema-linking miss.
        unanswerable_proof = compute_unanswerable_proof(coverage_report)
        winner, status, reason = biz_ctx.voting_executor.vote(
            candidates, unanswerable_proof=unanswerable_proof
        )
        t_vt_sec = time.perf_counter() - t_vt0

        # 4. Result Formatting
        t_fmt0 = time.perf_counter()
        final_type = "abstain"
        final_ans = None

        if winner is None:
            submission = QuestionSubmission(
                id=item.id,
                type="abstain",
                answer=None,
                elapsed_seconds=0.0,
                candidates=candidates,
                winning_candidate_id=None,
                voting_status=status,
                selection_reason=reason,
            )
        elif winner.predicted_type == "unanswerable":
            final_type = "unanswerable"
            submission = QuestionSubmission(
                id=item.id,
                type="unanswerable",
                answer=None,
                elapsed_seconds=0.0,
                candidates=candidates,
                winning_candidate_id=winner.candidate_id,
                voting_status=status,
                selection_reason=reason,
            )
        elif winner.predicted_type == "abstain" or winner.raw_result is None:
            submission = QuestionSubmission(
                id=item.id,
                type="abstain",
                answer=None,
                elapsed_seconds=0.0,
                candidates=candidates,
                winning_candidate_id=winner.candidate_id,
                voting_status=status,
                selection_reason=reason,
            )
        else:
            ans_type, ans_val = ResultFormatter.format_result(
                raw_rows=winner.raw_result,
                target_type=winner.predicted_type,
                raw_columns=winner.result_columns,
                sql=winner.final_sql,
            )
            final_type = ans_type
            final_ans = ans_val
            submission = QuestionSubmission(
                id=item.id,
                type=ans_type,
                answer=ans_val,
                elapsed_seconds=0.0,
                candidates=candidates,
                winning_candidate_id=winner.candidate_id,
                voting_status=status,
                selection_reason=reason,
            )

        # Build diagnostic formatting_trace
        ev_cands = [c for c in candidates if c.execution_success and has_evidence(c.result_set)]
        sig_groups: dict[Any, int] = {}
        candidate_signatures: list[dict[str, Any]] = []
        for c in candidates:
            s = answer_signature(c)
            if c in ev_cands and s is not None:
                sig_groups[s] = sig_groups.get(s, 0) + 1
            candidate_signatures.append({
                "candidate_id": c.candidate_id,
                "seed": c.seed,
                "predicted_type": c.predicted_type,
                "execution_success": c.execution_success,
                "has_evidence": has_evidence(c.result_set),
                "answer_signature": list(s) if s is not None else None,
            })
        cluster_sizes = sorted(sig_groups.values(), reverse=True)

        raw_val_preview = None
        pct_scale = None
        pct_scaled = False
        res_cols: list[str] = []

        if winner is not None:
            res_cols = list(winner.result_columns[:20])
            if winner.raw_result and len(winner.raw_result) > 0:
                first_row = winner.raw_result[0]
                raw_val_preview = first_row[0] if isinstance(first_row, (list, tuple)) else first_row
            if winner.predicted_type == "percent":
                pct_scale = detect_percent_scale(winner.final_sql)
                try:
                    if raw_val_preview is not None and not isinstance(raw_val_preview, bool):
                        num_f = float(raw_val_preview)
                        if pct_scale == "ratio" and abs(num_f) <= 1.0:
                            pct_scaled = True
                except (ValueError, TypeError):
                    pass

        submission.formatting_trace = {
            "answer_type": final_type,
            "final_answer": final_ans,
            "percent_scale": pct_scale,
            "percent_scaled": pct_scaled,
            "raw_value": str(raw_val_preview) if raw_val_preview is not None else None,
            "result_columns": res_cols,
            "evidence_candidates": len(ev_cands),
            "cluster_sizes": cluster_sizes,
        }
        t_fmt_sec = time.perf_counter() - t_fmt0
        elapsed = time.perf_counter() - t0
        submission.elapsed_seconds = elapsed

        if self.trace_enabled:
            cand_attempts = [att for c in candidates for att in c.attempts_history]
            cand_calls = len(cand_attempts)
            cand_ptok = sum(
                a["prompt_tokens"] for a in cand_attempts if isinstance(a.get("prompt_tokens"), int)
            )
            cand_ctok = sum(
                a["completion_tokens"] for a in cand_attempts if isinstance(a.get("completion_tokens"), int)
            )
            cand_lat = round(sum(float(a.get("latency_ms") or 0.0) for a in cand_attempts), 3)
            cand_trunc = sum(1 for a in cand_attempts if a.get("finish_reason") == "length")

            submission.schema_linking_trace = schema_linking_trace
            submission.run_id = self.run_id
            submission.business_load_detail = (
                {"memory_cache_hit": round(t_biz_sec, 4)}
                if business_was_cached
                else {k: round(float(v), 4) for k, v in biz_ctx.load_timings.items()}
            )
            submission.stage_timings_sec = {
                "business_load": round(t_biz_sec, 4),
                "schema_linking": round(t_sl_sec, 4),
                "candidate_generation": round(t_cg_sec, 4),
                "voting": round(t_vt_sec, 4),
                "formatting": round(t_fmt_sec, 4),
                "total": round(elapsed, 4),
            }
            submission.candidate_signatures = candidate_signatures
            submission.formatting_trace["coverage"] = (
                coverage_report.to_dict() if coverage_report is not None else None
            )
            submission.formatting_trace["unanswerable_proof"] = unanswerable_proof
            submission.formatting_trace["query_intent"] = intent_summary
            submission.formatting_trace["candidate_diagnostics"] = [
                {
                    "candidate_id": c.candidate_id,
                    "repair_stop_reason": c.repair_stop_reason,
                    "syntax_valid": c.is_valid_syntax,
                    "blocking_codes": _attempt_blocking_codes(c),
                    "attempts": len(c.attempts_history),
                }
                for c in candidates
            ]
            submission.llm_usage_summary = {
                "schema_linking": sl_usage,
                "candidate_generation": {
                    "llm_calls": cand_calls,
                    "total_prompt_tokens": cand_ptok,
                    "total_completion_tokens": cand_ctok,
                    "total_latency_ms": cand_lat,
                    "truncated_calls": cand_trunc,
                },
                "total_llm_calls": int(sl_usage.get("llm_calls", 0)) + cand_calls,
                "total_prompt_tokens": int(sl_usage.get("total_prompt_tokens", 0)) + cand_ptok,
                "total_completion_tokens": int(sl_usage.get("total_completion_tokens", 0)) + cand_ctok,
                "total_latency_ms": round(float(sl_usage.get("total_latency_ms", 0.0)) + cand_lat, 3),
                "total_truncated_calls": int(sl_usage.get("truncated_calls", 0)) + cand_trunc,
            }

            if biz_ctx.cache_dir is not None:
                write_json_trace(
                    biz_ctx.cache_dir / f"question_trace_{item.id}.json",
                    submission.to_full_cache_trace(item),
                )

        # 5. Flush diagnostic log to /result/logs/<id>.json (unchanged)
        log_file = self.logs_dir / f"{item.id}.json"
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(submission.to_trace_log(item), f, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write log trace for %s: %s", item.id, exc)

        return submission

    def run(
        self,
        questions_path: Path | str,
        limit: int | None = None,
    ) -> list[QuestionSubmission]:
        """Execute submission pipeline across questions file with streaming line-by-line flush."""
        q_path = Path(questions_path).resolve()
        if not q_path.is_file():
            raise FileNotFoundError(f"Questions file not found: {q_path}")

        questions: list[QuestionItem] = []
        with open(q_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        questions.append(QuestionItem.from_dict(json.loads(line_str)))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Skipping malformed question line: %s (%s)", line_str, exc)

        if limit is not None and limit > 0:
            questions = questions[:limit]

        submissions: list[QuestionSubmission] = []
        with open(self.output_path, "w", encoding="utf-8") as out_f:
            for item in questions:
                try:
                    sub = self.process_question(item)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Top-level exception on %s: %s", item.id, exc)
                    sub = QuestionSubmission(
                        id=item.id,
                        type="abstain",
                        answer=None,
                        voting_status="abstain_fallback",
                        selection_reason=f"abstain: Unhandled error ({exc})",
                    )
                submissions.append(sub)

                # Line-by-line flush to submission.jsonl with serialization guard
                try:
                    out_line = json.dumps(sub.to_submission_dict(), ensure_ascii=False)
                except (TypeError, ValueError) as exc:
                    logger.error(
                        "Failed to serialize submission for %s: %s. Emitting safe abstain fallback.",
                        item.id,
                        exc,
                    )
                    sub.type = "abstain"
                    sub.answer = None
                    out_line = json.dumps(sub.to_submission_dict(), ensure_ascii=False)

                out_f.write(out_line + "\n")
                out_f.flush()

        return submissions


__all__ = [
    "BusinessContext",
    "SubmissionPipeline",
]
