"""Submission pipeline orchestrator coordinating caching, schema linking, diversity generation, and streaming JSONL output."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zaic_2026_4.constants import (
    DEFAULT_CANDIDATE_SEEDS,
    DEFAULT_EXECUTION_MAX_RETRIES,
    DEFAULT_LOGS_DIR,
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
)

logger = logging.getLogger(__name__)


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
        rulebook_mode: str = "hybrid",
        cases: str | list[str] | set[str] | None = "abcde",
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

        self._business_cache: dict[str, BusinessContext] = {}
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
        rulebook_mode: str = "hybrid",
        cases: str | list[str] | set[str] | None = "abcde",
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
        )
        self._business_cache[business_id] = context
        return context

    def process_question(self, item: QuestionItem) -> QuestionSubmission:
        """Process a single question through schema linking, diversity generation, voting, and formatting."""
        t0 = time.perf_counter()

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

        # 1. Run Task-Aligned Schema Linking
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
            )
            reduced_schema = linking_res.final_linked_schema
            grounded_literals = linking_res.grounded_literals
        except Exception as exc:  # noqa: BLE001
            logger.warning("Schema linking error on %s: %s", item.id, exc)
            reduced_schema = set()
            grounded_literals = []

        # Fallback if reduced schema is empty: collect all columns from metadata
        if not reduced_schema:
            target_dict = biz_ctx.processed_metadata.get("metadata_profile", biz_ctx.processed_metadata)
            reduced_schema = {k for k in target_dict if "." in k}

        # 2. Candidate Diversity Generation (3 candidates with seeds & schema permutation)
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
        )

        # 3. Candidate Execution & Majority Voting
        biz_ctx.voting_executor.execute_candidates(candidates)
        winner, status, reason = biz_ctx.voting_executor.vote(candidates)

        elapsed = time.perf_counter() - t0

        # 4. Result Formatting
        final_type = "abstain"
        final_ans = None

        if winner is None:
            submission = QuestionSubmission(
                id=item.id,
                type="abstain",
                answer=None,
                elapsed_seconds=elapsed,
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
                elapsed_seconds=elapsed,
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
                elapsed_seconds=elapsed,
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
                elapsed_seconds=elapsed,
                candidates=candidates,
                winning_candidate_id=winner.candidate_id,
                voting_status=status,
                selection_reason=reason,
            )

        # Build diagnostic formatting_trace
        ev_cands = [c for c in candidates if c.execution_success and c.result_set is not None]
        sig_groups: dict[Any, int] = {}
        for c in ev_cands:
            s = answer_signature(c)
            if s is not None:
                sig_groups[s] = sig_groups.get(s, 0) + 1
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

        # 5. Flush diagnostic log to /result/logs/<id>.json
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
                sub = self.process_question(item)
                submissions.append(sub)

                # Line-by-line flush to submission.jsonl
                out_line = json.dumps(sub.to_submission_dict(), ensure_ascii=False)
                out_f.write(out_line + "\n")
                out_f.flush()

        return submissions


__all__ = [
    "BusinessContext",
    "SubmissionPipeline",
]
