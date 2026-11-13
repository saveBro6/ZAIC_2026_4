"""Module 3: Orchestration and caching pipeline for database profiling and dual indexing.

Executes RawDatabaseProfiler, ProfileLLMSummarizer, VietnameseLSHIndexer, and
SemanticVectorIndexer once per business and caches computed metadata and indexes
in /tmp to ensure rapid downstream query processing within competition time budgets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from zaic_2026_4.common.trace_logger import (
    append_llm_calls,
    utc_now_iso,
    write_json_trace,
)
from zaic_2026_4.ingestion import ingest_business, resolve_split_paths
from zaic_2026_4.profiling.lsh_indexer import VietnameseLSHIndexer
from zaic_2026_4.profiling.profile_summarizer import (
    DEFAULT_LOCAL_MODEL,
    LLMEngineProtocol,
    ProfileLLMSummarizer,
    create_engine,
)
from zaic_2026_4.profiling.raw_profiler import RawDatabaseProfiler
from zaic_2026_4.profiling.vector_indexer import (
    DEFAULT_EMBED_MODEL_PATH,
    MockEmbedder,
    SemanticVectorIndexer,
)

logger = logging.getLogger(__name__)


class ProfilingOrchestrator:
    """Central orchestrator coordinating database ingestion, raw profiling,
    LLM summarization, dual-indexing (LSH & FAISS), and caching in /tmp.
    """

    def __init__(
        self,
        business_dir: str | Path,
        rulebook_path: str | Path,
        llm_engine: LLMEngineProtocol | Any,
        embedder: Any | None = None,
        vector_model_path: str = DEFAULT_EMBED_MODEL_PATH,
        cache_dir: str | Path | None = None,
        calendar_path: str | Path | None = None,
        calendar_cache_path: str | Path | None = None,
        trace_enabled: bool = True,
    ) -> None:
        """Initialize ProfilingOrchestrator.

        Args:
            business_dir: Directory containing business tabular files and README.md.
            rulebook_path: Path to dataset split rulebook.md.
            llm_engine: LLM engine conforming to LLMEngineProtocol (vLLM or MockLLMEngine).
            embedder: Optional sentence embedder instance for FAISS indexing.
            vector_model_path: Path to embedding model (defaults to /app/models/bge-m3).
            cache_dir: Cache directory for metadata and indexes (defaults to /tmp/cache_<biz_id>).
            calendar_path: Optional path to calendar.csv file. Auto-discovered if omitted.
            calendar_cache_path: Optional calendar cache path retained for backward compatibility.
        """
        self.business_dir = Path(business_dir).resolve()
        self.rulebook_path = Path(rulebook_path).resolve()
        self.llm_engine = llm_engine
        self.embedder = embedder
        self.vector_model_path = vector_model_path
        cal_candidate = self.business_dir.parent.parent / "calendar.csv"
        self.calendar_path = (
            Path(calendar_path).resolve()
            if calendar_path
            else (cal_candidate if cal_candidate.is_file() else None)
        )
        self.calendar_cache_path = (
            Path(calendar_cache_path).resolve()
            if calendar_cache_path
            else None
        )

        if cache_dir is not None:
            self.cache_dir = Path(cache_dir).resolve()
        else:
            biz_name = self.business_dir.name or "default"
            self.cache_dir = Path(f"/tmp/cache_{biz_name}").resolve()

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.trace_enabled = trace_enabled
        self.db_path: Path | None = None
        self._pipeline_result: dict[str, Any] | None = None
        # Per-stage wall-clock breakdown of run_pipeline, exposed for pipeline tracing.
        self.last_stage_timings: dict[str, float] = {}

    def _read_file(self, path: str | Path) -> str:
        """Safely read a text file with UTF-8 encoding and fallback handling.

        Args:
            path: File path to read.

        Returns:
            File contents string or empty string if file does not exist.
        """
        p = Path(path)
        if not p.is_file():
            return ""
        try:
            return p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return ""
        except OSError:
            return ""

    def _input_fingerprint(self) -> str:
        """Fingerprint the business input files so stale cached artifacts can be detected."""
        hasher = hashlib.sha1()
        try:
            files = sorted(p for p in self.business_dir.rglob("*") if p.is_file())
        except OSError:
            files = []
        for path in files:
            try:
                stat = path.stat()
                relative = path.relative_to(self.business_dir).as_posix()
            except (OSError, ValueError):
                continue
            hasher.update(f"{relative}:{stat.st_size}:{int(stat.st_mtime)}".encode())
        if self.calendar_path is not None and self.calendar_path.is_file():
            try:
                cal_stat = self.calendar_path.stat()
                hasher.update(f"calendar:{cal_stat.st_size}:{int(cal_stat.st_mtime)}".encode())
            except OSError:
                pass
        return hasher.hexdigest()

    def _get_db_path(self, force_rebuild: bool = False) -> str:
        """Locate existing SQLite database or automatically ingest tabular data into SQLite.

        Args:
            force_rebuild: When True (or when the business inputs changed) the cached
                database is discarded so the changed data is actually re-ingested instead
                of being silently served from a stale file.

        Returns:
            Absolute path string to SQLite database file.
        """
        biz_name = self.business_dir.name
        cached_db_path = self.cache_dir / f"{biz_name}.db"

        if force_rebuild:
            self.db_path = None
            if cached_db_path.is_file():
                cached_db_path.unlink()
        elif self.db_path is not None and self.db_path.exists():
            return str(self.db_path)
        elif cached_db_path.is_file():
            self.db_path = cached_db_path
            return str(cached_db_path)

        # Single authoritative DB generation: ingest all tabular data and calendar
        created_db_path = ingest_business(
            business_dir=self.business_dir,
            output_dir=self.cache_dir,
            create_views=True,
            calendar_path=self.calendar_path,
        )
        self.db_path = Path(created_db_path)
        return str(self.db_path)

    def run_pipeline(self, force_rebuild: bool = False, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Execute the profiling and indexing pipeline freshly for this business.

        Args:
            force_rebuild: If True, regenerates all artifacts even if cached on disk.

        Returns:
            Dictionary with keys 'metadata', 'raw_profile', 'lsh_indexer', 'vector_indexer'.
        """
        if not force_rebuild and self._pipeline_result is not None:
            return self._pipeline_result

        profile_json_path = self.cache_dir / "metadata_profile.json"
        lsh_cache_path = self.cache_dir / "lsh_index.pkl"
        faiss_cache_dir = self.cache_dir / "faiss_index"
        fingerprint_path = self.cache_dir / "cache_fingerprint.txt"
        timings: dict[str, float] = {}

        fingerprint = self._input_fingerprint()
        if not force_rebuild and fingerprint_path.is_file():
            try:
                cached_fingerprint = fingerprint_path.read_text(encoding="utf-8").strip()
            except OSError:
                cached_fingerprint = ""
            if cached_fingerprint and cached_fingerprint != fingerprint:
                logger.info(
                    "Business inputs changed for %s; rebuilding cached profiling artifacts.",
                    self.business_dir.name,
                )
                force_rebuild = True

        # Step 1 & 2: Metadata Profiling (Disk cache reload when available). Artifacts are
        # reused independently: a missing index must not force a full LLM re-profiling run.
        if not force_rebuild and profile_json_path.is_file():
            with open(profile_json_path, "r", encoding="utf-8") as f:
                processed_metadata = json.load(f)

            raw_profile = None
            raw_profile_path = self.cache_dir / "raw_profile.json"
            if raw_profile_path.is_file():
                try:
                    with open(raw_profile_path, "r", encoding="utf-8") as f:
                        raw_profile = json.load(f)
                except (OSError, json.JSONDecodeError):
                    raw_profile = None
            self.raw_profile = raw_profile

            t_art0 = time.perf_counter()
            lsh_indexer = None
            if lsh_cache_path.is_file():
                try:
                    lsh_indexer = VietnameseLSHIndexer.load(lsh_cache_path)
                except Exception as exc:  # noqa: BLE001 - any corrupt/stale artifact is rebuilt
                    logger.warning("Rebuilding incompatible LSH index %s: %s", lsh_cache_path, exc)
            if lsh_indexer is None:
                lsh_indexer = VietnameseLSHIndexer()
                lsh_indexer.build_index(processed_metadata)
                lsh_indexer.save(lsh_cache_path)
            timings["lsh_index"] = time.perf_counter() - t_art0

            t_vec0 = time.perf_counter()
            if faiss_cache_dir.is_dir():
                vector_indexer = SemanticVectorIndexer.load(
                    faiss_cache_dir,
                    model_path=self.vector_model_path,
                    embedder=self.embedder,
                )
            else:
                readme_text_cached = self._read_file(self.business_dir / "README.md")
                vector_indexer = SemanticVectorIndexer(
                    model_path=self.vector_model_path,
                    embedder=self.embedder,
                )
                vector_indexer.build_index(
                    processed_metadata=processed_metadata,
                    rulebook_text=self._read_file(self.rulebook_path),
                    readme_text=readme_text_cached,
                )
                vector_indexer.save(faiss_cache_dir)
            timings["faiss_index"] = time.perf_counter() - t_vec0
            if self.embedder is None and vector_indexer is not None:
                self.embedder = vector_indexer.embedder

            # Resolve the database through ingestion so dry-run verification stays enabled
            # even when the cached artifacts predate the database file.
            self._get_db_path()

            self.last_stage_timings = {"cache_reuse": 1.0, **timings}
            result = {
                "metadata": processed_metadata,
                "raw_profile": raw_profile,
                "lsh_indexer": lsh_indexer,
                "vector_indexer": vector_indexer,
            }
            self._pipeline_result = result
            return result

        t_db0 = time.perf_counter()
        db_path = self._get_db_path(force_rebuild=force_rebuild)
        timings["db_ingest"] = time.perf_counter() - t_db0

        # Step 1: Raw Profiling on business tables and calendar
        t_raw0 = time.perf_counter()
        raw_profiler = RawDatabaseProfiler(db_path, calendar_path=self.calendar_path)
        raw_profile = raw_profiler.profile_database()
        self.raw_profile = raw_profile
        timings["raw_profile"] = time.perf_counter() - t_raw0

        # Save raw profile to cache for diagnostics
        raw_profile_path = self.cache_dir / "raw_profile.json"
        with open(raw_profile_path, "w", encoding="utf-8") as f:
            json.dump(raw_profile, f, ensure_ascii=False, indent=2)

        # Step 2: LLM Summarization across all tables (including calendar)
        t_llm0 = time.perf_counter()
        readme_text = self._read_file(self.business_dir / "README.md")
        summarizer = ProfileLLMSummarizer(self.llm_engine)
        processed_metadata = summarizer.generate_profiles(raw_profile, readme_text)
        timings["llm_profiling"] = time.perf_counter() - t_llm0

        # Persist updated metadata JSON
        with open(profile_json_path, "w", encoding="utf-8") as f:
            json.dump(processed_metadata, f, ensure_ascii=False, indent=2)

        # Persist profiling LLM traces in cache_dir when enabled
        if self.trace_enabled:
            cols_trace = summarizer.last_profiling_trace
            total_ptok = sum(c["prompt_tokens"] for c in cols_trace if isinstance(c.get("prompt_tokens"), int))
            total_ctok = sum(c["completion_tokens"] for c in cols_trace if isinstance(c.get("completion_tokens"), int))
            total_lat = round(sum(float(c.get("latency_ms") or 0.0) for c in cols_trace), 3)
            write_json_trace(
                self.cache_dir / "profiling_trace.json",
                {
                    "business_id": self.business_dir.name,
                    "generated_at": utc_now_iso(),
                    "total_columns": len(cols_trace),
                    "total_prompt_tokens": total_ptok,
                    "total_completion_tokens": total_ctok,
                    "total_latency_ms": total_lat,
                    "columns": cols_trace,
                },
            )
            for rec in summarizer.last_llm_call_records:
                rec.metadata["business_id"] = self.business_dir.name
            append_llm_calls(self.cache_dir, summarizer.last_llm_call_records)

        # Step 3: Vietnamese LSH Indexing
        t_lsh0 = time.perf_counter()
        lsh_indexer = VietnameseLSHIndexer()
        lsh_indexer.build_index(processed_metadata)
        lsh_indexer.save(lsh_cache_path)
        timings["lsh_index"] = time.perf_counter() - t_lsh0

        # Step 4: FAISS Semantic Vector Indexing
        t_vec0 = time.perf_counter()
        rulebook_text = self._read_file(self.rulebook_path)
        vector_indexer = SemanticVectorIndexer(
            model_path=self.vector_model_path,
            embedder=self.embedder,
        )
        vector_indexer.build_index(
            processed_metadata=processed_metadata,
            rulebook_text=rulebook_text,
            readme_text=readme_text,
        )
        vector_indexer.save(faiss_cache_dir)
        timings["faiss_index"] = time.perf_counter() - t_vec0

        if self.embedder is None and vector_indexer is not None:
            self.embedder = vector_indexer.embedder

        try:
            fingerprint_path.write_text(fingerprint, encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to persist cache fingerprint in %s: %s", self.cache_dir, exc)

        self.last_stage_timings = {"cache_reuse": 0.0, **timings}
        result = {
            "metadata": processed_metadata,
            "raw_profile": raw_profile,
            "lsh_indexer": lsh_indexer,
            "vector_indexer": vector_indexer,
        }
        self._pipeline_result = result
        return result

    def query_literals(self, entity_text: str, top_k: int = 10) -> list[tuple[str, str]]:
        """Query literal mention against indexed column values using LSH.

        Args:
            entity_text: Entity string from query.
            top_k: Maximum number of matches to return.

        Returns:
            List of (col_key, real_val) tuples sorted by similarity descending.
        """
        pipeline = self.run_pipeline()
        return pipeline["lsh_indexer"].query_literal(entity_text, top_k=top_k)

    def search_columns(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """Search top semantically matching database columns using FAISS.

        Args:
            query: Natural language query text.
            top_k: Maximum number of columns to return.

        Returns:
            List of matching document dicts with score.
        """
        pipeline = self.run_pipeline()
        return pipeline["vector_indexer"].search_columns(query, top_k=top_k)

    def search_rules(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Search top semantically matching rulebook clauses using FAISS.

        Args:
            query: Natural language query text.
            top_k: Maximum number of rulebook clauses to return.

        Returns:
            List of matching rulebook dicts with score.
        """
        pipeline = self.run_pipeline()
        return pipeline["vector_indexer"].search_rules(query, top_k=top_k)


def profile_split_calendar(
    calendar_path: str | Path,
    llm_engine: LLMEngineProtocol | Any,
    embedder: Any | None = None,
    vector_model_path: str = DEFAULT_EMBED_MODEL_PATH,
    cache_path: str | Path = "/tmp/calendar_profile.json",
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Profile split-level calendar.csv once and cache metadata and indexes.

    Args:
        calendar_path: Path to calendar.csv file.
        llm_engine: LLM engine conforming to LLMEngineProtocol.
        embedder: Optional sentence embedder instance for FAISS indexing.
        vector_model_path: Path to embedding model weights.
        cache_path: Target cache file path (default /tmp/calendar_profile.json).
        force_rebuild: If True, regenerates all artifacts even if cached.

    Returns:
        Dictionary containing metadata, raw_profile, lsh_indexer, and vector_indexer.
    """
    c_path = Path(cache_path).resolve()
    c_path.parent.mkdir(parents=True, exist_ok=True)

    cal_path = Path(calendar_path).resolve()
    if not cal_path.is_file():
        raise FileNotFoundError(f"Calendar file not found: {cal_path}")

    # Check cache
    if not force_rebuild and c_path.is_file():
        with open(c_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        lsh_indexer = VietnameseLSHIndexer()
        lsh_indexer.build_index(metadata)

        vector_indexer = SemanticVectorIndexer(
            model_path=vector_model_path,
            embedder=embedder,
        )
        vector_indexer.build_index(metadata)

        return {
            "metadata": metadata,
            "raw_profile": None,
            "lsh_indexer": lsh_indexer,
            "vector_indexer": vector_indexer,
            "cache_path": c_path,
        }

    # Step 1: Raw Profiling via Polars scan_csv
    raw_profile = RawDatabaseProfiler.profile_calendar(cal_path)

    # Step 2: LLM Summarization
    summarizer = ProfileLLMSummarizer(llm_engine)
    metadata = summarizer.summarize_calendar(raw_profile)

    # Write cache JSON
    with open(c_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Step 3 & 4: Build LSH & FAISS indexes
    lsh_indexer = VietnameseLSHIndexer()
    lsh_indexer.build_index(metadata)

    vector_indexer = SemanticVectorIndexer(
        model_path=vector_model_path,
        embedder=embedder,
    )
    vector_indexer.build_index(metadata)

    return {
        "metadata": metadata,
        "raw_profile": raw_profile,
        "lsh_indexer": lsh_indexer,
        "vector_indexer": vector_indexer,
        "cache_path": c_path,
    }


def main() -> None:
    """CLI execution entry point for ProfilingOrchestrator."""
    parser = argparse.ArgumentParser(
        description="Profile a business directory, generate LLM metadata, and construct LSH/FAISS indexes."
    )
    parser.add_argument(
        "--split",
        "--split-dir",
        dest="split",
        type=str,
        default="dataset/train",
        help="Name or path of dataset split (e.g. 'train', 'public_test', 'private_test'). Default: 'dataset/train'.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="dataset",
        help="Base directory containing dataset splits (default: 'dataset').",
    )
    parser.add_argument(
        "--biz-id",
        "--biz_id",
        dest="biz_id",
        type=str,
        default=None,
        help="Identifier of a single business to profile (e.g. 'biz_101', 'dev_001'). If omitted, profiles all businesses in split.",
    )
    parser.add_argument(
        "--calendar-cache",
        type=str,
        default=None,
        help="Optional calendar cache path retained for backward compatibility.",
    )
    parser.add_argument(
        "--output-dir",
        "--cache-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Target base output directory for profiling artifacts (redirects /tmp to this directory).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockLLMEngine and MockEmbedder for fast offline execution without GPU.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=f"Use local LLM model {DEFAULT_LOCAL_MODEL}.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Custom LLM model path or repo ID.",
    )
    parser.add_argument(
        "--vector-model-path",
        type=str,
        default=DEFAULT_EMBED_MODEL_PATH,
        help="Embedding model path for FAISS indexer.",
    )
    parser.add_argument(
        "--no-trace",
        action="store_false",
        dest="trace_enabled",
        default=True,
        help="Disable detailed profiling trace output in cache_dir.",
    )

    args = parser.parse_args()

    start_time = time.time()

    if args.mock:
        engine = create_engine(mock=True)
        embedder = MockEmbedder()
    else:
        engine = create_engine(local=args.local, model=args.model)
        embedder = None

    paths = resolve_split_paths(args.split, base_dir=args.base_dir, require_questions=False)
    split_p = paths["split_dir"]
    data_p = paths["data_dir"]
    rb_path = paths["rulebook_path"]
    cal_path = paths["calendar_path"]

    # Locate businesses to profile
    if args.biz_id:
        single_biz_dir = (data_p / args.biz_id).resolve()
        if not single_biz_dir.is_dir():
            parser.error(f"Business directory not found in split data: {single_biz_dir}")
        biz_dirs = [single_biz_dir]
    else:
        biz_dirs = sorted([d for d in data_p.iterdir() if d.is_dir()])
        if not biz_dirs:
            parser.error(f"No business directories found in {data_p}")

    print(f"Initializing profiling for {split_p.name} ({len(biz_dirs)} business(es))...")

    cal_target = None
    if args.calendar_cache:
        cal_target = Path(args.calendar_cache).resolve()
    elif args.output_dir:
        cal_target = Path(args.output_dir).resolve() / "calendar_profile.json"

    if cal_target and cal_path and cal_path.is_file():
        cal_target.parent.mkdir(parents=True, exist_ok=True)
        cal_res = profile_split_calendar(
            calendar_path=cal_path,
            llm_engine=engine,
            embedder=embedder,
            vector_model_path=args.vector_model_path,
            cache_path=cal_target,
            force_rebuild=True,
        )
        if embedder is None and "vector_indexer" in cal_res and cal_res["vector_indexer"] is not None:
            embedder = cal_res["vector_indexer"].embedder

    total_meta = 0
    total_lsh = 0
    total_vector = 0

    for idx, b_dir in enumerate(biz_dirs, 1):
        t_b0 = time.time()
        b_cache = (
            Path(args.output_dir).resolve() / f"cache_{b_dir.name}"
            if args.output_dir
            else Path(f"/tmp/cache_{b_dir.name}")
        )
        print(f"[{idx}/{len(biz_dirs)}] Processing {b_dir.name}...")

        orchestrator = ProfilingOrchestrator(
            business_dir=b_dir,
            rulebook_path=rb_path,
            llm_engine=engine,
            embedder=embedder,
            vector_model_path=args.vector_model_path,
            cache_dir=b_cache,
            calendar_path=cal_path if (cal_path and cal_path.is_file()) else None,
            trace_enabled=args.trace_enabled,
        )
        pipe = orchestrator.run_pipeline(force_rebuild=True)
        if embedder is None and "vector_indexer" in pipe and pipe["vector_indexer"] is not None:
            embedder = pipe["vector_indexer"].embedder

        m_cnt = len(pipe["metadata"])
        l_cnt = len(pipe["lsh_indexer"].value_to_col_map)
        v_cnt = len(pipe["vector_indexer"].doc_map)

        total_meta += m_cnt
        total_lsh += l_cnt
        total_vector += v_cnt
        print(f"     Done in {time.time() - t_b0:.2f}s ({m_cnt} columns, {l_cnt} LSH items, {v_cnt} docs).")

    elapsed_total = time.time() - start_time
    print(f"Split profiling completed successfully for {len(biz_dirs)} business(es) in {elapsed_total:.2f}s:")
    print(f"  - Total metadata columns: {total_meta}")
    print(f"  - Total LSH items:        {total_lsh}")
    print(f"  - Total FAISS documents:  {total_vector}")
    print(f"  - Average per business:   {elapsed_total / len(biz_dirs):.2f}s")


if __name__ == "__main__":
    main()
