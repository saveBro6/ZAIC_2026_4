"""Module 3: Orchestration and caching pipeline for database profiling and dual indexing.

Executes RawDatabaseProfiler, ProfileLLMSummarizer, VietnameseLSHIndexer, and
SemanticVectorIndexer once per business and caches computed metadata and indexes
in /tmp to ensure rapid downstream query processing within competition time budgets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from zaic_2026_4.ingestion.service import ingest_business
from zaic_2026_4.profiling.lsh_indexer import VietnameseLSHIndexer
from zaic_2026_4.profiling.profile_summarizer import (
    DEFAULT_LOCAL_MODEL,
    LLMEngineProtocol,
    ProfileLLMSummarizer,
    create_engine,
)
from zaic_2026_4.profiling.raw_profiler import RawDatabaseProfiler
from zaic_2026_4.profiling.vector_indexer import (
    DEFAULT_MODEL_PATH as DEFAULT_VECTOR_MODEL_PATH,
    MockEmbedder,
    SemanticVectorIndexer,
)


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
        vector_model_path: str = DEFAULT_VECTOR_MODEL_PATH,
        cache_dir: str | Path | None = None,
        calendar_path: str | Path | None = None,
        calendar_cache_path: str | Path = "/tmp/calendar_profile.json",
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
            calendar_cache_path: Path to shared split-level calendar cache JSON (default: /tmp/calendar_profile.json).
        """
        self.business_dir = Path(business_dir).resolve()
        self.rulebook_path = Path(rulebook_path).resolve()
        self.llm_engine = llm_engine
        self.embedder = embedder
        self.vector_model_path = vector_model_path
        self.calendar_path = Path(calendar_path).resolve() if calendar_path else None
        self.calendar_cache_path = Path(calendar_cache_path).resolve()

        if cache_dir is not None:
            self.cache_dir = Path(cache_dir).resolve()
        else:
            biz_name = self.business_dir.name or "default"
            self.cache_dir = Path(f"/tmp/cache_{biz_name}").resolve()

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path: Path | None = None
        self._pipeline_result: dict[str, Any] | None = None

    def _resolve_calendar_path(self) -> Path | None:
        """Attempt to auto-discover calendar.csv if not explicitly provided."""
        if self.calendar_path is not None and self.calendar_path.is_file():
            return self.calendar_path

        # 1. Look in dataset split directory: <split>/data/<biz_id> -> <split>/calendar.csv
        candidate_split = self.business_dir.parent.parent / "calendar.csv"
        if candidate_split.is_file():
            return candidate_split

        # 2. Look relative to rulebook: <split>/rulebook.md -> <split>/calendar.csv
        candidate_rulebook = self.rulebook_path.parent / "calendar.csv"
        if candidate_rulebook.is_file():
            return candidate_rulebook

        # 3. Competition container standard read-only mount
        candidate_root = Path("/data/calendar.csv")
        if candidate_root.is_file():
            return candidate_root

        return None

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
            except Exception:
                return ""
        except Exception:
            return ""

    def _get_db_path(self) -> str:
        """Locate existing SQLite database or automatically ingest CSV/XLSX into /tmp SQLite.

        Returns:
            Absolute path string to SQLite database file.
        """
        if self.db_path is not None and self.db_path.exists():
            return str(self.db_path)

        biz_name = self.business_dir.name
        cached_db_path = self.cache_dir / f"{biz_name}.db"

        # 1. Reuse existing converted DB in cache if present
        if cached_db_path.is_file():
            self.db_path = cached_db_path
            return str(cached_db_path)

        cached_split_dbs = list(self.cache_dir.rglob(f"{biz_name}.db"))
        if cached_split_dbs and cached_split_dbs[0].is_file():
            self.db_path = cached_split_dbs[0]
            return str(self.db_path)

        # 2. Check if business_dir contains pre-existing .sqlite or .db file
        # and has no raw tabular CSV/XLSX files that need ingestion
        if self.business_dir.is_dir():
            sqlite_files = sorted(
                list(self.business_dir.glob("*.sqlite")) + list(self.business_dir.glob("*.db"))
            )
            tabular_files = sorted(
                list(self.business_dir.glob("*.csv"))
                + list(self.business_dir.glob("*.xlsx"))
                + list(self.business_dir.glob("*.xls"))
            )
            if sqlite_files and not tabular_files:
                # Standalone SQLite database (e.g. biz_123/pos.sqlite)
                self.db_path = sqlite_files[0]
                return str(self.db_path)

        # 3. Convert CSV/XLSX (or mixed data) into temporary SQLite database in cache_dir
        created_db_path = ingest_business(
            business_dir=self.business_dir,
            output_dir=self.cache_dir,
            create_views=True,
        )
        self.db_path = Path(created_db_path)
        return str(self.db_path)

    def run_pipeline(self, force_rebuild: bool = False) -> dict[str, Any]:
        """Execute the profiling and indexing pipeline with multi-tier caching.

        Args:
            force_rebuild: If True, ignore existing cached artifacts and re-run all steps.

        Returns:
            Dictionary with keys 'metadata', 'lsh_indexer', 'vector_indexer'.
        """
        if not force_rebuild and self._pipeline_result is not None:
            return self._pipeline_result

        profile_json_path = self.cache_dir / "metadata_profile.json"
        lsh_cache_path = self.cache_dir / "lsh_index.pkl"
        faiss_cache_dir = self.cache_dir / "faiss_index"

        # Step 1 & 2: Metadata Profiling (Cache check or execute Raw Profiler + LLM Summarizer)
        if not force_rebuild and profile_json_path.is_file():
            with open(profile_json_path, "r", encoding="utf-8") as f:
                processed_metadata = json.load(f)
        else:
            db_path = self._get_db_path()

            # Step 1: Raw Profiling
            raw_profiler = RawDatabaseProfiler(db_path)
            raw_profile = raw_profiler.profile_database()

            # Save raw profile to cache for diagnostics
            raw_profile_path = self.cache_dir / "raw_profile.json"
            with open(raw_profile_path, "w", encoding="utf-8") as f:
                json.dump(raw_profile, f, ensure_ascii=False, indent=2)

            # Step 2: LLM Summarization
            readme_text = self._read_file(self.business_dir / "README.md")
            summarizer = ProfileLLMSummarizer(self.llm_engine)
            processed_metadata = summarizer.generate_profiles(raw_profile, readme_text)

        # Merge split-level calendar metadata from cache
        calendar_meta = None
        if not force_rebuild and self.calendar_cache_path.is_file():
            try:
                with open(self.calendar_cache_path, "r", encoding="utf-8") as f:
                    calendar_meta = json.load(f)
            except Exception:
                calendar_meta = None

        if calendar_meta is None:
            resolved_cal = self._resolve_calendar_path()
            if resolved_cal and resolved_cal.is_file():
                cal_res = profile_split_calendar(
                    calendar_path=resolved_cal,
                    llm_engine=self.llm_engine,
                    embedder=self.embedder,
                    vector_model_path=self.vector_model_path,
                    cache_path=self.calendar_cache_path,
                    force_rebuild=force_rebuild,
                )
                calendar_meta = cal_res.get("metadata", {})

        if calendar_meta:
            for k, v in calendar_meta.items():
                if k not in processed_metadata:
                    processed_metadata[k] = v

        # Persist updated metadata JSON
        with open(profile_json_path, "w", encoding="utf-8") as f:
            json.dump(processed_metadata, f, ensure_ascii=False, indent=2)

        # Step 3: Vietnamese LSH Indexing
        if not force_rebuild and lsh_cache_path.is_file():
            lsh_indexer = VietnameseLSHIndexer.load(lsh_cache_path)
        else:
            lsh_indexer = VietnameseLSHIndexer()
            lsh_indexer.build_index(processed_metadata)
            lsh_indexer.save(lsh_cache_path)

        # Step 4: FAISS Semantic Vector Indexing
        faiss_index_file = faiss_cache_dir / "index.faiss"
        faiss_doc_map = faiss_cache_dir / "doc_map.json"
        if not force_rebuild and faiss_index_file.is_file() and faiss_doc_map.is_file():
            vector_indexer = SemanticVectorIndexer.load(
                faiss_cache_dir,
                model_path=self.vector_model_path,
                embedder=self.embedder,
            )
        else:
            rulebook_text = self._read_file(self.rulebook_path)
            readme_text = self._read_file(self.business_dir / "README.md")
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

        result = {
            "metadata": processed_metadata,
            "lsh_indexer": lsh_indexer,
            "vector_indexer": vector_indexer,
        }
        self._pipeline_result = result
        return result

    def query_literals(self, entity_text: str, top_k: int = 5) -> list[tuple[str, str]]:
        """Query literal mention against indexed column values using LSH.

        Args:
            entity_text: Entity string from query.
            top_k: Maximum number of matches to return.

        Returns:
            List of (col_key, real_val) tuples sorted by similarity descending.
        """
        pipeline = self.run_pipeline()
        return pipeline["lsh_indexer"].query_literal(entity_text, top_k=top_k)

    def search_columns(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
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
    vector_model_path: str = DEFAULT_VECTOR_MODEL_PATH,
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
        "--business-dir",
        type=str,
        default=None,
        help="Path to business directory containing data files and README.md.",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default=None,
        help="Path to full split directory (e.g. dataset/public_test). Profiles calendar and all businesses in split.",
    )
    parser.add_argument(
        "--rulebook",
        type=str,
        default=None,
        help="Path to rulebook.md for the dataset split.",
    )
    parser.add_argument(
        "--calendar",
        type=str,
        default=None,
        help="Path to split-level calendar.csv.",
    )
    parser.add_argument(
        "--calendar-cache",
        type=str,
        default="/tmp/calendar_profile.json",
        help="Path to shared calendar profile cache JSON (default: /tmp/calendar_profile.json).",
    )
    parser.add_argument(
        "--profile-calendar-only",
        action="store_true",
        help="Profile only calendar.csv and exit.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Target cache directory (defaults to /tmp/cache_<biz_id>).",
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
        default=DEFAULT_VECTOR_MODEL_PATH,
        help="Embedding model path for FAISS indexer.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignore cached files and regenerate all profiles and indexes.",
    )

    args = parser.parse_args()

    start_time = time.time()

    if args.mock:
        engine = create_engine(mock=True)
        embedder = MockEmbedder()
    else:
        engine = create_engine(local=args.local, model=args.model)
        embedder = None

    if args.profile_calendar_only:
        if not args.calendar:
            parser.error("--profile-calendar-only requires --calendar path.")
        cal_res = profile_split_calendar(
            calendar_path=args.calendar,
            llm_engine=engine,
            embedder=embedder,
            vector_model_path=args.vector_model_path,
            cache_path=args.calendar_cache,
            force_rebuild=args.force_rebuild,
        )
        print(f"Calendar split profiling completed in {time.time() - start_time:.2f}s:")
        print(f"  - Cached metadata columns: {len(cal_res['metadata'])}")
        print(f"  - Cache path:              {cal_res['cache_path']}")
        return

    # Handle full split profiling mode
    if args.split_dir:
        split_p = Path(args.split_dir).resolve()
        if not split_p.is_dir():
            parser.error(f"Provided --split-dir is not a valid directory: {split_p}")

        cal_path = Path(args.calendar).resolve() if args.calendar else (split_p / "calendar.csv")
        rb_path = Path(args.rulebook).resolve() if args.rulebook else (split_p / "rulebook.md")

        if not rb_path.is_file():
            parser.error(f"Rulebook not found in split: {rb_path}")

        # Locate businesses
        data_p = split_p / "data" if (split_p / "data").is_dir() else split_p
        biz_dirs = sorted([d for d in data_p.iterdir() if d.is_dir()])
        if not biz_dirs:
            parser.error(f"No business directories found in {data_p}")

        print(f"Initializing split profiling for {split_p.name} ({len(biz_dirs)} businesses)...")

        # Step 1: Profile calendar.csv once for split if present
        if cal_path.is_file():
            print(f"Profiling split calendar: {cal_path}...")
            profile_split_calendar(
                calendar_path=cal_path,
                llm_engine=engine,
                embedder=embedder,
                vector_model_path=args.vector_model_path,
                cache_path=args.calendar_cache,
                force_rebuild=args.force_rebuild,
            )

        # Step 2: Iterate and profile all businesses
        total_meta = 0
        total_lsh = 0
        total_vector = 0

        for idx, b_dir in enumerate(biz_dirs, 1):
            t_b0 = time.time()
            b_cache = (
                Path(args.cache_dir) / f"cache_{b_dir.name}"
                if args.cache_dir
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
                calendar_path=cal_path if cal_path.is_file() else None,
                calendar_cache_path=args.calendar_cache,
            )
            pipe = orchestrator.run_pipeline(force_rebuild=args.force_rebuild)
            m_cnt = len(pipe["metadata"])
            l_cnt = len(pipe["lsh_indexer"].value_to_col_map)
            v_cnt = len(pipe["vector_indexer"].doc_map)

            total_meta += m_cnt
            total_lsh += l_cnt
            total_vector += v_cnt
            print(f"     Done in {time.time() - t_b0:.2f}s ({m_cnt} columns, {l_cnt} LSH items, {v_cnt} docs).")

        elapsed_total = time.time() - start_time
        print(f"Split profiling completed successfully for {len(biz_dirs)} businesses in {elapsed_total:.2f}s:")
        print(f"  - Total metadata columns: {total_meta}")
        print(f"  - Total LSH items:        {total_lsh}")
        print(f"  - Total FAISS documents:  {total_vector}")
        print(f"  - Average per business:   {elapsed_total / len(biz_dirs):.2f}s")
        return

    if not args.business_dir or not args.rulebook:
        parser.error("Either --split-dir, --profile-calendar-only, or both --business-dir and --rulebook must be provided.")

    print(f"Initializing ProfilingOrchestrator for {args.business_dir}...")

    orchestrator = ProfilingOrchestrator(
        business_dir=args.business_dir,
        rulebook_path=args.rulebook,
        llm_engine=engine,
        embedder=embedder,
        vector_model_path=args.vector_model_path,
        cache_dir=args.cache_dir,
        calendar_path=args.calendar,
        calendar_cache_path=args.calendar_cache,
    )

    pipeline = orchestrator.run_pipeline(force_rebuild=args.force_rebuild)
    elapsed = time.time() - start_time

    meta_count = len(pipeline["metadata"])
    lsh_items = len(pipeline["lsh_indexer"].value_to_col_map)
    vector_docs = len(pipeline["vector_indexer"].doc_map)

    print(f"Pipeline completed successfully in {elapsed:.2f}s:")
    print(f"  - Cached metadata columns: {meta_count}")
    print(f"  - LSH indexed literals:    {lsh_items}")
    print(f"  - FAISS indexed documents: {vector_docs}")
    print(f"  - Cache directory:         {orchestrator.cache_dir}")


if __name__ == "__main__":
    main()
