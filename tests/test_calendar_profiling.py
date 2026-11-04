"""Unit and integration tests for generic calendar profiling across all 4 profiling modules."""

from pathlib import Path
import tempfile
import polars as pl
import pytest

from zaic_2026_4.profiling.lsh_indexer import VietnameseLSHIndexer
from zaic_2026_4.profiling.profile_manager import (
    ProfilingOrchestrator,
    profile_split_calendar,
)
from zaic_2026_4.profiling.profile_summarizer import (
    MockLLMEngine,
    ProfileLLMSummarizer,
)
from zaic_2026_4.profiling.raw_profiler import (
    RawDatabaseProfiler,
    profile_calendar_csv,
)
from zaic_2026_4.profiling.vector_indexer import (
    MockEmbedder,
    SemanticVectorIndexer,
)

TRAIN_CALENDAR_PATH = Path("dataset/train/calendar.csv")


def test_raw_profiler_calendar_generic() -> None:
    """Verify generic profiling of real calendar.csv without hardcoded column checks."""
    assert TRAIN_CALENDAR_PATH.is_file(), f"Missing dataset: {TRAIN_CALENDAR_PATH}"

    db_profile = RawDatabaseProfiler.profile_calendar(TRAIN_CALENDAR_PATH)
    assert "calendar" in db_profile["tables"]

    cal_table = db_profile["tables"]["calendar"]
    assert cal_table["row_count"] == 1461
    assert not cal_table["is_view"]

    cols = cal_table["columns"]
    assert len(cols) == 7

    # Check date column
    assert "date" in cols
    date_col = cols["date"]
    assert date_col["null_count"] == 0
    assert date_col["data_format"] == "ISO_DATE"

    # Check holiday column
    assert "holiday" in cols
    holiday_col = cols["holiday"]
    assert holiday_col["null_count"] > 1000  # Mostly null for normal days
    samples = holiday_col["sample_values"]
    assert "Tết" in samples or "Giỗ tổ" in samples or "2/9" in samples


def test_raw_profiler_arbitrary_schema(tmp_path: Path) -> None:
    """Verify raw_profiler dynamically handles arbitrary renamed columns (Private Test resilience)."""
    custom_cal_path = tmp_path / "custom_calendar.csv"
    custom_cal_path.write_text(
        "ngay,thu_trong_tuan,ngay_am,thang_am,nam_am,thang_nhuan,le_hoi\n"
        "2024-01-01,1,20,11,2023,false,Tết Dương lịch\n"
        "2024-02-10,6,1,1,2024,false,Tết\n"
        "2024-04-18,4,10,3,2024,false,Giỗ tổ\n"
        "2024-09-02,1,30,7,2024,false,2/9\n",
        encoding="utf-8",
    )

    db_profile = profile_calendar_csv(custom_cal_path)
    cal_table = db_profile["tables"]["calendar"]
    assert cal_table["row_count"] == 4

    cols = cal_table["columns"]
    assert set(cols.keys()) == {
        "ngay",
        "thu_trong_tuan",
        "ngay_am",
        "thang_am",
        "nam_am",
        "thang_nhuan",
        "le_hoi",
    }
    assert cols["ngay"]["data_format"] == "ISO_DATE"
    assert "Tết" in cols["le_hoi"]["sample_values"]


def test_summarizer_calendar_autonomous() -> None:
    """Verify autonomous generation of minimal and maximal profiles without boilerplate dilution."""
    assert TRAIN_CALENDAR_PATH.is_file()

    raw_profile = RawDatabaseProfiler.profile_calendar(TRAIN_CALENDAR_PATH)
    engine = MockLLMEngine()
    summarizer = ProfileLLMSummarizer(engine)

    processed = summarizer.summarize_calendar(raw_profile)
    assert len(processed) == 7

    for col_key, meta in processed.items():
        assert col_key.startswith("calendar.")
        assert "minimal_profile" in meta
        assert "maximal_profile" in meta
        assert len(meta["minimal_profile"]) > 0
        assert len(meta["maximal_profile"]) > 0


def test_calendar_lsh_and_vector_indexing() -> None:
    """Verify literal matching and semantic vector retrieval on calendar columns."""
    assert TRAIN_CALENDAR_PATH.is_file()

    raw_profile = RawDatabaseProfiler.profile_calendar(TRAIN_CALENDAR_PATH)
    engine = MockLLMEngine()
    summarizer = ProfileLLMSummarizer(engine)
    metadata = summarizer.summarize_calendar(raw_profile)

    # 1. LSH Indexer
    lsh = VietnameseLSHIndexer()
    indexed_count = lsh.build_index(metadata)
    assert indexed_count > 0

    # Query holiday literal
    matches = lsh.query_literal("Tết", top_k=3)
    assert any(col == "calendar.holiday" for col, _ in matches)

    # 2. Semantic Vector Indexer with MockEmbedder
    embedder = MockEmbedder(dimension=128)
    vector_indexer = SemanticVectorIndexer(embedder=embedder)
    doc_count = vector_indexer.build_index(metadata)
    assert doc_count == 7

    results = vector_indexer.search_columns("thời gian ngày tháng", top_k=3)
    assert len(results) > 0


def test_lsh_and_vector_merge() -> None:
    """Verify merge methods on VietnameseLSHIndexer and SemanticVectorIndexer."""
    meta1 = {
        "tbl1.colA": {
            "sample_values": ["Sản phẩm A", "Sản phẩm B"],
            "minimal_profile": "Cột sản phẩm A",
            "maximal_profile": "Mô tả chi tiết cột sản phẩm A",
        }
    }
    meta2 = {
        "calendar.holiday": {
            "sample_values": ["Tết", "Giỗ tổ"],
            "minimal_profile": "Cột ngày lễ",
            "maximal_profile": "Mô tả chi tiết ngày lễ",
        }
    }

    # Test LSH Merge
    lsh1 = VietnameseLSHIndexer()
    lsh1.build_index(meta1)

    lsh2 = VietnameseLSHIndexer()
    lsh2.build_index(meta2)

    added_lsh = lsh1.merge(lsh2)
    assert added_lsh == 2
    assert any(col == "calendar.holiday" for col, _ in lsh1.query_literal("Tết"))

    # Test Vector Merge
    embedder = MockEmbedder(dimension=64)
    v1 = SemanticVectorIndexer(embedder=embedder)
    v1.build_index(meta1)

    v2 = SemanticVectorIndexer(embedder=embedder)
    v2.build_index(meta2)

    added_v = v1.merge(v2)
    assert added_v == 1
    assert any(doc["key"] == "calendar.holiday" for doc in v1.doc_map)


def test_split_caching_and_orchestrator_merging(tmp_path: Path) -> None:
    """Verify split-level caching to /tmp and merging with business metadata."""
    cal_cache_file = tmp_path / "test_calendar_profile.json"
    engine = MockLLMEngine()
    embedder = MockEmbedder(dimension=64)

    # 1. Step 1: Profile split-level calendar once
    cal_res = profile_split_calendar(
        calendar_path=TRAIN_CALENDAR_PATH,
        llm_engine=engine,
        embedder=embedder,
        cache_path=cal_cache_file,
    )
    assert cal_cache_file.is_file()
    assert len(cal_res["metadata"]) == 7

    # 2. Step 2: Second call must hit cache
    cal_res_cached = profile_split_calendar(
        calendar_path=TRAIN_CALENDAR_PATH,
        llm_engine=engine,
        embedder=embedder,
        cache_path=cal_cache_file,
    )
    assert cal_res_cached["raw_profile"] is None  # Hit cache

    # 3. Create dummy business directory
    biz_dir = tmp_path / "biz_dummy"
    biz_dir.mkdir()
    (biz_dir / "sales.csv").write_text(
        "id,product,amount\n1,Coca,10000\n2,Pepsi,12000\n",
        encoding="utf-8",
    )
    (biz_dir / "README.md").write_text("# Dummy Biz\n", encoding="utf-8")
    rulebook_file = tmp_path / "rulebook.md"
    rulebook_file.write_text("# Rulebook\n", encoding="utf-8")

    orchestrator = ProfilingOrchestrator(
        business_dir=biz_dir,
        rulebook_path=rulebook_file,
        llm_engine=engine,
        embedder=embedder,
        cache_dir=tmp_path / "cache_biz_dummy",
        calendar_path=TRAIN_CALENDAR_PATH,
        calendar_cache_path=cal_cache_file,
    )

    pipeline = orchestrator.run_pipeline()
    merged_meta = pipeline["metadata"]

    # Verify business tables AND calendar tables are both in merged metadata
    assert "sales.product" in merged_meta
    assert "calendar.holiday" in merged_meta

    # Verify LSH queries resolve both business products and calendar holidays
    prod_matches = orchestrator.query_literals("Coca")
    assert any(col == "sales.product" for col, _ in prod_matches)

    cal_matches = orchestrator.query_literals("Tết")
    assert any(col == "calendar.holiday" for col, _ in cal_matches)
