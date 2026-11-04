"""Unit and integration tests for Module 3: ProfilingOrchestrator (profile_manager.py)."""

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from zaic_2026_4.profiling.profile_manager import ProfilingOrchestrator, main
from zaic_2026_4.profiling.profile_summarizer import MockLLMEngine
from zaic_2026_4.profiling.vector_indexer import MockEmbedder


class ExplodingLLMEngine:
    """Mock engine that raises an error if invoked, used to test cache hits."""

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("LLM engine should not be called when cache is hit!")


class ExplodingEmbedder:
    """Mock embedder that raises an error if invoked, used to test cache hits."""

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Embedder should not be called when cache is hit!")


@pytest.fixture
def sample_business_csv(tmp_path: Path) -> tuple[Path, Path]:
    """Create a temporary business folder with a CSV file and README.md."""
    biz_dir = tmp_path / "biz_test_csv"
    biz_dir.mkdir(parents=True, exist_ok=True)

    csv_path = biz_dir / "sales.csv"
    csv_path.write_text(
        "id,ma_sp,ten_sp,so_luong,don_gia,ngay_ban\n"
        "1,SP001,Sting dâu 330ml,10,12000,2026-05-01\n"
        "2,SP002,Nước ngọt Coca 320ml,5,10000,2026-05-02\n"
        "3,SP001,Sting dâu 330ml,8,12000,2026-05-03\n",
        encoding="utf-8",
    )

    readme_path = biz_dir / "README.md"
    readme_path.write_text(
        "# Cửa Hàng Tạp Hóa\nBán lẻ nước giải khát và bánh kẹo.\n",
        encoding="utf-8",
    )

    rulebook_path = tmp_path / "rulebook.md"
    rulebook_path.write_text(
        "# Quy Tắc Nghiệp Vụ\n\n## Doanh Thu\nDoanh thu bằng số lượng nhân đơn giá.",
        encoding="utf-8",
    )

    return biz_dir, rulebook_path


@pytest.fixture
def sample_business_sqlite(tmp_path: Path) -> tuple[Path, Path]:
    """Create a temporary business folder with a standalone SQLite database."""
    biz_dir = tmp_path / "biz_test_sqlite"
    biz_dir.mkdir(parents=True, exist_ok=True)

    db_path = biz_dir / "pos.sqlite"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE DonHang (id INTEGER PRIMARY KEY, ma_kh TEXT, tong_tien INTEGER);"
    )
    cur.execute("INSERT INTO DonHang VALUES (1, 'KH001', 50000);")
    cur.execute("INSERT INTO DonHang VALUES (2, 'KH002', 120000);")
    conn.commit()
    conn.close()

    readme_path = biz_dir / "README.md"
    readme_path.write_text("# Dữ liệu POS\nQuản lý đơn hàng.\n", encoding="utf-8")

    rulebook_path = tmp_path / "rulebook.md"
    rulebook_path.write_text("# Quy tắc POS\n", encoding="utf-8")

    return biz_dir, rulebook_path


def test_get_db_path_with_standalone_sqlite(sample_business_sqlite: tuple[Path, Path], tmp_path: Path) -> None:
    """Verify _get_db_path directly returns pre-existing .sqlite file when no CSV/XLSX exist."""
    biz_dir, rulebook_path = sample_business_sqlite
    cache_dir = tmp_path / "cache_sqlite"

    orchestrator = ProfilingOrchestrator(
        business_dir=biz_dir,
        rulebook_path=rulebook_path,
        llm_engine=MockLLMEngine(),
        embedder=MockEmbedder(),
        cache_dir=cache_dir,
    )

    db_path = orchestrator._get_db_path()
    assert Path(db_path).resolve() == (biz_dir / "pos.sqlite").resolve()


def test_get_db_path_with_csv(sample_business_csv: tuple[Path, Path], tmp_path: Path) -> None:
    """Verify _get_db_path automatically converts CSV to SQLite database inside cache_dir."""
    biz_dir, rulebook_path = sample_business_csv
    cache_dir = tmp_path / "cache_csv"

    orchestrator = ProfilingOrchestrator(
        business_dir=biz_dir,
        rulebook_path=rulebook_path,
        llm_engine=MockLLMEngine(),
        embedder=MockEmbedder(),
        cache_dir=cache_dir,
    )

    db_path = orchestrator._get_db_path()
    assert Path(db_path).is_file()
    assert Path(db_path).parent.resolve() == cache_dir.resolve()
    assert Path(db_path).name == f"{biz_dir.name}.db"

    # Verify SQLite contains the sales table
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    conn.close()
    assert "sales" in tables


def test_get_db_path_with_split_hierarchy(tmp_path: Path) -> None:
    """Verify _get_db_path reuses DB when business_dir path contains split name."""
    biz_dir = tmp_path / "train" / "data" / "biz_split_test"
    biz_dir.mkdir(parents=True, exist_ok=True)
    (biz_dir / "t.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    rulebook = tmp_path / "train" / "rulebook.md"
    rulebook.parent.mkdir(parents=True, exist_ok=True)
    rulebook.write_text("# Rules\n", encoding="utf-8")

    cache_dir = tmp_path / "cache_split"
    orchestrator1 = ProfilingOrchestrator(
        business_dir=biz_dir,
        rulebook_path=rulebook,
        llm_engine=MockLLMEngine(),
        embedder=MockEmbedder(),
        cache_dir=cache_dir,
    )
    db1 = orchestrator1._get_db_path()
    assert Path(db1).is_file()

    # Second orchestrator instance must find existing cached DB without re-ingesting
    orchestrator2 = ProfilingOrchestrator(
        business_dir=biz_dir,
        rulebook_path=rulebook,
        llm_engine=MockLLMEngine(),
        embedder=MockEmbedder(),
        cache_dir=cache_dir,
    )
    db2 = orchestrator2._get_db_path()
    assert db2 == db1


def test_run_pipeline_end_to_end_mock(
    sample_business_csv: tuple[Path, Path], tmp_path: Path
) -> None:
    """Verify complete execution of ProfilingOrchestrator pipeline."""
    biz_dir, rulebook_path = sample_business_csv
    cache_dir = tmp_path / "cache_e2e"

    orchestrator = ProfilingOrchestrator(
        business_dir=biz_dir,
        rulebook_path=rulebook_path,
        llm_engine=MockLLMEngine(),
        embedder=MockEmbedder(dimension=128),
        cache_dir=cache_dir,
    )

    pipeline = orchestrator.run_pipeline()

    assert "metadata" in pipeline
    assert "lsh_indexer" in pipeline
    assert "vector_indexer" in pipeline

    metadata = pipeline["metadata"]
    assert "sales.ten_sp" in metadata
    assert "minimal_profile" in metadata["sales.ten_sp"]
    assert "maximal_profile" in metadata["sales.ten_sp"]

    # Verify cached artifacts were written to cache_dir
    assert (cache_dir / "metadata_profile.json").is_file()
    assert (cache_dir / "raw_profile.json").is_file()
    assert (cache_dir / "lsh_index.pkl").is_file()
    assert (cache_dir / "faiss_index" / "index.faiss").is_file()
    assert (cache_dir / "faiss_index" / "doc_map.json").is_file()


def test_pipeline_disk_caching(
    sample_business_csv: tuple[Path, Path], tmp_path: Path
) -> None:
    """Verify that a subsequent run reloads directly from disk cache without invoking LLM or embedder."""
    biz_dir, rulebook_path = sample_business_csv
    cache_dir = tmp_path / "cache_disk"

    # 1. First run creates the disk cache
    orchestrator1 = ProfilingOrchestrator(
        business_dir=biz_dir,
        rulebook_path=rulebook_path,
        llm_engine=MockLLMEngine(),
        embedder=MockEmbedder(dimension=128),
        cache_dir=cache_dir,
    )
    res1 = orchestrator1.run_pipeline()

    # 2. Second run with Exploding engines - must succeed purely from disk cache
    orchestrator2 = ProfilingOrchestrator(
        business_dir=biz_dir,
        rulebook_path=rulebook_path,
        llm_engine=ExplodingLLMEngine(),
        embedder=ExplodingEmbedder(),
        cache_dir=cache_dir,
    )
    res2 = orchestrator2.run_pipeline()

    assert len(res2["metadata"]) == len(res1["metadata"])
    assert res2["metadata"]["sales.ma_sp"]["minimal_profile"] == res1["metadata"]["sales.ma_sp"]["minimal_profile"]
    assert len(res2["lsh_indexer"].value_to_col_map) == len(res1["lsh_indexer"].value_to_col_map)
    assert len(res2["vector_indexer"].doc_map) == len(res1["vector_indexer"].doc_map)


def test_pipeline_in_memory_caching(
    sample_business_csv: tuple[Path, Path], tmp_path: Path
) -> None:
    """Verify in-memory caching returns identical dictionary object immediately."""
    biz_dir, rulebook_path = sample_business_csv
    cache_dir = tmp_path / "cache_mem"

    orchestrator = ProfilingOrchestrator(
        business_dir=biz_dir,
        rulebook_path=rulebook_path,
        llm_engine=MockLLMEngine(),
        embedder=MockEmbedder(dimension=128),
        cache_dir=cache_dir,
    )

    res1 = orchestrator.run_pipeline()
    res2 = orchestrator.run_pipeline()
    assert res1 is res2


def test_pipeline_force_rebuild(
    sample_business_csv: tuple[Path, Path], tmp_path: Path
) -> None:
    """Verify force_rebuild=True recomputes the pipeline."""
    biz_dir, rulebook_path = sample_business_csv
    cache_dir = tmp_path / "cache_force"

    orchestrator = ProfilingOrchestrator(
        business_dir=biz_dir,
        rulebook_path=rulebook_path,
        llm_engine=MockLLMEngine(),
        embedder=MockEmbedder(dimension=128),
        cache_dir=cache_dir,
    )

    res1 = orchestrator.run_pipeline()
    res2 = orchestrator.run_pipeline(force_rebuild=True)
    assert res1 is not res2
    assert "metadata" in res2


def test_convenience_methods(
    sample_business_csv: tuple[Path, Path], tmp_path: Path
) -> None:
    """Verify query_literals, search_columns, and search_rules return expected results."""
    biz_dir, rulebook_path = sample_business_csv
    cache_dir = tmp_path / "cache_queries"

    orchestrator = ProfilingOrchestrator(
        business_dir=biz_dir,
        rulebook_path=rulebook_path,
        llm_engine=MockLLMEngine(),
        embedder=MockEmbedder(dimension=128),
        cache_dir=cache_dir,
    )

    # 1. Literal matching via LSH
    matches = orchestrator.query_literals("Sting dâu", top_k=3)
    assert isinstance(matches, list)
    assert len(matches) > 0
    matched_values = [val for _, val in matches]
    assert any("Sting" in val for val in matched_values)

    # 2. Semantic column search
    col_results = orchestrator.search_columns("sản phẩm", top_k=2)
    assert isinstance(col_results, list)
    assert len(col_results) > 0

    # 3. Rulebook search
    rule_results = orchestrator.search_rules("doanh thu", top_k=2)
    assert isinstance(rule_results, list)
    assert len(rule_results) > 0


def test_missing_readme_and_rulebook(tmp_path: Path) -> None:
    """Verify orchestrator runs without error even if README or rulebook files are missing."""
    biz_dir = tmp_path / "biz_empty"
    biz_dir.mkdir(parents=True, exist_ok=True)
    csv_file = biz_dir / "t.csv"
    csv_file.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    rulebook_path = tmp_path / "non_existent_rulebook.md"
    cache_dir = tmp_path / "cache_empty"

    orchestrator = ProfilingOrchestrator(
        business_dir=biz_dir,
        rulebook_path=rulebook_path,
        llm_engine=MockLLMEngine(),
        embedder=MockEmbedder(dimension=64),
        cache_dir=cache_dir,
    )

    pipeline = orchestrator.run_pipeline()
    assert "metadata" in pipeline
    assert "t.a" in pipeline["metadata"]


def test_real_business_train_biz_101(tmp_path: Path) -> None:
    """Integration test against actual dataset/train/data/biz_101 if present."""
    biz_dir = Path("dataset/train/data/biz_101")
    rulebook_path = Path("dataset/train/rulebook.md")

    if not biz_dir.is_dir() or not rulebook_path.is_file():
        pytest.skip("dataset/train/data/biz_101 not present")

    cache_dir = tmp_path / "cache_biz_101"
    orchestrator = ProfilingOrchestrator(
        business_dir=biz_dir,
        rulebook_path=rulebook_path,
        llm_engine=MockLLMEngine(),
        embedder=MockEmbedder(dimension=128),
        cache_dir=cache_dir,
    )

    pipeline = orchestrator.run_pipeline()
    assert len(pipeline["metadata"]) > 0
    assert (cache_dir / "metadata_profile.json").is_file()
    assert (cache_dir / "lsh_index.pkl").is_file()
    assert (cache_dir / "faiss_index" / "index.faiss").is_file()


def test_cli_split_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI --split-dir correctly profiles multiple businesses in split."""
    split_dir = tmp_path / "test_split"
    data_dir = split_dir / "data"
    data_dir.mkdir(parents=True)

    # Business 1
    b1 = data_dir / "biz_a"
    b1.mkdir()
    (b1 / "sales.csv").write_text("id,val\n1,10\n", encoding="utf-8")
    (b1 / "README.md").write_text("# Biz A\n", encoding="utf-8")

    # Business 2
    b2 = data_dir / "biz_b"
    b2.mkdir()
    (b2 / "items.csv").write_text("sku,qty\nSKU1,5\n", encoding="utf-8")
    (b2 / "README.md").write_text("# Biz B\n", encoding="utf-8")

    # Split files
    (split_dir / "rulebook.md").write_text("# Rulebook\n", encoding="utf-8")
    (split_dir / "calendar.csv").write_text("date,weekday\n2024-01-01,1\n", encoding="utf-8")

    cal_cache = tmp_path / "cal_cache.json"
    cache_base = tmp_path / "caches"

    test_args = [
        "profile_manager.py",
        "--split-dir",
        str(split_dir),
        "--calendar-cache",
        str(cal_cache),
        "--cache-dir",
        str(cache_base),
        "--mock",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    main()

    assert cal_cache.is_file()
    assert (cache_base / "cache_biz_a" / "metadata_profile.json").is_file()
    assert (cache_base / "cache_biz_b" / "metadata_profile.json").is_file()


def test_cli_output_dir_redirection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify --output-dir cleanly redirects all outputs from /tmp to the specified directory."""
    split_dir = tmp_path / "custom_split"
    data_dir = split_dir / "data"
    data_dir.mkdir(parents=True)

    b1 = data_dir / "biz_x"
    b1.mkdir()
    (b1 / "data.csv").write_text("id,val\n1,100\n", encoding="utf-8")
    (b1 / "README.md").write_text("# Biz X\n", encoding="utf-8")

    (split_dir / "rulebook.md").write_text("# Rulebook\n", encoding="utf-8")
    (split_dir / "calendar.csv").write_text("date,weekday\n2024-01-01,1\n", encoding="utf-8")

    custom_out = tmp_path / "custom_output_base"

    test_args = [
        "profile_manager.py",
        "--split-dir",
        str(split_dir),
        "--output-dir",
        str(custom_out),
        "--mock",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    main()

    # calendar_profile.json redirected to custom_out
    assert (custom_out / "calendar_profile.json").is_file()
    # business cache redirected to custom_out / cache_biz_x
    assert (custom_out / "cache_biz_x" / "metadata_profile.json").is_file()
    assert (custom_out / "cache_biz_x" / "lsh_index.pkl").is_file()
    assert (custom_out / "cache_biz_x" / "faiss_index" / "index.faiss").is_file()
