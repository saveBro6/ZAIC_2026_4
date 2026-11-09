"""Unit tests for Module 2: ProfileLLMSummarizer, prompt templates, and engine factory."""

import json
from pathlib import Path
import sqlite3

import pytest

from zaic_2026_4.profiling.profile_summarizer import (
    DEFAULT_LOCAL_GPU_UTIL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_PROD_GPU_UTIL,
    DEFAULT_PROD_MODEL,
    MAXIMAL_PROFILE_PROMPT,
    MINIMAL_PROFILE_PROMPT,
    LLMEngineProtocol,
    MockLLMEngine,
    ProfileLLMSummarizer,
    create_engine,
)
from zaic_2026_4.profiling.raw_profiler import RawDatabaseProfiler


def test_default_constants() -> None:
    """Verify default models and GPU utilization constants adhere to technical requirements."""
    assert DEFAULT_PROD_MODEL == "Qwen/Qwen2.5-7B-Instruct"
    assert DEFAULT_LOCAL_MODEL == "Qwen/Qwen2.5-1.5B-Instruct"
    assert DEFAULT_PROD_GPU_UTIL == 0.70
    assert DEFAULT_LOCAL_GPU_UTIL == 0.60


def test_engine_factory_mock() -> None:
    """Verify engine factory returns MockLLMEngine when mock=True."""
    engine = create_engine(mock=True)
    assert isinstance(engine, MockLLMEngine)
    assert isinstance(engine, LLMEngineProtocol)

    local_engine = create_engine(local=True, mock=True)
    assert isinstance(local_engine, MockLLMEngine)


def test_mock_llm_engine_generation() -> None:
    """Test MockLLMEngine handling of single strings and batches."""
    engine = MockLLMEngine()

    # Empty batch
    assert engine.generate([]) == []
    assert engine.generate("") == ""

    # Single string minimal prompt
    min_prompt = (
        'You are an expert. 1-sentence SHORT SUMMARY. Table: "DonHang", Column: "ngay_ban"\n'
        'Format & Hints: DATE | ISO Date\n'
    )
    res_single = engine.generate(min_prompt)
    assert isinstance(res_single, str)
    assert "ngay_ban" in res_single
    assert "DonHang" in res_single

    # Batch prompts
    batch = [
        '1-sentence SHORT SUMMARY. Table: "DonHang", Column: "tong_tien"\nFormat & Hints: CURRENCY | VND',
        'Long detailed summary. Table: "KhachHang", Column: "ma_kh"\nFormat & Pattern: STRING',
    ]
    res_batch = engine.generate(batch)
    assert isinstance(res_batch, list)
    assert len(res_batch) == 2
    assert "tong_tien" in res_batch[0]
    assert "ma_kh" in res_batch[1]


def test_minimal_prompt_few_shots() -> None:
    """Verify MINIMAL_PROFILE_PROMPT contains XML structure and exactly 2 Vietnamese SMB few-shot examples."""
    assert "<system>" in MINIMAL_PROFILE_PROMPT
    assert "<instructions>" in MINIMAL_PROFILE_PROMPT
    assert "<business_readme>" in MINIMAL_PROFILE_PROMPT
    assert "<few_shot_examples>" in MINIMAL_PROFILE_PROMPT
    assert '<example id="1">' in MINIMAL_PROFILE_PROMPT
    assert '<example id="2">' in MINIMAL_PROFILE_PROMPT
    assert 'Cột: "ma_bien_so"' in MINIMAL_PROFILE_PROMPT
    assert 'Cột: "phi_dich_vu"' in MINIMAL_PROFILE_PROMPT
    assert 'Cột "ma_bien_so"' in MINIMAL_PROFILE_PROMPT
    assert 'Cột "phi_dich_vu"' in MINIMAL_PROFILE_PROMPT
    assert "<target_column>" in MINIMAL_PROFILE_PROMPT
    assert "<output>" in MINIMAL_PROFILE_PROMPT


def test_maximal_prompt_few_shots() -> None:
    """Verify MAXIMAL_PROFILE_PROMPT contains XML structure and exactly 2 Vietnamese SMB few-shot examples."""
    assert "<system>" in MAXIMAL_PROFILE_PROMPT
    assert "<instructions>" in MAXIMAL_PROFILE_PROMPT
    assert "<business_readme>" in MAXIMAL_PROFILE_PROMPT
    assert "<few_shot_examples>" in MAXIMAL_PROFILE_PROMPT
    assert '<example id="1">' in MAXIMAL_PROFILE_PROMPT
    assert '<example id="2">' in MAXIMAL_PROFILE_PROMPT
    assert 'Cột: "tien_do_sua"' in MAXIMAL_PROFILE_PROMPT
    assert 'Cột: "ngay_kham_hen"' in MAXIMAL_PROFILE_PROMPT
    assert 'Tóm tắt rút gọn: "' in MAXIMAL_PROFILE_PROMPT
    assert "Các cột khác trong bảng:" in MAXIMAL_PROFILE_PROMPT
    assert "<target_column>" in MAXIMAL_PROFILE_PROMPT
    assert "<output>" in MAXIMAL_PROFILE_PROMPT
    assert "{null_ratio" not in MAXIMAL_PROFILE_PROMPT
    assert "lần xuất hiện" not in MAXIMAL_PROFILE_PROMPT
    assert "{top_10}" not in MAXIMAL_PROFILE_PROMPT
    assert "Top giá trị phổ biến" not in MAXIMAL_PROFILE_PROMPT


def test_summarizer_prompt_builders() -> None:
    """Verify prompt formatting methods with realistic column metrics and full business_readme."""
    summarizer = ProfileLLMSummarizer(MockLLMEngine())

    col_stats = {
        "null_count": 10,
        "distinct_count": 50,
        "min_val": "1000",
        "max_val": "50000",
        "data_format": "CURRENCY",
        "currency_subtype": "THOUSAND_SEPARATOR",
        "llm_hint": "Formatted VND currency.",
        "top_10_samples": {"10,000": 20, "20,000": 15},
        "sample_values": ["10,000", "20,000", "30,000"],
    }

    min_p = summarizer.build_minimal_prompt(
        table_name="BanHang",
        col_name="tien_hang",
        col_stats=col_stats,
        sibling_cols=["id", "ngay"],
        total_rows=1000,
        business_readme="Tạp hóa Phương Linh\nQuy đổi: 1 lốc = 6 lon",
    )
    assert 'Bảng: "BanHang"' in min_p
    assert 'Cột: "tien_hang"' in min_p
    assert "CURRENCY" in min_p
    assert "<business_readme>" in min_p
    assert "Tạp hóa Phương Linh\nQuy đổi: 1 lốc = 6 lon" in min_p
    assert "Formatted VND currency." not in min_p

    max_p = summarizer.build_maximal_prompt(
        table_name="BanHang",
        col_name="tien_hang",
        col_stats=col_stats,
        total_rows=1000,
        business_readme="Tạp hóa Phương Linh\nQuy đổi: 1 lốc = 6 lon",
        minimal_profile='Cột "tien_hang" chứa số tiền bán hàng; định dạng số tiền VNĐ.',
        sibling_cols=["id", "ngay"],
    )
    assert 'Bảng: "BanHang"' in max_p
    assert 'Cột: "tien_hang"' in max_p
    assert 'Tóm tắt rút gọn: "Cột "tien_hang" chứa số tiền bán hàng; định dạng số tiền VNĐ."' in max_p
    assert "Các cột khác trong bảng: ['id', 'ngay']" in max_p
    assert "THOUSAND_SEPARATOR" in max_p
    assert "Formatted VND currency." not in max_p
    assert "<business_readme>" in max_p
    assert "Tạp hóa Phương Linh\nQuy đổi: 1 lốc = 6 lon" in max_p
    assert "null_ratio" not in max_p
    assert "lần xuất hiện" not in max_p
    assert "Top giá trị phổ biến" not in max_p
    assert "Giá trị NULL: 10/1000" in max_p


def test_summarizer_empty_database() -> None:
    """Verify summarizer gracefully handles an empty database profile."""
    summarizer = ProfileLLMSummarizer(MockLLMEngine())
    res = summarizer.generate_profiles({"tables": {}, "undocumented_joins": []})
    assert res == {}


def test_summarizer_mock_workflow() -> None:
    """Verify full summarization pipeline on synthetic multi-table database profile."""
    mock_db_profile = {
        "tables": {
            "HoaDon": {
                "row_count": 100,
                "is_view": False,
                "is_unified_view": False,
                "source_tables": [],
                "note": None,
                "columns": {
                    "ma_hd": {
                        "null_count": 0,
                        "distinct_count": 100,
                        "min_val": "HD001",
                        "max_val": "HD100",
                        "min_str_len": 5,
                        "max_str_len": 5,
                        "is_fixed_length": True,
                        "is_free_text": False,
                        "date_subtype": None,
                        "currency_subtype": None,
                        "currency_scale_multiplier": None,
                        "data_format": "STRING",
                        "is_json": False,
                        "llm_hint": "Primary key identifier.",
                        "top_10_samples": {"HD001": 1, "HD002": 1},
                        "sample_values": ["HD001", "HD002", "HD003"],
                    },
                    "ngay_tao": {
                        "null_count": 0,
                        "distinct_count": 50,
                        "min_val": "2025-01-01",
                        "max_val": "2025-02-19",
                        "min_str_len": 10,
                        "max_str_len": 10,
                        "is_fixed_length": True,
                        "is_free_text": False,
                        "date_subtype": "ISO_DATE",
                        "currency_subtype": None,
                        "currency_scale_multiplier": None,
                        "data_format": "DATE",
                        "is_json": False,
                        "llm_hint": "Standard ISO 8601 date.",
                        "top_10_samples": {"2025-01-01": 5, "2025-01-02": 4},
                        "sample_values": ["2025-01-01", "2025-01-02"],
                    },
                },
            }
        },
        "undocumented_joins": [],
    }

    summarizer = ProfileLLMSummarizer(MockLLMEngine(), batch_size=16)
    profiles = summarizer.generate_profiles(
        mock_db_profile,
        business_readme="Bán lẻ vật liệu xây dựng",
    )

    assert "HoaDon.ma_hd" in profiles
    assert "HoaDon.ngay_tao" in profiles

    ma_hd_meta = profiles["HoaDon.ma_hd"]
    assert "minimal_profile" in ma_hd_meta
    assert "maximal_profile" in ma_hd_meta
    assert "raw_samples" in ma_hd_meta
    assert ma_hd_meta["raw_samples"] == ["HD001", "HD002", "HD003"]
    assert ma_hd_meta["table_name"] == "HoaDon"
    assert ma_hd_meta["col_name"] == "ma_hd"
    assert ma_hd_meta["data_format"] == "STRING"
    assert len(ma_hd_meta["minimal_profile"]) > 0
    assert len(ma_hd_meta["maximal_profile"]) > 0
    assert ma_hd_meta["minimal_profile"] in ma_hd_meta["maximal_profile"]
    assert "Top-10 giá trị phổ biến: ['HD001', 'HD002']" in ma_hd_meta["maximal_profile"]

    ngay_meta = profiles["HoaDon.ngay_tao"]
    assert ngay_meta["data_format"] == "DATE"
    assert ngay_meta["date_subtype"] == "ISO_DATE"


def test_summarizer_e2e_with_raw_profiler(tmp_path: Path) -> None:
    """Test integration between RawDatabaseProfiler and ProfileLLMSummarizer."""
    db_file = tmp_path / "test_store.db"
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE SanPham (
            ma_sp TEXT,
            ten_sp TEXT,
            gia_von TEXT,
            ngay_nhap TEXT
        )
    """)
    cur.executemany(
        "INSERT INTO SanPham VALUES (?, ?, ?, ?)",
        [
            ("SP001", "Sting dâu 330ml", "10,000 VNĐ", "2025-05-01"),
            ("SP002", "Mì Hảo Hảo tôm chua cay", "4,500 VNĐ", "2025-05-02"),
            ("SP003", "Bia Tiger 330ml", "16,000 VNĐ", "2025-05-03"),
        ],
    )
    conn.commit()
    conn.close()

    # Step 1: Raw Profiler
    raw_profiler = RawDatabaseProfiler(db_path=db_file)
    raw_profile = raw_profiler.profile_database()

    # Step 2: Summarizer with Mock LLM
    summarizer = ProfileLLMSummarizer(MockLLMEngine())
    processed_metadata = summarizer.generate_profiles(
        db_profile=raw_profile,
        business_readme="Cửa hàng tạp hóa nhỏ tại TP.HCM",
    )

    assert "SanPham.ma_sp" in processed_metadata
    assert "SanPham.ten_sp" in processed_metadata
    assert "SanPham.gia_von" in processed_metadata
    assert "SanPham.ngay_nhap" in processed_metadata

    gia_von_meta = processed_metadata["SanPham.gia_von"]
    assert gia_von_meta["data_format"] == "CURRENCY"
    assert gia_von_meta["currency_subtype"] == "VND_TEXT"
    assert "minimal_profile" in gia_von_meta
    assert "maximal_profile" in gia_von_meta


def test_cli_parsing_and_execution(tmp_path: Path) -> None:
    """Verify CLI argument parser and profile_single_db execution with summarization."""
    from zaic_2026_4.profiling.cli import build_parser, profile_single_db

    parser = build_parser()
    args = parser.parse_args([
        "--db", "fake.db",
        "--summarize",
        "--mock-llm",
    ])
    assert args.summarize is True
    assert args.mock_llm is True
    assert args.local is False

    db_file = tmp_path / "cli_test.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE Test (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO Test VALUES (1, 'Item 1')")
    conn.commit()
    conn.close()

    readme_file = tmp_path / "README.md"
    readme_file.write_text("Test Store README", encoding="utf-8")

    out_json = tmp_path / "cli_out.json"
    args = parser.parse_args([
        "--db", str(db_file),
        "--summarize",
        "--mock-llm",
        "--readme", str(readme_file),
    ])

    result = profile_single_db(db_file, args, output_path=out_json)
    assert "raw_profile" in result
    assert "metadata_profile" in result
    assert "Test.name" in result["metadata_profile"]
    assert out_json.is_file()
