"""Unit tests for Module 2: ProfileLLMSummarizer, prompt templates, and engine factory."""

import sqlite3
from pathlib import Path

import pytest

from zaic_2026_4.profiling.profile_summarizer import (
    DEFAULT_LOCAL_GPU_UTIL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_PROD_GPU_UTIL,
    DEFAULT_PROD_MODEL,
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


def test_summarizer_prompt_builders() -> None:
    """Verify minimal prompt formatting methods with realistic column metrics and full business_readme."""
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
    assert "sample_values" in ma_hd_meta
    assert ma_hd_meta["sample_values"] == ["HD001", "HD002", "HD003"]
    assert ma_hd_meta["table_name"] == "HoaDon"
    assert ma_hd_meta["col_name"] == "ma_hd"
    assert ma_hd_meta["data_format"] == "STRING"
    assert len(ma_hd_meta["minimal_profile"]) > 0
    assert len(ma_hd_meta["maximal_profile"]) > 0
    assert ma_hd_meta["minimal_profile"] in ma_hd_meta["maximal_profile"]
    assert "Giá trị phổ biến: 'HD001', 'HD002'" in ma_hd_meta["maximal_profile"]
    assert "min_val" in ma_hd_meta
    assert "max_val" in ma_hd_meta
    assert "is_unique" in ma_hd_meta
    assert "join_candidates" in ma_hd_meta

    ngay_meta = profiles["HoaDon.ngay_tao"]
    assert ngay_meta["data_format"] == "DATE"
    assert ngay_meta["date_subtype"] == "ISO_DATE"


def test_summarizer_calls_llm_only_for_minimal() -> None:
    """Verify that ProfileLLMSummarizer only invokes the LLM engine for minimal prompts (0 maximal calls)."""
    class TrackingMockEngine(MockLLMEngine):
        def __init__(self) -> None:
            super().__init__()
            self.call_count = 0
            self.total_prompts = 0

        def generate(self, prompts, **kwargs):
            p_list = [prompts] if isinstance(prompts, str) else prompts
            self.call_count += 1
            self.total_prompts += len(p_list)
            return super().generate(prompts, **kwargs)

    tracking_engine = TrackingMockEngine()
    summarizer = ProfileLLMSummarizer(tracking_engine, batch_size=64)

    dummy_db = {
        "tables": {
            "T1": {
                "row_count": 10,
                "columns": {
                    "c1": {"data_format": "STRING", "sample_values": ["A", "B"]},
                    "c2": {"data_format": "NUMERIC", "sample_values": ["1", "2"]},
                },
            },
            "T2": {
                "row_count": 5,
                "columns": {
                    "c3": {"data_format": "STRING", "sample_values": ["X"]},
                },
            },
        },
        "undocumented_joins": [],
    }

    profiles = summarizer.generate_profiles(dummy_db)
    assert len(profiles) == 3
    # Exactly 3 minimal prompts submitted across 1 batch call, 0 maximal calls
    assert tracking_engine.call_count == 1
    assert tracking_engine.total_prompts == 3


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
    """Verify CLI argument parser and execution for zaic-profile."""
    from zaic_2026_4.profiling.cli import (
        _resolve_biz_dir,
        build_parser,
        profile_single_db,
    )

    parser = build_parser()
    args = parser.parse_args([
        "--split", "dataset/train",
        "--biz-id", "101",
        "--output-dir", str(tmp_path / "out"),
        "--mock",
    ])
    assert args.split == "dataset/train"
    assert args.biz_id == "101"
    assert args.output_dir == str(tmp_path / "out")
    assert args.mock is True
    assert args.local is False

    # Verify biz-id resolution supporting both numeric ID (101) and prefixed name (biz_101)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "biz_101").mkdir()

    assert _resolve_biz_dir(data_dir, "101") == (data_dir / "biz_101").resolve()
    assert _resolve_biz_dir(data_dir, "biz_101") == (data_dir / "biz_101").resolve()

    db_file = tmp_path / "cli_test.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE Test (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO Test VALUES (1, 'Item 1')")
    conn.commit()
    conn.close()

    readme_file = tmp_path / "README.md"
    readme_file.write_text("Test Store README", encoding="utf-8")

    out_json = tmp_path / "cli_out.json"
    result = profile_single_db(db_file, args, output_path=out_json)
    assert "raw_profile" in result
    assert "metadata_profile" in result
    assert "Test.name" in result["metadata_profile"]
    assert out_json.is_file()


def test_engine_factory_quantization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify create_engine accepts quantize parameter and forwards to VLLMEngine."""
    from unittest.mock import MagicMock

    import vllm

    from zaic_2026_4.profiling.profile_summarizer import VLLMEngine

    fake_llm = MagicMock()
    monkeypatch.setattr(vllm, "LLM", fake_llm)

    # 1. Default quantization is None
    engine_default = create_engine(mock=False)
    assert isinstance(engine_default, VLLMEngine)
    assert engine_default.quantization is None
    fake_llm.assert_called_with(
        model=engine_default.model_name,
        gpu_memory_utilization=engine_default.gpu_memory_utilization,
        max_model_len=engine_default.max_model_len,
        tensor_parallel_size=engine_default.tensor_parallel_size,
        trust_remote_code=True,
        dtype="auto",
        quantization=None,
        reasoning_config=engine_default.reasoning_config,
    )

    # 2. Explicit quantization e.g. fp8
    engine_fp8 = create_engine(mock=False, quantize="fp8")
    assert isinstance(engine_fp8, VLLMEngine)
    assert engine_fp8.quantization == "fp8"

