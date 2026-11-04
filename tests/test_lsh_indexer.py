"""Unit tests for Module 3: VietnameseLSHIndexer, character shingling, and literal matching."""

from pathlib import Path
import unicodedata

import pytest

from zaic_2026_4.profiling.lsh_indexer import (
    DEFAULT_MAX_SAMPLE_VALUES,
    DEFAULT_NUM_PERM,
    DEFAULT_SHINGLE_K,
    DEFAULT_THRESHOLD,
    VietnameseLSHIndexer,
    compute_minhash,
)


def test_default_constants() -> None:
    """Verify default LSH and MinHash configuration parameters."""
    assert DEFAULT_NUM_PERM == 128
    assert DEFAULT_THRESHOLD == 0.4
    assert DEFAULT_SHINGLE_K == 3
    assert DEFAULT_MAX_SAMPLE_VALUES == 10000


def test_indexer_initialization() -> None:
    """Verify VietnameseLSHIndexer initialization and attributes."""
    indexer = VietnameseLSHIndexer()
    assert indexer.threshold == 0.4
    assert indexer.num_perm == 128
    assert indexer.shingle_k == 3
    assert indexer.max_sample_values == 10000
    assert len(indexer.value_to_col_map) == 0
    assert len(indexer.minhashes) == 0

    custom = VietnameseLSHIndexer(
        threshold=0.6, num_perm=64, shingle_k=4, max_sample_values=500
    )
    assert custom.threshold == 0.6
    assert custom.num_perm == 64
    assert custom.shingle_k == 4
    assert custom.max_sample_values == 500


def test_shingle_generation_vietnamese() -> None:
    """Test character k-shingles with dual accented and unaccented Vietnamese representations."""
    indexer = VietnameseLSHIndexer(shingle_k=3)
    shingles = indexer._get_shingles("Sting dâu")

    # Accented shingles
    assert "sti" in shingles
    assert "tin" in shingles
    assert "ing" in shingles
    assert "dâu" in shingles

    # Unaccented shingles generated via unidecode
    assert "dau" in shingles


def test_shingle_nfc_normalization_equivalence() -> None:
    """Verify that NFD decomposed strings produce identical shingles to NFC precomposed strings."""
    indexer = VietnameseLSHIndexer(shingle_k=3)
    nfc_text = "Cà phê sữa đá"
    nfd_text = unicodedata.normalize("NFD", nfc_text)

    # In Python, NFD has separate combining accent codepoints
    assert nfc_text != nfd_text

    shingles_nfc = indexer._get_shingles(nfc_text)
    shingles_nfd = indexer._get_shingles(nfd_text)

    assert shingles_nfc == shingles_nfd
    assert len(shingles_nfc) > 0


def test_shingle_edge_cases_short_and_empty() -> None:
    """Verify shingle generation for short, empty, or whitespace strings."""
    indexer = VietnameseLSHIndexer(shingle_k=3)

    assert indexer._get_shingles("") == set()
    assert indexer._get_shingles("   ") == set()

    # 1-character string
    sh_1 = indexer._get_shingles("a")
    assert sh_1 == {"a"}

    # 2-character string (< k)
    sh_2 = indexer._get_shingles("ca")
    assert sh_2 == {"ca"}

    # 3-character string (== k)
    sh_3 = indexer._get_shingles("cat")
    assert sh_3 == {"cat"}


def test_minhash_dual_matching_similarity() -> None:
    """Verify MinHash Jaccard resemblance between accented and unaccented representations."""
    indexer = VietnameseLSHIndexer(num_perm=128, shingle_k=3)

    m_accented = indexer._create_minhash("Sting dâu 330ml")
    m_no_accent = indexer._create_minhash("sting dau 330ml")

    jaccard_score = m_accented.jaccard(m_no_accent)
    # Dual shingling guarantees very high Jaccard resemblance
    assert jaccard_score >= 0.70


def test_compute_minhash_helper() -> None:
    """Verify the compute_minhash standalone helper."""
    tokens = ["sti", "tin", "ing"]
    m = compute_minhash(tokens, num_perm=64)
    assert m.num_perm == 64

    # Empty iterable
    empty_m = compute_minhash([], num_perm=64)
    assert empty_m.num_perm == 64


def test_build_index_processed_metadata() -> None:
    """Verify build_index extracts and indexes sample values from processed_metadata format."""
    indexer = VietnameseLSHIndexer(threshold=0.5, num_perm=128)

    sample_meta = {
        "SanPham.ten_sp": {
            "minimal_profile": "Tên sản phẩm",
            "raw_samples": [
                "Sting dâu 330ml",
                "Cà phê sữa đá Sài Gòn",
                "Bạc xỉu",
                "-",  # Should be skipped (< 2 chars)
                " ",  # Should be skipped
                None,  # Should be skipped
                "Sting dâu 330ml",  # Duplicate in same column, should be deduplicated
            ],
        },
        "KhachHang.ten_kh": {
            "minimal_profile": "Tên khách hàng",
            "raw_samples": [
                "Nguyễn Văn An",
                "Trần Thị Mai",
                "Chị Lan",
            ],
        },
    }

    count = indexer.build_index(sample_meta)
    # SanPham has 3 valid unique items, KhachHang has 3 valid items = 6 items
    assert count == 6
    assert len(indexer.value_to_col_map) == 6
    assert len(indexer.minhashes) == 6


def test_build_index_raw_profile_format() -> None:
    """Verify build_index supports raw_profile nested table structure."""
    indexer = VietnameseLSHIndexer(threshold=0.5, num_perm=128)

    raw_profile = {
        "tables": {
            "HoaDon": {
                "row_count": 100,
                "columns": {
                    "ma_hd": {
                        "sample_values": ["HD0001", "HD0002", "HD0003"],
                    },
                    "ghi_chu": {
                        "sample_values": ["Giao hàng nhanh", "Đã thanh toán"],
                    },
                },
            }
        }
    }

    count = indexer.build_index(raw_profile)
    assert count == 5
    assert len(indexer.value_to_col_map) == 5


def test_query_literal_exact_and_fuzzy() -> None:
    """Verify literal query matches with exact, diacritic-free, and colloquial questions."""
    indexer = VietnameseLSHIndexer(threshold=0.4, num_perm=128)

    sample_meta = {
        "SanPham.ten_sp": {
            "raw_samples": [
                "Sting dâu 330ml",
                "Xi măng Hà Tiên PCB40 bao 50kg",
                "Trà xanh Không Độ 455ml",
                "Bia Heineken lon 330ml",
            ]
        },
        "KhachHang.ten_kh": {
            "raw_samples": [
                "Nguyễn Văn Tuấn",
                "Lê Thị Hồng Vân",
            ]
        },
    }
    indexer.build_index(sample_meta)

    # 1. Exact query
    res_exact = indexer.query_literal("Xi măng Hà Tiên PCB40 bao 50kg")
    assert len(res_exact) > 0
    assert res_exact[0] == ("SanPham.ten_sp", "Xi măng Hà Tiên PCB40 bao 50kg")

    # 2. Diacritic-free query
    res_no_accent = indexer.query_literal("sting dau 330ml")
    assert len(res_no_accent) > 0
    assert res_no_accent[0][0] == "SanPham.ten_sp"
    assert "Sting dâu" in res_no_accent[0][1]

    # 3. Substring / colloquial query
    res_sub = indexer.query_literal("Xi măng Hà Tiên")
    assert len(res_sub) > 0
    assert res_sub[0] == ("SanPham.ten_sp", "Xi măng Hà Tiên PCB40 bao 50kg")

    # 4. Customer name query without accents
    res_kh = indexer.query_literal("nguyen van tuan")
    assert len(res_kh) > 0
    assert res_kh[0] == ("KhachHang.ten_kh", "Nguyễn Văn Tuấn")


def test_query_literal_edge_cases_and_limits() -> None:
    """Verify query limits, empty strings, and non-matching literals."""
    indexer = VietnameseLSHIndexer(threshold=0.5, num_perm=128)
    sample_meta = {
        "SanPham.ten_sp": {
            "raw_samples": ["Coca Cola 320ml", "Pepsi Cola 320ml", "7Up 320ml"]
        }
    }
    indexer.build_index(sample_meta)

    # Empty and short queries return empty list
    assert indexer.query_literal("") == []
    assert indexer.query_literal(" ") == []
    assert indexer.query_literal("a") == []

    # Irrelevant query returns empty list
    assert indexer.query_literal("Hoàn toàn không liên quan zzz123") == []

    # top_k limit
    res = indexer.query_literal("Cola 320ml", top_k=1)
    assert len(res) <= 1


def test_query_literal_with_scores_ranking() -> None:
    """Verify query_literal_with_scores returns matches sorted by similarity score descending."""
    indexer = VietnameseLSHIndexer(threshold=0.3, num_perm=128)
    sample_meta = {
        "SanPham.ten_sp": {
            "raw_samples": [
                "Bia Tiger lon 330ml",
                "Bia Tiger Crystal lon 330ml",
                "Bia Larue lon 330ml",
            ]
        }
    }
    indexer.build_index(sample_meta)

    scored = indexer.query_literal_with_scores("Bia Tiger lon 330ml")
    assert len(scored) > 0
    # First match should be the exact match with highest score
    assert scored[0][1] == "Bia Tiger lon 330ml"
    assert scored[0][2] >= scored[1][2]

    # Scores should be within [0.0, 1.0]
    for _, _, sc in scored:
        assert 0.0 <= sc <= 1.0


def test_save_and_load_serialization(tmp_path: Path) -> None:
    """Verify pickle serialization roundtrip produces identical query results."""
    indexer = VietnameseLSHIndexer(threshold=0.4, num_perm=128)
    sample_meta = {
        "SanPham.ten_sp": {
            "raw_samples": ["Mì Hảo Hảo tôm chua cay", "Mì Omachi sườn hầm"]
        }
    }
    indexer.build_index(sample_meta)

    save_file = tmp_path / "cache_lsh" / "test_indexer.pkl"
    indexer.save(save_file)
    assert save_file.is_file()

    loaded = VietnameseLSHIndexer.load(save_file)
    assert loaded.threshold == indexer.threshold
    assert loaded.num_perm == indexer.num_perm
    assert len(loaded.value_to_col_map) == len(indexer.value_to_col_map)

    # Query results should match
    res_orig = indexer.query_literal("hao hao tom chua cay")
    res_load = loaded.query_literal("hao hao tom chua cay")
    assert res_orig == res_load
    assert len(res_orig) > 0

    # Loading non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        VietnameseLSHIndexer.load(tmp_path / "does_not_exist.pkl")
