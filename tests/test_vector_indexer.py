"""Unit tests for Module 4: SemanticVectorIndexer and MockEmbedder."""

from pathlib import Path
import json
import numpy as np
import pytest

from zaic_2026_4.profiling.vector_indexer import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_PATH,
    DEFAULT_TOP_K,
    MockEmbedder,
    SemanticVectorIndexer,
)


def test_mock_embedder_dimension_and_normalization() -> None:
    """Verify MockEmbedder generates unit-normalized vectors with correct dimension."""
    dim = 256
    embedder = MockEmbedder(dimension=dim)
    assert embedder.get_sentence_embedding_dimension() == dim

    texts = ["Doanh thu bán hàng", "Mã khách hàng", "Đơn hàng bị huỷ"]
    out = embedder.encode(texts)
    assert isinstance(out, dict)
    assert "dense_vecs" in out

    vecs = out["dense_vecs"]
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (3, dim)
    assert vecs.dtype == np.float32

    # Check unit L2 normalization
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, np.ones(3), rtol=1e-5, atol=1e-5)


def test_mock_embedder_determinism() -> None:
    """Verify identical text produces identical embeddings."""
    embedder = MockEmbedder(dimension=128)
    out1 = embedder.encode(["Sữa tươi Vinamilk"])["dense_vecs"]
    out2 = embedder.encode(["Sữa tươi Vinamilk"])["dense_vecs"]
    out_diff = embedder.encode(["Nước tăng lực Sting"])["dense_vecs"]

    np.testing.assert_array_equal(out1, out2)
    assert not np.array_equal(out1, out_diff)


def test_mock_embedder_empty_input() -> None:
    """Verify MockEmbedder safely handles empty input lists."""
    embedder = MockEmbedder(dimension=64)
    out = embedder.encode([])["dense_vecs"]
    assert out.shape == (0, 64)


def test_indexer_initialization_with_mock() -> None:
    """Verify SemanticVectorIndexer initializes with MockEmbedder correctly."""
    indexer = SemanticVectorIndexer(embedder=MockEmbedder(dimension=128))
    assert indexer.dimension == 128
    assert indexer.index.d == 128
    assert indexer.index.ntotal == 0
    assert len(indexer.doc_map) == 0
    assert indexer.max_length == DEFAULT_MAX_LENGTH


def test_build_index_column_profiles() -> None:
    """Verify indexing database columns from processed_metadata."""
    embedder = MockEmbedder(dimension=128)
    indexer = SemanticVectorIndexer(embedder=embedder)

    metadata = {
        "DonHang.tong_tien": {
            "maximal_profile": "Tổng số tiền thanh toán của hóa đơn bán hàng đã trừ chiết khấu dòng.",
            "minimal_profile": "Tổng tiền thanh toán hóa đơn.",
            "data_format": "CURRENCY",
            "table_name": "DonHang",
            "col_name": "tong_tien",
        },
        "DonHang.trang_thai": {
            "maximal_profile": "Trạng thái đơn hàng (Đã giao, Đã huỷ, Chờ xử lý).",
            "data_format": "STRING",
        },
        "KhachHang.ma_kh": {
            "minimal_profile": "Mã định danh khách hàng duy nhất.",
            "data_format": "STRING",
        },
    }

    doc_count = indexer.build_index(processed_metadata=metadata)
    assert doc_count == 3
    assert indexer.index.ntotal == 3
    assert len(indexer.doc_map) == 3

    assert indexer.doc_map[0]["key"] == "DonHang.tong_tien"
    assert indexer.doc_map[0]["type"] == "column"
    assert "Tổng số tiền thanh toán" in indexer.doc_map[0]["content"]

    # Fallback to minimal_profile when maximal_profile is missing
    assert indexer.doc_map[2]["key"] == "KhachHang.ma_kh"
    assert "Mã định danh khách hàng" in indexer.doc_map[2]["content"]


def test_build_index_rulebook_parsing() -> None:
    """Verify rulebook markdown is parsed into rule clauses."""
    embedder = MockEmbedder(dimension=128)
    indexer = SemanticVectorIndexer(embedder=embedder)

    rulebook_text = """# DataMate Rulebook

Đơn tính vào doanh thu là đơn không huỷ và không trả hàng. Đơn có trả hàng bỏ cả đơn.

Thời gian: Tuần này bắt đầu từ thứ Hai đến hết ngày as_of_date.

Bán chạy nhất: Tính theo tổng số lượng quy về đơn vị cơ bản.
"""

    doc_count = indexer.build_index(processed_metadata={}, rulebook_text=rulebook_text)
    assert doc_count == 4  # Header + 3 rule sections
    assert indexer.index.ntotal == 4

    rules = [d for d in indexer.doc_map if d["type"] == "rulebook"]
    assert len(rules) == 4
    assert any("không huỷ và không trả hàng" in r["content"] for r in rules)
    assert any("thứ Hai" in r["content"] for r in rules)


def test_build_index_readme_parsing() -> None:
    """Verify business README.md is parsed and indexed as context."""
    embedder = MockEmbedder(dimension=128)
    indexer = SemanticVectorIndexer(embedder=embedder)

    readme_text = """# Cửa hàng tạp hóa Cô Ba

Cửa hàng chuyên kinh doanh các mặt hàng tiêu dùng nhanh, bánh kẹo, đồ uống và gia vị.

Bảng DonHang lưu trữ toàn bộ các giao dịch phát sinh từ máy tính tiền POS.

```markdown
ignored code block
```
"""

    doc_count = indexer.build_index(
        processed_metadata={}, rulebook_text="", readme_text=readme_text
    )
    assert doc_count >= 2
    readmes = [d for d in indexer.doc_map if d["type"] == "readme"]
    assert len(readmes) >= 2


def test_search_similar_schema_2d_fix() -> None:
    """Verify FAISS 2D array output unwrap fix and valid score computation."""
    embedder = MockEmbedder(dimension=128)
    indexer = SemanticVectorIndexer(embedder=embedder)

    metadata = {
        "DonHang.doanh_thu": {
            "maximal_profile": "Tổng doanh thu bán hàng sau chiết khấu.",
        },
        "DonHang.ngay_tao": {
            "maximal_profile": "Thời gian tạo đơn hàng định dạng ISO date YYYY-MM-DD.",
        },
        "SanPham.ten_sp": {
            "maximal_profile": "Tên sản phẩm hàng hóa hoặc dịch vụ.",
        },
    }
    rulebook = "Doanh thu là tổng thành tiền các dòng hàng không bị huỷ trả."

    indexer.build_index(processed_metadata=metadata, rulebook_text=rulebook)

    results = indexer.search_similar_schema("doanh thu bán hàng", top_k=2)
    assert len(results) == 2
    for res in results:
        assert "score" in res
        assert isinstance(res["score"], float)
        # Cosine similarity for unit vectors is in [-1.0, 1.0]
        assert -1.05 <= res["score"] <= 1.05


def test_search_padding_index_neg1() -> None:
    """Verify that asking top_k greater than indexed docs does not crash or use -1 padding."""
    embedder = MockEmbedder(dimension=64)
    indexer = SemanticVectorIndexer(embedder=embedder)

    indexer.build_index(
        processed_metadata={
            "Table.Col1": {"maximal_profile": "Description 1"},
        }
    )
    assert indexer.index.ntotal == 1

    # Request top_k = 10 when only 1 item exists
    results = indexer.search_similar_schema("query", top_k=10)
    assert len(results) == 1
    assert results[0]["key"] == "Table.Col1"


def test_search_doc_type_filtering() -> None:
    """Verify search filtering by document type ('column' vs 'rulebook')."""
    embedder = MockEmbedder(dimension=128)
    indexer = SemanticVectorIndexer(embedder=embedder)

    metadata = {
        "DonHang.tien": {"maximal_profile": "Tiền bán hàng hóa."},
        "DonHang.sl": {"maximal_profile": "Số lượng hàng hóa bán ra."},
    }
    rules = "Quy tắc 1: Đơn huỷ không tính vào doanh thu."
    indexer.build_index(processed_metadata=metadata, rulebook_text=rules)

    col_results = indexer.search_columns("tiền bán", top_k=5)
    assert all(r["type"] == "column" for r in col_results)

    rule_results = indexer.search_rules("đơn huỷ", top_k=3)
    assert all(r["type"] == "rulebook" for r in rule_results)


def test_search_empty_index_and_blank_query() -> None:
    """Verify graceful handling when searching empty index or blank queries."""
    indexer = SemanticVectorIndexer(embedder=MockEmbedder(dimension=64))
    assert indexer.search_similar_schema("any query") == []

    indexer.build_index({"T.C": {"maximal_profile": "Desc"}})
    assert indexer.search_similar_schema("") == []
    assert indexer.search_similar_schema("   ") == []


def test_score_no_in_place_mutation() -> None:
    """Verify query score assignment does not mutate original doc_map."""
    indexer = SemanticVectorIndexer(embedder=MockEmbedder(dimension=64))
    indexer.build_index({"T.C": {"maximal_profile": "Sample col"}})

    assert "score" not in indexer.doc_map[0]
    res1 = indexer.search_similar_schema("query 1", top_k=1)
    assert "score" in res1[0]
    # Internal doc_map must remain clean of query-specific score
    assert "score" not in indexer.doc_map[0]


def test_persistence_save_load(tmp_path: Path) -> None:
    """Verify saving FAISS index and doc_map, then loading back and querying."""
    save_dir = tmp_path / "faiss_cache"

    embedder = MockEmbedder(dimension=64)
    indexer = SemanticVectorIndexer(embedder=embedder)
    metadata = {
        "Hoadon.id": {"maximal_profile": "Mã định danh hóa đơn."},
        "Hoadon.ngay": {"maximal_profile": "Ngày lập hóa đơn."},
    }
    rules = "Quy tắc tính doanh thu theo ngày lập hóa đơn."
    indexer.build_index(metadata, rulebook_text=rules)

    # Save to directory
    indexer.save(save_dir)
    assert (save_dir / "index.faiss").is_file()
    assert (save_dir / "doc_map.json").is_file()
    assert (save_dir / "vector_meta.json").is_file()

    # Load back with same embedder
    loaded = SemanticVectorIndexer.load(save_dir, embedder=embedder)
    assert loaded.index.ntotal == indexer.index.ntotal
    assert len(loaded.doc_map) == len(indexer.doc_map)
    assert loaded.dimension == 64

    # Queries on original and loaded index must match exactly
    orig_res = indexer.search_similar_schema("hóa đơn", top_k=2)
    load_res = loaded.search_similar_schema("hóa đơn", top_k=2)
    assert len(orig_res) == len(load_res)
    for r1, r2 in zip(orig_res, load_res):
        assert r1["key"] == r2["key"]
        assert pytest.approx(r1["score"], 1e-4) == r2["score"]
