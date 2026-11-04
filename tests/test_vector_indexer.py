"""Unit tests for Module 4: SemanticVectorIndexer, MockEmbedder, and Hierarchical Chunking."""

from pathlib import Path

import numpy as np
import pytest

from zaic_2026_4.profiling.vector_indexer import (
    DEFAULT_MAX_LENGTH,
    MockEmbedder,
    SemanticVectorIndexer,
    parse_markdown_hierarchical,
    recursive_split_paragraph,
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


def test_recursive_split_paragraph_under_threshold() -> None:
    """Verify text <= max_chars is preserved as a single chunk without splitting."""
    text = "Đơn tính vào doanh thu là đơn không huỷ và không trả hàng."
    chunks = recursive_split_paragraph(text, max_chars=1000)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_recursive_split_paragraph_delimiters_priority() -> None:
    """Verify recursive splitting respects priority delimiters and never cuts words in half."""
    # Create long text exceeding 1000 chars composed of distinct sentences
    sentence = "Đây là một câu hoàn chỉnh về nghiệp vụ bán lẻ của cửa hàng tạp hóa. "
    long_text = sentence * 25  # ~1725 chars
    chunks = recursive_split_paragraph(long_text, max_chars=1000)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 1000
        # Ensure sentence boundary was used and no word was truncated mid-letter
        assert chunk.endswith(".")


def test_parse_markdown_hierarchical_table_threshold_small() -> None:
    """Verify tables with <= 3 rows are NOT split row-by-row and kept as 1 block."""
    table_text = """## Thông tin chi nhánh
| Mã CN | Tên chi nhánh |
|---|---|
| CN01 | Quận 1 |
| CN02 | Quận 3 |
"""
    # Total rows = 1 header + 2 data rows = 3 rows (<= 3)
    chunks = parse_markdown_hierarchical(
        table_text, min_chunk_chars=50, min_table_rows_for_split=3
    )
    assert len(chunks) == 1
    assert "| Mã CN | Tên chi nhánh |" in chunks[0]
    assert "| CN01 | Quận 1 |" in chunks[0]
    assert "| CN02 | Quận 3 |" in chunks[0]


def test_parse_markdown_hierarchical_table_threshold_large() -> None:
    """Verify tables with > 3 rows ARE decomposed into individual row chunks."""
    table_text = """## 1. Đơn hàng và doanh thu
| Luật | Mặc định |
|---|---|
| Đơn tính vào doanh thu | đơn không huỷ và không trả hàng |
| Bán được bao nhiêu tiền | là doanh thu theo mục này |
| Đơn bị huỷ | đơn có trạng thái huỷ |
"""
    # Total rows = 1 header + 3 data rows = 4 rows (> 3)
    chunks = parse_markdown_hierarchical(table_text, min_table_rows_for_split=3)
    assert len(chunks) == 3
    assert chunks[0].startswith("[Mục: 1. Đơn hàng và doanh thu]")
    assert "Luật: Đơn tính vào doanh thu -> Mặc định: đơn không huỷ và không trả hàng" in chunks[0]
    assert "Luật: Bán được bao nhiêu tiền -> Mặc định: là doanh thu theo mục này" in chunks[1]
    assert "Luật: Đơn bị huỷ -> Mặc định: đơn có trạng thái huỷ" in chunks[2]


def test_h1_excluded_from_heading_context() -> None:
    """Verify H1 document title is omitted from heading_ctx to prevent semantic dilution."""
    text = """# DataMate — Rulebook (bộ train)

## 1. Đơn hàng và doanh thu
| Luật | Mặc định |
|---|---|
| Đơn huỷ | không tính doanh thu |
| Trả hàng | trừ tiền hàng |
| Đơn nợ | ghi sổ công nợ |
| Giảm giá | trừ từng dòng |
"""
    chunks = parse_markdown_hierarchical(text, min_table_rows_for_split=3)
    assert len(chunks) == 4
    for c in chunks:
        # Must start with H2, not H1
        assert c.startswith("[Mục: 1. Đơn hàng và doanh thu]")
        assert "DataMate — Rulebook" not in c


def test_coalescing_buffer_short_paragraphs() -> None:
    """Verify short paragraphs are accumulated and emitted once min_chunk_chars (150) is reached."""
    markdown_sample = """# Dữ liệu biz_126

Sổ bán hàng chủ tiệm tự ghi trên Excel, một sheet: mỗi dòng một mặt hàng khách mua, không ghi mã đơn, không ghi tên khách.

## File
- `SoBanHang.xlsx` (xlsx)
"""
    chunks = parse_markdown_hierarchical(markdown_sample, min_chunk_chars=150)
    # The snippet has ~150 chars total across all lines; should be coalesced without losing SoBanHang.xlsx
    assert any("SoBanHang.xlsx" in c for c in chunks)
    for c in chunks:
        assert len(c) >= 30  # No empty or single-token micro-chunks emitted


def test_coalescing_buffer_flush_on_major_heading() -> None:
    """Verify pending buffer is flushed when an H1 or H2 heading arrives."""
    text = """## Mục A
Đoạn ngắn A.

## Mục B
Đoạn ngắn B.
"""
    chunks = parse_markdown_hierarchical(text, min_chunk_chars=150)
    # Both short paragraphs must be emitted rather than lost due to H2 flush guard
    assert len(chunks) == 2
    assert "Đoạn ngắn A." in chunks[0]
    assert "Đoạn ngắn B." in chunks[1]


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


def test_build_index_rulebook_table_parsing() -> None:
    """Verify rulebook markdown table with > 3 rows is parsed into atomic rule chunks."""
    embedder = MockEmbedder(dimension=128)
    indexer = SemanticVectorIndexer(embedder=embedder)

    rulebook_text = """## 1. Đơn hàng và doanh thu
| Luật | Mặc định |
|---|---|
| Đơn tính vào doanh thu | đơn không huỷ và không trả hàng. Đơn có trả hàng bỏ cả đơn |
| Bán được bao nhiêu tiền | là doanh thu theo mục này, không phải tiền khách đã trả |
| Đơn bị huỷ | đơn có trạng thái huỷ. Đơn trả hàng không phải đơn huỷ |
| Số đơn | chỉ tính đơn không huỷ, không trả |
"""

    doc_count = indexer.build_index(processed_metadata={}, rulebook_text=rulebook_text)
    assert doc_count == 4
    assert indexer.index.ntotal == 4

    rules = [d for d in indexer.doc_map if d["type"] == "rulebook"]
    assert len(rules) == 4
    assert any("không huỷ và không trả hàng" in r["content"] for r in rules)
    assert any("Đơn bị huỷ" in r["content"] for r in rules)


def test_build_index_readme_parsing() -> None:
    """Verify business README.md is parsed and indexed as context."""
    embedder = MockEmbedder(dimension=128)
    indexer = SemanticVectorIndexer(embedder=embedder)

    readme_text = """# Cửa hàng tạp hóa Cô Ba

Cửa hàng chuyên kinh doanh các mặt hàng tiêu dùng nhanh, bánh kẹo, đồ uống và gia vị cho cư dân khu vực lân cận.

Bảng DonHang lưu trữ toàn bộ các giao dịch phát sinh từ máy tính tiền POS hàng ngày của cửa hàng.

```markdown
ignored code block
```
"""

    doc_count = indexer.build_index(
        processed_metadata={}, rulebook_text="", readme_text=readme_text
    )
    assert doc_count >= 1
    readmes = [d for d in indexer.doc_map if d["type"] == "readme"]
    assert len(readmes) >= 1
    assert any("tiêu dùng nhanh" in r["content"] for r in readmes)


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
    rulebook = "Doanh thu là tổng thành tiền các dòng hàng không bị huỷ trả phát sinh trong kỳ của cửa hàng."

    indexer.build_index(processed_metadata=metadata, rulebook_text=rulebook)

    results = indexer.search_similar_schema("doanh thu bán hàng", top_k=2)
    assert len(results) == 2
    for res in results:
        assert "score" in res
        assert isinstance(res["score"], float)
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
    rules = """## 1. Doanh thu
| Luật | Mặc định |
|---|---|
| Đơn huỷ | không tính vào doanh thu |
| Trả hàng | bỏ toàn bộ đơn |
| Đơn nợ | ghi nhận công nợ |
| Giảm giá | trừ trên từng dòng |
"""
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
    rules = "Quy tắc tính doanh thu theo ngày lập hóa đơn của cửa hàng bán lẻ phát sinh trong kỳ kế toán."
    indexer.build_index(metadata, rulebook_text=rules)

    indexer.save(save_dir)
    assert (save_dir / "index.faiss").is_file()
    assert (save_dir / "doc_map.json").is_file()
    assert (save_dir / "vector_meta.json").is_file()

    loaded = SemanticVectorIndexer.load(save_dir, embedder=embedder)
    assert loaded.index.ntotal == indexer.index.ntotal
    assert len(loaded.doc_map) == len(indexer.doc_map)
    assert loaded.dimension == 64

    orig_res = indexer.search_similar_schema("hóa đơn", top_k=2)
    load_res = loaded.search_similar_schema("hóa đơn", top_k=2)
    assert len(orig_res) == len(load_res)
    for r1, r2 in zip(orig_res, load_res):
        assert r1["key"] == r2["key"]
        assert pytest.approx(r1["score"], 1e-4) == r2["score"]


def test_indexer_with_vllm_embed_interface() -> None:
    """Verify SemanticVectorIndexer works with vLLM-style embedder producing outputs.embedding."""
    class MockVllmOutput:
        def __init__(self, vec: list[float]) -> None:
            class Inner:
                pass
            self.outputs = Inner()
            self.outputs.embedding = vec

    class MockVllmEmbedder:
        def __init__(self, dim: int = 32) -> None:
            self.dim = dim

        def embed(self, texts: list[str]) -> list[MockVllmOutput]:
            results = []
            for t in texts:
                # Deterministic pseudo vector
                val = float(len(t))
                vec = [val] * self.dim
                results.append(MockVllmOutput(vec))
            return results

    embedder = MockVllmEmbedder(dim=32)
    indexer = SemanticVectorIndexer(embedder=embedder)
    assert indexer.dimension == 32

    metadata = {
        "orders.amount": {"maximal_profile": "Order total amount in VND"},
    }
    doc_count = indexer.build_index(metadata)
    assert doc_count >= 1
    assert indexer.index.ntotal == doc_count

    hits = indexer.search_columns("amount", top_k=1)
    assert len(hits) == 1
    assert hits[0]["key"] == "orders.amount"
    assert "score" in hits[0]
