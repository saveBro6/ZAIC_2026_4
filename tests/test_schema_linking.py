"""Unit tests for Task-Aligned Schema Linking.

Verifies:
  - Checklist 1: Context builder & 5 variants (Cases a to e)
  - Checklist 2: SQL AST Parser (aliases, fields, literals, fallback)
  - Checklist 3: LSH lookup & re-prompting loop on literal mismatch
  - Checklist 4: Paper benchmark (Fresno County Office of Education example)
  - Checklist 5: Union / Fusion & Key preservation (PK, FK, undocumented joins)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from zaic_2026_4.profiling.lsh_indexer import VietnameseLSHIndexer
from zaic_2026_4.profiling.profile_summarizer import MockLLMEngine
from zaic_2026_4.profiling.vector_indexer import MockEmbedder, SemanticVectorIndexer
from zaic_2026_4.schema_linking.ast_parser import SQLASTExtractor
from zaic_2026_4.schema_linking.context_builder import (
    SchemaContextBuilder,
)
from zaic_2026_4.schema_linking.task_aligned_linker import (
    TaskAlignedSchemaLinker,
    task_aligned_schema_linking,
)


@pytest.fixture
def ast_extractor() -> SQLASTExtractor:
    return SQLASTExtractor()


@pytest.fixture
def sample_metadata() -> dict[str, Any]:
    return {
        "schools.CDSCode": {
            "minimal_profile": "Mã trường CDSCode duy nhất.",
            "maximal_profile": "Cột mã trường CDSCode đóng vai trò khóa chính trong bảng schools.",
            "data_format": "STRING",
            "raw_samples": ["01001", "01002", "01003"],
        },
        "schools.Zip": {
            "minimal_profile": "Mã bưu chính Zip của trường.",
            "maximal_profile": "Mã bưu chính của trường phục vụ phân vùng địa lý.",
            "data_format": "STRING",
            "raw_samples": ["93701", "93702", "93720"],
        },
        "schools.District": {
            "minimal_profile": "Tên quận/học khu của trường.",
            "maximal_profile": "Tên học khu mà trường trực thuộc.",
            "data_format": "STRING",
            "raw_samples": ["Fresno County Office of Education", "Clovis Unified"],
        },
        "frpm.CDSCode": {
            "minimal_profile": "Mã trường liên kết frpm.",
            "maximal_profile": "Mã trường dùng để join với bảng schools.",
            "data_format": "STRING",
            "raw_samples": ["01001", "01002", "01003"],
        },
        "frpm.Charter School (Y/N)": {
            "minimal_profile": "Cờ trường bán công (1/0).",
            "maximal_profile": "Cờ nhị phân phân loại trường bán công.",
            "data_format": "INTEGER",
            "raw_samples": ["1", "0"],
        },
        "frpm.County Name": {
            "minimal_profile": "Tên hạt.",
            "maximal_profile": "Tên hạt cấp tỉnh bang.",
            "data_format": "STRING",
            "raw_samples": ["Fresno", "Madera", "Kern"],
        },
        "frpm.District Name": {
            "minimal_profile": "Tên học khu trong bảng frpm.",
            "maximal_profile": "Tên học khu phân bổ nguồn ngân sách.",
            "data_format": "STRING",
            "raw_samples": ["Fresno County Office of Education", "Central Unified"],
        },
    }


@pytest.fixture
def sample_raw_profile() -> dict[str, Any]:
    return {
        "tables": {
            "schools": {
                "columns": {
                    "CDSCode": {},
                    "Zip": {},
                    "District": {},
                }
            },
            "frpm": {
                "columns": {
                    "CDSCode": {},
                    "Charter School (Y/N)": {},
                    "County Name": {},
                    "District Name": {},
                }
            },
        },
        "undocumented_joins": [
            {
                "col1": "frpm.CDSCode",
                "col2": "schools.CDSCode",
                "resemblance": 0.95,
            }
        ],
    }


@pytest.fixture
def sample_lsh_indexer(sample_metadata: dict[str, Any]) -> VietnameseLSHIndexer:
    indexer = VietnameseLSHIndexer(threshold=0.4)
    indexer.build_index(sample_metadata)
    return indexer


@pytest.fixture
def sample_vector_indexer(sample_metadata: dict[str, Any]) -> SemanticVectorIndexer:
    embedder = MockEmbedder(dimension=64)
    indexer = SemanticVectorIndexer(embedder=embedder)
    rulebook = (
        "## Quy ước học khu\nMọi phép tính trường học ưu tiên phân nhóm theo District.\n\n"
        "## Báo cáo ngân sách\nChỉ tổng hợp số liệu cho các trường có gắn mã CDSCode hợp lệ."
    )
    readme = (
        "# Dữ liệu trường học Fresno\nBảng schools lưu thông tin địa chỉ và zip code. "
        "Bảng frpm lưu số liệu tài chính."
    )
    indexer.build_index(
        processed_metadata=sample_metadata,
        rulebook_text=rulebook,
        readme_text=readme,
    )
    return indexer


# ---------------------------------------------------------------------------
# Test Checklist 2: SQL AST Parser
# ---------------------------------------------------------------------------


def test_ast_parser_fresno_query(ast_extractor: SQLASTExtractor) -> None:
    """Verify AST parsing on the exact Fresno query from the research paper."""
    sql = """
    SELECT T2.Zip FROM frpm AS T1
    INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode
    WHERE T1.`Charter School (Y/N)` = 1 AND T1.`County Name` = 'Fresno County Office of Education'
    """
    fields, lits = ast_extractor.parse_fields_and_literals(sql)

    assert "schools.Zip" in fields
    assert "schools.CDSCode" in fields
    assert "frpm.CDSCode" in fields
    assert "frpm.Charter School (Y/N)" in fields
    assert "frpm.County Name" in fields
    assert "Fresno County Office of Education" in lits
    assert "1" in lits


def test_ast_parser_vietnamese_identifiers(ast_extractor: SQLASTExtractor) -> None:
    """Verify parsing Vietnamese quoted table and column names."""
    sql = """
    SELECT T1."Tên sản phẩm", SUM(T1."Số lượng") AS tong_sl
    FROM "Chi tiết đơn hàng" AS T1
    JOIN "Đơn hàng" AS T2 ON T1."Mã đơn" = T2."id"
    WHERE T2."Trạng thái" = 'Đã hoàn thành' AND T1."Đơn giá" >= 50000
    GROUP BY T1."Tên sản phẩm"
    ORDER BY tong_sl DESC
    """
    fields, lits = ast_extractor.parse_fields_and_literals(sql)

    assert "Chi tiết đơn hàng.Tên sản phẩm" in fields
    assert "Chi tiết đơn hàng.Số lượng" in fields
    assert "Chi tiết đơn hàng.Mã đơn" in fields
    assert "Chi tiết đơn hàng.Đơn giá" in fields
    assert "Đơn hàng.id" in fields
    assert "Đơn hàng.Trạng thái" in fields
    assert "Đã hoàn thành" in lits
    assert "50000" in lits


def test_ast_parser_unqualified_column_resolution(ast_extractor: SQLASTExtractor) -> None:
    """Verify resolving unqualified column names to their parent table."""
    sql = "SELECT Zip FROM schools WHERE District = 'Fresno'"
    known = {"schools.Zip", "schools.District", "frpm.CDSCode"}
    fields, lits = ast_extractor.parse_fields_and_literals(sql, known_columns=known)

    assert "schools.Zip" in fields
    assert "schools.District" in fields
    assert "Fresno" in lits


def test_ast_parser_clean_sql(ast_extractor: SQLASTExtractor) -> None:
    """Verify stripping markdown code fences, comments, echoed questions, and semicolons."""
    raw = "```sql\n-- Calculate total\nSELECT col1 FROM table1;\n```"
    cleaned = ast_extractor.clean_sql(raw)
    assert cleaned == "SELECT col1 FROM table1"

    # Echoed user question before SELECT query
    echoed_question = (
        "Thống kê tổng tiền bán của mặt hàng Thuốc trong tháng 02/2026 là bao nhiêu?\n"
        'SELECT SUM(T1."Thanh tien") FROM "ban_hang" AS T1;'
    )
    assert ast_extractor.clean_sql(echoed_question) == 'SELECT SUM(T1."Thanh tien") FROM "ban_hang" AS T1'


def test_ast_parser_fallback_on_malformed_sql(ast_extractor: SQLASTExtractor) -> None:
    """Verify fallback parser recovers fields and literals on malformed SQL."""
    malformed = "SELECT FROM WHERE T1.County = 'Fresno' AND"
    fields, lits = ast_extractor.parse_fields_and_literals(malformed)
    assert "T1.County" in fields
    assert "Fresno" in lits


# ---------------------------------------------------------------------------
# Test Checklist 1: Context Builder & 5 Variants
# ---------------------------------------------------------------------------


def test_context_builder_focused_schema(
    sample_metadata: dict[str, Any],
    sample_lsh_indexer: VietnameseLSHIndexer,
    sample_vector_indexer: SemanticVectorIndexer,
) -> None:
    """Verify focused schema retrieval from FAISS and LSH."""
    builder = SchemaContextBuilder()
    question = "List the zip code of schools in Fresno County Office of Education"

    focused = builder.get_focused_schema(
        question=question,
        lsh_indexer=sample_lsh_indexer,
        vector_indexer=sample_vector_indexer,
    )

    assert len(focused) > 0
    # Must identify district or schools related columns
    assert any("District" in col or "schools" in col for col in focused)


def test_context_builder_5_variants(
    sample_metadata: dict[str, Any],
    sample_lsh_indexer: VietnameseLSHIndexer,
    sample_vector_indexer: SemanticVectorIndexer,
) -> None:
    """Verify building all 5 context variants (Cases a to e)."""
    builder = SchemaContextBuilder()
    question = "List the zip code of all charter schools in Fresno"

    variants = builder.build_5_variants(
        question=question,
        processed_metadata=sample_metadata,
        lsh_indexer=sample_lsh_indexer,
        vector_indexer=sample_vector_indexer,
    )

    assert len(variants) == 5
    case_names = [v.case_name for v in variants]
    assert case_names == ["case_a", "case_b", "case_c", "case_d", "case_e"]
    for v in variants:
        assert "<few_shot_examples>" in v.formatted_prompt
        assert 'SELECT SUM(T1."Thành tiền")' in v.formatted_prompt

    # Case a: Focused + Minimal
    assert variants[0].profile_type == "minimal_profile"

    # Case b: Focused + Maximal
    assert variants[1].profile_type == "maximal_profile"

    # Case c: Full + Minimal
    assert variants[2].profile_type == "minimal_profile"
    assert len(variants[2].schema_cols) == len(sample_metadata)

    # Case d: Full + Maximal
    assert variants[3].profile_type == "maximal_profile"
    assert len(variants[3].schema_cols) == len(sample_metadata)

    # Case e: Focused + Full (with prompt-level SME & Rulebook context)
    assert variants[4].profile_type == "full_profile"
    assert "<business_rules>" in variants[4].formatted_prompt
    assert "<business_readme>" in variants[4].formatted_prompt


def test_parse_case_spec() -> None:
    """Verify parsing and validation of case specification strings and collections."""
    from zaic_2026_4.schema_linking.context_builder import parse_case_spec

    assert parse_case_spec(None) == ["a", "b", "c", "d", "e"]
    assert parse_case_spec("") == ["a", "b", "c", "d", "e"]
    assert parse_case_spec("abcde") == ["a", "b", "c", "d", "e"]
    assert parse_case_spec("abce") == ["a", "b", "c", "e"]
    assert parse_case_spec("case_a, case_b, case_e") == ["a", "b", "e"]
    assert parse_case_spec("a;c;e") == ["a", "c", "e"]
    assert parse_case_spec(["a", "b", "e"]) == ["a", "b", "e"]
    assert parse_case_spec("ABCE") == ["a", "b", "c", "e"]

    # Deduplication and order preservation
    assert parse_case_spec("aabcee") == ["a", "b", "c", "e"]

    # Invalid input
    import pytest
    with pytest.raises(ValueError, match="Invalid cases specification"):
        parse_case_spec("xyz")


def test_context_builder_subset_variants(
    sample_metadata: dict[str, Any],
    sample_lsh_indexer: VietnameseLSHIndexer,
    sample_vector_indexer: SemanticVectorIndexer,
) -> None:
    """Verify building only a specified subset of variants (e.g. abce)."""
    builder = SchemaContextBuilder()
    question = "List the zip code of all charter schools in Fresno"

    variants = builder.build_5_variants(
        question=question,
        processed_metadata=sample_metadata,
        lsh_indexer=sample_lsh_indexer,
        vector_indexer=sample_vector_indexer,
        cases="abce",
    )

    assert len(variants) == 4
    case_names = [v.case_name for v in variants]
    assert case_names == ["case_a", "case_b", "case_c", "case_e"]
    assert "case_d" not in case_names


# ---------------------------------------------------------------------------
# Test Checklist 3 & 4: Paper Benchmark & Re-prompting Loop (Fresno Example)
# ---------------------------------------------------------------------------


def test_task_aligned_schema_linking_fresno_benchmark(
    sample_metadata: dict[str, Any],
    sample_lsh_indexer: VietnameseLSHIndexer,
    sample_vector_indexer: SemanticVectorIndexer,
    sample_raw_profile: dict[str, Any],
) -> None:
    """Verify the exact Fresno County Office of Education benchmark from the paper.

    Initial SQL uses frpm.`County Name` = 'Fresno County Office of Education'.
    LSH lookup detects that 'Fresno County Office of Education' is in schools.District.
    Re-prompting instructs LLM to use schools.District.
    Harvested schema correctly includes schools.District.
    """
    mock_llm = MockLLMEngine()
    linker = TaskAlignedSchemaLinker(
        llm_engine=mock_llm,
        max_retries=2,
    )

    question = "List the zip code of all the charter schools in Fresno County Office of Education"

    result = linker.link_schema(
        question=question,
        processed_metadata=sample_metadata,
        lsh_indexer=sample_lsh_indexer,
        vector_indexer=sample_vector_indexer,
        raw_profile=sample_raw_profile,
    )

    # Verify re-prompting triggered and resolved
    assert result.total_retries >= 1
    assert 'schools."District"' in result.final_linked_schema
    assert "schools.District" in result.canonical_schema
    assert 'schools."Zip"' in result.final_linked_schema
    assert "schools.Zip" in result.canonical_schema
    assert 'frpm."CDSCode"' in result.final_linked_schema
    assert "frpm.CDSCode" in result.canonical_schema
    assert 'schools."CDSCode"' in result.final_linked_schema
    assert "schools.CDSCode" in result.canonical_schema


# ---------------------------------------------------------------------------
# Test Checklist 5: Union / Fusion & Key Preservation
# ---------------------------------------------------------------------------


def test_preserve_linking_keys(
    sample_raw_profile: dict[str, Any],
    sample_metadata: dict[str, Any],
) -> None:
    """Verify preserving undocumented join keys and primary keys across referenced tables."""
    linker = TaskAlignedSchemaLinker(llm_engine=MockLLMEngine())

    # Initially only Zip from schools and Charter from frpm are harvested
    harvested = {"schools.Zip", "frpm.Charter School (Y/N)"}

    extended = linker.preserve_linking_keys(
        linked_fields=harvested,
        raw_profile=sample_raw_profile,
        processed_metadata=sample_metadata,
    )

    # Undocumented join (frpm.CDSCode <-> schools.CDSCode) must be preserved
    assert "frpm.CDSCode" in extended
    assert "schools.CDSCode" in extended


def test_is_candidate_key_column() -> None:
    """Verify detection of various English and Vietnamese primary/foreign key conventions."""
    is_key = TaskAlignedSchemaLinker.is_candidate_key_column

    # English key formats
    assert is_key("id", "orders") is True
    assert is_key("order_id", "orders") is True
    assert is_key("CustomerID", "customers") is True
    assert is_key("CDSCode", "schools") is True
    assert is_key("code", "products") is True

    # Vietnamese key formats (with accents, mixed cases, spaces, abbreviations)
    assert is_key("Mã hóa đơn", "ban_hang") is True
    assert is_key("Mã KH", "khach_hang") is True
    assert is_key("Mã SP", "san_pham") is True
    assert is_key("Mã phiếu thu", "thu_chi") is True
    assert is_key("Mã barcode", "san_pham") is True
    assert is_key("Ma don", "don_hang") is True
    assert is_key("Số HĐ", "hoa_don") is True
    assert is_key("Số CT", "chung_tu") is True
    assert is_key("Số phiếu", "phieu_xuat") is True
    assert is_key("Số KH", "khach_hang") is True

    # Non-key fields (measures, phone numbers, descriptive text, notes, boolean flags ending in 'id')
    assert is_key("is_paid", "invoices") is False
    assert is_key("paid", "orders") is False
    assert is_key("valid", "tokens") is False
    assert is_key("is_valid", "users") is False
    assert is_key("grid", "layout") is False
    assert is_key("mid", "audio") is False
    assert is_key("solid", "geometry") is False
    assert is_key("idea", "notes") is False
    assert is_key("idle", "status") is False
    assert is_key("Số lượng lẻ", "ban_hang") is False
    assert is_key("Số lượng", "ban_hang") is False
    assert is_key("Số điện thoại", "khach_hang") is False
    assert is_key("Mẫu ghi chú", "ban_hang") is False
    assert is_key("Thanh tien", "ban_hang") is False
    assert is_key("Giá bán", "ban_hang") is False
    assert is_key("Đối tác", "ban_hang") is False


def test_preserve_vietnamese_primary_keys() -> None:
    """Verify automatic preservation of Vietnamese primary keys for referenced tables."""
    linker = TaskAlignedSchemaLinker(llm_engine=MockLLMEngine())
    raw_profile = {
        "tables": {
            "ban_hang": {
                "columns": {
                    "Mã hóa đơn": {},
                    "Ngày tháng": {},
                    "Số lượng lẻ": {},
                    "Thanh tien": {},
                    "Mẫu ghi chú": {},
                }
            }
        }
    }
    metadata = {
        "ban_hang.Mã hóa đơn": {"data_format": "STRING"},
        "ban_hang.Ngày tháng": {"data_format": "DATE"},
        "ban_hang.Số lượng lẻ": {"data_format": "INTEGER"},
        "ban_hang.Thanh tien": {"data_format": "INTEGER"},
        "ban_hang.Mẫu ghi chú": {"data_format": "STRING"},
    }

    # Only 'Thanh tien' harvested initially
    harvested = {"ban_hang.Thanh tien"}
    extended = linker.preserve_linking_keys(
        linked_fields=harvested,
        raw_profile=raw_profile,
        processed_metadata=metadata,
    )

    # 'Mã hóa đơn' must be preserved as table PK
    assert "ban_hang.Mã hóa đơn" in extended
    # Non-key metrics and text should NOT be preserved
    assert "ban_hang.Số lượng lẻ" not in extended
    assert "ban_hang.Mẫu ghi chú" not in extended


def test_functional_task_aligned_schema_linking(
    sample_metadata: dict[str, Any],
    sample_lsh_indexer: VietnameseLSHIndexer,
    sample_vector_indexer: SemanticVectorIndexer,
    sample_raw_profile: dict[str, Any],
) -> None:
    """Verify functional interface task_aligned_schema_linking returns set of columns."""
    mock_llm = MockLLMEngine()
    question = "List the zip code of all the charter schools in Fresno County Office of Education"

    final_schema = task_aligned_schema_linking(
        question=question,
        processed_metadata=sample_metadata,
        lsh_indexer=sample_lsh_indexer,
        vector_indexer=sample_vector_indexer,
        llm_engine=mock_llm,
        raw_profile=sample_raw_profile,
        max_retries=2,
    )

    assert isinstance(final_schema, set)
    assert 'schools."District"' in final_schema
    assert 'schools."Zip"' in final_schema


# ---------------------------------------------------------------------------
# Test Checklist 6: Schema Post-Processor & Ground-Truth Quoting
# ---------------------------------------------------------------------------


def test_schema_post_processor_hallucination_and_syntax_cleaning() -> None:
    """Verify post-processor eliminates hallucinations, cleans aliases, and quotes verified columns."""
    from zaic_2026_4.schema_linking.post_processor import SchemaPostProcessor

    metadata = {
        "ban_hang.Giá bán": {"data_format": "REAL"},
        "ban_hang.Kết quả": {"data_format": "STRING"},
        "ban_hang.Nội dung": {"data_format": "STRING"},
        "ban_hang.Ngày tháng": {"data_format": "STRING"},
        "ban_hang.Đơn vị (tính)": {"data_format": "STRING"},
        "ban_hang.so_luong": {"data_format": "INTEGER"},
        "ban_hang.Thanh tien": {"data_format": "REAL"},
        "ban_hang.Quy ra ĐV lẻ": {"data_format": "INTEGER"},
        "ban_hang.Mã hóa đơn": {"data_format": "STRING"},
        "ban_hang.Đối tác": {"data_format": "STRING"},
    }

    raw_candidates = {
        "T1.Ngay thang",
        "T1.Noi dung LIKE",
        "T1.Ten hang LIKE",
        "T1.Thanh tien) AS",
        "T1.Quy_ra_DV_le",
        "ban_hang.Ngay_thang",
        "ban_hang.Noi_dung",
        "ban_hang.Ten_hang",
        "ban_hang.Thanh_tien",
        "ban_hang.Doi_tac",
    }

    aliases = {"T1": "ban_hang"}

    quoted, canonical = SchemaPostProcessor.post_process_linked_schema(
        raw_fields=raw_candidates,
        processed_metadata=metadata,
        table_aliases=aliases,
        always_quote=True,
    )

    # Hallucinated columns Ten_hang and Ten hang must be discarded
    assert not any("ten" in c.split(".", 1)[-1].lower() for c in canonical)
    assert 'ban_hang."Ten_hang"' not in quoted
    assert 'ban_hang."Ten hang"' not in quoted
    assert "ban_hang.Ten_hang" not in canonical
    assert "ban_hang.Ten hang" not in canonical

    # Valid columns must be canonicalized and quoted (including Vietnamese Đ/đ normalization)
    assert 'ban_hang."Ngày tháng"' in quoted
    assert "ban_hang.Ngày tháng" in canonical
    assert 'ban_hang."Nội dung"' in quoted
    assert "ban_hang.Nội dung" in canonical
    assert 'ban_hang."Thanh tien"' in quoted
    assert "ban_hang.Thanh tien" in canonical
    assert 'ban_hang."Quy ra ĐV lẻ"' in quoted
    assert "ban_hang.Quy ra ĐV lẻ" in canonical
    assert 'ban_hang."Đối tác"' in quoted
    assert "ban_hang.Đối tác" in canonical

    # Exactly 5 unique ground-truth columns should remain
    assert len(quoted) == 5
    assert len(canonical) == 5


def test_schema_post_processor_multi_table_resolution() -> None:
    """Verify post-processor resolves aliases and columns across multiple tables."""
    from zaic_2026_4.schema_linking.post_processor import SchemaPostProcessor

    metadata = {
        "ban_hang.Mã khách": {"data_format": "STRING"},
        "ban_hang.Thanh tien": {"data_format": "REAL"},
        "khach_hang.Mã khách": {"data_format": "STRING"},
        "khach_hang.Tên khách": {"data_format": "STRING"},
    }

    raw_candidates = {
        "T1.Thanh tien",
        "T1.Ma_khach",
        "T2.Ma_khach",
        "T2.Ten khach LIKE",
        "T2.Email_fake",  # Hallucination on table 2
    }

    aliases = {"T1": "ban_hang", "T2": "khach_hang"}

    quoted, canonical = SchemaPostProcessor.post_process_linked_schema(
        raw_fields=raw_candidates,
        processed_metadata=metadata,
        table_aliases=aliases,
        always_quote=True,
    )

    # Hallucinated column Email_fake rejected
    assert not any("Email" in c for c in canonical)

    # Verified multi-table columns
    assert 'ban_hang."Thanh tien"' in quoted
    assert 'ban_hang."Mã khách"' in quoted
    assert 'khach_hang."Mã khách"' in quoted
    assert 'khach_hang."Tên khách"' in quoted
    assert len(quoted) == 4

    # Format for SQL Generation XML
    xml_output = SchemaPostProcessor.format_for_sql_generation(quoted, metadata)
    assert '<table name="ban_hang">' in xml_output
    assert '<table name="khach_hang">' in xml_output
    assert '<column name="Tên khách"' in xml_output


def test_schema_linking_auto_save_cache(
    sample_metadata: dict[str, Any],
    sample_lsh_indexer: Any,
    sample_vector_indexer: Any,
    sample_raw_profile: dict[str, Any],
    tmp_path: Path,
) -> None:
    from zaic_2026_4.schema_linking.task_aligned_linker import TaskAlignedSchemaLinker

    linker = TaskAlignedSchemaLinker(
        llm_engine=MockLLMEngine(),
        max_retries=1,
    )

    question = "List the zip code of all charter schools in Fresno"
    cache_dir = tmp_path / "cache_test"

    result = linker.link_schema(
        question=question,
        processed_metadata=sample_metadata,
        lsh_indexer=sample_lsh_indexer,
        vector_indexer=sample_vector_indexer,
        raw_profile=sample_raw_profile,
        cache_dir=cache_dir,
        question_id="q_123",
    )
    assert result.final_linked_schema is not None

    res_file = cache_dir / "schema_linking_result.json"
    assert res_file.is_file()

    data = json.loads(res_file.read_text(encoding="utf-8"))
    assert "final_linked_schema" in data
    assert "variant_traces" in data
    assert data["question"] == question
    assert data["question_id"] == "q_123"

    per_q_file = cache_dir / "schema_linking_result_q_123.json"
    assert per_q_file.is_file()


def test_zaic_schema_link_cli_parser_and_matching() -> None:
    """Verify zaic-schema-link CLI parser and business ID question matching."""
    from zaic_2026_4.schema_linking.cli import _matches_biz_id, build_parser

    parser = build_parser()
    args = parser.parse_args([
        "--split", "dataset/train",
        "--biz-id", "101",
        "--output-dir", "/tmp/custom_cache",
        "--mock",
    ])
    assert args.split == "dataset/train"
    assert args.biz_id == "101"
    assert args.output_dir == "/tmp/custom_cache"
    assert args.mock is True
    assert args.local is False

    # Check _matches_biz_id logic
    assert _matches_biz_id("101", "biz_101") is True
    assert _matches_biz_id("biz_101", "biz_101") is True
    assert _matches_biz_id("101", "101") is True
    assert _matches_biz_id("biz_101", "101") is True
    assert _matches_biz_id("1", "biz_01") is True
    assert _matches_biz_id("101", "biz_102") is False


