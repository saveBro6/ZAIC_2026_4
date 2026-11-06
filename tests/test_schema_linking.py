"""Unit tests for Task-Aligned Schema Linking.

Verifies:
  - Checklist 1: Context builder & 5 variants (Cases a to e)
  - Checklist 2: SQL AST Parser (aliases, fields, literals, fallback)
  - Checklist 3: LSH lookup & re-prompting loop on literal mismatch
  - Checklist 4: Paper benchmark (Fresno County Office of Education example)
  - Checklist 5: Union / Fusion & Key preservation (PK, FK, undocumented joins)
"""

from __future__ import annotations

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
    """Verify stripping markdown code fences, comments, and semicolons."""
    raw = "```sql\n-- Calculate total\nSELECT col1 FROM table1;\n```"
    cleaned = ast_extractor.clean_sql(raw)
    assert cleaned == "SELECT col1 FROM table1"


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
    assert "schools.District" in result.final_linked_schema
    assert "schools.Zip" in result.final_linked_schema
    assert "frpm.CDSCode" in result.final_linked_schema
    assert "schools.CDSCode" in result.final_linked_schema


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
    assert "schools.District" in final_schema
    assert "schools.Zip" in final_schema
