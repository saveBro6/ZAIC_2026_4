"""Unit tests for centralized constants module and backward-compatible re-exports."""

import re
from pathlib import Path

import zaic_2026_4
import zaic_2026_4.ingestion
import zaic_2026_4.ingestion.service
import zaic_2026_4.profiling
import zaic_2026_4.profiling.lsh_indexer
import zaic_2026_4.profiling.profile_summarizer
import zaic_2026_4.profiling.raw_profiler
import zaic_2026_4.profiling.vector_indexer
import zaic_2026_4.schema_linking.context_builder
from zaic_2026_4 import constants


def test_package_exports_constants() -> None:
    """Verify that the top-level package exposes constants."""
    assert hasattr(zaic_2026_4, "constants")
    assert zaic_2026_4.constants is constants


def test_constants_count_and_all() -> None:
    """Verify that __all__ contains exactly all expected constants without omissions."""
    expected_constants = {
        # Ingestion
        "DEFAULT_BASE_OUTPUT_DIR",
        # LSH Indexer
        "DEFAULT_NUM_PERM",
        "DEFAULT_THRESHOLD",
        "DEFAULT_SHINGLE_K",
        "DEFAULT_MAX_SAMPLE_VALUES",
        # Profile Summarizer
        "DEFAULT_PROD_MODEL",
        "DEFAULT_LOCAL_MODEL",
        "DEFAULT_PROD_GPU_UTIL",
        "DEFAULT_LOCAL_GPU_UTIL",
        "DEFAULT_LLM_MAX_LENGTH",
        "MINIMAL_PROFILE_PROMPT",
        "MAXIMAL_PROFILE_PROMPT",
        # Raw Profiler
        "ISO_DATE_REGEX",
        "ISO_DATETIME_REGEX",
        "SLASH_DATE_REGEX",
        "SLASH_DATE_SHORT_REGEX",
        "VN_DATE_ABBR_REGEX",
        "EN_DATE_ABBR_REGEX",
        "VN_DATE_FULL_REGEX",
        "EN_DATE_FULL_REGEX",
        "VN_DATE_WEEKDAY_REGEX",
        "EN_DATE_WEEKDAY_REGEX",
        "EXCEL_SERIAL_REGEX",
        "VND_TEXT_REGEX",
        "VND_SYMBOL_REGEX",
        "ACCOUNTING_PARENTHESES_REGEX",
        "ACCOUNTING_DASH_ZERO_REGEX",
        "SCALED_THOUSAND_K_REGEX",
        "SCALED_MILLION_TR_REGEX",
        "SCALED_BILLION_TY_REGEX",
        "STANDARD_THOUSAND_SEPARATOR_REGEX",
        "PLAIN_NUMBER_REGEX",
        "VIEW_FROM_REGEX",
        "KEY_TOKENS",
        "NON_KEY_METRIC_TOKENS",
        # Vector Indexer
        "DEFAULT_EMBED_MODEL_PATH",
        "DEFAULT_EMBED_MAX_LENGTH",
        "DEFAULT_BATCH_SIZE",
        "DEFAULT_TOP_K",
        "DEFAULT_EMBEDDING_DIM",
        "DEFAULT_MAX_CHUNK_CHARS",
        "DEFAULT_MIN_CHUNK_CHARS",
        "DEFAULT_EMBED_GPU_UTIL",
        "MIN_TABLE_ROWS_FOR_SPLIT",
        # Schema Linking
        "FEW_SHOT_SQL_EXAMPLES",
        # Submission
        "DEFAULT_CANDIDATE_SEEDS",
        "DEFAULT_EXECUTION_MAX_RETRIES",
        "DEFAULT_LOGS_DIR",
        "DEFAULT_NUM_CANDIDATES",
        "DEFAULT_SQL_MAX_RETRIES",
        "DEFAULT_SQL_TIMEOUT_SECONDS",
        "DEFAULT_SUBMISSION_PATH",
    }
    assert set(constants.__all__) == expected_constants
    assert len(constants.__all__) == 52

    for name in expected_constants:
        assert hasattr(constants, name), f"constants module missing {name}"


def test_constant_types_and_values() -> None:
    """Verify types and value integrity of critical constants."""
    assert isinstance(constants.DEFAULT_BASE_OUTPUT_DIR, Path)

    assert isinstance(constants.DEFAULT_NUM_PERM, int)
    assert constants.DEFAULT_NUM_PERM == 128
    assert isinstance(constants.DEFAULT_THRESHOLD, float)
    assert constants.DEFAULT_THRESHOLD == 0.4
    assert isinstance(constants.DEFAULT_SHINGLE_K, int)
    assert constants.DEFAULT_SHINGLE_K == 3
    assert isinstance(constants.DEFAULT_MAX_SAMPLE_VALUES, int)
    assert constants.DEFAULT_MAX_SAMPLE_VALUES == 10000

    assert isinstance(constants.DEFAULT_PROD_MODEL, str)
    assert constants.DEFAULT_LOCAL_MODEL == "Qwen/Qwen2.5-1.5B-Instruct"
    assert isinstance(constants.DEFAULT_PROD_GPU_UTIL, float)
    assert isinstance(constants.DEFAULT_LOCAL_GPU_UTIL, float)
    assert constants.DEFAULT_LLM_MAX_LENGTH == 16384
    assert "<system>" in constants.MINIMAL_PROFILE_PROMPT
    assert "<system>" in constants.MAXIMAL_PROFILE_PROMPT

    # Regex patterns
    assert isinstance(constants.ISO_DATE_REGEX, re.Pattern)
    assert constants.ISO_DATE_REGEX.match("2026-08-15") is not None
    assert constants.SLASH_DATE_REGEX.match("15/08/2026") is not None
    assert constants.VND_TEXT_REGEX.match("120,000 vnđ") is not None
    assert constants.VIEW_FROM_REGEX.search('FROM "don_hang"') is not None

    # Token sets
    assert isinstance(constants.KEY_TOKENS, frozenset)
    assert "id" in constants.KEY_TOKENS
    assert isinstance(constants.NON_KEY_METRIC_TOKENS, frozenset)
    assert "tien" in constants.NON_KEY_METRIC_TOKENS

    # Vector Indexer
    assert constants.DEFAULT_EMBED_MODEL_PATH == "/app/models/bge-m3"
    assert constants.DEFAULT_EMBED_MAX_LENGTH == 4096
    assert constants.DEFAULT_EMBEDDING_DIM == 1024
    assert constants.DEFAULT_BATCH_SIZE == 32
    assert constants.DEFAULT_EMBED_GPU_UTIL == 0.20

    # Schema Linking
    assert "<few_shot_examples>" in constants.FEW_SHOT_SQL_EXAMPLES

    # Submission
    assert constants.DEFAULT_CANDIDATE_SEEDS == (42, 100, 2026)
    assert constants.DEFAULT_NUM_CANDIDATES == 3
    assert constants.DEFAULT_SQL_MAX_RETRIES == 2
    assert constants.DEFAULT_EXECUTION_MAX_RETRIES == 2
    assert constants.DEFAULT_SQL_TIMEOUT_SECONDS == 10.0
    assert isinstance(constants.DEFAULT_SUBMISSION_PATH, Path)
    assert isinstance(constants.DEFAULT_LOGS_DIR, Path)


def test_backward_compatibility_reexports() -> None:
    """Verify that original submodules successfully re-export the identical objects."""
    # Ingestion
    assert zaic_2026_4.ingestion.service.DEFAULT_BASE_OUTPUT_DIR is constants.DEFAULT_BASE_OUTPUT_DIR
    assert zaic_2026_4.ingestion.DEFAULT_BASE_OUTPUT_DIR is constants.DEFAULT_BASE_OUTPUT_DIR

    # LSH Indexer
    assert zaic_2026_4.profiling.lsh_indexer.DEFAULT_NUM_PERM is constants.DEFAULT_NUM_PERM
    assert zaic_2026_4.profiling.DEFAULT_NUM_PERM is constants.DEFAULT_NUM_PERM

    # Profile Summarizer
    assert zaic_2026_4.profiling.profile_summarizer.DEFAULT_PROD_MODEL is constants.DEFAULT_PROD_MODEL
    assert zaic_2026_4.profiling.DEFAULT_PROD_MODEL is constants.DEFAULT_PROD_MODEL
    assert zaic_2026_4.profiling.profile_summarizer.MINIMAL_PROFILE_PROMPT is constants.MINIMAL_PROFILE_PROMPT

    # Raw Profiler
    assert zaic_2026_4.profiling.raw_profiler.ISO_DATE_REGEX is constants.ISO_DATE_REGEX
    assert zaic_2026_4.profiling.raw_profiler.KEY_TOKENS is constants.KEY_TOKENS
    assert zaic_2026_4.profiling.raw_profiler.NON_KEY_METRIC_TOKENS is constants.NON_KEY_METRIC_TOKENS

    # Vector Indexer
    assert zaic_2026_4.profiling.vector_indexer.DEFAULT_EMBED_MODEL_PATH is constants.DEFAULT_EMBED_MODEL_PATH
    assert zaic_2026_4.profiling.DEFAULT_EMBED_MODEL_PATH is constants.DEFAULT_EMBED_MODEL_PATH
    assert zaic_2026_4.profiling.vector_indexer.DEFAULT_EMBED_MAX_LENGTH is constants.DEFAULT_EMBED_MAX_LENGTH
    assert zaic_2026_4.profiling.DEFAULT_EMBED_MAX_LENGTH is constants.DEFAULT_EMBED_MAX_LENGTH
    assert zaic_2026_4.profiling.vector_indexer.DEFAULT_EMBEDDING_DIM is constants.DEFAULT_EMBEDDING_DIM
    assert zaic_2026_4.profiling.vector_indexer.DEFAULT_EMBED_GPU_UTIL is constants.DEFAULT_EMBED_GPU_UTIL

    # Schema Linking
    assert zaic_2026_4.schema_linking.context_builder.FEW_SHOT_SQL_EXAMPLES is constants.FEW_SHOT_SQL_EXAMPLES
