"""Regression tests for the pipeline optimizations derived from the trace analysis.

Covers the deterministic temporal resolver, grounding hygiene, schema coverage repair,
truncation-aware output parsing, the new SQL guardrails, batched candidate generation and
the unanswerable voting majority rule.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pytest

from zaic_2026_4.common.query_intent import detect_query_intent
from zaic_2026_4.common.temporal_resolver import (
    format_temporal_anchor_xml,
    resolve_primary_period,
    resolve_temporal_periods,
)
from zaic_2026_4.constants import DEFAULT_LSH_INDEX_VERSION
from zaic_2026_4.profiling.lsh_indexer import VietnameseLSHIndexer
from zaic_2026_4.schema_linking.coverage import validate_schema_coverage
from zaic_2026_4.schema_linking.grounding import filter_grounded_literals
from zaic_2026_4.submission.candidate_generator import CandidateGenerator
from zaic_2026_4.submission.models import CandidateQuery
from zaic_2026_4.submission.sql_validator import (
    SQLValidator,
    infer_currency_separators,
    is_truncated_json_output,
    parse_structured_sql_output,
)
from zaic_2026_4.submission.voting_executor import VotingExecutor

# ---------------------------------------------------------------------------
# Temporal resolver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "as_of", "expected"),
    [
        (
            "Tổng doanh thu tháng này (tính đến hôm nay) là bao nhiêu?",
            "2026-09-07",
            ("2026-09-01", "2026-09-07"),
        ),
        ("Doanh thu tháng trước là bao nhiêu?", "2026-05-25", ("2026-04-01", "2026-04-30")),
        ("Số đơn trong 30 ngày gần đây", "2026-05-25", ("2026-04-26", "2026-05-25")),
        ("Doanh thu quý 1/2026", "2026-05-25", ("2026-01-01", "2026-03-31")),
        ("Doanh thu năm 2025", "2026-05-25", ("2025-01-01", "2025-12-31")),
        ("Doanh thu tuần trước", "2026-05-25", ("2026-05-18", "2026-05-24")),
    ],
)
def test_temporal_resolver_periods(question: str, as_of: str, expected: tuple[str, str]) -> None:
    """Verify relative expressions resolve to concrete inclusive ISO ranges."""
    period = resolve_primary_period(question, as_of)
    assert period is not None
    assert (period.start, period.end) == expected


def test_temporal_resolver_ignores_untimed_questions() -> None:
    """A question without a temporal expression must not fabricate a period."""
    assert resolve_temporal_periods("Giá thấp nhất là bao nhiêu?", "2026-05-25") == []


def test_temporal_resolver_clamps_future_periods() -> None:
    """Periods that start after the anchor date are discarded, not shifted."""
    assert resolve_temporal_periods("Doanh thu tháng 12/2026", "2026-05-25") == []


def test_temporal_anchor_xml_contains_period_bounds() -> None:
    """The rendered anchor must carry the resolved start/end so the model never guesses."""
    anchor = format_temporal_anchor_xml("Doanh thu tháng này?", "2026-09-07")
    assert "as_of_date: 2026-09-07" in anchor
    assert 'start="2026-09-01"' in anchor
    assert 'end="2026-09-07"' in anchor


# ---------------------------------------------------------------------------
# Query intent
# ---------------------------------------------------------------------------


def test_query_intent_detects_signals() -> None:
    """Aggregate, exact-match and distinct cues are derived from the question text only."""
    intent = detect_query_intent(
        "Cho tôi biết chính xác tổng số lượng duy nhất đã bán trong tháng 5/2026",
        "2026-05-25",
    )
    assert intent.has_temporal_cue is True
    assert intent.has_aggregate is True
    assert intent.wants_exact_match is True
    assert intent.wants_distinct is True
    assert intent.primary_period is not None


# ---------------------------------------------------------------------------
# Grounding hygiene
# ---------------------------------------------------------------------------


def test_grounding_filter_drops_speculative_literals() -> None:
    """Only question-derived entities that survive in the final schema may reach the prompt."""
    final_schema = {'"HoaDon"."MaHoaDon"', '"ChiTiet"."HoatChat"'}
    candidates = [
        {
            "mention": "Natri clorid",
            "canonical_value": "Natri clorid",
            "column": "ChiTiet.HoatChat",
            "score": 0.95,
        },
        # Derived from trial SQL only: the phrase is absent from the question.
        {
            "mention": "Nước muối",
            "canonical_value": "Nước muối sinh lý",
            "column": "ChiTiet.HoatChat",
            "score": 0.92,
        },
        # Date values belong to the temporal resolver, not to grounding.
        {
            "mention": "tháng 5",
            "canonical_value": "2026-05-01",
            "column": "HoaDon.NgayBan",
            "score": 0.90,
        },
        # Calendar dimensions are excluded outright.
        {
            "mention": "tháng 5",
            "canonical_value": "Tháng Năm",
            "column": "calendar.month_name",
            "score": 0.90,
        },
        # Column dropped from the final linked schema.
        {
            "mention": "Natri clorid",
            "canonical_value": "Natri clorid",
            "column": "ChiTiet.GhiChu",
            "score": 0.88,
        },
    ]

    kept, dropped = filter_grounded_literals(
        candidates,
        final_schema=final_schema,
        question="Tổng tiền cần trả cho hoạt chất Natri clorid trong tháng 5/2026 là bao nhiêu?",
    )

    assert [item["canonical_value"] for item in kept] == ["Natri clorid"]
    assert kept[0]["source"] == "question"
    assert kept[0]["column"] == "ChiTiet.HoatChat"
    reasons = {item["reason"] for item in dropped}
    assert {"not_in_question", "date_value", "calendar_column", "column_not_in_schema"} <= reasons


# ---------------------------------------------------------------------------
# Schema coverage
# ---------------------------------------------------------------------------


def _temporal_metadata() -> dict[str, Any]:
    """Metadata whose only date column lives in a table reachable through a join edge."""
    return {
        "ChiTiet.MaHoaDon": {
            "data_format": "STRING",
            "minimal_profile": "Mã hóa đơn của dòng chi tiết.",
            "sample_values": ["HD001", "HD002"],
            "join_candidates": [{"target": "HoaDon.MaHoaDon", "resemblance": 0.95}],
        },
        "ChiTiet.ThanhTien": {
            "data_format": "CURRENCY",
            "minimal_profile": "Thành tiền của dòng chi tiết.",
            "sample_values": ["1.200.000"],
        },
        "HoaDon.MaHoaDon": {
            "data_format": "STRING",
            "minimal_profile": "Mã hóa đơn của chứng từ bán hàng.",
            "sample_values": ["HD001", "HD002"],
        },
        "HoaDon.NgayBan": {
            "data_format": "SLASH_DATE",
            "minimal_profile": "Ngày bán của hóa đơn.",
            "sample_values": ["02/05/2026"],
        },
    }


def test_coverage_pulls_missing_date_table_through_join_edge() -> None:
    """A temporal question must not lose the only table that owns a sale date."""
    intent = detect_query_intent("Tổng thành tiền bán trong tháng 5/2026?", "2026-05-25")
    schema = {"ChiTiet.MaHoaDon", "ChiTiet.ThanhTien"}

    expanded, report = validate_schema_coverage(schema, _temporal_metadata(), intent)

    assert "HoaDon.NgayBan" in expanded
    assert "HoaDon.MaHoaDon" in expanded
    assert "temporal_date_column" in report.satisfied_aspects
    assert report.has_unresolved_gaps is False
    assert "HoaDon" in report.added_tables


def test_coverage_reports_unresolved_temporal_gap() -> None:
    """With no reachable date column the gap must be reported instead of silently passing."""
    intent = detect_query_intent("Tổng thành tiền bán trong tháng 5/2026?", "2026-05-25")
    metadata = {
        "ChiTiet.ThanhTien": {"data_format": "CURRENCY", "minimal_profile": "Thành tiền."},
    }

    _expanded, report = validate_schema_coverage({"ChiTiet.ThanhTien"}, metadata, intent)

    assert report.has_unresolved_gaps is True
    assert "temporal_date_column" in report.missing_aspects


# ---------------------------------------------------------------------------
# Output parsing: truncation vs unparseable
# ---------------------------------------------------------------------------


def test_is_truncated_json_output() -> None:
    """Mid-JSON cut-offs are detected, complete objects are not."""
    assert is_truncated_json_output('{"reasoning": " thinkingđang phân tích') is True
    assert is_truncated_json_output('{"sql": "SELECT 1", "type": "number"}') is False
    assert is_truncated_json_output("SELECT 1") is False


def test_parse_truncated_output_never_returns_raw_json_as_sql() -> None:
    """A truncated response must yield an empty SQL plus an explicit blocking code."""
    sql, _target_type, issues = parse_structured_sql_output('{"reasoning": " thinkingcắt giữa')
    assert sql == ""
    assert any(issue.code == "OUTPUT_TRUNCATED" for issue in issues)


def test_parse_plain_sql_still_supported() -> None:
    """Backward compatibility: a bare SQL statement is still accepted."""
    sql, target_type, issues = parse_structured_sql_output("SELECT * FROM users;")
    assert sql == "SELECT * FROM users;"
    assert target_type == "number"
    assert issues == []


# ---------------------------------------------------------------------------
# SQL guardrails
# ---------------------------------------------------------------------------


def _guardrail_metadata() -> dict[str, Any]:
    return {
        "HoaDon.NgayBan": {"data_format": "ISO_DATE", "sample_values": ["2026-09-02"]},
        "HoaDon.CanTraNCC": {
            "data_format": "CURRENCY",
            "sample_values": ["89,526,400", "1,181"],
        },
        "ChiTiet.HoatChat": {
            "data_format": "STRING",
            "distinct_count": 3,
            "sample_values": ["Natri clorid", "Glucose 5%", "NaCl 0.9%"],
        },
    }


def test_validator_blocks_out_of_period_date_window() -> None:
    """A January window must be rejected for a September question."""
    intent = detect_query_intent("Tổng doanh thu tháng này là bao nhiêu?", "2026-09-07")
    sql = (
        'SELECT SUM("CanTraNCC") FROM "HoaDon" '
        "WHERE \"NgayBan\" BETWEEN '2026-01-01' AND '2026-01-31'"
    )

    result = SQLValidator.validate(
        sql,
        target_type="number",
        metadata=_guardrail_metadata(),
        as_of_date="2026-09-07",
        intent=intent,
    )

    assert result.is_valid is False
    assert "DATE_RANGE_OUT_OF_PERIOD" in {issue.code for issue in result.issues}


def test_validator_allows_historical_filter_outside_anchor_year() -> None:
    """A legitimate historical bound in another year is not an out-of-period query."""
    intent = detect_query_intent("Doanh thu tháng này?", "2026-09-07")
    sql = (
        'SELECT SUM(CAST(REPLACE("CanTraNCC", \',\', \'\') AS INTEGER)) FROM "HoaDon" '
        "WHERE \"NgayBan\" BETWEEN '2026-09-01' AND '2026-09-07' "
        "AND \"NgayBan\" >= '2000-01-01'"
    )

    result = SQLValidator.validate(
        sql,
        target_type="number",
        metadata=_guardrail_metadata(),
        as_of_date="2026-09-07",
        intent=intent,
    )

    assert "DATE_RANGE_OUT_OF_PERIOD" not in {issue.code for issue in result.issues}


def test_infer_currency_separators() -> None:
    """Separator inference distinguishes VND comma grouping from dot grouping."""
    assert infer_currency_separators({"89,526,400", "1,181"}) == (",", None)
    assert infer_currency_separators({"1.200.000", "3.500.000 đ"}) == (".", None)


def test_validator_blocks_currency_cleaning_with_wrong_separator() -> None:
    """Removing the decimal dot while keeping comma grouping corrupts the magnitude."""
    sql = (
        'SELECT SUM(CAST(REPLACE("CanTraNCC", \'.\', \'\') AS INTEGER)) FROM "HoaDon" '
        "WHERE \"NgayBan\" BETWEEN '2026-09-01' AND '2026-09-07'"
    )
    intent = detect_query_intent("Tổng chi phí tháng này?", "2026-09-07")

    result = SQLValidator.validate(
        sql,
        target_type="number",
        metadata=_guardrail_metadata(),
        as_of_date="2026-09-07",
        intent=intent,
    )

    assert result.is_valid is False
    assert "CURRENCY_SEPARATOR_MISMATCH" in {issue.code for issue in result.issues}


def test_validator_hints_exact_match_for_verbatim_literal() -> None:
    """A LIKE on a value that exists verbatim is reported as a partial-match hint."""
    sql = """SELECT "HoatChat" FROM "ChiTiet" WHERE "HoatChat" LIKE '%Natri clorid%' LIMIT 1"""
    intent = detect_query_intent("Hoạt chất nào?", "2026-09-07")

    result = SQLValidator.validate(
        sql,
        target_type="string",
        metadata=_guardrail_metadata(),
        as_of_date="2026-09-07",
        intent=intent,
    )

    assert result.is_valid is True
    assert "LITERAL_PARTIAL_MATCH" in {issue.code for issue in result.issues}


# ---------------------------------------------------------------------------
# Candidate generation: batching and truncation recovery
# ---------------------------------------------------------------------------


class _CountingEngine:
    """Engine returning a fixed set of answers while recording every call."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[Any] = []
        self.last_call_traces: list[dict[str, Any]] = []

    def generate(self, prompts: list[str] | str, **kwargs: Any) -> list[str] | str:
        self.calls.append(prompts)
        is_batch = not isinstance(prompts, str)
        count = len(prompts) if is_batch else 1
        outputs = [self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]] * count
        self.last_call_traces = [
            {"finish_reason": "stop", "prompt_tokens": 10, "completion_tokens": 5, "latency_ms": 1.0}
            for _ in range(count)
        ]
        return outputs if is_batch else outputs[0]


def _metadata_for_candidates() -> dict[str, Any]:
    return {
        "users.id": {"data_format": "INTEGER", "minimal_profile": "ID"},
        "users.name": {"data_format": "STRING", "minimal_profile": "Tên"},
    }


def test_candidates_initial_generation_is_batched() -> None:
    """All candidates must share one batched initial call instead of N sequential ones."""
    engine = _CountingEngine(['{"sql": "SELECT name FROM users LIMIT 1", "type": "string"}'])
    generator = CandidateGenerator(llm_engine=engine, seeds=(42, 100, 2026))

    candidates = generator.generate_all_candidates(
        question="Danh sách tên người dùng?",
        as_of_date="2026-05-25",
        reduced_schema={"users.id", "users.name"},
        processed_metadata=_metadata_for_candidates(),
    )

    assert len(candidates) == 3
    assert len(engine.calls) == 1
    assert isinstance(engine.calls[0], list)
    assert len(engine.calls[0]) == 3
    assert all(c.initial_sql for c in candidates)


def test_truncated_candidate_is_retried_with_short_budget() -> None:
    """A length-truncated generation is flagged and repaired instead of being accepted."""
    engine = _CountingEngine(
        [
            '{"reasoning": " thinkingphân tích rất dài',
            '{"sql": "SELECT name FROM users LIMIT 1", "type": "string"}',
        ]
    )
    generator = CandidateGenerator(llm_engine=engine, seeds=(42,), sql_max_retries=1)

    candidate = generator.generate_candidate(
        candidate_id=0,
        seed=42,
        question="Danh sách tên người dùng?",
        as_of_date="2026-05-25",
        reduced_schema={"users.id", "users.name"},
        processed_metadata=_metadata_for_candidates(),
        trace_enabled=True,
    )

    assert candidate.initial_sql == ""
    assert candidate.final_sql == "SELECT name FROM users LIMIT 1"
    assert candidate.sql_retry_count == 1
    truncated_attempt = candidate.attempts_history[0]
    codes = {issue["code"] for issue in truncated_attempt["issues"]}
    assert "OUTPUT_TRUNCATED" in codes
    # The repair turn is the second attempt and produced the accepted SQL.
    assert candidate.best_attempt_idx == 1


# ---------------------------------------------------------------------------
# Voting: unanswerable needs a majority and a proven schema
# ---------------------------------------------------------------------------


def _unanswerable(candidate_id: int) -> CandidateQuery:
    return CandidateQuery(
        candidate_id=candidate_id,
        seed=candidate_id,
        final_sql="",
        predicted_type="unanswerable",
        deliberate_abstain=True,
        execution_success=True,
    )


def _failed(candidate_id: int) -> CandidateQuery:
    return CandidateQuery(
        candidate_id=candidate_id,
        seed=candidate_id,
        final_sql="SELECT 1",
        predicted_type="number",
        execution_success=False,
        is_valid_syntax=False,
    )


@pytest.mark.parametrize(
    ("candidates", "proof", "expected_status"),
    [
        ([_unanswerable(0), _failed(1), _failed(2)], True, "abstain_unverified_unanswerable"),
        ([_unanswerable(0), _unanswerable(1), _unanswerable(2)], True, "unanswerable_consensus"),
        ([_unanswerable(0), _unanswerable(1), _unanswerable(2)], False, "abstain_unverified_unanswerable"),
        ([_unanswerable(0)], True, "unanswerable_consensus"),
    ],
)
def test_vote_unanswerable_requires_majority_and_proof(
    candidates: list[CandidateQuery],
    proof: bool,
    expected_status: str,
) -> None:
    """A single giving-up candidate must never be submitted as unanswerable."""
    executor = VotingExecutor(db_path="/non/existent.db")
    _winner, status, _reason = executor.vote(candidates, unanswerable_proof=proof)
    assert status == expected_status


# ---------------------------------------------------------------------------
# LSH index hygiene
# ---------------------------------------------------------------------------


def test_lsh_index_skips_dates_and_caps_values(tmp_path: Path) -> None:
    """Date-only values are skipped and per-column volume is bounded."""
    indexer = VietnameseLSHIndexer(max_sample_values=3)
    metadata = {
        "orders.date": {"data_format": "ISO_DATE", "sample_values": ["2026-05-01", "2026-05-02"]},
        "orders.name": {
            "data_format": "STRING",
            "sample_values": ["An", "Binh", "Chi", "Dung", "Em"],
        },
    }

    indexed = indexer.build_index(metadata)
    indexed_columns = {col for col, _val in indexer.value_to_col_map.values()}

    assert indexed == 3
    assert "orders.date" not in indexed_columns

    path = tmp_path / "lsh_index.pkl"
    indexer.save(path)
    assert path.is_file()


def test_lsh_index_rejects_stale_version(tmp_path: Path) -> None:
    """A stale pickle is refused so the caller rebuilds instead of loading a huge old index."""
    indexer = VietnameseLSHIndexer(max_sample_values=3)
    indexer.build_index({"orders.name": {"sample_values": ["An", "Binh"]}})
    path = tmp_path / "lsh_index.pkl"
    indexer.save(path)

    with open(path, "rb") as f:
        payload = pickle.load(f)
    payload["version"] = 0
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    with pytest.raises(ValueError, match="Unsupported LSH index version"):
        VietnameseLSHIndexer.load(path)


# ---------------------------------------------------------------------------
# Prompt hygiene
# ---------------------------------------------------------------------------


def test_prompts_never_embed_example_schema_or_dates() -> None:
    """Neither prompt template may leak the illustrative schema or calendar years."""
    from zaic_2026_4.constants import FEW_SHOT_SQL_EXAMPLES
    from zaic_2026_4.submission.prompt_template import FEW_SHOT_SQL_EXAMPLES_VI

    for block in (FEW_SHOT_SQL_EXAMPLES, FEW_SHOT_SQL_EXAMPLES_VI):
        for leaked in ("don_hang", "chi_tiet_ban_hang", "khach_hang", "ban_hang", "2026-01-01"):
            assert leaked not in block
        assert "<few_shot_examples>" in block


def test_generation_prompt_states_question_period() -> None:
    """The SQL prompt must carry the resolved period of the question."""
    from zaic_2026_4.submission.prompt_template import build_sql_generation_prompt

    prompt = build_sql_generation_prompt(
        question="Tổng doanh thu tháng này là bao nhiêu?",
        as_of_date="2026-09-07",
        schema_keys=["bang.Cot"],
        processed_metadata={"bang.Cot": {"data_format": "STRING", "maximal_profile": "Cột"}},
    )

    anchor = prompt.split("\n<temporal_anchor>\nas_of_date:")[1].split("</temporal_anchor>")[0]
    assert " 2026-09-07" in anchor
    assert 'start="2026-09-01"' in anchor
    assert 'end="2026-09-07"' in anchor


# ---------------------------------------------------------------------------
# Review regressions: temporal boundaries, coverage proof, stale cache artifacts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        # Whole-month bound on the resolved September period.
        (
            'SELECT SUM("CanTraNCC") FROM "HoaDon" '
            "WHERE \"NgayBan\" BETWEEN '2026-09-01' AND '2026-09-30'"
        ),
        # Exclusive first day of the next month.
        (
            'SELECT SUM("CanTraNCC") FROM "HoaDon" '
            "WHERE \"NgayBan\" >= '2026-09-01' AND \"NgayBan\" < '2026-10-01'"
        ),
    ],
)
def test_temporal_window_accepts_natural_period_bounds(sql: str) -> None:
    """Natural calendar bounds of the resolved period are legal, not out-of-period."""
    intent = detect_query_intent("Tổng doanh thu tháng này là bao nhiêu?", "2026-09-07")

    result = SQLValidator.validate(
        sql,
        target_type="number",
        metadata=_guardrail_metadata(),
        as_of_date="2026-09-07",
        intent=intent,
    )

    assert "DATE_RANGE_OUT_OF_PERIOD" not in {issue.code for issue in result.issues}


def test_temporal_window_skipped_without_resolved_period() -> None:
    """A question without a relative expression must not be date-window checked."""
    intent = detect_query_intent("Giá trị lớn nhất của cột chi phí là bao nhiêu?", "2026-09-07")
    sql = (
        'SELECT MAX(CAST(REPLACE("CanTraNCC", \',\', \'\') AS INTEGER)) FROM "HoaDon" '
        "WHERE \"NgayBan\" <= '2026-12-31'"
    )

    result = SQLValidator.validate(
        sql,
        target_type="number",
        metadata=_guardrail_metadata(),
        as_of_date="2026-09-07",
        intent=intent,
    )

    assert "DATE_RANGE_OUT_OF_PERIOD" not in {issue.code for issue in result.issues}


def test_temporal_window_still_blocks_wrong_month_window() -> None:
    """The original January-for-September mistake stays blocking."""
    intent = detect_query_intent("Tổng doanh thu tháng này là bao nhiêu?", "2026-09-07")
    sql = (
        'SELECT SUM("CanTraNCC") FROM "HoaDon" '
        "WHERE \"NgayBan\" BETWEEN '2026-01-01' AND '2026-01-31'"
    )

    result = SQLValidator.validate(
        sql,
        target_type="number",
        metadata=_guardrail_metadata(),
        as_of_date="2026-09-07",
        intent=intent,
    )

    assert "DATE_RANGE_OUT_OF_PERIOD" in {issue.code for issue in result.issues}


def test_exact_match_becomes_blocking_when_question_demands_it() -> None:
    """An exact-value question must force the LIKE rewrite instead of only hinting it."""
    sql = """SELECT "HoatChat" FROM "ChiTiet" WHERE "HoatChat" LIKE '%Natri clorid%' LIMIT 1"""
    metadata = _guardrail_metadata()

    strict = SQLValidator.validate(
        sql,
        target_type="string",
        metadata=metadata,
        as_of_date="2026-09-07",
        intent=detect_query_intent(
            "Cho tôi biết chính xác hoạt chất Natri clorid dùng bao nhiêu?", "2026-09-07"
        ),
    )
    lenient = SQLValidator.validate(
        sql,
        target_type="string",
        metadata=metadata,
        as_of_date="2026-09-07",
        intent=detect_query_intent("Danh sách hoạt chất liên quan?", "2026-09-07"),
    )

    strict_codes = {(issue.code, issue.severity) for issue in strict.issues}
    lenient_codes = {(issue.code, issue.severity) for issue in lenient.issues}
    assert ("LITERAL_PARTIAL_MATCH", "blocking") in strict_codes
    assert ("LITERAL_PARTIAL_MATCH", "warning") in lenient_codes


def test_unanswerable_proof_requires_a_coverage_report() -> None:
    """A failed schema link (no report) and a reported gap both withhold the proof."""
    from zaic_2026_4.schema_linking.coverage import CoverageReport
    from zaic_2026_4.submission.pipeline import compute_unanswerable_proof

    covered = CoverageReport(required_aspects=["temporal_date_column"], satisfied_aspects=["temporal_date_column"])
    gapped = CoverageReport(required_aspects=["temporal_date_column"], missing_aspects=["temporal_date_column"])

    assert compute_unanswerable_proof(covered) is True
    assert compute_unanswerable_proof(gapped) is False
    assert compute_unanswerable_proof(None) is False


def test_stale_lsh_pickle_is_rebuilt_without_crashing(tmp_path: Path) -> None:
    """A pre-versioning pickle must trigger a rebuild, never an AttributeError."""
    import pickle

    from zaic_2026_4.profiling.profile_manager import ProfilingOrchestrator
    from zaic_2026_4.profiling.profile_summarizer import MockLLMEngine
    from zaic_2026_4.profiling.vector_indexer import MockEmbedder

    data_dir = tmp_path / "data"
    biz_dir = data_dir / "biz_stale"
    biz_dir.mkdir(parents=True)
    (biz_dir / "sales.csv").write_text(
        "id,ten_sp\n1,Sting\n2,Coca\n", encoding="utf-8"
    )
    rulebook = tmp_path / "rulebook.md"
    rulebook.write_text("# Rulebook\nDoanh thu.", encoding="utf-8")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # A metadata profile must exist, otherwise run_pipeline takes the fresh-build branch and
    # never reaches the stale-artifact handler this test is about.
    (cache_dir / "metadata_profile.json").write_text(
        json.dumps(
            {
                "sales.ten_sp": {
                    "minimal_profile": "Tên sản phẩm.",
                    "maximal_profile": "Tên sản phẩm bán ra.",
                    "data_format": "STRING",
                    "sample_values": ["Sting", "Coca"],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    indexer = VietnameseLSHIndexer(max_sample_values=5)
    indexer.build_index({"sales.ten_sp": {"sample_values": ["Sting", "Coca"]}})
    stale_path = cache_dir / "lsh_index.pkl"
    with open(stale_path, "wb") as f:
        pickle.dump(
            {
                "threshold": 0.4,
                "num_perm": 64,
                "shingle_k": 3,
                "max_sample_values": 5,
                "lsh": indexer.lsh,
                "value_to_col_map": indexer.value_to_col_map,
                "minhashes": indexer.minhashes,
            },
            f,
        )

    load_calls: list[Path] = []
    original_load = VietnameseLSHIndexer.load
    real_load = staticmethod(original_load)

    def _spy_load(path: str | Path) -> VietnameseLSHIndexer:
        load_calls.append(Path(path))
        return real_load(path)

    orchestrator = ProfilingOrchestrator(
        business_dir=biz_dir,
        rulebook_path=rulebook,
        llm_engine=MockLLMEngine(),
        embedder=MockEmbedder(),
        vector_model_path="mock",
        cache_dir=cache_dir,
    )

    VietnameseLSHIndexer.load = staticmethod(_spy_load)  # type: ignore[method-assign]
    try:
        # Must not raise even though the cached pickle predates the version stamp.
        result = orchestrator.run_pipeline(force_rebuild=False)
    finally:
        VietnameseLSHIndexer.load = staticmethod(original_load)  # type: ignore[method-assign]

    # The stale pickle really was loaded (and rejected) rather than skipped.
    assert load_calls == [stale_path]

    with open(stale_path, "rb") as f:
        assert pickle.load(f)["version"] == DEFAULT_LSH_INDEX_VERSION

    rebuilt = VietnameseLSHIndexer.load(stale_path)
    assert rebuilt.value_to_col_map
    assert result["lsh_indexer"] is not None
    assert orchestrator.db_path is not None

def test_empty_linked_schema_is_a_coverage_gap() -> None:
    """An empty schema must not yield a clean report that would grant unanswerable proof."""
    from zaic_2026_4.submission.pipeline import compute_unanswerable_proof

    intent = detect_query_intent("Tổng doanh thu tháng này?", "2026-09-07")
    _expanded, report = validate_schema_coverage(set(), _temporal_metadata(), intent)

    assert report.has_unresolved_gaps is True
    assert "linked_schema_present" in report.missing_aspects
    assert compute_unanswerable_proof(report) is False


def test_changed_business_input_reingests_the_database(tmp_path: Path) -> None:
    """A changed source file must invalidate the cached SQLite database, not just metadata."""
    import sqlite3

    from zaic_2026_4.profiling.profile_manager import ProfilingOrchestrator
    from zaic_2026_4.profiling.profile_summarizer import MockLLMEngine
    from zaic_2026_4.profiling.vector_indexer import MockEmbedder

    data_dir = tmp_path / "data"
    biz_dir = data_dir / "biz_fresh"
    biz_dir.mkdir(parents=True)
    csv_path = biz_dir / "sales.csv"
    csv_path.write_text("id,ten_sp\n1,Sting\n2,Coca\n", encoding="utf-8")
    rulebook = tmp_path / "rulebook.md"
    rulebook.write_text("# Rulebook\nDoanh thu.", encoding="utf-8")
    cache_dir = tmp_path / "cache"

    def _run() -> int:
        orchestrator = ProfilingOrchestrator(
            business_dir=biz_dir,
            rulebook_path=rulebook,
            llm_engine=MockLLMEngine(),
            embedder=MockEmbedder(),
            vector_model_path="mock",
            cache_dir=cache_dir,
        )
        orchestrator.run_pipeline(force_rebuild=False)
        db_path = orchestrator.cache_dir / "biz_fresh.db"
        with sqlite3.connect(db_path) as conn:
            return int(conn.execute('SELECT COUNT(*) FROM "sales"').fetchone()[0])

    assert _run() == 2

    # Grow the source file: the fingerprint changes and the database must be re-ingested.
    csv_path.write_text(
        "id,ten_sp\n1,Sting\n2,Coca\n3,Pepsi\n4,7Up\n", encoding="utf-8"
    )
    assert _run() == 4


def test_lsh_near_unique_cut_only_applies_to_identifier_columns() -> None:
    """Categorical name columns keep full sample recall; identifier columns are trimmed."""
    indexer = VietnameseLSHIndexer(max_sample_values=100)
    name_values = [f"Hoạt chất {i}" for i in range(20)]
    id_values = [f"SP{i:03d}" for i in range(20)]

    indexer.build_index(
        {
            "bang.ten_hoat_chat": {"data_format": "STRING", "sample_values": name_values},
            "bang.ma_san_pham": {"data_format": "FIXED_CODE", "sample_values": id_values},
        }
    )

    indexed_names = {val for col, val in indexer.value_to_col_map.values() if col.endswith("ten_hoat_chat")}
    indexed_ids = {val for col, val in indexer.value_to_col_map.values() if col.endswith("ma_san_pham")}

    assert len(indexed_names) == 20
    assert 0 < len(indexed_ids) < 20
