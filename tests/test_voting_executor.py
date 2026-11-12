"""Unit tests for VotingExecutor: SQLite execution, set conversion, and majority voting."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from zaic_2026_4.submission.models import CandidateQuery
from zaic_2026_4.submission.voting_executor import (
    VotingExecutor,
    answer_signature,
    has_evidence,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "test_biz.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE products (id INT, name TEXT, price REAL);")
    conn.execute("INSERT INTO products VALUES (1, 'Bút bi', 5000);")
    conn.execute("INSERT INTO products VALUES (2, 'Vở kẻ ngang', 12000);")
    conn.execute("INSERT INTO products VALUES (3, 'Tẩy chì', 3000);")
    conn.commit()
    conn.close()
    return db_file


def test_execute_query(temp_db: Path) -> None:
    executor = VotingExecutor(db_path=temp_db)

    # Valid query (5-tuple return)
    success, err, raw, r_set, cols = executor.execute_query("SELECT name FROM products WHERE price > 4000;")
    assert success
    assert err is None
    assert raw is not None
    assert len(raw) == 2
    assert r_set == {("Bút bi",), ("Vở kẻ ngang",)}
    assert cols == ["name"]

    # Invalid query
    bad_success, bad_err, _, _, bad_cols = executor.execute_query("SELECT * FROM non_existing_table;")
    assert not bad_success
    assert bad_err is not None
    assert "no such table" in bad_err.lower()
    assert bad_cols == []


def test_voting_majority_2_of_3(temp_db: Path) -> None:
    executor = VotingExecutor(db_path=temp_db)

    c0 = CandidateQuery(candidate_id=0, seed=42, final_sql="SELECT name FROM products WHERE id = 1;", predicted_type="string")
    c1 = CandidateQuery(candidate_id=1, seed=100, final_sql="SELECT name FROM products WHERE price = 5000;", predicted_type="string")
    c2 = CandidateQuery(candidate_id=2, seed=2026, final_sql="SELECT name FROM products WHERE id = 2;", predicted_type="string")

    candidates = [c0, c1, c2]
    executor.execute_candidates(candidates)

    winner, status, reason = executor.vote(candidates)
    assert winner is not None
    assert winner.candidate_id == 0  # min(candidate_id) in agreeing group [0, 1]
    assert status == "majority_2_of_3"
    assert "majority:" in reason
    assert winner.result_set == {("Bút bi",)}


def test_voting_unanimous_3_of_3(temp_db: Path) -> None:
    executor = VotingExecutor(db_path=temp_db)

    c0 = CandidateQuery(candidate_id=0, seed=42, final_sql="SELECT COUNT(*) FROM products;")
    c1 = CandidateQuery(candidate_id=1, seed=100, final_sql="SELECT COUNT(id) FROM products;")
    c2 = CandidateQuery(candidate_id=2, seed=2026, final_sql="SELECT 3;")

    candidates = [c0, c1, c2]
    executor.execute_candidates(candidates)

    winner, status, reason = executor.vote(candidates)
    assert winner is not None
    assert status == "unanimous_3_of_3"
    assert "majority:" in reason
    assert winner.result_set == {(3,)}


def test_voting_abstain_low_agreement_1_1_1(temp_db: Path) -> None:
    """Metric M6: In a 1-1-1 split, agreement is 1/3 <= 0.5, so vote abstains without RNG."""
    executor = VotingExecutor(db_path=temp_db)

    c0 = CandidateQuery(candidate_id=0, seed=42, final_sql="SELECT name FROM products WHERE id = 1;", predicted_type="string")
    c1 = CandidateQuery(candidate_id=1, seed=100, final_sql="SELECT name FROM products WHERE id = 2;", predicted_type="string")
    c2 = CandidateQuery(candidate_id=2, seed=2026, final_sql="SELECT name FROM products WHERE id = 3;", predicted_type="string")

    candidates = [c0, c1, c2]
    executor.execute_candidates(candidates)

    winner, status, reason = executor.vote(candidates)
    assert winner is None
    assert status == "abstain_low_agreement"
    assert "abstain_low_agreement:" in reason
    assert "1/1/1" in reason


def test_voting_all_failed(temp_db: Path) -> None:
    executor = VotingExecutor(db_path=temp_db)

    c0 = CandidateQuery(candidate_id=0, seed=42, final_sql="SELECT * FROM missing1;")
    c1 = CandidateQuery(candidate_id=1, seed=100, final_sql="SELECT * FROM missing2;")

    candidates = [c0, c1]
    executor.execute_candidates(candidates)

    winner, status, reason = executor.vote(candidates)
    assert winner is None
    assert status == "abstain_fallback"
    assert "abstain:" in reason


def test_has_evidence_logic() -> None:
    """Verify evidence detection distinguishes empty/NULL from valid results including zero."""
    assert not has_evidence(None)
    assert not has_evidence(set())
    assert not has_evidence({(None,)})
    assert not has_evidence({(None, None)})
    assert not has_evidence({(None,), (None,)})

    # Legitimate non-None evidence
    assert has_evidence({(0,)})
    assert has_evidence({(0.0,)})
    assert has_evidence({("0",)})
    assert has_evidence({(1, "x")})
    assert has_evidence({(None, "x")})


def test_voting_none_consensus_insufficient_evidence(temp_db: Path) -> None:
    """Metric M4: When all candidates return {(None,)}, vote yields insufficient_evidence instead of unanimous."""
    executor = VotingExecutor(db_path=temp_db)

    c0 = CandidateQuery(candidate_id=0, seed=42, final_sql="SELECT SUM(price) FROM products WHERE id = 99;")
    c1 = CandidateQuery(candidate_id=1, seed=100, final_sql="SELECT AVG(price) FROM products WHERE id = 99;")
    c2 = CandidateQuery(candidate_id=2, seed=2026, final_sql="SELECT MAX(price) FROM products WHERE id = 99;")

    candidates = [c0, c1, c2]
    executor.execute_candidates(candidates)

    winner, status, reason = executor.vote(candidates)
    assert winner is None
    assert status == "insufficient_evidence"
    assert "insufficient_evidence:" in reason


def test_voting_deliberate_unanswerable_consensus() -> None:
    """Verify deliberate abstain and unanswerable candidates achieve unanswerable_consensus."""
    executor = VotingExecutor(db_path="/dummy/path.db")

    c0 = CandidateQuery(
        candidate_id=0,
        seed=42,
        final_sql="",
        predicted_type="unanswerable",
        deliberate_abstain=True,
    )
    c1 = CandidateQuery(
        candidate_id=1,
        seed=100,
        final_sql="",
        predicted_type="unanswerable",
        deliberate_abstain=True,
    )

    candidates = [c0, c1]
    executor.execute_candidates(candidates)
    assert c0.execution_success is True
    assert c1.execution_success is True

    winner, status, _reason = executor.vote(candidates)
    assert winner is not None
    assert winner.predicted_type == "unanswerable"
    assert status == "unanswerable_consensus"


def test_voting_abstain_low_agreement_excludes_none(temp_db: Path) -> None:
    """Verify 1-1 agreement among valid candidates with 1 NULL candidate abstains (1/1 split, ratio <= 0.5)."""
    executor = VotingExecutor(db_path=temp_db)

    c0 = CandidateQuery(candidate_id=0, seed=42, final_sql="SELECT name FROM products WHERE id = 1;", predicted_type="string")
    c1 = CandidateQuery(candidate_id=1, seed=100, final_sql="SELECT name FROM products WHERE id = 2;", predicted_type="string")
    # c2 produces {(None,)} and is excluded from evidence candidates
    c2 = CandidateQuery(candidate_id=2, seed=2026, final_sql="SELECT SUM(price) FROM products WHERE id = 99;", predicted_type="number")

    candidates = [c0, c1, c2]
    executor.execute_candidates(candidates)

    winner, status, _reason = executor.vote(candidates)
    assert winner is None
    assert status == "abstain_low_agreement"


def test_voting_type_aware_separation() -> None:
    """Metric M4: Number and percent producing identical value 2.08 have different signatures."""
    executor = VotingExecutor(db_path="/dummy/path.db")

    # c0: number 2.08
    c0 = CandidateQuery(
        candidate_id=0,
        seed=42,
        final_sql="SELECT 2.08;",
        predicted_type="number",
        execution_success=True,
        raw_result=[(2.08,)],
        result_set={(2.08,)},
    )
    # c1: percent 2.08
    c1 = CandidateQuery(
        candidate_id=1,
        seed=100,
        final_sql="SELECT 2.08;",
        predicted_type="percent",
        execution_success=True,
        raw_result=[(2.08,)],
        result_set={(2.08,)},
    )

    # 1 vs 1 with different types -> 1/1 split -> abstain
    winner, status, _ = executor.vote([c0, c1])
    assert winner is None
    assert status == "abstain_low_agreement"

    # c2: percent 2.08
    c2 = CandidateQuery(
        candidate_id=2,
        seed=2026,
        final_sql="SELECT 2.08;",
        predicted_type="percent",
        execution_success=True,
        raw_result=[(2.08,)],
        result_set={(2.08,)},
    )

    # c1 and c2 agree on percent 2.08 (2/3 > 0.5) -> winner is percent (c1)
    winner_3, status_3, _ = executor.vote([c0, c1, c2])
    assert winner_3 is not None
    assert winner_3.candidate_id == 1
    assert winner_3.predicted_type == "percent"
    assert status_3 == "majority_2_of_3"


def test_voting_list_order_aware() -> None:
    """Metric M5: List order differences produce different signatures."""
    executor = VotingExecutor(db_path="/dummy/path.db")

    c0 = CandidateQuery(
        candidate_id=0,
        seed=42,
        final_sql="SELECT x;",
        predicted_type="list",
        execution_success=True,
        raw_result=[("A",), ("B",)],
        result_set={("A",), ("B",)},
    )
    c1 = CandidateQuery(
        candidate_id=1,
        seed=100,
        final_sql="SELECT x;",
        predicted_type="list",
        execution_success=True,
        raw_result=[("B",), ("A",)],
        result_set={("A",), ("B",)},
    )

    sig0 = answer_signature(c0)
    sig1 = answer_signature(c1)
    assert sig0 != sig1

    winner, status, _ = executor.vote([c0, c1])
    assert winner is None
    assert status == "abstain_low_agreement"


def test_voting_table_row_order_invariant() -> None:
    """Metric M5: Table row order differences canonicalize to the same signature."""
    executor = VotingExecutor(db_path="/dummy/path.db")

    c0 = CandidateQuery(
        candidate_id=0,
        seed=42,
        final_sql="SELECT name, price FROM products;",
        predicted_type="table",
        execution_success=True,
        raw_result=[("A", 1), ("B", 2)],
        result_set={("A", 1), ("B", 2)},
        result_columns=["name", "price"],
    )
    c1 = CandidateQuery(
        candidate_id=1,
        seed=100,
        final_sql="SELECT name, price FROM products;",
        predicted_type="table",
        execution_success=True,
        raw_result=[("B", 2), ("A", 1)],
        result_set={("A", 1), ("B", 2)},
        result_columns=["name", "price"],
    )

    sig0 = answer_signature(c0)
    sig1 = answer_signature(c1)
    assert sig0 == sig1

    winner, status, _ = executor.vote([c0, c1])
    assert winner is not None
    assert status == "unanimous_2_of_2"


def test_voting_single_survivor_allowed() -> None:
    """1 survivor candidate (ratio 1.0 > 0.5) is allowed to win without abstaining."""
    executor = VotingExecutor(db_path="/dummy/path.db")

    c0 = CandidateQuery(
        candidate_id=0,
        seed=42,
        final_sql="SELECT 10;",
        predicted_type="number",
        execution_success=True,
        raw_result=[(10,)],
        result_set={(10,)},
    )
    c1 = CandidateQuery(
        candidate_id=1,
        seed=100,
        final_sql="SELECT bad;",
        predicted_type="number",
        execution_success=False,
    )
    c2 = CandidateQuery(
        candidate_id=2,
        seed=2026,
        final_sql="SELECT null_only;",
        predicted_type="number",
        execution_success=True,
        raw_result=[(None,)],
        result_set={(None,)},
    )

    winner, status, _ = executor.vote([c0, c1, c2])
    assert winner is not None
    assert winner.candidate_id == 0
    assert status == "majority_1_of_3"


def test_voting_deterministic() -> None:
    """Verify voting is strictly deterministic across repeated invocations."""
    executor = VotingExecutor(db_path="/dummy/path.db")

    c0 = CandidateQuery(
        candidate_id=0,
        seed=42,
        final_sql="SELECT 10;",
        predicted_type="number",
        execution_success=True,
        raw_result=[(10,)],
        result_set={(10,)},
    )
    c1 = CandidateQuery(
        candidate_id=1,
        seed=100,
        final_sql="SELECT 10;",
        predicted_type="number",
        execution_success=True,
        raw_result=[(10,)],
        result_set={(10,)},
    )

    res1 = executor.vote([c0, c1])
    res2 = executor.vote([c0, c1])
    assert res1 == res2
