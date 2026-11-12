"""Unit tests for VotingExecutor: SQLite execution, set conversion, and majority voting."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from zaic_2026_4.submission.models import CandidateQuery
from zaic_2026_4.submission.voting_executor import VotingExecutor, has_evidence


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

    # Valid query
    success, err, raw, r_set = executor.execute_query("SELECT name FROM products WHERE price > 4000;")
    assert success
    assert err is None
    assert raw is not None
    assert len(raw) == 2
    assert r_set == {("Bút bi",), ("Vở kẻ ngang",)}

    # Invalid query
    bad_success, bad_err, _, _ = executor.execute_query("SELECT * FROM non_existing_table;")
    assert not bad_success
    assert bad_err is not None
    assert "no such table" in bad_err.lower()


def test_voting_majority_2_of_3(temp_db: Path) -> None:
    executor = VotingExecutor(db_path=temp_db)

    c0 = CandidateQuery(candidate_id=0, seed=42, final_sql="SELECT name FROM products WHERE id = 1;")
    c1 = CandidateQuery(candidate_id=1, seed=100, final_sql="SELECT name FROM products WHERE price = 5000;")
    c2 = CandidateQuery(candidate_id=2, seed=2026, final_sql="SELECT name FROM products WHERE id = 2;")

    candidates = [c0, c1, c2]
    executor.execute_candidates(candidates)

    winner, status, reason = executor.vote(candidates)
    assert winner is not None
    assert winner.candidate_id in (0, 1)
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


def test_voting_random_tiebreak(temp_db: Path) -> None:
    executor = VotingExecutor(db_path=temp_db)

    c0 = CandidateQuery(candidate_id=0, seed=42, final_sql="SELECT name FROM products WHERE id = 1;")
    c1 = CandidateQuery(candidate_id=1, seed=100, final_sql="SELECT name FROM products WHERE id = 2;")
    c2 = CandidateQuery(candidate_id=2, seed=2026, final_sql="SELECT name FROM products WHERE id = 3;")

    candidates = [c0, c1, c2]
    executor.execute_candidates(candidates)

    winner, status, reason = executor.vote(candidates)
    assert winner is not None
    assert status == "random_tiebreak"
    assert "random:" in reason
    assert winner.candidate_id in (0, 1, 2)


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


def test_voting_tiebreak_excludes_none_candidate(temp_db: Path) -> None:
    """Verify random tiebreak only selects from candidates that produced non-None evidence."""
    executor = VotingExecutor(db_path=temp_db)

    c0 = CandidateQuery(candidate_id=0, seed=42, final_sql="SELECT name FROM products WHERE id = 1;")
    c1 = CandidateQuery(candidate_id=1, seed=100, final_sql="SELECT name FROM products WHERE id = 2;")
    # c2 produces {(None,)}
    c2 = CandidateQuery(candidate_id=2, seed=2026, final_sql="SELECT SUM(price) FROM products WHERE id = 99;")

    candidates = [c0, c1, c2]
    executor.execute_candidates(candidates)

    winner, status, _reason = executor.vote(candidates)
    assert winner is not None
    assert winner.candidate_id in (0, 1)  # Never picks candidate 2!
    assert status == "random_tiebreak"
