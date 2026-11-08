"""Unit tests for VotingExecutor: SQLite execution, set conversion, and majority voting."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from zaic_2026_4.submission.models import CandidateQuery
from zaic_2026_4.submission.voting_executor import VotingExecutor


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
