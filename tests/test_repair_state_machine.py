"""Unit tests for the unified self-correction state machine in candidate_generator."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from zaic_2026_4.submission.candidate_generator import CandidateGenerator


def test_scenario_1_clean_turn_0_with_warning_does_not_retry(tmp_path: Path) -> None:
    """Scenario 1: Valid query with warning (ORDER_ASC_NULLABLE) terminates at turn 0 without LLM retries (Metric M2)."""
    db_file = tmp_path / "test_s1.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE products (id INT, price REAL);")
    conn.execute("INSERT INTO products VALUES (1, 10000), (2, 20000);")
    conn.commit()
    conn.close()

    class WarningCleanEngine:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompts: Any, **kwargs: Any) -> str:
            self.calls += 1
            return '{"sql": "SELECT id, price FROM products ORDER BY price ASC;", "type": "table"}'

    engine = WarningCleanEngine()
    gen = CandidateGenerator(llm_engine=engine, seeds=(42,), db_path=db_file)
    cand = gen.generate_candidate(
        candidate_id=0,
        seed=42,
        question="Danh sách sản phẩm?",
        as_of_date="2026-05-25",
        reduced_schema={"products.id", "products.price"},
        processed_metadata={
            "products.id": {"data_format": "NUMERIC"},
            "products.price": {"data_format": "NUMERIC", "null_count": 2},
        },
        db_path=db_file,
    )

    assert engine.calls == 1  # 0 retries!
    assert cand.is_valid_syntax is True
    assert cand.execution_success is True
    assert cand.sql_retry_count == 0
    assert cand.execution_retry_count == 0
    assert cand.repair_stop_reason == "clean"


def test_scenario_2_canonical_sql_stagnation_stops_early(tmp_path: Path) -> None:
    """Scenario 2: When model repeats canonical SQL, stagnation stops early after <= 2 attempts (Metric M2)."""
    db_file = tmp_path / "test_s2.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE t (a INT);")
    conn.commit()
    conn.close()

    class StagnantEngine:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompts: Any, **kwargs: Any) -> str:
            self.calls += 1
            # Returns identical SQL with minor spacing variation
            if self.calls == 1:
                return '{"sql": "SELECT bad_col FROM t;", "type": "number"}'
            return '{"sql": "SELECT   bad_col   FROM   t ;", "type": "number"}'

    engine = StagnantEngine()
    gen = CandidateGenerator(
        llm_engine=engine,
        seeds=(42,),
        db_path=db_file,
        sql_max_retries=5,  # Generous budget
    )
    cand = gen.generate_candidate(
        candidate_id=0,
        seed=42,
        question="Test?",
        as_of_date="2026-05-25",
        reduced_schema={"t.a"},
        processed_metadata={"t.a": {"data_format": "NUMERIC"}},
        db_path=db_file,
    )

    assert engine.calls == 2  # Stops at attempt 2 due to stagnation, not 5!
    assert cand.repair_stop_reason == "stagnation"
    assert cand.sql_retry_count == 1


def test_scenario_3_successful_repair_at_turn_2(tmp_path: Path) -> None:
    """Scenario 3: Faulty query in turn 0 successfully repaired in turn 1."""
    db_file = tmp_path / "test_s3.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE orders (id INT, total INT);")
    conn.execute("INSERT INTO orders VALUES (1, 50000), (2, 70000);")
    conn.commit()
    conn.close()

    class SuccessfulRepairEngine:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompts: Any, **kwargs: Any) -> str:
            self.calls += 1
            if self.calls == 1:
                # Turn 0: unknown column
                return '{"sql": "SELECT SUM(tong_tien) FROM orders;", "type": "number"}'
            # Turn 1: fixed column
            return '{"sql": "SELECT SUM(total) FROM orders;", "type": "number"}'

    engine = SuccessfulRepairEngine()
    gen = CandidateGenerator(llm_engine=engine, seeds=(42,), db_path=db_file)
    cand = gen.generate_candidate(
        candidate_id=0,
        seed=42,
        question="Tổng doanh thu?",
        as_of_date="2026-05-25",
        reduced_schema={"orders.id", "orders.total"},
        processed_metadata={
            "orders.id": {"data_format": "NUMERIC"},
            "orders.total": {"data_format": "NUMERIC"},
        },
        db_path=db_file,
    )

    assert engine.calls == 2
    assert cand.sql_retry_count == 1
    assert cand.execution_success is True
    assert cand.raw_result == [(120000,)]
    assert cand.repair_stop_reason == "clean"


def test_scenario_4_last_retry_broken_preserves_best_executed(tmp_path: Path) -> None:
    """Scenario 4: When last retry introduces a syntax error, best candidate with execution results is preserved."""
    db_file = tmp_path / "test_s4.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE sales (id INT, qty INT);")
    conn.execute("INSERT INTO sales VALUES (1, 15), (2, 25);")
    conn.commit()
    conn.close()

    class RegressingEngine:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompts: Any, **kwargs: Any) -> str:
            self.calls += 1
            if self.calls == 1:
                # Turn 0: valid execution, but with a warning (ORDER_ASC_NULLABLE)
                return '{"sql": "SELECT SUM(qty) FROM sales;", "type": "number"}'
            # Turn 1: syntax error introduced
            return '{"sql": "SELECT FROM WHERE ;", "type": "number"}'

    engine = RegressingEngine()
    gen = CandidateGenerator(
        llm_engine=engine,
        seeds=(42,),
        db_path=db_file,
        sql_max_retries=1,
        execution_max_retries=1,
    )
    # Simulate Turn 0 having a warning that triggers retry if wanted
    cand = gen.generate_candidate(
        candidate_id=0,
        seed=42,
        question="Tổng số lượng?",
        as_of_date="2026-05-25",
        reduced_schema={"sales.id", "sales.qty"},
        processed_metadata={
            "sales.id": {"data_format": "NUMERIC"},
            "sales.qty": {"data_format": "NUMERIC"},
        },
        db_path=db_file,
    )

    # In turn 0, clean execution succeeded so it preserved that best result
    assert cand.execution_success is True
    assert cand.raw_result == [(40,)]
    assert "SELECT SUM(qty) FROM sales;" in cand.final_sql


def test_scenario_5_deliberate_abstain_stops_immediately() -> None:
    """Scenario 5: Deliberate abstain or unanswerable prediction stops immediately."""
    class AbstainEngine:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompts: Any, **kwargs: Any) -> str:
            self.calls += 1
            return '{"sql": "", "type": "unanswerable", "reasoning": "Dữ liệu không đủ."}'

    engine = AbstainEngine()
    gen = CandidateGenerator(llm_engine=engine, seeds=(42,))
    cand = gen.generate_candidate(
        candidate_id=0,
        seed=42,
        question="Tỷ lệ lạm phát năm 2050?",
        as_of_date="2026-05-25",
        reduced_schema=set(),
        processed_metadata={},
    )

    assert engine.calls == 1
    assert cand.deliberate_abstain is True
    assert cand.predicted_type == "unanswerable"
    assert cand.final_sql == ""
    assert cand.repair_stop_reason == "deliberate_abstain"


def test_scenario_6_unexplained_zero_stops_without_evidence_issues(tmp_path: Path) -> None:
    """Scenario 6: Clean query returning 0 stops with unexplained_zero=True and 0 retries (Metric M3)."""
    db_file = tmp_path / "test_s6.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE orders (id INT, status TEXT);")
    conn.execute("INSERT INTO orders VALUES (1, 'active');")
    conn.commit()
    conn.close()

    class CleanZeroEngine:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompts: Any, **kwargs: Any) -> str:
            self.calls += 1
            return '{"sql": "SELECT COUNT(*) FROM orders WHERE status = \'cancelled\';", "type": "number"}'

    engine = CleanZeroEngine()
    gen = CandidateGenerator(
        llm_engine=engine,
        seeds=(42,),
        db_path=db_file,
        execution_max_retries=2,
    )
    cand = gen.generate_candidate(
        candidate_id=0,
        seed=42,
        question="Số đơn bị hủy?",
        as_of_date="2026-05-25",
        reduced_schema={"orders.id", "orders.status"},
        processed_metadata={},
        db_path=db_file,
    )

    assert engine.calls == 1  # 0 retries on clean legitimate zero!
    assert cand.execution_success is True
    assert cand.raw_result == [(0,)]
    assert cand.unresolved_zero is True
    assert cand.repair_stop_reason == "unexplained_zero"
    assert cand.execution_retry_count == 0
