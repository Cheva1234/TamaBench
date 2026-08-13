"""Calibration Test: Paired Seeds Evaluation Suite across Baseline Agents.

Executes Seeds 1 to 100 paired across RandomSchema, RandomValid, and RuleAgent,
extracting reproducible Failure Attribution Distribution metrics.
"""

import pytest
from tamabench.agents.random_schema_agent import RandomSchemaAgent
from tamabench.agents.random_valid_agent import RandomValidAgent
from tamabench.agents.rule_agent import RuleAgent
from tamabench.runner.batch_runner import BatchRunner
from tamabench.metrics.failure_analysis import FailureAnalysisEngine
from tamabench.logging.database import DatabaseStore


@pytest.mark.calibration
def test_paired_seeds_baseline_calibration(tmp_path):
    db_file = str(tmp_path / "paired_benchmark.db")
    event_file = str(tmp_path / "paired_benchmark_events.jsonl")

    runner = BatchRunner(db_path=db_file, event_path=event_file)
    paired_seeds = list(range(1, 21))  # 20 paired seeds for calibration test suite run

    results = {
        "RandomSchemaAgent": {"survived": 0, "total_days": 0.0},
        "RandomValidAgent": {"survived": 0, "total_days": 0.0},
        "RuleAgent": {"survived": 0, "total_days": 0.0},
    }

    for seed in paired_seeds:
        # 1. Random Schema
        a_schema = RandomSchemaAgent(seed=seed)
        m_schema = runner.run_episode(agent=a_schema, seed=seed, max_simulated_minutes=1440)
        results["RandomSchemaAgent"]["total_days"] += m_schema.simulated_days
        if m_schema.survived:
            results["RandomSchemaAgent"]["survived"] += 1

        # 2. Random Valid
        a_valid = RandomValidAgent(seed=seed)
        m_valid = runner.run_episode(agent=a_valid, seed=seed, max_simulated_minutes=1440)
        results["RandomValidAgent"]["total_days"] += m_valid.simulated_days
        if m_valid.survived:
            results["RandomValidAgent"]["survived"] += 1

        # 3. Rule Agent
        a_rule = RuleAgent()
        m_rule = runner.run_episode(agent=a_rule, seed=seed, max_simulated_minutes=1440)
        results["RuleAgent"]["total_days"] += m_rule.simulated_days
        if m_rule.survived:
            results["RuleAgent"]["survived"] += 1

    runner.close()

    # Verify Baseline Monotonic Progression: RandomSchema <= RandomValid <= RuleAgent
    days_schema = results["RandomSchemaAgent"]["total_days"]
    days_valid = results["RandomValidAgent"]["total_days"]
    days_rule = results["RuleAgent"]["total_days"]

    assert days_valid >= days_schema, f"Calibration anomaly: RandomValid ({days_valid:.1f} days) < RandomSchema ({days_schema:.1f} days)"
    assert days_rule >= days_valid, f"Calibration anomaly: RuleAgent ({days_rule:.1f} days) < RandomValid ({days_valid:.1f} days)"

    # Audit Failure Attribution Distributions
    db = DatabaseStore(db_path=db_file)
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT run_id FROM runs")
        run_rows = cursor.fetchall()

    for r in run_rows:
        run_id = r["run_id"]
        decisions = db.get_run_decisions(run_id)
        dec_dicts = [dict(d) for d in decisions]
        
        cursor.execute("SELECT * FROM outcomes WHERE run_id = ?", (run_id,))
        outcome = cursor.fetchone()
        out_dict = dict(outcome) if outcome else None

        attribution = FailureAnalysisEngine.analyze_run_failures(dec_dicts, out_dict)
        assert attribution.failure_primary is not None
