"""Integration tests for Database Store, Event Stream, and Replay Engine."""

import os
import pytest
from tamabench.agents.rule_agent import RuleAgent
from tamabench.runner.batch_runner import BatchRunner
from tamabench.logging.replay import ReplayEngine


@pytest.mark.integration
def test_logging_and_replay_verification(tmp_path):
    db_file = str(tmp_path / "replay_test.db")
    event_file = str(tmp_path / "replay_test_events.jsonl")

    runner = BatchRunner(db_path=db_file, event_path=event_file)
    agent = RuleAgent()

    # Run episode
    metrics = runner.run_episode(agent=agent, seed=1842, max_simulated_minutes=720)
    run_id = metrics.run_id
    runner.close()

    # Verify SQLite DB has recorded decisions
    engine = ReplayEngine(db_path=db_file)
    decisions = engine.db.get_run_decisions(run_id)
    assert len(decisions) > 0

    # Execute Replay Verification (Check SHA-256 state_hash trajectory)
    success, mismatches = engine.replay_run(run_id)
    assert success is True, f"Replay failed with mismatches: {mismatches}"
    assert len(mismatches) == 0
