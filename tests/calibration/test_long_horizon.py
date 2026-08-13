"""Calibration Test: Long-Horizon 7-Day Determinism Audit across 100 Seeds.

Runs 100 seeds for 7 simulated days each (10,080 simulation minutes/episode = >1,000,000 total minutes).
Replays all recorded episodes and verifies 100.0% SHA-256 state_hash matching rate.
"""

import pytest
from tamabench.agents.rule_agent import RuleAgent
from tamabench.runner.batch_runner import BatchRunner
from tamabench.logging.replay import ReplayEngine


@pytest.mark.calibration
def test_long_horizon_100_seeds_determinism(tmp_path):
    db_file = str(tmp_path / "long_horizon.db")
    event_file = str(tmp_path / "long_horizon_events.jsonl")

    runner = BatchRunner(db_path=db_file, event_path=event_file)
    replay_engine = ReplayEngine(db_path=db_file)

    seeds = list(range(1, 101))  # 100 seeds
    seven_days_minutes = 10080  # 7 simulated days

    failed_replays = []

    for seed in seeds:
        agent = RuleAgent()
        metrics = runner.run_episode(
            agent=agent,
            seed=seed,
            max_simulated_minutes=seven_days_minutes,
        )

        success, mismatches = replay_engine.replay_run(metrics.run_id)
        if not success:
            failed_replays.append((seed, metrics.run_id, mismatches))

    runner.close()

    assert len(failed_replays) == 0, f"Long horizon replay failed for {len(failed_replays)} seed(s): {failed_replays[:3]}"
