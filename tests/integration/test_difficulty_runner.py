from tamabench.agents.rule_agent import RuleAgent
from tamabench.runner.batch_runner import BatchRunner


def test_runner_uses_easy_difficulty_horizon(tmp_path):
    runner = BatchRunner(
        db_path=str(tmp_path / "difficulty.db"),
        event_path=str(tmp_path / "difficulty.jsonl"),
    )

    metrics = runner.run_episode(agent=RuleAgent(), seed=42, difficulty="easy")
    runner.close()

    # Blocking actions commit atomically, so the final action may cross the
    # one-day boundary by at most its duration.
    assert 1.0 <= metrics.simulated_days <= 1.1
    assert metrics.survived is True
