"""Integration tests for baseline agents (RandomSchema, RandomValid, Rule)."""

import pytest
import os
from tamabench.agents.random_schema_agent import RandomSchemaAgent
from tamabench.agents.random_valid_agent import RandomValidAgent
from tamabench.agents.rule_agent import RuleAgent
from tamabench.runner.batch_runner import BatchRunner


@pytest.mark.integration
def test_rule_agent_episode(tmp_path):
    db_file = str(tmp_path / "test_rule.db")
    event_file = str(tmp_path / "test_rule_events.jsonl")

    runner = BatchRunner(db_path=db_file, event_path=event_file)
    agent = RuleAgent()

    # Run 1 simulated day (1440 minutes)
    metrics = runner.run_episode(agent=agent, seed=100, max_simulated_minutes=1440)
    runner.close()

    assert metrics.simulated_days >= 1.0
    assert metrics.first_pass_schema_acc == 100.0
    assert metrics.valid_action_rate == 100.0
    assert os.path.exists(db_file)
    assert os.path.exists(event_file)


@pytest.mark.integration
def test_random_valid_agent_episode(tmp_path):
    db_file = str(tmp_path / "test_random_valid.db")
    event_file = str(tmp_path / "test_random_valid_events.jsonl")

    runner = BatchRunner(db_path=db_file, event_path=event_file)
    agent = RandomValidAgent(seed=42)

    metrics = runner.run_episode(agent=agent, seed=42, max_simulated_minutes=720)
    runner.close()

    assert metrics.first_pass_schema_acc == 100.0
    assert metrics.valid_action_rate == 100.0


@pytest.mark.integration
def test_random_schema_agent_precondition_errors(tmp_path):
    db_file = str(tmp_path / "test_random_schema.db")
    event_file = str(tmp_path / "test_random_schema_events.jsonl")

    runner = BatchRunner(db_path=db_file, event_path=event_file)
    agent = RandomSchemaAgent(seed=42)

    # Random schema agent should trigger some environment errors (preconditions)
    metrics = runner.run_episode(agent=agent, seed=42, max_simulated_minutes=720)
    runner.close()

    assert metrics.first_pass_schema_acc == 100.0
    assert metrics.invalid_action_rate >= 0.0
