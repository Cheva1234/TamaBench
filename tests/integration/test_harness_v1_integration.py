"""Integration tests for Harness V1: call-count reduction vs raw model.

The harness resolves non-critical decision boundaries with deterministic
actions, so a full episode makes fewer model decisions than the raw agent
while keeping the pet alive at least as long.

With a dense optimal policy (RuleAgent) the harness matches raw exactly —
every boundary is actionable, so there is nothing to skip. The reduction
shows with a lazier model that would otherwise waste calls on wait/hesitate
decisions; the harness catches those and rescues the pet.
"""

import json

import pytest

from tamabench.agents.base import BaseAgent
from tamabench.agents.harness_v1_agent import HarnessV1Agent
from tamabench.agents.rule_agent import RuleAgent
from tamabench.runner.batch_runner import BatchRunner
from tamabench.schemas.actions import ActionProposal


class LazyAgent(BaseAgent):
    """A model that always waits — never works, never buys, never feeds
    proactively. Used to prove the harness provides autonomy the model
    lacks."""

    def __init__(self):
        super().__init__(name="LazyLLM")

    def select_action(self, observation):
        proposal = ActionProposal(action="wait", minutes=60)
        return json.dumps(proposal.model_dump(exclude_none=True)), proposal, None


@pytest.mark.integration
def test_harness_v1_episode_runs_and_survives(tmp_path):
    db_file = str(tmp_path / "test_harness.db")
    event_file = str(tmp_path / "test_harness_events.jsonl")

    runner = BatchRunner(db_path=db_file, event_path=event_file)
    agent = HarnessV1Agent(model_agent=RuleAgent())

    metrics = runner.run_episode(agent=agent, seed=100, max_simulated_minutes=1440)
    runner.close()

    assert metrics.simulated_days >= 1.0
    assert metrics.first_pass_schema_acc == 100.0
    assert metrics.valid_action_rate == 100.0
    # Wait decisions are schema-valid and env-valid by construction.
    assert metrics.total_decisions > 0


@pytest.mark.integration
def test_harness_v1_matches_rule_agent_survival(tmp_path):
    """With a dense optimal policy the harness must not hurt survival or
    inflate the decision count."""
    db_file = str(tmp_path / "test_harness_parity.db")
    event_file = str(tmp_path / "test_harness_parity_events.jsonl")

    runner = BatchRunner(db_path=db_file, event_path=event_file)
    raw_agent = RuleAgent()
    harness_agent = HarnessV1Agent(model_agent=RuleAgent())

    raw_metrics = runner.run_episode(agent=raw_agent, seed=100, max_simulated_minutes=1440)
    harness_metrics = runner.run_episode(agent=harness_agent, seed=100, max_simulated_minutes=1440)
    runner.close()

    assert harness_metrics.simulated_days >= raw_metrics.simulated_days
    assert harness_metrics.total_decisions <= raw_metrics.total_decisions + 1


@pytest.mark.integration
def test_harness_v1_reduces_calls_and_rescues_lazy_model(tmp_path):
    """Same lazy model, same seed: the harness must consult the model less
    often than raw and keep the pet alive at least as long."""
    db_file = str(tmp_path / "test_harness_lazy.db")
    event_file = str(tmp_path / "test_harness_lazy_events.jsonl")

    runner = BatchRunner(db_path=db_file, event_path=event_file)
    raw_agent = LazyAgent()
    harness_agent = HarnessV1Agent(model_agent=LazyAgent())

    raw_metrics = runner.run_episode(agent=raw_agent, seed=100, max_simulated_minutes=1440)
    harness_metrics = runner.run_episode(agent=harness_agent, seed=100, max_simulated_minutes=1440)
    runner.close()

    # The harness keeps the pet alive at least as long as raw (rescue).
    assert harness_metrics.simulated_days >= raw_metrics.simulated_days
    # The harness consults the model fewer times than raw makes decisions.
    assert harness_agent.model_decisions < raw_metrics.total_decisions
