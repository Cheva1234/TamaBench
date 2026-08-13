import json

from tamabench.agents.base import BaseAgent
from tamabench.runner.batch_runner import BatchRunner
from tamabench.schemas.actions import ActionProposal


class NoProgressAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="NoProgressAgent")

    def select_action(self, observation):
        proposal = ActionProposal(action="feed")
        return json.dumps(proposal.model_dump(exclude_none=True)), proposal, None


def test_runner_terminates_an_agent_that_repeats_zero_time_actions(tmp_path):
    runner = BatchRunner(
        db_path=str(tmp_path / "stall.db"),
        event_path=str(tmp_path / "stall.jsonl"),
        max_stalled_decisions=3,
    )

    metrics = runner.run_episode(
        agent=NoProgressAgent(),
        seed=42,
        max_simulated_minutes=120,
    )
    runner.close()

    assert metrics.survived is False
    assert metrics.simulated_days == 0.0
    assert metrics.total_decisions == 4
