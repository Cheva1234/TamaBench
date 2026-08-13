import json

from tamabench.agents.base import BaseAgent
from tamabench.runner.batch_runner import BatchRunner
from tamabench.schemas.actions import ActionProposal


class BlockingActionAgent(BaseAgent):
    def __init__(self, action: ActionProposal):
        super().__init__("BlockingActionAgent")
        self.action = action
        self.calls = 0

    def select_action(self, observation):
        self.calls += 1
        return json.dumps(self.action.model_dump(exclude_none=True)), self.action, None


def test_work_does_not_poll_agent_during_blocking_time_skip(tmp_path):
    agent = BlockingActionAgent(ActionProposal(action="work", job_id="delivery"))
    runner = BatchRunner(
        db_path=str(tmp_path / "boundary.db"),
        event_path=str(tmp_path / "boundary.jsonl"),
    )

    metrics = runner.run_episode(agent, seed=42, max_simulated_minutes=121)
    runner.close()

    assert metrics.total_decisions == agent.calls
    assert agent.calls == 2
    assert metrics.api_calls == 0
