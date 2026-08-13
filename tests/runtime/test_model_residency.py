import json

from tamabench.agents.base import BaseAgent
from tamabench.runner.batch_runner import BatchRunner
from tamabench.schemas.actions import ActionProposal


class ResidentFakeAgent(BaseAgent):
    model_name = "resident-fake"
    model_resident = False

    def __init__(self):
        super().__init__("ResidentFakeAgent")
        self.warmup_calls = 0
        self.close_calls = 0
        self.action = ActionProposal(action="wait", minutes=30)

    def warmup(self):
        self.warmup_calls += 1
        self.model_resident = True
        return 1.0

    def close(self):
        self.close_calls += 1
        self.model_resident = False

    def select_action(self, observation):
        return json.dumps(self.action.model_dump(exclude_none=True)), self.action, None


def test_runner_warms_shared_agent_once_for_multiple_episodes(tmp_path):
    agent = ResidentFakeAgent()
    runner = BatchRunner(
        db_path=str(tmp_path / "resident.db"),
        event_path=str(tmp_path / "resident.jsonl"),
    )

    runner.run_episode(agent, seed=1, max_simulated_minutes=30)
    runner.run_episode(agent, seed=2, max_simulated_minutes=30)
    runner.close()

    assert agent.warmup_calls == 1
    assert agent.close_calls == 1
