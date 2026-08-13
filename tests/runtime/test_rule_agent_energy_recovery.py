from tamabench.agents.rule_agent import RuleAgent
from tamabench.env.core import TamaEnv


def test_rule_agent_sleeps_when_no_job_is_executable_due_to_energy():
    env = TamaEnv()
    observation = env.reset(seed=42)
    env.state.agent.energy = 0
    observation = env.observe()

    _, proposal, error = RuleAgent().select_action(observation)

    assert error is None
    assert proposal is not None
    assert proposal.action == "sleep"
