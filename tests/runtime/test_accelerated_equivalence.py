from tamabench.env.core import TamaEnv
from tamabench.env.time_engine import BenchmarkMode
from tamabench.schemas.actions import ActionProposal


def _run_fixed_actions(mode: BenchmarkMode) -> TamaEnv:
    env = TamaEnv(mode=mode)
    env.reset(seed=1842)
    for action in (
        ActionProposal(action="work", job_id="cafe_shift"),
        ActionProposal(action="feed"),
        ActionProposal(action="wait", minutes=120),
        ActionProposal(action="sleep", hours=3),
    ):
        result = env.commit(action)
        assert result.success is True
    return env


def test_accelerated_and_reference_modes_have_identical_state_and_events():
    reference = _run_fixed_actions(BenchmarkMode.LOGICAL)
    accelerated = _run_fixed_actions(BenchmarkMode.ACCELERATED)

    assert accelerated.state.compute_hash() == reference.state.compute_hash()
    assert accelerated.event_history == reference.event_history
    assert accelerated.terminated == reference.terminated
    assert accelerated.state.agent.money == reference.state.agent.money


def test_wait_is_a_single_blocking_time_skip():
    env = TamaEnv(mode=BenchmarkMode.ACCELERATED)
    env.reset(seed=42)

    result = env.commit(ActionProposal(action="wait", minutes=120))

    assert result.success is True
    assert result.execution_minutes == 120
    assert env.state.total_minutes == 120
