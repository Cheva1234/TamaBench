import pytest

from tamabench.env.core import TamaEnv
from tamabench.env.time_engine import BenchmarkMode
from tamabench.metrics.live_reporter import LiveReporter
from tamabench.schemas.actions import ActionProposal


def test_live_reporter_exposes_idle_and_generating_model_states():
    reporter = LiveReporter()

    reporter.set_model_status("generating")
    assert reporter.model_status == "generating"

    reporter.set_model_status("idle")
    assert reporter.model_status == "idle"

    with pytest.raises(ValueError):
        reporter.set_model_status("running")


def test_live_reporter_uses_observed_economy_values():
    env = TamaEnv(mode=BenchmarkMode.ACCELERATED)
    env.reset(seed=42)
    reporter = LiveReporter()

    work_observation = env.observe()
    work_proposal = ActionProposal(action="work", job_id="cafe_shift")
    work_result = env.commit(work_proposal)
    reporter.update(work_observation, 1, work_proposal, work_result)

    buy_observation = env.observe()
    buy_proposal = ActionProposal(action="buy", item="food", amount=1)
    buy_result = env.commit(buy_proposal)
    reporter.update(buy_observation, 2, buy_proposal, buy_result)

    assert reporter.total_income == 25
    assert reporter.total_spending == 30
