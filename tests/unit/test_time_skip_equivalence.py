"""Unit tests for Time-Skip Fast-Forward Engine & Equivalence Verification."""

import pytest
from tamabench.env.core import TamaEnv
from tamabench.schemas.actions import ActionProposal


@pytest.mark.unit
def test_tick_vs_fast_forward_equivalence():
    """Verifies event-driven fast-forward advance_until(180) matches 180 individual 1-minute ticks."""
    env1 = TamaEnv()
    env1.reset(seed=42)
    env1.advance_until(180)

    env2 = TamaEnv()
    env2.reset(seed=42)
    for _ in range(180):
        env2.advance_time(1)

    assert env1.observe().state_hash == env2.observe().state_hash
    assert env1.state.pet.health == env2.state.pet.health
    assert env1.state.pet.hunger == env2.state.pet.hunger


@pytest.mark.unit
def test_work_time_skip_action():
    """Verifies work(cafe_shift) fast-forwards 60 minutes in a single block while maintaining valid state and money reward."""
    env = TamaEnv()
    env.reset(seed=42)
    initial_money = env.state.agent.money
    initial_minutes = env.state.total_minutes

    res = env.step(ActionProposal(action="work", job_id="cafe_shift"))
    assert res.success is True
    assert res.execution_minutes == 60
    assert env.state.total_minutes == initial_minutes + 60
    # Dynamic economy gives varying reward based on total_minutes
    expected_reward = 60
    assert env.state.agent.money == initial_money + expected_reward
    assert env.state.agent.current_activity == "idle"


@pytest.mark.unit
@pytest.mark.parametrize("sleep_hours, expected_mins", [(3, 180), (5, 300), (8, 480)])
def test_sleep_energy_recovery(sleep_hours, expected_mins):
    """Verifies sleep action with custom hours (3, 5, 8) recovers agent energy (+0.5/min)."""
    env = TamaEnv()
    env.reset(seed=42)
    env.state.agent.energy = 20.0
    initial_energy = env.state.agent.energy

    res = env.step(ActionProposal(action="sleep", hours=sleep_hours))
    assert res.success is True
    assert res.execution_minutes == expected_mins
    assert env.state.agent.energy > initial_energy
    assert env.state.agent.current_activity == "idle"
