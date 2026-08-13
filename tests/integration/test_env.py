"""Unit tests for TamaEnv, dynamics, and seed reproducibility."""

import pytest
from tamabench.env.core import TamaEnv
from tamabench.schemas.actions import ActionProposal
from tamabench.schemas.errors import ErrorCategory, ErrorType


@pytest.mark.integration
def test_env_reset_seed_reproducibility():
    env1 = TamaEnv()
    obs1 = env1.reset(seed=1842)

    env2 = TamaEnv()
    obs2 = env2.reset(seed=1842)

    assert obs1.state_hash == obs2.state_hash
    assert obs1.pet.health == obs2.pet.health
    assert obs1.agent.money == obs2.agent.money


@pytest.mark.integration
def test_delta_time_equivalence():
    env1 = TamaEnv()
    env1.reset(seed=42)
    env1.advance_time(120)

    env2 = TamaEnv()
    env2.reset(seed=42)
    env2.advance_time(60)
    env2.advance_time(60)

    assert env1.observe().state_hash == env2.observe().state_hash


@pytest.mark.integration
def test_feed_action_precondition_and_effects():
    env = TamaEnv()
    env.reset(seed=42)

    # Valid feed
    initial_food = env.state.inventory.food
    result = env.step(ActionProposal(action="feed"))
    assert result.success is True
    assert env.state.inventory.food == initial_food - 1

    # Deplete food inventory
    env.state.inventory.food = 0
    result_empty = env.step(ActionProposal(action="feed"))
    assert result_empty.success is False
    assert result_empty.error is not None
    assert result_empty.error.category == ErrorCategory.ENVIRONMENT
    assert result_empty.error.error_type == ErrorType.INSUFFICIENT_RESOURCE


@pytest.mark.integration
def test_work_and_buy_economy():
    env = TamaEnv()
    env.reset(seed=42)
    initial_money = env.state.agent.money

    # Perform cafe shift (+25)
    res_work = env.step(ActionProposal(action="work", job_id="cafe_shift"))
    assert res_work.success is True
    assert env.state.agent.money == initial_money + 25

    # Buy 1 food item (cost 30)
    res_buy = env.step(ActionProposal(action="buy", item="food", amount=1))
    assert res_buy.success is True
    assert env.state.agent.money == initial_money + 25 - 30
    assert env.state.inventory.food == 2
