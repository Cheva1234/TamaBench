"""Unit tests for Harness V1 (WakeScheduler + HarnessV1Agent)."""

import pytest
from tamabench.agents.harness_v1_agent import WakeScheduler, HarnessV1Agent
from tamabench.agents.rule_agent import RuleAgent
from tamabench.env.core import TamaEnv
from tamabench.schemas.actions import ActionProposal


@pytest.mark.unit
def test_wake_scheduler_critical_flags():
    env = TamaEnv()
    obs = env.reset(seed=42)
    scheduler = WakeScheduler()

    # Fresh state: nothing critical
    assert scheduler.is_critical(obs) is False

    # Sick pet is always critical
    obs.pet.is_sick = True
    assert scheduler.is_critical(obs) is True
    obs.pet.is_sick = False

    # Low fullness is critical
    obs.pet.hunger = 40.0
    assert scheduler.is_critical(obs) is True
    obs.pet.hunger = 80.0

    # Low cleanliness is critical
    obs.pet.cleanliness = 30.0
    assert scheduler.is_critical(obs) is True
    obs.pet.cleanliness = 90.0

    # Low agent energy is critical
    obs.agent.energy = 5
    assert scheduler.is_critical(obs) is True


@pytest.mark.unit
def test_wake_scheduler_next_wake_math():
    env = TamaEnv()
    obs = env.reset(seed=42)
    scheduler = WakeScheduler()

    # Feeding raises fullness to 100; next wake when it decays to 50.
    # 50 / 0.30 per minute = ~167 minutes. Give the agent food so the
    # low-food check (60 min) does not override the hunger math.
    obs.inventory.food = 3
    minutes = scheduler.next_wake_minutes(obs, ActionProposal(action="feed"))
    assert 160 <= minutes <= 170

    # Cleaning resets cleanliness to 100; next wake when it decays to 40.
    # 60 / 0.15 per minute = 400 minutes, but agent energy caps it sooner.
    minutes = scheduler.next_wake_minutes(obs, ActionProposal(action="clean"))
    assert minutes > 0

    # Work blocks for the full job duration.
    minutes = scheduler.next_wake_minutes(obs, ActionProposal(action="work", job_id="cafe_shift"))
    assert minutes == 60

    # Sleep blocks for the requested hours.
    minutes = scheduler.next_wake_minutes(obs, ActionProposal(action="sleep", hours=5))
    assert minutes == 300

    # An unexecutable action forces an immediate re-consult (1 minute).
    obs.inventory.food = 0
    minutes = scheduler.next_wake_minutes(obs, ActionProposal(action="feed"))
    assert minutes == 1


@pytest.mark.unit
def test_harness_agent_resolves_noncritical_without_model_calls():
    """The harness must resolve non-critical boundaries with deterministic
    policy actions that carry attempt_count=0 (no API call) and never
    invoke the model."""
    env = TamaEnv()
    obs = env.reset(seed=42)
    agent = HarnessV1Agent(model_agent=RuleAgent())

    raw, proposal, err = agent.select_action(obs)
    assert err is None
    assert proposal is not None
    # First decision always consults the model.
    assert agent.model_decisions == 1

    # Simulate the post-decision state: advance time to the scheduled wake.
    next_wake = agent._next_wake_minute
    assert next_wake is not None and next_wake > obs.time.total_minutes

    # A non-critical observation at a non-wake boundary must be resolved by
    # the deterministic policy with zero model involvement.
    obs2 = env.observe()
    raw2, proposal2, err2 = agent.select_action(obs2)
    assert err2 is None
    assert proposal2 is not None
    assert agent.model_decisions == 1  # model not consulted
    assert agent.scheduled_waits == 1
    assert agent.last_decision.attempt_count == 0
    assert agent.last_decision.total_tokens == 0


@pytest.mark.unit
def test_harness_agent_resets_schedule_on_new_episode():
    """reset_episode() must clear the pending schedule and counters so
    state never leaks between episodes run by the same agent instance."""
    env = TamaEnv()
    obs = env.reset(seed=42)
    agent = HarnessV1Agent(model_agent=RuleAgent())

    agent.select_action(obs)
    assert agent._next_wake_minute is not None
    assert agent.model_decisions == 1

    # New episode: explicit reset hook (called by the runner).
    agent.reset_episode()
    assert agent._next_wake_minute is None
    assert agent.model_decisions == 0
    assert agent.scheduled_waits == 0

    obs2 = env.reset(seed=43)
    agent.select_action(obs2)
    assert agent.model_decisions == 1  # model consulted again
    assert agent._next_wake_minute is not None
