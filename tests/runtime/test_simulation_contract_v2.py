"""Behavioral 2.0 invariants: replay, exact welfare, and boundary semantics."""

from dataclasses import asdict
import random

import pytest

from tamabench.env.core import TamaEnv
from tamabench.env.dynamics import DynamicsEngine
from tamabench.env.economy import EconomySystem
from tamabench.env.scenarios import get_config_for_scenario
from tamabench.env.time_engine import BenchmarkMode
from tamabench.schemas.actions import ActionProposal
from tamabench.validation.env_validator import EnvironmentValidator


def _env(mode=BenchmarkMode.ACCELERATED, seed=42, horizon=None):
    env = TamaEnv(mode)
    env.reset(seed=seed, max_simulated_minutes=horizon)
    return env


def _assert_equal(reference, accelerated):
    assert asdict(accelerated.state) == asdict(reference.state)
    assert accelerated.event_history == reference.event_history
    assert accelerated.terminated == reference.terminated
    assert accelerated.horizon_reached == reference.horizon_reached
    assert accelerated.state.compute_hash() == reference.state.compute_hash()


def test_seed14_sleep_recovery_regression():
    """Before 2.0: logical 84.3 versus accelerated 77.7 health at minute 441."""
    reference = _env(BenchmarkMode.LOGICAL, 14)
    accelerated = _env(BenchmarkMode.ACCELERATED, 14)
    actions = [
        ActionProposal(action="sleep", hours=3),
        ActionProposal(action="buy", item="food"),
        ActionProposal(action="buy", item="medicine"),
        ActionProposal(action="work", job_id="freelance"),
        ActionProposal(action="feed"),
        *[ActionProposal(action="play") for _ in range(3)],
        ActionProposal(action="sleep", hours=3),
    ]
    for proposal in actions:
        assert reference.step(proposal).success
        assert accelerated.step(proposal).success
        _assert_equal(reference, accelerated)
    assert accelerated.state.total_minutes == 441
    assert accelerated.state.pet.health == 84.3


def test_1000_reachable_trajectories_match_reference():
    """Actions, never injected states, generate 1,000 reachable trajectories."""
    for seed in range(1000):
        rng = random.Random(seed)
        horizon = rng.randint(1, 2200)
        reference = _env(BenchmarkMode.LOGICAL, seed, horizon)
        accelerated = _env(BenchmarkMode.ACCELERATED, seed, horizon)
        for _ in range(35):
            if not reference.requires_decision():
                break
            candidates = [ActionProposal(action=action) for action in (
                "observe", "feed", "play", "clean", "heal", "wake")]
            candidates += [ActionProposal(action="sleep", hours=hours) for hours in (3, 5, 8)]
            candidates += [ActionProposal(action="wait", minutes=rng.randint(1, 600))]
            candidates += [ActionProposal(action="work", job_id=job.id)
                           for job in reference.state.jobs_available]
            candidates += [ActionProposal(action="buy", item=item, amount=rng.randint(1, 3))
                           for item in ("food", "medicine")]
            legal = [p for p in candidates
                     if EnvironmentValidator.validate_preconditions(p, reference.state) is None]
            proposal = rng.choice(legal)
            a = reference.step(proposal)
            b = accelerated.step(proposal)
            assert a.model_dump() == b.model_dump(), (seed, proposal)
            _assert_equal(reference, accelerated)


def test_welfare_is_exact_and_segmentation_invariant():
    one = _env(seed=14)
    split = _env(seed=14)
    reference = _env(BenchmarkMode.LOGICAL, 14)
    # Covers critical hunger, cleanliness, sleep recovery, automatic transitions,
    # happiness zero, death clipping, and zero-health terminal minute.
    for env in (one, split, reference):
        env.state.pet.health = 93.21
        env.state.pet.hunger = 97.6
        env.state.pet.cleanliness = 67.75
        env.state.pet.is_sleeping = True
        env.state.pet.energy = 18.4
        env.state.agent.energy = 27.8
        env.state.agent.current_activity = "sleeping"
    one.advance_time(1500)
    reference.advance_time(1500)
    rng = random.Random(247)
    remaining = 1500
    while remaining:
        count = min(remaining, rng.randint(1, 87))
        split.advance_time(count)
        remaining -= count
    _assert_equal(reference, one)
    _assert_equal(reference, split)
    assert one.statistics.elapsed_minutes == one.state.total_minutes
    assert one.statistics.min_health == 0
    assert one.statistics.avg_health == one.statistics.health_integral / one.state.total_minutes


def test_welfare_matches_independent_minute_end_oracle():
    env = _env()
    env.state.pet.health = 70
    env.state.pet.hunger = 60
    env.state.pet.cleanliness = 40
    env.state.pet.energy = 50
    env.state.pet.is_sleeping = True
    # The oracle deliberately does not call any DynamicsEngine helper.
    health, fullness, cleanliness, energy, happiness = 7000, 6000, 4000, 5000, 7000
    sleeping = True
    health_sum = happiness_sum = 0
    minimum = health
    for _ in range(290):
        fullness = max(0, fullness - 30)
        cleanliness = max(0, cleanliness - 15)
        energy = min(10000, energy + 50) if sleeping else max(0, energy - 20)
        happiness = max(0, happiness - 8)
        health += (-20 if fullness < 1500 else 0) + (-10 if cleanliness < 2000 else 0)
        if sleeping and fullness >= 5000:
            health += 5
        health = min(10000, max(0, health))
        minimum = min(minimum, health)
        health_sum += health
        happiness_sum += happiness
        if sleeping and energy == 10000:
            sleeping = False
        elif not sleeping and energy == 0:
            sleeping = True
    env.advance_time(290)
    assert env.statistics.health_centiminutes == health_sum
    assert env.statistics.happiness_centiminutes == happiness_sum
    assert env.statistics.min_health == minimum / 100
    assert env.state.pet.health == health / 100


def test_accelerated_engine_uses_jumps(monkeypatch):
    calls, segments = [], []
    original = DynamicsEngine.apply_delta_time
    original_sum = DynamicsEngine._clipped_sum

    def counted(state, minutes):
        calls.append(minutes)
        return original(state, minutes)

    def counted_sum(start, rate, count):
        segments.append(count)
        return original_sum(start, rate, count)

    monkeypatch.setattr(DynamicsEngine, "apply_delta_time", counted)
    monkeypatch.setattr(DynamicsEngine, "_clipped_sum", staticmethod(counted_sum))
    env = _env()
    env.advance_time(400)
    assert sum(calls) == 400
    assert len(calls) == 1
    assert calls[0] > 1
    # Two exact sums per constant-rate segment, not 800 one-minute sums.
    assert len(segments) < 20
    assert max(segments) > 100


@pytest.mark.parametrize("mode", [BenchmarkMode.LOGICAL, BenchmarkMode.ACCELERATED])
def test_horizon_clamps_work_and_time_and_does_not_pay_unfinished_job(mode):
    env = _env(mode, horizon=17)
    money = env.state.agent.money
    result = env.step({"action": "work", "job_id": "cafe_shift"})
    assert result.success and not result.completed
    assert result.execution_minutes == 17
    assert result.horizon_reached and not result.terminated
    assert env.state.agent.money == money
    assert env.statistics.elapsed_minutes == 17
    assert not env.requires_decision()
    before = env.state.compute_hash()
    env.advance_time(100)
    assert not env.step({"action": "feed"}).success
    assert env.state.compute_hash() == before


def test_job_finishing_exactly_at_horizon_earns_start_quote():
    env = _env(horizon=60)
    result = env.step({"action": "work", "job_id": "cafe_shift"})
    assert result.completed and result.horizon_reached
    assert result.execution_minutes == 60
    assert env.state.agent.money == 110


@pytest.mark.parametrize("health, elapsed", [(0.2, 1), (12.0, 60)])
def test_death_before_or_on_job_completion_earns_nothing(health, elapsed):
    env = _env()
    env.state.pet.health = health
    env.state.pet.hunger = 0
    result = env.step({"action": "work", "job_id": "cafe_shift"})
    assert result.success and result.terminated and not result.completed
    assert result.execution_minutes == elapsed
    assert env.state.agent.money == 50


@pytest.mark.parametrize("energy, action", [(9.6, "play"), (4.6, "clean"), (19.6, "work")])
def test_preconditions_agree_with_observed_rounded_energy(energy, action):
    env = _env()
    env.state.agent.energy = energy
    proposal = ActionProposal(action=action, job_id="cafe_shift" if action == "work" else None)
    cost = {"play": 10, "clean": 5, "work": 20}[action]
    assert env.observe().agent.energy >= cost
    assert env.step(proposal).success


def test_scenario_identity_and_seven_day_default():
    env = _env()
    assert env.max_simulated_minutes == 7 * 24 * 60
    assert env.state.scenario_id == "dynamic_v2"
    assert env.state.scenario_version == 2
    assert env.state.benchmark_version == env.state.environment_version == "2.0.0"
    assert env.reset(scenario_id="standard_v1").scenario_id == "dynamic_v2"
    with pytest.raises(ValueError, match="Unknown scenario"):
        env.reset(scenario_id="typo_v2")
    with pytest.raises(ValueError, match="Unknown scenario"):
        get_config_for_scenario("not_a_scenario")


def test_actual_spec_tracks_dynamics_and_economy():
    from tamabench.spec.spec_loader import EnvironmentSpecLoader
    raw = EnvironmentSpecLoader.load_spec()["raw_content"]
    assert 'environment_version: "2.0.0"' in raw
    assert 'scenario_id: "dynamic_v2"' in raw
    for name in ("hunger_rate", "cleanliness_rate", "happiness_decay_rate",
                 "agent_awake_energy_decay", "awake_pet_energy_decay"):
        assert f"{name}: {getattr(DynamicsEngine, name.upper()):.2f}" in raw
    jobs = EconomySystem.get_dynamic_jobs(0)
    assert jobs[0].reward == 60 and jobs[0].energy_cost == 20
    assert "initial_reward: 60, limit_reward: 30, energy_cost: 20" in raw


def test_scheduled_event_order_and_welfare_match_across_engine_boundaries():
    from tamabench.env.scheduler import WorldEvent
    reference = _env(BenchmarkMode.LOGICAL)
    accelerated = _env()
    for env in (reference, accelerated):
        env.scheduler.scheduled_events = [WorldEvent(
            timestamp_minute=37, event_type="FORCED_SICKNESS", description="Test event",
            handler=lambda state: setattr(state.pet, "is_sick", True),
        )]
        env.advance_time(37)
        assert env.state.pet.health == 100
        assert env.state.pet.is_sick
        env.advance_time(1)
        assert env.state.pet.health == 99.7
    _assert_equal(reference, accelerated)
    assert accelerated.statistics.health_centiminutes == 37 * 10000 + 9970
    assert accelerated.event_history == [(37, "FORCED_SICKNESS")]


@pytest.mark.parametrize("fullness, cleanliness, sleeping, expected_health", [
    (15.3, 20.15, False, 90.0),
    (15.0, 20.0, False, 89.7),
    (50.3, 100.0, True, 90.05),
    (50.0, 100.0, True, 90.0),
])
def test_exact_post_update_thresholds(fullness, cleanliness, sleeping, expected_health):
    env = _env()
    env.state.pet.health = 90
    env.state.pet.hunger = fullness
    env.state.pet.cleanliness = cleanliness
    env.state.pet.is_sleeping = sleeping
    env.advance_time(1)
    assert env.state.pet.health == expected_health


def test_direct_time_advance_refreshes_quote_before_next_action():
    env = _env()
    env.advance_time(60)
    expected = EconomySystem.get_dynamic_jobs(60)[0].reward
    result = env.step({"action": "work", "job_id": "cafe_shift"})
    assert result.completed
    assert env.state.agent.money == 50 + expected


@pytest.mark.parametrize("duration", [-1, True, 1.5])
def test_direct_time_api_rejects_invalid_durations(duration):
    env = _env()
    with pytest.raises(ValueError):
        env.advance_time(duration)
    with pytest.raises(ValueError):
        env.advance_until(duration)


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5])
def test_reset_rejects_invalid_horizons(horizon):
    with pytest.raises(ValueError):
        _env(horizon=horizon)
