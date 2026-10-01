"""Exact discrete-minute dynamics with event-driven analytical acceleration.

Contract 2.0 uses integer hundredths for state transitions. A minute first
updates needs, then health using those updated needs and the sleep state for
that minute, then applies automatic wake/sleep transitions. Constant-rate
segments are summed analytically, including the minute-end welfare samples.
"""

from typing import Tuple
from tamabench.env.state import WorldState


class DynamicsEngine:
    HUNGER_RATE = 0.30
    CRITICAL_HUNGER_THRESHOLD = 15.0
    SLEEP_HEALTH_RECOVERY_THRESHOLD = 50.0
    AWAKE_PET_ENERGY_DECAY = 0.20
    SLEEP_PET_ENERGY_RECOVERY = 0.50
    CLEANLINESS_RATE = 0.15
    HAPPINESS_DECAY_RATE = 0.08
    CRITICAL_HUNGER_HEALTH_PENALTY = 0.20
    LOW_CLEANLINESS_HEALTH_PENALTY = 0.10
    SICKNESS_HEALTH_PENALTY = 0.30
    SLEEP_HEALTH_RECOVERY = 0.05
    AGENT_AWAKE_ENERGY_DECAY = 0.12
    AGENT_SLEEP_ENERGY_RECOVERY = 0.40

    @staticmethod
    def _clipped_sum(start: int, rate: int, count: int) -> int:
        """Sum clamp(start + i*rate, 0, 10000), i=1..count, exactly."""
        if rate == 0:
            return start * count
        if rate < 0:
            linear = min(count, start // -rate)
            return linear * start + rate * linear * (linear + 1) // 2
        linear = min(count, (10000 - start) // rate)
        return (linear * start + rate * linear * (linear + 1) // 2
                + (count - linear) * 10000)

    @classmethod
    def apply_delta_time(cls, state: WorldState, minutes: int) -> Tuple[WorldState, bool, str]:
        """Jump between rule discontinuities, never between individual minutes."""
        if type(minutes) is not int or minutes < 0:
            raise ValueError("minutes must be a nonnegative integer")
        pet, agent = state.pet, state.agent
        health, hunger, energy, happiness, cleanliness, agent_energy = (
            int(round(value * 100)) for value in (
                pet.health, pet.hunger, pet.energy, pet.happiness,
                pet.cleanliness, agent.energy,
            )
        )
        remaining = minutes
        while remaining and health > 0:
            # Normalize instantaneous transitions before the first minute.
            if pet.is_sleeping and energy >= 10000:
                pet.is_sleeping = False
            elif not pet.is_sleeping and energy <= 0:
                pet.is_sleeping = True
            if agent.current_activity == "sleeping" and agent_energy >= 10000:
                agent.current_activity = "idle"

            sleeping = pet.is_sleeping
            agent_sleeping = agent.current_activity == "sleeping"
            segment = remaining
            # Number of ticks through (including) the automatic transition.
            until_pet_transition = ((10000 - energy + 49) // 50 if sleeping
                                    else (energy + 19) // 20)
            segment = min(segment, until_pet_transition)
            if agent_sleeping:
                segment = min(segment, (10000 - agent_energy + 39) // 40)

            # Keep post-update predicates constant for every tick in a segment.
            # Integer division avoids float ceil/floor drift at exact thresholds.
            thresholds = [(hunger, 1500, 30), (cleanliness, 2000, 15)]
            if sleeping:
                thresholds.append((hunger, 5000, 30))
            for value, threshold, decay in thresholds:
                ticks_before_crossing = (value - threshold) // decay
                if ticks_before_crossing >= 1:
                    segment = min(segment, ticks_before_crossing)

            next_hunger = max(0, hunger - 30)
            next_cleanliness = max(0, cleanliness - 15)
            health_rate = (
                (-20 if next_hunger < 1500 else 0)
                + (-10 if next_cleanliness < 2000 else 0)
                + (-30 if pet.is_sick else 0)
                + (5 if sleeping and not pet.is_sick and next_hunger >= 5000 else 0)
            )
            if health_rate < 0:
                segment = min(segment, (health - health_rate - 1) // -health_rate)

            stats = state.statistics
            stats.health_centiminutes += cls._clipped_sum(health, health_rate, segment)
            stats.happiness_centiminutes += cls._clipped_sum(happiness, -8, segment)
            next_health = max(0, min(10000, health + health_rate * segment))
            stats.minimum_health_centi = min(stats.minimum_health_centi, health, next_health)
            stats.elapsed_minutes += segment

            health = next_health
            hunger = max(0, hunger - 30 * segment)
            energy = (min(10000, energy + 50 * segment) if sleeping
                      else max(0, energy - 20 * segment))
            happiness = max(0, happiness - 8 * segment)
            cleanliness = max(0, cleanliness - 15 * segment)
            agent_energy = (min(10000, agent_energy + 40 * segment) if agent_sleeping
                            else max(0, agent_energy - 12 * segment))
            state.total_minutes += segment
            pet.age += segment
            remaining -= segment

            if sleeping and energy >= 10000:
                pet.is_sleeping = False
            elif not sleeping and energy <= 0:
                pet.is_sleeping = True
            if agent_sleeping and agent_energy >= 10000:
                agent.current_activity = "idle"

        (pet.health, pet.hunger, pet.energy, pet.happiness,
         pet.cleanliness, agent.energy) = (
            value / 100 for value in (health, hunger, energy, happiness, cleanliness, agent_energy)
        )
        terminated = health <= 0
        return state, terminated, "Pet died due to health reaching 0." if terminated else ""
