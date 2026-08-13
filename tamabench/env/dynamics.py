"""Delta-time based state dynamics engine for TamaBench V1.

Uses closed-form linear rate equations for advance_time(minutes)
avoiding expensive minute-by-minute loops while supporting event scheduler jumps.
"""

import math
from typing import Tuple
from tamabench.env.state import WorldState


class DynamicsEngine:
    # Standard rates per simulation minute
    HUNGER_RATE: float = 0.30              # +18 / hour
    AWAKE_PET_ENERGY_DECAY: float = 0.20   # -12 / hour
    SLEEP_PET_ENERGY_RECOVERY: float = 0.5 # +30 / hour
    CLEANLINESS_RATE: float = 0.15         # -9 / hour
    HAPPINESS_DECAY_RATE: float = 0.08     # -4.8 / hour

    # Health decay rates per minute when in negative conditions
    CRITICAL_HUNGER_HEALTH_PENALTY: float = 0.2
    LOW_CLEANLINESS_HEALTH_PENALTY: float = 0.1
    SICKNESS_HEALTH_PENALTY: float = 0.3
    SLEEP_HEALTH_RECOVERY: float = 0.05

    # Agent energy decay rates per minute
    AGENT_AWAKE_ENERGY_DECAY: float = 0.12 # -7.2 / hour
    AGENT_SLEEP_ENERGY_RECOVERY: float = 0.4 # +24 / hour

    @classmethod
    def apply_delta_time(cls, state: WorldState, minutes: int) -> Tuple[WorldState, bool, str]:
        """Apply closed-form updates while preserving automatic state transitions."""
        if minutes <= 0:
            return state, False, ""

        remaining = minutes
        while remaining > 0:
            pet = state.pet
            agent = state.agent

            if pet.is_sleeping and pet.energy >= 100.0:
                pet.is_sleeping = False
                continue
            if not pet.is_sleeping and pet.energy <= 0.0:
                pet.is_sleeping = True
                continue
            if agent.current_activity == "sleeping" and agent.energy >= 100.0:
                agent.current_activity = "idle"
                continue

            segment = remaining
            if pet.is_sleeping:
                segment = min(
                    segment,
                    max(1, math.ceil((100.0 - pet.energy) / cls.SLEEP_PET_ENERGY_RECOVERY)),
                )
            else:
                segment = min(
                    segment,
                    max(1, math.ceil(pet.energy / cls.AWAKE_PET_ENERGY_DECAY)),
                )
            if agent.current_activity == "sleeping":
                segment = min(
                    segment,
                    max(1, math.ceil((100.0 - agent.energy) / cls.AGENT_SLEEP_ENERGY_RECOVERY)),
                )

            # Split one minute before discontinuous health/recovery rules so
            # the crossing minute is evaluated with the same post-update
            # condition as the reference tick engine.
            if pet.hunger <= 85.0:
                crossing = int((85.0 - pet.hunger) / cls.HUNGER_RATE) + 1
                segment = min(segment, max(1, crossing - 1))
            if pet.cleanliness >= 20.0:
                crossing = int((pet.cleanliness - 20.0) / cls.CLEANLINESS_RATE) + 1
                segment = min(segment, max(1, crossing - 1))
            if pet.is_sleeping and pet.hunger <= 50.0:
                crossing = int((50.0 - pet.hunger) / cls.HUNGER_RATE) + 1
                segment = min(segment, max(1, crossing - 1))

            health_rate = 0.0
            if pet.hunger > 85.0:
                health_rate -= cls.CRITICAL_HUNGER_HEALTH_PENALTY
            if pet.cleanliness < 20.0:
                health_rate -= cls.LOW_CLEANLINESS_HEALTH_PENALTY
            if pet.is_sick:
                health_rate -= cls.SICKNESS_HEALTH_PENALTY
            if pet.is_sleeping and not pet.is_sick and pet.hunger <= 50.0:
                health_rate += cls.SLEEP_HEALTH_RECOVERY
            if health_rate < 0.0:
                segment = min(segment, max(1, math.ceil(pet.health / -health_rate)))

            cls._apply_segment(state, segment)
            remaining -= segment

            if state.pet.health <= 0.0:
                remaining = 0

            if state.pet.is_sleeping and state.pet.energy >= 100.0:
                state.pet.is_sleeping = False
            elif not state.pet.is_sleeping and state.pet.energy <= 0.0:
                state.pet.is_sleeping = True
            if state.agent.current_activity == "sleeping" and state.agent.energy >= 100.0:
                state.agent.current_activity = "idle"

        terminated = state.pet.health <= 0.0
        reason = "Pet died due to health reaching 0." if terminated else ""
        return state, terminated, reason

    @classmethod
    def _apply_segment(cls, state: WorldState, minutes: int) -> None:
        """Apply one segment with no automatic wake/sleep transition inside it."""

        pet = state.pet
        agent = state.agent

        # 1. Update Pet Hunger
        pet.hunger = min(100.0, pet.hunger + (cls.HUNGER_RATE * minutes))

        # 2. Update Pet Energy
        if pet.is_sleeping:
            pet.energy = min(100.0, pet.energy + (cls.SLEEP_PET_ENERGY_RECOVERY * minutes))
        else:
            pet.energy = max(0.0, pet.energy - (cls.AWAKE_PET_ENERGY_DECAY * minutes))

        # 3. Update Cleanliness & Happiness
        pet.cleanliness = max(0.0, pet.cleanliness - (cls.CLEANLINESS_RATE * minutes))
        pet.happiness = max(0.0, pet.happiness - (cls.HAPPINESS_DECAY_RATE * minutes))

        # 4. Calculate Health Change
        health_delta = 0.0

        if pet.hunger > 85.0:
            health_delta -= cls.CRITICAL_HUNGER_HEALTH_PENALTY * minutes

        if pet.cleanliness < 20.0:
            health_delta -= cls.LOW_CLEANLINESS_HEALTH_PENALTY * minutes

        if pet.is_sick:
            health_delta -= cls.SICKNESS_HEALTH_PENALTY * minutes

        if pet.is_sleeping and not pet.is_sick and pet.hunger <= 50.0:
            health_delta += cls.SLEEP_HEALTH_RECOVERY * minutes

        pet.health = max(0.0, min(100.0, pet.health + health_delta))
        pet.age += minutes

        # 5. Update Agent State
        if agent.current_activity == "sleeping":
            agent.energy = min(100.0, agent.energy + (cls.AGENT_SLEEP_ENERGY_RECOVERY * minutes))
        else:
            agent.energy = max(0.0, agent.energy - (cls.AGENT_AWAKE_ENERGY_DECAY * minutes))

        # 6. Advance World Time
        state.total_minutes += minutes

        # Keep analytical jumps and one-minute reference ticks numerically
        # identical. This only removes floating-point residue; the configured
        # V1 rates and transition rules remain unchanged.
        pet.hunger = round(pet.hunger, 10)
        pet.energy = round(pet.energy, 10)
        pet.cleanliness = round(pet.cleanliness, 10)
        pet.happiness = round(pet.happiness, 10)
        pet.health = round(pet.health, 10)
        agent.energy = round(agent.energy, 10)
