"""Event Scheduler and Seeded Random Events Generator for TamaBench V1.

Ensures fully reproducible random world events (sickness, job changes, shop price updates)
and enables analytical time jumps to scheduled events & state thresholds.
"""

import random
from dataclasses import dataclass
from typing import Callable, Optional
from tamabench.env.state import WorldState


@dataclass
class WorldEvent:
    timestamp_minute: int
    event_type: str
    description: str
    handler: Callable[[WorldState], None]


class EventScheduler:
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)
        self.scheduled_events: list[WorldEvent] = []

    def seed_initial_events(self, max_minutes: int = 43200):
        """Generates deterministic random events for the simulation horizon (up to 30 simulated days)."""
        self.scheduled_events.clear()

        # Schedule Random Sickness Events (e.g. 1-2 per 3 days)
        current = self.rng.randint(720, 1440)
        while current < max_minutes:
            def make_sickness_handler():
                def handler(state: WorldState):
                    sickness_chance = 0.5 if state.pet.cleanliness < 50 else 0.15
                    if not state.pet.is_sick and random.Random(state.total_minutes).random() < sickness_chance:
                        state.pet.is_sick = True
                return handler

            self.scheduled_events.append(
                WorldEvent(
                    timestamp_minute=current,
                    event_type="PET_BECAME_SICK",
                    description="Pet sickness risk event triggered.",
                    handler=make_sickness_handler(),
                )
            )
            current += self.rng.randint(1440, 2880)

        self.scheduled_events.sort(key=lambda e: e.timestamp_minute)

    def get_next_event_minute(self, current_minute: int, target_minute: int, state: Optional[WorldState] = None) -> Optional[int]:
        """Finds the timestamp of the earliest pending event (scheduled or threshold) between current_minute and target_minute."""
        earliest: Optional[int] = None

        # 1. Check Scheduled Random Events
        for event in self.scheduled_events:
            if current_minute < event.timestamp_minute <= target_minute:
                if earliest is None or event.timestamp_minute < earliest:
                    earliest = event.timestamp_minute
                break

        # 2. Check Analytical State Threshold Events if state is provided
        if state is not None:
            pet = state.pet

            # Threshold A: Critical Hunger (hunger >= 85.0)
            if pet.hunger < 85.0:
                mins_to_hunger = int((85.0 - pet.hunger) / 0.30) + 1
                threshold_min = current_minute + max(1, mins_to_hunger)
                if current_minute < threshold_min <= target_minute:
                    if earliest is None or threshold_min < earliest:
                        earliest = threshold_min

            # Threshold B: Low Cleanliness (cleanliness <= 20.0)
            if pet.cleanliness > 20.0:
                mins_to_clean = int((pet.cleanliness - 20.0) / 0.15) + 1
                threshold_min = current_minute + max(1, mins_to_clean)
                if current_minute < threshold_min <= target_minute:
                    if earliest is None or threshold_min < earliest:
                        earliest = threshold_min

            # Sleep health recovery ends immediately after hunger rises above
            # 50, so the accelerated engine must split at that boundary too.
            if pet.is_sleeping and pet.hunger <= 50.0:
                mins_to_sleep_recovery_end = int((50.0 - pet.hunger) / 0.30) + 1
                threshold_min = current_minute + max(1, mins_to_sleep_recovery_end)
                if current_minute < threshold_min <= target_minute:
                    if earliest is None or threshold_min < earliest:
                        earliest = threshold_min

        return earliest

    def trigger_events_at(self, minute: int, state: WorldState) -> list[str]:
        """Triggers all events scheduled exactly at or before `minute`."""
        triggered_types: list[str] = []
        remaining: list[WorldEvent] = []

        for event in self.scheduled_events:
            if event.timestamp_minute <= minute:
                event.handler(state)
                triggered_types.append(event.event_type)
            else:
                remaining.append(event)

        self.scheduled_events = remaining
        return triggered_types
