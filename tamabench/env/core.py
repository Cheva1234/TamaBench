"""Headless Environment Simulator Core for TamaBench V1.

Implements Time-Skip Actions (work, sleep, wait) backed by an Event-Driven Fast-Forward Engine.
"""

from typing import Union, Optional, Tuple
from tamabench.env.state import WorldState, AgentState, PetState, Inventory
from tamabench.env.dynamics import DynamicsEngine
from tamabench.env.economy import EconomySystem
from tamabench.env.scheduler import EventScheduler
from tamabench.env.time_engine import TimeEngine, BenchmarkMode
from tamabench.schemas.observation import Observation
from tamabench.schemas.actions import ActionProposal, StepResult
from tamabench.validation.syntax_validator import SyntaxValidator
from tamabench.validation.env_validator import EnvironmentValidator


class TamaEnv:
    def __init__(self, mode: BenchmarkMode = BenchmarkMode.LOGICAL):
        self.mode = mode
        self.time_engine = TimeEngine(mode=mode)
        self.state: WorldState = WorldState()
        self.scheduler: Optional[EventScheduler] = None
        self.terminated: bool = False
        self.termination_reason: str = ""
        self.event_history: list[tuple[int, str]] = []

    def reset(
        self,
        seed: int = 42,
        scenario_id: str = "standard_v1",
        scenario_version: int = 1,
    ) -> Observation:
        """Resets environment to reproducible initial state using `seed`."""
        self.state = WorldState(
            total_minutes=0,
            agent=AgentState(money=30, energy=100, current_activity="idle"),
            pet=PetState(
                health=100.0,
                hunger=80.0,
                energy=80.0,
                happiness=70.0,
                cleanliness=90.0,
                is_sick=False,
                is_sleeping=False,
                age=0,
            ),
            inventory=Inventory(food=1, medicine=0),
            jobs_available=EconomySystem.get_default_jobs(),
            shop_items_available=EconomySystem.get_default_shop(),
            benchmark_version="1.1.0",
            environment_version="1.1.0",
            scenario_id=scenario_id,
            scenario_version=scenario_version,
            seed=seed,
        )
        self.scheduler = EventScheduler(seed=seed)
        self.scheduler.seed_initial_events()
        self.terminated = False
        self.termination_reason = ""
        self.event_history = []

        return self.observe()

    def observe(self) -> Observation:
        """Returns current canonical observation snapshot."""
        return self.state.to_observation()

    def requires_decision(self) -> bool:
        """Returns True if simulation is at a decision boundary requiring an API call."""
        if self.terminated:
            return False
        return not self.state.agent.current_activity.startswith("working:")

    def commit(self, action_input: Union[ActionProposal, dict, str]) -> StepResult:
        """Commit one agent decision at a boundary.

        V1 actions already perform their complete blocking time-skip inside
        ``step``. This named method makes that boundary explicit for runners
        without changing the V1 environment protocol.
        """
        return self.step(action_input)

    def advance_to_next_boundary(self) -> Observation:
        """Advance any pending blocking work without requesting a new action.

        The V1 implementation commits blocking actions atomically, so there is
        no pending state to advance. The method is still part of the runner
        contract and is ready for a future split commit/advance implementation.
        """
        return self.observe()

    def advance_time(self, minutes: int) -> Tuple[bool, str]:
        """Advances simulation clock using analytical event-driven jumps."""
        if minutes <= 0 or self.terminated:
            return self.terminated, self.termination_reason

        target_minute = self.state.total_minutes + minutes
        return self.advance_until(target_minute)

    def advance_until(self, target_minute: int) -> Tuple[bool, str]:
        """Advance to a target using the selected reference or accelerated engine."""
        if self.mode == BenchmarkMode.LOGICAL:
            return self._advance_reference_until(target_minute)
        return self._advance_event_driven_until(target_minute)

    def _advance_reference_until(self, target_minute: int) -> Tuple[bool, str]:
        """Reference implementation: apply dynamics and events one minute at a time."""
        while self.state.total_minutes < target_minute and not self.terminated:
            self.state, self.terminated, self.termination_reason = DynamicsEngine.apply_delta_time(
                self.state, 1
            )
            if self.terminated:
                break
            triggered = self.scheduler.trigger_events_at(self.state.total_minutes, self.state)
            self.event_history.extend((self.state.total_minutes, event_type) for event_type in triggered)
        return self.terminated, self.termination_reason

    def _advance_event_driven_until(self, target_minute: int) -> Tuple[bool, str]:
        """Accelerated implementation: jump between event timestamps analytically."""
        current_minute = self.state.total_minutes

        while current_minute < target_minute and not self.terminated:
            next_event_min = self.scheduler.get_next_event_minute(current_minute, target_minute, self.state)
            step_to_min = next_event_min if next_event_min is not None else target_minute

            delta = step_to_min - current_minute
            if delta > 0:
                self.state, self.terminated, self.termination_reason = DynamicsEngine.apply_delta_time(
                    self.state, delta
                )
                current_minute = self.state.total_minutes

            if self.terminated:
                break

            if next_event_min is not None and current_minute == next_event_min:
                triggered = self.scheduler.trigger_events_at(current_minute, self.state)
                self.event_history.extend((current_minute, event_type) for event_type in triggered)

        return self.terminated, self.termination_reason

    def step(self, action_input: Union[ActionProposal, dict, str]) -> StepResult:
        """Executes action proposal through 2-stage validation and time-skip commitment."""
        if self.terminated:
            return StepResult(
                success=False,
                observation=self.observe(),
                terminated=True,
                termination_reason=self.termination_reason,
                state_hash=self.state.compute_hash(),
            )

        # Stage 1 Syntax & Schema Validation
        if isinstance(action_input, ActionProposal):
            proposal = action_input
            schema_err = None
        elif isinstance(action_input, dict):
            proposal, schema_err = SyntaxValidator.validate_raw(
                action_input.get("raw", "") if "raw" in action_input else str(action_input).replace("'", '"')
            )
        else:
            proposal, schema_err = SyntaxValidator.validate_raw(str(action_input))

        if schema_err:
            return StepResult(
                success=False,
                observation=self.observe(),
                terminated=self.terminated,
                termination_reason=self.termination_reason,
                error=schema_err,
                execution_minutes=0,
                state_hash=self.state.compute_hash(),
            )

        # Stage 2 Environment Precondition Validation
        env_err = EnvironmentValidator.validate_preconditions(proposal, self.state)
        if env_err:
            return StepResult(
                success=False,
                observation=self.observe(),
                terminated=self.terminated,
                termination_reason=self.termination_reason,
                error=env_err,
                execution_minutes=0,
                state_hash=self.state.compute_hash(),
            )

        action = proposal.action
        exec_minutes = 1

        if action == "feed":
            self.state.inventory.food -= 1
            self.state.pet.hunger = min(
                100.0,
                self.state.pet.hunger + EconomySystem.FOOD_HUNGER_RESTORE,
            )
            self.state.pet.happiness = min(100.0, self.state.pet.happiness + 5.0)
            exec_minutes = 2
            self.advance_time(exec_minutes)

        elif action == "play":
            self.state.agent.energy = max(0, self.state.agent.energy - 10)
            self.state.pet.happiness = min(100.0, self.state.pet.happiness + 20.0)
            self.state.pet.energy = max(0.0, self.state.pet.energy - 10.0)
            self.state.pet.cleanliness = max(0.0, self.state.pet.cleanliness - 5.0)
            exec_minutes = 15
            self.advance_time(exec_minutes)

        elif action == "clean":
            self.state.agent.energy = max(0, self.state.agent.energy - 5)
            self.state.pet.cleanliness = 100.0
            self.state.pet.happiness = min(100.0, self.state.pet.happiness + 5.0)
            exec_minutes = 10
            self.advance_time(exec_minutes)

        elif action == "heal":
            self.state.inventory.medicine -= 1
            self.state.pet.is_sick = False
            self.state.pet.health = min(100.0, self.state.pet.health + 15.0)
            exec_minutes = 5
            self.advance_time(exec_minutes)

        elif action == "sleep":
            hours = proposal.hours
            if hours not in (3, 5, 8):
                if proposal.minutes and (proposal.minutes // 60 in (3, 5, 8)):
                    hours = proposal.minutes // 60
                else:
                    hours = 3  # Default 3 hours if unassigned

            exec_minutes = hours * 60
            self.state.agent.current_activity = "sleeping"
            self.state.pet.is_sleeping = True
            self.advance_time(exec_minutes)
            if not self.terminated:
                self.state.agent.current_activity = "idle"
                self.state.pet.is_sleeping = False

        elif action == "wake":
            self.state.agent.current_activity = "idle"
            self.state.pet.is_sleeping = False
            exec_minutes = 1
            self.advance_time(exec_minutes)

        elif action == "work":
            # Time-Skip Action: Fast-forwards full job duration in a single block
            job = EconomySystem.find_job(proposal.job_id or "", self.state)
            if job:
                self.state.agent.current_activity = f"working:{job.id}"
                exec_minutes = job.duration_minutes
                self.advance_until(self.state.total_minutes + exec_minutes)

                # Deduct energy and grant reward money upon job completion
                self.state.agent.energy = max(0, self.state.agent.energy - job.energy_cost)
                self.state.agent.money += job.reward
                self.state.agent.current_activity = "idle"

        elif action == "buy":
            item = EconomySystem.find_shop_item(proposal.item or "", self.state)
            amount = proposal.amount or 1
            if item:
                self.state.agent.money -= item.cost * amount
                if item.item == "food":
                    self.state.inventory.food += amount
                elif item.item == "medicine":
                    self.state.inventory.medicine += amount
            exec_minutes = 2
            self.advance_time(exec_minutes)

        elif action == "wait":
            # Time-Skip Action: Fast-forwards requested wait duration
            exec_minutes = proposal.minutes or 30
            self.advance_until(self.state.total_minutes + exec_minutes)

        elif action == "observe":
            exec_minutes = 1
            self.advance_time(exec_minutes)

        return StepResult(
            success=True,
            observation=self.observe(),
            terminated=self.terminated,
            termination_reason=self.termination_reason,
            execution_minutes=exec_minutes,
            state_hash=self.state.compute_hash(),
        )
