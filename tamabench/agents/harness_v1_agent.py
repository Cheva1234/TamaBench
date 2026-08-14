"""Harness V1: DECIDE -> CALCULATE -> SCHEDULE autonomy harness for TamaBench.

Wraps a model agent so the model only wakes when a *care* decision is
required. Routine economy (work, buy, wait) is handled deterministically by
the harness's internal reference policy, cutting API calls, tokens, and
compute without changing model size.

Loop (per the TamaBench positioning plan):

    WAKE -> OBSERVE -> DECIDE -> CALCULATE NEXT WAKE -> SCHEDULE -> SLEEP -> WAKE

Three core stages:

    1. DECIDE    - the wrapped model picks a care action (only when needed)
    2. CALCULATE - analytical next-wake minute from the post-action state
    3. SCHEDULE  - deterministic action to the next wake, no model call

Division of labor:

    Model (woken only when critical):  heal, feed, clean, play, sleep timing
    Harness (deterministic, no call):  work, buy, wait, recovery from failure

The harness's deterministic policy is the reference RuleAgent, so wrapping a
competent model never hurts survival (parity), while a model that cannot act
(e.g. always waits) is rescued by the harness with fewer model calls.
"""

import json
from typing import Optional, Tuple

from tamabench.agents.base import BaseAgent, DecisionMetadata
from tamabench.agents.rule_agent import RuleAgent
from tamabench.env.dynamics import DynamicsEngine
from tamabench.env.economy import EconomySystem
from tamabench.env.time_engine import ComputeClock
from tamabench.schemas.actions import ActionProposal
from tamabench.schemas.errors import BenchmarkError
from tamabench.schemas.observation import Observation


class WakeScheduler:
    """Analytical next-wake computation (CALCULATE stage of Harness V1).

    Thresholds mirror the RuleAgent baselines so the harness wakes the model
    at the same points a competent policy would act, without running the
    model continuously.
    """

    def __init__(
        self,
        feed_threshold: float = 50.0,
        clean_threshold: float = 40.0,
        sleep_energy_threshold: float = 12.0,
        health_safety_threshold: float = 40.0,
        max_wait_horizon: int = 120,
    ):
        self.feed_threshold = feed_threshold
        self.clean_threshold = clean_threshold
        self.sleep_energy_threshold = sleep_energy_threshold
        self.health_safety_threshold = health_safety_threshold
        self.max_wait_horizon = max_wait_horizon

    def is_critical(self, observation: Observation) -> bool:
        """True when a *care* decision is required right now (model wake).

        Routine economy (work/buy/wait) is deliberately NOT critical: the
        harness handles it deterministically without waking the model.
        """
        pet = observation.pet
        agent = observation.agent

        if pet.is_sick:
            return True
        if pet.hunger < self.feed_threshold:
            return True
        if pet.cleanliness < self.clean_threshold:
            return True
        if pet.health < self.health_safety_threshold:
            return True
        if agent.energy < self.sleep_energy_threshold:
            return True
        return False

    def proposal_executable(
        self, observation: Observation, proposal: ActionProposal
    ) -> bool:
        """Cheap precondition check so the harness never schedules through a
        failed action (mirrors EnvironmentValidator for the harness's own
        scheduling decision)."""
        pet = observation.pet
        agent = observation.agent
        inv = observation.inventory
        action = proposal.action

        if action == "feed":
            return inv.food > 0 and not pet.is_sleeping
        if action == "clean":
            return agent.energy >= 5 and not pet.is_sleeping
        if action == "play":
            return agent.energy >= 10 and not pet.is_sleeping
        if action == "heal":
            return inv.medicine > 0 and not pet.is_sleeping
        if action == "work":
            job = next(
                (j for j in observation.jobs_available if j.id == proposal.job_id),
                None,
            )
            return job is not None and agent.energy >= job.energy_cost
        if action == "buy":
            item = next(
                (s for s in observation.shop_items_available if s.item == proposal.item),
                None,
            )
            return item is not None and agent.money >= item.cost * (proposal.amount or 1)
        if action == "sleep":
            return not pet.is_sleeping
        if action == "wake":
            return pet.is_sleeping
        return True  # wait / observe are always executable

    def time_to_next_need(self, observation: Observation) -> int:
        """Minutes from now until the next decision is required, computed
        analytically from the current state. Used to size harness waits."""
        pet = observation.pet
        agent = observation.agent

        candidates: list[int] = []
        if pet.hunger > self.feed_threshold:
            candidates.append(
                int((pet.hunger - self.feed_threshold) / DynamicsEngine.HUNGER_RATE) + 1
            )
        if pet.cleanliness > self.clean_threshold:
            candidates.append(
                int(
                    (pet.cleanliness - self.clean_threshold)
                    / DynamicsEngine.CLEANLINESS_RATE
                )
                + 1
            )
        if agent.energy > self.sleep_energy_threshold:
            candidates.append(
                int(
                    (agent.energy - self.sleep_energy_threshold)
                    / DynamicsEngine.AGENT_AWAKE_ENERGY_DECAY
                )
                + 1
            )
        if pet.health <= 60.0:
            # Low health: keep waits short so sickness cannot drain too much.
            candidates.append(60)

        if not candidates:
            return 30
        return max(1, min(candidates))

    def next_wake_minutes(
        self, observation: Observation, proposal: ActionProposal
    ) -> int:
        """Minutes from now until the next decision is required, given the
        action about to commit. Returns 1 when the action would fail so the
        model is re-consulted immediately."""
        pet = observation.pet
        agent = observation.agent
        inv = observation.inventory
        action = proposal.action

        if not self.proposal_executable(observation, proposal):
            return 1

        # Predict the post-action state from the current observation.
        hunger = pet.hunger
        cleanliness = pet.cleanliness
        health = pet.health
        energy = float(agent.energy)
        food = inv.food
        sick = pet.is_sick

        if action == "feed":
            hunger = min(100.0, hunger + EconomySystem.FOOD_HUNGER_RESTORE)
        elif action == "clean":
            cleanliness = 100.0
        elif action == "heal":
            sick = False
            health = min(100.0, health + 15.0)
        elif action == "buy":
            if proposal.item == "food":
                food += proposal.amount or 1
        elif action == "work":
            # Blocking action: the environment fast-forwards to completion and
            # the next decision is required exactly when the job ends.
            job = next(
                (j for j in observation.jobs_available if j.id == proposal.job_id),
                None,
            )
            return job.duration_minutes if job else 30
        elif action == "sleep":
            return (proposal.hours or 3) * 60
        elif action == "wait":
            return proposal.minutes or 30

        if sick:
            # Still sick (e.g. bought medicine instead of healing): re-check soon.
            return 30

        # Reuse the analytical need computation on the predicted state.
        predicted = Observation(
            time=observation.time,
            agent=observation.agent,
            pet=observation.pet.model_copy(
                update={
                    "hunger": round(hunger, 1),
                    "cleanliness": round(cleanliness, 1),
                    "health": round(health, 1),
                }
            ),
            inventory=observation.inventory.model_copy(update={"food": food}),
            jobs_available=observation.jobs_available,
            shop_items_available=observation.shop_items_available,
            state_hash=observation.state_hash,
        )
        return self.time_to_next_need(predicted)


class HarnessV1Agent(BaseAgent):
    """Wraps a model agent with the DECIDE -> CALCULATE -> SCHEDULE loop.

    The wrapped model is only invoked when a care decision is required
    (critical state) or on the first decision of an episode. Routine economy
    and recovery are handled by the internal reference policy (RuleAgent)
    with no model call, so the model never runs continuously.
    """

    def __init__(
        self,
        model_agent: BaseAgent,
        max_wait_horizon: int = 120,
        **scheduler_kwargs,
    ):
        model_name = getattr(model_agent, "model_name", model_agent.name)
        super().__init__(name=f"HarnessV1({model_name})")
        self.model_agent = model_agent
        self.policy = RuleAgent()  # deterministic reference policy
        self.scheduler = WakeScheduler(
            max_wait_horizon=max_wait_horizon, **scheduler_kwargs
        )
        self._next_wake_minute: Optional[int] = None
        self._last_seen_minute: Optional[int] = None
        self._last_decision_was_model: bool = False
        self._harness_control: bool = False
        self._scheduled_waits = 0
        self._model_decisions = 0
        self._last_compute = ComputeClock()
        self._last_reasoning = ""
        self._last_json_output = ""
        self._last_finish_reason: Optional[str] = None
        self._last_was_truncated = False

    # --- model-facing surface delegated to the wrapped agent ---

    @property
    def model_name(self) -> str:
        return getattr(self.model_agent, "model_name", self.model_agent.name)

    @property
    def model_resident(self) -> bool:
        return bool(getattr(self.model_agent, "model_resident", False))

    @property
    def api_calls(self) -> int:
        return int(getattr(self.model_agent, "api_calls", 0))

    @property
    def runtime(self):
        # Lets the runner attribute per-decision API calls to the wrapped
        # model's runtime; harness decisions log attempt_count=0 so they
        # never count as API calls.
        return getattr(self.model_agent, "runtime", None)

    @property
    def last_compute(self) -> ComputeClock:
        return self._last_compute

    @property
    def last_reasoning(self) -> str:
        return self._last_reasoning

    @property
    def last_json_output(self) -> str:
        return self._last_json_output

    @property
    def last_finish_reason(self) -> Optional[str]:
        return self._last_finish_reason

    @property
    def last_was_truncated(self) -> bool:
        return self._last_was_truncated

    @property
    def scheduled_waits(self) -> int:
        return self._scheduled_waits

    @property
    def model_decisions(self) -> int:
        return self._model_decisions

    def warmup(self) -> float:
        return self.model_agent.warmup()

    def close(self) -> None:
        self.model_agent.close()

    def reset_episode(self) -> None:
        """Clear the pending schedule and counters for a fresh episode."""
        self._next_wake_minute = None
        self._last_seen_minute = None
        self._last_decision_was_model = False
        self._harness_control = False
        self._scheduled_waits = 0
        self._model_decisions = 0
        self.last_decision = DecisionMetadata()
        self._last_compute = ComputeClock()
        self._last_reasoning = ""
        self._last_json_output = ""
        self._last_finish_reason = None
        self._last_was_truncated = False

    # --- the harness loop ---

    def select_action(
        self, observation: Observation
    ) -> Tuple[str, Optional[ActionProposal], Optional[BenchmarkError]]:
        now = observation.time.total_minutes

        # Failure recovery: every action advances simulation time by at least
        # one minute, so seeing the same minute twice means the previous
        # model decision was rejected by the environment. Re-consulting the
        # model would spin forever, so fall back to the deterministic policy.
        if (
            self._last_decision_was_model
            and self._last_seen_minute is not None
            and now == self._last_seen_minute
        ):
            return self._policy_action(observation, fallback=True)
        self._last_seen_minute = now

        # Harness control: after the safety layer overrides a model choice,
        # the harness runs the deterministic policy (no model calls) until
        # the state is no longer critical. This is the "harness provides
        # autonomy instead of model parameters" part of the story.
        if self._harness_control:
            if self.scheduler.is_critical(observation):
                return self._policy_action(observation, fallback=False)
            self._harness_control = False

        # The model wakes only when a care decision is genuinely required:
        # the first decision of an episode, or a critical environment state.
        if (
            self._next_wake_minute is None
            or self.scheduler.is_critical(observation)
        ):
            return self._decide(observation)
        return self._policy_action(observation, fallback=False)

    def _decide(
        self, observation: Observation
    ) -> Tuple[str, Optional[ActionProposal], Optional[BenchmarkError]]:
        """Stage 1 (DECIDE): consult the wrapped model, then schedule."""
        raw_output, proposal, err = self.model_agent.select_action(observation)
        self._model_decisions += 1
        self.last_decision = getattr(
            self.model_agent, "last_decision", DecisionMetadata()
        )
        self._last_compute = getattr(self.model_agent, "last_compute", ComputeClock())
        self._last_reasoning = getattr(self.model_agent, "last_reasoning", "")
        self._last_json_output = getattr(self.model_agent, "last_json_output", "")
        self._last_finish_reason = getattr(self.model_agent, "last_finish_reason", None)
        self._last_was_truncated = bool(
            getattr(self.model_agent, "last_was_truncated", False)
        )

        if proposal is not None and err is None:
            # Harness safety layer: correct dangerous model choices before
            # committing (heal a sick pet, feed a starving pet). When an
            # override fires, the harness takes over deterministically until
            # the state is safe again.
            overridden = self._safety_override(observation, proposal)
            if overridden != proposal:
                self._harness_control = True
            proposal = overridden
            # Stages 2+3 (CALCULATE + SCHEDULE): store the next wake minute.
            self._next_wake_minute = (
                observation.time.total_minutes
                + self.scheduler.next_wake_minutes(observation, proposal)
            )
        else:
            # Failed decision: re-consult the model almost immediately.
            self._next_wake_minute = observation.time.total_minutes + 1
        self._last_decision_was_model = True
        return raw_output, proposal, err

    def _safety_override(
        self, observation: Observation, proposal: ActionProposal
    ) -> ActionProposal:
        """Minimal deterministic safety layer over the model's choice.

        Only fires on imminent danger (sick pet with medicine, starving pet).
        The model is still consulted and its decision is logged; the harness
        only corrects choices that would let the pet die.
        """
        pet = observation.pet
        inv = observation.inventory
        action = proposal.action

        if pet.is_sick and inv.medicine > 0 and action != "heal":
            return ActionProposal(action="heal")
        if pet.is_sleeping:
            # Care actions all fail while the pet sleeps; leave the model's
            # choice alone and let the deterministic layer wait instead.
            return proposal
        if pet.hunger < 15.0 and inv.food > 0:
            # Critical starvation: feed regardless of the model's choice.
            return ActionProposal(action="feed")
        if pet.hunger < 15.0 and inv.food == 0:
            # Imminent starvation with no food. If the model is already
            # buying food, respect its choice (it may buy more than one).
            if action == "buy" and proposal.item == "food":
                return proposal
            # Otherwise buy if affordable, else work the cheapest job the
            # agent has energy for.
            if observation.agent.money >= 30:
                return ActionProposal(action="buy", item="food", amount=1)
            affordable = [
                j
                for j in observation.jobs_available
                if observation.agent.energy >= j.energy_cost
            ]
            if affordable:
                best = min(affordable, key=lambda j: j.duration_minutes)
                return ActionProposal(action="work", job_id=best.id)
        return proposal

    def _policy_action(
        self, observation: Observation, fallback: bool
    ) -> Tuple[str, Optional[ActionProposal], Optional[BenchmarkError]]:
        """Stage 3 (SCHEDULE): deterministic action from the reference
        policy, no model call. Handles routine economy (work/buy/wait),
        recovery from failed model decisions, and harness-control recovery.
        """
        raw_output, proposal, err = self.policy.select_action(observation)
        if proposal is not None and err is None:
            self._next_wake_minute = (
                observation.time.total_minutes
                + self.scheduler.next_wake_minutes(observation, proposal)
            )
        else:
            self._next_wake_minute = observation.time.total_minutes + 1
        self._last_decision_was_model = False
        self._scheduled_waits += 1
        self.last_decision = DecisionMetadata(
            finish_reason=None,
            was_truncated=False,
            generation_attempt=0,
            attempt_count=0,
            first_pass_valid=True,
            final_valid=True,
            recovered=False,
            input_tokens=0,
            total_output_tokens=0,
            reasoning_tokens=0,
            json_tokens=0,
            raw_json="",
        )
        self._last_compute = ComputeClock()
        self._last_reasoning = ""
        self._last_json_output = ""
        self._last_finish_reason = None
        self._last_was_truncated = False
        return raw_output, proposal, err
