"""Action Effectiveness Classifier for TamaBench V1.

Categorizes actions into PRODUCTIVE, NEUTRAL, WASTEFUL, or HARMFUL based on state delta utility.
"""

from enum import Enum
from typing import Optional
from tamabench.schemas.observation import Observation
from tamabench.schemas.actions import ActionProposal
from tamabench.schemas.errors import BenchmarkError


class ActionCategory(str, Enum):
    PRODUCTIVE = "productive"
    NEUTRAL = "neutral"
    WASTEFUL = "wasteful"
    HARMFUL = "harmful"


class ActionEffectivenessClassifier:
    @classmethod
    def classify_action(
        self,
        before_obs: Observation,
        proposal: Optional[ActionProposal],
        error: Optional[BenchmarkError],
        after_obs: Observation,
    ) -> ActionCategory:
        """Classifies action effectiveness based on state transition utility."""
        if error is not None:
            # Environment or Schema Error
            if before_obs.pet.is_sick or before_obs.pet.hunger > 80:
                return ActionCategory.HARMFUL
            return ActionCategory.WASTEFUL

        if proposal is None:
            return ActionCategory.HARMFUL

        action = proposal.action
        pet_before = before_obs.pet
        inv_before = before_obs.inventory
        agent_before = before_obs.agent

        if action == "feed":
            if pet_before.hunger >= 30:
                return ActionCategory.PRODUCTIVE
            return ActionCategory.WASTEFUL

        elif action == "heal":
            if pet_before.is_sick:
                return ActionCategory.PRODUCTIVE
            return ActionCategory.WASTEFUL

        elif action == "clean":
            if pet_before.cleanliness <= 50:
                return ActionCategory.PRODUCTIVE
            return ActionCategory.WASTEFUL

        elif action == "play":
            if pet_before.happiness <= 60 and agent_before.energy >= 10:
                return ActionCategory.PRODUCTIVE
            return ActionCategory.NEUTRAL

        elif action == "work":
            if agent_before.money < 150 or inv_before.food <= 1:
                return ActionCategory.PRODUCTIVE
            return ActionCategory.NEUTRAL

        elif action == "buy":
            if proposal.item == "food" and inv_before.food <= 2:
                return ActionCategory.PRODUCTIVE
            if proposal.item == "medicine" and (inv_before.medicine == 0 or pet_before.is_sick):
                return ActionCategory.PRODUCTIVE
            if inv_before.food > 8:
                return ActionCategory.WASTEFUL
            return ActionCategory.NEUTRAL

        elif action in ["wait", "observe", "sleep", "wake"]:
            return ActionCategory.NEUTRAL

        return ActionCategory.NEUTRAL
