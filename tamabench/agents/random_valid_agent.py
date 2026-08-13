"""Random Executable Baseline Agent for TamaBench V1.

Selects uniformly ONLY from currently executable/precondition-valid actions.
"""

import json
import random
from typing import Tuple, Optional
from tamabench.agents.base import BaseAgent
from tamabench.schemas.observation import Observation
from tamabench.schemas.actions import ActionProposal
from tamabench.schemas.errors import BenchmarkError


class RandomValidAgent(BaseAgent):
    def __init__(self, seed: int = 42):
        super().__init__(name="RandomValidAgent")
        self.rng = random.Random(seed)

    def select_action(
        self, observation: Observation
    ) -> Tuple[str, Optional[ActionProposal], Optional[BenchmarkError]]:
        valid_proposals: list[ActionProposal] = []
        pet = observation.pet
        agent = observation.agent
        inv = observation.inventory

        if pet.is_sleeping:
            valid_proposals.append(ActionProposal(action="wake"))
            valid_proposals.append(ActionProposal(action="wait", minutes=30))
        else:
            valid_proposals.append(ActionProposal(action="observe"))
            valid_proposals.append(ActionProposal(action="sleep", hours=3))
            valid_proposals.append(ActionProposal(action="wait", minutes=30))

            if inv.food > 0:
                valid_proposals.append(ActionProposal(action="feed"))

            if inv.medicine > 0 and pet.is_sick:
                valid_proposals.append(ActionProposal(action="heal"))

            if agent.energy >= 10:
                valid_proposals.append(ActionProposal(action="play"))

            if agent.energy >= 5:
                valid_proposals.append(ActionProposal(action="clean"))

            for j in observation.jobs_available:
                if agent.energy >= j.energy_cost:
                    valid_proposals.append(ActionProposal(action="work", job_id=j.id))

            for s in observation.shop_items_available:
                if agent.money >= s.cost:
                    valid_proposals.append(ActionProposal(action="buy", item=s.item, amount=1))

        chosen = self.rng.choice(valid_proposals)
        raw_output = json.dumps(chosen.model_dump(exclude_none=True))
        return raw_output, chosen, None
