"""Random Schema Baseline Agent for TamaBench V1.

Selects uniformly from ALL schema-valid actions, regardless of state preconditions.
Used as the primary lower-bound baseline to evaluate precondition comprehension.
"""

import json
import random
from typing import Tuple, Optional
from tamabench.agents.base import BaseAgent
from tamabench.schemas.observation import Observation
from tamabench.schemas.actions import ActionProposal, ActionType
from tamabench.schemas.errors import BenchmarkError


class RandomSchemaAgent(BaseAgent):
    def __init__(self, seed: int = 42):
        super().__init__(name="RandomSchemaAgent")
        self.rng = random.Random(seed)

    def select_action(
        self, observation: Observation
    ) -> Tuple[str, Optional[ActionProposal], Optional[BenchmarkError]]:
        possible_actions = [
            ActionProposal(action="observe"),
            ActionProposal(action="feed"),
            ActionProposal(action="play"),
            ActionProposal(action="clean"),
            ActionProposal(action="heal"),
            ActionProposal(action="sleep"),
            ActionProposal(action="wake"),
            ActionProposal(action="wait", minutes=30),
            ActionProposal(action="work", job_id="cafe_shift"),
            ActionProposal(action="work", job_id="delivery"),
            ActionProposal(action="work", job_id="freelance"),
            ActionProposal(action="buy", item="food", amount=1),
            ActionProposal(action="buy", item="medicine", amount=1),
        ]

        chosen = self.rng.choice(possible_actions)
        raw_output = json.dumps(chosen.model_dump(exclude_none=True))
        return raw_output, chosen, None
