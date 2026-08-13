"""Rule-Based Baseline Agent for TamaBench V1.

Uses deterministic priority rules to maintain pet health and work balance.
"""

import json
from typing import Tuple, Optional
from tamabench.agents.base import BaseAgent
from tamabench.schemas.observation import Observation
from tamabench.schemas.actions import ActionProposal
from tamabench.schemas.errors import BenchmarkError


class RuleAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="RuleAgent")

    def select_action(
        self, observation: Observation
    ) -> Tuple[str, Optional[ActionProposal], Optional[BenchmarkError]]:
        pet = observation.pet
        agent = observation.agent
        inv = observation.inventory

        # 0. If pet is sleeping and fully rested or hungry, wake up
        if pet.is_sleeping:
            if pet.energy >= 90 or pet.hunger > 60:
                proposal = ActionProposal(action="wake")
                return json.dumps(proposal.model_dump(exclude_none=True)), proposal, None
            proposal = ActionProposal(action="wait", minutes=60)
            return json.dumps(proposal.model_dump(exclude_none=True)), proposal, None

        # 1. Critical Priority: Heal pet if sick
        if pet.is_sick:
            if inv.medicine > 0:
                proposal = ActionProposal(action="heal")
                return json.dumps(proposal.model_dump(exclude_none=True)), proposal, None
            elif agent.money >= 75:
                proposal = ActionProposal(action="buy", item="medicine", amount=1)
                return json.dumps(proposal.model_dump(exclude_none=True)), proposal, None

        # 2. High Priority: Feed pet if hungry
        if pet.hunger >= 50:
            if inv.food > 0:
                proposal = ActionProposal(action="feed")
                return json.dumps(proposal.model_dump(exclude_none=True)), proposal, None
            elif agent.money >= 30:
                amount = min(3, agent.money // 30)
                proposal = ActionProposal(action="buy", item="food", amount=amount)
                return json.dumps(proposal.model_dump(exclude_none=True)), proposal, None

        # 3. Priority: Clean pet if dirty
        if pet.cleanliness < 40 and agent.energy >= 5:
            proposal = ActionProposal(action="clean")
            return json.dumps(proposal.model_dump(exclude_none=True)), proposal, None

        # 4. Work Priority: Earn money if low on food/medicine or cash
        if inv.food <= 1 or agent.money < 100:
            # Pick best available job agent has energy for
            affordable_jobs = [j for j in observation.jobs_available if agent.energy >= j.energy_cost]
            if affordable_jobs:
                # Prefer shorter cafe shift if pet is hungry, else highest reward
                if pet.hunger > 40:
                    best_job = min(affordable_jobs, key=lambda j: j.duration_minutes)
                else:
                    best_job = max(affordable_jobs, key=lambda j: j.reward)
                proposal = ActionProposal(action="work", job_id=best_job.id)
                return json.dumps(proposal.model_dump(exclude_none=True)), proposal, None

        # 5. Play / Rest Priority
        if pet.happiness < 60 and agent.energy >= 10:
            proposal = ActionProposal(action="play")
            return json.dumps(proposal.model_dump(exclude_none=True)), proposal, None

        # Fallback: Wait 60 minutes
        proposal = ActionProposal(action="wait", minutes=60)
        return json.dumps(proposal.model_dump(exclude_none=True)), proposal, None
