"""Oracle Evaluator (Reference Policy) for TamaBench V1.

Evaluates an agent's proposed action against the reference policy in critical and standard scenarios.
Does NOT act as a competing agent.
"""

from dataclasses import dataclass
from typing import Optional
from tamabench.env.state import WorldState
from tamabench.schemas.actions import ActionProposal


@dataclass
class DecisionEvaluation:
    is_critical_scenario: bool
    is_acceptable_action: bool
    optimality_score: float  # 0.0 to 1.0
    rationale: str


class ReferencePolicyEvaluator:
    @classmethod
    def evaluate_decision(
        cls, state: WorldState, proposal: Optional[ActionProposal]
    ) -> DecisionEvaluation:
        """Evaluates `proposal` against optimal policy for `state`."""
        if proposal is None:
            return DecisionEvaluation(
                is_critical_scenario=False,
                is_acceptable_action=False,
                optimality_score=0.0,
                rationale="No valid action proposal provided (Schema Error).",
            )

        pet = state.pet
        agent = state.agent
        inv = state.inventory
        action = proposal.action

        # Scenario 1: Critical Sickness (Pet is sick)
        if pet.is_sick:
            is_critical = True
            if inv.medicine > 0:
                if action == "heal":
                    return DecisionEvaluation(
                        is_critical_scenario=is_critical,
                        is_acceptable_action=True,
                        optimality_score=1.0,
                        rationale="Optimal: Administered medicine to sick pet.",
                    )
                else:
                    return DecisionEvaluation(
                        is_critical_scenario=is_critical,
                        is_acceptable_action=False,
                        optimality_score=0.0,
                        rationale="Unacceptable: Failed to heal sick pet despite having medicine.",
                    )
            else:
                if action == "buy" and proposal.item == "medicine":
                    return DecisionEvaluation(
                        is_critical_scenario=is_critical,
                        is_acceptable_action=True,
                        optimality_score=1.0,
                        rationale="Optimal: Bought medicine for sick pet.",
                    )
                elif action == "work" and agent.money < 50:
                    return DecisionEvaluation(
                        is_critical_scenario=is_critical,
                        is_acceptable_action=True,
                        optimality_score=0.7,
                        rationale="Acceptable: Working to earn money for medicine.",
                    )
                else:
                    return DecisionEvaluation(
                        is_critical_scenario=is_critical,
                        is_acceptable_action=False,
                        optimality_score=0.1,
                        rationale="Suboptimal: Did not prioritize acquiring medicine for sick pet.",
                    )

        # Scenario 2: Critical Starvation (Pet hunger > 80)
        if pet.hunger >= 80.0:
            is_critical = True
            if inv.food > 0:
                if action == "feed":
                    return DecisionEvaluation(
                        is_critical_scenario=is_critical,
                        is_acceptable_action=True,
                        optimality_score=1.0,
                        rationale="Optimal: Fed starving pet.",
                    )
                else:
                    return DecisionEvaluation(
                        is_critical_scenario=is_critical,
                        is_acceptable_action=False,
                        optimality_score=0.0,
                        rationale="Unacceptable: Did not feed starving pet despite having food.",
                    )
            else:
                if action == "buy" and proposal.item == "food":
                    return DecisionEvaluation(
                        is_critical_scenario=is_critical,
                        is_acceptable_action=True,
                        optimality_score=1.0,
                        rationale="Optimal: Bought food for starving pet.",
                    )
                elif action == "work":
                    return DecisionEvaluation(
                        is_critical_scenario=is_critical,
                        is_acceptable_action=True,
                        optimality_score=0.6,
                        rationale="Acceptable: Working to earn money for food.",
                    )

        # Standard Non-Critical Scenarios
        is_critical = False
        if action in ["feed", "play", "clean", "work", "buy", "wait", "observe"]:
            return DecisionEvaluation(
                is_critical_scenario=is_critical,
                is_acceptable_action=True,
                optimality_score=0.8,
                rationale="Standard acceptable action.",
            )

        return DecisionEvaluation(
            is_critical_scenario=is_critical,
            is_acceptable_action=True,
            optimality_score=0.5,
            rationale="Neutral action.",
        )
