"""Stage 2: Environment and Precondition Validator for TamaBench V1.

Validates whether a schema-valid action can be executed in the current WorldState.
Returns Environment Errors (PRECONDITION_FAILED, INSUFFICIENT_RESOURCE, etc.).
"""

from typing import Optional
from tamabench.env.state import WorldState
from tamabench.env.economy import EconomySystem
from tamabench.schemas.actions import ActionProposal
from tamabench.schemas.errors import BenchmarkError, ErrorCategory, ErrorType


class EnvironmentValidator:
    @classmethod
    def validate_preconditions(
        cls, proposal: ActionProposal, state: WorldState
    ) -> Optional[BenchmarkError]:
        """Checks domain preconditions and returns Environment Error if invalid."""
        action = proposal.action
        pet = state.pet
        agent = state.agent
        inv = state.inventory

        if action == "feed":
            if pet.is_sleeping:
                return BenchmarkError(
                    category=ErrorCategory.ENVIRONMENT,
                    error_type=ErrorType.PRECONDITION_FAILED,
                    message="Cannot feed pet while pet is sleeping.",
                )
            if inv.food <= 0:
                return BenchmarkError(
                    category=ErrorCategory.ENVIRONMENT,
                    error_type=ErrorType.INSUFFICIENT_RESOURCE,
                    message="Insufficient food inventory (food count: 0).",
                )

        elif action == "play":
            if pet.is_sleeping:
                return BenchmarkError(
                    category=ErrorCategory.ENVIRONMENT,
                    error_type=ErrorType.PRECONDITION_FAILED,
                    message="Cannot play with pet while pet is sleeping.",
                )
            if agent.energy < 10:
                return BenchmarkError(
                    category=ErrorCategory.ENVIRONMENT,
                    error_type=ErrorType.INSUFFICIENT_RESOURCE,
                    message="Agent energy too low to play (requires 10 energy).",
                )

        elif action == "clean":
            if pet.is_sleeping:
                return BenchmarkError(
                    category=ErrorCategory.ENVIRONMENT,
                    error_type=ErrorType.PRECONDITION_FAILED,
                    message="Cannot clean pet while pet is sleeping.",
                )
            if agent.energy < 5:
                return BenchmarkError(
                    category=ErrorCategory.ENVIRONMENT,
                    error_type=ErrorType.INSUFFICIENT_RESOURCE,
                    message="Agent energy too low to clean (requires 5 energy).",
                )

        elif action == "heal":
            if pet.is_sleeping:
                return BenchmarkError(
                    category=ErrorCategory.ENVIRONMENT,
                    error_type=ErrorType.PRECONDITION_FAILED,
                    message="Cannot administer medicine while pet is sleeping.",
                )
            if inv.medicine <= 0:
                return BenchmarkError(
                    category=ErrorCategory.ENVIRONMENT,
                    error_type=ErrorType.INSUFFICIENT_RESOURCE,
                    message="Insufficient medicine inventory (medicine count: 0).",
                )

        elif action == "sleep":
            if pet.is_sleeping:
                return BenchmarkError(
                    category=ErrorCategory.ENVIRONMENT,
                    error_type=ErrorType.PRECONDITION_FAILED,
                    message="Pet is already sleeping.",
                )

        elif action == "wake":
            if not pet.is_sleeping:
                return BenchmarkError(
                    category=ErrorCategory.ENVIRONMENT,
                    error_type=ErrorType.PRECONDITION_FAILED,
                    message="Pet is already awake.",
                )

        elif action == "work":
            job = EconomySystem.find_job(proposal.job_id or "", state)
            if not job:
                return BenchmarkError(
                    category=ErrorCategory.ENVIRONMENT,
                    error_type=ErrorType.ACTION_UNAVAILABLE,
                    message=f"Job '{proposal.job_id}' is not available.",
                )
            if agent.energy < job.energy_cost:
                return BenchmarkError(
                    category=ErrorCategory.ENVIRONMENT,
                    error_type=ErrorType.INSUFFICIENT_RESOURCE,
                    message=f"Agent energy ({agent.energy}) is insufficient for job '{job.id}' (requires {job.energy_cost}).",
                )

        elif action == "buy":
            item = EconomySystem.find_shop_item(proposal.item or "", state)
            if not item:
                return BenchmarkError(
                    category=ErrorCategory.ENVIRONMENT,
                    error_type=ErrorType.ACTION_UNAVAILABLE,
                    message=f"Item '{proposal.item}' is not sold in shop.",
                )
            amount = proposal.amount or 1
            total_cost = item.cost * amount
            if agent.money < total_cost:
                return BenchmarkError(
                    category=ErrorCategory.ENVIRONMENT,
                    error_type=ErrorType.INSUFFICIENT_RESOURCE,
                    message=f"Insufficient funds: cost is ${total_cost}, agent has ${agent.money}.",
                )

        return None
