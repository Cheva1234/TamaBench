"""Schemas and error taxonomy for TamaBench V1."""

from tamabench.schemas.errors import ErrorType, ErrorCategory, BenchmarkError
from tamabench.schemas.observation import Observation, TimeState, AgentObservation, PetObservation, InventoryObservation, JobObservation, ShopItemObservation
from tamabench.schemas.actions import ActionType, ActionProposal, DecisionTrace, ActionPrediction, StepResult

__all__ = [
    "ErrorType",
    "ErrorCategory",
    "BenchmarkError",
    "Observation",
    "TimeState",
    "AgentObservation",
    "PetObservation",
    "InventoryObservation",
    "JobObservation",
    "ShopItemObservation",
    "ActionType",
    "ActionProposal",
    "DecisionTrace",
    "ActionPrediction",
    "StepResult",
]
