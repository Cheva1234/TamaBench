"""Base Agent Interface for TamaBench V1."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
from tamabench.schemas.observation import Observation
from tamabench.schemas.actions import ActionProposal
from tamabench.schemas.errors import BenchmarkError


class BaseAgent(ABC):
    def __init__(self, name: str = "BaseAgent"):
        self.name = name

    @abstractmethod
    def select_action(
        self, observation: Observation
    ) -> Tuple[str, Optional[ActionProposal], Optional[BenchmarkError]]:
        """Given an Observation, returns (raw_output_str, ActionProposal, SchemaError)."""
        pass
