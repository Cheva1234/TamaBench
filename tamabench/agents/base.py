"""Base Agent Interface for TamaBench V1."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Tuple, Optional
from tamabench.schemas.observation import Observation
from tamabench.schemas.actions import ActionProposal
from tamabench.schemas.errors import BenchmarkError


@dataclass
class DecisionMetadata:
    """Per-decision runtime facts shared by model and baseline agents."""

    finish_reason: Optional[str] = None
    was_truncated: bool = False
    generation_attempt: int = 1
    attempt_count: int = 1
    first_pass_valid: bool = True
    final_valid: bool = True
    recovered: bool = False
    first_failure_type: Optional[str] = None
    first_failure_message: Optional[str] = None
    input_tokens: int = 0
    total_output_tokens: int = 0
    reasoning_tokens: int = 0
    json_tokens: int = 0
    raw_json: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.total_output_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"total_tokens": self.total_tokens}


class BaseAgent(ABC):
    def __init__(self, name: str = "BaseAgent"):
        self.name = name
        self.last_decision = DecisionMetadata()

    def reset_decision_metadata(self) -> None:
        self.last_decision = DecisionMetadata()

    def reset_episode(self) -> None:
        """Hook called by the runner at the start of every episode.

        Agents that carry cross-decision state (e.g. harness schedulers)
        must clear it here so state never leaks between episodes. Baselines
        have a no-op implementation.
        """
        return None

    def warmup(self) -> float:
        """Warm a model if the agent owns one; baselines have no-op warmup."""
        return 0.0

    def close(self) -> None:
        """Release model resources; baselines have no-op close."""
        return None

    @abstractmethod
    def select_action(
        self, observation: Observation
    ) -> Tuple[str, Optional[ActionProposal], Optional[BenchmarkError]]:
        """Given an Observation, returns (raw_output_str, ActionProposal, SchemaError)."""
        pass
