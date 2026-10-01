"""Canonical strict action and prediction schemas for contract 2.0."""

from enum import Enum
from typing import Any, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from tamabench.schemas.errors import BenchmarkError


class ActionType(str, Enum):
    OBSERVE = "observe"
    FEED = "feed"
    PLAY = "play"
    CLEAN = "clean"
    HEAL = "heal"
    SLEEP = "sleep"
    WAKE = "wake"
    WAIT = "wait"
    WORK = "work"
    BUY = "buy"


class StrictActionModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", validate_assignment=True,
                              revalidate_instances="always", allow_inf_nan=False)


class ActionPrediction(StrictActionModel):
    pet_safe_until_completion: Optional[bool] = Field(
        default=None, description="Agent prediction: will pet stay healthy/alive during action?"
    )
    expected_money_after: Optional[int] = Field(
        default=None, description="Agent prediction: expected agent money balance after action"
    )
    expected_hunger_after: Optional[float] = Field(
        default=None,
        description="Agent prediction: expected pet hunger/fullness level after action (0 starving, 100 full)",
    )
    expected_health_after: Optional[float] = Field(
        default=None, description="Agent prediction: expected pet health level after action"
    )


class DecisionTrace(StrictActionModel):
    situation_summary: str = Field(default="", description="Current situation analysis by agent")
    current_priority: str = Field(default="", description="Primary priority governing this decision")
    options_considered: list[str] = Field(default_factory=list, description="List of actions considered")
    chosen_action: str = Field(default="", description="Name of action selected")
    decision_rationale: str = Field(default="", description="Detailed rationale for choosing action")
    expected_result: str = Field(default="", description="High-level description of expected outcome")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Agent confidence score (0.0 to 1.0)")


class ActionProposal(StrictActionModel):
    action: Literal["observe", "feed", "play", "clean", "heal", "sleep",
                    "wake", "wait", "work", "buy"]
    # Action specific parameters
    job_id: Optional[str] = Field(default=None, description="Required for work action")
    item: Optional[str] = Field(default=None, description="Required for buy action (e.g. food, medicine)")
    amount: int = Field(default=1, gt=0, description="Optional parameter for buy action (unit count)")
    minutes: int = Field(default=30, gt=0, description="Optional duration in minutes for wait action")
    hours: Optional[int] = Field(default=None, description="Optional duration in hours for sleep action (3, 5, or 8)")
    
    # Structured prediction artifact from model
    prediction: Optional[ActionPrediction] = Field(default=None)
    # Requested decision trace from model
    trace: Optional[DecisionTrace] = Field(default=None)


    @field_validator("hours")
    @classmethod
    def supported_sleep_hours(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value not in (3, 5, 8):
            raise ValueError("hours must be one of 3, 5, or 8")
        return value

    @model_validator(mode="after")
    def required_arguments(self) -> "ActionProposal":
        if self.action == "work" and not self.job_id:
            raise ValueError("Action 'work' requires nonempty string argument 'job_id'")
        if self.action == "buy" and not self.item:
            raise ValueError("Action 'buy' requires nonempty string argument 'item'")
        return self


class StepResult(BaseModel):
    success: bool = Field(description="True if action executed cleanly without error")
    observation: Any = Field(description="Observation snapshot following step")
    completed: bool = Field(default=False, description="Requested action finished alive; success alone means accepted")
    horizon_reached: bool = Field(default=False, description="Configured simulation horizon reached")
    terminated: bool = Field(default=False, description="True if episode ended (e.g., pet died)")
    termination_reason: Optional[str] = Field(default=None, description="Reason if episode ended")
    error: Optional[BenchmarkError] = Field(default=None, description="Error detail if action failed")
    execution_minutes: int = Field(default=0, description="Simulation minutes advanced during action")
    state_hash: str = Field(description="SHA-256 state snapshot hash after action commit")
