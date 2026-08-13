"""Canonical Action and Prediction Schemas for TamaBench V1."""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
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


class ActionPrediction(BaseModel):
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


class DecisionTrace(BaseModel):
    situation_summary: str = Field(default="", description="Current situation analysis by agent")
    current_priority: str = Field(default="", description="Primary priority governing this decision")
    options_considered: list[str] = Field(default_factory=list, description="List of actions considered")
    chosen_action: str = Field(default="", description="Name of action selected")
    decision_rationale: str = Field(default="", description="Detailed rationale for choosing action")
    expected_result: str = Field(default="", description="High-level description of expected outcome")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Agent confidence score (0.0 to 1.0)")


class ActionProposal(BaseModel):
    action: str = Field(description="Action name (e.g. feed, work, buy, wait, etc.)")
    # Action specific parameters
    job_id: Optional[str] = Field(default=None, description="Required for work action")
    item: Optional[str] = Field(default=None, description="Required for buy action (e.g. food, medicine)")
    amount: Optional[int] = Field(default=1, description="Optional parameter for buy action (unit count)")
    minutes: Optional[int] = Field(default=30, description="Optional duration in minutes for wait action")
    hours: Optional[int] = Field(default=None, description="Optional duration in hours for sleep action (3, 5, or 8)")
    
    # Structured prediction artifact from model
    prediction: Optional[ActionPrediction] = Field(default=None)
    # Requested decision trace from model
    trace: Optional[DecisionTrace] = Field(default=None)


class StepResult(BaseModel):
    success: bool = Field(description="True if action executed cleanly without error")
    observation: Any = Field(description="Observation snapshot following step")
    terminated: bool = Field(default=False, description="True if episode ended (e.g., pet died)")
    termination_reason: Optional[str] = Field(default=None, description="Reason if episode ended")
    error: Optional[BenchmarkError] = Field(default=None, description="Error detail if action failed")
    execution_minutes: int = Field(default=0, description="Simulation minutes advanced during action")
    state_hash: str = Field(description="SHA-256 state snapshot hash after action commit")
