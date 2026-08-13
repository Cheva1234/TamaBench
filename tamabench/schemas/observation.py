"""Canonical Observation Schema for TamaBench V1."""

from typing import Any
from pydantic import BaseModel, Field


class TimeState(BaseModel):
    day: int = Field(ge=1, description="Simulation day number")
    hour: int = Field(ge=0, le=23, description="Simulation hour (0-23)")
    minute: int = Field(ge=0, le=59, description="Simulation minute (0-59)")
    total_minutes: int = Field(ge=0, description="Total elapsed simulation minutes")


class AgentObservation(BaseModel):
    money: int = Field(ge=0, description="Current agent balance")
    energy: int = Field(ge=0, le=100, description="Current agent energy (0-100)")
    activity: str = Field(description="Current activity (e.g., idle, working, sleeping)")


class PetObservation(BaseModel):
    health: float = Field(ge=0.0, le=100.0, description="Pet health (0-100)")
    hunger: float = Field(
        ge=0.0,
        le=100.0,
        description="Pet hunger/fullness level: 0 means starving and 100 means fully fed",
    )
    energy: float = Field(ge=0.0, le=100.0, description="Pet energy level (0-100)")
    happiness: float = Field(ge=0.0, le=100.0, description="Pet happiness level (0-100)")
    cleanliness: float = Field(ge=0.0, le=100.0, description="Pet cleanliness level (0-100)")
    is_sick: bool = Field(description="Whether pet is currently sick")
    is_sleeping: bool = Field(description="Whether pet is currently sleeping")
    age: int = Field(ge=0, description="Pet age in simulation minutes")


class InventoryObservation(BaseModel):
    food: int = Field(ge=0, description="Quantity of food in inventory")
    medicine: int = Field(ge=0, description="Quantity of medicine in inventory")


class JobObservation(BaseModel):
    id: str = Field(description="Unique job identifier")
    name: str = Field(description="Readable job title")
    duration_minutes: int = Field(gt=0, description="Required job duration in simulation minutes")
    reward: int = Field(gt=0, description="Monetary compensation earned upon completion")
    energy_cost: int = Field(ge=0, description="Agent energy consumed during job")


class ShopItemObservation(BaseModel):
    item: str = Field(description="Item name (e.g. food, medicine)")
    cost: int = Field(gt=0, description="Cost per unit in money")
    description: str = Field(description="Item description and usage effect")


class Observation(BaseModel):
    time: TimeState
    agent: AgentObservation
    pet: PetObservation
    inventory: InventoryObservation
    jobs_available: list[JobObservation]
    shop_items_available: list[ShopItemObservation]
    state_hash: str = Field(description="SHA-256 state snapshot hash for replay & verification")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
