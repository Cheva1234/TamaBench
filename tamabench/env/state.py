"""State models and SHA-256 state hashing for TamaBench V1."""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any
from tamabench.schemas.observation import (
    Observation,
    TimeState,
    AgentObservation,
    PetObservation,
    InventoryObservation,
    JobObservation,
    ShopItemObservation,
)


@dataclass
class AgentState:
    money: int = 100
    energy: float = 100.0
    current_activity: str = "idle"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PetState:
    health: float = 100.0
    hunger: float = 20.0
    energy: float = 80.0
    happiness: float = 70.0
    cleanliness: float = 90.0
    is_sick: bool = False
    is_sleeping: bool = False
    age: int = 0  # In simulation minutes

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Inventory:
    food: int = 3
    medicine: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Job:
    id: str
    name: str
    duration_minutes: int
    reward: int
    energy_cost: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShopItem:
    item: str
    cost: int
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorldState:
    total_minutes: int = 0
    agent: AgentState = field(default_factory=AgentState)
    pet: PetState = field(default_factory=PetState)
    inventory: Inventory = field(default_factory=Inventory)
    jobs_available: list[Job] = field(default_factory=list)
    shop_items_available: list[ShopItem] = field(default_factory=list)

    # Metadata for scenario versioning
    benchmark_version: str = "0.1.0"
    environment_version: str = "0.1.0"
    scenario_id: str = "standard_v1"
    scenario_version: int = 1
    seed: int = 42

    @property
    def day(self) -> int:
        return (self.total_minutes // 1440) + 1

    @property
    def hour(self) -> int:
        return (self.total_minutes % 1440) // 60

    @property
    def minute(self) -> int:
        return self.total_minutes % 60

    def compute_hash(self) -> str:
        """Computes deterministic SHA-256 state snapshot hash for auditability and replay verification."""
        pet_dict = self.pet.to_dict()
        for k, v in pet_dict.items():
            if isinstance(v, float):
                pet_dict[k] = round(v, 4)

        state_dict = {
            "time": self.total_minutes,
            "agent": {
                "money": self.agent.money,
                "energy": round(self.agent.energy, 4),
                "current_activity": self.agent.current_activity,
            },
            "pet": pet_dict,
            "inventory": self.inventory.to_dict(),
            "jobs": [j.to_dict() for j in self.jobs_available],
            "shop": [s.to_dict() for s in self.shop_items_available],
            "scenario": {
                "id": self.scenario_id,
                "version": self.scenario_version,
                "seed": self.seed,
            },
        }
        json_bytes = json.dumps(state_dict, sort_keys=True).encode("utf-8")
        return f"sha256:{hashlib.sha256(json_bytes).hexdigest()}"

    def to_observation(self) -> Observation:
        return Observation(
            time=TimeState(
                day=self.day,
                hour=self.hour,
                minute=self.minute,
                total_minutes=self.total_minutes,
            ),
            agent=AgentObservation(
                money=self.agent.money,
                energy=int(round(self.agent.energy)),
                activity=self.agent.current_activity,
            ),
            pet=PetObservation(
                health=round(self.pet.health, 1),
                hunger=round(self.pet.hunger, 1),
                energy=round(self.pet.energy, 1),
                happiness=round(self.pet.happiness, 1),
                cleanliness=round(self.pet.cleanliness, 1),
                is_sick=self.pet.is_sick,
                is_sleeping=self.pet.is_sleeping,
                age=self.pet.age,
            ),
            inventory=InventoryObservation(
                food=self.inventory.food,
                medicine=self.inventory.medicine,
            ),
            jobs_available=[
                JobObservation(
                    id=j.id,
                    name=j.name,
                    duration_minutes=j.duration_minutes,
                    reward=j.reward,
                    energy_cost=j.energy_cost,
                )
                for j in self.jobs_available
            ],
            shop_items_available=[
                ShopItemObservation(
                    item=s.item,
                    cost=s.cost,
                    description=s.description,
                )
                for s in self.shop_items_available
            ],
            state_hash=self.compute_hash(),
        )
