"""State models, exact welfare accumulators, and contract 2.0 state hashing."""

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
    money: int = 50
    energy: float = 100.0
    current_activity: str = "idle"

    @property
    def available_energy(self) -> int:
        """Public whole-unit energy used for all action preconditions."""
        return int(round(self.energy))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PetState:
    health: float = 100.0
    # Fullness meter kept under the legacy `hunger` field name: 100 is full,
    # 0 is starving.
    hunger: float = 100.0
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
class SimulationStatistics:
    """Exact sum of every minute-end sample (units: statistic * minutes)."""

    elapsed_minutes: int = 0
    health_centiminutes: int = 0
    happiness_centiminutes: int = 0
    minimum_health_centi: int = 10000

    @property
    def health_integral(self) -> float:
        return self.health_centiminutes / 100

    @property
    def happiness_integral(self) -> float:
        return self.happiness_centiminutes / 100

    @property
    def min_health(self) -> float:
        return self.minimum_health_centi / 100

    @property
    def avg_health(self) -> float:
        return self.health_integral / self.elapsed_minutes if self.elapsed_minutes else 100.0

    @property
    def avg_happiness(self) -> float:
        return self.happiness_integral / self.elapsed_minutes if self.elapsed_minutes else 70.0

    def to_dict(self) -> dict[str, Any]:
        return {"elapsed_minutes": self.elapsed_minutes,
                "health_integral": self.health_integral,
                "happiness_integral": self.happiness_integral,
                "min_health": self.min_health,
                "avg_health": self.avg_health,
                "avg_happiness": self.avg_happiness}


@dataclass
class WorldState:
    total_minutes: int = 0
    agent: AgentState = field(default_factory=AgentState)
    pet: PetState = field(default_factory=PetState)
    inventory: Inventory = field(default_factory=Inventory)
    jobs_available: list[Job] = field(default_factory=list)
    shop_items_available: list[ShopItem] = field(default_factory=list)

    statistics: SimulationStatistics = field(default_factory=SimulationStatistics)
    max_simulated_minutes: int = 7 * 24 * 60

    # Metadata for scenario versioning
    benchmark_version: str = "2.0.0"
    environment_version: str = "2.0.0"
    scenario_id: str = "dynamic_v2"
    scenario_version: int = 2
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
            "horizon": self.max_simulated_minutes,
            "statistics": asdict(self.statistics),
            "benchmark_version": self.benchmark_version,
            "environment_version": self.environment_version,
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
                energy=self.agent.available_energy,
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
            scenario_id=self.scenario_id,
            scenario_version=self.scenario_version,
        )
