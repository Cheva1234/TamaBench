"""Economy, Jobs registry, and Shop system for TamaBench V1.

Implements a dynamic economy that scales in difficulty over time based on an asymptotic limit.
"""

import math
from typing import Optional
from tamabench.env.state import Job, ShopItem, WorldState


class EconomySystem:
    FOOD_HUNGER_RESTORE: int = 35

    @classmethod
    def _calculate_limit(cls, initial: float, limit: float, total_minutes: int, k: float = 0.0005) -> int:
        """
        Calculates the asymptotic limit for the economy.
        As total_minutes -> infinity, the value approaches 'limit'.
        k controls the rate of decay.
        """
        val = limit + (initial - limit) * math.exp(-k * total_minutes)
        return int(round(val))

    @classmethod
    def get_dynamic_jobs(cls, total_minutes: int) -> list[Job]:
        # Cafe Shift starts at 60, approaches 30
        cafe_reward = cls._calculate_limit(60, 30, total_minutes)
        # Express Delivery starts at 120, approaches 60
        delivery_reward = cls._calculate_limit(120, 60, total_minutes)
        # Freelance Task starts at 20, approaches 10
        freelance_reward = cls._calculate_limit(20, 10, total_minutes)
        
        return [
            Job(
                id="cafe_shift",
                name="Cafe Shift",
                duration_minutes=60,
                reward=cafe_reward,
                energy_cost=20,
            ),
            Job(
                id="delivery",
                name="Express Delivery",
                duration_minutes=120,
                reward=delivery_reward,
                energy_cost=40,
            ),
            Job(
                id="freelance",
                name="Freelance Task",
                duration_minutes=30,
                reward=freelance_reward,
                energy_cost=12,
            ),
        ]

    @classmethod
    def get_dynamic_shop(cls, total_minutes: int) -> list[ShopItem]:
        # Food cost starts at 10, approaches 30
        food_cost = cls._calculate_limit(10, 30, total_minutes)
        # Medicine cost starts at 25, approaches 75
        medicine_cost = cls._calculate_limit(25, 75, total_minutes)
        
        return [
            ShopItem(
                item="food",
                cost=food_cost,
                description="Increases the hunger/fullness meter by 35 and increases happiness by 5.",
            ),
            ShopItem(
                item="medicine",
                cost=medicine_cost,
                description="Cures sickness and restores 15 health.",
            ),
        ]

    @classmethod
    def find_job(cls, job_id: str, state: WorldState) -> Optional[Job]:
        for job in state.jobs_available:
            if job.id == job_id:
                return job
        return None

    @classmethod
    def find_shop_item(cls, item_name: str, state: WorldState) -> Optional[ShopItem]:
        for item in state.shop_items_available:
            if item.item == item_name:
                return item
        return None
