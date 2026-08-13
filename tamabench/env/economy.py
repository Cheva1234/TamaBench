"""Economy, Jobs registry, and Shop system for TamaBench V1.

Implements a balanced, challenging economic environment for small AI models.
"""

from typing import Optional
from tamabench.env.state import Job, ShopItem, WorldState


class EconomySystem:
    DEFAULT_JOBS: list[Job] = [
        Job(
            id="cafe_shift",
            name="Cafe Shift",
            duration_minutes=60,
            reward=25,
            energy_cost=20,
        ),
        Job(
            id="delivery",
            name="Express Delivery",
            duration_minutes=120,
            reward=55,
            energy_cost=40,
        ),
        Job(
            id="freelance",
            name="Freelance Task",
            duration_minutes=30,
            reward=10,
            energy_cost=12,
        ),
    ]

    DEFAULT_SHOP: list[ShopItem] = [
        ShopItem(
            item="food",
            cost=30,
            description="Restores 35 hunger and increases happiness by 5.",
        ),
        ShopItem(
            item="medicine",
            cost=75,
            description="Cures sickness and restores 15 health.",
        ),
    ]

    @classmethod
    def get_default_jobs(cls) -> list[Job]:
        return [Job(**j.to_dict()) for j in cls.DEFAULT_JOBS]

    @classmethod
    def get_default_shop(cls) -> list[ShopItem]:
        return [ShopItem(**s.to_dict()) for s in cls.DEFAULT_SHOP]

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
