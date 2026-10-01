"""Versioned scenario registry. Legacy names are explicit compatibility aliases."""

from pydantic import BaseModel, ConfigDict


class ScenarioConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    max_simulated_minutes: int
    initial_money: int
    initial_food: int
    initial_medicine: int = 1
    sickness_events: bool = True
    initial_fullness: float = 100.0


_SCENARIOS = {
    "dynamic_v2": ScenarioConfig(
        id="dynamic_v2", max_simulated_minutes=7 * 24 * 60,
        initial_money=50, initial_food=3, initial_medicine=1,
        sickness_events=True, initial_fullness=100.0,
    ),
}
# standard_v1 selects current dynamic_v2 rules, not historical V1 dynamics.
SCENARIO_ALIASES = {"standard_v1": "dynamic_v2"}


def get_config_for_scenario(scenario_id: str) -> ScenarioConfig:
    canonical = SCENARIO_ALIASES.get(scenario_id, scenario_id)
    if canonical not in _SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario_id}'. Choose dynamic_v2 (standard_v1 is a compatibility alias).")
    return _SCENARIOS[canonical]


def get_difficulty_config(difficulty: str | None) -> ScenarioConfig:
    """Legacy difficulty flags all select the documented dynamic scenario."""
    return _SCENARIOS["dynamic_v2"]
