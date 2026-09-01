from pydantic import BaseModel

class ScenarioConfig(BaseModel):
    id: str
    max_simulated_minutes: int
    initial_money: int
    initial_food: int
    initial_medicine: int = 1
    sickness_events: bool = True
    initial_fullness: float = 80.0

_SCENARIOS = {
    "dynamic_v2": ScenarioConfig(
        id="dynamic_v2",
        max_simulated_minutes=7 * 24 * 60,  # 7 days max to survive
        initial_money=50,
        initial_food=3,
        initial_medicine=1,
        sickness_events=True,
        initial_fullness=100.0,
    ),
}

def get_config_for_scenario(scenario_id: str) -> ScenarioConfig:
    return _SCENARIOS.get(scenario_id, _SCENARIOS["dynamic_v2"])

def get_difficulty_config(difficulty: str | None) -> ScenarioConfig:
    return _SCENARIOS["dynamic_v2"]
