"""Reproducible difficulty presets for TamaBench episodes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DifficultyConfig:
    name: str
    scenario_id: str
    max_simulated_minutes: int
    initial_money: int
    initial_food: int
    initial_medicine: int
    initial_fullness: float
    sickness_events: bool


DIFFICULTY_PRESETS: dict[str, DifficultyConfig] = {
    "easy": DifficultyConfig(
        name="easy",
        scenario_id="easy_v1",
        max_simulated_minutes=1440,
        initial_money=100,
        initial_food=5,
        initial_medicine=1,
        initial_fullness=100.0,
        sickness_events=False,
    ),
    "standard": DifficultyConfig(
        name="standard",
        scenario_id="standard_v1",
        max_simulated_minutes=4320,
        initial_money=30,
        initial_food=1,
        initial_medicine=0,
        initial_fullness=80.0,
        sickness_events=True,
    ),
    "hard": DifficultyConfig(
        name="hard",
        scenario_id="hard_v1",
        max_simulated_minutes=10080,
        initial_money=20,
        initial_food=1,
        initial_medicine=0,
        initial_fullness=80.0,
        sickness_events=True,
    ),
}


def get_difficulty_config(difficulty: str) -> DifficultyConfig:
    try:
        return DIFFICULTY_PRESETS[difficulty]
    except KeyError as exc:
        valid = ", ".join(DIFFICULTY_PRESETS)
        raise ValueError(f"Unknown difficulty '{difficulty}'. Choose one of: {valid}.") from exc


def get_config_for_scenario(scenario_id: str) -> DifficultyConfig:
    for config in DIFFICULTY_PRESETS.values():
        if config.scenario_id == scenario_id:
            return config
    return get_difficulty_config("standard")
