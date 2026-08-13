from tamabench.context.builder import ContextBuilder
from tamabench.env.core import TamaEnv
from tamabench.env.scenarios import get_difficulty_config
from tamabench.env.time_engine import BenchmarkMode


def test_difficulty_presets_define_a_baseline_range():
    easy = get_difficulty_config("easy")
    standard = get_difficulty_config("standard")
    hard = get_difficulty_config("hard")

    assert easy.max_simulated_minutes == 1440
    assert easy.initial_money == 100
    assert easy.initial_food == 5
    assert easy.sickness_events is False

    assert standard.max_simulated_minutes == 4320
    assert standard.initial_money == 30
    assert standard.initial_food == 1
    assert standard.sickness_events is True

    assert hard.max_simulated_minutes == 10080
    assert hard.initial_money == 20
    assert hard.initial_food == 1
    assert hard.sickness_events is True


def test_environment_applies_selected_difficulty_and_prompt_horizon():
    env = TamaEnv(mode=BenchmarkMode.ACCELERATED)
    observation = env.reset(seed=42, difficulty="easy")

    assert observation.scenario_id == "easy_v1"
    assert observation.agent.money == 100
    assert observation.inventory.food == 5
    assert observation.inventory.medicine == 1
    assert observation.pet.hunger == 100.0
    assert env.scheduler.scheduled_events == []

    prompt = ContextBuilder.build_prompt(observation, [])
    assert "Keep the pet alive for 1 simulated day" in prompt
    assert "Difficulty: easy" in prompt
