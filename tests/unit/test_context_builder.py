from tamabench.context.builder import ContextBuilder
from tamabench.env.core import TamaEnv
from tamabench.env.time_engine import BenchmarkMode


def test_prompt_explains_fullness_direction_and_continuous_health_damage():
    observation = TamaEnv(mode=BenchmarkMode.ACCELERATED).reset(seed=42)
    observation = observation.model_copy(
        update={
            "pet": observation.pet.model_copy(update={"hunger": 35.5}),
        }
    )

    prompt = ContextBuilder.build_prompt(observation, [])

    prompt = " ".join(prompt.lower().split())

    assert "hunger is a fullness meter, not a starvation score" in prompt
    assert "0 means starving and 100 means fully fed" in prompt
    assert "feeding increases hunger by 35" in prompt
    assert "hunger decreases by 18 per simulated hour" in prompt
    assert "health decreases continuously" in prompt
    assert "3-hour sleep removes about 54 hunger" in prompt
    assert "feed first when fullness is low" in prompt


def test_prompt_gives_small_models_minimal_valid_json_examples():
    observation = TamaEnv(mode=BenchmarkMode.ACCELERATED).reset(seed=42)

    prompt = ContextBuilder.build_prompt(observation, [])

    assert "prediction and trace fields are optional" in prompt.lower()
    assert '{"action":"feed"}' in prompt
    assert '{"action":"wait","minutes":60}' in prompt
