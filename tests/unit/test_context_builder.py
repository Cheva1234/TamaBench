from tamabench.context.builder import ContextBuilder
from tamabench.env.core import TamaEnv
from tamabench.env.time_engine import BenchmarkMode


def test_prompt_explains_hunger_direction_and_continuous_health_damage():
    observation = TamaEnv(mode=BenchmarkMode.ACCELERATED).reset(seed=42)
    observation = observation.model_copy(
        update={
            "pet": observation.pet.model_copy(update={"hunger": 65.5}),
        }
    )

    prompt = ContextBuilder.build_prompt(observation, [])

    prompt = " ".join(prompt.lower().split())

    assert "hunger is a danger meter, not fullness" in prompt
    assert "0 means satisfied and 100 means starving" in prompt
    assert "health decreases continuously" in prompt
    assert "3-hour sleep adds about 54 hunger" in prompt
    assert "feed first if food is available" in prompt
