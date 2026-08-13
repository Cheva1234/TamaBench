"""Calibration Test: Stationary Context Growth & Slope Regression Test.

Simulates 30 days (43,200 simulation minutes) and fits a linear regression model
tokens = a + b * day, enforcing max_growth_slope <= 2.0 tokens/day and max_tokens <= 3000.
"""

import pytest
import numpy as np
from tamabench.env.core import TamaEnv
from tamabench.context.builder import ContextBuilder
from tamabench.agents.rule_agent import RuleAgent


@pytest.mark.calibration
def test_context_growth_stationarity_over_30_days():
    env = TamaEnv()
    obs = env.reset(seed=1842)
    agent = RuleAgent()

    sampled_days: list[float] = []
    sampled_context_tokens: list[int] = []
    recent_events: list[str] = []

    max_minutes = 43200  # 30 days
    last_sampled_day = -1

    while env.state.total_minutes < max_minutes:
        # Keep pet alive and state stable for 30-day context capacity benchmark
        if env.terminated:
            env.terminated = False
            env.state.pet.health = 100.0
            env.state.pet.hunger = 20.0
            env.state.inventory.food = 5

        obs = env.observe()
        prompt = ContextBuilder.build_prompt(
            observation=obs,
            recent_event_descriptions=recent_events,
            schema_mode="raw_json",
        )
        token_estimate = len(prompt) // 4

        current_day = env.state.total_minutes // 1440
        if current_day > last_sampled_day:
            sampled_days.append(float(current_day))
            sampled_context_tokens.append(token_estimate)
            last_sampled_day = current_day

        raw_output, proposal, _ = agent.select_action(obs)
        if proposal:
            step_res = env.step(proposal)
            recent_events.append(f"Action '{proposal.action}' executed at minute {env.state.total_minutes}")

    assert len(sampled_context_tokens) >= 20, f"Insufficient sample points over 30 days ({len(sampled_context_tokens)} sampled)."

    # Fit linear regression model: tokens = a + b * day
    x = np.array(sampled_days)
    y = np.array(sampled_context_tokens)

    # Calculate slope b = Cov(x, y) / Var(x)
    n = len(x)
    slope = float(((n * np.sum(x * y)) - (np.sum(x) * np.sum(y))) / ((n * np.sum(x ** 2)) - (np.sum(x) ** 2)))
    max_tokens = int(np.max(y))
    mean_tokens = float(np.mean(y))

    # Assert Stationarity: Max tokens <= 3000 AND slope <= 2.0 tokens/day
    assert max_tokens <= 3000, f"Context cap exceeded! Max tokens was {max_tokens} > 3000."
    assert slope <= 2.0, f"Context growth detected! Slope was {slope:.2f} tokens/day > 2.0 tokens/day threshold."
