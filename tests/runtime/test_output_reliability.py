import json

import pytest

from tamabench.agents.raw_llm_agent import RawLLMAgent
from tamabench.env.core import TamaEnv
from tamabench.runner.batch_runner import BatchRunner
from tamabench.runtime.model_runtime import ModelGenerationResponse, ModelRuntime
from tamabench.schemas.errors import ErrorType


class FakeRuntime(ModelRuntime):
    def __init__(self, responses: list[ModelGenerationResponse]):
        super().__init__(model_name="fake")
        self.responses = responses
        self.payloads: list[dict] = []

    def warmup(self) -> float:
        self.model_resident = True
        return 0.0

    def generate(self, payload: dict) -> ModelGenerationResponse:
        self.payloads.append(payload)
        return self.responses.pop(0)


def test_raw_llm_agent_uses_configured_output_limit_and_recovers_truncation():
    runtime = FakeRuntime(
        [
            ModelGenerationResponse(
                content='{"action":"wait"',
                finish_reason="length",
                output_tokens=16,
            ),
            ModelGenerationResponse(
                content=json.dumps({"action": "wait", "minutes": 30}),
                finish_reason="stop",
                output_tokens=12,
            ),
        ]
    )
    agent = RawLLMAgent(
        runtime=runtime,
        max_output_tokens=4096,
        max_retries=1,
    )

    observation = TamaEnv().reset(seed=42)
    _, proposal, error = agent.select_action(observation)

    assert runtime.payloads[0]["max_tokens"] == 4096
    assert error is None
    assert proposal is not None
    assert agent.last_decision.first_pass_valid is False
    assert agent.last_decision.final_valid is True
    assert agent.last_decision.recovered is True
    assert agent.last_decision.attempt_count == 2
    assert agent.last_decision.first_failure_type == ErrorType.OUTPUT_TRUNCATED.value


def test_raw_llm_agent_passes_reasoning_effort_to_runtime():
    runtime = FakeRuntime(
        [
            ModelGenerationResponse(
                content=json.dumps({"action": "wait", "minutes": 30}),
                finish_reason="stop",
                output_tokens=12,
            )
        ]
    )
    agent = RawLLMAgent(runtime=runtime, reasoning_effort="none", max_retries=0)

    agent.select_action(TamaEnv().reset(seed=42))

    assert runtime.payloads[0]["reasoning_effort"] == "none"


def test_exhausted_length_generation_is_not_reported_as_invalid_json():
    runtime = FakeRuntime(
        [
            ModelGenerationResponse(
                content='{"action":"wait"',
                finish_reason="length",
            )
        ]
    )
    agent = RawLLMAgent(runtime=runtime, max_retries=0)

    _, proposal, error = agent.select_action(TamaEnv().reset(seed=42))

    assert proposal is None
    assert error is not None
    assert error.error_type == ErrorType.OUTPUT_TRUNCATED
    assert agent.last_decision.was_truncated is True
    assert agent.last_decision.final_valid is False


def test_reasoning_and_json_token_counts_are_recorded_separately():
    runtime = FakeRuntime(
        [
            ModelGenerationResponse(
                content='<think>choose the safest action</think>{"action":"wait"}',
                finish_reason="stop",
                output_tokens=20,
            )
        ]
    )
    agent = RawLLMAgent(runtime=runtime, max_retries=0)

    agent.select_action(TamaEnv().reset(seed=42))

    assert agent.last_decision.reasoning_tokens > 0
    assert agent.last_decision.json_tokens > 0
    assert agent.last_decision.total_output_tokens == 20


def test_metrics_keep_first_pass_failure_separate_from_recovery(tmp_path):
    runtime = FakeRuntime(
        [
            ModelGenerationResponse(
                content='{"action":"wait"',
                finish_reason="length",
                output_tokens=16,
            ),
            ModelGenerationResponse(
                content=json.dumps({"action": "wait", "minutes": 30}),
                finish_reason="stop",
                output_tokens=12,
            ),
        ]
    )
    agent = RawLLMAgent(runtime=runtime, max_retries=1)
    runner = BatchRunner(
        db_path=str(tmp_path / "metrics.db"),
        event_path=str(tmp_path / "metrics.jsonl"),
    )

    metrics = runner.run_episode(agent, seed=42, max_simulated_minutes=1)
    runner.close()

    assert metrics.first_pass_schema_acc == 0.0
    assert metrics.final_schema_acc == 100.0
    assert metrics.truncation_rate == 100.0
    assert metrics.retry_rate == 100.0
    assert metrics.final_schema_recovery_rate == 100.0
