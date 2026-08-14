"""Raw LLM agent adapter for small models in TamaBench V1.1."""

import json
import time
from typing import Any, Optional, Tuple

from tamabench.agents.base import BaseAgent
from tamabench.context.builder import ContextBuilder
from tamabench.env.time_engine import ComputeClock
from tamabench.runtime.model_runtime import ModelRuntime
from tamabench.schemas.actions import ActionProposal
from tamabench.schemas.errors import BenchmarkError, ErrorCategory, ErrorType
from tamabench.schemas.observation import Observation
from tamabench.validation.syntax_validator import SyntaxValidator


def _estimate_tokens(value: str) -> int:
    return len(value) // 4


def _extract_reasoning_and_json(content: str, provider_reasoning: str = "") -> tuple[str, str]:
    """Separate provider/thinking text from the JSON decision payload."""
    reasoning = provider_reasoning.strip()
    decision_text = content.strip()

    if "<think>" in decision_text:
        start = decision_text.find("<think>") + len("<think>")
        end = decision_text.find("</think>", start)
        if end >= 0:
            embedded_reasoning = decision_text[start:end].strip()
            reasoning = "\n".join(part for part in (reasoning, embedded_reasoning) if part)
            decision_text = decision_text[end + len("</think>"):].strip()

    first_brace = decision_text.find("{")
    if first_brace >= 0:
        decision_text = decision_text[first_brace:]

    return reasoning, decision_text


class RawLLMAgent(BaseAgent):
    """Calls an OpenAI-compatible model and records attempt-level facts."""

    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        api_base: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        schema_mode: str = "raw_json",
        temperature: float = 0.2,
        max_retries: int = 2,
        max_output_tokens: int = 4096,
        timeout: float = 120.0,
        runtime: Optional[ModelRuntime] = None,
        keep_alive: str | int = -1,
    ):
        super().__init__(name=f"RawLLM({model_name})")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than 0")

        self.model_name = model_name
        self.schema_mode = schema_mode
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens
        self.recent_events: list[str] = []
        self.last_compute: ComputeClock = ComputeClock()
        self.last_reasoning = ""
        self.last_json_output = ""
        self.last_finish_reason: Optional[str] = None
        self.last_was_truncated = False
        self.runtime = runtime or ModelRuntime(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            keep_alive=keep_alive,
            timeout=timeout,
        )

    @property
    def model_resident(self) -> bool:
        return self.runtime.model_resident

    @property
    def api_calls(self) -> int:
        return self.runtime.api_calls

    @property
    def model_warmup_ms(self) -> float:
        return self.runtime.warmup_ms

    def warmup(self) -> float:
        self.last_compute.model_load_ms = self.runtime.warmup()
        return self.last_compute.model_load_ms

    def close(self) -> None:
        self.runtime.close()

    def record_event(self, event_description: str):
        self.recent_events.append(event_description)

    @staticmethod
    def _truncation_error(raw_output: str, finish_reason: str) -> BenchmarkError:
        return BenchmarkError(
            category=ErrorCategory.SCHEMA,
            error_type=ErrorType.OUTPUT_TRUNCATED,
            message="Model generation stopped because the output token limit was reached.",
            details={"finish_reason": finish_reason, "raw_output": raw_output},
        )

    def select_action(
        self, observation: Observation
    ) -> Tuple[str, Optional[ActionProposal], Optional[BenchmarkError]]:
        self.reset_decision_metadata()
        self.last_compute.reset()
        self.last_reasoning = ""
        self.last_json_output = ""
        self.last_finish_reason = None
        self.last_was_truncated = False

        prompt = ContextBuilder.build_prompt(
            observation=observation,
            recent_event_descriptions=self.recent_events,
            schema_mode=self.schema_mode,
        )

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise autonomous TamaBench agent. "
                        "Respond with exactly one JSON object and no prose."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }

        if self.schema_mode == "provider_constrained":
            payload["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        raw_output = ""
        last_error: Optional[BenchmarkError] = None
        final_proposal: Optional[ActionProposal] = None
        first_failure: Optional[BenchmarkError] = None
        first_pass_valid = False
        final_finish_reason: Optional[str] = None

        for attempt in range(1, self.max_retries + 2):
            final_proposal = None
            try:
                response = self.runtime.generate(payload)
                raw_output = response.content
                final_finish_reason = response.finish_reason
                self.last_compute.generation_ms += response.generation_ms
                self.last_decision.input_tokens += response.input_tokens
                self.last_decision.total_output_tokens += response.output_tokens

                reasoning, json_output = _extract_reasoning_and_json(
                    raw_output,
                    response.reasoning,
                )
                self.last_reasoning = reasoning
                self.last_json_output = json_output
                self.last_decision.reasoning_tokens += _estimate_tokens(reasoning)
                self.last_decision.json_tokens += _estimate_tokens(json_output)
                self.last_decision.raw_json = json_output

                validation_started = time.perf_counter()
                candidate, validation_error = SyntaxValidator.validate_raw(raw_output)
                self.last_compute.schema_validation_ms += (
                    time.perf_counter() - validation_started
                ) * 1000.0

                if response.finish_reason == "length":
                    last_error = self._truncation_error(raw_output, response.finish_reason)
                    self.last_decision.was_truncated = True
                elif candidate is not None:
                    final_proposal = candidate
                    last_error = None
                else:
                    last_error = validation_error or BenchmarkError(
                        category=ErrorCategory.SCHEMA,
                        error_type=ErrorType.INVALID_JSON,
                        message="Model returned no parseable JSON decision.",
                    )
            except Exception as exc:
                raw_output = f"API Error: {exc}"
                last_error = BenchmarkError(
                    category=ErrorCategory.SCHEMA,
                    error_type=ErrorType.INVALID_JSON,
                    message=f"LLM API call failed: {exc}",
                )
                final_finish_reason = None

            candidate_valid = final_proposal is not None and last_error is None
            if attempt == 1:
                first_pass_valid = candidate_valid
                if not candidate_valid:
                    first_failure = last_error

            if candidate_valid:
                self.last_decision.final_valid = True
                self.last_decision.recovered = attempt > 1 and not first_pass_valid
                self.last_decision.raw_json = self.last_json_output
                break

            if attempt <= self.max_retries:
                error_message = last_error.message if last_error else "invalid output"
                payload["messages"].append({"role": "assistant", "content": raw_output})
                payload["messages"].append(
                    {
                        "role": "user",
                        "content": (
                            f"Schema validation failed: {error_message} "
                            "Return only one complete JSON object."
                        ),
                    }
                )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.last_compute.total_decision_ms = elapsed_ms
        self.last_compute.retry_ms = max(
            0.0,
            elapsed_ms - self.last_compute.generation_ms - self.last_compute.schema_validation_ms,
        )

        self.last_decision.finish_reason = final_finish_reason
        self.last_finish_reason = final_finish_reason
        self.last_was_truncated = self.last_decision.was_truncated
        self.last_decision.generation_attempt = attempt
        self.last_decision.attempt_count = attempt
        self.last_decision.first_pass_valid = first_pass_valid
        self.last_decision.final_valid = final_proposal is not None and last_error is None
        self.last_decision.recovered = self.last_decision.final_valid and not first_pass_valid
        if first_failure is not None:
            self.last_decision.first_failure_type = first_failure.error_type.value
            self.last_decision.first_failure_message = first_failure.message

        if final_proposal is not None and last_error is None:
            return raw_output, final_proposal, None
        return raw_output, None, last_error
