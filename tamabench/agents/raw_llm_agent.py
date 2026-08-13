"""Raw LLM Agent Adapter for Small Models in TamaBench V1.

Connects to any OpenAI-compatible API (Ollama, vLLM, OpenAI, LiteLLM)
and supports both `raw_json` (unconstrained) and `provider_constrained` schema modes.
"""

import json
import time
from typing import Tuple, Optional, Any
import requests
from tamabench.agents.base import BaseAgent
from tamabench.context.builder import ContextBuilder
from tamabench.schemas.observation import Observation
from tamabench.schemas.actions import ActionProposal
from tamabench.schemas.errors import BenchmarkError
from tamabench.validation.syntax_validator import SyntaxValidator
from tamabench.env.time_engine import ComputeClock


class RawLLMAgent(BaseAgent):
    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        api_base: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        schema_mode: str = "raw_json",  # raw_json or provider_constrained
        temperature: float = 0.2,
        max_retries: int = 2,
    ):
        super().__init__(name=f"RawLLM({model_name})")
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.schema_mode = schema_mode
        self.temperature = temperature
        self.max_retries = max_retries
        self.recent_events: list[str] = []
        self.last_compute: ComputeClock = ComputeClock()

    def record_event(self, event_description: str):
        self.recent_events.append(event_description)

    def select_action(
        self, observation: Observation
    ) -> Tuple[str, Optional[ActionProposal], Optional[BenchmarkError]]:
        prompt = ContextBuilder.build_prompt(
            observation=observation,
            recent_event_descriptions=self.recent_events,
            schema_mode=self.schema_mode,
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a precise autonomous agent for TamaBench. Respond ONLY with a single valid JSON object containing your action proposal. Do not output conversational preamble."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": 2048,
        }

        if self.schema_mode == "provider_constrained":
            payload["response_format"] = {"type": "json_object"}

        self.last_compute.reset()
        start_time = time.time()

        raw_output = ""
        last_error: Optional[BenchmarkError] = None
        proposal: Optional[ActionProposal] = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120.0,
                )

                if resp.status_code == 200:
                    resp_json = resp.json()
                    choices = resp_json.get("choices", [])
                    if choices:
                        msg_obj = choices[0].get("message", {})
                        raw_output = msg_obj.get("content", "")
                        reasoning_content = msg_obj.get("reasoning", "")
                        
                        # Extract <think>...</think> tags if embedded in content
                        if "<think>" in raw_output and "</think>" in raw_output:
                            start_t = raw_output.find("<think>") + 7
                            end_t = raw_output.find("</think>")
                            reasoning_content = raw_output[start_t:end_t].strip()
                        
                        self.last_reasoning = reasoning_content
                        if not raw_output or not raw_output.strip():
                            raw_output = reasoning_content
                        
                        # Calculate latency metrics
                        gen_duration = (time.time() - start_time) * 1000.0
                        self.last_compute.generation_ms = gen_duration
                        self.last_compute.total_decision_ms = gen_duration

                        usage = resp_json.get("usage", {})
                        input_tok = usage.get("prompt_tokens", len(prompt) // 4)
                        output_tok = usage.get("completion_tokens", len(raw_output) // 4)

                        # Stage 1 Syntax & Schema Validation
                        v_start = time.time()
                        proposal, last_error = SyntaxValidator.validate_raw(raw_output)
                        self.last_compute.schema_validation_ms = (time.time() - v_start) * 1000.0

                        if proposal is not None:
                            return raw_output, proposal, None

            except Exception as e:
                raw_output = f"API Error: {str(e)}"
                last_error = BenchmarkError(
                    category="SCHEMA",
                    error_type="INVALID_JSON",
                    message=f"LLM API Call failed: {str(e)}",
                )

            # Retry attempt prompt append with strict brevity instruction
            payload["messages"].append({"role": "assistant", "content": raw_output})
            payload["messages"].append(
                {"role": "user", "content": f"Schema validation failed: {last_error.message if last_error else 'error'}. Output ONLY the raw valid JSON object without conversational text."}
            )

        return raw_output, None, last_error
