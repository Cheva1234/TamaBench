"""Persistent model runtime for OpenAI-compatible inference servers.

The runtime owns the HTTP session and model residency state. Agents only ask
it to warm the model or generate a response, which keeps lifecycle concerns
out of benchmark decision logic.
"""

from dataclasses import dataclass
import time
from typing import Any, Optional

import requests


@dataclass
class ModelGenerationResponse:
    content: str
    finish_reason: Optional[str] = None
    reasoning: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    generation_ms: float = 0.0


class ModelRuntime:
    """Keeps a model warm across decisions and benchmark episodes."""

    def __init__(
        self,
        model_name: str,
        api_base: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        keep_alive: str | int = -1,
        timeout: float = 120.0,
        session: Optional[requests.Session] = None,
    ):
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.session = session or requests.Session()
        self.model_resident = False
        self.warmup_ms = 0.0
        self.api_calls = 0

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def warmup(self) -> float:
        """Load the model once and request persistent residency from Ollama."""
        if self.model_resident:
            return 0.0

        started = time.perf_counter()
        response = self.session.post(
            f"{self.api_base}/chat/completions",
            headers=self._headers(),
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": "Respond with 1."}],
                "temperature": 0,
                "max_tokens": 1,
                "keep_alive": self.keep_alive,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        self.model_resident = True
        self.warmup_ms = (time.perf_counter() - started) * 1000.0
        return self.warmup_ms

    def generate(self, payload: dict[str, Any]) -> ModelGenerationResponse:
        """Generate one response while preserving model residency."""
        if not self.model_resident:
            self.warmup()

        request_payload = dict(payload)
        request_payload.setdefault("model", self.model_name)
        request_payload["keep_alive"] = self.keep_alive

        started = time.perf_counter()
        self.api_calls += 1
        response = self.session.post(
            f"{self.api_base}/chat/completions",
            headers=self._headers(),
            json=request_payload,
            timeout=self.timeout,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()

        response_json = response.json()
        choices = response_json.get("choices", [])
        if not choices:
            raise RuntimeError("Model response did not contain any choices")

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""
        usage = response_json.get("usage", {}) or {}

        return ModelGenerationResponse(
            content=content,
            finish_reason=choice.get("finish_reason"),
            reasoning=message.get("reasoning", "") or message.get("reasoning_content", "") or "",
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", len(content) // 4) or 0),
            generation_ms=elapsed_ms,
        )

    def close(self) -> None:
        """Release the client session and mark residency as ended."""
        self.model_resident = False
        self.session.close()
