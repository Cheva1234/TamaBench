"""Explicit Ollama and OpenAI-compatible request adapters with measured usage."""
from dataclasses import dataclass
import time
from typing import Any, Optional
from urllib.parse import urlsplit
import requests


class InferenceBudgetExceeded(RuntimeError):
    pass


class ProviderError(RuntimeError):
    """Safe diagnostic: never contains provider bodies, headers, or key values."""


@dataclass
class ModelGenerationResponse:
    content: str
    finish_reason: Optional[str] = None
    reasoning: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    generation_ms: float = 0.0
    usage_available: bool = True
    response_model: str | None = None
    response_id: str | None = None
    system_fingerprint: str | None = None


class ModelRuntime:
    def __init__(self, model_name, api_base="http://localhost:11434/v1", api_key="ollama",
                 keep_alive="5m", timeout=120.0, session=None, backend="ollama",
                 output_token_parameter="max_tokens"):
        if backend not in {"ollama", "openai_compatible"}:
            raise ValueError("backend must be ollama or openai_compatible")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        endpoint = urlsplit(api_base)
        if api_key and endpoint.scheme != "https" and endpoint.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Authenticated remote endpoints require HTTPS")
        self.model_name, self.api_base, self.api_key = model_name, api_base.rstrip("/"), api_key
        self.keep_alive, self.timeout, self.backend = keep_alive, timeout, backend
        if output_token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("Unsupported output token parameter")
        self.output_token_parameter = output_token_parameter
        self.last_error = None
        self.session = session or requests.Session()
        self.model_resident = False  # Last successful Ollama load; not ongoing server telemetry.
        self.warmup_ms = 0.0
        self.generation_ms = 0.0
        self.api_calls = self.warmup_calls = self.cleanup_calls = 0
        self.warmup_input_tokens = self.warmup_output_tokens = 0
        self.input_tokens = self.output_tokens = 0
        self.usage_complete = True
        self.remaining_calls = self.remaining_tokens = self.deadline = None
        self._closed = False

    @property
    def native_base(self):
        return self.api_base.removesuffix("/v1")

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, url, payload, kind):
        if self.remaining_calls is not None and self.remaining_calls <= 0:
            raise InferenceBudgetExceeded("max_api_calls")
        timeout = self.timeout
        if self.deadline is not None:
            timeout = min(timeout, self.deadline - time.perf_counter())
            if timeout <= 0:
                raise InferenceBudgetExceeded("max_wall_seconds")
        self.api_calls += 1
        if self.remaining_calls is not None:
            self.remaining_calls -= 1
        if kind == "warmup":
            self.warmup_calls += 1
        elif kind == "cleanup":
            self.cleanup_calls += 1
        started = time.perf_counter()
        try:
            response = self.session.post(url, headers=self._headers(), json=payload, timeout=timeout,
                                         allow_redirects=False)
            status = getattr(response, "status_code", 200)
            if 300 <= status < 400:
                self.last_error = "Provider redirected the request; verify the API base URL. Redirects are disabled to protect credentials."
                raise ProviderError(self.last_error)
            response.raise_for_status()
            elapsed = (time.perf_counter() - started) * 1000
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Expected a response object")
            return data, elapsed
        except requests.RequestException as exc:
            if isinstance(exc, requests.Timeout) and self.deadline is not None and time.perf_counter() >= self.deadline:
                raise InferenceBudgetExceeded("max_wall_seconds") from exc
            # Do not include potentially credential-bearing provider response bodies.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            hints = {400: "Check the model's supported parameters, token-limit field, and schema mode.",
                     401: "Check the API key environment variable and provider account.",
                     403: "Check the account's model access and API permissions.",
                     404: "Check the API base URL and exact model ID.",
                     408: "Provider request timed out; inspect saved results before retrying.",
                     429: "Rate limit or quota exceeded; inspect provider limits before using --resume."}
            if status is not None:
                hint = hints.get(status, "Provider service failed; inspect saved results before retrying." if status >= 500 else "Check provider configuration.")
                message = f"Provider HTTP {status}. {hint} No automatic HTTP retry was made."
            elif isinstance(exc, requests.Timeout):
                message = "Provider request timed out; adjust --timeout or check service availability. No automatic HTTP retry was made."
            else:
                message = "Could not connect to provider; check the API base URL, network, and TLS configuration."
            self.last_error = message
            raise ProviderError(message) from None
        except (ValueError, TypeError) as exc:
            self.last_error = "Provider response was not valid JSON"
            raise ProviderError(self.last_error) from None
        finally:
            if kind == "generation":
                self.generation_ms += (time.perf_counter() - started) * 1000

    def warmup(self):
        if self.model_resident or self.backend != "ollama":
            return 0.0
        data, elapsed = self._post(self.native_base + "/api/generate",
            {"model": self.model_name, "prompt": "", "stream": False, "keep_alive": self.keep_alive}, "warmup")
        self.warmup_input_tokens += int(data.get("prompt_eval_count", 0))
        self.warmup_output_tokens += int(data.get("eval_count", 0))
        self.model_resident = True
        self.warmup_ms += elapsed
        return elapsed

    def generate(self, payload: dict[str, Any]):
        if self.backend == "ollama" and not self.model_resident:
            self.warmup()
        request = dict(payload)
        request.setdefault("model", self.model_name)
        if self.remaining_tokens is not None:
            if self.remaining_tokens <= 0:
                raise InferenceBudgetExceeded("max_total_tokens")
            request["max_tokens"] = min(request.get("max_tokens", self.remaining_tokens), self.remaining_tokens)
        if self.backend == "ollama":
            options = {"num_predict": request.get("max_tokens", 4096)}
            if "temperature" in request:
                options["temperature"] = request["temperature"]
            if "seed" in request:
                options["seed"] = request["seed"]
            native = {"model": request["model"], "messages": request["messages"], "stream": False,
                      "options": options, "keep_alive": self.keep_alive}
            if "response_format" in request:
                native["format"] = "json"
            if "reasoning_effort" in request:
                native["think"] = False if request["reasoning_effort"] == "none" else request["reasoning_effort"]
            data, elapsed = self._post(self.native_base + "/api/chat", native, "generation")
            message = data.get("message", {})
            content, reasoning = message.get("content", "") or "", message.get("thinking", "") or ""
            finish = data.get("done_reason")
            usage_available = "prompt_eval_count" in data and "eval_count" in data
            input_tokens = self._usage_count(data.get("prompt_eval_count", 0))
            output_tokens = self._usage_count(data.get("eval_count", 0))
        else:
            request.pop("keep_alive", None)
            if self.output_token_parameter != "max_tokens" and "max_tokens" in request:
                request[self.output_token_parameter] = request.pop("max_tokens")
            data, elapsed = self._post(self.api_base + "/chat/completions", request, "generation")
            if not data.get("choices"):
                raise ProviderError("Provider response did not contain choices")
            choice = data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "") or ""
            reasoning = message.get("reasoning", "") or message.get("reasoning_content", "") or ""
            finish = choice.get("finish_reason")
            usage = data.get("usage") or {}
            usage_available = "prompt_tokens" in usage and "completion_tokens" in usage
            input_tokens = self._usage_count(usage.get("prompt_tokens", 0))
            output_tokens = self._usage_count(usage.get("completion_tokens", 0))
        if not isinstance(content, str) or not isinstance(reasoning, str):
            raise ProviderError("Provider must return text content in a Chat Completions message")
        self.usage_complete = self.usage_complete and usage_available
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if self.remaining_tokens is not None:
            self.remaining_tokens -= input_tokens + output_tokens
        return ModelGenerationResponse(content, finish, reasoning, input_tokens, output_tokens, elapsed, usage_available,
            **{name: data.get(key) if isinstance(data.get(key), str) else None for name, key in
               [("response_model", "model"), ("response_id", "id"), ("system_fingerprint", "system_fingerprint")]})

    @staticmethod
    def _usage_count(value):
        if type(value) is not int or value < 0:
            raise ProviderError("Provider token usage must contain non-negative integers")
        return value

    def unload(self):
        if self.backend != "ollama":
            raise ValueError("Cold server lifecycle is supported only by the Ollama backend")
        self._post(self.native_base + "/api/generate",
                   {"model": self.model_name, "prompt": "", "stream": False, "keep_alive": 0}, "cleanup")
        self.model_resident = False

    def close(self):
        if not self._closed:
            self.session.close()
            self._closed = True
        self.model_resident = False
