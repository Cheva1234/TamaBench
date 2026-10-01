"""Validated, secret-free configuration shared by Python, CLI, and Colab."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tamabench.env.scenarios import get_config_for_scenario
from tamabench.spec.spec_loader import EnvironmentSpecLoader
from tamabench.providers import resolve_provider


@lru_cache(maxsize=1)
def implementation_digest() -> str:
    """Hash installed source, including uncommitted local fixes, without Git."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.suffix in {".py", ".yaml"} and path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

    benchmark_version: Literal["2.0.0"] = "2.0.0"
    scenario_id: str = "dynamic_v2"
    scenario_version: Literal[2] = 2
    max_simulated_minutes: int = Field(default=7 * 1440, gt=0)
    agent: Literal["rule", "random_valid", "random_schema", "raw_llm", "harness_v1"] = "rule"
    backend: Literal["none", "ollama", "openai_compatible"] = "none"
    provider: Literal["ollama", "openai", "openrouter", "groq", "custom"] | None = None
    model: str = Field(default="qwen2.5:3b", min_length=1)
    api_base: str = "http://localhost:11434"
    api_key_env: str = "TAMABENCH_API_KEY"
    require_api_key: bool = False
    output_token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    schema_mode: Literal["raw_json", "provider_constrained"] = "raw_json"
    temperature: float | None = Field(default=0.2, ge=0, le=2)
    max_retries: int = Field(default=2, ge=0, le=20)
    max_output_tokens: int = Field(default=4096, gt=0)
    inference_seed: int | None = Field(default=None, ge=0)
    seed_mode: Literal["episode", "omit"] = "episode"
    timeout: float = Field(default=120.0, gt=0)
    reasoning_effort: Literal["none", "low", "medium", "high"] | None = None
    keep_alive: str = "5m"
    model_lifecycle: Literal["warm", "cold"] = "warm"
    speed: Literal["accelerated", "reference"] = "accelerated"
    max_decisions: int = Field(default=5000, gt=0)
    max_api_calls: int = Field(default=1000, gt=0)
    max_total_tokens: int = Field(default=1_000_000, gt=0)
    max_wall_seconds: float = Field(default=3600.0, gt=0)
    max_consecutive_failures: int = Field(default=5, gt=0)
    max_stalled_decisions: int = Field(default=5, gt=0)
    trace_logs: bool = True
    episodes: int = Field(default=1, gt=0)
    seed_start: int = Field(default=42, ge=0)
    display: Literal["live", "compact", "quiet"] = "compact"
    output_dir: str = "tamabench-output"
    db_path: str | None = None
    event_path: str | None = None
    resume: bool = False

    @model_validator(mode="before")
    @classmethod
    def provider_defaults(cls, values):
        return resolve_provider(values)

    @field_validator("model")
    @classmethod
    def nonempty_model(cls, value):
        if not value.strip():
            raise ValueError("model must not be blank")
        return value

    @field_validator("output_dir", "db_path", "event_path", mode="before")
    @classmethod
    def path_text(cls, value):
        return os.fspath(value) if isinstance(value, os.PathLike) else value

    @field_validator("output_dir", "db_path", "event_path")
    @classmethod
    def nonempty_path(cls, value):
        if value is not None and not value.strip():
            raise ValueError("Output paths must not be empty")
        return value

    @field_validator("scenario_id")
    @classmethod
    def canonical_scenario(cls, value):
        return get_config_for_scenario(value).id

    @field_validator("api_base")
    @classmethod
    def safe_api_base(cls, value):
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("api_base must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("api_base must not contain credentials, query parameters, or fragments")
        if parsed.path.rstrip("/").endswith(("/chat/completions", "/responses", "/messages")):
            raise ValueError("api_base must be the API base, not a full inference endpoint (for example https://host/v1)")
        return value.rstrip("/")

    @field_validator("api_key_env")
    @classmethod
    def environment_name(cls, value):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("api_key_env must name an environment variable, not contain a credential")
        return value

    @field_validator("keep_alive")
    @classmethod
    def finite_keep_alive(cls, value):
        if not re.fullmatch(r"[1-9][0-9]*(?:ms|s|m|h)", value):
            raise ValueError("keep_alive must be a finite positive duration, such as 5m")
        return value

    @model_validator(mode="after")
    def consistent_provider(self):
        is_model = self.agent in {"raw_llm", "harness_v1"}
        if is_model and self.backend == "none":
            raise ValueError("Model agents require backend='ollama' or 'openai_compatible'")
        if not is_model and self.backend != "none":
            raise ValueError("CPU baseline agents require backend='none'")
        if self.model_lifecycle == "cold" and self.backend != "ollama":
            raise ValueError("Cold server lifecycle requires the Ollama backend")
        if self.backend == "ollama" and self.output_token_parameter != "max_tokens":
            raise ValueError("output_token_parameter is for compatible APIs; Ollama uses num_predict internally")
        return self

    def model_seed(self, episode_seed: int) -> int | None:
        if self.inference_seed is not None:
            return self.inference_seed
        return episode_seed if self.seed_mode == "episode" else None

    @property
    def agent_name(self) -> str:
        names = {"rule": "RuleAgent", "random_valid": "RandomValidAgent", "random_schema": "RandomSchemaAgent"}
        return names.get(self.agent, f"{'HarnessV1' if self.agent == 'harness_v1' else 'RawLLM'}({self.model})")

    @property
    def seeds(self) -> list[int]:
        return list(range(self.seed_start, self.seed_start + self.episodes))

    def behavioral_config(self) -> dict:
        """Exactly the payload persisted/hashed by BatchRunner, no secrets."""
        data = self.model_dump(exclude={"episodes", "seed_start", "display", "output_dir", "db_path", "event_path", "resume"})
        data.update(mode="accelerated" if self.speed == "accelerated" else "logical",
                    agent_type=self.agent_name, spec_hash=EnvironmentSpecLoader.get_spec_hash(),
                    source_tree_sha256=implementation_digest())
        return data

    @property
    def config_hash(self) -> str:
        content = json.dumps(self.behavioral_config(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode()).hexdigest()

    @property
    def experiment_id(self) -> str:
        return "exp_" + self.config_hash[:16]

    @classmethod
    def from_json_file(cls, filename: str | Path) -> "RunConfig":
        return cls.model_validate_json(Path(filename).read_text(encoding="utf-8"))
