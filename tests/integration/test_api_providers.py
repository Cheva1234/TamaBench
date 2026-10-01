"""Provider setup and complete mocked CLI runs; never contacts a provider."""
import json
import requests
import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from tamabench.cli import cli
from tamabench.config import RunConfig
from tamabench.experiment import doctor, run_experiment, PreflightError
from tamabench.providers import PROVIDERS
from tamabench.runtime.model_runtime import ModelRuntime, ProviderError


@pytest.mark.parametrize("provider", ["openai", "openrouter", "groq"])
def test_named_cloud_preset_round_trip(provider):
    config = RunConfig(provider=provider, model="chosen-model")
    assert config.backend == "openai_compatible" and config.agent == "raw_llm"
    assert config.require_api_key is True
    assert config.api_base == PROVIDERS[provider]["api_base"]
    assert config.temperature is None and config.model_seed(42) is None
    assert RunConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize("values", [
    {"provider": "openai"}, {"provider": "custom", "model": "test"},
    {"provider": "openai", "model": "test", "api_base": "https://other.invalid/v1"},
    {"provider": "openai", "model": "test", "require_api_key": False},
    {"provider": "groq", "model": "test", "backend": "ollama"},
    {"provider": "oops", "model": "test"}, {"provider": []},
    {"provider": "openai", "model": 42}, {"provider": "openai", "model": "test", "api_base": None},
    {"api_base": "https://example.invalid/v1/chat/completions"},
])
def test_invalid_provider_combinations(values):
    with pytest.raises(ValidationError):
        RunConfig(**values)


def test_required_key_preflight_fails_before_artifacts_or_requests(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(requests.Session, "post", lambda *a, **k: pytest.fail("sent inference"))
    output = tmp_path / "absent"
    config = RunConfig(provider="openai", model="chosen", output_dir=output)
    assert not doctor(config)["ready"]
    with pytest.raises(PreflightError, match="OPENAI_API_KEY"):
        run_experiment(config)
    assert not output.exists()
    result = CliRunner().invoke(cli, ["doctor", "--provider", "openai", "--model", "chosen"])
    assert result.exit_code != 0 and "OPENAI_API_KEY" in result.output


def test_custom_unauthenticated_endpoint_and_https_guard(monkeypatch):
    monkeypatch.delenv("TEST_PROVIDER_KEY", raising=False)
    config = RunConfig(provider="custom", model="test", api_base="http://example.invalid/v1", api_key_env="TEST_PROVIDER_KEY")
    assert doctor(config)["ready"]
    monkeypatch.setenv("TEST_PROVIDER_KEY", "private")
    assert not doctor(config)["ready"]
    local = RunConfig(**dict(config.model_dump(), api_base="http://127.0.0.1:1234/v1"))
    assert doctor(local)["ready"]


@pytest.mark.parametrize("provider", ["openai", "openrouter", "groq"])
def test_provider_cli_end_to_end_mocked(tmp_path, monkeypatch, provider):
    secret = "never-save-this-api-key"
    preset = PROVIDERS[provider]
    monkeypatch.setenv(preset["api_key_env"], secret)
    calls = []
    def post(self, url, **kwargs):
        calls.append((url, kwargs))
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"model": "resolved-model-v1", "id": "fixture-id", "system_fingerprint": "fixture-fingerprint",
            "choices": [{"message": {"content": '{"action":"wait","minutes":30}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10}}).encode()
        return response
    monkeypatch.setattr(requests.Session, "post", post)
    args = ["run", "--provider", provider, "--model", "chosen-model", "--max-simulated-minutes", "3",
            "--output-dir", str(tmp_path), "--display", "quiet"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    url, request = calls[0]
    assert url == preset["api_base"] + "/chat/completions"
    assert request["headers"]["Authorization"] == f"Bearer {secret}"
    assert request["allow_redirects"] is False
    assert set(request["json"]) == {"model", "messages", preset["output_token_parameter"]}
    rows = json.loads((tmp_path / "summary.json").read_text())["episodes"]
    assert rows[0]["status"] == "completed" and rows[0]["api_calls"] == 1
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()
    assert secret not in result.output
    resumed = CliRunner().invoke(cli, args + ["--resume"])
    assert resumed.exit_code == 0 and len(calls) == 1


@pytest.mark.parametrize("status,hint", [(400,"parameters"), (401,"API key"), (403,"permissions"), (404,"model ID"), (429,"quota"), (500,"service failed")])
def test_provider_http_errors_are_safe_actionable_and_not_retried(tmp_path, monkeypatch, status, hint):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-marker")
    calls = []
    def post(self, *args, **kwargs):
        calls.append(1)
        response = requests.Response()
        response.status_code = status
        response._content = b'{"error":"secret-marker-do-not-log"}'
        return response
    monkeypatch.setattr(requests.Session, "post", post)
    result = CliRunner().invoke(cli, ["run", "--provider", "openai", "--model", "fixture", "--output-dir", str(tmp_path)])
    assert result.exit_code != 0 and hint in result.output
    assert len(calls) == 1 and "secret-marker" not in result.output
    summary = json.loads((tmp_path / "summary.json").read_text())["episodes"][0]
    assert summary["status"] == "infrastructure_failed" and summary["api_calls"] == 1
    assert summary["schema_decisions"] == 0
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert b"secret-marker" not in path.read_bytes()


def test_output_token_field_clamp_and_explicit_sampling(monkeypatch):
    sent = []
    def post(self, *args, **kwargs):
        sent.append(kwargs["json"])
        response = requests.Response(); response.status_code = 200
        response._content = b'{"choices":[{"message":{"content":"{}"}}],"usage":{"prompt_tokens":3,"completion_tokens":2}}'
        return response
    monkeypatch.setattr(requests.Session, "post", post)
    with_runtime = ModelRuntime("test", backend="openai_compatible", output_token_parameter="max_completion_tokens")
    with_runtime.remaining_tokens = 7
    try:
        with_runtime.generate({"messages": [], "max_tokens": 100, "seed": 3, "temperature": 0.1})
        assert sent[0]["max_completion_tokens"] == 7 and "max_tokens" not in sent[0]
        assert sent[0]["seed"] == 3 and with_runtime.remaining_tokens == 2
    finally:
        with_runtime.close()


def test_redirect_is_not_followed(monkeypatch):
    def post(self, *args, **kwargs):
        assert kwargs["allow_redirects"] is False
        response = requests.Response(); response.status_code = 307
        return response
    monkeypatch.setattr(requests.Session, "post", post)
    runtime = ModelRuntime("test", backend="openai_compatible")
    with pytest.raises(ProviderError, match="Redirects are disabled"):
        runtime.generate({"messages": []})
    runtime.close()
    assert runtime.api_calls == 1 and runtime.generation_ms > 0


@pytest.mark.parametrize("usage", [-1, True, "5", 1.5, None])
def test_invalid_usage_cannot_reduce_budgets(usage):
    with pytest.raises(ProviderError, match="non-negative integers"):
        ModelRuntime._usage_count(usage)


def test_providers_command_and_explicit_overrides(monkeypatch):
    monkeypatch.setattr(requests.Session, "post", lambda *a, **k: pytest.fail("request"))
    result = CliRunner().invoke(cli, ["providers"])
    assert result.exit_code == 0 and "OPENROUTER_API_KEY" in result.output
    config = RunConfig(provider="groq", model="test", temperature=0.1, inference_seed=11)
    assert config.model_seed(42) == 11 and config.temperature == 0.1
    assert RunConfig(provider="groq", model="test", seed_mode="episode").model_seed(42) == 42


def test_switching_preset_does_not_carry_credentials_or_model_options(tmp_path):
    from tamabench.cli import _config
    original = RunConfig(provider='openai', model='old', temperature=0.9, inference_seed=5,
                         reasoning_effort='high', schema_mode='provider_constrained', episodes=3)
    path = tmp_path/'config.json'; path.write_text(original.model_dump_json())
    switched = _config(config_file=str(path), provider='custom', api_base='https://other.invalid/v1', model='new')
    assert switched.api_key_env == 'TAMABENCH_API_KEY'
    assert switched.require_api_key is False
    assert switched.temperature is None and switched.inference_seed is None
    assert switched.reasoning_effort is None and switched.schema_mode == 'raw_json'
    assert switched.episodes == 3
    explicit = _config(config_file=str(path), provider='groq', model='chosen', api_key_env='MY_GROQ_KEY', temperature=0.3)
    assert explicit.api_key_env == 'MY_GROQ_KEY' and explicit.temperature == 0.3


def test_runtime_rejects_plaintext_credential_before_creating_session():
    with pytest.raises(ValueError, match='HTTPS'):
        ModelRuntime('model', api_base='http://public.invalid/v1', api_key='secret', backend='openai_compatible')


def test_omitted_temperature_stays_omitted_in_native_ollama(monkeypatch):
    sent=[]
    def post(self,url,**kwargs):
        sent.append(kwargs['json'])
        response=requests.Response();response.status_code=200
        response._content=b'{"message":{"content":"{}"},"prompt_eval_count":1,"eval_count":1}'
        return response
    monkeypatch.setattr(requests.Session,'post',post)
    runtime=ModelRuntime('model',backend='ollama')
    runtime.model_resident=True
    runtime.generate({'messages':[], 'max_tokens':5})
    runtime.close()
    assert 'temperature' not in sent[0]['options']
