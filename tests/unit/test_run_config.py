import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tamabench.config import RunConfig
from tamabench.experiment import create_agent, doctor


def test_cpu_defaults_and_canonical_scenario():
    config = RunConfig()
    assert config.agent == "rule" and config.backend == "none"
    assert config.max_simulated_minutes == 10080
    assert config.speed == "accelerated"
    assert config.benchmark_version == "2.0.0"
    assert RunConfig(scenario_id="standard_v1").scenario_id == "dynamic_v2"


@pytest.mark.parametrize("values", [
    {"episodes": 0}, {"seed_start": -1}, {"max_simulated_minutes": 0},
    {"max_api_calls": True}, {"max_output_tokens": False}, {"timeout": 0},
    {"temperature": float("nan")}, {"max_wall_seconds": float("inf")},
    {"scenario_id": "typo"}, {"unknown": 1}, {"api_key": "do-not-store-secrets"},
    {"api_base": "https://secret:password@example.com/v1"},
    {"api_base": "https://example.com/v1?key=secret"},
    {"api_base": "ftp://example.com/v1"}, {"keep_alive": "-1"},
    {"agent": "raw_llm"}, {"backend": "ollama"},
    {"agent": "raw_llm", "backend": "openai_compatible", "model_lifecycle": "cold"},
])
def test_invalid_config_rejected_before_execution(values):
    with pytest.raises(ValidationError):
        RunConfig(**values)


def test_hash_ignores_orchestration_but_not_behavior_or_provider():
    original = RunConfig()
    changed = RunConfig(episodes=10, seed_start=999, output_dir="another", resume=True, display="quiet")
    assert original.config_hash == changed.config_hash
    assert RunConfig(max_decisions=1).config_hash != original.config_hash
    assert RunConfig(max_simulated_minutes=100).config_hash != original.config_hash
    assert "source_tree_sha256" in original.behavioral_config()
    assert "spec_hash" in original.behavioral_config()


def test_factory_reads_credential_only_at_construction_and_never_logs_it(monkeypatch):
    sentinel = "secret-value-that-must-not-appear"
    monkeypatch.setenv("TEST_TAMA_SECRET", sentinel)
    config = RunConfig(agent="raw_llm", backend="openai_compatible", api_key_env="TEST_TAMA_SECRET",
                       api_base="https://example.invalid/v1")
    agent = create_agent(config, 47)
    try:
        assert agent.runtime.api_key == sentinel
        assert agent.inference_seed == 47
        assert sentinel not in config.model_dump_json()
        assert sentinel not in json.dumps(config.behavioral_config())
        assert sentinel not in json.dumps(doctor(config))
    finally:
        agent.close()


def test_doctor_is_nonmutating_and_never_calls_network(tmp_path, monkeypatch):
    import requests
    monkeypatch.setattr(requests.Session, "request", lambda *a, **k: pytest.fail("doctor made a request"))
    destination = tmp_path / "not-created"
    info = doctor(RunConfig(output_dir=destination))
    assert info["inference_requests"] == 0
    assert info["model_service_checked"] is False
    assert info["output_parent_writable"] is True
    assert not destination.exists()


def test_json_config_round_trip(tmp_path):
    config = RunConfig(output_dir=tmp_path / "out", max_simulated_minutes=120)
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json())
    assert RunConfig.from_json_file(path) == config
