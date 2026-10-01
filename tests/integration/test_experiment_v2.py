import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from tamabench.cli import cli
from tamabench.config import RunConfig
from tamabench.experiment import FINISHED_STATUSES, run_experiment
from tamabench.logging.database import DatabaseStore


def test_experiment_creates_complete_artifact_bundle_and_exact_hash(tmp_path):
    config = RunConfig(output_dir=tmp_path, max_simulated_minutes=73, display="quiet")
    result = run_experiment(config)
    assert len(result.run_ids) == 1
    for name in ("manifest.json", "results.sqlite", "events.jsonl", "traces", "summary.json", "summary.csv", "summary.md"):
        assert (tmp_path / name).exists()
    row = DatabaseStore(result.database_path).get_run(result.run_ids[0])
    assert row["config_hash"] == config.config_hash
    assert row["status"] == "completed"
    assert row["simulated_duration_minutes"] == 73
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["episodes"][0]["run_id"] == row["run_id"]
    assert summary["episodes"][0]["critical_decision_acc"] is None


def test_resume_extends_seeds_and_reruns_changed_configuration(tmp_path):
    config = RunConfig(output_dir=tmp_path, max_simulated_minutes=61, display="quiet")
    first = run_experiment(config)
    second = run_experiment(RunConfig(**dict(config.model_dump(), episodes=2, resume=True)))
    assert second.skipped_run_ids == first.run_ids
    assert len(second.run_ids) == 1
    third = run_experiment(RunConfig(**dict(config.model_dump(), max_simulated_minutes=62, resume=True)))
    assert not third.skipped_run_ids and len(third.run_ids) == 1


@pytest.mark.parametrize("status", sorted(FINISHED_STATUSES))
def test_resume_never_discards_scientific_failures(tmp_path, status):
    config = RunConfig(output_dir=tmp_path, max_simulated_minutes=2, display="quiet")
    first = run_experiment(config)
    db = DatabaseStore(first.database_path)
    with db._get_connection() as conn:
        conn.execute("UPDATE runs SET status=? WHERE run_id=?", (status, first.run_ids[0]))
    resumed = run_experiment(RunConfig(**dict(config.model_dump(), resume=True)))
    assert resumed.skipped_run_ids == first.run_ids
    assert not resumed.run_ids


@pytest.mark.parametrize("status", ["running", "interrupted", "infrastructure_failed"])
def test_resume_retries_unfinished_attempts(tmp_path, status):
    config = RunConfig(output_dir=tmp_path, max_simulated_minutes=2, display="quiet")
    first = run_experiment(config)
    db = DatabaseStore(first.database_path)
    with db._get_connection() as conn:
        conn.execute("UPDATE runs SET status=? WHERE run_id=?", (status, first.run_ids[0]))
    resumed = run_experiment(RunConfig(**dict(config.model_dump(), resume=True)))
    assert len(resumed.run_ids) == 1
    assert not resumed.skipped_run_ids


def test_cli_config_override_export_report_alias_and_replay(tmp_path):
    config_path = tmp_path / "config.json"
    out = tmp_path / "out"
    config_path.write_text(RunConfig(output_dir=out, max_simulated_minutes=2).model_dump_json())
    cli_runner = CliRunner()
    run = cli_runner.invoke(cli, ["run", "--config", str(config_path), "--max-simulated-minutes", "73"])
    assert run.exit_code == 0, run.output
    db = DatabaseStore(str(out / "results.sqlite"))
    row = db.list_runs()[0]
    assert row["simulated_duration_minutes"] == 73
    replay = cli_runner.invoke(cli, ["replay", "--output-dir", str(out), "--run-id", row["run_id"]])
    assert replay.exit_code == 0, replay.output
    for command in ("report", "report-v1", "export"):
        result = cli_runner.invoke(cli, [command, "--output-dir", str(out)])
        assert result.exit_code == 0, result.output


def test_cli_validation_fails_before_creating_outputs(tmp_path):
    out = tmp_path / "out"
    result = CliRunner().invoke(cli, ["run", "--episodes", "0", "--output-dir", str(out)])
    assert result.exit_code != 0
    assert not out.exists()


def test_keyboard_interrupt_preserves_manifest_and_exports(tmp_path, monkeypatch):
    from tamabench.runner.batch_runner import BatchRunner
    monkeypatch.setattr(BatchRunner, "run_episode", lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        run_experiment(RunConfig(output_dir=tmp_path, max_simulated_minutes=2))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert next(iter(manifest["experiments"].values()))["status"] == "interrupted"
    assert (tmp_path / "summary.json").exists()


def test_cli_does_not_echo_rejected_secret_in_configuration(tmp_path):
    sentinel = "sensitive-value-never-echo-this"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"api_key": sentinel}))
    result = CliRunner().invoke(cli, ["doctor", "--config", str(path)])
    assert result.exit_code != 0
    assert sentinel not in result.output
    assert "api_key" in result.output


def test_cold_ollama_lifecycle_is_explicit_and_counted_without_network(tmp_path, monkeypatch):
    import tamabench.experiment as experiment
    from tamabench.agents.rule_agent import RuleAgent

    created = []

    class Runtime:
        warmup_calls = warmup_input_tokens = warmup_output_tokens = 0
        input_tokens = output_tokens = api_calls = 0
        remaining_calls = remaining_tokens = deadline = None
        model_resident = False

        def __init__(self):
            self.cleanup_calls = 0

        def unload(self):
            self.cleanup_calls += 1

    class FakeModel(RuleAgent):
        def __init__(self):
            super().__init__()
            self.name = "RawLLM(fake-model)"
            self.runtime = Runtime()
            self.inference_seed = None

    def factory(config, seed):
        agent = FakeModel()
        created.append(agent)
        return agent

    monkeypatch.setattr(experiment, "create_agent", factory)
    config = RunConfig(agent="raw_llm", backend="ollama", model="fake-model", model_lifecycle="cold",
                       episodes=2, output_dir=tmp_path, max_simulated_minutes=2, display="quiet")
    result = run_experiment(config)
    assert len(result.run_ids) == 2
    assert len(created) == 2
    assert [agent.runtime.cleanup_calls for agent in created] == [1, 1]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["experiments"][config.experiment_id]["lifecycle_cleanup_calls"] == 2


def test_provider_failure_metrics_make_cli_fail_with_preserved_artifacts(tmp_path, monkeypatch):
    from tamabench.runner.batch_runner import BatchRunner
    original = BatchRunner.run_episode

    def failed_result(self, *args, **kwargs):
        metric = original(self, *args, **kwargs)
        metric.status = "infrastructure_failed"
        return metric

    monkeypatch.setattr(BatchRunner, "run_episode", failed_result)
    result = CliRunner().invoke(cli, ["run", "--max-simulated-minutes", "2", "--output-dir", str(tmp_path)])
    assert result.exit_code != 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    entry = next(iter(manifest["experiments"].values()))
    assert entry["status"] == "failed"
    assert len(entry["run_ids"]) == 1
    assert json.loads((tmp_path / "summary.json").read_text())["episodes"]
    assert (tmp_path / "summary.csv").exists()


def test_all_agents_close_and_artifacts_survive_close_failure(tmp_path, monkeypatch):
    import tamabench.experiment as experiment
    from tamabench.agents.random_valid_agent import RandomValidAgent

    closed = []
    created = []

    class BrokenClose(RandomValidAgent):
        def __init__(self, seed):
            super().__init__(seed)
            self.seed = seed

        def close(self):
            closed.append(self.seed)
            raise RuntimeError("Close fixture failed")

    def factory(config, seed):
        agent = BrokenClose(seed)
        created.append(agent)
        return agent

    monkeypatch.setattr(experiment, "create_agent", factory)
    with pytest.raises(RuntimeError):
        run_experiment(RunConfig(agent="random_valid", episodes=2, output_dir=tmp_path,
                                 max_simulated_minutes=2, display="quiet"))
    assert len(created) == 2
    assert set(closed) == {42, 43}
    entry = next(iter(json.loads((tmp_path / "manifest.json").read_text())["experiments"].values()))
    assert entry["status"] == "failed"
    assert entry["cleanup_error_types"]
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary.md").exists()


def test_export_failure_still_updates_manifest(tmp_path, monkeypatch):
    import tamabench.experiment as experiment
    monkeypatch.setattr(experiment, "export_summary", lambda *a, **k: (_ for _ in ()).throw(OSError("export failed")))
    with pytest.raises(experiment.ExperimentInfrastructureError):
        run_experiment(RunConfig(output_dir=tmp_path, max_simulated_minutes=2, display="quiet"))
    entry = next(iter(json.loads((tmp_path / "manifest.json").read_text())["experiments"].values()))
    assert entry["status"] == "failed"
    assert "OSError" in entry["cleanup_error_types"]
