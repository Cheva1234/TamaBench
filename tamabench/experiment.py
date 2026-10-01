"""Single experiment entry point and output bundle for CLI and notebooks."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from urllib.parse import urlsplit
from typing import Callable

from tamabench.config import RunConfig
from tamabench.env.time_engine import BenchmarkMode
from tamabench.logging.database import DatabaseStore
from tamabench.metrics.calculator import BenchmarkMetricsCalculator, EpisodeMetrics
from tamabench.runner.batch_runner import BatchRunner

# Finished experimental outcomes must not be selectively rerun: doing so would
# preferentially discard failures and inflate measured survival.
FINISHED_STATUSES = frozenset({"completed", "died", "invalid_action_abort", "stalled", "budget_exhausted"})


def create_agent(config: RunConfig, seed: int):
    """One construction path; does not load a model or send network requests."""
    from tamabench.agents.rule_agent import RuleAgent
    from tamabench.agents.random_valid_agent import RandomValidAgent
    from tamabench.agents.random_schema_agent import RandomSchemaAgent
    if config.agent == "rule":
        return RuleAgent()
    if config.agent == "random_valid":
        return RandomValidAgent(seed=seed)
    if config.agent == "random_schema":
        return RandomSchemaAgent(seed=seed)
    from tamabench.agents.raw_llm_agent import RawLLMAgent
    from tamabench.agents.harness_v1_agent import HarnessV1Agent
    key = os.environ.get(config.api_key_env, "")
    model = RawLLMAgent(
        model_name=config.model, api_base=config.api_base, api_key=key,
        backend=config.backend, schema_mode=config.schema_mode,
        temperature=config.temperature, max_retries=config.max_retries,
        max_output_tokens=config.max_output_tokens, timeout=config.timeout,
        reasoning_effort=config.reasoning_effort, keep_alive=config.keep_alive,
        inference_seed=config.model_seed(seed),
        output_token_parameter=config.output_token_parameter,
    )
    return HarnessV1Agent(model_agent=model) if config.agent == "harness_v1" else model


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, content):
    """Atomic replacement keeps manifests/export readable after interruption."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def output_paths(config: RunConfig) -> dict[str, Path]:
    root = Path(config.output_dir).expanduser()
    return {"root": root, "manifest": root / "manifest.json",
            "database": Path(config.db_path).expanduser() if config.db_path else root / "results.sqlite",
            "events": Path(config.event_path).expanduser() if config.event_path else root / "events.jsonl",
            "traces": root / "traces"}


def collect_summary(db_path: str | Path, experiment_id: str | None = None) -> list[dict]:
    db = DatabaseStore(str(db_path))
    calculator = BenchmarkMetricsCalculator(str(db_path))
    rows = []
    for run in db.list_runs(experiment_id):
        metric = asdict(calculator.calculate_run_metrics(run["run_id"]))
        metric.update(seed=run["seed"], agent=run["agent_type"], config_hash=run["config_hash"],
                      experiment_id=run["experiment_id"], benchmark_version=run["benchmark_version"],
                      max_simulated_minutes=run["max_simulated_minutes"])
        rows.append(metric)
    return rows


def export_summary(db_path: str | Path, output_dir: str | Path, experiment_id: str | None = None) -> list[dict]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = collect_summary(db_path, experiment_id)
    _write_json(root / "summary.json", {"benchmark_version": "2.0.0", "episodes": rows})
    columns = list(rows[0]) if rows else ["run_id", "seed", "status", "config_hash"]
    temporary = root / "summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / "summary.csv")
    lines = ["# TamaBench 2.0 results", "", "One row per attempt. Interrupted/infrastructure failures are not scientific successes.", "",
             "| Run | Agent | Seed | Status | Days | Avg health | Min health | API calls |", "|---|---|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        agent = str(row["agent"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f'| {row["run_id"]} | {agent} | {row["seed"]} | {row["status"]} | {row["simulated_days"]:.3f} | {row["avg_health"]:.1f} | {row["min_health"]:.1f} | {row["api_calls"]} |')
    lines += ["", "Health/happiness averages use every simulated minute. Unimplemented metrics are null/blank, never invented.", ""]
    temporary = root / "summary.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(root / "summary.md")
    return rows


class ExperimentInfrastructureError(RuntimeError):
    """A provider or cleanup failure makes the experiment unsuccessful."""


class PreflightError(ValueError):
    """Safe local setup errors with no secrets or provider response bodies."""


@dataclass
class ExperimentResult:
    experiment_id: str
    config_hash: str
    output_dir: str
    database_path: str
    run_ids: list[str]
    skipped_run_ids: list[str]
    metrics: list[EpisodeMetrics]


def run_experiment(config: RunConfig, on_episode: Callable | None = None) -> ExperimentResult:
    """Run configured seed pairs, persist provenance, export even after failure."""
    config = RunConfig.model_validate(config.model_dump())
    checks = doctor(config)
    if checks["errors"]:
        raise PreflightError("; ".join(checks["errors"]))
    paths = output_paths(config)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["traces"].mkdir(parents=True, exist_ok=True)
    manifest = {"benchmark_version": "2.0.0", "created_at": _now(), "experiments": {}}
    if paths["manifest"].exists():
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if manifest.get("benchmark_version") != "2.0.0" or not isinstance(manifest.get("experiments"), dict):
            raise ValueError("Output directory contains an incompatible manifest; choose a new output directory")
    versions = {}
    for name in ("pydantic", "click", "rich", "requests"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    entry = manifest["experiments"].setdefault(config.experiment_id, {})
    entry.update(config_hash=config.config_hash, configuration=config.model_dump(),
                 behavioral_configuration=config.behavioral_config(), requested_seeds=config.seeds,
                 started_at=_now(), status="running", python=sys.version.split()[0],
                 platform=platform.platform(), dependency_versions=versions,
                 artifacts={key: str(value) for key, value in paths.items() if key != "root"})
    entry.setdefault("lifecycle_cleanup_calls", 0)
    _write_json(paths["manifest"], manifest)
    database = DatabaseStore(str(paths["database"]))
    completed = {}
    if config.resume:
        for row in database.list_runs(config.experiment_id):
            if row["config_hash"] == config.config_hash and row["status"] in FINISHED_STATUSES:
                completed[row["seed"]] = row["run_id"]
    run_ids, skipped, metrics = [], [], []
    shared_agent = None
    created_agents = []
    try:
        with BatchRunner(db_path=str(paths["database"]), event_path=str(paths["events"]),
                         mode=BenchmarkMode.ACCELERATED if config.speed == "accelerated" else BenchmarkMode.LOGICAL,
                         max_stalled_decisions=config.max_stalled_decisions,
                         log_dir=str(paths["traces"]), trace_logs=config.trace_logs) as runner:
            for index, seed in enumerate(config.seeds, 1):
                if seed in completed:
                    skipped.append(completed[seed])
                    continue
                reusable = config.model_lifecycle == "warm" and config.agent not in {"random_schema", "random_valid"}
                agent = shared_agent if shared_agent is not None else create_agent(config, seed)
                if agent not in created_agents:
                    created_agents.append(agent)
                if reusable:
                    shared_agent = agent
                model_agent = getattr(agent, "model_agent", agent)
                if hasattr(model_agent, "inference_seed"):
                    model_agent.inference_seed = config.model_seed(seed)
                if config.model_lifecycle == "cold":
                    before = agent.runtime.cleanup_calls
                    try:
                        agent.runtime.unload()
                    finally:
                        entry["lifecycle_cleanup_calls"] += agent.runtime.cleanup_calls - before
                metric = runner.run_episode(agent=agent, seed=seed,
                    max_simulated_minutes=config.max_simulated_minutes, scenario_id=config.scenario_id,
                    scenario_version=config.scenario_version, live_monitor=config.display == "live",
                    max_consecutive_failures=config.max_consecutive_failures,
                    run_config=config.behavioral_config(), experiment_id=config.experiment_id)
                metrics.append(metric)
                run_ids.append(metric.run_id)
                if on_episode:
                    on_episode(metric, seed, index, config.episodes)
                if metric.status == "infrastructure_failed":
                    hint = getattr(getattr(model_agent, "runtime", None), "last_error", None)
                    raise ExperimentInfrastructureError(hint or "Episode infrastructure failed; recorded results were preserved")
        entry["status"] = "finished"
    except BaseException as error:
        entry["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        entry["error_type"] = type(error).__name__  # no credential-bearing exception text
        raise
    finally:
        # Cleanup cannot prevent the remaining closes or artifact finalization.
        # Preserve an original inference/interruption error instead of hiding it
        # behind a later close/export error.
        original_error = sys.exc_info()[1]
        cleanup_errors = []
        for agent in created_agents:
            try:
                agent.close()
            except BaseException as error:
                cleanup_errors.append(error)
        entry.update(ended_at=_now(), run_ids=run_ids, skipped_run_ids=skipped)
        try:
            entry["runs"] = [{"run_id": row["run_id"], "seed": row["seed"], "status": row["status"]}
                             for row in database.list_runs(config.experiment_id)
                             if row["config_hash"] == config.config_hash]
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            export_summary(paths["database"], paths["root"])
        except BaseException as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            entry["status"] = "interrupted" if isinstance(original_error, KeyboardInterrupt) else "failed"
            entry["cleanup_error_types"] = [type(error).__name__ for error in cleanup_errors]
        try:
            _write_json(paths["manifest"], manifest)
        except BaseException as error:
            cleanup_errors.append(error)
        if cleanup_errors and original_error is None:
            if isinstance(cleanup_errors[0], KeyboardInterrupt):
                raise cleanup_errors[0]
            raise ExperimentInfrastructureError("Experiment cleanup or artifact export failed") from None
    return ExperimentResult(config.experiment_id, config.config_hash, str(paths["root"]),
                            str(paths["database"]), run_ids, skipped, metrics)


def doctor(config: RunConfig) -> dict:
    """Local preflight only: no inference, warmup, API request, or model pull."""
    root = Path(config.output_dir).expanduser()
    parent = root
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    key_present = bool(os.environ.get(config.api_key_env, "").strip())
    errors, warnings = [], []
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        errors.append("Output parent is not a writable directory")
    if config.require_api_key and not key_present:
        errors.append(f"Set {config.api_key_env} in your environment or Colab Secrets before running; never put the key in a config file")
    endpoint = urlsplit(config.api_base)
    if config.backend != "none" and key_present and endpoint.scheme != "https" and endpoint.hostname not in {"localhost", "127.0.0.1", "::1"}:
        errors.append("Authenticated remote endpoints require HTTPS; use a secure API base URL")
    if config.backend == "openai_compatible":
        if not key_present:
            warnings.append("No API credential is set; this works only with unauthenticated compatible endpoints")
        if config.temperature is None:
            warnings.append("Temperature omitted: provider/model default applies and is recorded as null")
        if config.model_seed(config.seed_start) is None:
            warnings.append("No sampling seed is sent; episode seeds still control the simulator")
        warnings.append("Supports Chat Completions text responses with token usage; native Responses, Anthropic Messages, and Gemini APIs need separate adapters")
    return {"benchmark_version": "2.0.0", "python": sys.version.split()[0],
            "configuration_valid": True, "backend": config.backend, "agent": config.agent,
            "provider": config.provider, "model": config.model if config.backend != "none" else None,
            "api_base": config.api_base if config.backend != "none" else None,
            "output_token_parameter": config.output_token_parameter if config.backend == "openai_compatible" else None,
            "ready": not errors, "errors": errors, "warnings": warnings,
            "output_parent_writable": parent.is_dir() and os.access(parent, os.W_OK),
            "credentials_present": key_present if config.backend == "openai_compatible" else None,
            "credential_environment_variable": config.api_key_env if config.backend == "openai_compatible" else None,
            "model_service_checked": False, "inference_requests": 0,
            "note": "Local preflight only; provider connectivity and model availability were not tested."}
