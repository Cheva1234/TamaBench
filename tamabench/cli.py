"""Thin CLI over the same validated experiment API used by Python and Colab."""

import json
from pathlib import Path

import click
from pydantic import ValidationError

from tamabench.config import RunConfig
from tamabench.experiment import doctor as preflight, export_summary, run_experiment, PreflightError, ExperimentInfrastructureError
from tamabench.providers import PROVIDERS
from tamabench.logging.replay import ReplayEngine
from tamabench.metrics.reporter import BenchmarkReporter


def config_options(function):
    """Apply shared options without duplicating configuration defaults."""
    options = [
        (("--config", "config_file"), dict(type=click.Path(exists=True, dir_okay=False), help="RunConfig JSON file; explicit flags override it")),
        (("--agent",), dict(type=click.Choice(["rule", "random_valid", "random_schema", "raw_llm", "harness_v1"]), help="Default: CPU rule baseline")),
        (("--backend",), dict(type=click.Choice(["none", "ollama", "openai_compatible"]))),
        (("--provider",), dict(type=click.Choice(list(PROVIDERS)), help="Endpoint and credential-env preset; API presets require --model")),
        (("--model",), dict(type=str)),
        (("--api-base",), dict(type=str)),
        (("--api-key-env",), dict(type=str, help="Environment variable name; never put a credential in CLI arguments")),
        (("--require-api-key/--no-require-api-key",), dict(help="Fail locally if the credential variable is empty; required for named cloud providers")),
        (("--output-token-parameter",), dict(type=click.Choice(["max_tokens", "max_completion_tokens"]), help="Explicit Chat Completions token-limit field override")),
        (("--scenario-id",), dict(type=str, help="dynamic_v2; standard_v1 is a compatibility alias")),
        (("--max-simulated-minutes", "--horizon-minutes"), dict(type=click.IntRange(min=1), help="Default: 10080 (seven days)")),
        (("--episodes",), dict(type=click.IntRange(min=1))),
        (("--seed-start",), dict(type=click.IntRange(min=0))),
        (("--schema-mode",), dict(type=click.Choice(["raw_json", "provider_constrained"]))),
        (("--temperature",), dict(type=click.FloatRange(min=0, max=2))),
        (("--omit-temperature",), dict(is_flag=True, help="Use provider/model default temperature, including with a saved config")),
        (("--inference-seed",), dict(type=click.IntRange(min=0))),
        (("--seed-mode",), dict(type=click.Choice(["episode", "omit"]), help="Cloud presets omit sampling seeds unless explicitly set")),
        (("--max-retries",), dict(type=click.IntRange(min=0, max=20))),
        (("--max-output-tokens",), dict(type=click.IntRange(min=1))),
        (("--timeout",), dict(type=click.FloatRange(min=0, min_open=True))),
        (("--reasoning-effort",), dict(type=click.Choice(["none", "low", "medium", "high"]))),
        (("--keep-alive",), dict(type=str, help="Finite Ollama residency TTL, default 5m")),
        (("--model-lifecycle",), dict(type=click.Choice(["warm", "cold"]))),
        (("--speed",), dict(type=click.Choice(["accelerated", "reference"]))),
        (("--display",), dict(type=click.Choice(["live", "compact", "quiet"]))),
        (("--max-decisions",), dict(type=click.IntRange(min=1))),
        (("--max-api-calls",), dict(type=click.IntRange(min=1))),
        (("--max-total-tokens",), dict(type=click.IntRange(min=1))),
        (("--max-wall-seconds",), dict(type=click.FloatRange(min=0, min_open=True))),
        (("--max-consecutive-failures",), dict(type=click.IntRange(min=1))),
        (("--max-stalled-decisions",), dict(type=click.IntRange(min=1))),
        (("--output-dir",), dict(type=click.Path(file_okay=False), help="One artifact directory, default tamabench-output")),
        (("--db-path",), dict(type=click.Path(dir_okay=False), help="Legacy database path override")),
        (("--event-path",), dict(type=click.Path(dir_okay=False), help="Legacy event path override")),
        (("--trace-logs/--no-trace-logs",), dict(help="Keep human-readable/replay traces (default on)")),
        (("--resume/--no-resume",), dict(help="Skip finished exact config+seed pairs; retry infrastructure interruptions")),
    ]
    for names, kwargs in reversed(options):
        function = click.option(*names, default=None, **kwargs)(function)
    return function


def _config(config_file=None, **overrides):
    values = {}
    try:
        if config_file:
            values = json.loads(Path(config_file).read_text(encoding="utf-8"))
            if not isinstance(values, dict):
                raise ValueError("Configuration file must contain a JSON object")
        if overrides.get("provider") is not None and overrides["provider"] != values.get("provider"):
            # Never carry a previous provider's endpoint or credential variable
            # into a different destination. Explicit flags below still win.
            for key in ("agent", "backend", "model", "api_base", "api_key_env", "require_api_key",
                        "output_token_parameter", "temperature", "seed_mode", "inference_seed",
                        "reasoning_effort", "schema_mode", "model_lifecycle"):
                values.pop(key, None)
        omit_temperature = overrides.pop("omit_temperature", False)
        if omit_temperature and overrides.get("temperature") is not None:
            raise ValueError("Choose --temperature or --omit-temperature, not both")
        values.update({key: value for key, value in overrides.items() if value is not None})
        if omit_temperature:
            values["temperature"] = None
        if "provider" not in values and "backend" not in values and values.get("agent") in {"raw_llm", "harness_v1"}:
            values["backend"] = "ollama"
        # Keep legacy tests/scripts' artifacts next to their requested database.
        if "output_dir" not in values and (values.get("db_path") or values.get("event_path")):
            path = Path(values.get("db_path") or values["event_path"])
            values["output_dir"] = str(path.parent / (path.stem + "-output"))
        return RunConfig.model_validate(values)
    except ValidationError as error:
        # Pydantic's default repr includes rejected input values, which might be
        # a mistakenly supplied secret. Show fields/reasons only.
        reasons = [f"{'.'.join(map(str, item['loc'])) or 'configuration'}: {item['msg']}"
                   for item in error.errors(include_input=False, include_context=False)]
        raise click.ClickException("Invalid configuration: " + "; ".join(reasons)) from None
    except (ValueError, OSError) as error:
        raise click.ClickException(f"Invalid configuration: {error}") from error


@click.group()
@click.version_option("2.0.0")
def cli():
    """TamaBench 2.0: reproducible, accelerated autonomous-agent experiments."""


@cli.command()
def providers():
    """List provider presets without contacting any service."""
    click.echo(json.dumps(PROVIDERS, indent=2, sort_keys=True))
    click.echo("API presets require an explicit model ID. Custom supports Chat Completions, not every native provider API.")


@cli.command()
@config_options
def run(**kwargs):
    """Run CPU baselines or an explicitly configured model backend."""
    config = _config(**kwargs)
    reporter = BenchmarkReporter()

    def progress(metric, seed, index, episodes):
        if config.display == "compact":
            click.echo(f"[{index:03d}/{episodes:03d}] | Seed #{seed} | Days {metric.simulated_days:.3f} | Health {metric.avg_health:.1f} | Status: {metric.status}")
        elif config.display == "live":
            reporter.print_summary(metric, model_name=config.agent_name, episodes=1)

    try:
        result = run_experiment(config, on_episode=progress)
    except (PreflightError, ExperimentInfrastructureError) as error:
        raise click.ClickException(str(error)) from None
    except KeyboardInterrupt:
        raise click.ClickException("Interrupted; partial results were preserved. Use --resume to retry unfinished episodes.") from None
    except Exception as error:
        # Provider errors are deliberately sanitized by the runtime; do not dump
        # a requests exception chain that may contain headers or provider bodies.
        raise click.ClickException(f"Experiment failed ({type(error).__name__}); inspect the output manifest and episode status.") from None
    if config.display != "quiet":
        click.echo(f"Artifacts: {result.output_dir}")
        if result.skipped_run_ids:
            click.echo(f"Resumed: skipped {len(result.skipped_run_ids)} finished config+seed pairs")


@cli.command()
@config_options
def doctor(**kwargs):
    """Validate local setup without model inference or provider requests."""
    result = preflight(_config(**kwargs))
    click.echo(json.dumps(result, indent=2, sort_keys=True))
    if not result["ready"]:
        raise click.ClickException("; ".join(result["errors"]))


def _database(db_path, output_dir):
    path = Path(db_path) if db_path else Path(output_dir) / "results.sqlite"
    if not path.is_file():
        raise click.ClickException(f"Database not found: {path}")
    return str(path)


@cli.command()
@click.option("--output-dir", default="tamabench-output", type=click.Path(file_okay=False))
@click.option("--db-path", default=None, type=click.Path(dir_okay=False))
@click.option("--experiment-id", default=None)
@click.option("--run-id", "run_ids", multiple=True)
def report(output_dir, db_path, experiment_id, run_ids):
    """Render measured per-run results; retain separate configuration groups."""
    from tamabench.metrics.report_v1 import ReportV1Generator
    generator = ReportV1Generator(db_path=_database(db_path, output_dir))
    generator.generate_report(run_ids=list(run_ids) or None, experiment_id=experiment_id)


# Compatibility name; both commands share the same current report implementation.
cli.add_command(report, name="report-v1")


@cli.command(name="export")
@click.option("--output-dir", default="tamabench-output", type=click.Path(file_okay=False))
@click.option("--db-path", default=None, type=click.Path(dir_okay=False))
@click.option("--experiment-id", default=None)
def export_cmd(output_dir, db_path, experiment_id):
    """Export summary.json, summary.csv, and summary.md from saved results."""
    rows = export_summary(_database(db_path, output_dir), output_dir, experiment_id)
    click.echo(f"Exported {len(rows)} episodes to {output_dir}")


@cli.command()
@click.option("--run-id", required=True)
@click.option("--output-dir", default="tamabench-output", type=click.Path(file_okay=False))
@click.option("--db-path", default=None, type=click.Path(dir_okay=False))
def replay(run_id, output_dir, db_path):
    """Replay recorded decisions and verify every deterministic state hash."""
    engine = ReplayEngine(db_path=_database(db_path, output_dir))
    success, mismatches = engine.replay_run(run_id)
    if success:
        click.echo(f"Replay successful for '{run_id}': all state hashes matched")
    else:
        for message in mismatches:
            click.echo(message)
        raise click.ClickException(f"Replay failed for '{run_id}' ({len(mismatches)} mismatches)")


if __name__ == "__main__":
    cli()
