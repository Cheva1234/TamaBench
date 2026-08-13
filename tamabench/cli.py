"""Command Line Interface for TamaBench V1."""

import click
import sys
from tamabench.agents.random_schema_agent import RandomSchemaAgent
from tamabench.agents.random_valid_agent import RandomValidAgent
from tamabench.agents.rule_agent import RuleAgent
from tamabench.agents.raw_llm_agent import RawLLMAgent
from tamabench.agents.harness_v1_agent import HarnessV1Agent
from tamabench.runner.batch_runner import BatchRunner
from tamabench.metrics.reporter import BenchmarkReporter
from tamabench.logging.replay import ReplayEngine
from tamabench.env.time_engine import BenchmarkMode


@click.group()
def cli():
    """TamaBench V1: Persistent Sandbox Agent Benchmark for Small AI Models."""
    pass


@cli.command()
@click.option("--agent", type=click.Choice(["random_schema", "random_valid", "rule", "raw_llm", "harness_v1"]), default="rule", help="Agent baseline type")
@click.option("--model", type=str, default="qwen2.5:3b", help="Model name if using raw_llm agent")
@click.option("--api-base", type=str, default="http://localhost:11434/v1", help="OpenAI-compatible API base URL")
@click.option("--max-output-tokens", type=click.IntRange(min=1), default=4096, show_default=True, help="Maximum generated tokens per model attempt")
@click.option("--timeout", type=float, default=120.0, show_default=True, help="Per-request timeout in seconds (raise for thinking models)")
@click.option("--reasoning-effort", type=click.Choice(["none", "low", "medium", "high"]), default="none", show_default=True, help="Reasoning budget sent to compatible models")
@click.option("--episodes", type=int, default=1, help="Number of benchmark episodes to run")
@click.option("--seed-start", type=int, default=42, help="Starting RNG seed")
@click.option("--schema-mode", type=click.Choice(["raw_json", "provider_constrained"]), default="raw_json", help="Schema benchmark mode")
@click.option("--display", type=click.Choice(["live", "compact", "quiet"]), default="live", help="Display mode for terminal output")
@click.option("--speed", type=click.Choice(["accelerated", "reference"]), default="accelerated", help="Execution speed mode")
@click.option("--difficulty", type=click.Choice(["easy", "standard", "hard"]), default="standard", show_default=True, help="Scenario difficulty preset")
@click.option("--model-lifecycle", type=click.Choice(["warm", "cold"]), default="warm", help="Model lifecycle management mode")
@click.option("--db-path", type=str, default="tamabench_results.db", help="SQLite results database path")
@click.option("--event-path", type=str, default="tamabench_events.jsonl", help="JSONL event stream file path")
def run(agent, model, api_base, max_output_tokens, timeout, reasoning_effort, episodes, seed_start, schema_mode, display, speed, difficulty, model_lifecycle, db_path, event_path):
    """Executes a benchmark experiment run."""
    mode_enum = BenchmarkMode.ACCELERATED if speed == "accelerated" else BenchmarkMode.LOGICAL
    runner = BatchRunner(db_path=db_path, event_path=event_path, mode=mode_enum)
    reporter = BenchmarkReporter()

    if display != "live":
        click.echo(f"Starting TamaBench V1 Benchmark: Agent='{agent}', Difficulty='{difficulty}', Speed='{speed}', Display='{display}', Lifecycle='{model_lifecycle}'")

    shared_agent = None
    if model_lifecycle == "warm":
        if agent == "raw_llm":
            shared_agent = RawLLMAgent(
                model_name=model,
                api_base=api_base,
                schema_mode=schema_mode,
                max_output_tokens=max_output_tokens,
                timeout=timeout,
                reasoning_effort=reasoning_effort,
            )
        elif agent == "harness_v1":
            shared_agent = HarnessV1Agent(
                model_agent=RawLLMAgent(
                    model_name=model,
                    api_base=api_base,
                    schema_mode=schema_mode,
                    max_output_tokens=max_output_tokens,
                    timeout=timeout,
                    reasoning_effort=reasoning_effort,
                )
            )
        elif agent == "rule":
            shared_agent = RuleAgent()

    for ep in range(episodes):
        current_seed = seed_start + ep

        if shared_agent is not None:
            agent_obj = shared_agent
        else:
            if agent == "random_schema":
                agent_obj = RandomSchemaAgent(seed=current_seed)
            elif agent == "random_valid":
                agent_obj = RandomValidAgent(seed=current_seed)
            elif agent == "rule":
                agent_obj = RuleAgent()
            elif agent == "raw_llm":
                agent_obj = RawLLMAgent(
                    model_name=model,
                    api_base=api_base,
                    schema_mode=schema_mode,
                    max_output_tokens=max_output_tokens,
                    timeout=timeout,
                    reasoning_effort=reasoning_effort,
                )
            elif agent == "harness_v1":
                agent_obj = HarnessV1Agent(
                    model_agent=RawLLMAgent(
                        model_name=model,
                        api_base=api_base,
                        schema_mode=schema_mode,
                        max_output_tokens=max_output_tokens,
                        timeout=timeout,
                        reasoning_effort=reasoning_effort,
                    )
                )
            else:
                agent_obj = RuleAgent()

        live_flag = (display == "live")
        metrics = runner.run_episode(
            agent=agent_obj,
            seed=current_seed,
            live_monitor=live_flag,
            difficulty=difficulty,
        )
        if display == "compact":
            click.echo(f"[{ep+1:03d}/{episodes:03d}] Difficulty {difficulty} | Seed #{current_seed} | Days {metrics.simulated_days:.1f} | Health {metrics.avg_health:.1f} | Survived: {metrics.survived}")
        elif display == "quiet" and (ep + 1) % 10 == 0:
            click.echo(f"Completed {ep + 1} / {episodes} episodes")
        elif display != "quiet":
            reporter.print_summary(metrics, model_name=agent_obj.name, episodes=1)

    runner.close()


@cli.command()
@click.option("--run-id", required=True, type=str, help="Run ID to replay and verify")
@click.option("--db-path", type=str, default="tamabench_results.db", help="SQLite database path")
def replay(run_id, db_path):
    """Replays a recorded episode and verifies state_hash determinism."""
    click.echo(f"Replaying run '{run_id}' from database '{db_path}'...")
    engine = ReplayEngine(db_path=db_path)
    success, mismatches = engine.replay_run(run_id)

    if success:
        click.secho(f"✓ Replay successful for run '{run_id}'! State SHA-256 hashes matched 100%.", fg="green", bold=True)
    else:
        click.secho(f"✗ Replay failed for run '{run_id}' with {len(mismatches)} state hash mismatch(es):", fg="red", bold=True)
        for m in mismatches:
            click.echo(f"  - {m}")
        sys.exit(1)


@cli.command(name="report-v1")
@click.option("--db-path", type=str, default="tamabench_results.db", help="SQLite database path")
def report_v1_cmd(db_path):
    """Renders 4-Layer Real Model Benchmark Report V1."""
    from tamabench.metrics.report_v1 import ReportV1Generator
    generator = ReportV1Generator(db_path=db_path)
    generator.generate_report()


if __name__ == "__main__":
    cli()
