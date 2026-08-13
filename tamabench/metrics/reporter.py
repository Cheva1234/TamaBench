"""Rich Terminal Reporter and Summary Exporter for TamaBench V1."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tamabench.metrics.calculator import EpisodeMetrics


class BenchmarkReporter:
    def __init__(self):
        self.console = Console()

    def print_summary(self, metrics: EpisodeMetrics, model_name: str = "RuleAgent", episodes: int = 1):
        self.console.print("\n[bold cyan]TamaBench V1 Benchmark Report[/bold cyan]\n")

        # Overview Panel
        self.console.print(
            Panel(
                f"[bold white]Model:[/bold white] [green]{model_name}[/green]\n"
                f"[bold white]Episodes:[/bold white] {episodes}\n"
                f"[bold white]Simulated Days:[/bold white] {metrics.simulated_days} days\n"
                f"[bold white]Survival Status:[/bold white] [{'green' if metrics.survived else 'red'}]{'SURVIVED' if metrics.survived else 'DIED'}[/{'green' if metrics.survived else 'red'}]",
                title="Experiment Overview",
                border_style="cyan",
            )
        )

        # Long Horizon & Pet Care Table
        table_pet = Table(title="Long-Horizon & Pet Care Metrics", border_style="bright_blue")
        table_pet.add_column("Metric", style="bold white")
        table_pet.add_column("Value", style="yellow", justify="right")

        table_pet.add_row("Survival Rate", f"{100.0 if metrics.survived else 0.0:.1f}%")
        table_pet.add_row("Average Pet Health", f"{metrics.avg_health:.1f}")
        table_pet.add_row("Minimum Pet Health", f"{metrics.min_health:.1f}")
        table_pet.add_row("Average Happiness", f"{metrics.avg_happiness:.1f}")
        table_pet.add_row("Critical Decision Accuracy", f"{metrics.critical_decision_acc:.1f}%")
        self.console.print(table_pet)

        # Schema & Decision Quality Table
        table_schema = Table(title="Schema Compliance & Decision Quality", border_style="magenta")
        table_schema.add_column("Metric", style="bold white")
        table_schema.add_column("Value", style="green", justify="right")

        table_schema.add_row("First-Pass Schema Accuracy", f"{metrics.first_pass_schema_acc:.1f}%")
        table_schema.add_row("Final Schema Accuracy", f"{metrics.final_schema_acc:.1f}%")
        table_schema.add_row("Schema Recovery Rate", f"{metrics.final_schema_recovery_rate:.1f}%")
        table_schema.add_row("Output Truncation Rate", f"{metrics.truncation_rate:.1f}%")
        table_schema.add_row("Retry Rate", f"{metrics.retry_rate:.1f}%")
        table_schema.add_row("Valid Action Rate", f"{metrics.valid_action_rate:.1f}%")
        table_schema.add_row("Invalid Action Rate", f"{metrics.invalid_action_rate:.1f}%")
        table_schema.add_row("Productive Action Rate", f"{metrics.productive_action_rate:.1f}%")
        table_schema.add_row("Wasteful Action Rate", f"{metrics.wasteful_action_rate:.1f}%")
        table_schema.add_row("Prediction Accuracy", f"{metrics.prediction_accuracy:.1f}%")
        self.console.print(table_schema)

        # Compute & Work Table
        table_compute = Table(title="Compute Efficiency & Economy", border_style="yellow")
        table_compute.add_column("Metric", style="bold white")
        table_compute.add_column("Value", style="cyan", justify="right")

        table_compute.add_row("Total Income Earned", f"${metrics.total_income}")
        table_compute.add_row("Total Spending", f"${metrics.total_spending}")
        table_compute.add_row("Jobs Completed", f"{metrics.jobs_completed}")
        table_compute.add_row("Average Context / Decision", f"{metrics.avg_context_tokens} tokens")
        table_compute.add_row("Total Input Tokens", f"{metrics.total_input_tokens:,}")
        table_compute.add_row("Total Output Tokens", f"{metrics.total_output_tokens:,}")
        table_compute.add_row("Reasoning Tokens / Decision", f"{metrics.reasoning_tokens_per_decision:.1f}")
        table_compute.add_row("JSON Tokens / Decision", f"{metrics.json_tokens_per_decision:.1f}")
        table_compute.add_row("API Calls / Simulated Day", f"{metrics.api_calls_per_simulated_day:.1f}")
        table_compute.add_row("Average Decision Latency", f"{metrics.avg_decision_latency_ms:.1f}ms")
        table_compute.add_row("p95 Decision Latency", f"{metrics.p95_decision_latency_ms:.1f}ms")
        table_compute.add_row("Profiler: Inference", f"{metrics.inference_ms:.1f}ms")
        table_compute.add_row("Profiler: Simulation", f"{metrics.simulation_ms:.1f}ms")
        table_compute.add_row("Profiler: Validation", f"{metrics.validation_ms:.1f}ms")
        table_compute.add_row("Profiler: Logging", f"{metrics.logging_ms:.1f}ms")
        table_compute.add_row("Profiler: Other", f"{metrics.other_ms:.1f}ms")
        table_compute.add_row("Profiler: Episode Wall", f"{metrics.episode_wall_time_ms:.1f}ms")
        self.console.print(table_compute)
