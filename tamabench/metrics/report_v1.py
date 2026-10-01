"""Truthful, configuration-separated reports; report-v1 remains a command alias."""
from dataclasses import asdict
import json
import math
from rich.console import Console
from rich.table import Table
from tamabench.logging.database import DatabaseStore
from tamabench.metrics.calculator import BenchmarkMetricsCalculator


SCIENTIFIC_STATUSES = {"completed", "died", "invalid_action_abort", "stalled", "budget_exhausted"}


def wilson_interval(successes, count):
    """95% binomial Wilson interval; descriptive uncertainty, not a ranking test."""
    if count == 0:
        return None
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z*z/count
    center = (p + z*z/(2*count))/denominator
    half = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count))/denominator
    return [max(0.0, center-half), min(1.0, center+half)]


class ReportV1Generator:
    def __init__(self, db_path="tamabench_results.db"):
        self.db = DatabaseStore(db_path)
        self.console = Console()
        self.calculator = BenchmarkMetricsCalculator(db_path)

    def records(self, run_ids=None, experiment_id=None):
        selected = set(run_ids) if run_ids else None
        result = []
        for run in self.db.list_runs(experiment_id):
            if selected is not None and run["run_id"] not in selected:
                continue
            row = dict(run)
            if row["status"] in SCIENTIFIC_STATUSES:
                row["metrics"] = asdict(self.calculator.calculate_run_metrics(row["run_id"]))
            else:
                row["metrics"] = None
            result.append(row)
        return result

    def generate_report(self, run_ids=None, experiment_id=None):
        records = self.records(run_ids, experiment_id)
        if not records:
            self.console.print("No runs found in database.")
            return records
        self.console.print("\n[bold]TamaBench V2 evidence report[/bold]")
        groups = {}
        for row in records:
            key = (row["config_hash"] or "legacy", row["agent_type"], row["environment_version"])
            groups.setdefault(key, []).append(row)
        for (fingerprint, agent, version), rows in groups.items():
            table = Table(title=f"{agent} | environment {version} | config {fingerprint[:12]}")
            for title in ("Run", "Seed", "Status", "Days", "Time avg health", "Valid actions", "Income", "Requests"):
                table.add_column(title)
            for row in rows:
                metric = row["metrics"]
                values = (f"{metric['simulated_days']:.3f}", f"{metric['avg_health']:.2f}",
                          (f"{metric['valid_action_rate']:.1f}%" if metric["valid_action_rate"] is not None else "N/A"), str(metric['total_income']), str(metric['api_calls'])) if metric else ("N/A",)*5
                table.add_row(row["run_id"], str(row["seed"]), row["status"], *values)
            self.console.print(table)
            scientific = [r for r in rows if r["status"] in SCIENTIFIC_STATUSES]
            # Repeated attempts of the same seed are not independent replicates.
            per_seed = {}
            for row in scientific:
                per_seed.setdefault(row["seed"], row)
            success = sum(bool(r["survived"]) for r in per_seed.values())
            interval = wilson_interval(success, len(per_seed))
            if interval:
                self.console.print(f"First finished attempt per seed: {success}/{len(per_seed)} survived; "
                                   f"95% Wilson interval {interval[0]*100:.1f}–{interval[1]*100:.1f}%")
            extra = len(scientific) - len(per_seed)
            if extra:
                self.console.print(f"{extra} additional finished attempts shown above but excluded from the per-seed summary.")
            excluded = len(rows)-len(scientific)
            if excluded:
                self.console.print(f"{excluded} incomplete/infrastructure/legacy records excluded; rerun or inspect them before comparison.")
        self.console.print("Configurations are not pooled. Intervals are descriptive; compare preregistered paired seed sets. "
                           "The legacy composite score is secondary, and unsupported metrics remain unavailable.")
        return records
