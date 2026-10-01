"""Benchmark Metrics Calculator Engine for TamaBench V1."""

import json
from dataclasses import dataclass, field
from typing import Any
from tamabench.logging.database import DatabaseStore
from tamabench.schemas.actions import ActionPrediction


@dataclass
class EpisodeMetrics:
    run_id: str
    survived: bool
    simulated_days: float
    avg_health: float
    min_health: float
    avg_happiness: float
    first_pass_schema_acc: float | None
    final_schema_acc: float | None
    schema_retry_rate: float | None
    valid_action_rate: float | None
    invalid_action_rate: float | None
    critical_decision_acc: float | None
    productive_action_rate: float | None
    wasteful_action_rate: float | None
    harmful_action_rate: float | None
    prediction_accuracy: float | None
    confidence_calibration_error: float | None
    total_income: int
    total_spending: int
    jobs_completed: int
    avg_decision_latency_ms: float
    avg_context_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    final_schema_recovery_rate: float | None = 0.0
    truncation_rate: float | None = 0.0
    retry_rate: float | None = 0.0
    recovered_decisions: int = 0
    total_decisions: int = 0
    reasoning_tokens_per_decision: float = 0.0
    json_tokens_per_decision: float = 0.0
    total_tokens_per_simulated_day: float = 0.0
    p95_decision_latency_ms: float = 0.0
    api_calls: int = 0
    api_calls_per_simulated_day: float = 0.0
    model_warmup_ms: float = 0.0
    model_resident: bool = False
    inference_ms: float = 0.0
    simulation_ms: float = 0.0
    validation_ms: float = 0.0
    logging_ms: float = 0.0
    other_ms: float = 0.0
    episode_wall_time_ms: float = 0.0
    score: float = 0.0
    status: str = "legacy"
    termination_reason: str | None = None
    schema_decisions: int = 0
    model_decisions: int = 0
    policy_decisions: int = 0
    overrides: int = 0
    warmup_api_calls: int = 0
    token_usage_complete: bool = True
    overhead_ms: float = 0.0
    inference_fraction: float = 0.0


class BenchmarkMetricsCalculator:
    def __init__(self, db_path: str = "tamabench_results.db"):
        self.db = DatabaseStore(db_path=db_path)

    def calculate_run_metrics(self, run_id: str) -> EpisodeMetrics:
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            run_row = cursor.fetchone()
            if not run_row:
                raise ValueError(f"Run ID '{run_id}' not found.")

            decisions = self.db.get_run_decisions(run_id)
            cursor.execute("SELECT * FROM outcomes WHERE run_id = ?", (run_id,))
            outcome_row = cursor.fetchone()

            runtime_rows = conn.execute(
                "SELECT * FROM runtime_metrics WHERE run_id = ?", (run_id,)
            ).fetchall()

        total_decisions = len(decisions)
        survived = bool(outcome_row["survived"] if outcome_row is not None else run_row["survived"])

        schema_rows = [d for d in decisions if d["action_source"] != "policy" and d["schema_observed"]]
        schema_metrics = self._calculate_schema_metrics(schema_rows, len(schema_rows))
        action_metrics = self._calculate_action_metrics(decisions, total_decisions)
        outcome_metrics = self._calculate_outcome_metrics(outcome_row, run_row)
        runtime_metrics = self._calculate_runtime_metrics(decisions, runtime_rows, outcome_metrics["simulated_days"], total_decisions)

        wall_ms = float((outcome_row["episode_wall_time_ms"] if outcome_row else None) or run_row["episode_wall_time_ms"] or runtime_metrics["episode_wall_time_ms"])
        warmup_ms = float(run_row["warmup_ms"] or 0)
        runtime_metrics["model_warmup_ms"] = warmup_ms
        runtime_metrics["episode_wall_time_ms"] = wall_ms
        warmup_calls = int(run_row["warmup_api_calls"] or 0)
        if run_row["generation_api_calls"] is not None:
            runtime_metrics["api_calls"] = run_row["generation_api_calls"]
            runtime_metrics["total_input_tokens"] = run_row["generation_input_tokens"] or 0
            runtime_metrics["total_output_tokens"] = run_row["generation_output_tokens"] or 0
            runtime_metrics["inference_ms"] = run_row["generation_ms"] or 0.0
        runtime_metrics["api_calls"] += warmup_calls
        runtime_metrics["total_input_tokens"] += int(run_row["warmup_input_tokens"] or 0)
        runtime_metrics["total_output_tokens"] += int(run_row["warmup_output_tokens"] or 0)
        days = outcome_metrics["simulated_days"]
        runtime_metrics["api_calls_per_simulated_day"] = runtime_metrics["api_calls"] / days if days else 0.0
        runtime_metrics["total_tokens_per_simulated_day"] = (runtime_metrics["total_input_tokens"] + runtime_metrics["total_output_tokens"]) / days if days else 0.0
        inference_total = runtime_metrics["inference_ms"] + warmup_ms

        # Legacy composite retained only as a secondary diagnostic in v2.
        score = (outcome_metrics["simulated_days"] * 1000) + (outcome_metrics["avg_health"] * 10) + (outcome_metrics["total_income"] - outcome_metrics["total_spending"])

        return EpisodeMetrics(
            run_id=run_id,
            survived=survived,
            score=round(score, 2) if outcome_metrics["simulated_days"] else 0.0,
            **schema_metrics,
            **action_metrics,
            **outcome_metrics,
            **runtime_metrics,
            status=run_row["status"],
            termination_reason=run_row["termination_reason"],
            schema_decisions=len(schema_rows),
            model_decisions=sum(d["action_source"] == "model" for d in decisions),
            policy_decisions=sum(d["action_source"] == "policy" for d in decisions),
            overrides=sum(bool(d["override_reason"]) for d in decisions),
            warmup_api_calls=warmup_calls,
            token_usage_complete=bool(run_row["token_usage_complete"]),
            overhead_ms=max(0.0, wall_ms - inference_total),
            inference_fraction=min(1.0, inference_total / wall_ms) if wall_ms else 0.0,
            critical_decision_acc=None,
            productive_action_rate=None,
            wasteful_action_rate=None,
            harmful_action_rate=None,
            prediction_accuracy=None,
            confidence_calibration_error=None,
            total_decisions=total_decisions,
        )

    def _calculate_schema_metrics(self, decisions, total_decisions) -> dict:
        if not total_decisions:
            return {"first_pass_schema_acc": None, "final_schema_acc": None, "schema_retry_rate": None,
                    "final_schema_recovery_rate": None, "truncation_rate": None, "retry_rate": None,
                    "recovered_decisions": 0}
        denominator = total_decisions
        first_pass_count = sum(1 for d in decisions if d["first_pass_valid"])
        final_valid_count = sum(1 for d in decisions if d["final_valid"])
        recovered_count = sum(1 for d in decisions if d["recovered"])
        truncation_count = sum(1 for d in decisions if d["was_truncated"])
        retry_count = sum(1 for d in decisions if d["attempt_count"] > 1)

        return {
            "first_pass_schema_acc": round((first_pass_count / denominator) * 100.0, 1),
            "final_schema_acc": round((final_valid_count / denominator) * 100.0, 1),
            "schema_retry_rate": round((retry_count / denominator) * 100.0, 1),
            "final_schema_recovery_rate": round((recovered_count / denominator) * 100.0, 1),
            "truncation_rate": round((truncation_count / denominator) * 100.0, 1),
            "retry_rate": round((retry_count / denominator) * 100.0, 1),
            "recovered_decisions": recovered_count,
        }

    def _calculate_action_metrics(self, decisions, total_decisions) -> dict:
        if not total_decisions:
            return {"valid_action_rate": None, "invalid_action_rate": None}
        denominator = total_decisions
        valid_env_count = sum(1 for d in decisions if d["is_env_valid"])
        valid_action_rate = (valid_env_count / denominator) * 100.0
        return {
            "valid_action_rate": round(valid_action_rate, 1),
            "invalid_action_rate": round(100.0 - valid_action_rate, 1),
        }

    def _calculate_outcome_metrics(self, outcome_row, run_row) -> dict:
        return {
            "simulated_days": float(outcome_row["simulated_days"] if outcome_row else (run_row["simulated_duration_minutes"] / 1440.0)),
            "avg_health": round(outcome_row["avg_health"] if outcome_row else 100.0, 1),
            "min_health": round(outcome_row["min_health"] if outcome_row else 100.0, 1),
            "avg_happiness": round(outcome_row["avg_happiness"] if outcome_row else 100.0, 1),
            "total_income": outcome_row["total_income"] if outcome_row else 0,
            "total_spending": outcome_row["total_spending"] if outcome_row else 0,
            "jobs_completed": outcome_row["jobs_completed"] if outcome_row else 0,
        }

    def _calculate_runtime_metrics(self, decisions, runtime_rows, simulated_days, total_decisions) -> dict:
        denominator = total_decisions or 1
        runtime_by_decision = {row["decision_id"]: row for row in runtime_rows}
        runtime_values = [runtime_by_decision[d["decision_id"]] for d in decisions if d["decision_id"] in runtime_by_decision]

        latencies = sorted(float(row["total_decision_ms"] or 0.0) for row in runtime_values)
        p95_latency = _percentile(latencies, 0.95)

        total_input_tokens = sum(int(row["input_tokens"] or 0) for row in runtime_values)
        total_output_tokens = sum(int(row["output_tokens"] or 0) for row in runtime_values)
        total_reasoning_tokens = sum(int(row["reasoning_tokens"] or 0) for row in runtime_values)
        total_json_tokens = sum(int(row["json_tokens"] or 0) for row in runtime_values)

        api_calls = sum(int(row["api_calls"] or 0) for row in runtime_values)
        inference_ms = sum(float(row["generation_ms"] or 0.0) for row in runtime_values)
        validation_ms = sum(float(row["schema_validation_ms"] or 0.0) for row in runtime_values)
        simulation_ms = sum(float(row["simulation_ms"] or 0.0) for row in runtime_values)
        logging_ms = sum(float(row["logging_ms"] or 0.0) for row in runtime_values)
        other_ms = sum(float(row["other_ms"] or 0.0) for row in runtime_values)

        warmup_ms = max((float(row["model_warmup_ms"] or 0.0) for row in runtime_values), default=0.0)
        resident = any(bool(row["model_resident"]) for row in runtime_values)

        return {
            "avg_decision_latency_ms": round(sum(latencies) / len(latencies) if latencies else 0.0, 1),
            "p95_decision_latency_ms": round(p95_latency, 1),
            "avg_context_tokens": round(total_input_tokens / len(runtime_values)) if runtime_values else 0,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "reasoning_tokens_per_decision": round(total_reasoning_tokens / denominator, 1),
            "json_tokens_per_decision": round(total_json_tokens / denominator, 1),
            "total_tokens_per_simulated_day": round((total_input_tokens + total_output_tokens) / simulated_days, 1) if simulated_days else 0.0,
            "api_calls": api_calls,
            "api_calls_per_simulated_day": round(api_calls / simulated_days, 1) if simulated_days else 0.0,
            "model_warmup_ms": round(warmup_ms, 1),
            "model_resident": resident,
            "inference_ms": round(inference_ms, 1),
            "validation_ms": round(validation_ms, 1),
            "simulation_ms": round(simulation_ms, 1),
            "logging_ms": round(logging_ms, 1),
            "other_ms": round(other_ms, 1),
            "episode_wall_time_ms": round(inference_ms + validation_ms + simulation_ms + logging_ms + other_ms, 1),
        }

def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] + (values[upper] - values[lower]) * weight
