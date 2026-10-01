"""Bounded episode execution, independent of CLI/notebook presentation."""

import datetime
import hashlib
import json
import time
import uuid
from pathlib import Path

from tamabench.agents.base import BaseAgent, DecisionMetadata
from tamabench.env.core import TamaEnv
from tamabench.env.time_engine import BenchmarkMode
from tamabench.logging.file_logger import FileLogger
from tamabench.logging.logger_process import LoggerProcess
from tamabench.metrics.calculator import BenchmarkMetricsCalculator, EpisodeMetrics
from tamabench.metrics.live_reporter import LiveReporter
from tamabench.schemas.actions import StepResult
from tamabench.schemas.errors import ErrorCategory, ErrorType
from tamabench.runtime.model_runtime import InferenceBudgetExceeded


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class BatchRunner:
    def __init__(self, db_path="tamabench_results.db", event_path="tamabench_events.jsonl",
                 mode=BenchmarkMode.LOGICAL, max_stalled_decisions=5, log_dir=None, trace_logs=True):
        if max_stalled_decisions <= 0:
            raise ValueError("max_stalled_decisions must be positive")
        self.db_path, self.event_path, self.mode = db_path, event_path, mode
        self.log_dir = str(log_dir or Path(db_path).parent / "logs")
        self.trace_logs = trace_logs
        self.max_stalled_decisions = max_stalled_decisions
        self.logger = LoggerProcess(db_path, event_path)
        self.logger.start()
        self._warmed_agents = set()
        self._agents = {}
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _warm_agent(self, agent):
        self._agents[id(agent)] = agent
        if id(agent) in self._warmed_agents:
            return 0.0
        elapsed = agent.warmup()
        self._warmed_agents.add(id(agent))
        return elapsed

    def run_episode(self, agent: BaseAgent, seed=42, max_simulated_minutes=None,
                    scenario_id="dynamic_v2", scenario_version=2, live_monitor=False,
                    max_consecutive_failures=5, run_config=None, experiment_id=None) -> EpisodeMetrics:
        if self._closed:
            raise RuntimeError("Runner is closed")
        limit = max_simulated_minutes if max_simulated_minutes is not None else 7 * 1440
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("max_simulated_minutes must be a positive integer")
        if max_consecutive_failures <= 0:
            raise ValueError("max_consecutive_failures must be positive")
        config = dict(run_config or {})
        config.update(max_simulated_minutes=limit, scenario_id=scenario_id, scenario_version=scenario_version,
                      mode=self.mode.value, agent_type=agent.name)
        # Callers must supply secret-free configuration, never credentials or auth headers.
        if any(k.lower() in {"api_key", "password", "authorization", "token"} for k in config):
            raise ValueError("Run configuration must not contain credentials")
        config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()
        run_id = "run_" + uuid.uuid4().hex[:12]
        started = time.perf_counter()
        env = TamaEnv(mode=self.mode)
        obs = env.reset(seed=seed, scenario_id=scenario_id, scenario_version=scenario_version,
                        max_simulated_minutes=limit)
        self.logger.log_run({"run_id": run_id, "seed": seed, "scenario_id": env.state.scenario_id,
            "scenario_version": env.state.scenario_version, "benchmark_version": "2.0.0",
            "environment_version": "2.0.0", "mode": self.mode.value, "agent_type": agent.name,
            "model_name": getattr(agent, "model_name", agent.name),
            "schema_mode": getattr(agent, "schema_mode", "typed"), "started_at": utcnow(),
            "ended_at": None, "survived": 0, "simulated_duration_minutes": 0, "status": "running",
            "experiment_id": experiment_id, "config_hash": config_hash, "config_json": config_json,
            "max_simulated_minutes": limit})
        file_logger = live = None
        warmup_ms = 0.0
        warmup_calls = warmup_input = warmup_output = 0
        steps = failures = stalled = income = spending = jobs = jobs_failed = 0
        used_calls = used_tokens = 0
        token_usage_complete = True
        status, reason = "running", None
        caught = None
        runtime = getattr(agent, "runtime", None)
        runtime_before = {key: getattr(runtime, key, 0) for key in ("api_calls", "warmup_calls", "cleanup_calls", "input_tokens", "output_tokens", "generation_ms")}
        if runtime is not None:
            runtime.remaining_calls = config.get("max_api_calls")
            runtime.remaining_tokens = config.get("max_total_tokens")
            runtime.deadline = started + config["max_wall_seconds"] if config.get("max_wall_seconds") else None
        try:
            file_logger = FileLogger(run_id, self.log_dir) if self.trace_logs else None
            agent.reset_episode()
            before_warm = tuple(getattr(runtime, x, 0) for x in ("warmup_calls", "warmup_input_tokens", "warmup_output_tokens"))
            warmup_ms = self._warm_agent(agent)
            warmup_calls, warmup_input, warmup_output = tuple(
                getattr(runtime, x, 0) - old for x, old in zip(("warmup_calls", "warmup_input_tokens", "warmup_output_tokens"), before_warm))
            used_calls = warmup_calls
            used_tokens = warmup_input + warmup_output
            if live_monitor:
                live = LiveReporter(model_name=getattr(agent, "model_name", agent.name), seed=seed)
                live.start()
            self.logger.log_event(run_id, "SIMULATION_STARTED", 0, {"seed": seed, "agent": agent.name}, obs.state_hash)
            while not env.horizon_reached and not env.terminated:
                elapsed_seconds = time.perf_counter() - started
                limits = (("max_decisions", steps), ("max_api_calls", used_calls),
                          ("max_total_tokens", used_tokens), ("max_wall_seconds", elapsed_seconds))
                exceeded = next((key for key, value in limits if config.get(key) is not None and value >= config[key]), None)
                if exceeded:
                    status, reason = "budget_exhausted", exceeded
                    break
                current = env.observe()
                steps += 1
                if live:
                    live.set_model_status("generating")
                decision_started = time.perf_counter()
                raw, proposal, schema_error = agent.select_action(current)
                decision_ms = (time.perf_counter() - decision_started) * 1000
                metadata = getattr(agent, "last_decision", DecisionMetadata())
                source = getattr(metadata, "action_source", None) or ("model" if runtime is not None else "baseline")
                api_calls = metadata.attempt_count if runtime is not None else 0
                if source == "policy":
                    api_calls = 0
                used_calls += api_calls
                used_tokens += metadata.input_tokens + metadata.total_output_tokens
                token_usage_complete = token_usage_complete and metadata.usage_available
                if live:
                    live.set_model_status("idle")
                simulation_started = time.perf_counter()
                previous_minute = env.state.total_minutes
                if schema_error is not None or proposal is None:
                    result = StepResult(success=False, observation=current, error=schema_error,
                                        execution_minutes=0, state_hash=current.state_hash)
                else:
                    result = env.commit(proposal)
                simulation_ms = (time.perf_counter() - simulation_started) * 1000
                failures = 0 if result.success else failures + 1
                stalled = stalled + 1 if env.state.total_minutes == previous_minute else 0
                if proposal and proposal.action == "work":
                    if result.completed:
                        job = next(j for j in current.jobs_available if j.id == proposal.job_id)
                        income += job.reward
                        jobs += 1
                    else:
                        jobs_failed += 1
                if proposal and proposal.action == "buy" and result.success:
                    item = next(i for i in current.shop_items_available if i.item == proposal.item)
                    spending += item.cost * proposal.amount
                error = schema_error or result.error
                schema_valid = proposal is not None and schema_error is None and metadata.final_valid
                first_valid = metadata.first_pass_valid and schema_error is None
                decision_id = f"dec_{run_id}_{steps:06d}"
                compute = getattr(agent, "last_compute", None)
                generation_ms = float(getattr(compute, "generation_ms", 0))
                validation_ms = float(getattr(compute, "schema_validation_ms", 0))
                log_start = time.perf_counter()
                if file_logger:
                    file_logger.log_step(steps, current, raw, proposal, result,
                        latency_ms=generation_ms, thinking_process=getattr(agent, "last_reasoning", ""),
                        decision_metadata=metadata.to_dict())
                if live:
                    live.update(current, steps, proposal, result, latency_ms=generation_ms)
                logging_ms = (time.perf_counter() - log_start) * 1000
                self.logger.log_decision({"decision_id": decision_id, "run_id": run_id, "step_index": steps,
                    "day": current.time.day, "hour": current.time.hour, "minute": current.time.minute,
                    "state_hash": current.state_hash, "next_state_hash": result.state_hash,
                    "observation_json": json.dumps(current.to_dict(), separators=(",", ":")),
                    "raw_model_output": raw,
                    "parsed_action_json": json.dumps(proposal.model_dump(exclude_none=True)) if proposal else None,
                    "action_name": proposal.action if proposal else "unknown", "is_schema_valid": int(schema_valid),
                    "is_env_valid": int(result.success), "action_completed": int(result.completed),
                    "error_category": error.category.value if error else None,
                    "error_type": error.error_type.value if error else None,
                    "error_message": error.message if error else None, "execution_minutes": result.execution_minutes,
                    "finish_reason": metadata.finish_reason, "was_truncated": int(metadata.was_truncated),
                    "generation_attempt": metadata.generation_attempt, "attempt_count": metadata.attempt_count,
                    "first_pass_valid": int(first_valid), "final_valid": int(schema_valid),
                    "recovered": int(metadata.recovered), "first_failure_type": metadata.first_failure_type,
                    "action_source": source, "original_action_json": getattr(metadata, "original_action_json", None),
                    "override_reason": getattr(metadata, "override_reason", None),
                    "schema_observed": int(metadata.schema_observed),
                    "attempts_json": json.dumps(metadata.attempts, separators=(",", ":"))},
                    trace_data={"decision_id": decision_id, "run_id": run_id,
                        "provider_reasoning": getattr(agent, "last_reasoning", ""),
                        "decision_rationale": proposal.trace.decision_rationale if proposal and proposal.trace else None,
                        "confidence": proposal.trace.confidence if proposal and proposal.trace else None},
                    runtime_data={"decision_id": decision_id, "run_id": run_id,
                        "model_warmup_ms": warmup_ms if steps == 1 else 0, "model_resident": int(bool(getattr(agent, "model_resident", False))),
                        "api_calls": api_calls, "generation_ms": generation_ms, "schema_validation_ms": validation_ms,
                        "total_decision_ms": decision_ms, "input_tokens": metadata.input_tokens,
                        "output_tokens": metadata.total_output_tokens, "reasoning_tokens": metadata.reasoning_tokens,
                        "json_tokens": metadata.json_tokens, "total_tokens": metadata.total_tokens,
                        "simulation_ms": simulation_ms, "logging_ms": logging_ms,
                        "other_ms": max(0.0, decision_ms - generation_ms - validation_ms)})
                self.logger.log_event(run_id, "ACTION_COMPLETE", env.state.total_minutes,
                    {"step_index": steps, "action": proposal.action if proposal else None,
                     "action_source": source, "success": result.success, "completed": result.completed,
                     "execution_minutes": result.execution_minutes}, result.state_hash)
                if not metadata.usage_available and config.get("max_total_tokens") is not None:
                    status, reason = "infrastructure_failed", "Provider omitted token usage required for the configured token budget"
                    break
                if error and error.category == ErrorCategory.INFRASTRUCTURE:
                    status = "budget_exhausted" if error.error_type == ErrorType.BUDGET_EXHAUSTED else "infrastructure_failed"
                    reason = error.message
                    break
                if config.get("max_total_tokens") is not None and used_tokens > config["max_total_tokens"]:
                    status, reason = "budget_exhausted", "max_total_tokens observed after response"
                    break
                if failures >= max_consecutive_failures:
                    status, reason = "invalid_action_abort", "max_consecutive_failures"
                    break
                if stalled >= self.max_stalled_decisions:
                    status, reason = "stalled", "agent_stalled"
                    break
            if status == "running":
                status = "died" if env.terminated else "completed"
                reason = env.termination_reason or "horizon_reached"
        except InferenceBudgetExceeded as exc:
            status, reason = "budget_exhausted", str(exc)
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "infrastructure_failed"
            reason = f"{type(exc).__name__}: {exc}"
            caught = exc
        finally:
            cleanup_errors = []
            if live:
                try:
                    live.stop()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            stats = env.statistics
            outcome = {"run_id": run_id, "survived": int(status == "completed"),
                "simulated_days": env.state.total_minutes / 1440, "simulated_minutes": env.state.total_minutes,
                "final_health": env.state.pet.health, "min_health": stats.min_health,
                "avg_health": stats.avg_health, "final_happiness": env.state.pet.happiness,
                "avg_happiness": stats.avg_happiness, "final_money": env.state.agent.money,
                "final_energy": env.state.agent.energy, "total_income": income, "total_spending": spending,
                "jobs_completed": jobs, "jobs_failed": jobs_failed, "status": status, "termination_reason": reason,
                "health_integral": stats.health_integral, "happiness_integral": stats.happiness_integral}
            if file_logger:
                try:
                    file_logger.log_summary(status == "completed", outcome["simulated_days"], outcome["final_health"], outcome["final_money"], status=status)
                except BaseException as exc:
                    cleanup_errors.append(exc)
                finally:
                    try:
                        file_logger.close()
                    except BaseException as exc:
                        cleanup_errors.append(exc)
            if cleanup_errors:
                status = "infrastructure_failed"
                reason = f"Trace/display finalization failed ({type(cleanup_errors[0]).__name__})"
                caught = caught or cleanup_errors[0]
                outcome.update(status=status, termination_reason=reason, survived=0)
            self.logger.log_event(run_id, "SIMULATION_ENDED", env.state.total_minutes,
                                  {"status": status, "termination_reason": reason}, env.state.compute_hash())
            final = {"run_id": run_id, "status": status, "termination_reason": reason, "ended_at": utcnow(),
                     "survived": outcome["survived"], "simulated_duration_minutes": env.state.total_minutes,
                     "warmup_ms": warmup_ms, "warmup_api_calls": warmup_calls,
                     "warmup_input_tokens": warmup_input, "warmup_output_tokens": warmup_output,
                     "token_usage_complete": int(token_usage_complete)}
            if runtime is not None:
                runtime_calls = getattr(runtime, "api_calls", 0) - runtime_before["api_calls"]
                if runtime_calls > 0:
                    final.update(generation_api_calls=max(0, runtime_calls - warmup_calls),
                        generation_input_tokens=getattr(runtime, "input_tokens", 0) - runtime_before["input_tokens"],
                        generation_output_tokens=getattr(runtime, "output_tokens", 0) - runtime_before["output_tokens"],
                        generation_ms=getattr(runtime, "generation_ms", 0) - runtime_before["generation_ms"])
            self.logger.finalize_run(final, outcome)
            self.logger.flush()
            wall = (time.perf_counter() - started) * 1000
            final["episode_wall_time_ms"] = wall
            outcome["episode_wall_time_ms"] = wall
            self.logger.finalize_run(final, outcome)
            self.logger.flush()
        if caught is not None:
            raise caught
        return BenchmarkMetricsCalculator(self.db_path).calculate_run_metrics(run_id)

    def close(self):
        if self._closed:
            return
        self._closed = True
        errors = []
        for agent in self._agents.values():
            try:
                agent.close()
            except BaseException as exc:
                errors.append(exc)
        try:
            self.logger.stop()
        except BaseException as exc:
            errors.append(exc)
        if errors:
            raise errors[0]
