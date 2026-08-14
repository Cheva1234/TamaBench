"""Benchmark runner with model calls only at environment decision boundaries."""

import datetime
import json
import time
import uuid

from tamabench.agents.base import BaseAgent, DecisionMetadata
from tamabench.env.core import TamaEnv
from tamabench.env.time_engine import BenchmarkMode
from tamabench.logging.file_logger import FileLogger
from tamabench.logging.logger_process import LoggerProcess
from tamabench.metrics.calculator import BenchmarkMetricsCalculator, EpisodeMetrics
from tamabench.metrics.live_reporter import LiveReporter
from tamabench.evaluation.reference_policy import ReferencePolicyEvaluator


class BatchRunner:
    """Runs episodes without polling the model during blocking simulation time."""

    def __init__(
        self,
        db_path: str = "tamabench_results.db",
        event_path: str = "tamabench_events.jsonl",
        mode: BenchmarkMode = BenchmarkMode.LOGICAL,
    ):
        self.db_path = db_path
        self.event_path = event_path
        self.mode = mode
        self.logger = LoggerProcess(db_path=db_path, event_path=event_path)
        self.logger.start()
        self._warmed_agents: set[int] = set()
        self._agents: dict[int, BaseAgent] = {}

    def _warm_agent(self, agent: BaseAgent) -> float:
        self._agents[id(agent)] = agent
        if id(agent) in self._warmed_agents:
            return 0.0
        warmup_ms = agent.warmup()
        self._warmed_agents.add(id(agent))
        return warmup_ms

    def run_episode(
        self,
        agent: BaseAgent,
        seed: int = 42,
        max_simulated_minutes: int = 4320,
        scenario_id: str = "standard_v1",
        scenario_version: int = 1,
        live_monitor: bool = False,
        max_consecutive_failures: int = 5,
    ) -> EpisodeMetrics:
        """Execute an episode, making exactly one agent call per decision boundary."""
        episode_started = time.perf_counter()
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        file_logger = FileLogger(run_id=run_id)

        env = TamaEnv(mode=self.mode)
        obs = env.reset(seed=seed, scenario_id=scenario_id, scenario_version=scenario_version)
        agent.reset_episode()
        warmup_ms = self._warm_agent(agent)

        live_reporter = (
            LiveReporter(model_name=getattr(agent, "model_name", agent.name), seed=seed)
            if live_monitor
            else None
        )
        if live_reporter:
            live_reporter.start()

        self.logger.log_run(
            {
                "run_id": run_id,
                "seed": seed,
                "scenario_id": scenario_id,
                "scenario_version": scenario_version,
                "benchmark_version": "1.1.0",
                "environment_version": "1.1.0",
                "mode": self.mode.value,
                "agent_type": agent.name,
                "model_name": getattr(agent, "model_name", agent.name),
                "schema_mode": getattr(agent, "schema_mode", "raw_json"),
                "started_at": started_at,
                "ended_at": None,
                "simulated_duration_minutes": 0,
                "survived": 0,
            }
        )
        self.logger.log_event(
            run_id=run_id,
            event_type="SIMULATION_STARTED",
            simulation_minute=0,
            details={
                "seed": seed,
                "agent": agent.name,
                "model_resident": bool(getattr(agent, "model_resident", False)),
                "model_warmup_ms": warmup_ms,
            },
            state_hash=obs.state_hash,
        )

        step_index = 0
        total_income = 0
        total_spending = 0
        jobs_completed = 0
        jobs_failed = 0
        health_samples = [obs.pet.health]
        happiness_samples = [obs.pet.happiness]
        consecutive_failures = 0
        aborted = False
        termination_reason = None

        while env.state.total_minutes < max_simulated_minutes and not env.terminated:
            # A future environment implementation may leave a blocking action
            # pending. The runner advances it without asking the agent again.
            if not env.requires_decision():
                env.advance_to_next_boundary()
                continue

            step_index += 1
            current_obs = env.observe()
            pre_hash = current_obs.state_hash
            if live_reporter:
                live_reporter.set_model_status("generating")

            self.logger.log_event(
                run_id=run_id,
                event_type="MODEL_CALL_STARTED",
                simulation_minute=env.state.total_minutes,
                details={"step_index": step_index},
                state_hash=pre_hash,
            )

            decision_started = time.perf_counter()
            raw_output, proposal, schema_err = agent.select_action(current_obs)
            decision_wall_ms = (time.perf_counter() - decision_started) * 1000.0
            metadata = getattr(agent, "last_decision", DecisionMetadata())
            if schema_err is not None and proposal is None:
                metadata.final_valid = False
                metadata.first_pass_valid = False
                metadata.first_failure_type = schema_err.error_type.value
                metadata.first_failure_message = schema_err.message

            if live_reporter:
                live_reporter.set_model_status("idle")

            ReferencePolicyEvaluator.evaluate_decision(env.state, proposal)

            simulation_started = time.perf_counter()
            # `commit` is the explicit decision-boundary API. Blocking actions
            # advance analytically inside this call and never re-enter the loop.
            step_res = env.commit(proposal if proposal is not None else raw_output)
            simulation_ms = (time.perf_counter() - simulation_started) * 1000.0

            if step_res.success:
                consecutive_failures = 0
            else:
                consecutive_failures += 1

            post_obs = step_res.observation
            health_samples.append(post_obs.pet.health)
            happiness_samples.append(post_obs.pet.happiness)

            if proposal and proposal.action == "work" and step_res.success:
                job = next((j for j in current_obs.jobs_available if j.id == proposal.job_id), None)
                if job:
                    total_income += job.reward
                jobs_completed += 1
            elif proposal and proposal.action == "work":
                jobs_failed += 1

            if proposal and proposal.action == "buy" and step_res.success:
                item = next((s for s in current_obs.shop_items_available if s.item == proposal.item), None)
                total_spending += (item.cost if item else 0) * (proposal.amount or 1)

            decision_id = f"dec_{run_id}_{step_index:04d}"
            final_valid = bool(metadata.final_valid and proposal is not None and schema_err is None)
            first_pass_valid = bool(metadata.first_pass_valid and schema_err is None)
            metadata.final_valid = final_valid
            metadata.first_pass_valid = first_pass_valid

            is_schema_valid = int(final_valid)
            is_env_valid = int(step_res.success)
            error = step_res.error or schema_err
            err_cat = error.category.value if error else None
            err_type = error.error_type.value if error else None
            err_msg = error.message if error else None

            log_started = time.perf_counter()
            file_logger.log_step(
                step_index=step_index,
                obs=current_obs,
                raw_output=raw_output,
                proposal=proposal,
                step_result=step_res,
                latency_ms=getattr(agent, "last_compute", None).generation_ms
                if getattr(agent, "last_compute", None)
                else 0.0,
                thinking_process=getattr(agent, "last_reasoning", ""),
                decision_metadata=metadata.to_dict(),
            )
            logging_ms = (time.perf_counter() - log_started) * 1000.0

            if live_reporter:
                live_reporter.update(
                    current_obs,
                    step_index,
                    proposal,
                    step_res,
                    latency_ms=getattr(agent, "last_compute", None).generation_ms
                    if getattr(agent, "last_compute", None)
                    else 0.0,
                )

            compute = getattr(agent, "last_compute", None)
            generation_ms = compute.generation_ms if compute else 0.0
            schema_validation_ms = compute.schema_validation_ms if compute else 0.0
            total_decision_ms = max(
                compute.total_decision_ms if compute else 0.0,
                decision_wall_ms,
            )
            self.logger.log_decision(
                decision_data={
                    "decision_id": decision_id,
                    "run_id": run_id,
                    "step_index": step_index,
                    "day": current_obs.time.day,
                    "hour": current_obs.time.hour,
                    "minute": current_obs.time.minute,
                    "state_hash": pre_hash,
                    "next_state_hash": step_res.state_hash,
                    "observation_json": json.dumps(current_obs.to_dict()),
                    "raw_model_output": raw_output,
                    "parsed_action_json": json.dumps(proposal.model_dump(exclude_none=True))
                    if proposal
                    else None,
                    "action_name": proposal.action if proposal else "unknown",
                    "is_schema_valid": is_schema_valid,
                    "is_env_valid": is_env_valid,
                    "error_category": err_cat,
                    "error_type": err_type,
                    "error_message": err_msg,
                    "execution_minutes": step_res.execution_minutes,
                    "finish_reason": metadata.finish_reason,
                    "was_truncated": int(metadata.was_truncated),
                    "generation_attempt": metadata.generation_attempt,
                    "attempt_count": metadata.attempt_count,
                    "first_pass_valid": int(first_pass_valid),
                    "final_valid": int(final_valid),
                    "recovered": int(metadata.recovered),
                    "first_failure_type": metadata.first_failure_type,
                },
                trace_data={
                    "decision_id": decision_id,
                    "run_id": run_id,
                    "situation_summary": proposal.trace.situation_summary
                    if proposal and proposal.trace
                    else "",
                    "current_priority": proposal.trace.current_priority
                    if proposal and proposal.trace
                    else "",
                    "options_considered": json.dumps(proposal.trace.options_considered)
                    if proposal and proposal.trace
                    else "[]",
                    "chosen_action": proposal.action if proposal else "",
                    "decision_rationale": proposal.trace.decision_rationale
                    if proposal and proposal.trace
                    else "",
                    "expected_result": proposal.trace.expected_result
                    if proposal and proposal.trace
                    else "",
                    "confidence": proposal.trace.confidence if proposal and proposal.trace else 1.0,
                    "provider_reasoning": getattr(agent, "last_reasoning", "") or None,
                },
                runtime_data={
                    "decision_id": decision_id,
                    "run_id": run_id,
                    "model_load_ms": 0.0,
                    "model_warmup_ms": warmup_ms if step_index == 1 else 0.0,
                    "model_resident": int(bool(getattr(agent, "model_resident", False))),
                    "api_calls": metadata.attempt_count if hasattr(agent, "runtime") else 0,
                    "ttft_ms": 0.0,
                    "generation_ms": generation_ms,
                    "schema_validation_ms": schema_validation_ms,
                    "total_decision_ms": total_decision_ms,
                    "input_tokens": metadata.input_tokens,
                    "output_tokens": metadata.total_output_tokens,
                    "reasoning_tokens": metadata.reasoning_tokens,
                    "json_tokens": metadata.json_tokens,
                    "total_tokens": metadata.total_tokens,
                    "simulation_ms": simulation_ms,
                    "logging_ms": logging_ms,
                    "other_ms": max(
                        0.0,
                        total_decision_ms - generation_ms - schema_validation_ms,
                    ),
                    "ram_peak_mb": 0.0,
                    "vram_peak_mb": 0.0,
                    "cost_usd": 0.0,
                },
            )

            self.logger.log_event(
                run_id=run_id,
                event_type="ACTION_COMPLETE",
                simulation_minute=env.state.total_minutes,
                details={
                    "step_index": step_index,
                    "action": proposal.action if proposal else "unknown",
                    "execution_minutes": step_res.execution_minutes,
                    "success": step_res.success,
                },
                state_hash=step_res.state_hash,
            )

            if consecutive_failures >= max_consecutive_failures:
                aborted = True
                termination_reason = f"max_consecutive_failures={max_consecutive_failures}"
                self.logger.log_event(
                    run_id=run_id,
                    event_type="EPISODE_TERMINATED",
                    simulation_minute=env.state.total_minutes,
                    details={
                        "reason": termination_reason,
                        "consecutive_failures": consecutive_failures,
                        "step_index": step_index,
                    },
                    state_hash=step_res.state_hash,
                )
                break

        if live_reporter:
            live_reporter.stop()

        ended_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        survived = (not env.terminated) and (not aborted)
        sim_days = env.state.total_minutes / 1440.0
        avg_health = sum(health_samples) / len(health_samples)
        min_health = min(health_samples)
        avg_happiness = sum(happiness_samples) / len(happiness_samples)

        self.logger.log_outcome(
            {
                "run_id": run_id,
                "survived": int(survived),
                "simulated_days": round(sim_days, 2),
                "final_health": round(env.state.pet.health, 1),
                "min_health": round(min_health, 1),
                "avg_health": round(avg_health, 1),
                "final_happiness": round(env.state.pet.happiness, 1),
                "avg_happiness": round(avg_happiness, 1),
                "final_money": env.state.agent.money,
                "final_energy": int(round(env.state.agent.energy)),
                "total_income": total_income,
                "total_spending": total_spending,
                "jobs_completed": jobs_completed,
                "jobs_failed": jobs_failed,
                "termination_reason": termination_reason,
            }
        )
        self.logger.log_event(
            run_id=run_id,
            event_type="SIMULATION_COMPLETED",
            simulation_minute=env.state.total_minutes,
            details={
                "survived": survived,
                "duration_minutes": env.state.total_minutes,
                "episode_wall_time_ms": (time.perf_counter() - episode_started) * 1000.0,
                "termination_reason": termination_reason,
                "aborted": aborted,
            },
            state_hash=env.state.compute_hash(),
        )
        self.logger.flush()

        file_logger.log_summary(
            survived=survived,
            simulated_days=sim_days,
            final_health=env.state.pet.health,
            final_money=env.state.agent.money,
        )

        return BenchmarkMetricsCalculator(db_path=self.db_path).calculate_run_metrics(run_id)

    def close(self):
        for agent in self._agents.values():
            agent.close()
        self.logger.stop()
