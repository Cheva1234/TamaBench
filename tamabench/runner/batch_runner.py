"""Accelerated Batch Benchmark Runner for TamaBench V1."""

import datetime
import json
import uuid
from typing import Optional
from tamabench.agents.base import BaseAgent
from tamabench.env.core import TamaEnv
from tamabench.env.time_engine import BenchmarkMode
from tamabench.logging.logger_process import LoggerProcess
from tamabench.metrics.calculator import BenchmarkMetricsCalculator, EpisodeMetrics
from tamabench.evaluation.reference_policy import ReferencePolicyEvaluator


from tamabench.metrics.live_reporter import LiveReporter
from tamabench.logging.file_logger import FileLogger


class BatchRunner:
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

    def run_episode(
        self,
        agent: BaseAgent,
        seed: int = 42,
        max_simulated_minutes: int = 4320,  # Default 3 simulated days
        scenario_id: str = "standard_v1",
        scenario_version: int = 1,
        live_monitor: bool = False,
    ) -> EpisodeMetrics:
        """Executes a single benchmark episode."""
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        file_logger = FileLogger(run_id=run_id)

        env = TamaEnv(mode=self.mode)
        obs = env.reset(seed=seed, scenario_id=scenario_id, scenario_version=scenario_version)

        live_reporter = LiveReporter(model_name=getattr(agent, "model_name", agent.name), seed=seed) if live_monitor else None
        if live_reporter:
            live_reporter.start()

        # Log Run Initialization
        self.logger.log_run({
            "run_id": run_id,
            "seed": seed,
            "scenario_id": scenario_id,
            "scenario_version": scenario_version,
            "benchmark_version": "0.1.0",
            "environment_version": "0.1.0",
            "mode": self.mode.value,
            "agent_type": agent.name,
            "model_name": getattr(agent, "model_name", agent.name),
            "schema_mode": getattr(agent, "schema_mode", "raw_json"),
            "started_at": started_at,
            "ended_at": None,
            "simulated_duration_minutes": 0,
            "survived": 0,
        })

        self.logger.log_event(
            run_id=run_id,
            event_type="SIMULATION_STARTED",
            simulation_minute=0,
            details={"seed": seed, "agent": agent.name},
            state_hash=obs.state_hash,
        )

        step_index = 0
        total_income = 0
        total_spending = 0
        jobs_completed = 0
        jobs_failed = 0
        health_samples = [obs.pet.health]
        happiness_samples = [obs.pet.happiness]

        while env.state.total_minutes < max_simulated_minutes and not env.terminated:
            step_index += 1
            current_obs = env.observe()
            pre_hash = current_obs.state_hash

            # 1. Agent Select Action
            raw_output, proposal, schema_err = agent.select_action(current_obs)

            # Evaluate against reference policy
            ref_eval = ReferencePolicyEvaluator.evaluate_decision(env.state, proposal)

            # 2. Step Environment
            if proposal is not None:
                step_res = env.step(proposal)
            else:
                step_res = env.step(raw_output)

            post_obs = step_res.observation
            post_hash = step_res.state_hash

            health_samples.append(post_obs.pet.health)
            happiness_samples.append(post_obs.pet.happiness)

            gen_ms = getattr(agent, "last_compute", None).generation_ms if getattr(agent, "last_compute", None) else 0.0
            if live_reporter:
                live_reporter.update(current_obs, step_index, proposal, step_res, latency_ms=gen_ms)

            thinking_str = getattr(agent, "last_reasoning", "")
            # Log to FileLogger (standalone reasoning txt and replay jsonl)
            file_logger.log_step(
                step_index=step_index,
                obs=current_obs,
                raw_output=raw_output,
                proposal=proposal,
                step_result=step_res,
                latency_ms=gen_ms,
                thinking_process=thinking_str,
            )

            # Track economy stats
            if proposal and proposal.action == "work" and step_res.success:
                job_id = proposal.job_id or ""
                if job_id == "cafe_shift":
                    total_income += 40
                elif job_id == "delivery":
                    total_income += 90
                elif job_id == "freelance":
                    total_income += 20
                jobs_completed += 1

            if proposal and proposal.action == "buy" and step_res.success:
                item_name = proposal.item or ""
                amt = proposal.amount or 1
                cost_per = 20 if item_name == "food" else 50
                total_spending += cost_per * amt

            # 3. Log Decision & Decision Trace
            decision_id = f"dec_{run_id}_{step_index:04d}"
            
            is_schema_valid = 1 if (schema_err is None and step_res.error is None or step_res.error.category != "SCHEMA") else 0
            is_env_valid = 1 if step_res.success else 0

            err_cat = step_res.error.category.value if step_res.error else (schema_err.category.value if schema_err else None)
            err_type = step_res.error.error_type.value if step_res.error else (schema_err.error_type.value if schema_err else None)
            err_msg = step_res.error.message if step_res.error else (schema_err.message if schema_err else None)

            self.logger.log_decision(
                decision_data={
                    "decision_id": decision_id,
                    "run_id": run_id,
                    "step_index": step_index,
                    "day": current_obs.time.day,
                    "hour": current_obs.time.hour,
                    "minute": current_obs.time.minute,
                    "state_hash": pre_hash,
                    "next_state_hash": post_hash,
                    "observation_json": json.dumps(current_obs.to_dict()),
                    "raw_model_output": raw_output,
                    "parsed_action_json": json.dumps(proposal.model_dump(exclude_none=True)) if proposal else None,
                    "action_name": proposal.action if proposal else "unknown",
                    "is_schema_valid": is_schema_valid,
                    "is_env_valid": is_env_valid,
                    "error_category": err_cat,
                    "error_type": err_type,
                    "error_message": err_msg,
                    "execution_minutes": step_res.execution_minutes,
                },
                trace_data={
                    "decision_id": decision_id,
                    "run_id": run_id,
                    "situation_summary": proposal.trace.situation_summary if proposal and proposal.trace else "",
                    "current_priority": proposal.trace.current_priority if proposal and proposal.trace else "",
                    "options_considered": json.dumps(proposal.trace.options_considered) if proposal and proposal.trace else "[]",
                    "chosen_action": proposal.action if proposal else "",
                    "decision_rationale": proposal.trace.decision_rationale if proposal and proposal.trace else "",
                    "expected_result": proposal.trace.expected_result if proposal and proposal.trace else "",
                    "confidence": proposal.trace.confidence if proposal and proposal.trace else 1.0,
                    "provider_reasoning": None,
                },
                runtime_data={
                    "decision_id": decision_id,
                    "run_id": run_id,
                    "model_load_ms": 0.0,
                    "ttft_ms": 0.0,
                    "generation_ms": getattr(agent, "last_compute", None).generation_ms if getattr(agent, "last_compute", None) else 0.0,
                    "schema_validation_ms": 0.0,
                    "total_decision_ms": getattr(agent, "last_compute", None).total_decision_ms if getattr(agent, "last_compute", None) else 0.0,
                    "input_tokens": 2150,
                    "output_tokens": len(raw_output) // 4,
                    "ram_peak_mb": 0.0,
                    "vram_peak_mb": 0.0,
                    "cost_usd": 0.0,
                },
            )

        if live_reporter:
            live_reporter.stop()

        # Episode Ended
        ended_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        survived = not env.terminated
        sim_days = env.state.total_minutes / 1440.0
        avg_h = sum(health_samples) / len(health_samples)
        min_h = min(health_samples)
        avg_hap = sum(happiness_samples) / len(happiness_samples)

        # Record Outcome
        self.logger.log_outcome({
            "run_id": run_id,
            "survived": 1 if survived else 0,
            "simulated_days": round(sim_days, 2),
            "final_health": round(env.state.pet.health, 1),
            "min_health": round(min_h, 1),
            "avg_health": round(avg_h, 1),
            "final_happiness": round(env.state.pet.happiness, 1),
            "avg_happiness": round(avg_hap, 1),
            "final_money": env.state.agent.money,
            "final_energy": env.state.agent.energy,
            "total_income": total_income,
            "total_spending": total_spending,
            "jobs_completed": jobs_completed,
            "jobs_failed": jobs_failed,
        })

        self.logger.log_event(
            run_id=run_id,
            event_type="SIMULATION_COMPLETED",
            simulation_minute=env.state.total_minutes,
            details={"survived": survived, "duration_minutes": env.state.total_minutes},
            state_hash=env.state.compute_hash(),
        )

        self.logger.flush()

        file_logger.log_summary(
            survived=survived,
            simulated_days=sim_days,
            final_health=env.state.pet.health,
            final_money=env.state.agent.money,
        )

        calc = BenchmarkMetricsCalculator(db_path=self.db_path)
        return calc.calculate_run_metrics(run_id)

    def close(self):
        self.logger.stop()
