"""Replay canonical accepted actions and verify every transition under its contract."""
import json
from tamabench.env.core import TamaEnv
from tamabench.env.time_engine import BenchmarkMode
from tamabench.logging.database import DatabaseStore
from tamabench.schemas.actions import ActionProposal


class ReplayEngine:
    def __init__(self, db_path="tamabench_results.db"):
        self.db = DatabaseStore(db_path)

    def replay_run(self, run_id):
        run = self.db.get_run(run_id)
        if run is None:
            return False, [f"Run ID '{run_id}' not found"]
        if run["environment_version"] != "2.0.0":
            return False, ["Replay requires the original environment version; v1 results cannot be verified with v2 rules"]
        if run["status"] in {"running", "legacy", "infrastructure_failed", "interrupted"}:
            return False, [f"Run status '{run['status']}' is incomplete; no complete-episode verification claimed"]
        env = TamaEnv(mode=BenchmarkMode(run["mode"]))
        env.reset(seed=run["seed"], scenario_id=run["scenario_id"], scenario_version=run["scenario_version"],
                  max_simulated_minutes=run["max_simulated_minutes"])
        mismatches = []
        income = spending = jobs = jobs_failed = 0
        decisions = self.db.get_run_decisions(run_id)
        for step in decisions:
            number = step["step_index"]
            if env.state.compute_hash() != step["state_hash"]:
                mismatches.append(f"Step {number} pre-state mismatch")
            if step["is_schema_valid"]:
                if not step["parsed_action_json"]:
                    mismatches.append(f"Step {number} missing accepted action")
                    continue
                try:
                    action = ActionProposal.model_validate(json.loads(step["parsed_action_json"]))
                    before = env.observe()
                    result = env.step(action)
                    if action.action == "work":
                        if result.completed:
                            jobs += 1
                            income += next(j.reward for j in before.jobs_available if j.id == action.job_id)
                        else:
                            jobs_failed += 1
                    if action.action == "buy" and result.success:
                        spending += next(i.cost for i in before.shop_items_available if i.item == action.item) * action.amount
                except Exception as exc:
                    mismatches.append(f"Step {number} cannot replay action: {type(exc).__name__}")
                    continue
                if result.success != bool(step["is_env_valid"]):
                    mismatches.append(f"Step {number} acceptance mismatch")
                if result.execution_minutes != step["execution_minutes"]:
                    mismatches.append(f"Step {number} elapsed-time mismatch")
                if result.completed != bool(step["action_completed"]):
                    mismatches.append(f"Step {number} completion mismatch")
            elif step["parsed_action_json"]:
                rejected = json.loads(step["parsed_action_json"])
                if rejected.get("action") == "work":
                    jobs_failed += 1
            if env.state.compute_hash() != step["next_state_hash"]:
                mismatches.append(f"Step {number} post-state mismatch")
        if env.state.total_minutes != run["simulated_duration_minutes"]:
            mismatches.append("Final duration mismatch")
        if bool(run["survived"]) != (run["status"] == "completed" and env.horizon_reached and not env.terminated):
            mismatches.append("Final survival mismatch")
        with self.db._get_connection() as conn:
            outcome = conn.execute("SELECT * FROM outcomes WHERE run_id=?", (run_id,)).fetchone()
        if outcome is None:
            mismatches.append("Missing final outcome")
        else:
            expected = {"final_health": env.state.pet.health, "final_money": env.state.agent.money,
                        "final_happiness": env.state.pet.happiness, "final_energy": env.state.agent.energy,
                        "avg_health": env.statistics.avg_health, "min_health": env.statistics.min_health,
                        "avg_happiness": env.statistics.avg_happiness,
                        "health_integral": env.statistics.health_integral,
                        "happiness_integral": env.statistics.happiness_integral,
                        "total_income": income, "total_spending": spending,
                        "jobs_completed": jobs, "jobs_failed": jobs_failed,
                        "simulated_minutes": env.state.total_minutes, "simulated_days": env.state.total_minutes/1440,
                        "survived": run["survived"]}
            if outcome["status"] != run["status"] or outcome["termination_reason"] != run["termination_reason"]:
                mismatches.append("Final status/reason mismatch")
            for key, actual in expected.items():
                if outcome[key] is None or abs(float(outcome[key]) - actual) > 1e-7:
                    mismatches.append(f"Final {key} mismatch")
        return not mismatches, mismatches
