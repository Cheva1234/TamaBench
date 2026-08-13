"""Replay Engine for TamaBench V1.

Re-executes recorded episode decisions against a fresh TamaEnv with identical seed,
verifying state_hash equality at every step to validate simulator determinism.
"""

import json
from typing import Tuple
from tamabench.env.core import TamaEnv
from tamabench.logging.database import DatabaseStore


class ReplayEngine:
    def __init__(self, db_path: str = "tamabench_results.db"):
        self.db = DatabaseStore(db_path=db_path)

    def replay_run(self, run_id: str) -> Tuple[bool, list[str]]:
        """Replays all decisions for `run_id` and checks state_hash consistency."""
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            run_row = cursor.fetchone()
            if not run_row:
                return False, [f"Run ID '{run_id}' not found in database."]

        seed = run_row["seed"]
        scenario_id = run_row["scenario_id"]
        scenario_version = run_row["scenario_version"]

        env = TamaEnv()
        env.reset(seed=seed, scenario_id=scenario_id, scenario_version=scenario_version)

        decisions = self.db.get_run_decisions(run_id)
        mismatches: list[str] = []

        for step in decisions:
            expected_pre_hash = step["state_hash"]
            expected_next_hash = step["next_state_hash"]
            step_index = step["step_index"]

            # 1. Verify pre-action hash
            current_hash = env.state.compute_hash()
            if current_hash != expected_pre_hash:
                mismatches.append(
                    f"Step {step_index} pre-hash mismatch! Expected {expected_pre_hash}, got {current_hash}"
                )

            # Parse action proposal
            parsed_action_json = step["parsed_action_json"]
            if not parsed_action_json:
                continue

            action_data = json.loads(parsed_action_json)

            # Re-step action
            step_res = env.step(action_data)

            # 2. Verify post-action hash
            if expected_next_hash and step_res.state_hash != expected_next_hash:
                mismatches.append(
                    f"Step {step_index} post-hash mismatch! Expected {expected_next_hash}, got {step_res.state_hash}"
                )

        replay_success = len(mismatches) == 0
        return replay_success, mismatches
