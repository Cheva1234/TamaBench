"""Calibration Test: High-Scale Logger & SQLite WAL Stress Test.

Simulates 100,000 synthetic events and decisions across parallel worker threads,
verifying zero missing events, zero SQLite lock failures, and zero JSONL corruptions.
"""

import json
import os
import threading
import pytest
from tamabench.logging.logger_process import LoggerProcess
from tamabench.logging.database import DatabaseStore


@pytest.mark.calibration
def test_logger_100k_synthetic_events_stress(tmp_path):
    db_file = str(tmp_path / "stress_test.db")
    event_file = str(tmp_path / "stress_events.jsonl")

    logger = LoggerProcess(db_path=db_file, event_path=event_file)
    logger.start()

    run_id = "run_stress_100k"
    logger.log_run({
        "run_id": run_id,
        "seed": 42,
        "scenario_id": "standard_v1",
        "scenario_version": 1,
        "benchmark_version": "0.1.0",
        "environment_version": "0.1.0",
        "mode": "logical",
        "agent_type": "StressWorkerAgent",
        "model_name": "synthetic",
        "schema_mode": "raw_json",
        "started_at": "2026-08-13T00:00:00Z",
        "ended_at": None,
        "simulated_duration_minutes": 100000,
        "survived": 1,
    })

    num_threads = 10
    decisions_per_thread = 10000
    total_expected_decisions = num_threads * decisions_per_thread

    def worker_task(thread_id: int):
        for i in range(decisions_per_thread):
            step_idx = (thread_id * decisions_per_thread) + i + 1
            dec_id = f"dec_{run_id}_{step_idx:06d}"

            logger.log_decision(
                decision_data={
                    "decision_id": dec_id,
                    "run_id": run_id,
                    "step_index": step_idx,
                    "day": (step_idx // 1440) + 1,
                    "hour": (step_idx % 1440) // 60,
                    "minute": step_idx % 60,
                    "state_hash": f"sha256:pre_{step_idx}",
                    "next_state_hash": f"sha256:post_{step_idx}",
                    "observation_json": "{}",
                    "raw_model_output": '{"action": "wait"}',
                    "parsed_action_json": '{"action": "wait"}',
                    "action_name": "wait",
                    "is_schema_valid": 1,
                    "is_env_valid": 1,
                    "error_category": None,
                    "error_type": None,
                    "error_message": None,
                    "execution_minutes": 30,
                }
            )
            logger.log_event(
                run_id=run_id,
                event_type="SYNTHETIC_EVENT",
                simulation_minute=step_idx,
                details={"thread": thread_id, "step": step_idx},
                state_hash=f"sha256:post_{step_idx}",
            )

    threads = []
    for t in range(num_threads):
        thread = threading.Thread(target=worker_task, args=(t,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    logger.stop()

    # Integrity Audit
    db = DatabaseStore(db_path=db_file)
    decisions = db.get_run_decisions(run_id)
    assert len(decisions) == total_expected_decisions, f"Missing decisions! Expected {total_expected_decisions}, recorded {len(decisions)}"

    # Audit JSONL event count
    line_count = 0
    with open(event_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                json.loads(line)  # Validates JSON formatting
                line_count += 1

    assert line_count == total_expected_decisions, f"JSONL missing lines! Expected {total_expected_decisions}, recorded {line_count}"
