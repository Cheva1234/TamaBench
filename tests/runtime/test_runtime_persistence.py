from tamabench.logging.database import DatabaseStore


def test_database_persists_attempt_and_runtime_reliability_fields(tmp_path):
    db = DatabaseStore(str(tmp_path / "runtime.db"))
    db.record_run(
        {
            "run_id": "run_1",
            "seed": 1,
            "scenario_id": "standard_v1",
            "scenario_version": 1,
            "benchmark_version": "1.1.0",
            "environment_version": "1.0.0",
            "mode": "accelerated",
            "agent_type": "RawLLM(fake)",
            "model_name": "fake",
            "schema_mode": "raw_json",
            "started_at": "now",
            "ended_at": None,
            "simulated_duration_minutes": 0,
            "survived": 0,
        }
    )
    db.record_decision(
        {
            "decision_id": "dec_1",
            "run_id": "run_1",
            "step_index": 1,
            "day": 1,
            "hour": 0,
            "minute": 0,
            "state_hash": "pre",
            "next_state_hash": "post",
            "observation_json": "{}",
            "raw_model_output": "{}",
            "parsed_action_json": "{}",
            "action_name": "wait",
            "is_schema_valid": 1,
            "is_env_valid": 1,
            "error_category": None,
            "error_type": None,
            "error_message": None,
            "execution_minutes": 30,
            "finish_reason": "stop",
            "was_truncated": 0,
            "generation_attempt": 2,
            "attempt_count": 2,
            "first_pass_valid": 0,
            "final_valid": 1,
            "recovered": 1,
            "first_failure_type": "OUTPUT_TRUNCATED",
        }
    )
    db.record_runtime_metrics(
        {
            "decision_id": "dec_1",
            "run_id": "run_1",
            "model_load_ms": 12.0,
            "generation_ms": 34.0,
            "total_decision_ms": 50.0,
            "input_tokens": 100,
            "output_tokens": 20,
            "reasoning_tokens": 15,
            "json_tokens": 5,
            "total_tokens": 120,
            "model_resident": 1,
            "api_calls": 2,
            "model_warmup_ms": 12.0,
        }
    )

    decision = db.get_run_decisions("run_1")[0]
    with db._get_connection() as conn:
        runtime = conn.execute(
            "SELECT * FROM runtime_metrics WHERE decision_id = ?", ("dec_1",)
        ).fetchone()

    assert decision["finish_reason"] == "stop"
    assert decision["was_truncated"] == 0
    assert decision["attempt_count"] == 2
    assert decision["recovered"] == 1
    assert runtime["reasoning_tokens"] == 15
    assert runtime["json_tokens"] == 5
    assert runtime["model_resident"] == 1
