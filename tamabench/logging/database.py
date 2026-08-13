"""SQLite Database Storage for TamaBench V1 with WAL mode support."""

import json
import sqlite3
from typing import Any, Optional


class DatabaseStore:
    def __init__(self, db_path: str = "tamabench_results.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Runs Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                seed INTEGER NOT NULL,
                scenario_id TEXT NOT NULL,
                scenario_version INTEGER NOT NULL,
                benchmark_version TEXT NOT NULL,
                environment_version TEXT NOT NULL,
                mode TEXT NOT NULL,
                agent_type TEXT NOT NULL,
                model_name TEXT,
                schema_mode TEXT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                simulated_duration_minutes INTEGER DEFAULT 0,
                survived INTEGER DEFAULT 0
            );
            """)

            # 2. Decisions Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                day INTEGER NOT NULL,
                hour INTEGER NOT NULL,
                minute INTEGER NOT NULL,
                state_hash TEXT NOT NULL,
                next_state_hash TEXT,
                observation_json TEXT NOT NULL,
                raw_model_output TEXT,
                parsed_action_json TEXT,
                action_name TEXT,
                is_schema_valid INTEGER NOT NULL,
                is_env_valid INTEGER NOT NULL,
                error_category TEXT,
                error_type TEXT,
                error_message TEXT,
                execution_minutes INTEGER DEFAULT 0,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            """)

            # 3. Decision Traces & Provider Reasoning Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS decision_traces (
                decision_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                situation_summary TEXT,
                current_priority TEXT,
                options_considered TEXT,
                chosen_action TEXT,
                decision_rationale TEXT,
                expected_result TEXT,
                confidence REAL,
                provider_reasoning TEXT,
                FOREIGN KEY(decision_id) REFERENCES decisions(decision_id),
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            """)

            # 4. Runtime Metrics Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS runtime_metrics (
                decision_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                model_load_ms REAL DEFAULT 0,
                ttft_ms REAL DEFAULT 0,
                generation_ms REAL DEFAULT 0,
                schema_validation_ms REAL DEFAULT 0,
                total_decision_ms REAL DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                ram_peak_mb REAL DEFAULT 0,
                vram_peak_mb REAL DEFAULT 0,
                cost_usd REAL DEFAULT 0,
                FOREIGN KEY(decision_id) REFERENCES decisions(decision_id),
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            """)

            # 5. Outcomes Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                run_id TEXT PRIMARY KEY,
                survived INTEGER NOT NULL,
                simulated_days REAL NOT NULL,
                final_health REAL NOT NULL,
                min_health REAL NOT NULL,
                avg_health REAL NOT NULL,
                final_happiness REAL NOT NULL,
                avg_happiness REAL NOT NULL,
                final_money INTEGER NOT NULL,
                final_energy INTEGER NOT NULL,
                total_income INTEGER DEFAULT 0,
                total_spending INTEGER DEFAULT 0,
                jobs_completed INTEGER DEFAULT 0,
                jobs_failed INTEGER DEFAULT 0,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            """)

            conn.commit()

    def record_run(self, run_data: dict[str, Any]):
        with self._get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO runs (
                run_id, seed, scenario_id, scenario_version, benchmark_version,
                environment_version, mode, agent_type, model_name, schema_mode,
                started_at, ended_at, simulated_duration_minutes, survived
            ) VALUES (
                :run_id, :seed, :scenario_id, :scenario_version, :benchmark_version,
                :environment_version, :mode, :agent_type, :model_name, :schema_mode,
                :started_at, :ended_at, :simulated_duration_minutes, :survived
            );
            """, run_data)

    def record_decision(self, decision_data: dict[str, Any]):
        with self._get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO decisions (
                decision_id, run_id, step_index, day, hour, minute, state_hash,
                next_state_hash, observation_json, raw_model_output, parsed_action_json,
                action_name, is_schema_valid, is_env_valid, error_category, error_type,
                error_message, execution_minutes
            ) VALUES (
                :decision_id, :run_id, :step_index, :day, :hour, :minute, :state_hash,
                :next_state_hash, :observation_json, :raw_model_output, :parsed_action_json,
                :action_name, :is_schema_valid, :is_env_valid, :error_category, :error_type,
                :error_message, :execution_minutes
            );
            """, decision_data)

    def record_decision_trace(self, trace_data: dict[str, Any]):
        with self._get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO decision_traces (
                decision_id, run_id, situation_summary, current_priority,
                options_considered, chosen_action, decision_rationale, expected_result,
                confidence, provider_reasoning
            ) VALUES (
                :decision_id, :run_id, :situation_summary, :current_priority,
                :options_considered, :chosen_action, :decision_rationale, :expected_result,
                :confidence, :provider_reasoning
            );
            """, trace_data)

    def record_runtime_metrics(self, runtime_data: dict[str, Any]):
        with self._get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO runtime_metrics (
                decision_id, run_id, model_load_ms, ttft_ms, generation_ms,
                schema_validation_ms, total_decision_ms, input_tokens, output_tokens,
                ram_peak_mb, vram_peak_mb, cost_usd
            ) VALUES (
                :decision_id, :run_id, :model_load_ms, :ttft_ms, :generation_ms,
                :schema_validation_ms, :total_decision_ms, :input_tokens, :output_tokens,
                :ram_peak_mb, :vram_peak_mb, :cost_usd
            );
            """, runtime_data)

    def record_outcome(self, outcome_data: dict[str, Any]):
        with self._get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO outcomes (
                run_id, survived, simulated_days, final_health, min_health, avg_health,
                final_happiness, avg_happiness, final_money, final_energy,
                total_income, total_spending, jobs_completed, jobs_failed
            ) VALUES (
                :run_id, :survived, :simulated_days, :final_health, :min_health, :avg_health,
                :final_happiness, :avg_happiness, :final_money, :final_energy,
                :total_income, :total_spending, :jobs_completed, :jobs_failed
            );
            """, outcome_data)

    def get_run_decisions(self, run_id: str) -> list[sqlite3.Row]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM decisions WHERE run_id = ? ORDER BY step_index ASC", (run_id,))
            return cursor.fetchall()
