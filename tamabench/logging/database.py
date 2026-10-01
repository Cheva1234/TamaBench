"""SQLite Database Storage for TamaBench V1 with WAL mode support."""

import json
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Optional


class DatabaseStore:
    def __init__(self, db_path: str = "tamabench_results.db"):
        self.db_path = db_path
        Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        with self._get_connection() as conn:
            self._columns = {table: {row[1] for row in conn.execute(f"PRAGMA table_info({table})")} for table in ("runs", "decisions", "decision_traces", "runtime_metrics", "outcomes")}
        self._sql_cache = {}

    def connect(self) -> sqlite3.Connection:
        """A caller-owned connection; one is reused by the writer thread."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _get_connection(self):
        conn = self.connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
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
                finish_reason TEXT,
                was_truncated INTEGER NOT NULL DEFAULT 0,
                generation_attempt INTEGER NOT NULL DEFAULT 1,
                attempt_count INTEGER NOT NULL DEFAULT 1,
                first_pass_valid INTEGER NOT NULL DEFAULT 1,
                final_valid INTEGER NOT NULL DEFAULT 1,
                recovered INTEGER NOT NULL DEFAULT 0,
                first_failure_type TEXT,
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
                reasoning_tokens INTEGER DEFAULT 0,
                json_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                ram_peak_mb REAL DEFAULT 0,
                vram_peak_mb REAL DEFAULT 0,
                cost_usd REAL DEFAULT 0,
                model_resident INTEGER NOT NULL DEFAULT 0,
                api_calls INTEGER DEFAULT 0,
                model_warmup_ms REAL DEFAULT 0,
                simulation_ms REAL DEFAULT 0,
                logging_ms REAL DEFAULT 0,
                other_ms REAL DEFAULT 0,
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
            self._ensure_columns(conn)

    @staticmethod
    def _ensure_columns(conn: sqlite3.Connection):
        """Add V1.1 columns to databases created by V1 without destructive migration."""
        migrations = {
            "runs": {
                "experiment_id": "TEXT", "config_hash": "TEXT", "config_json": "TEXT",
                "status": "TEXT NOT NULL DEFAULT 'legacy'", "termination_reason": "TEXT",
                "max_simulated_minutes": "INTEGER", "episode_wall_time_ms": "REAL",
                "warmup_api_calls": "INTEGER DEFAULT 0", "warmup_input_tokens": "INTEGER DEFAULT 0",
                "warmup_output_tokens": "INTEGER DEFAULT 0", "warmup_ms": "REAL DEFAULT 0",
                "generation_api_calls": "INTEGER", "generation_input_tokens": "INTEGER",
                "generation_output_tokens": "INTEGER", "generation_ms": "REAL", "token_usage_complete": "INTEGER DEFAULT 1",
            },
            "outcomes": {
                "status": "TEXT NOT NULL DEFAULT 'legacy'", "termination_reason": "TEXT",
                "simulated_minutes": "INTEGER", "episode_wall_time_ms": "REAL",
                "health_integral": "REAL", "happiness_integral": "REAL",
            },
            "decisions": {
                "action_source": "TEXT DEFAULT 'unknown'", "original_action_json": "TEXT",
                "override_reason": "TEXT", "action_completed": "INTEGER DEFAULT 1",
                "schema_observed": "INTEGER DEFAULT 1", "attempts_json": "TEXT",
                "finish_reason": "TEXT",
                "was_truncated": "INTEGER NOT NULL DEFAULT 0",
                "generation_attempt": "INTEGER NOT NULL DEFAULT 1",
                "attempt_count": "INTEGER NOT NULL DEFAULT 1",
                "first_pass_valid": "INTEGER NOT NULL DEFAULT 1",
                "final_valid": "INTEGER NOT NULL DEFAULT 1",
                "recovered": "INTEGER NOT NULL DEFAULT 0",
                "first_failure_type": "TEXT",
            },
            "runtime_metrics": {
                "reasoning_tokens": "INTEGER DEFAULT 0",
                "json_tokens": "INTEGER DEFAULT 0",
                "total_tokens": "INTEGER DEFAULT 0",
                "model_resident": "INTEGER NOT NULL DEFAULT 0",
                "api_calls": "INTEGER DEFAULT 0",
                "model_warmup_ms": "REAL DEFAULT 0",
                "simulation_ms": "REAL DEFAULT 0",
                "logging_ms": "REAL DEFAULT 0",
                "other_ms": "REAL DEFAULT 0",
            },
        }
        for table, columns in migrations.items():
            existing = {
                row[1]
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for column, definition in columns.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        conn.commit()

    def _write(self, table: str, data: dict[str, Any], conn=None):
        columns = tuple(k for k in data if k in self._columns[table])
        if not columns:
            raise ValueError(f"No recognized fields for {table}")
        primary = "decision_id" if table in {"decisions", "decision_traces", "runtime_metrics"} else "run_id"
        key = (table, columns)
        sql = self._sql_cache.get(key)
        if sql is None:
            update = ",".join(f"{k}=excluded.{k}" for k in columns if k != primary)
            sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)}) ON CONFLICT({primary}) DO UPDATE SET {update}"
            self._sql_cache[key] = sql
        values = tuple(data[k] for k in columns)
        if conn is not None:
            conn.execute(sql, values)
        else:
            with self._get_connection() as connection:
                connection.execute(sql, values)

    def record_run(self, data, conn=None):
        self._write("runs", data, conn)

    def record_decision(self, data, conn=None):
        self._write("decisions", data, conn)

    def record_decision_trace(self, data, conn=None):
        self._write("decision_traces", data, conn)

    def record_runtime_metrics(self, data, conn=None):
        self._write("runtime_metrics", data, conn)

    def record_outcome(self, data, conn=None):
        self._write("outcomes", data, conn)

    def finalize_run(self, run_data, outcome_data, conn=None):
        """Run status and outcome are committed in the same transaction."""
        if conn is None:
            with self._get_connection() as connection:
                self.finalize_run(run_data, outcome_data, connection)
            return
        if outcome_data is not None:
            self.record_outcome(outcome_data, conn)
        allowed = {k: v for k, v in run_data.items() if k in self._columns["runs"] and k != "run_id"}
        assignment = ",".join(f"{k}=?" for k in allowed)
        conn.execute(f"UPDATE runs SET {assignment} WHERE run_id=?", (*allowed.values(), run_data["run_id"]))

    def get_run_decisions(self, run_id: str) -> list[sqlite3.Row]:
        with self._get_connection() as conn:
            return conn.execute("SELECT * FROM decisions WHERE run_id=? ORDER BY step_index", (run_id,)).fetchall()

    def get_run(self, run_id):
        with self._get_connection() as conn:
            return conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()

    def list_runs(self, experiment_id=None):
        with self._get_connection() as conn:
            if experiment_id:
                return conn.execute("SELECT * FROM runs WHERE experiment_id=? ORDER BY started_at,run_id", (experiment_id,)).fetchall()
            return conn.execute("SELECT * FROM runs ORDER BY started_at,run_id").fetchall()
