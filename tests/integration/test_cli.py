"""Integration tests for TamaBench CLI commands."""

import pytest
from click.testing import CliRunner
from tamabench.cli import cli
from tamabench.logging.database import DatabaseStore


@pytest.mark.integration
def test_cli_run_and_replay(tmp_path):
    db_file = str(tmp_path / "cli_test.db")
    event_file = str(tmp_path / "cli_test_events.jsonl")

    runner = CliRunner()

    # 1. Run episode via CLI
    res_run = runner.invoke(
        cli,
        [
            "run",
            "--agent",
            "rule",
            "--episodes",
            "1",
            "--seed-start",
            "42",
            "--display",
            "compact",
            "--db-path",
            db_file,
            "--event-path",
            event_file,
        ],
    )
    assert res_run.exit_code == 0
    assert "Seed #42" in res_run.output

    # Get run_id from DB
    db = DatabaseStore(db_path=db_file)
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT run_id FROM runs LIMIT 1")
        run_row = cursor.fetchone()
        assert run_row is not None
        run_id = run_row["run_id"]

    # 2. Replay run via CLI
    res_replay = runner.invoke(
        cli,
        [
            "replay",
            "--run-id",
            run_id,
            "--db-path",
            db_file,
        ],
    )
    assert res_replay.exit_code == 0
    assert "Replay successful" in res_replay.output
