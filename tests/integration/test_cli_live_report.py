from click.testing import CliRunner

from tamabench.cli import cli


def test_live_display_prints_final_benchmark_report(tmp_path):
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "--agent",
            "rule",
            "--display",
            "live",
            "--db-path",
            str(tmp_path / "live.db"),
            "--event-path",
            str(tmp_path / "live.jsonl"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "TamaBench V1 Benchmark Report" in result.output
    assert "Survival Status:" in result.output
