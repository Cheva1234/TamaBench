import pytest

from tamabench.metrics.live_reporter import LiveReporter


def test_live_reporter_exposes_idle_and_generating_model_states():
    reporter = LiveReporter()

    reporter.set_model_status("generating")
    assert reporter.model_status == "generating"

    reporter.set_model_status("idle")
    assert reporter.model_status == "idle"

    with pytest.raises(ValueError):
        reporter.set_model_status("running")
