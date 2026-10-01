"""Offline release regressions for truthful results, failure handling and speed paths."""
import json
from dataclasses import asdict
import threading
import time

import pytest
from rich.console import Console
from io import StringIO

from tamabench.agents.base import BaseAgent, DecisionMetadata
from tamabench.agents.raw_llm_agent import RawLLMAgent
from tamabench.env.time_engine import BenchmarkMode
from tamabench.logging.database import DatabaseStore
from tamabench.logging.logger_process import LoggerProcess, StorageError
from tamabench.logging.replay import ReplayEngine
from tamabench.metrics.report_v1 import ReportV1Generator
from tamabench.runner.batch_runner import BatchRunner
from tamabench.runtime.model_runtime import ModelRuntime, ProviderError, InferenceBudgetExceeded
from tamabench.schemas.actions import ActionProposal
from tamabench.schemas.errors import BenchmarkError, ErrorCategory, ErrorType


class WaitAgent(BaseAgent):
    def __init__(self, minutes=30):
        super().__init__('wait_fixture')
        self.minutes = minutes

    def select_action(self, obs):
        proposal = ActionProposal(action='wait', minutes=self.minutes)
        return proposal.model_dump_json(exclude_none=True), proposal, None


def runner(tmp_path, name='case'):
    return BatchRunner(str(tmp_path/f'{name}.sqlite'), str(tmp_path/f'{name}.jsonl'),
                       mode=BenchmarkMode.ACCELERATED, log_dir=str(tmp_path/f'{name}-traces'))


def test_completed_run_is_finalized_and_replayable(tmp_path):
    with runner(tmp_path) as run:
        metric = run.run_episode(WaitAgent(300), max_simulated_minutes=10)
    row = DatabaseStore(str(tmp_path/'case.sqlite')).get_run(metric.run_id)
    assert row['status'] == 'completed'
    assert row['survived'] == 1
    assert row['simulated_duration_minutes'] == 10
    assert row['ended_at']
    assert metric.simulated_days == 10/1440
    assert metric.critical_decision_acc is None
    assert metric.productive_action_rate is None
    assert ReplayEngine(str(tmp_path/'case.sqlite')).replay_run(metric.run_id) == (True, [])


def test_rejected_valid_looking_output_never_executes(tmp_path):
    class Rejected(WaitAgent):
        def select_action(self, obs):
            self.last_decision = DecisionMetadata(first_pass_valid=False, final_valid=False, was_truncated=True)
            return '{"action":"work","job_id":"cafe_shift"}', None, BenchmarkError(
                category=ErrorCategory.SCHEMA, error_type=ErrorType.OUTPUT_TRUNCATED, message='fixture')
    with runner(tmp_path) as run:
        metric = run.run_episode(Rejected(), max_simulated_minutes=120, max_consecutive_failures=2)
    db = DatabaseStore(str(tmp_path/'case.sqlite'))
    decisions = db.get_run_decisions(metric.run_id)
    assert metric.status == 'invalid_action_abort'
    assert metric.simulated_days == metric.total_income == metric.jobs_completed == 0
    assert all(d['is_env_valid']==0 and d['state_hash']==d['next_state_hash'] for d in decisions)
    assert ReplayEngine(str(tmp_path/'case.sqlite')).replay_run(metric.run_id) == (True, [])


def test_segmenting_identical_trajectory_does_not_change_welfare(tmp_path):
    with runner(tmp_path) as run:
        long = run.run_episode(WaitAgent(600), max_simulated_minutes=600)
        short = run.run_episode(WaitAgent(30), max_simulated_minutes=600)
    assert long.avg_health == short.avg_health
    assert long.avg_happiness == short.avg_happiness
    assert long.min_health == short.min_health
    assert long.score == short.score


def test_policy_steps_do_not_inflate_model_schema_accuracy(tmp_path):
    class Mixed(WaitAgent):
        count = 0
        def select_action(self, obs):
            self.count += 1
            if self.count == 1:
                self.last_decision = DecisionMetadata(action_source='model', first_pass_valid=False, final_valid=False)
                return 'bad', None, BenchmarkError(category=ErrorCategory.SCHEMA, error_type=ErrorType.INVALID_JSON, message='bad')
            self.last_decision = DecisionMetadata(action_source='policy', attempt_count=0)
            return super().select_action(obs)
    with runner(tmp_path) as run:
        metric = run.run_episode(Mixed(), max_simulated_minutes=1)
    assert metric.total_decisions == 2
    assert metric.schema_decisions == metric.model_decisions == metric.policy_decisions == 1
    assert metric.first_pass_schema_acc == metric.final_schema_acc == 0


def test_infrastructure_failure_is_persisted_not_scored_as_schema(tmp_path):
    class Broken(WaitAgent):
        def select_action(self, obs):
            raise ProviderError('offline provider fixture')
    with runner(tmp_path) as run:
        with pytest.raises(ProviderError):
            run.run_episode(Broken())
    db = DatabaseStore(str(tmp_path/'case.sqlite'))
    row = db.list_runs()[0]
    assert row['status'] == 'infrastructure_failed'
    assert row['ended_at']
    assert not db.get_run_decisions(row['run_id'])


def test_keyboard_interrupt_is_persisted_and_cleanup_succeeds(tmp_path):
    class Interrupted(WaitAgent):
        def select_action(self, obs):
            raise KeyboardInterrupt()
    run = runner(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        run.run_episode(Interrupted())
    run.close()
    run.close()
    assert DatabaseStore(str(tmp_path/'case.sqlite')).list_runs()[0]['status'] == 'interrupted'


def test_logger_write_failure_propagates_without_unfinished_tasks(tmp_path, monkeypatch):
    logger = LoggerProcess(str(tmp_path/'error.sqlite'), str(tmp_path/'error.jsonl'))
    def fail(*args, **kwargs):
        raise OSError('injected write failure')
    monkeypatch.setattr(logger.db, 'record_run', fail)
    logger.start()
    logger.log_run({'run_id':'injected'})
    errors = []
    def finish():
        try:
            logger.flush()
        except StorageError as exc:
            errors.append(str(exc))
    thread = threading.Thread(target=finish, daemon=True)
    thread.start(); thread.join(timeout=2)
    assert not thread.is_alive(), 'flush hung after storage error'
    assert errors and logger.msg_queue.unfinished_tasks == 0
    with pytest.raises(StorageError):
        logger.stop()
    assert not logger._worker_thread.is_alive()


def test_repeated_logger_stop_is_safe(tmp_path):
    logger = LoggerProcess(str(tmp_path/'repeat.sqlite'), str(tmp_path/'repeat.jsonl'))
    logger.start(); logger.stop(); logger.stop(); logger.flush()
    assert logger.msg_queue.unfinished_tasks == 0


def test_report_displays_real_fixture_values(tmp_path):
    with runner(tmp_path) as run:
        metric = run.run_episode(WaitAgent(600), max_simulated_minutes=600)
    report = ReportV1Generator(str(tmp_path/'case.sqlite'))
    stream = StringIO(); report.console = Console(file=stream, width=220, color_system=None)
    report.generate_report()
    output = stream.getvalue()
    assert f'{metric.avg_health:.2f}' in output
    assert 'completed' in output
    assert '98.5%' not in output and '$140' not in output
    assert '1/1 survived' in output


def test_hard_decision_budget_records_distinct_terminal_status(tmp_path):
    with runner(tmp_path) as run:
        metric = run.run_episode(WaitAgent(1), max_simulated_minutes=100, run_config={'max_decisions':3})
    assert metric.total_decisions == 3
    assert metric.status == 'budget_exhausted'
    assert metric.simulated_days == 3/1440
    assert not metric.survived
    assert ReplayEngine(str(tmp_path/'case.sqlite')).replay_run(metric.run_id) == (True, [])


class Response:
    def __init__(self, payload):
        self.payload = payload
    def raise_for_status(self):
        pass
    def json(self):
        return self.payload


class Session:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.requests = []
        self.closed = False
    def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return Response(next(self.payloads))
    def close(self):
        self.closed = True


def test_generic_endpoint_has_no_ollama_fields_or_warmup():
    session = Session([{'choices':[{'message':{'content':'{"action":"wait"}'},'finish_reason':'stop'}],
                        'usage':{'prompt_tokens':10,'completion_tokens':5}}])
    runtime = ModelRuntime('fake', api_base='https://example.invalid/v1', api_key='test-only',
                           session=session, backend='openai_compatible')
    assert runtime.warmup() == 0
    result = runtime.generate({'messages':[], 'max_tokens':20})
    assert len(session.requests) == runtime.api_calls == 1
    assert runtime.warmup_calls == 0
    assert 'keep_alive' not in session.requests[0][1]['json']
    assert result.input_tokens == 10 and result.output_tokens == 5
    runtime.close(); runtime.close(); assert session.closed


def test_ollama_uses_native_lifecycle_and_counts_warmup():
    session = Session([{'prompt_eval_count':0,'eval_count':0},
        {'message':{'content':'{"action":"wait"}'},'done_reason':'stop','prompt_eval_count':11,'eval_count':7}, {}])
    runtime = ModelRuntime('fake', session=session)
    runtime.generate({'messages':[], 'max_tokens':30, 'temperature':0.2, 'seed':42})
    assert runtime.api_calls == 2 and runtime.warmup_calls == 1
    assert session.requests[0][0].endswith('/api/generate')
    assert session.requests[1][0].endswith('/api/chat')
    assert session.requests[1][1]['json']['options']['seed'] == 42
    runtime.unload()
    assert runtime.cleanup_calls == 1
    assert session.requests[-1][1]['json']['keep_alive'] == 0
    runtime.close()


def test_request_budget_prevents_request():
    session = Session([])
    runtime = ModelRuntime('fake', session=session, backend='openai_compatible')
    runtime.remaining_calls = 0
    with pytest.raises(InferenceBudgetExceeded):
        runtime.generate({'messages':[]})
    assert not session.requests
    runtime.close()


def test_trace_summary_failure_still_finalizes_authoritative_run(tmp_path, monkeypatch):
    from tamabench.logging.file_logger import FileLogger
    def fail(*args, **kwargs):
        raise OSError('injected full trace disk')
    monkeypatch.setattr(FileLogger, 'log_summary', fail)
    with runner(tmp_path) as run:
        with pytest.raises(OSError):
            run.run_episode(WaitAgent(), max_simulated_minutes=1)
    db = DatabaseStore(str(tmp_path/'case.sqlite'))
    row = db.list_runs()[0]
    assert row['status'] == 'infrastructure_failed'
    assert row['ended_at'] and row['simulated_duration_minutes'] == 1
    with db._get_connection() as connection:
        assert connection.execute('SELECT count(*) FROM outcomes').fetchone()[0] == 1


def test_budget_mid_retry_keeps_observed_invalid_outputs(tmp_path):
    payloads = [
        {'choices':[{'message':{'content':'{"action":"wait","minutes":1}'},'finish_reason':'stop'}], 'usage':{'prompt_tokens':5,'completion_tokens':2}},
        {'choices':[{'message':{'content':'bad'},'finish_reason':'stop'}], 'usage':{'prompt_tokens':5,'completion_tokens':2}},
        {'choices':[{'message':{'content':'bad'},'finish_reason':'stop'}], 'usage':{'prompt_tokens':5,'completion_tokens':2}},
    ]
    runtime = ModelRuntime('fixture', backend='openai_compatible', session=Session(payloads))
    agent = RawLLMAgent(runtime=runtime, max_retries=3)
    with runner(tmp_path) as run:
        metric = run.run_episode(agent, max_simulated_minutes=30, run_config={'max_api_calls':3})
    rows = DatabaseStore(str(tmp_path/'case.sqlite')).get_run_decisions(metric.run_id)
    assert metric.status == 'budget_exhausted'
    assert len(rows) == metric.total_decisions == 2
    assert metric.api_calls == 3
    assert metric.first_pass_schema_acc == metric.final_schema_acc == 50
    assert len(json.loads(rows[1]['attempts_json'])) == 2
    assert rows[1]['schema_observed'] == 1


def test_provider_failure_without_output_is_not_a_schema_observation(tmp_path):
    class FailedSession(Session):
        def post(self, *args, **kwargs):
            import requests
            raise requests.ConnectionError('fixture')
    runtime = ModelRuntime('fixture', backend='openai_compatible', session=FailedSession([]))
    with runner(tmp_path) as run:
        metric = run.run_episode(RawLLMAgent(runtime=runtime), max_simulated_minutes=30)
    assert metric.status == 'infrastructure_failed'
    assert metric.schema_decisions == 0
    assert metric.api_calls == 1
    row = DatabaseStore(str(tmp_path/'case.sqlite')).get_run_decisions(metric.run_id)[0]
    assert row['error_category'] == 'INFRASTRUCTURE' and row['schema_observed'] == 0


def test_logger_start_failure_rejects_producers(tmp_path, monkeypatch):
    logger = LoggerProcess(str(tmp_path/'startup.sqlite'), str(tmp_path/'startup.jsonl'))
    def fail():
        raise OSError('connect failed')
    monkeypatch.setattr(logger.db, 'connect', fail)
    with pytest.raises(StorageError):
        logger.start()
    with pytest.raises(StorageError):
        logger.log_run({'run_id':'too late'})
    with pytest.raises(StorageError):
        logger.flush()
    assert logger.msg_queue.unfinished_tasks == 0


def test_deadline_timeout_is_budget_exhausted(tmp_path):
    class DeadlineSession(Session):
        def post(self, url, **kwargs):
            import requests
            time.sleep(kwargs['timeout'] + 0.002)
            raise requests.ReadTimeout('fixture')
    runtime = ModelRuntime('fixture', backend='openai_compatible', session=DeadlineSession([]))
    with runner(tmp_path) as run:
        metric = run.run_episode(RawLLMAgent(runtime=runtime), max_simulated_minutes=30,
                                  run_config={'max_wall_seconds':0.02})
    assert metric.status == 'budget_exhausted'


def test_missing_usage_stops_token_budgeted_experiment(tmp_path):
    runtime = ModelRuntime('fixture', backend='openai_compatible', session=Session([
        {'choices':[{'message':{'content':'{"action":"wait","minutes":1}'},'finish_reason':'stop'}]}]))
    with runner(tmp_path) as run:
        metric = run.run_episode(RawLLMAgent(runtime=runtime), max_simulated_minutes=30,
                                 run_config={'max_total_tokens':10})
    assert metric.status == 'infrastructure_failed'
    assert metric.token_usage_complete is False
    assert metric.api_calls == 1


def test_replay_rejects_corrupted_reported_outcomes(tmp_path):
    with runner(tmp_path) as run:
        metric = run.run_episode(WaitAgent(10), max_simulated_minutes=10)
    db = DatabaseStore(str(tmp_path/'case.sqlite'))
    with db._get_connection() as conn:
        conn.execute('UPDATE outcomes SET avg_health=12,total_income=999,jobs_completed=99,survived=0 WHERE run_id=?',(metric.run_id,))
    success, mismatches = ReplayEngine(str(tmp_path/'case.sqlite')).replay_run(metric.run_id)
    assert success is False
    assert any('avg_health' in message for message in mismatches)
    assert any('total_income' in message for message in mismatches)
    assert any('survived' in message for message in mismatches)
