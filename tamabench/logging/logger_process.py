"""Centralized Logger Process for thread-safe asynchronous multi-worker runs."""

import queue
import threading
from typing import Any, Optional
from tamabench.logging.database import DatabaseStore
from tamabench.logging.event_stream import EventStreamLogger


class LoggerProcess:
    def __init__(self, db_path: str = "tamabench_results.db", event_path: str = "tamabench_events.jsonl"):
        self.db = DatabaseStore(db_path=db_path)
        self.event_logger = EventStreamLogger(log_filepath=event_path)
        self.msg_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        self._running = True
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()

    def flush(self):
        self.msg_queue.join()

    def stop(self):
        self.flush()
        self._running = False
        self.msg_queue.put(None)  # Sentinel
        if self._worker_thread:
            self._worker_thread.join(timeout=5)

    def log_run(self, data: dict[str, Any]):
        self.msg_queue.put(("record_run", data))

    def log_decision(
        self,
        decision_data: dict[str, Any],
        trace_data: Optional[dict[str, Any]] = None,
        runtime_data: Optional[dict[str, Any]] = None,
    ):
        self.msg_queue.put(("record_decision", (decision_data, trace_data, runtime_data)))

    def log_event(self, run_id: str, event_type: str, simulation_minute: int, details: dict, state_hash: str):
        self.msg_queue.put(("log_event", (run_id, event_type, simulation_minute, details, state_hash)))

    def log_outcome(self, outcome_data: dict[str, Any]):
        self.msg_queue.put(("record_outcome", outcome_data))

    def _process_queue(self):
        while self._running or not self.msg_queue.empty():
            try:
                item = self.msg_queue.get(timeout=0.2)
                if item is None:
                    break

                cmd, payload = item
                if cmd == "record_run":
                    self.db.record_run(payload)
                elif cmd == "record_decision":
                    d_data, t_data, r_data = payload
                    self.db.record_decision(d_data)
                    if t_data:
                        self.db.record_decision_trace(t_data)
                    if r_data:
                        self.db.record_runtime_metrics(r_data)
                elif cmd == "log_event":
                    run_id, event_type, sim_min, details, s_hash = payload
                    self.event_logger.log_event(run_id, event_type, sim_min, details, s_hash)
                elif cmd == "record_outcome":
                    self.db.record_outcome(payload)

                self.msg_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[LoggerProcess Error]: {e}")
