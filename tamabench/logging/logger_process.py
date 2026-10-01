"""Bounded asynchronous storage writer with transactional batches and error propagation."""

import queue
import threading
import time
from typing import Any

from tamabench.logging.database import DatabaseStore
from tamabench.logging.event_stream import EventStreamLogger


class StorageError(RuntimeError):
    """The authoritative result store could not persist a complete record."""


class LoggerProcess:
    def __init__(self, db_path="tamabench_results.db", event_path="tamabench_events.jsonl", batch_size=256):
        self.db = DatabaseStore(db_path)
        self.event_logger = EventStreamLogger(event_path)
        self.msg_queue = queue.Queue(maxsize=4096)
        self.batch_size = batch_size
        self._worker_thread = None
        self._ready = threading.Event()
        self._running = False
        self._closed = False
        self._failure = None
        self.write_ms = 0.0
        self.transactions = 0

    def start(self):
        if self._closed:
            raise StorageError("Logger is already closed")
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()
        if not self._ready.wait(timeout=10):
            self._failure = StorageError("Storage worker initialization timed out")
            self._closed = True
            self.msg_queue.put(None)
        self._check()

    def _check(self):
        if self._failure is not None:
            raise StorageError(f"Result persistence failed: {self._failure}") from self._failure

    def _put(self, item):
        self._check()
        if self._closed or not self._running:
            raise StorageError("Logger is not running")
        while True:
            self._check()
            try:
                self.msg_queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def flush(self):
        # Poll worker health rather than blocking forever in Queue.join.
        with self.msg_queue.all_tasks_done:
            while self.msg_queue.unfinished_tasks:
                self._check()
                if self._worker_thread is None or not self._worker_thread.is_alive():
                    raise StorageError("Storage worker stopped before all records were acknowledged")
                self.msg_queue.all_tasks_done.wait(timeout=0.1)
        self._check()

    def stop(self):
        if self._closed:
            self._check()
            return
        try:
            self.flush()
        finally:
            self._closed = True
            if self._worker_thread and self._worker_thread.is_alive():
                self.msg_queue.put(None)
                self._worker_thread.join(timeout=10)
                if self._worker_thread.is_alive():
                    raise StorageError("Storage worker did not stop")
            self._running = False
        self._check()

    def log_run(self, data):
        self._put(("record_run", data))

    def log_decision(self, decision_data, trace_data=None, runtime_data=None):
        self._put(("record_decision", (decision_data, trace_data, runtime_data)))

    def log_event(self, run_id, event_type, simulation_minute, details, state_hash):
        self._put(("log_event", (run_id, event_type, simulation_minute, details, state_hash)))

    def log_outcome(self, data):
        self._put(("record_outcome", data))

    def finalize_run(self, run_data, outcome_data):
        self._put(("finalize_run", (run_data, outcome_data)))

    def _process_queue(self):
        conn = None
        try:
            conn = self.db.connect()
            self._ready.set()
            while True:
                first = self.msg_queue.get()
                if first is None:
                    self.msg_queue.task_done()
                    break
                batch = [first]
                while len(batch) < self.batch_size:
                    try:
                        item = self.msg_queue.get_nowait()
                    except queue.Empty:
                        break
                    batch.append(item)
                    if item is None:
                        break
                started = time.perf_counter()
                try:
                    if self._failure is None:
                        with conn:
                            for cmd, payload in (item for item in batch if item is not None):
                                if cmd == "record_decision":
                                    decision, trace, runtime = payload
                                    self.db.record_decision(decision, conn)
                                    if trace:
                                        self.db.record_decision_trace(trace, conn)
                                    if runtime:
                                        self.db.record_runtime_metrics(runtime, conn)
                                elif cmd == "log_event":
                                    self.event_logger.log_event(*payload)
                                elif cmd == "finalize_run":
                                    self.db.finalize_run(*payload, conn=conn)
                                else:
                                    getattr(self.db, cmd)(payload, conn)
                        self.event_logger.flush()
                        self.transactions += 1
                except BaseException as exc:
                    self._failure = exc
                finally:
                    self.write_ms += (time.perf_counter() - started) * 1000
                    for _ in batch:
                        self.msg_queue.task_done()
                if None in batch:
                    break
        except BaseException as exc:
            self._failure = exc
        finally:
            self._ready.set()
            if conn is not None:
                conn.close()
            try:
                self.event_logger.close()
            except BaseException as exc:
                self._failure = self._failure or exc
            # Unblock callers even when connection initialization failed.
            while True:
                try:
                    self.msg_queue.get_nowait()
                    self.msg_queue.task_done()
                except queue.Empty:
                    break
            self._running = False
