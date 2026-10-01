"""Buffered append-only event stream; SQLite is the authoritative replay store."""
import json
import time
from pathlib import Path


class EventStreamLogger:
    def __init__(self, log_filepath="tamabench_events.jsonl"):
        self.log_filepath = log_filepath
        Path(log_filepath).parent.mkdir(parents=True, exist_ok=True)
        self._file = None

    def log_event(self, run_id, event_type, simulation_minute, details, state_hash=""):
        if self._file is None:
            self._file = open(self.log_filepath, "a", encoding="utf-8", buffering=65536)
        self._file.write(json.dumps({"timestamp_wall": time.time(), "run_id": run_id,
            "event_type": event_type, "simulation_minute": simulation_minute,
            "state_hash": state_hash, "details": details}, separators=(",", ":")) + "\n")

    def flush(self):
        if self._file:
            self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
