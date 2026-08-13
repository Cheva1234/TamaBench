"""JSONL Append-Only Event Stream Logger for TamaBench V1."""

import json
import os
import time
from typing import Any


class EventStreamLogger:
    def __init__(self, log_filepath: str = "tamabench_events.jsonl"):
        self.log_filepath = log_filepath

    def log_event(
        self,
        run_id: str,
        event_type: str,
        simulation_minute: int,
        details: dict[str, Any],
        state_hash: str = "",
    ):
        event_record = {
            "timestamp_wall": time.time(),
            "run_id": run_id,
            "event_type": event_type,
            "simulation_minute": simulation_minute,
            "state_hash": state_hash,
            "details": details,
        }
        json_line = json.dumps(event_record, sort_keys=True)
        
        # Open in append mode
        with open(self.log_filepath, "a", encoding="utf-8") as f:
            f.write(json_line + "\n")
