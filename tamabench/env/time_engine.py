"""Time Architecture Engine for TamaBench V1.

Manages Simulation Clock, Compute Clock, and Wall Clock.
Implements Logical Mode where Simulation Clock pauses during AI model inference.
"""

import time
from dataclasses import dataclass, field
from enum import Enum


class BenchmarkMode(str, Enum):
    LOGICAL = "logical"
    ACCELERATED = "accelerated"
    REALTIME = "realtime"


@dataclass
class ComputeClock:
    model_load_ms: float = 0.0
    ttft_ms: float = 0.0
    generation_ms: float = 0.0
    schema_validation_ms: float = 0.0
    retry_ms: float = 0.0
    total_decision_ms: float = 0.0

    def reset(self):
        self.model_load_ms = 0.0
        self.ttft_ms = 0.0
        self.generation_ms = 0.0
        self.schema_validation_ms = 0.0
        self.retry_ms = 0.0
        self.total_decision_ms = 0.0


class TimeEngine:
    def __init__(self, mode: BenchmarkMode = BenchmarkMode.LOGICAL):
        self.mode = mode
        self.wall_clock_start: float = time.time()
        self.total_compute_ms: float = 0.0

    def record_compute_time(self, compute: ComputeClock):
        self.total_compute_ms += compute.total_decision_ms

    def get_elapsed_wall_seconds(self) -> float:
        return time.time() - self.wall_clock_start
