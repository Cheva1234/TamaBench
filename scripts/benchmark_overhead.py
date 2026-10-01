"""Offline latency-isolation benchmark. Never sends inference requests."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import tempfile
import time
from types import SimpleNamespace

from tamabench.agents.rule_agent import RuleAgent
from tamabench.agents.base import BaseAgent, DecisionMetadata
from tamabench.env.time_engine import BenchmarkMode, ComputeClock
from tamabench.runner.batch_runner import BatchRunner


class DelayedRuleModel(BaseAgent):
    def __init__(self, delay_ms):
        super().__init__("offline_delayed_rule")
        self.policy = RuleAgent()
        self.delay_ms = delay_ms
        self.runtime = SimpleNamespace(api_calls=0, warmup_calls=0, cleanup_calls=0,
            input_tokens=0, output_tokens=0, generation_ms=0.0)
        self.last_compute = ComputeClock()

    def select_action(self, observation):
        start = time.perf_counter()
        time.sleep(self.delay_ms / 1000)
        generation = (time.perf_counter()-start)*1000
        self.last_compute = ComputeClock(generation_ms=generation)
        self.last_decision = DecisionMetadata(input_tokens=100, total_output_tokens=16, action_source="model")
        self.runtime.api_calls += 1
        self.runtime.input_tokens += 100
        self.runtime.output_tokens += 16
        self.runtime.generation_ms += generation
        return self.policy.select_action(observation)


def measure(episodes=3, delay_ms=10.0):
    rows = []
    with tempfile.TemporaryDirectory(prefix="tamabench-profile-") as directory:
        for mode in (BenchmarkMode.LOGICAL, BenchmarkMode.ACCELERATED):
            for latency in (0, delay_ms):
                root = Path(directory) / f"{mode.value}-{latency}"
                with BatchRunner(str(root/'results.sqlite'), str(root/'events.jsonl'), mode=mode,
                                 log_dir=str(root/'traces'), trace_logs=True) as runner:
                    for seed in range(episodes):
                        agent = DelayedRuleModel(latency)
                        result = asdict(runner.run_episode(agent, seed=seed, max_simulated_minutes=1440))
                        result.update(mode=mode.value, injected_latency_ms=latency)
                        rows.append(result)
    groups = []
    for mode in ("logical", "accelerated"):
        for latency in (0, delay_ms):
            group = [row for row in rows if row['mode']==mode and row['injected_latency_ms']==latency]
            groups.append({"mode":mode, "injected_latency_ms":latency, "episodes":len(group),
                "median_episode_ms":statistics.median(r['episode_wall_time_ms'] for r in group),
                "median_non_inference_ms":statistics.median(r['overhead_ms'] for r in group),
                "median_inference_fraction":statistics.median(r['inference_fraction'] for r in group),
                "median_simulation_ms":statistics.median(r['simulation_ms'] for r in group),
                "median_decisions":statistics.median(r['total_decisions'] for r in group)})
    return {"kind":"offline synthetic latency; no LLM/GPU performance claim", "traces_enabled":True,
            "summary":groups,"episodes":rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--delay-ms', type=float, default=10)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or args.delay_ms <= 0:
        parser.error('episodes and delay must be positive')
    result = measure(args.episodes, args.delay_ms)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text+'\n')
    print(json.dumps(result['summary'], indent=2))
