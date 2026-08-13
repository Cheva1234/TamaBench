"""Persistent File Logger for Agent Reasoning Traces & Replay Logs.

Generates standalone, collectable text log files (logs/run_<RUN_ID>_reasoning.txt)
and replay JSONL streams (logs/run_<RUN_ID>_replay.jsonl) per benchmark episode run.
"""

import os
import json
from typing import Any, Optional
from tamabench.schemas.observation import Observation
from tamabench.schemas.actions import ActionProposal, StepResult


class FileLogger:
    def __init__(self, run_id: str, log_dir: str = "logs"):
        self.run_id = run_id
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.reasoning_path = os.path.join(self.log_dir, f"{self.run_id}_reasoning.txt")
        self.replay_path = os.path.join(self.log_dir, f"{self.run_id}_replay.jsonl")

        self.latest_reasoning_path = os.path.join(self.log_dir, "latest_reasoning.txt")
        self.latest_replay_path = os.path.join(self.log_dir, "latest_replay.jsonl")

        # Initialize reasoning text file header
        header = (
            "================================================================================\n"
            "TAMABENCH V1 — AGENT REASONING & DECISION TRACE LOG\n"
            f"RUN ID: {self.run_id}\n"
            "================================================================================\n\n"
        )
        with open(self.reasoning_path, "w", encoding="utf-8") as f:
            f.write(header)
        with open(self.latest_reasoning_path, "w", encoding="utf-8") as f:
            f.write(header)

        # Clear latest replay JSONL file
        with open(self.latest_replay_path, "w", encoding="utf-8") as f:
            f.write("")

    def log_step(
        self,
        step_index: int,
        obs: Observation,
        raw_output: str,
        proposal: Optional[ActionProposal],
        step_result: StepResult,
        latency_ms: float = 0.0,
        thinking_process: str = "",
        decision_metadata: Optional[dict[str, Any]] = None,
    ):
        p = obs.pet
        a = obs.agent
        inv = obs.inventory

        action_str = proposal.action if proposal else "unknown"
        if proposal and proposal.job_id:
            action_str += f" (job_id={proposal.job_id})"
        elif proposal and proposal.item:
            action_str += f" (item={proposal.item}, amount={proposal.amount})"
        elif proposal and proposal.hours:
            action_str += f" (hours={proposal.hours})"
        elif proposal and proposal.minutes:
            action_str += f" (minutes={proposal.minutes})"

        priority = proposal.trace.current_priority if (proposal and proposal.trace) else "-"
        rationale = proposal.trace.decision_rationale if (proposal and proposal.trace) else "-"
        summary = proposal.trace.situation_summary if (proposal and proposal.trace) else "-"

        err_str = f"❌ ERROR: {step_result.error.error_type} ({step_result.error.message})" if step_result.error else "✅ SUCCESS"

        thinking_block = f"• Model Internal Thinking Process (<think>):\n  {thinking_process.strip()}\n" if thinking_process and thinking_process.strip() else ""
        metadata = decision_metadata or {}
        generation_block = (
            f"• Generation       : finish_reason={metadata.get('finish_reason', '-')}, "
            f"attempts={metadata.get('attempt_count', 1)}, "
            f"truncated={bool(metadata.get('was_truncated', False))}\n"
            f"• Token Split      : reasoning={metadata.get('reasoning_tokens', 0)}, "
            f"json={metadata.get('json_tokens', 0)}, "
            f"total={metadata.get('total_output_tokens', 0)}\n"
        )

        block = (
            "--------------------------------------------------------------------------------\n"
            f"STEP #{step_index:04d} | Sim Time: Day {obs.time.day} {obs.time.hour:02d}:{obs.time.minute:02d} | Latency: {latency_ms:.1f}ms\n"
            "--------------------------------------------------------------------------------\n"
            f"• Proposed Action : {action_str}\n"
            f"• Execution Status: {err_str}\n"
            f"• Pet State       : Health={p.health:.1f}/100 | Hunger/Fullness={p.hunger:.1f}/100 | Cleanliness={p.cleanliness:.1f}/100 | Sick={p.is_sick} | Sleeping={p.is_sleeping}\n"
            f"• Agent State     : Money=${a.money} | Energy={a.energy}/100 | Activity={a.activity}\n"
            f"• Inventory       : Food={inv.food} | Medicine={inv.medicine}\n"
            f"• Primary Priority: {priority}\n"
            f"• Situation Summary: {summary}\n"
            f"• Decision Rationale:\n"
            f"  {rationale}\n"
            f"{thinking_block}"
            f"{generation_block}"
            f"• Raw Model Response:\n"
            f"  {raw_output.strip()}\n\n"
        )

        with open(self.reasoning_path, "a", encoding="utf-8") as f:
            f.write(block)
        with open(self.latest_reasoning_path, "a", encoding="utf-8") as f:
            f.write(block)

        # Log standalone replay event JSONL line
        replay_record = {
            "run_id": self.run_id,
            "step_index": step_index,
            "simulated_minute": (obs.time.day - 1) * 1440 + obs.time.hour * 60 + obs.time.minute,
            "time": f"Day {obs.time.day} {obs.time.hour:02d}:{obs.time.minute:02d}",
            "observation": obs.to_dict(),
            "raw_output": raw_output,
            "proposal": proposal.model_dump(exclude_none=True) if proposal else None,
            "result": {
                "success": step_result.success,
                "error": step_result.error.model_dump(exclude_none=True) if step_result.error else None,
                "state_hash": step_result.state_hash,
            },
            "generation": metadata,
        }

        line = json.dumps(replay_record) + "\n"
        with open(self.replay_path, "a", encoding="utf-8") as f:
            f.write(line)
        with open(self.latest_replay_path, "a", encoding="utf-8") as f:
            f.write(line)

    def log_summary(self, survived: bool, simulated_days: float, final_health: float, final_money: int):
        status_str = "PASSED (SURVIVED)" if survived else "FAILED (PET DIED)"
        summary_text = (
            "================================================================================\n"
            "BENCHMARK EPISODE SUMMARY\n"
            f"• Outcome        : {status_str}\n"
            f"• Simulated Days : {simulated_days:.2f} days\n"
            f"• Final Health   : {final_health:.1f}/100\n"
            f"• Final Money    : ${final_money}\n"
            "================================================================================\n"
        )
        with open(self.reasoning_path, "a", encoding="utf-8") as f:
            f.write(summary_text)
        with open(self.latest_reasoning_path, "a", encoding="utf-8") as f:
            f.write(summary_text)
