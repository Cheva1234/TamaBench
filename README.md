<div align="center">

# 🐾 TamaBench

### Small Model Autonomy Benchmark

**A lightweight long-horizon benchmark for small and local LLM agents — and the automation harnesses that extend them.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-compatible-black.svg)](https://ollama.com/)

</div>

---

## Can a 2.6B model autonomously survive for days?

TamaBench places an AI agent inside a persistent sandbox where it must:

- care for a virtual pet
- work and earn money
- manage limited resources
- make structured tool calls
- plan around delayed consequences
- recover from mistakes

TamaBench measures:

**Planning · Tool Reliability · Resource Management · Failure Modes · Tokens · Latency · Compute Efficiency**

## The Bigger Question

> How much autonomy can we extract from a small model before we need a larger model?

And:

> How much can an automation harness close that gap?

---

## What is TamaBench?

TamaBench is an **open benchmark for small, local, and quantized language models**. It evaluates how well a model can act as an **autonomous agent** in a real-time resource-management environment — a virtual pet (Tamagotchi) that decays over time and needs constant care.

The agent must **survive 3 simulated days** by:
- Feeding, cleaning, and healing a pet
- Earning money through jobs to buy supplies
- Balancing its own energy with the pet's needs
- Reacting to random events like sickness

This tests **real-world agentic capabilities** — not text generation quality, but the model's ability to **plan, prioritize, manage resources, and adapt under pressure**.

The Tamagotchi-style environment is the **simulation mechanism**, not the whole identity of the project. The benchmark itself is about long-horizon autonomy: persistent decision making, tool/JSON reliability, planning under delayed consequences, resource management, failure recovery, context efficiency, and compute efficiency.

### The Simulation Loop

```text
 MODEL
   ↓
 DECIDE
   ↓
 WORK / CARE / WAIT
   ↓
 WORLD FAST-FORWARD
   ↓
 CONSEQUENCES
   ↓
 NEXT WAKE
   ↓
 MODEL
```

## Quick Demo

Start a complete local autonomous-agent benchmark with:

```bash
ollama pull llama3.2:3b
python -m tamabench.cli run \
  --agent raw_llm \
  --model llama3.2:3b \
  --episodes 1 \
  --display live
```

Replace `llama3.2:3b` with any model served by Ollama or an OpenAI-compatible
local endpoint. The live monitor shows the simulation state, actions, schema
quality, economy, and the final benchmark report after the episode ends.
### Hunger Meter Semantics

The `hunger` value is a **fullness meter** so that larger values are better:

- `100` = fully fed
- `0` = starving
- Feeding increases the meter by `35`
- Time lowers the meter by `18` per simulated hour
- Health damage begins when the meter falls below `15`

This meaning is intentionally explicit because autonomous agents must reason about
whether a time-based action will leave enough food energy before they sleep, work,
or wait.

---

## Why Use TamaBench to Select an Automation Model?

Choosing an LLM for automation tasks is difficult. Standard benchmarks (MMLU, HumanEval, GSM8K) measure knowledge recall or code generation — but **they do not measure how a model behaves as an autonomous agent**.

TamaBench closes this gap by requiring the model to:

| Capability | How TamaBench Tests It |
|---|---|
| **Multi-step planning** | Resources decay — survival requires 3-day resource plans |
| **Priority reasoning** | Multiple needs compete simultaneously (hunger vs. money vs. cleanliness) |
| **Tool use / JSON output** | Every action is a structured JSON payload with strict schema |
| **Budget & resource management** | Money must be earned before supplies can be bought |
| **Adaptation** | Random sickness events force reactive replanning |
| **Avoiding catastrophic failures** | Health reaching 0 terminates the episode immediately |

## Designed for Small-Model Autonomous Agents

TamaBench is especially useful for evaluating **small, local, and quantized model weights** running on consumer hardware. The goal is not only to ask whether a model can produce a correct answer, but whether it can repeatedly run an autonomous control loop:

```text
observe state → choose an action → use a tool → wait for consequences → observe again
```

This loop is common in practical automation systems. A model used for an autonomous task may need to:

| Autonomous task | What the model must do |
|---|---|
| **IoT monitoring** | Read sensor state, detect abnormal conditions, trigger an actuator or alert, and continue monitoring |
| **Email automation** | Inspect incoming messages, classify priority, draft or send a response, avoid duplicate actions, and escalate uncertain cases |
| **Home automation** | Balance temperature, energy usage, schedules, and safety constraints over time |
| **Server and service monitoring** | Read health signals, restart a failed component when safe, collect evidence, and notify an operator when recovery fails |
| **Workflow automation** | Break a task into steps, call tools in the correct order, manage budgets and deadlines, and recover from errors |

The Tamagotchi environment is a controlled sandbox for these same capabilities. It tests whether a model can:

- maintain state across many decisions;
- choose actions based on changing observations;
- use structured JSON as a tool/action interface;
- plan around time, resources, and delayed consequences;
- react to unexpected events;
- avoid unsafe or irreversible actions; and
- complete tasks efficiently with limited inference calls and token usage.

TamaBench does **not** claim that surviving the pet simulation directly proves that a model can safely operate an email account, IoT device, or production server. Those applications require their own tools, permissions, safety policies, and domain-specific tests. TamaBench measures the reusable agent skills underneath them: observation, planning, tool use, recovery, and long-horizon control.

### ✅ Pros

- **Fully local** — runs with [Ollama](https://ollama.com/) on consumer hardware, no API keys required
- **Reproducible** — seeded random events ensure fair comparison between models
- **Fast** — one episode completes in **under 5 minutes** on CPU-only hardware
- **Observable** — full reasoning trace captured per step (`logs/reasoning_<date>.txt`)
- **Model-agnostic** — any model served via Ollama or OpenAI-compatible API works
- **Zero prompt leakage** — the system prompt is purely specification-based (no hints about when to use actions)
- **Quantified results** — outputs survival rate, average health, happiness score, economic efficiency

---

## Harness Evaluation: The Differentiator

TamaBench measures not only:

```text
Model A  vs  Model B
```

but also:

```text
Raw Model  vs  Raw Model + Harness
```

This is one of the main distinguishing features of TamaBench — it answers:

> Does better automation architecture compensate for smaller model size?

### Harness V1

The first harness is intentionally minimal:

```text
 WAKE
   ↓
 OBSERVE
   ↓
 DECIDE
   ↓
 CALCULATE NEXT WAKE
   ↓
 SCHEDULE
   ↓
 SLEEP
   ↓
 WAKE
```

Three core stages:

```text
1. DECIDE
2. CALCULATE
3. SCHEDULE
```

The model does not need to run continuously — it wakes only when a care decision is required. Routine economy (work, buy, wait) is handled deterministically by the harness's reference policy, creating a controlled experiment around:

- API call reduction
- Token reduction
- Compute reduction
- Better timing
- Better long-horizon survival

…without changing model size.

### Signature Experiment: Same Model, Different Harness

```text
LFM2.5 2.6B Raw
 ↓
LFM2.5 + Wake Scheduler
 ↓
LFM2.5 + Harness V1
```

Keep identical: model, quantization, prompt budget, environment, scenario, seed set, temperature. Then compare:

```text
Survival Rate
API Calls / Day
Tokens / Day
Planning Failures
Schema Failures
Resource Failures
Latency
```

The key result becomes:

> **How much effective autonomy came from the harness rather than the model?**

---

## Current Results

| Model | Survival | Schema | Calls / Day | Tokens / Day |
|---|---:|---:|---:|---:|
| RandomSchema | TBD | TBD | — | — |
| RandomValid | TBD | 100% | — | — |
| RuleAgent | TBD | 100% | — | — |
| LFM2.5 2.6B Raw | TBD | TBD | TBD | TBD |
| LFM2.5 + Harness V1 | TBD | TBD | TBD | TBD |

Planned expansion: Average Health, Planning Failures, Resource Failures, Truncation Rate, Average Latency, Reasoning Tokens, Cost / Simulated Day.

---

## Current State (v1.1 runtime)

> ⚠️ **Early Research Preview** — APIs and scoring may change between minor versions.

### What Works
- ✅ Full simulation engine with event-driven time-skipping
- ✅ 6 actions: `feed`, `clean`, `heal`, `play`, `sleep`, `work`, `buy`, `wait`, `wake`
- ✅ Economy system: 3 jobs (café shift, delivery, freelance), shop (food, medicine)
- ✅ Sickness events with probabilistic triggers based on cleanliness
- ✅ Per-step reasoning trace extraction (`<think>` tag support)
- ✅ SQLite result database + JSONL event stream
- ✅ Rule-based baseline agent for reference comparison
- ✅ Tested with: `llama3.2:3b`, `qwen2.5:7b`, `lfm2.5-2.6b`
- ✅ Configurable generation limit (`--max-output-tokens`, default `4096`)
- ✅ Persistent model residency with Ollama `keep_alive` and warmup telemetry
- ✅ Inference only at decision boundaries; blocking actions use analytical time-skips
- ✅ Separate first-pass schema, recovery, retry, and truncation metrics
- ✅ Reference (one-minute) and accelerated (event-driven) modes with state-hash equivalence tests
- ✅ **Harness V1 agent** (`--agent harness_v1`): DECIDE → CALCULATE → SCHEDULE loop that wakes the model only for care decisions and handles routine economy deterministically (fewer API calls, same survival)

### Known Limitations
- ❌ No multi-agent or parallel episode runner yet
- ❌ No web dashboard for result visualization
- ❌ Results from older environment versions are not cross-version comparable

---

## Installation

**Requirements:** Python 3.11+, [Ollama](https://ollama.com/)

```bash
# 1. Clone the repository
git clone https://github.com/Cheva1234/TamaBench.git
cd TamaBench

# 2. Install (editable mode recommended)
pip install -e .

# 3. Pull a model via Ollama
ollama pull llama3.2:3b
```

---

## How to Use

### Run a Benchmark Episode

```bash
# Run 1 episode with any Ollama model
python -m tamabench.cli run --agent raw_llm --model llama3.2:3b --episodes 1

# Use the V1.1 default output budget explicitly
python -m tamabench.cli run --agent raw_llm --model lfm2.5-2.6b \
  --max-output-tokens 4096 --episodes 1

# Run 3 episodes with a quantized model
python -m tamabench.cli run --agent raw_llm --model qwen2.5:7b --episodes 3

# Run the rule-based baseline (no model needed)
python -m tamabench.cli run --agent rule --episodes 1

# Run the Harness V1 agent (wraps a model; harness handles routine economy)
python -m tamabench.cli run --agent harness_v1 --model <your-model> --episodes 1
```

### Watch the Reasoning Log (Live)

```bash
tail -f logs/reasoning_latest.txt
```

### Compare Models

```bash
# Run baseline
python -m tamabench.cli run --agent rule --episodes 5

# Run target model
python -m tamabench.cli run --agent raw_llm --model <your-model> --episodes 5

# Query results
sqlite3 tamabench_results.db "
  SELECT model_name, COUNT(*) as episodes,
         ROUND(AVG(survived)*100,1) as survival_pct,
         ROUND(AVG(final_health),1) as avg_health
  FROM outcomes JOIN runs USING(run_id)
  GROUP BY model_name;
"
```

### Shareable Model Comparison

For a useful comparison, run the same number of episodes and seeds for every
model, then share survival, health, schema quality, and efficiency together:

```bash
python -m tamabench.cli run --agent rule --episodes 5 --seed-start 42 --display compact
python -m tamabench.cli run --agent raw_llm --model YOUR_MODEL \
  --episodes 5 --seed-start 42 --display compact
python -m tamabench.cli report-v1
```

Use this format when posting results:

| Model | Episodes | Survival | Average health | Final schema | Output tokens | p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| `your-model` | 5 | fill in | fill in | fill in | fill in | fill in |

Always include the TamaBench/environment version, seed range, execution mode,
and generation limit so other people can reproduce the comparison.

---

## Output & Logs

| File | Description |
|---|---|
| `tamabench_results.db` | SQLite database with all episode results |
| `tamabench_events.jsonl` | Per-step event stream (JSONL format) |
| `logs/reasoning_<date>.txt` | Reasoning, JSON output, finish reason, and token split per step |
| `logs/replay_<run_id>.jsonl` | Full episode replay for post-analysis |

---

## Project Structure

```
TamaBench/
├── tamabench/
│   ├── env/              # Simulation engine (core, dynamics, scheduler, economy)
│   ├── agents/           # Agent implementations (rule-based, raw LLM, harness V1)
│   ├── context/          # System prompt builder
│   ├── validation/       # JSON schema + environment precondition validators
│   ├── logging/          # File logger, DB logger, event stream
│   ├── metrics/          # Scoring calculator, live reporter
│   ├── runner/           # Batch runner
│   └── cli.py            # Entry point
├── tests/                # Unit + integration test suite
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## How Scoring Works

An episode is evaluated across survival, care quality, economy, decision quality,
and inference efficiency:

| Metric | Description |
|---|---|
| **Survival** | Binary: did the pet survive all 3 simulated days? |
| **Average Health** | Mean health across all sampled steps (0–100) |
| **Average Happiness** | Mean happiness across all sampled steps (0–100) |
| **Economic Efficiency** | Income earned relative to spending |
| **First-Pass Schema Compliance** | % of decisions valid before recovery |
| **Final Schema Recovery** | % recovered or valid after retries |
| **Truncation / Retry Rate** | Runtime-cut generation and retry frequency |
| **Inference Efficiency** | p95 latency, API calls/day, reasoning/JSON tokens, and profiler breakdown |

---

## License

[MIT](LICENSE) — free to use, modify, and distribute.

---

<div align="center">
<sub>Small Model. Long Horizon. Persistent Consequences. · Built as part of Project Aether · TamaBench v1.1.0</sub>
</div>
