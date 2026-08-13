<div align="center">

# 🐾 TamaBench

**A Tamagotchi-style survival benchmark for evaluating autonomous LLM agents**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-compatible-black.svg)](https://ollama.com/)

</div>

---

## What is TamaBench?

TamaBench is an **open benchmark** that evaluates how well a language model can act as an **autonomous agent** in a real-time resource-management environment — a virtual pet (Tamagotchi) that decays over time and needs constant care.

The agent must **survive 3 simulated days** by:
- Feeding, cleaning, and healing a pet
- Earning money through jobs to buy supplies
- Balancing its own energy with the pet's needs
- Reacting to random events like sickness

This tests **real-world agentic capabilities** — not text generation quality, but the model's ability to **plan, prioritize, manage resources, and adapt under pressure**.

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

### ✅ Pros

- **Fully local** — runs with [Ollama](https://ollama.com/) on consumer hardware, no API keys required
- **Reproducible** — seeded random events ensure fair comparison between models
- **Fast** — one episode completes in **under 5 minutes** on CPU-only hardware
- **Observable** — full reasoning trace captured per step (`logs/reasoning_<date>.txt`)
- **Model-agnostic** — any model served via Ollama or OpenAI-compatible API works
- **Zero prompt leakage** — the system prompt is purely specification-based (no hints about when to use actions)
- **Quantified results** — outputs survival rate, average health, happiness score, economic efficiency

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

### Known Limitations
- ❌ No multi-agent or parallel episode runner yet
- ❌ No web dashboard for result visualization
- ❌ Benchmark v0.1.0 results are not cross-version comparable

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
│   ├── agents/           # Agent implementations (rule-based, raw LLM)
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

An episode is scored across 5 dimensions:

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
<sub>Built as part of Project Aether · TamaBench v0.1.0</sub>
</div>
