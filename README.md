# TamaBench 2.0

A small, reproducible autonomous-agent benchmark: keep a virtual pet healthy while managing work, money, inventory, and energy over seven simulated days.

Version 2.0 separates simulation time from model latency. Its accelerated engine jumps between events and exact integer-state thresholds; the reference engine advances minute by minute. Both produce the same states, event histories, elapsed time, and minute-weighted welfare statistics. No artificial sleeps are added.

## Quick start: CPU only

Python 3.10 or newer. From this checkout:

```bash
python -m pip install -e '.[dev]'
python -m tamabench doctor
python -m tamabench run --agent rule --output-dir results/rule
python -m tamabench report --output-dir results/rule
```

No GPU, model download, credentials, Ollama service, or network inference is needed for the default rule baseline. For a short smoke test add `--max-simulated-minutes 240`. `tamabench` and `python -m tamabench.cli` are equivalent entry points.

The [four-cell Colab notebook](notebooks/TamaBench_Colab.ipynb) uses exactly the same configuration and experiment functions. See [Colab setup](docs/colab.md), including the default reviewed-source-ZIP upload and optional published-ref installation.

## One configuration, three entry points

CLI, Python, and Colab all use `RunConfig` and `run_experiment`:

```python
from tamabench.config import RunConfig
from tamabench.experiment import run_experiment

config = RunConfig(
    agent="rule",
    episodes=3,
    seed_start=42,
    max_simulated_minutes=7 * 1440,
    output_dir="results/rule",
    display="compact",
)
result = run_experiment(config)
print(result.run_ids)
```

Save `config.model_dump_json(indent=2)` as a JSON file and pass `--config config.json`. Explicit CLI flags override the file. Invalid values, unknown configuration fields, unknown scenarios, non-finite numbers, and incompatible provider/lifecycle combinations fail before an episode starts.

Defaults: `dynamic_v2`, seven days, accelerated execution, one seed starting at 42, CPU rule agent, compact display. `standard_v1` is an explicit compatibility spelling for current `dynamic_v2` rules; it does not emulate the historical simulator. Actual metadata always records scenario version 2 and behavioral contract 2.0.0.

## Model backends

### Ollama

Install/run Ollama and obtain your model separately. TamaBench never silently downloads a model or installs a GPU stack.

```bash
python -m tamabench doctor --agent raw_llm --backend ollama --model qwen2.5:3b
python -m tamabench run \
  --agent raw_llm --backend ollama --model qwen2.5:3b \
  --api-base http://localhost:11434 \
  --keep-alive 5m --output-dir results/ollama
```

Ollama uses native `/api/generate` with an empty prompt for measured warmup, then `/api/chat` for decisions. Both send a finite `keep_alive` duration, default five minutes. Closing the HTTP client does not unload the server model; the TTL applies. `--model-lifecycle cold` explicitly unloads before each episode and records these cleanup requests separately in the manifest. This can affect other clients sharing that Ollama server. [Ollama API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)

### Hosted API providers (no GPU required)

Presets support OpenAI, OpenRouter, and Groq Chat Completions APIs. Pick your own exact model ID and set the corresponding environment variable: `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, or `GROQ_API_KEY`.

```bash
read -rsp 'API key: ' OPENAI_API_KEY; export OPENAI_API_KEY; echo
tamabench providers
tamabench doctor --provider openai --model YOUR_MODEL_ID
tamabench run --provider openai --model YOUR_MODEL_ID \
  --horizon-minutes 240 --max-api-calls 30 --max-output-tokens 512 \
  --max-total-tokens 30000 --max-wall-seconds 300 --max-retries 0 \
  --output-dir results/api-smoke
```

Change the provider and credential variable to use OpenRouter or Groq. `--provider` selects the model agent, backend, endpoint, named credential variable, and compatible output-token field. API presets require an explicit model instead of guessing a billable choice. `doctor` checks local configuration and credential presence without requests; `run` may incur provider charges. Missing required keys fail before an experiment starts. This short run is a smoke test, not a full ranking.

For other compatible endpoints, use `--provider custom --model YOUR_MODEL_ID --api-base https://your-provider.example/v1 --api-key-env TAMABENCH_API_KEY --require-api-key`. An unauthenticated local server can omit `--require-api-key`. The existing `--backend openai_compatible` interface remains supported. Credentials never belong in configuration files, CLI key-value arguments, or URLs. Named cloud endpoints are pinned; authenticated remote custom endpoints require HTTPS and redirects are disabled.

Cloud presets omit temperature, sampling seeds, reasoning options, and JSON mode unless explicitly chosen. This avoids sending unsupported model-specific options. These omissions are recorded and mean provider defaults apply. Generic services receive `/chat/completions` with no Ollama fields or artificial warmup. Native Responses, Anthropic Messages, and Gemini APIs are not implemented. API availability and model-option support were checked in documentation, not live-paid runs.

See [provider setup, custom endpoints, troubleshooting, and comparison guidance](docs/providers.md). In Colab, select the provider from the form, enter the model ID, and add the matching Colab Secret. Hosted APIs do not require a Colab GPU.

## Bounds and outcomes

Use `--max-decisions`, `--max-api-calls`, `--max-total-tokens`, `--max-wall-seconds`, `--max-consecutive-failures`, and `--max-stalled-decisions` to bound runs. Provider timeout, output-token limit, retry count, temperature, inference seed, schema mode, and reasoning effort are configurable. Ollama and legacy backend-only configurations follow the episode seed by default. Cloud/custom presets omit sampling seeds unless `--inference-seed` or `--seed-mode episode` is supplied.

Input-token usage is known only after a provider response. Token budgets stop further calls after observed usage and clamp requested output tokens; they are not a guaranteed pre-call token or monetary spending cap. In-flight HTTP requests are bounded by timeouts. Provider-reported usage can be inaccurate. With a token budget enabled, missing usage is treated as an infrastructure failure rather than allowing unmetered calls.

Episodes distinguish:

- `completed`: alive at the configured horizon
- `died`: health reached zero
- `invalid_action_abort` or `stalled`: repeated rejected/non-advancing decisions
- `budget_exhausted`: configured budget reached
- `interrupted` or `infrastructure_failed`: execution did not produce a finished scientific outcome

A rejected action advances no time and does not become a fallback wait. Time-skips stop at death or the horizon. Work earns its start-time quote only after the full job duration completes alive, including jobs ending exactly at the horizon. `success` means accepted; `completed` means the requested action finished alive. `execution_minutes` is actual elapsed time.

## Outputs, resume, replay

One output directory contains:

```text
manifest.json       configuration, source/spec hashes, seeds, dependencies, status
results.sqlite      run metadata, decisions, timing, and outcomes
events.jsonl        durable event stream
traces/             per-run replay JSONL and human-readable traces
summary.json        machine-readable measured episode metrics
summary.csv         one row per episode attempt
summary.md          readable results
```

`--no-trace-logs` disables the optional duplicate trace files; SQLite and events remain available. The legacy `--db-path` and `--event-path` flags override those two destinations; other outputs stay in `--output-dir` (or beside the requested database when no output directory is specified).

```bash
python -m tamabench run --agent rule --episodes 10 --output-dir results/rule --resume
python -m tamabench export --output-dir results/rule
python -m tamabench replay --output-dir results/rule --run-id RUN_ID_FROM_SUMMARY
python -m tamabench report --output-dir results/rule --experiment-id EXP_ID_FROM_MANIFEST
```

Resume skips exact configuration-and-seed pairs with finished statuses: `completed`, `died`, `invalid_action_abort`, `stalled`, or `budget_exhausted`. It retries `running`, `interrupted`, and `infrastructure_failed` attempts from the beginning. It does not selectively rerun scientific failures or continue mid-episode. Changing provider settings, budgets, the specification, or installed source creates a new fingerprint. Extending the seed range, changing display, or moving output paths does not change the behavioral fingerprint. Without `--resume`, another attempt is recorded.

Reports keep configurations separate and calculate values from recorded results. `report-v1` remains an alias for the current report command. V1 results are not directly comparable with the corrected V2 contract.

## What is measured

- Survival and actual simulated duration
- Health/happiness averages over every simulated minute, plus minimum health
- Strict JSON/schema reliability, first-pass failures, retries, and truncation
- Accepted actions, completed jobs, income, and spending
- Model calls versus deterministic harness/policy decisions and overrides
- Provider usage, model latency, warmup, wall time, and non-inference overhead

Unimplemented causal/planning/prediction metrics remain null or unavailable; no placeholder benchmark scores are presented as measured facts. Harness results include deterministic policy assistance and must not be interpreted as pure model autonomy.

The canonical action parser accepts one complete JSON object. It preserves sleep hours, rejects booleans as quantities, rejects negative/zero amounts and durations, rejects unknown fields/actions and duplicate keys, and does not repair truncated or prose-wrapped JSON. JSON, dictionaries, and typed proposals share one validation boundary.

The [behavioral specification](tamabench/spec/environment_v2.yaml) documents exact rates, action ordering, automatic sleep transitions, dynamic prices/rewards, sickness events, and welfare sampling. `hunger` is the legacy field name for fullness: 100 is full and 0 is starving.

## Development and verification

```bash
python -m pytest
python scripts/benchmark_overhead.py --help
```

The offline regression suite includes the seed-14 reference/accelerated health divergence, 1,000 reachable randomized trajectories, welfare segmentation invariance, an independent minute-end oracle, horizon and reward edges, strict parser parity, and mocked provider/logging failures. The overhead benchmark uses deterministic fake models and makes no real inference requests. Real GPU/provider performance is not claimed by these checks.

License: [MIT](LICENSE)

The optional local submission prototype can be installed with `python -m pip install -e '.[api]'`. It stores unverified client-submitted results and is not a trusted public leaderboard. Development/CI API tests use in-process test clients, with no production database or network server.

The GitHub Actions workflow runs the full offline suite on Python 3.10 and 3.12, builds a wheel, and smoke-tests the installed wheel outside the checkout. This workflow is provided for a future authorized push; creating the file does not mean hosted CI has run.
