# API providers: quick setup and safe comparisons

Hosted APIs need no local model download or Colab GPU. TamaBench currently supports non-streaming **Chat Completions text responses**, plus native Ollama. This is not universal support for every API: native Anthropic Messages, Gemini, OpenAI Responses, Azure-specific authentication, tool-only output, and multimodal output require separate adapters or a compatible gateway.

## Presets

Run `tamabench providers` to inspect the built-in settings locally. Pick an exact model ID from your provider account; presets deliberately do not select a billable model for you.

| Provider | API base | Credential environment variable / Colab Secret | Output token field |
|---|---|---|---|
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` | `max_completion_tokens` |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | `max_tokens` |
| `groq` | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` | `max_completion_tokens` |
| `ollama` | `http://localhost:11434` | Optional `TAMABENCH_API_KEY` | Native `num_predict` |
| `custom` | Explicit `--api-base` required | Optional `TAMABENCH_API_KEY` | `max_tokens`, overridable |

The cloud preset bases and API formats were checked against official [OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create), [OpenRouter setup](https://openrouter.ai/docs/quickstart), [Groq compatibility](https://console.groq.com/docs/openai), and [Groq request fields](https://console.groq.com/docs/api-reference) documentation on 2026-10-01. Provider/model availability, options, limits, and prices can change. All integration tests here use mocked responses, not live provider certification.

## Fast CLI smoke test

For OpenAI, first set your key without placing it in shell history:

```bash
read -rsp 'OpenAI API key: ' OPENAI_API_KEY; export OPENAI_API_KEY; echo
# Replace YOUR_MODEL_ID with a Chat Completions model available to your account.
tamabench doctor --provider openai --model YOUR_MODEL_ID
tamabench run --provider openai --model YOUR_MODEL_ID \
  --horizon-minutes 240 --max-api-calls 30 --max-output-tokens 512 \
  --max-total-tokens 30000 --max-wall-seconds 300 --max-retries 0 \
  --output-dir results/api-smoke
tamabench report --output-dir results/api-smoke
```

For OpenRouter or Groq, change `--provider` and set the matching variable from the table. Add `--api-key-env MY_KEY_VARIABLE` if you use another environment-variable name. Never pass the key value as an argument, put it in an endpoint URL, or save it in a configuration file.

`doctor` makes **zero provider requests**. It validates local configuration, output access, secret presence, and transport settings. Missing required credentials cause a nonzero exit before creating an experiment. It cannot establish model access, available balance, actual connectivity, or API compatibility. `run` sends real requests and may incur charges. The short horizon and budgets above are a setup smoke test, not sufficient evidence for a published ranking. Reasoning-heavy models may need a larger output allowance to produce an action.

## Custom compatible server

```bash
tamabench run --provider custom --model YOUR_MODEL_ID \
  --api-base https://your-provider.example/v1 \
  --api-key-env TAMABENCH_API_KEY --require-api-key \
  --output-dir results/custom
```

Replace the example base/model; the base should not include `/chat/completions`. The custom profile allows no-key local servers by default; omit `--require-api-key` for those. A configured key is sent even if it is optional, so select an unused credential variable for an intentionally unauthenticated server. Authenticated remote endpoints must use HTTPS. Explicit loopback addresses (`localhost`, `127.0.0.1`, `::1`) allow local HTTP development. Redirects are not followed, preventing implicit changes to the credential destination.

Named cloud endpoints cannot be overridden; use `custom` and choose its credential variable explicitly. With `--config`, changing `--provider` clears the previous provider's model, endpoint, auth-variable name, and model-specific options before applying new explicit CLI flags. Seed counts and benchmark budgets remain. This prevents a previous provider's key from accidentally following a changed destination.

## Model-specific options and fair comparisons

Cloud/custom presets omit temperature and sampling seed unless explicitly set. Null temperature and `seed_mode="omit"` are recorded, meaning the provider/model default applies. Episode seeds still control the deterministic simulator. Existing backend-only and Ollama configurations retain their established 0.2 temperature and episode-seed defaults.

- `--temperature 0.2`: send an explicit temperature if the model supports it
- `--omit-temperature`: omit temperature, including when overriding a saved configuration
- `--inference-seed 42` or `--seed-mode episode`: explicitly send a supported sampling seed; providers may ignore it
- `--output-token-parameter max_tokens` or `max_completion_tokens`: choose the compatible endpoint's accepted output-limit field
- `--reasoning-effort low`: optional model-specific setting, never sent automatically; only use supported values
- `--schema-mode provider_constrained`: optional JSON-object response mode; unsupported models may reject it

No retry silently changes parameters, endpoint, or model. Presets are convenience settings, not a guarantee that every hosted model accepts the same extras. Use matched explicit supported sampling settings, schema mode, token/call/wall budgets, horizon, and seed set for comparisons. Where models require different controls, disclose that difference instead of calling the comparison identical. Provider defaults, model aliases, backend routing, and stochastic inference can change; pin model versions where available. Attempts record response model ID, response ID, and system fingerprint when supplied, without treating those fields as independently verified provenance.

## Failures, accounting, and credentials

- `--max-retries` controls additional model responses after invalid/truncated action output; each response is counted and preserved
- HTTP errors, auth errors, timeouts, and rate limits are **not automatically retried**; this avoids hidden duplicated or billed requests
- Safe HTTP status diagnostics explain auth/model/parameter/quota problems without copying response bodies, request headers, or credentials into logs
- A failed API episode exits nonzero, records `infrastructure_failed`, and still exports its evidence; inspect it before `--resume`
- Request count and elapsed latency include failed requests. Provider token usage is only known after usable responses; absent or invalid usage is not treated as a free successful inference
- Token budgets stop subsequent calls based on observed usage and clamp requested output tokens. They are not a guaranteed pre-call monetary cap. Configure provider-side spending limits separately
- API secrets are loaded only from the named environment variable. Manifests contain the variable name, never its value. Colab Secrets are read into the runtime environment without printing the secret

## Colab

Open the [four-cell notebook](../notebooks/TamaBench_Colab.ipynb), run the default pinned-commit install cell, select a provider, enter your model ID, and add the matching Secret in Colab's Secrets panel with notebook access enabled. The configuration cell performs the same local preflight. The next cell runs the same `run_experiment` as the CLI; the final cell downloads results. CPU is the initial selection so simply opening/running an unconfigured notebook does not make hosted API calls.

The notebook and provider paths have been executed with mocked Colab interfaces and mocked HTTP responses. A fresh live Colab session and real API run remain untested.
