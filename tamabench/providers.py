"""Small explicit Chat Completions presets, not a universal API adapter."""

PROVIDERS = {
    "ollama": {"backend": "ollama", "api_base": "http://localhost:11434",
               "api_key_env": "TAMABENCH_API_KEY", "require_api_key": False,
               "output_token_parameter": "max_tokens"},
    "openai": {"backend": "openai_compatible", "api_base": "https://api.openai.com/v1",
               "api_key_env": "OPENAI_API_KEY", "require_api_key": True,
               "output_token_parameter": "max_completion_tokens"},
    "openrouter": {"backend": "openai_compatible", "api_base": "https://openrouter.ai/api/v1",
                   "api_key_env": "OPENROUTER_API_KEY", "require_api_key": True,
                   "output_token_parameter": "max_tokens"},
    "groq": {"backend": "openai_compatible", "api_base": "https://api.groq.com/openai/v1",
             "api_key_env": "GROQ_API_KEY", "require_api_key": True,
             "output_token_parameter": "max_completion_tokens"},
    "custom": {"backend": "openai_compatible", "api_key_env": "TAMABENCH_API_KEY",
               "require_api_key": False, "output_token_parameter": "max_tokens"},
}


def resolve_provider(values):
    """Fill missing values only; resolved configs round-trip without changes."""
    if not isinstance(values, dict) or values.get("provider") is None:
        return values
    values = dict(values)
    name = values["provider"]
    if not isinstance(name, str) or name not in PROVIDERS:
        raise ValueError("Unknown provider; use ollama, openai, openrouter, groq, or custom")
    preset = PROVIDERS[name]
    if name != "ollama" and (not isinstance(values.get("model"), str) or not values["model"].strip()):
        raise ValueError("An API provider requires an explicit model ID")
    if name == "custom" and not values.get("api_base"):
        raise ValueError("The custom provider requires an explicit api_base URL")
    if name in {"openai", "openrouter", "groq"}:
        if "api_base" in values and (not isinstance(values["api_base"], str) or values["api_base"].rstrip("/") != preset["api_base"]):
            raise ValueError("Named provider endpoints cannot be overridden; use provider='custom' and choose its credential variable explicitly")
        if values.get("require_api_key") is False:
            raise ValueError("Named cloud providers require an API key")
    if "backend" in values and values["backend"] != preset["backend"]:
        raise ValueError("provider and backend are incompatible")
    values.setdefault("agent", "raw_llm")
    for key, value in preset.items():
        values.setdefault(key, value)
    if name != "ollama":
        values.setdefault("temperature", None)
        values.setdefault("seed_mode", "omit")
    return values
