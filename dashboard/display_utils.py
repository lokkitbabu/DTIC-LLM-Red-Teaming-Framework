"""
dashboard/display_utils.py

Single source of truth for constants and display helpers used across all views.
Import from here — do not redefine locally.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Metrics — canonical ordered list
# ---------------------------------------------------------------------------

METRICS = [
    "identity_consistency",
    "cultural_authenticity",
    "naturalness",
    "information_yield",
]

METRIC_LABELS = {
    "identity_consistency":  "IC",
    "cultural_authenticity": "CA",
    "naturalness":           "Nat",
    "information_yield":     "IY",
}

METRIC_LABELS_FULL = {
    "identity_consistency":  "Identity Consistency",
    "cultural_authenticity": "Cultural Authenticity",
    "naturalness":           "Naturalness",
    "information_yield":     "Information Yield",
}

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

PROBE_SCENARIOS = ["terrorism_recruitment_probe"]

ABLATION_SCENARIOS = [
    "terrorism_recruitment_full",
    "terrorism_recruitment_medium",
    "terrorism_recruitment_bare",
]

ALL_SCENARIOS = PROBE_SCENARIOS + ABLATION_SCENARIOS

FIDELITY_LEVELS = {
    "terrorism_recruitment_full":   "High",
    "terrorism_recruitment_medium": "Medium",
    "terrorism_recruitment_bare":   "Low",
}

# ---------------------------------------------------------------------------
# Model display names — single source of truth
# ---------------------------------------------------------------------------

# Maps any of: adapter repr, provider:model, bare model_id → clean display name
_MODEL_MAP: dict[str, str] = {
    # Bare model IDs
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "Llama 3.3 70B",
    "Llama-3.3-70B-Instruct-Turbo":             "Llama 3.3 70B",
    "deepseek-ai/DeepSeek-V3.1":                "DeepSeek V3.1",
    "DeepSeek-V3.1":                            "DeepSeek V3.1",
    "mistral-large-latest":                     "Mistral Large 3",
    "claude-sonnet-4-6":                        "Claude Sonnet 4.6",
    "gpt-5.4":                                  "GPT-5.4",
    "grok-4.20-0309-non-reasoning":             "Grok 4.20",
    # Provider-prefixed
    "together:meta-llama/Llama-3.3-70B-Instruct-Turbo": "Llama 3.3 70B",
    "together:deepseek-ai/DeepSeek-V3.1":               "DeepSeek V3.1",
    "mistral:mistral-large-latest":                     "Mistral Large 3",
    "anthropic:claude-sonnet-4-6":                      "Claude Sonnet 4.6",
    "openai:gpt-5.4":                                   "GPT-5.4",
    "grok:grok-4.20-0309-non-reasoning":                "Grok 4.20",
    # Adapter repr strings (from Supabase subject_model column)
    "TogetherAdapter(model=meta-llama/Llama-3.3-70B-Instruct-Turbo)": "Llama 3.3 70B",
    "TogetherAdapter(model=deepseek-ai/DeepSeek-V3.1)":               "DeepSeek V3.1",
    "MistralAdapter(model=mistral-large-latest)":                     "Mistral Large 3",
    "AnthropicAdapter(model=claude-sonnet-4-6)":                      "Claude Sonnet 4.6",
    "OpenAIAdapter(model=gpt-5.4)":                                   "GPT-5.4",
    "GrokAdapter(model=grok-4.20-0309-non-reasoning)":                "Grok 4.20",
}

# Model colors for consistent chart styling
MODEL_COLORS: dict[str, str] = {
    "DeepSeek V3.1":     "#2ecc71",
    "GPT-5.4":           "#3498db",
    "Grok 4.20":         "#9b59b6",
    "Claude Sonnet 4.6": "#e74c3c",
    "Llama 3.3 70B":     "#f39c12",
    "Mistral Large 3":   "#1abc9c",
}


def shorten_model(raw: str) -> str:
    """
    Convert any model string format to a clean display name.

    Handles:
      - Adapter repr:        'TogetherAdapter(model=deepseek-ai/DeepSeek-V3.1)'
      - Provider-prefixed:   'together:deepseek-ai/DeepSeek-V3.1'
      - Bare model ID:       'deepseek-ai/DeepSeek-V3.1'
      - Already clean name:  'DeepSeek V3.1'
    """
    if not raw:
        return raw

    # Try direct lookup first
    direct = _MODEL_MAP.get(raw)
    if direct:
        return direct

    # Strip provider prefix (e.g. "together:model" -> "model")
    model_id = raw.split(":")[-1] if ":" in raw else raw
    if model_id in _MODEL_MAP:
        return _MODEL_MAP[model_id]

    # Strip adapter wrapper (e.g. "TogetherAdapter(model=X)" -> "X")
    import re
    m = re.match(r"\w+Adapter\(model=(.+)\)$", raw.strip())
    if m:
        inner = m.group(1)
        if inner in _MODEL_MAP:
            return _MODEL_MAP[inner]
        # Try just the final path segment
        short = inner.split("/")[-1]
        if short in _MODEL_MAP:
            return _MODEL_MAP[short]
        return short[:28]

    # Last resort: take last path segment, strip any remaining wrapper
    fallback = model_id.split("/")[-1].split("(model=")[-1].rstrip(")")
    return fallback[:28]


def load_human_scores() -> "pd.DataFrame":
    """
    Load all run_scores from Supabase as a flat DataFrame.
    Includes: run_id, rater_id, 4 metrics, total — plus model + scenario_id joined in.
    Falls back gracefully if Supabase unavailable.
    """
    import pandas as pd
    try:
        from dashboard.supabase_store import get_store
        store = get_store()
        if not store.available:
            return pd.DataFrame()
        resp = store._client.table("run_scores").select(
            "run_id, rater_id, identity_consistency, cultural_authenticity,"
            " naturalness, information_yield, total, notes"
        ).execute()
        if not resp.data:
            return pd.DataFrame()
        df = pd.DataFrame(resp.data)
        run_resp = store._client.table("run_logs").select(
            "run_id, scenario_id, subject_model"
        ).execute()
        if run_resp.data:
            run_df = pd.DataFrame(run_resp.data)
            run_df["model"] = run_df["subject_model"].apply(shorten_model)
            df = df.merge(run_df[["run_id", "scenario_id", "model"]], on="run_id", how="left")
        for m in METRICS + ["total"]:
            if m in df.columns:
                df[m] = _to_numeric(df[m])
        return df
    except Exception:
        return pd.DataFrame()


def _to_numeric(s: "pd.Series") -> "pd.Series":
    import pandas as pd
    return pd.to_numeric(s, errors="coerce")
