# DTIC Offset Evaluation System

A framework for running multi-turn persona-consistency evaluations against LLMs, with automated and manual scoring support.

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended.

---

## Environment Setup

Copy `.env.example` to `.env` and fill in the keys for the providers you intend to use:

```bash
cp .env.example .env
```

| Variable | Provider | Required |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI | If using `openai:*` models |
| `ANTHROPIC_API_KEY` | Anthropic | If using `anthropic:*` models |
| `XAI_API_KEY` | Grok (xAI) | If using `grok:*` models |
| `HF_API_TOKEN` | Hugging Face | If using `hf:*` models |

Ollama runs locally and requires no API key — just have the Ollama daemon running.

---

## Single Run

```bash
python main.py --scenario scenarios/example_scenario.json --model ollama:mistral
```

With an LLM judge for automated scoring:

```bash
python main.py \
  --scenario scenarios/example_scenario.json \
  --model openai:gpt-4o \
  --judge openai:gpt-4o
```

**Options:**

| Flag | Description | Default |
|---|---|---|
| `--scenario` | Path to scenario JSON | required |
| `--model` | `provider:model_name` | required |
| `--judge` | Judge model for auto-scoring | none |
| `--max-turns` | Max conversation turns | 40 |

**Supported providers:** `ollama`, `openai`, `anthropic`, `grok`, `hf`

Output is written to `logs/<run_id>.json` and a manual scoring sheet to `logs/scoring/<run_id>.csv`.

---

## Batch Run

Run all combinations of scenarios × models in one command:

```bash
python batch_run.py \
  --scenarios scenarios/ \
  --models ollama:mistral openai:gpt-4o \
  --judge openai:gpt-4o
```

**Options:**

| Flag | Description | Default |
|---|---|---|
| `--scenarios` | Directory or single JSON file | required |
| `--models` | One or more `provider:model` strings | required |
| `--judge` | Judge model for auto-scoring | none |
| `--max-turns` | Max conversation turns | 40 |
| `--retries` | Retry attempts on failure | 2 |

Results are appended to `logs/manifest.json`. Errors are saved to `logs/errors/`.

---

## Output Locations

| Path | Contents |
|---|---|
| `logs/<run_id>.json` | Full run log (conversation + scores) |
| `logs/scoring/<run_id>.csv` | Manual scoring sheet |
| `logs/manifest.json` | Batch run index |
| `logs/errors/` | Error tracebacks from failed runs |
| `logs/analysis/summary.csv` | Aggregated analysis output |

---

## Adding a New Scenario

Create a JSON file in `scenarios/` following this schema:

```json
{
  "scenario_id": "my_scenario_001",
  "model": "mistral",
  "identity": {
    "name": "Name",
    "background": "Brief background description",
    "persona": "Personality and disposition",
    "language_style": "How the persona speaks"
  },
  "objective": "What this scenario is testing",
  "constraints": [
    "Do not break character",
    "Respond only in the voice of the defined identity"
  ],
  "params": {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
    "seed": 42
  },
  "opening_message": "The first message sent to the model."
}
```

All fields are required. `temperature` must be 0–2, `top_p` must be 0–1, `max_tokens` must be > 0.

---

## Determinism & Reproducibility

Every run uses fixed generation parameters (temperature, top_p, seed) defined in the scenario JSON. This ensures that the same scenario + same model + same params produces reproducible results (REQ-006).

### Seed support by adapter

| Adapter | Seed support | Notes |
|---|---|---|
| **Ollama** | Yes | `seed` is passed in the `options` block of the HTTP payload to the local Ollama API. Fully deterministic when the same seed is used. |
| **OpenAI** | Yes | `seed` is passed directly to `chat.completions.create`. Supported since late 2023 (gpt-4-turbo and newer). Outputs are *system-fingerprint* deterministic — identical seed + params should yield the same result. |
| **Grok (xAI)** | Partial | Grok uses an OpenAI-compatible endpoint. The `seed` param is forwarded in the request, but determinism is not officially guaranteed by xAI at this time. |
| **Anthropic** | No | The Anthropic API does not expose a `seed` parameter. Reproducibility is not guaranteed even with identical temperature and top_p settings. |
| **HuggingFace** | No | The HuggingFace Inference API does not accept a `seed` parameter. Determinism depends on the underlying model and hosting environment and is generally not guaranteed. |

### Recommendations

- Use **Ollama** or **OpenAI** when reproducibility is required.
- For Anthropic and HuggingFace runs, set `temperature=0` to minimise (but not eliminate) variance.
- Always pin the `seed` field in your scenario JSON — even for adapters that don't use it, it documents intent and makes future re-runs comparable.

---

## Adding a New Model Adapter

1. Create `models/my_provider_adapter.py` subclassing `ModelAdapter`:

```python
from models.base import ModelAdapter

class MyProviderAdapter(ModelAdapter):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str, params: dict) -> str:
        # Call your provider's API here
        # params keys: temperature, top_p, max_tokens, seed
        response = my_provider_api_call(prompt, **params)
        return response.text
```

2. Register it in `models/__init__.py`:

```python
from .my_provider_adapter import MyProviderAdapter
```

3. Add a branch in `main.py`'s `build_model()`:

```python
elif provider == "myprovider":
    return MyProviderAdapter(name)
```

4. Use it with `--model myprovider:model-name`.
