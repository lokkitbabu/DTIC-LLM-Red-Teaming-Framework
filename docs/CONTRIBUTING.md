# Contributing & Development

How to extend the framework, run tests, and add new capabilities.

---

## Setup

```bash
git clone https://github.com/lokkitbabu/DTIC-LLM-Red-Teaming-Framework.git
cd DTIC-LLM-Red-Teaming-Framework
pip install -r requirements.txt
cp .env.example .env
```

---

## Project Layout

```
models/           Provider adapters — one file per provider
runner/           Conversation orchestration
evaluation/       Scoring logic and rubric definitions
dashboard/        Streamlit UI and data layers
  views/          One file per dashboard page
analysis/         Offline analysis scripts
utils/            Shared: logger, schema, scenario loader
tests/            Unit + property-based tests
scenarios/        Experiment definitions (JSON)
```

---

## Adding a Model Provider

1. Create `models/{provider}_adapter.py`:

```python
import os
from openai import OpenAI          # if OpenAI-compatible
from models.base import ModelAdapter

class MyAdapter(ModelAdapter):
    def __init__(self, model_name: str):
        self.model_name = model_name
        api_key = os.environ.get("MY_API_KEY")
        if not api_key:
            raise EnvironmentError("MY_API_KEY not set")
        self.client = OpenAI(api_key=api_key, base_url="https://api.my.ai/v1")

    def generate(self, prompt: str, params: dict) -> str:
        # parse [SYSTEM]/[CONVERSATION] prompt into messages
        messages = self._parse_prompt(prompt)
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=params.get("temperature", 0.7),
            max_tokens=params.get("max_tokens", 512),
        )
        return resp.choices[0].message.content or ""

    def _parse_prompt(self, prompt: str) -> list[dict]:
        # standard split — copy from together_adapter.py
        ...
```

2. Register in `models/__init__.py`:

```python
from models.my_adapter import MyAdapter
# add "MyAdapter" to __all__
```

3. Add routing in `main.py`'s `build_model()`:

```python
elif provider == "myprovider":
    return MyAdapter(name)
```

4. Add `MY_API_KEY=` to `.env.example`

5. Add preset button in `dashboard/views/run_executor.py`'s `MODEL_PRESETS`

---

## Adding a Dashboard Page

1. Create `dashboard/views/my_page.py`:

```python
import streamlit as st
import pandas as pd

def render_my_page(run_index: pd.DataFrame) -> None:
    st.subheader("My Page")
    # your view code here
```

2. Import in `dashboard/app.py`:

```python
from dashboard.views.my_page import render_my_page
```

3. Add to the nav radio and routing:

```python
page = st.sidebar.radio("Navigate", [..., "My Page"])

elif page == "My Page":
    render_my_page(filtered_index)
```

---

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_conversation_runner.py -v

# Property-based tests only
python -m pytest tests/test_property_*.py -v

# With coverage
python -m pytest tests/ --cov=. --cov-report=term-missing
```

The test suite has ~4400 lines across 20 files covering:
- Unit tests: conversation runner, evaluation, data loader, scenario loader
- Property-based tests: filter state, flag manager, score writer, scoring UI, summary, charts, comparison, conversation view

---

## Code Style

- Python 3.10+
- Type hints on all function signatures
- Docstrings on all public functions and classes
- No bare `except:` — always catch specific exceptions
- Streamlit views: one public `render_*()` function per file, private helpers prefixed `_`

---

## Key Design Decisions

### Why two separate LLMs?

Most red-teaming frameworks use scripted probes. Using a live LLM as interviewer means:
- Questions adapt to what the subject actually said
- Natural rapport-building and escalation
- More realistic pressure test

The tradeoff is reproducibility — same scenario + same models may produce different conversations. Mitigated by running 3 repeats per combination and using fixed seeds where supported.

### Why Supabase?

Four raters need to score the same runs independently without sharing CSV files by email. Supabase provides:
- Single source of truth for all run data
- Real-time score updates visible to everyone
- Zero infrastructure overhead (managed Postgres)
- Graceful local file fallback when not configured

### Why three prompt formats?

The client (Offset Labs) explicitly requested investigation of hierarchical prompting as a research area. The A/B comparison across flat/hierarchical/XML directly addresses the Statement of Work while producing a clean publishable finding.

---

## Supabase Schema

```sql
-- Run logs: full JSON blob per run
run_logs (
  run_id TEXT PRIMARY KEY,
  scenario_id TEXT,
  subject_model TEXT,
  interviewer_model TEXT,
  prompt_format TEXT,
  session_group_id TEXT,
  session_number INT,
  timestamp TIMESTAMPTZ,
  data JSONB           -- full run log
)

-- Per-rater overall scores
run_scores (
  run_id TEXT,
  rater_id TEXT,
  identity_consistency SMALLINT,
  cultural_authenticity SMALLINT,
  naturalness SMALLINT,
  information_yield SMALLINT,
  total SMALLINT,
  notes TEXT,
  UNIQUE(run_id, rater_id)
)

-- Per-turn scores
turn_scores (
  run_id TEXT,
  rater_id TEXT,
  turn INT,
  [4 metrics],
  UNIQUE(run_id, rater_id, turn)
)

-- Cross-session memory
session_memories (
  session_group_id TEXT,
  session_number INT,
  run_id TEXT,
  memory_facts JSONB,
  UNIQUE(session_group_id, session_number)
)
```

---

## Common Issues

**Run not appearing in dashboard**
- Check `logs/` for the JSON file
- If on Streamlit Cloud: run `python sync.py` locally to push to Supabase
- Or use the Upload run log expander in the sidebar

**Model returning empty string**
- Check API key is set in `.env`
- Check provider string format: `provider:model-name`
- Run with `--max-turns 3` first to test connectivity

**Validation error on existing log**
- Schema is now permissive (`extra="ignore"`, no `raw_prompt` requirement)
- Check `run_id` field exists in the JSON

**Dashboard shows 0 runs on Streamlit Cloud**
- Set `SUPABASE_URL` and `SUPABASE_KEY` in Streamlit secrets
- Run `python sync.py` locally to backfill existing runs

**Scoring CSV not writing**
- Check `logs/scoring/` directory exists (created automatically)
- Check `rater_id` is not empty in the scoring UI
