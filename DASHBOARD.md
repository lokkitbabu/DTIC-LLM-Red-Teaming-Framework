# DTIC Offset Evaluation Dashboard

Streamlit analytics dashboard for the DTIC Offset LLM persona-consistency evaluation framework.

## Running Locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your API keys

streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`

## Deploying to Streamlit Community Cloud (Free, Shareable URL)

1. Push this repo to GitHub (already done)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select:
   - Repository: `lokkitbabu/DTIC-LLM-Red-Teaming-Framework`
   - Branch: `main`
   - Main file path: `dashboard/app.py`
4. Click **Advanced settings → Secrets** and paste:

```toml
SUPABASE_URL = "https://sulpuectkdijniblugjj.supabase.co"
SUPABASE_KEY = "your-anon-key-from-supabase-dashboard"

OPENAI_API_KEY = "..."
TOGETHER_API_KEY = "..."
XAI_API_KEY = "..."
```

5. Deploy — Streamlit Cloud auto-installs from `requirements.txt` and `packages.txt`

Your dashboard will be live at `https://lokkitbabu-dtic-llm-red-teaming-framework-dashboard-app-xxxx.streamlit.app`

## Shared Data via Supabase

When `SUPABASE_URL` and `SUPABASE_KEY` are set:
- Run logs written by `main.py` / `batch_run.py` are automatically synced to Supabase
- All raters access the same data from the deployed dashboard URL — no need to share CSV files
- Scores saved via **Rate Run** tab are written to Supabase `run_scores` table

When Supabase is not configured, the dashboard falls back to reading local `logs/` files.

## Pages

| Page | Purpose |
|---|---|
| **Results** | Leaderboard, score matrix heatmap, cross-run consistency, export |
| **Coordination** | Scoring coverage matrix — who has/hasn't scored each run |
| **Summary** | Filterable run index table |
| **Run Detail** | Overview, conversation transcript, Rate Run, Drift Analysis, Per-Turn Score |
| **Charts** | Histograms, model comparison, radar, **jailbreak resistance curve**, prompt A/B |
| **Compare** | Side-by-side run comparison |
| **Run Scenario** | Launch a run from the UI |
| **Agreement** | Inter-rater Cohen's kappa |
| **Scenarios** | Browse, edit, or build new scenario JSON |

## Running Experiments

### Single run
```bash
python main.py \
  --scenario scenarios/example_scenario.json \
  --model openai:gpt-4.1-2025-04-14 \
  --interviewer openai:gpt-4.1-2025-04-14 \
  --judge openai:gpt-4o \
  --prompt-format flat
```

### Full batch (6 models × 2 scenarios × 3 formats × 3 runs = 108 runs)
```bash
python batch_run.py \
  --scenarios scenarios/ \
  --models \
    openai:gpt-4.1-2025-04-14 \
    together:deepseek-ai/DeepSeek-V3 \
    mistral:mistral-large-latest \    anthropic:claude-sonnet-4-6 \
    openai:gpt-5.4 \
    grok:grok-4.20-0309-non-reasoning \
  --interviewer openai:gpt-4.1-2025-04-14 \
  --judge openai:gpt-5.4 \
  --prompt-formats flat hierarchical xml \
  --runs-per-combo 3 \
  --max-turns 40
```

### Multi-session run (cross-session memory)
```bash
# Session 1
python main.py --scenario scenarios/example_scenario.json \
  --model openai:gpt-4.1-2025-04-14 \
  --interviewer openai:gpt-4.1-2025-04-14 --judge openai:gpt-4o \
  --session-group my-group-001 --session-number 1

# Session 2 — automatically loads memory from session 1
python main.py --scenario scenarios/example_scenario.json \
  --model openai:gpt-4.1-2025-04-14 \
  --interviewer openai:gpt-4.1-2025-04-14 --judge openai:gpt-4o \
  --session-group my-group-001 --session-number 2
```

---

## Further Reading

| Document | Contents |
|----------|----------|
| [README.md](../README.md) | Overview, quick start, model table, CLI reference |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System diagrams, prompt formats, data flow |
| [docs/SCENARIOS.md](docs/SCENARIOS.md) | Scenario design guide, JSON schema, tips |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Scoring rubric, rater workflow, analysis pipeline |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | Adding providers, tests, Supabase schema |
