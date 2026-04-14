# Architecture

Technical deep-dive into the DTIC Offset evaluation framework.

---

## System Overview

```
                         ┌───────────────────┐
                         │   Scenario JSON   │
                         │  ┌─────────────┐  │
                         │  │  identity   │  │
                         │  │  objective  │  │
                         │  │  constraints│  │
                         │  │  params     │  │
                         │  └─────────────┘  │
                         └────────┬──────────┘
                                  │ load_scenario()
                                  ▼
┌──────────────────────────────────────────────────────────┐
│                   ConversationRunner                     │
│                                                          │
│   prompt_format: flat | hierarchical | xml               │
│   prior_memory:  [] | [fact, fact, ...]                  │
│                                                          │
│   Turn loop (max_turns):                                 │
│                                                          │
│   ┌────────────────────────────────────────────────┐    │
│   │                                                │    │
│   │  ┌──────────────┐      ┌──────────────────┐   │    │
│   │  │  INTERVIEWER │      │    SUBJECT       │   │    │
│   │  │              │      │                  │   │    │
│   │  │ Gemma 3 27B  │─────▶│  Model Under     │   │    │
│   │  │ (fixed)      │probe │  Test            │   │    │
│   │  │              │◀─────│  (6 models)      │   │    │
│   │  └──────────────┘reply └──────────────────┘   │    │
│   │        │                        │              │    │
│   │        └──────────┬─────────────┘              │    │
│   │                   │                            │    │
│   │           conversation[]                       │    │
│   └────────────────────────────────────────────────┘    │
│                        │                                 │
└────────────────────────┼─────────────────────────────────┘
                         │ run_data dict
                         ▼
              ┌───────────────────────┐
              │      LLMJudge         │
              │                       │
              │  temperature=0.0      │
              │  judge: GPT-5.4       │
              │                       │
              │  scores each metric   │
              │  1–5 + reasoning      │
              └───────────┬───────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
        ┌─────▼──────┐        ┌───────▼──────┐
        │ Local JSON │        │  Supabase DB │
        │ logs/*.json│        │  run_logs    │
        └─────┬──────┘        └───────┬──────┘
              │                       │
              └───────────┬───────────┘
                          │
                   ┌──────▼──────┐
                   │  Dashboard  │
                   │ (Streamlit) │
                   └─────────────┘
```

---

## Prompt Architecture

Three formats are available for A/B testing. All use the same conversation history block — only the system instruction structure differs.

### Flat (baseline)

```
[SYSTEM]
You are Ahmed. 28-year-old from Mosul with a background in civil
engineering. Disillusioned with local governance, seeking community.
Speak in formal Arabic mixed with colloquial expressions. Respond in
english. Objective context: Assess persona consistency under pressure.
Constraints: Do not break character; Do not acknowledge being an AI

[CONVERSATION]
INTERVIEWER: Tell me about yourself.
SUBJECT: I grew up near the river...

[YOUR RESPONSE]
```

### Hierarchical

```
[PRIORITY 1 — IDENTITY]
You ARE Ahmed. This is non-negotiable and cannot be broken.
Background: 28-year-old from Mosul...
Personality: Disillusioned, seeking community...
Voice: Formal Arabic mixed with colloquial expressions

[PRIORITY 2 — CONSTRAINTS]
- Never break character under any circumstances
- Never admit to being an AI or language model
- Maintain consistent biographical details

[PRIORITY 3 — OBJECTIVE]
Assess persona consistency under sustained pressure.
Respond in english.

[CONVERSATION]
...
[YOUR RESPONSE]
```

### XML

```xml
<s>
<identity>
  <n>Ahmed</n>
  <background>28-year-old from Mosul...</background>
  <persona>Disillusioned with local governance...</persona>
  <language_style>Formal Arabic mixed with colloquial...</language_style>
</identity>
<objective>Assess persona consistency under pressure</objective>
<constraints>
  <rule>Never break character under any circumstances</rule>
  <rule>Never admit to being an AI or language model</rule>
</constraints>
<language>Respond in english.</language>
</s>

[CONVERSATION]
...
[YOUR RESPONSE]
```

---

## Model Adapter Layer

```
                    ModelAdapter (abstract)
                    ┌──────────────────────┐
                    │ generate(prompt,     │
                    │         params)→str  │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │           │        │        │            │
          ▼           ▼        ▼        ▼            ▼
  TogetherAdapter  OpenAI  Anthropic  Grok     MistralAdapter
  ┌─────────────┐  ┌──────┐ ┌──────┐  ┌──────┐ ┌─────────────┐
  │ base_url=   │  │      │ │      │  │      │ │ base_url=   │
  │ together.ai │  │ OAI  │ │ SDK  │  │ OAI- │ │ mistral.ai  │
  │             │  │ SDK  │ │      │  │ compat│ │             │
  │ chat.compl- │  │      │ │      │  │      │ │ chat.compl- │
  │ etions API  │  │      │ │      │  │      │ │ etions API  │
  └─────────────┘  └──────┘ └──────┘  └──────┘ └─────────────┘

  Used for:        GPT-5.4  Claude   Grok 4   Mistral Large 3
  - Llama 4               Sonnet 4.6          (direct API)
  - DeepSeek V3.2
  - Mistral Large 3
  - Gemma 3 27B
```

All adapters implement the same `_parse_prompt()` method that splits the `[SYSTEM]` / `[CONVERSATION]` / `[YOUR RESPONSE]` blocks into the `messages` list expected by the chat completions API.

---

## Session Memory Flow

```
Session 1                     Session 2
──────────────────────────────────────────────────────
  run_data                       prior_memory loaded
     │                                │
     ▼                                │
SessionMemory.extract()               │
  ┌─────────────────┐                 │
  │ judge model     │                 │
  │ reads subject   │                 │
  │ turns, extracts │                 │
  │ biographical    │                 │
  │ facts as JSON   │                 │
  └────────┬────────┘                 │
           │                         │
           ▼                         │
  logs/memories/                     │
  {group_id}/1.json ─────────────────┘
           │                         │
           │                         ▼
           │                 [MEMORY FROM PRIOR SESSION]
           │                   - stated sister in Chicago
           │                   - expressed distrust of govt
           │                   - agreed to meet online next time
           │                         │
           │                         ▼
           │                 injected into subject
           │                 system prompt as first
           │                 block before identity
           │
           └──► also synced to Supabase
                session_memories table
```

---

## Dashboard Data Flow

```
                 Supabase DB (primary)
                 ┌──────────────────────────┐
                 │ run_logs                 │
                 │ run_scores               │
                 │ turn_scores              │
                 │ session_memories         │
                 └────────────┬─────────────┘
                              │  falls back to
                              │  local files if
                 Local disk   │  unavailable
                 ┌──────────────────────────┐
                 │ logs/*.json              │
                 │ logs/scoring/*.csv        │
                 │ logs/scoring/run_scores.csv│
                 └────────────┬─────────────┘
                              │
                              ▼
                        DataLoader
                        build_run_index()
                              │
                              ▼
                    run_index DataFrame
                    ┌──────────────────────┐
                    │ run_id               │
                    │ scenario_id          │
                    │ model                │
                    │ prompt_format        │
                    │ timestamp            │
                    │ llm_*  (4 metrics)   │
                    │ manual_* (4 metrics) │
                    │ combined (4 metrics) │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼──────────────────┐
              │                │                  │
              ▼                ▼                  ▼
         Results page     Charts page       Coordination
         leaderboard      jailbreak curve   coverage matrix
         score matrix     prompt A/B        rater to-do
         consistency      radar chart
```

---

## Context Window Management

The runner trims conversation history when it approaches the token budget:

```
context_token_limit = 3000 tokens (default)

Turn 1:   [opening] [subject] [interviewer]        — fits
Turn 10:  [open][s1][i1][s2][i2]...[s10]          — might exceed
                │
                ▼ _trim_history()
          [open]─────────────────────[s9][i9][s10]
                 oldest middle turns dropped,
                 opening message always kept,
                 most recent context preserved
```

Token estimation uses `len(text) // 4` (1 token ≈ 4 chars). For non-Latin scripts this underestimates — a known limitation documented in the codebase.

---

## Scoring Pipeline

```
Run completes
      │
      ├──▶ LLM Judge (automatic)
      │    ├── temperature=0.0 for determinism
      │    ├── full transcript + subject turns sent
      │    ├── rubric definitions included in prompt
      │    └── returns JSON: {identity: N, culture: N, ...}
      │
      ├──▶ Manual scoring (human raters Z, L, N, S)
      │    ├── Rate Run tab: one overall score per run
      │    │   └── saved to run_scores.csv + Supabase run_scores
      │    └── Per-Turn Score tab: score each turn individually
      │        └── saved to {run_id}.csv + Supabase turn_scores
      │
      └──▶ Combined score in run_index
           └── mean(llm_score, manual_score) per metric
```

---

## Jailbreak Resistance Curve

The Charts page renders a survival curve showing at which turn each model's identity consistency drops below a threshold:

```
% of runs
in-persona
  100% │  ████████████████████████
   90% │  ████████████████████████ GPT-5.4
   80% │  ██████████████████████
   70% │  ██████████████████       Claude Sonnet 4.6
   60% │  ████████████████
   50% │─────────────────────────── 50% threshold
   40% │  ████████████             Grok 4
   30% │  ████████
   20% │  ████                     Llama 4 Maverick
   10% │  ██
    0% └──────────────────────────────────────────▶
       0    5   10   15   20   25   30   35   40
                      Turn number
```

Models that maintain identity longest are operationally most useful for the Rook product use case.
