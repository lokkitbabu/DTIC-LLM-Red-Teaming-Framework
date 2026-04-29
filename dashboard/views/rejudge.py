"""
RejudgeView: bulk re-score runs with a different judge model.
Stores each judge's scores separately — keeps all versions, never overwrites.
"""

from __future__ import annotations
from dashboard.display_utils import METRICS as _METRICS

import json
import threading
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

_METRICS = [
    "identity_consistency",
    "cultural_authenticity",
    "naturalness",
    "information_yield",
]

_JUDGE_PRESETS = [
    "openai:gpt-5.4",
    "openai:gpt-4.1-2025-04-14",
    "anthropic:claude-sonnet-4-6",
    "grok:grok-4-1-fast-reasoning",
]


def render_rejudge_view(
    run_index: pd.DataFrame,
    logs_dir: Path = Path("logs"),
) -> None:
    st.subheader("Bulk Re-Judge")
    st.caption(
        "Re-score runs with a different judge model. "
        "Each judge's scores are stored separately — original scores are never overwritten."
    )

    if run_index.empty:
        st.info("No runs available.")
        return

    # ── Judge model selector ─────────────────────────────────────────────────
    st.markdown("#### 1 — Choose judge model")
    cols = st.columns(len(_JUDGE_PRESETS))
    for col, preset in zip(cols, _JUDGE_PRESETS):
        label = preset.split(":")[-1].split("/")[-1][:18]
        if col.button(label, key=f"rj_preset_{preset}"):
            st.session_state["rj_judge_model"] = preset

    judge_model = st.text_input(
        "Judge model",
        value=st.session_state.get("rj_judge_model", "openai:gpt-5.4"),
        key="rj_judge_model",
        label_visibility="collapsed",
    )

    eval_target = st.radio(
        "Evaluate",
        ["subject", "interviewer"],
        horizontal=True,
        key="rj_eval_target",
    )

    eval_prompt = st.radio(
        "Eval prompt",
        ["strict", "standard", "lenient"],
        horizontal=True,
        key="rj_eval_prompt",
        help="strict = harder to score high, penalises AI-isms and generic responses (recommended). standard = balanced. lenient = upper bound.",
    )

    use_dual_judge = st.checkbox(
        "Use dual judge (GPT-5.4 + Claude Sonnet 4.6, scores averaged)",
        value=True,
        key="rj_dual_judge",
    )

    # ── Run selection ────────────────────────────────────────────────────────
    st.markdown("#### 2 — Select runs")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        scenarios = ["All"] + sorted(run_index["scenario_id"].unique().tolist())
        sel_scenario = st.selectbox("Filter scenario", scenarios, key="rj_scenario")
    with col_b:
        models = ["All"] + sorted(run_index["model"].unique().tolist())
        sel_model = st.selectbox("Filter model", models, key="rj_model")

    df = run_index.copy()
    if sel_scenario != "All":
        df = df[df["scenario_id"] == sel_scenario]
    if sel_model != "All":
        df = df[df["model"] == sel_model]

    if df.empty:
        st.info("No runs match filters.")
        return

    col_sel_all, col_desel_all, _ = st.columns([1, 1, 4])
    if col_sel_all.button("Select all", key="rj_sel_all"):
        for rid in df["run_id"].tolist():
            st.session_state[f"rj_chk_{rid}"] = True
    if col_desel_all.button("Deselect all", key="rj_desel_all"):
        for rid in df["run_id"].tolist():
            st.session_state[f"rj_chk_{rid}"] = False

    selected_ids: list[str] = []
    for _, row in df.iterrows():
        run_id = row["run_id"]
        short = run_id[:8]
        model_label = str(row.get("model", "")).split("/")[-1].split("(model=")[-1].rstrip(")")[:24]
        scenario = str(row.get("scenario_id", ""))[:20]
        fmt = str(row.get("prompt_format", ""))

        # Show existing judges for this run
        existing_judges = _get_existing_judges(run_id)
        judge_badge = f" · judges: {', '.join(existing_judges)}" if existing_judges else ""

        if st.checkbox(
            f"`{short}…`  {model_label}  ·  {scenario}  ·  {fmt}{judge_badge}",
            key=f"rj_chk_{run_id}",
        ):
            selected_ids.append(run_id)

    if not selected_ids:
        st.info("Select at least one run.")
        return

    # ── Launch ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.info(f"**{len(selected_ids)} run(s)** will be judged by `{judge_model}`")

    rj_state = st.session_state.get("rj_state")
    is_running = rj_state is not None and rj_state.get("running", False)

    if st.button(
        f"⚖ Start re-judging ({len(selected_ids)} runs)",
        type="primary",
        key="rj_launch",
        disabled=is_running,
    ):
        _start_rejudge(
            selected_ids, judge_model, eval_target, logs_dir,
            eval_prompt=st.session_state.get("rj_eval_prompt", "strict"),
            dual_judge=st.session_state.get("rj_dual_judge", True),
        )
        st.rerun()

    # ── Progress ─────────────────────────────────────────────────────────────
    _render_rejudge_status()

    # ── Historical judge scores table ────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### All judge scores on record")
    _render_judge_scores_table(run_index)


@st.fragment(run_every=1)
def _render_rejudge_status() -> None:
    rj_state = st.session_state.get("rj_state")
    if not rj_state:
        return

    is_running = rj_state.get("running", False)
    completed = rj_state.get("completed", 0)
    total = rj_state.get("total", 1)
    lines = rj_state.get("lines", [])
    error = rj_state.get("error")

    if is_running:
        st.progress(completed / total, text=f"⚖ Judging {completed}/{total}…")
    elif error:
        st.error(f"Failed: {error}")
    else:
        st.success(f"✅ Done — {completed}/{total} runs judged")

    if lines:
        with st.expander("Log", expanded=is_running):
            st.code("\n".join(lines[-30:]), language=None)


def _start_rejudge(run_ids: list[str], judge_model_str: str, eval_target: str, logs_dir: Path, eval_prompt: str = 'strict', dual_judge: bool = True) -> None:
    state = {
        "running": True,
        "completed": 0,
        "total": len(run_ids),
        "lines": [],
        "error": None,
    }
    st.session_state["rj_state"] = state

    t = threading.Thread(
        target=_rejudge_worker,
        args=(run_ids, judge_model_str, eval_target, logs_dir, state, eval_prompt, dual_judge),
        daemon=True,
    )
    t.start()


def _rejudge_worker(
    run_ids: list[str],
    judge_model_str: str,
    eval_target: str,
    logs_dir: Path,
    state: dict,
    eval_prompt: str = 'strict',
    dual_judge: bool = True,
) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from main import build_model
    from evaluation.llm_judge import LLMJudge

    try:
        # Build judge(s) — dual judge averages GPT-5.4 + Claude Sonnet
        if dual_judge:
            judge_models = [
                build_model(judge_model_str),
                build_model("anthropic:claude-sonnet-4-6") if "gpt" in judge_model_str else build_model("openai:gpt-5.4"),
            ]
            state["lines"].append(f"  Using dual judge: {judge_model_str.split(':')[-1]} + {'claude-sonnet-4-6' if 'gpt' in judge_model_str else 'gpt-5.4'}")
        else:
            judge_models = [build_model(judge_model_str)]
        judge = LLMJudge(judge_models, eval_target=eval_target, prompt_name=eval_prompt)
        state["lines"].append(f"  Eval prompt: {eval_prompt}")
    except Exception as e:
        state["error"] = f"Failed to init judge: {e}"
        state["running"] = False
        return

    for run_id in run_ids:
        try:
            # Load run data
            run_data = _load_run_data(run_id, logs_dir)
            if not run_data:
                state["lines"].append(f"  ✗ {run_id[:8]}… — not found")
                state["completed"] += 1
                continue

            state["lines"].append(f"  ⚖ {run_id[:8]}…  judging with {judge_model_str.split(':')[-1]}…")

            result = judge.evaluate(run_data)
            scores = result.get("scores", result) if isinstance(result, dict) else {}
            reasoning = result.get("reasoning", "") if isinstance(result, dict) else ""

            # Save to Supabase judge_scores table
            _save_judge_score(run_id, judge_model_str, scores, reasoning, prompt_name=eval_prompt)

            # Also update run JSON locally
            _update_run_json(run_id, judge_model_str, result, logs_dir)

            total = sum(
                v for k, v in scores.items()
                if k in _METRICS and isinstance(v, (int, float))
            )
            state["lines"].append(
                f"  ✓ {run_id[:8]}…  total={total}/20  "
                f"IC={scores.get('identity_consistency','?')} "
                f"CA={scores.get('cultural_authenticity','?')} "
                f"N={scores.get('naturalness','?')} "
                f"IY={scores.get('information_yield','?')}"
            )

        except Exception as e:
            state["lines"].append(f"  ✗ {run_id[:8]}… — {str(e)[:80]}")

        state["completed"] += 1

    state["running"] = False
    state["lines"].append(f"\nDone — {state['completed']}/{state['total']} runs processed.")


def _load_run_data(run_id: str, logs_dir: Path) -> Optional[dict]:
    local = logs_dir / f"{run_id}.json"
    if local.exists():
        try:
            return json.loads(local.read_text())
        except Exception:
            pass
    try:
        from dashboard.supabase_store import get_store
        store = get_store()
        if store.available:
            data = store.load_run(run_id)
            if data:
                return data.get("data") or data
    except Exception:
        pass
    return None


def _save_judge_score(run_id: str, judge_model: str, scores: dict, reasoning: str, prompt_name: str = "strict") -> None:
    try:
        from dashboard.supabase_store import get_store
        store = get_store()
        if not store.available:
            return
        row = {
            "run_id": run_id,
            "judge_model": judge_model,
            "prompt_name": prompt_name,
            "identity_consistency": int(scores.get("identity_consistency") or 0) or None,
            "cultural_authenticity": int(scores.get("cultural_authenticity") or 0) or None,
            "naturalness": int(scores.get("naturalness") or 0) or None,
            "information_yield": int(scores.get("information_yield") or 0) or None,
            "total": sum(
                int(scores.get(m) or 0) for m in _METRICS if scores.get(m)
            ),
            "reasoning": str(reasoning)[:2000],
        }
        store._client.table("judge_scores").upsert(
            row, on_conflict="run_id,judge_model,prompt_name"
        ).execute()
    except Exception as e:
        print(f"  [save_judge_score] {e}", flush=True)


def _update_run_json(run_id: str, judge_model: str, result: dict, logs_dir: Path) -> None:
    """Append judge result under scores.judges.{judge_model} in the local JSON."""
    local = logs_dir / f"{run_id}.json"
    if not local.exists():
        return
    try:
        data = json.loads(local.read_text())
        scores = data.setdefault("scores", {})
        judges = scores.setdefault("judges", {})
        judges[judge_model] = result
        # Keep backward-compatible llm_judge key pointing to first/primary judge
        if "llm_judge" not in scores:
            scores["llm_judge"] = result
        local.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _get_existing_judges(run_id: str) -> list[str]:
    try:
        from dashboard.supabase_store import get_store
        store = get_store()
        if store.available:
            resp = store._client.table("judge_scores").select("judge_model").eq("run_id", run_id).execute()
            return [r["judge_model"].split(":")[-1].split("/")[-1][:16] for r in (resp.data or [])]
    except Exception:
        pass
    return []


def _render_judge_scores_table(run_index: pd.DataFrame) -> None:
    try:
        from dashboard.supabase_store import get_store
        store = get_store()
        if not store.available:
            st.caption("Supabase not configured.")
            return

        resp = store._client.table("judge_scores").select(
            "run_id, judge_model, prompt_name, identity_consistency, cultural_authenticity, naturalness, information_yield, total, created_at"
        ).order("created_at", desc=True).limit(200).execute()

        if not resp.data:
            st.caption("No judge scores recorded yet.")
            return

        df = pd.DataFrame(resp.data)
        df["run_id"] = df["run_id"].str[:8] + "…"
        df["judge_model"] = df["judge_model"].apply(lambda x: str(x).split(":")[-1].split("/")[-1][:24])
        df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
        df = df.rename(columns={
            "identity_consistency": "IC",
            "cultural_authenticity": "CA",
            "naturalness": "Nat",
            "information_yield": "IY",
        })
        st.dataframe(df, width="stretch", hide_index=True)

    except Exception as e:
        st.caption(f"Could not load judge scores: {e}")
