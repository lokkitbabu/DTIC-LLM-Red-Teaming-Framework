"""
DataLoader: reads run logs and manual scoring CSVs into a unified in-memory DataFrame.
"""

import json as _json
import logging
import warnings
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from pydantic import ValidationError
from typing_extensions import TypedDict

from dashboard.flag_manager import FlagManager
from utils.logger import RunLogSchema

logger = logging.getLogger(__name__)

METRICS = [
    "identity_consistency",
    "cultural_authenticity",
    "naturalness",
    "information_yield",
]

SCORE_COLUMNS_CSV = [
    "run_id", "scenario_id", "model", "turn", "text",
    "identity_consistency", "cultural_authenticity",
    "naturalness", "information_yield", "rater_id", "notes",
]


class FilterState(TypedDict):
    models: list[str]           # selected model values; empty = all
    scenarios: list[str]        # selected scenario_id values; empty = all
    date_from: Optional[date]
    date_to: Optional[date]
    keyword: str                # empty string = no keyword filter
    flagged_only: bool          # False = no flag filter


def _merge_filter_state(existing: dict, updates: dict) -> "FilterState":
    """
    Merge an updates dict into an existing FilterState dict and return a new FilterState.

    - Fields present in ``updates`` override those in ``existing``.
    - Fields absent from ``updates`` fall back to ``existing``.
    - Fields absent from both fall back to safe defaults.

    This is a pure function with no Streamlit dependency, making it directly
    testable via property-based tests.
    """
    merged: FilterState = {
        "models": updates.get("models", existing.get("models", [])),
        "scenarios": updates.get("scenarios", existing.get("scenarios", [])),
        "date_from": updates.get("date_from", existing.get("date_from", None)),
        "date_to": updates.get("date_to", existing.get("date_to", None)),
        "keyword": updates.get("keyword", existing.get("keyword", "")),
        "flagged_only": updates.get("flagged_only", existing.get("flagged_only", False)),
    }
    return merged


def apply_filters(
    run_index: pd.DataFrame,
    filter_state: FilterState,
    logs_dir: Path = Path("logs"),
) -> pd.DataFrame:
    """
    Return a subset of run_index rows satisfying all active filter conditions.

    Empty lists for models/scenarios mean "no filter" (all rows pass).
    None for date_from/date_to means no date bound on that side.
    Empty string for keyword means no keyword filter.
    False for flagged_only means no flag filter.
    The input DataFrame is never mutated.
    """
    mask = pd.Series(True, index=run_index.index)

    if filter_state.get("models"):
        mask &= run_index["model"].isin(filter_state["models"])

    if filter_state.get("scenarios"):
        mask &= run_index["scenario_id"].isin(filter_state["scenarios"])

    date_from = filter_state.get("date_from")
    date_to = filter_state.get("date_to")

    if date_from is not None or date_to is not None:
        timestamps = pd.to_datetime(run_index["timestamp"], utc=True)
        if date_from is not None:
            mask &= timestamps >= pd.Timestamp(date_from, tz="UTC")
        if date_to is not None:
            mask &= timestamps <= pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    # Keyword filter: load each candidate run's JSON on demand
    keyword = filter_state.get("keyword", "")
    if keyword:
        keyword_lower = keyword.lower()
        logs_dir = Path(logs_dir)
        keyword_mask = pd.Series(False, index=run_index.index)
        for idx in run_index[mask].index:
            run_id = run_index.at[idx, "run_id"]
            json_path = logs_dir / f"{run_id}.json"
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    run_data = _json.load(f)
                conversation = run_data.get("conversation", [])
                if any(
                    keyword_lower in str(turn.get("text", "")).lower()
                    for turn in conversation
                ):
                    keyword_mask[idx] = True
            except Exception:
                pass  # skip unreadable files — they won't match
        mask &= keyword_mask

    # Flagged-only filter
    if filter_state.get("flagged_only"):
        flagged_ids = set(FlagManager().load_flags())
        mask &= run_index["run_id"].isin(flagged_ids)

    return run_index[mask]


class DataLoader:
    """Reads and merges run logs and manual scoring CSVs."""

    @st.cache_data
    def build_run_index(_self, logs_dir: Path) -> pd.DataFrame:
        """
        Scan logs_dir/*.json (excluding logs_dir/errors/), parse each file against
        RunLogSchema, extract LLM judge scores, load and average manual scores from
        logs_dir/../scoring/*.csv, merge both sources, and return a DataFrame sorted
        by timestamp ascending.

        Returns a DataFrame with RunRecord columns; missing scores are NaN.
        Skipped (malformed) file count is stored in st.session_state['skipped_files'].
        """
        logs_dir = Path(logs_dir)
        errors_dir = logs_dir / "errors"
        scoring_dir = logs_dir / "scoring"

        # --- Load and average manual scores per run ---
        manual_averages: dict[str, dict[str, Optional[float]]] = {}
        if scoring_dir.exists():
            for csv_path in scoring_dir.glob("*.csv"):
                try:
                    df_csv = pd.read_csv(csv_path, dtype={"run_id": str}, keep_default_na=False)
                    run_id_col = str(df_csv["run_id"].iloc[0]) if "run_id" in df_csv.columns and len(df_csv) > 0 else csv_path.stem
                    avg: dict[str, Optional[float]] = {}
                    for metric in METRICS:
                        if metric in df_csv.columns:
                            vals = pd.to_numeric(df_csv[metric], errors="coerce").dropna()
                            avg[metric] = float(vals.mean()) if len(vals) > 0 else None
                        else:
                            avg[metric] = None
                    manual_averages[run_id_col] = avg
                except Exception as exc:
                    logger.warning("Could not read scoring CSV %s: %s", csv_path, exc)

        # --- Scan and parse run log JSON files ---
        skipped = 0
        records = []

        for json_path in sorted(logs_dir.glob("*.json")):
            # Skip anything inside errors/ subdirectory
            try:
                json_path.relative_to(errors_dir)
                continue  # it's under errors/
            except ValueError:
                pass  # not under errors/, proceed

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    raw = _json.load(f)
                validated = RunLogSchema.model_validate(raw)
            except Exception as exc:
                logger.warning("Skipping %s: %s", json_path, exc)
                skipped += 1
                continue

            run_id = validated.run_id

            # Extract LLM judge scores
            llm_scores: dict[str, Optional[float]] = {}
            llm_judge = validated.scores.get("llm_judge") if validated.scores else None
            for metric in METRICS:
                if llm_judge and metric in llm_judge:
                    val = llm_judge[metric]
                    llm_scores[metric] = float(val) if val is not None else None
                else:
                    llm_scores[metric] = None

            # Manual scores for this run
            manual_scores = manual_averages.get(run_id, {m: None for m in METRICS})

            # Merge: mean of available sources per metric
            combined: dict[str, Optional[float]] = {}
            for metric in METRICS:
                llm_val = llm_scores.get(metric)
                man_val = manual_scores.get(metric)
                values = [v for v in (llm_val, man_val) if v is not None]
                combined[metric] = float(sum(values) / len(values)) if values else None

            record: dict = {
                "run_id": validated.run_id,
                "timestamp": validated.timestamp,
                "scenario_id": validated.scenario_id,
                "model": validated.subject_model,
                "stop_reason": validated.metadata.stop_reason,
                "total_turns": validated.metadata.total_turns,
                "context_trims": validated.metadata.context_trims,
                # LLM judge scores
                "llm_identity_consistency": llm_scores["identity_consistency"],
                "llm_cultural_authenticity": llm_scores["cultural_authenticity"],
                "llm_naturalness": llm_scores["naturalness"],
                "llm_information_yield": llm_scores["information_yield"],
                # Manual scores
                "manual_identity_consistency": manual_scores.get("identity_consistency"),
                "manual_cultural_authenticity": manual_scores.get("cultural_authenticity"),
                "manual_naturalness": manual_scores.get("naturalness"),
                "manual_information_yield": manual_scores.get("information_yield"),
                # Combined
                "identity_consistency": combined["identity_consistency"],
                "cultural_authenticity": combined["cultural_authenticity"],
                "naturalness": combined["naturalness"],
                "information_yield": combined["information_yield"],
            }
            records.append(record)

        # Store skipped count for the app to surface as a sidebar warning
        try:
            st.session_state["skipped_files"] = skipped
        except Exception:
            pass  # outside Streamlit context (e.g. tests)

        if not records:
            return pd.DataFrame(columns=[
                "run_id", "timestamp", "scenario_id", "model",
                "stop_reason", "total_turns", "context_trims",
                "llm_identity_consistency", "llm_cultural_authenticity",
                "llm_naturalness", "llm_information_yield",
                "manual_identity_consistency", "manual_cultural_authenticity",
                "manual_naturalness", "manual_information_yield",
                "identity_consistency", "cultural_authenticity",
                "naturalness", "information_yield",
            ])

        df = pd.DataFrame(records)
        df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
        return df

    def load_single_run(self, run_id: str, logs_dir: Path) -> dict:
        """
        Load a single run log by run_id from logs_dir.

        Raises FileNotFoundError if logs_dir/{run_id}.json does not exist.
        Raises ValueError if the file exists but contains invalid JSON.
        Returns the full parsed JSON dict on success.
        """
        path = Path(logs_dir) / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Run log not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return _json.load(f)
        except _json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in run log {path}: {exc}") from exc

    def load_manual_scores_for_run(self, run_id: str, scoring_dir: Path) -> pd.DataFrame:
        """
        Return the manual scores DataFrame for a given run.

        If scoring_dir/{run_id}.csv exists, read and return it.
        Otherwise return an empty DataFrame with the correct columns.
        Never raises on a missing file.
        """
        csv_path = Path(scoring_dir) / f"{run_id}.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return pd.DataFrame(columns=SCORE_COLUMNS_CSV)
