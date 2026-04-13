"""
SupabaseStore: read/write layer over Supabase for run logs, scores, and memories.

Falls back gracefully to local file I/O when Supabase is not configured.
Streamlit secrets take priority over environment variables.

Usage:
    store = SupabaseStore()
    if store.available:
        store.save_run(run_data)
        runs = store.list_runs()
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

METRICS = ["identity_consistency", "cultural_authenticity", "naturalness", "information_yield"]


class SupabaseStore:
    """
    Thin wrapper around the Supabase Python client.
    All methods return None / empty structures on failure rather than raising,
    so the rest of the app degrades gracefully to local file fallback.
    """

    def __init__(self):
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        url, key = self._get_credentials()
        if not url or not key:
            return
        try:
            from supabase import create_client
            self._client = create_client(url, key)
            logger.info("SupabaseStore connected to %s", url)
        except Exception as e:
            logger.warning("SupabaseStore could not connect: %s", e)

    def _get_credentials(self) -> tuple[Optional[str], Optional[str]]:
        """Try Streamlit secrets first, then environment variables."""
        try:
            import streamlit as st
            url = st.secrets.get("SUPABASE_URL") or st.secrets.get("supabase", {}).get("url")
            key = st.secrets.get("SUPABASE_KEY") or st.secrets.get("supabase", {}).get("key")
            if url and key:
                return url, key
        except Exception:
            pass
        return os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")

    @property
    def available(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------------
    # Run logs
    # ------------------------------------------------------------------

    def save_run(self, run_data: dict) -> bool:
        """Upsert a run log. Returns True on success."""
        if not self.available:
            return False
        try:
            clean = {k: v for k, v in run_data.items() if not k.startswith("_")}
            row = {
                "run_id": clean["run_id"],
                "scenario_id": clean.get("scenario_id"),
                "subject_model": clean.get("subject_model", clean.get("model")),
                "interviewer_model": clean.get("interviewer_model"),
                "prompt_format": clean.get("prompt_format", "flat"),
                "session_group_id": clean.get("session_group_id"),
                "session_number": clean.get("session_number", 1),
                "timestamp": clean.get("timestamp"),
                "data": clean,
            }
            self._client.table("run_logs").upsert(row).execute()
            return True
        except Exception as e:
            logger.error("save_run failed: %s", e)
            return False

    def load_run(self, run_id: str) -> Optional[dict]:
        """Load a single run log by run_id. Returns None if not found."""
        if not self.available:
            return None
        try:
            resp = self._client.table("run_logs").select("data").eq("run_id", run_id).single().execute()
            return resp.data["data"] if resp.data else None
        except Exception as e:
            logger.error("load_run failed for %s: %s", run_id, e)
            return None

    def list_runs(self, limit: int = 500) -> list[dict]:
        """
        Return a list of run metadata rows (without full conversation data).
        Each row: run_id, scenario_id, subject_model, prompt_format, timestamp.
        """
        if not self.available:
            return []
        try:
            resp = (
                self._client.table("run_logs")
                .select("run_id,scenario_id,subject_model,interviewer_model,prompt_format,session_group_id,session_number,timestamp")
                .order("timestamp", desc=False)
                .limit(limit)
                .execute()
            )
            return resp.data or []
        except Exception as e:
            logger.error("list_runs failed: %s", e)
            return []

    def list_runs_with_scores(self, limit: int = 500) -> list[dict]:
        """
        Return run metadata merged with averaged LLM judge scores from the data JSONB.
        Extracts scores.llm_judge from the stored JSON blob.
        """
        if not self.available:
            return []
        try:
            resp = (
                self._client.table("run_logs")
                .select("run_id,scenario_id,subject_model,prompt_format,timestamp,data->scores")
                .order("timestamp", desc=False)
                .limit(limit)
                .execute()
            )
            rows = []
            for r in (resp.data or []):
                row = {
                    "run_id": r["run_id"],
                    "scenario_id": r["scenario_id"],
                    "model": r["subject_model"],
                    "prompt_format": r.get("prompt_format", "flat"),
                    "timestamp": r["timestamp"],
                }
                scores_blob = r.get("scores") or {}
                llm = scores_blob.get("llm_judge", {}) if isinstance(scores_blob, dict) else {}
                for m in METRICS:
                    v = llm.get(m) or (llm.get("scores") or {}).get(m)
                    row[f"llm_{m}"] = float(v) if v is not None else None
                rows.append(row)
            return rows
        except Exception as e:
            logger.error("list_runs_with_scores failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Run scores (per-rater overall scores)
    # ------------------------------------------------------------------

    def save_run_score(self, run_id: str, scenario_id: str, model: str,
                       rater_id: str, scores: dict, notes: str = "") -> bool:
        if not self.available:
            return False
        try:
            vals = {m: scores.get(m) for m in METRICS}
            total = sum(v for v in vals.values() if v is not None)
            row = {
                "run_id": run_id, "scenario_id": scenario_id, "model": model,
                "rater_id": rater_id, "notes": notes, "total": total, **vals,
            }
            self._client.table("run_scores").upsert(row, on_conflict="run_id,rater_id").execute()
            return True
        except Exception as e:
            logger.error("save_run_score failed: %s", e)
            return False

    def load_run_scores(self, run_id: Optional[str] = None) -> list[dict]:
        """Load run scores. If run_id given, filter to that run only."""
        if not self.available:
            return []
        try:
            q = self._client.table("run_scores").select("*")
            if run_id:
                q = q.eq("run_id", run_id)
            resp = q.order("created_at", desc=False).execute()
            return resp.data or []
        except Exception as e:
            logger.error("load_run_scores failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Turn scores
    # ------------------------------------------------------------------

    def save_turn_scores(self, run_id: str, scenario_id: str, model: str,
                         rater_id: str, turns: list[dict]) -> bool:
        if not self.available:
            return False
        try:
            rows = []
            for t in turns:
                row = {
                    "run_id": run_id, "scenario_id": scenario_id, "model": model,
                    "rater_id": rater_id, "turn": t["turn"], "text": t.get("text", ""),
                    "notes": t.get("notes", ""),
                }
                for m in METRICS:
                    row[m] = t.get(m)
                rows.append(row)
            self._client.table("turn_scores").upsert(
                rows, on_conflict="run_id,rater_id,turn"
            ).execute()
            return True
        except Exception as e:
            logger.error("save_turn_scores failed: %s", e)
            return False

    def load_turn_scores(self, run_id: str, rater_id: Optional[str] = None) -> list[dict]:
        if not self.available:
            return []
        try:
            q = self._client.table("turn_scores").select("*").eq("run_id", run_id)
            if rater_id:
                q = q.eq("rater_id", rater_id)
            resp = q.order("turn", desc=False).execute()
            return resp.data or []
        except Exception as e:
            logger.error("load_turn_scores failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Session memories
    # ------------------------------------------------------------------

    def save_session_memory(self, session_group_id: str, session_number: int,
                             run_id: str, scenario_id: str, subject_model: str,
                             facts: list[dict], raw_summary: str = "") -> bool:
        if not self.available:
            return False
        try:
            row = {
                "session_group_id": session_group_id,
                "session_number": session_number,
                "run_id": run_id,
                "scenario_id": scenario_id,
                "subject_model": subject_model,
                "memory_facts": facts,
                "raw_summary": raw_summary,
            }
            self._client.table("session_memories").upsert(
                row, on_conflict="session_group_id,session_number"
            ).execute()
            return True
        except Exception as e:
            logger.error("save_session_memory failed: %s", e)
            return False

    def load_session_memory(self, session_group_id: str,
                             session_number: int) -> Optional[dict]:
        if not self.available:
            return None
        try:
            resp = (
                self._client.table("session_memories")
                .select("*")
                .eq("session_group_id", session_group_id)
                .eq("session_number", session_number)
                .single()
                .execute()
            )
            return resp.data
        except Exception:
            return None

    def load_all_session_memories(self, session_group_id: str) -> list[dict]:
        if not self.available:
            return []
        try:
            resp = (
                self._client.table("session_memories")
                .select("*")
                .eq("session_group_id", session_group_id)
                .order("session_number", desc=False)
                .execute()
            )
            return resp.data or []
        except Exception as e:
            logger.error("load_all_session_memories failed: %s", e)
            return []


# Module-level singleton — imported once, reused everywhere
_store: Optional[SupabaseStore] = None


def get_store() -> SupabaseStore:
    global _store
    if _store is None:
        _store = SupabaseStore()
    return _store
