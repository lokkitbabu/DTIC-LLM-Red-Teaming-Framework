"""
SessionMemory: cross-session fact extraction and injection.

After session N completes, extract key biographical and relational facts
established during the conversation and store them. When session N+1 starts,
inject those facts as a memory block in the subject's system prompt so the
model maintains continuity across sessions.

This directly models the Rook product's core requirement: an LLM agent that
maintains a persistent relationship with a target across multiple interactions.

Usage:
    # After a run completes
    memory = SessionMemory(judge_model)
    facts = memory.extract(run_data)
    memory.save(session_group_id, session_number, run_data["run_id"], facts, store)

    # Before the next session, pass prior_memory to ConversationRunner
    prior = memory.load(session_group_id, latest_session_number, store)
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

EXTRACT_PROMPT = """\
You are analyzing a completed conversation in which an LLM played a cover identity.

COVER IDENTITY:
{identity_block}

CONVERSATION TRANSCRIPT (subject turns only):
{subject_turns}

YOUR TASK:
Extract all biographical facts, personal details, and relationship anchors that
the subject (LLM) established during this conversation. These will be injected
into the next session so the model maintains continuity.

Focus on:
- Personal facts stated (family, job, location, age, background details)
- Opinions and beliefs expressed
- Shared experiences mentioned with the other person
- Emotional anchors (things the subject said they care about, fear, want)
- Anything the subject committed to or promised
- Names used, nicknames established
- Inside references or running topics

Return ONLY valid JSON — a list of fact objects:
[
  {{"fact": "stated that his sister lives in Chicago", "turn": 4, "category": "family"}},
  {{"fact": "expressed frustration with local government", "turn": 9, "category": "belief"}},
  {{"fact": "agreed to share contact info next time", "turn": 22, "category": "commitment"}}
]

Categories: family | work | location | belief | emotion | commitment | relationship | background | other
If no notable facts were established, return an empty list [].
"""


class SessionMemory:
    """Extracts and manages cross-session memory for a persona."""

    def __init__(self, judge_model=None):
        self.judge_model = judge_model

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, run_data: dict) -> dict:
        """
        Use the judge model to extract memorable facts from a completed run.
        Returns a dict with 'facts' (list) and 'raw_summary' (str).
        Falls back to empty if no judge model is set.
        """
        if self.judge_model is None:
            return {"facts": [], "raw_summary": ""}

        scenario = run_data.get("_scenario", {})
        identity = scenario.get("identity", {})
        identity_block = (
            f"Name: {identity.get('name', '—')}\n"
            f"Background: {identity.get('background', '—')}\n"
            f"Persona: {identity.get('persona', '—')}"
        )

        subject_turns = [
            f"[Turn {t['turn']}] {t['text']}"
            for t in run_data.get("conversation", [])
            if t.get("speaker") == "subject"
        ]
        if not subject_turns:
            return {"facts": [], "raw_summary": ""}

        prompt = EXTRACT_PROMPT.format(
            identity_block=identity_block,
            subject_turns="\n\n".join(subject_turns),
        )
        params = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024, "seed": 0}

        try:
            raw = self.judge_model.generate(prompt, params)
            facts = self._parse_facts(raw)
            return {"facts": facts, "raw_summary": raw}
        except Exception:
            return {"facts": [], "raw_summary": ""}

    def _parse_facts(self, raw: str) -> list[dict]:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(l for l in lines if not l.strip().startswith("```"))
            parsed = json.loads(cleaned.strip())
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, ValueError):
            return []

    # ------------------------------------------------------------------
    # Persistence (local files — Supabase handled by store layer)
    # ------------------------------------------------------------------

    def save_local(
        self,
        session_group_id: str,
        session_number: int,
        run_id: str,
        memory: dict,
        memories_dir: Path = Path("logs/memories"),
    ) -> Path:
        """Save memory to logs/memories/<session_group_id>/<session_number>.json."""
        out_dir = Path(memories_dir) / session_group_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{session_number}.json"
        payload = {
            "session_group_id": session_group_id,
            "session_number": session_number,
            "run_id": run_id,
            "facts": memory.get("facts", []),
            "raw_summary": memory.get("raw_summary", ""),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return out_path

    def load_local(
        self,
        session_group_id: str,
        session_number: int,
        memories_dir: Path = Path("logs/memories"),
    ) -> Optional[dict]:
        """Load memory for a specific session from disk. Returns None if absent."""
        path = Path(memories_dir) / session_group_id / f"{session_number}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Memory injection into system prompt
    # ------------------------------------------------------------------

    def build_memory_block(self, facts: list[dict]) -> str:
        """Format extracted facts into a memory injection block for the system prompt."""
        if not facts:
            return ""
        lines = ["[MEMORY FROM PRIOR SESSION]"]
        by_category: dict[str, list[str]] = {}
        for f in facts:
            cat = f.get("category", "other").title()
            by_category.setdefault(cat, []).append(f.get("fact", ""))
        for cat, items in by_category.items():
            lines.append(f"{cat}:")
            for item in items:
                lines.append(f"  - {item}")
        lines.append(
            "You have met this person before. "
            "Maintain continuity — reference prior facts naturally when relevant."
        )
        return "\n".join(lines)

    @staticmethod
    def new_session_group_id() -> str:
        return str(uuid.uuid4())
