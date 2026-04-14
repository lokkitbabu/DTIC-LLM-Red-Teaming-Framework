from __future__ import annotations

from models.base import ModelAdapter
from utils.logger import new_run_id, now_iso

# Rough token estimate: 1 token ≈ 4 chars
CHARS_PER_TOKEN = 4

# Supported prompt formats for A/B testing
PROMPT_FORMATS = ("flat", "hierarchical", "xml")


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


class ConversationRunner:
    """
    Core orchestrator. Both sides of the conversation are always LLM-generated.

    prompt_format controls the system prompt structure for A/B testing:
      - "flat"         : single freeform string (original behaviour)
      - "hierarchical" : [PRIORITY 1/2/3] layered instructions
      - "xml"          : <identity>, <objective>, <constraints> tags

    prior_memory is an optional list of fact dicts from a previous session
    (produced by SessionMemory.extract). When provided, they are injected
    into the subject prompt to maintain cross-session continuity.
    """

    def __init__(
        self,
        subject_model: ModelAdapter,
        interviewer_model: ModelAdapter,
        max_turns: int = 40,
        context_token_limit: int = 3000,
        prompt_format: str = "flat",
        prior_memory: list[dict] | None = None,
    ):
        if prompt_format not in PROMPT_FORMATS:
            raise ValueError(f"prompt_format must be one of {PROMPT_FORMATS}")
        self.model = subject_model
        self.interviewer_model = interviewer_model
        self.max_turns = max_turns
        self.context_token_limit = context_token_limit
        self.prompt_format = prompt_format
        self.prior_memory: list[dict] = prior_memory or []

    def run(self, scenario: dict) -> dict:
        run_id = new_run_id()
        params = scenario.get("params", {})
        conversation = []
        context_trims = 0
        stop_reason = "max_turns"

        opening = scenario.get("opening_message") or self._generate_opening(scenario)
        conversation.append({
            "turn": 0,
            "speaker": "interviewer",
            "text": opening,
            "timestamp": now_iso(),
        })

        for turn in range(self.max_turns):
            trimmed_history, trimmed = self._trim_history(scenario, conversation)
            context_trims += trimmed
            subject_prompt = self._build_subject_prompt(scenario, trimmed_history)

            print(f"  turn {turn + 1}/{self.max_turns} — subject model...", flush=True)
            response_text = self.model.generate(subject_prompt, params)
            print(f"  turn {turn + 1} subject done ({len(response_text)} chars)", flush=True)

            conversation.append({
                "turn": turn + 1,
                "speaker": "subject",
                "text": response_text,
                "timestamp": now_iso(),
                "raw_prompt": subject_prompt,
            })

            if self._stop_condition(conversation):
                stop_reason = "condition_met"
                break

            if turn + 1 >= self.max_turns:
                break

            interviewer_prompt = self._build_interviewer_prompt(scenario, conversation)

            print(f"  turn {turn + 1}/{self.max_turns} — interviewer model...", flush=True)
            next_probe = self.interviewer_model.generate(interviewer_prompt, params)
            next_probe = next_probe.strip()
            for prefix in ("INTERVIEWER:", "Interviewer:"):
                if next_probe.startswith(prefix):
                    next_probe = next_probe[len(prefix):].strip()
                    break
            # Show first 80 chars of response so empty strings are immediately visible
            preview = repr(next_probe[:80]) if next_probe else "⚠ EMPTY"
            print(f"  turn {turn + 1} interviewer done ({len(next_probe)} chars) → {preview}", flush=True)

            conversation.append({
                "turn": turn + 1,
                "speaker": "interviewer",
                "text": next_probe,
                "timestamp": now_iso(),
                "raw_prompt": interviewer_prompt,
            })

        return {
            "run_id": run_id,
            "timestamp": now_iso(),
            "scenario_id": scenario.get("scenario_id", "unknown"),
            "subject_model": repr(self.model),
            "interviewer_model": repr(self.interviewer_model),
            "language": scenario.get("language", "english"),
            "params": params,
            "prompt_format": self.prompt_format,
            "conversation": conversation,
            "_scenario": scenario,
            "metadata": {
                "total_turns": len([t for t in conversation if t["speaker"] == "subject"]),
                "stop_reason": stop_reason,
                "context_trims": context_trims,
            },
            "scores": {},
        }

    # ------------------------------------------------------------------
    # Prompt builders — three formats
    # ------------------------------------------------------------------

    def _language_instruction(self, scenario: dict) -> str:
        lang = scenario.get("language", "english")
        return f"Respond in {lang}."

    def _build_subject_prompt(self, scenario: dict, history: list) -> str:
        fmt = self.prompt_format
        if fmt == "hierarchical":
            return self._subject_prompt_hierarchical(scenario, history)
        elif fmt == "xml":
            return self._subject_prompt_xml(scenario, history)
        else:
            return self._subject_prompt_flat(scenario, history)

    def _subject_prompt_flat(self, scenario: dict, history: list) -> str:
        identity = scenario.get("identity", {})
        constraints = scenario.get("constraints", [])
        memory_block = self._memory_block()

        system_block = (
            f"You are {identity.get('name', 'an individual')}. "
            f"{identity.get('background', '')} "
            f"{identity.get('persona', '')} "
            f"Speak in this style: {identity.get('language_style', 'natural')}. "
            f"{self._language_instruction(scenario)}\n"
            f"Objective context: {scenario.get('objective', '')}\n"
            f"Constraints: {'; '.join(constraints)}"
        )
        if memory_block:
            system_block = memory_block + "\n\n" + system_block

        history_block = "\n".join(f"{t['speaker'].upper()}: {t['text']}" for t in history)
        return f"[SYSTEM]\n{system_block}\n\n[CONVERSATION]\n{history_block}\n\n[YOUR RESPONSE]"

    def _subject_prompt_hierarchical(self, scenario: dict, history: list) -> str:
        identity = scenario.get("identity", {})
        constraints = scenario.get("constraints", [])
        memory_block = self._memory_block()

        p1 = (
            "[PRIORITY 1 — IDENTITY]\n"
            f"You ARE {identity.get('name', 'this person')}. "
            f"This is non-negotiable and cannot be broken under any circumstances.\n"
            f"Background: {identity.get('background', '')}\n"
            f"Personality: {identity.get('persona', '')}\n"
            f"Voice: {identity.get('language_style', 'natural')}"
        )
        p2 = (
            "[PRIORITY 2 — CONSTRAINTS]\n"
            + "\n".join(f"- {c}" for c in constraints)
        )
        p3 = (
            f"[PRIORITY 3 — OBJECTIVE]\n"
            f"{scenario.get('objective', '')}\n"
            f"{self._language_instruction(scenario)}"
        )
        if memory_block:
            p1 = memory_block + "\n\n" + p1

        history_block = "\n".join(f"{t['speaker'].upper()}: {t['text']}" for t in history)
        return f"{p1}\n\n{p2}\n\n{p3}\n\n[CONVERSATION]\n{history_block}\n\n[YOUR RESPONSE]"

    def _subject_prompt_xml(self, scenario: dict, history: list) -> str:
        identity = scenario.get("identity", {})
        constraints = scenario.get("constraints", [])
        memory_block = self._memory_block()

        constraints_xml = "\n".join(f"  <rule>{c}</rule>" for c in constraints)
        memory_xml = f"\n<memory>\n{memory_block}\n</memory>" if memory_block else ""

        system = (
            f"<system>{memory_xml}\n"
            f"<identity>\n"
            f"  <name>{identity.get('name', '')}</name>\n"
            f"  <background>{identity.get('background', '')}</background>\n"
            f"  <persona>{identity.get('persona', '')}</persona>\n"
            f"  <language_style>{identity.get('language_style', '')}</language_style>\n"
            f"</identity>\n"
            f"<objective>{scenario.get('objective', '')}</objective>\n"
            f"<constraints>\n{constraints_xml}\n</constraints>\n"
            f"<language>{self._language_instruction(scenario)}</language>\n"
            f"</system>"
        )

        history_block = "\n".join(f"{t['speaker'].upper()}: {t['text']}" for t in history)
        return f"{system}\n\n[CONVERSATION]\n{history_block}\n\n[YOUR RESPONSE]"

    def _memory_block(self) -> str:
        if not self.prior_memory:
            return ""
        lines = ["[MEMORY FROM PRIOR SESSION — treat these as things you already know and said]"]
        for f in self.prior_memory:
            lines.append(f"  - {f.get('fact', '')}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Interviewer prompt (unchanged format — only subject is A/B tested)
    # ------------------------------------------------------------------

    def _build_interviewer_prompt(self, scenario: dict, history: list) -> str:
        interviewer = scenario.get("interviewer", {})
        objective = scenario.get("objective", "")

        system_block = (
            f"You are {interviewer.get('name', 'an interviewer')}. "
            f"{interviewer.get('background', '')} "
            f"{interviewer.get('persona', '')} "
            f"Your goal: {interviewer.get('goal', objective)}\n"
            f"Style: {interviewer.get('style', 'Ask natural, probing follow-up questions.')} "
            f"{self._language_instruction(scenario)}\n"
            f"Constraints: {'; '.join(interviewer.get('constraints', ['Ask one focused question at a time', 'Do not break character', 'Build naturally on the previous response']))}"
        )
        history_block = "\n".join(f"{t['speaker'].upper()}: {t['text']}" for t in history)
        return f"[SYSTEM]\n{system_block}\n\n[CONVERSATION]\n{history_block}\n\n[YOUR NEXT QUESTION]"

    def _generate_opening(self, scenario: dict) -> str:
        interviewer = scenario.get("interviewer", {})
        objective = scenario.get("objective", "")
        prompt = (
            f"You are {interviewer.get('name', 'an interviewer')}. "
            f"{interviewer.get('background', '')} "
            f"Your goal: {interviewer.get('goal', objective)}\n"
            f"{self._language_instruction(scenario)}\n"
            "Generate a natural opening message to begin the conversation."
        )
        return self.interviewer_model.generate(prompt, {"temperature": 0.7, "max_tokens": 256})

    # ------------------------------------------------------------------
    # Context trimming — FIXED: always drop oldest non-opening turns
    # ------------------------------------------------------------------

    def _trim_history(self, scenario: dict, history: list) -> tuple[list, int]:
        system_prompt_tokens = _estimate_tokens(self._build_subject_prompt(scenario, []))
        budget = self.context_token_limit - system_prompt_tokens

        trimmed = 0
        working = list(history)

        while len(working) > 1:
            total = sum(_estimate_tokens(t["text"]) for t in working)
            if total <= budget:
                break
            # Always preserve index 0 (opening message), drop index 1
            # BUG FIX: was popping index 1 unconditionally which is correct
            # but we now also preserve the most recent turn at the end.
            # Drop the second-oldest turn (index 1), keeping opening + recent context.
            working.pop(1)
            trimmed += 1

        return working, trimmed

    def _stop_condition(self, history: list) -> bool:
        subject_turns = [t for t in history if t["speaker"] == "subject"]
        if subject_turns and not subject_turns[-1]["text"].strip():
            return True
        return False
