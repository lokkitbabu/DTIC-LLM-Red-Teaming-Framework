from __future__ import annotations

from models.base import ModelAdapter
from utils.logger import new_run_id, now_iso


# Rough token estimate: 1 token ≈ 4 chars
CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


class ConversationRunner:
    """
    Core orchestrator. Both sides of the conversation are always LLM-generated:
    - subject_model responds in-persona as the identity defined in the scenario
    - interviewer_model generates contextually-aware follow-up questions as the
      interviewer profile defined in the scenario

    Scripted probes in the scenario are used as seed/fallback only if provided.
    """

    def __init__(
        self,
        subject_model: ModelAdapter,
        interviewer_model: ModelAdapter,
        max_turns: int = 40,
        context_token_limit: int = 3000,
    ):
        self.model = subject_model
        self.interviewer_model = interviewer_model
        self.max_turns = max_turns
        self.context_token_limit = context_token_limit

    def run(self, scenario: dict) -> dict:
        run_id = new_run_id()
        params = scenario.get("params", {})
        conversation = []
        context_trims = 0
        stop_reason = "max_turns"

        # Seed with the opening message (from scenario or LLM-generated)
        opening = scenario.get("opening_message") or self._generate_opening(scenario)
        conversation.append({
            "turn": 0,
            "speaker": "interviewer",
            "text": opening,
            "timestamp": now_iso(),
        })

        for turn in range(self.max_turns):
            # --- Subject responds ---
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

            # --- Interviewer responds ---
            interviewer_prompt = self._build_interviewer_prompt(scenario, conversation)

            print(f"  turn {turn + 1}/{self.max_turns} — interviewer model...", flush=True)
            next_probe = self.interviewer_model.generate(interviewer_prompt, params)
            next_probe = next_probe.strip()
            # Strip any speaker prefix the model echoes back (e.g. "INTERVIEWER: ...")
            for prefix in ("INTERVIEWER:", "Interviewer:"):
                if next_probe.startswith(prefix):
                    next_probe = next_probe[len(prefix):].strip()
                    break
            print(f"  turn {turn + 1} interviewer done ({len(next_probe)} chars)", flush=True)

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
            "conversation": conversation,
            "_scenario": scenario,
            "metadata": {
                "total_turns": len([t for t in conversation if t["speaker"] == "subject"]),
                "stop_reason": stop_reason,
                "context_trims": context_trims,
            },
            "scores": {},
        }

    def _language_instruction(self, scenario: dict) -> str:
        lang = scenario.get("language", "english")
        return f"Respond in {lang}."

    def _build_subject_prompt(self, scenario: dict, history: list) -> str:
        identity = scenario.get("identity", {})
        constraints = scenario.get("constraints", [])
        objective = scenario.get("objective", "")
        lang_instruction = self._language_instruction(scenario)

        system_block = (
            f"You are {identity.get('name', 'an individual')}. "
            f"{identity.get('background', '')} "
            f"{identity.get('persona', '')} "
            f"Speak in this style: {identity.get('language_style', 'natural')}. "
            f"{lang_instruction}\n"
            f"Objective context: {objective}\n"
            f"Constraints: {'; '.join(constraints)}"
        )

        history_block = "\n".join(
            f"{t['speaker'].upper()}: {t['text']}" for t in history
        )

        return f"[SYSTEM]\n{system_block}\n\n[CONVERSATION]\n{history_block}\n\n[YOUR RESPONSE]"

    def _build_interviewer_prompt(self, scenario: dict, history: list) -> str:
        interviewer = scenario.get("interviewer", {})
        objective = scenario.get("objective", "")
        lang_instruction = self._language_instruction(scenario)

        system_block = (
            f"You are {interviewer.get('name', 'an interviewer')}. "
            f"{interviewer.get('background', '')} "
            f"{interviewer.get('persona', '')} "
            f"Your goal: {interviewer.get('goal', objective)}\n"
            f"Style: {interviewer.get('style', 'Ask natural, probing follow-up questions based on what was just said.')} "
            f"{lang_instruction}\n"
            f"Constraints: {'; '.join(interviewer.get('constraints', ['Ask one focused question at a time', 'Do not break character', 'Build naturally on the previous response']))}"
        )

        history_block = "\n".join(
            f"{t['speaker'].upper()}: {t['text']}" for t in history
        )

        return f"[SYSTEM]\n{system_block}\n\n[CONVERSATION]\n{history_block}\n\n[YOUR NEXT QUESTION]"

    def _generate_opening(self, scenario: dict) -> str:
        """Generate an opening message from the interviewer model if none is scripted."""
        interviewer = scenario.get("interviewer", {})
        objective = scenario.get("objective", "")
        lang_instruction = self._language_instruction(scenario)
        prompt = (
            f"You are {interviewer.get('name', 'an interviewer')}. "
            f"{interviewer.get('background', '')} "
            f"Your goal: {interviewer.get('goal', objective)}\n"
            f"{lang_instruction}\n"
            f"Generate a natural opening message to begin the conversation."
        )
        return self.interviewer_model.generate(prompt, {"temperature": 0.7, "max_tokens": 256})

    def _trim_history(self, scenario: dict, history: list) -> tuple[list, int]:
        system_prompt_tokens = _estimate_tokens(self._build_subject_prompt(scenario, []))
        budget = self.context_token_limit - system_prompt_tokens

        trimmed = 0
        working = list(history)

        while working:
            total = sum(_estimate_tokens(t["text"]) for t in working)
            if total <= budget:
                break
            if len(working) > 1:
                working.pop(1)
                trimmed += 1
            else:
                break

        return working, trimmed

    def _stop_condition(self, history: list) -> bool:
        subject_turns = [t for t in history if t["speaker"] == "subject"]
        if subject_turns and not subject_turns[-1]["text"].strip():
            return True
        return False
