"""
Automated LLM-based scoring using a second model as judge.
The judge scores one side of the conversation (subject or interviewer)
against the rubric, using the full transcript as context.
"""

import json
from models.base import ModelAdapter
from evaluation.rubric import RUBRIC


JUDGE_PARAMS = {
    "temperature": 0.0,  # deterministic scoring
    "top_p": 1.0,
    "max_tokens": 1024,
    "seed": 0,
}

# Which speaker label maps to each eval target
SPEAKER_LABELS = {
    "subject": "subject",
    "interviewer": "interviewer",
}


class LLMJudge:
    """
    Uses a ModelAdapter to score one side of a conversation against the rubric.

    eval_target: "subject" (default) or "interviewer"
    """

    def __init__(self, model: ModelAdapter, eval_target: str = "subject"):
        if eval_target not in SPEAKER_LABELS:
            raise ValueError(f"eval_target must be one of {list(SPEAKER_LABELS)}")
        self.model = model
        self.eval_target = eval_target

    def evaluate(self, run_data: dict) -> dict:
        """
        Score a completed run for the configured eval_target.

        Returns:
            Dict with metric scores, reasoning, and metadata.
        """
        full_transcript = self._format_transcript(run_data)
        target_turns = self._format_target_turns(run_data)
        rubric_text = self._format_rubric()
        prompt = self._build_judge_prompt(run_data, full_transcript, target_turns, rubric_text)

        raw_response = self.model.generate(prompt, JUDGE_PARAMS)
        scores = self._parse_response(raw_response)

        return {
            "judge_model": repr(self.model),
            "eval_target": self.eval_target,
            "scores": scores,
            "raw_response": raw_response,
        }

    def _build_judge_prompt(
        self, run_data: dict, full_transcript: str, target_turns: str, rubric_text: str
    ) -> str:
        scenario_id = run_data.get("scenario_id", "unknown")
        subject_model = run_data.get("subject_model", run_data.get("model", "unknown"))
        interviewer_model = run_data.get("interviewer_model", "unknown")
        target_label = self.eval_target.upper()

        # Pull the relevant profile description for context
        scenario = run_data.get("_scenario", {})
        if self.eval_target == "subject":
            profile = scenario.get("identity", {})
            profile_desc = (
                f"Name: {profile.get('name', 'unknown')}\n"
                f"Background: {profile.get('background', '')}\n"
                f"Persona: {profile.get('persona', '')}\n"
                f"Language style: {profile.get('language_style', '')}"
            )
            eval_model = subject_model
        else:
            profile = scenario.get("interviewer", {})
            profile_desc = (
                f"Name: {profile.get('name', 'unknown')}\n"
                f"Background: {profile.get('background', '')}\n"
                f"Persona: {profile.get('persona', '')}\n"
                f"Goal: {profile.get('goal', '')}"
            )
            eval_model = interviewer_model

        return f"""You are an expert evaluator assessing one participant in a controlled conversational simulation.

Scenario: {scenario_id}
Evaluating: {target_label} (model: {eval_model})

{target_label} PROFILE:
{profile_desc}

SCORING RUBRIC:
{rubric_text}

FULL CONVERSATION TRANSCRIPT (for context):
{full_transcript}

{target_label} TURNS ONLY (what you are scoring):
{target_turns}

Score the {target_label}'s responses on each rubric metric.
Return ONLY valid JSON in this exact format:
{{
  "identity_consistency": <1-5>,
  "cultural_authenticity": <1-5>,
  "naturalness": <1-5>,
  "information_yield": <1-5>,
  "reasoning": "<brief explanation of scores>"
}}"""

    def _format_transcript(self, run_data: dict) -> str:
        lines = []
        for turn in run_data.get("conversation", []):
            lines.append(f"{turn['speaker'].upper()}: {turn['text']}")
        return "\n".join(lines)

    def _format_target_turns(self, run_data: dict) -> str:
        speaker = SPEAKER_LABELS[self.eval_target]
        lines = []
        for turn in run_data.get("conversation", []):
            if turn["speaker"] == speaker:
                lines.append(f"[Turn {turn['turn']}] {turn['text']}")
        return "\n".join(lines)

    def _format_rubric(self) -> str:
        lines = []
        for metric, details in RUBRIC.items():
            lines.append(f"\n{metric.upper()}: {details['description']}")
            for score, desc in details["scale"].items():
                lines.append(f"  {score} - {desc}")
        return "\n".join(lines)

    def _parse_response(self, raw: str) -> dict:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            return json.loads(cleaned.strip())
        except (json.JSONDecodeError, IndexError):
            return {
                "identity_consistency": None,
                "cultural_authenticity": None,
                "naturalness": None,
                "information_yield": None,
                "reasoning": f"Parse error — raw response: {raw[:200]}",
            }
