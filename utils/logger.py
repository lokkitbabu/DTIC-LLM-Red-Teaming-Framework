import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, field_validator, model_validator


LOGS_DIR = Path("logs")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TurnSchema(BaseModel):
    turn: int
    speaker: Literal["interviewer", "subject"]
    text: str
    text_en: Optional[str] = None
    timestamp: str
    raw_prompt: Optional[str] = None  # present for subject turns; optional for compatibility


class MetadataSchema(BaseModel):
    total_turns: int
    stop_reason: Literal["max_turns", "condition_met", "error", "probes_exhausted"]
    context_trims: int


class RunLogSchema(BaseModel):
    model_config = {"extra": "ignore"}  # tolerate extra fields from any runner

    run_id: str
    timestamp: str
    scenario_id: str
    subject_model: str
    interviewer_model: str
    language: str = "english"
    params: dict[str, Any]
    conversation: list[TurnSchema]
    metadata: MetadataSchema
    scores: dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def new_run_id() -> str:
    return str(uuid.uuid4())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_run(run_data: dict, run_id: str = None) -> Path:
    """
    Validate run_data against RunLogSchema, then save to logs/<run_id>.json.
    Strips internal-only keys (e.g. _scenario) before validation.
    Returns the path of the saved file.
    """
    clean = {k: v for k, v in run_data.items() if not k.startswith("_")}
    validated = RunLogSchema.model_validate(clean)

    LOGS_DIR.mkdir(exist_ok=True)
    run_id = run_id or validated.run_id or new_run_id()
    out_path = LOGS_DIR / f"{run_id}.json"

    with open(out_path, "w") as f:
        json.dump(validated.model_dump(), f, indent=2)

    return out_path


def translate_run(run_data: dict, model) -> dict:
    """
    Translate all conversation turns to English in a single model call.
    Sends the full conversation as a numbered JSON array, gets back a matching
    array of English translations, and populates text_en on each turn.

    Only runs when run_data["language"] is not "english".
    Returns the mutated run_data (also mutates in place).
    """
    if run_data.get("language", "english").lower() == "english":
        return run_data

    conversation = run_data.get("conversation", [])
    if not conversation:
        return run_data

    # Build a compact numbered list for the model to translate
    turns_payload = json.dumps(
        [{"i": i, "speaker": t["speaker"], "text": t["text"]} for i, t in enumerate(conversation)],
        ensure_ascii=False,
        indent=2,
    )

    prompt = (
        f"The following is a conversation in {run_data['language']}. "
        "Translate each turn's 'text' field into accurate English, preserving meaning, tone, and register. "
        "Return ONLY a valid JSON array where each object has 'i' (the original index) and 'text_en' (the English translation). "
        "Do not include any explanation or extra text.\n\n"
        f"{turns_payload}"
    )

    raw = model.generate(prompt, {"temperature": 0.0, "max_tokens": 4096})

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        translations = json.loads(cleaned.strip())
        index = {item["i"]: item["text_en"] for item in translations}
        for i, turn in enumerate(conversation):
            turn["text_en"] = index.get(i, "")
    except (json.JSONDecodeError, KeyError):
        # Best-effort: leave text_en empty rather than crash
        for turn in conversation:
            turn.setdefault("text_en", "")

    return run_data


def save_error(scenario_id: str, model: str, error: str) -> Path:
    """Log a failed run to logs/errors/."""
    error_dir = LOGS_DIR / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id()
    out_path = error_dir / f"{run_id}.json"

    with open(out_path, "w") as f:
        json.dump({
            "run_id": run_id,
            "timestamp": now_iso(),
            "scenario_id": scenario_id,
            "model": model,
            "error": error,
        }, f, indent=2)

    return out_path
