import json
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field, field_validator


class IdentitySchema(BaseModel):
    name: str
    background: str
    persona: str
    language_style: str


class ParamsSchema(BaseModel):
    temperature: float = Field(..., ge=0.0, le=2.0)
    top_p: float = Field(..., ge=0.0, le=1.0)
    max_tokens: int = Field(..., gt=0)
    seed: int


class ScenarioSchema(BaseModel):
    model_config = {"extra": "ignore"}
    scenario_id: str
    model: str
    identity: IdentitySchema
    objective: str
    constraints: List[str]
    params: ParamsSchema
    opening_message: str


def load_scenario(path: str) -> dict:
    """Load, validate, and return a scenario JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scenario not found: {path}")
    with open(p) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Scenario file is not valid JSON: {path} — {e}") from e

    try:
        scenario = ScenarioSchema.model_validate(data)
    except Exception as e:
        raise ValueError(f"Invalid scenario '{path}': {e}") from e

    return scenario.model_dump()
