import json
import pytest
from utils.scenario_loader import load_scenario

VALID_SCENARIO = {
    "scenario_id": "test_scenario",
    "model": "mistral",
    "identity": {
        "name": "Test Person",
        "background": "Test background",
        "persona": "Test persona",
        "language_style": "Formal",
    },
    "objective": "Test objective",
    "constraints": ["Never break character"],
    "params": {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512,
        "seed": 42,
    },
    "opening_message": "Hello",
}


# 21.1 — Valid scenario loads correctly
def test_valid_scenario_loads(tmp_path):
    scenario_file = tmp_path / "scenario.json"
    scenario_file.write_text(json.dumps(VALID_SCENARIO))

    result = load_scenario(str(scenario_file))

    assert result["scenario_id"] == "test_scenario"
    assert result["model"] == "mistral"
    assert result["objective"] == "Test objective"
    assert result["opening_message"] == "Hello"
    assert result["constraints"] == ["Never break character"]
    assert result["identity"]["name"] == "Test Person"
    assert result["identity"]["background"] == "Test background"
    assert result["identity"]["persona"] == "Test persona"
    assert result["identity"]["language_style"] == "Formal"
    assert result["params"]["temperature"] == 0.7
    assert result["params"]["top_p"] == 0.9
    assert result["params"]["max_tokens"] == 512
    assert result["params"]["seed"] == 42


# 21.2 — Missing file raises FileNotFoundError
def test_missing_file_raises(tmp_path):
    missing = tmp_path / "nonexistent.json"
    with pytest.raises(FileNotFoundError):
        load_scenario(str(missing))


# 21.3 — Malformed JSON raises ValueError
def test_malformed_json_raises(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{bad json")
    with pytest.raises(ValueError):
        load_scenario(str(bad_file))


# 21.4a — Missing required field raises ValueError
def test_missing_required_field_raises(tmp_path):
    scenario = {k: v for k, v in VALID_SCENARIO.items() if k != "objective"}
    scenario_file = tmp_path / "missing_field.json"
    scenario_file.write_text(json.dumps(scenario))
    with pytest.raises(ValueError):
        load_scenario(str(scenario_file))


# 21.4b — Invalid param value (temperature > 2.0) raises ValueError
def test_invalid_param_value_raises(tmp_path):
    scenario = json.loads(json.dumps(VALID_SCENARIO))
    scenario["params"]["temperature"] = 5.0
    scenario_file = tmp_path / "invalid_param.json"
    scenario_file.write_text(json.dumps(scenario))
    with pytest.raises(ValueError):
        load_scenario(str(scenario_file))
