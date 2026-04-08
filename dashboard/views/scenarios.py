"""
ScenariosView: list, display, edit, validate, and save scenario JSON files.

Responsibilities:
  - List all scenarios_dir/*.json files in a selectbox
  - Load selected file content into st.text_area for editing
  - "Validate" button: parse JSON and check for REQUIRED_SCENARIO_KEYS;
    show green success or red error list
  - "Save" button: validate JSON parseability first; write back to
    scenarios_dir/<filename>.json on success; display confirmation
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

REQUIRED_SCENARIO_KEYS = [
    "scenario_id",
    "identity",
    "interviewer",
    "objective",
    "constraints",
    "params",
]


def _validate_scenario_keys(data: dict) -> list[str]:
    """
    Return a list of missing required keys from data.

    An empty list means all required keys are present (valid).
    A non-empty list contains the names of missing keys.
    """
    return [key for key in REQUIRED_SCENARIO_KEYS if key not in data]


def render_scenarios_view(scenarios_dir: Path) -> None:
    """
    Render the Scenarios page.

    Args:
        scenarios_dir: Path to the directory containing scenario JSON files.
    """
    st.subheader("Scenario Editor")

    # ------------------------------------------------------------------
    # 1. File selector
    # ------------------------------------------------------------------
    json_files = sorted(scenarios_dir.glob("*.json"))

    if not json_files:
        st.info(f"No scenario JSON files found in `{scenarios_dir}/`.")
        return

    file_names = [f.name for f in json_files]
    selected_name = st.selectbox("Select scenario", file_names)

    if not selected_name:
        return

    selected_path = scenarios_dir / selected_name

    # ------------------------------------------------------------------
    # 2. Load file content into text area
    # ------------------------------------------------------------------
    session_key = f"scenario_content_{selected_name}"

    # Only read from disk when the selection changes or content not yet cached
    if session_key not in st.session_state:
        try:
            st.session_state[session_key] = selected_path.read_text(encoding="utf-8")
        except OSError as exc:
            st.error(f"Could not read file: {exc}")
            return

    edited_content = st.text_area(
        "Scenario JSON",
        value=st.session_state[session_key],
        height=400,
        key=f"textarea_{selected_name}",
    )
    # Keep session state in sync with the text area
    st.session_state[session_key] = edited_content

    # ------------------------------------------------------------------
    # 3. Action buttons
    # ------------------------------------------------------------------
    col_validate, col_save = st.columns(2)

    with col_validate:
        if st.button("Validate", key=f"validate_{selected_name}"):
            _handle_validate(edited_content)

    with col_save:
        if st.button("Save", key=f"save_{selected_name}"):
            _handle_save(edited_content, selected_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _handle_validate(content: str) -> None:
    """Parse content as JSON and check for required keys; display result."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")
        return

    missing = _validate_scenario_keys(data)
    if missing:
        st.error("Validation failed. Missing required keys:\n" + "\n".join(f"- `{k}`" for k in missing))
    else:
        st.success("Validation passed. All required keys are present.")


def _handle_save(content: str, target_path: Path) -> None:
    """Validate JSON parseability, then write to target_path."""
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        st.error(f"Cannot save: invalid JSON — {exc}")
        return

    try:
        target_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        st.error(f"Failed to write file: {exc}")
        return

    st.success(f"Saved to `{target_path}`.")
