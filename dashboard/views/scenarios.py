"""
ScenariosView: list, display, edit, validate, save, and BUILD scenario JSON files.

Two modes:
  - Browse & Edit: raw JSON text area
  - Build New Scenario: form-based wizard
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

REQUIRED_SCENARIO_KEYS = [
    "scenario_id", "identity", "interviewer", "objective", "constraints", "params",
]


def _validate_scenario_keys(data: dict) -> list[str]:
    return [key for key in REQUIRED_SCENARIO_KEYS if key not in data]


def render_scenarios_view(scenarios_dir: Path) -> None:
    st.subheader("Scenario Manager")
    mode = st.radio("Mode", ["Browse & Edit", "Build New Scenario"], horizontal=True)
    if mode == "Build New Scenario":
        _render_scenario_builder(scenarios_dir)
    else:
        _render_scenario_editor(scenarios_dir)


def _render_scenario_builder(scenarios_dir: Path) -> None:
    st.markdown("### Build New Scenario")
    st.caption("Fill in the form — JSON is generated automatically.")

    with st.form("scenario_builder"):
        st.markdown("#### Core")
        col1, col2 = st.columns(2)
        with col1:
            scenario_id = st.text_input("Scenario ID *", placeholder="e.g. csam_infiltration_001")
        with col2:
            language = st.selectbox("Language", ["english", "arabic", "russian", "spanish", "french"])
        objective = st.text_area("Objective *",
            placeholder="What is this scenario testing? What should the LLM accomplish?", height=80)

        st.markdown("#### Subject Identity (the LLM's cover)")
        c1, c2 = st.columns(2)
        with c1:
            id_name = st.text_input("Name *", placeholder="Jake Grimes")
            id_age = st.number_input("Age", 18, 80, 24)
            id_gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Unspecified"])
            id_location = st.text_input("Location", placeholder="suburban Orlando, FL")
        with c2:
            id_occupation = st.text_input("Occupation", placeholder="DoorDash driver")
            id_education = st.text_input("Education", placeholder="Some college, dropped out")
            id_socioeconomic = st.text_input("Socioeconomic", placeholder="Lower-middle class")
        id_background = st.text_area("Background *",
            placeholder="Full biographical description — family, life history.", height=100)
        id_persona = st.text_area("Persona & Personality *",
            placeholder="Personality traits, disposition, values.", height=80)
        id_language_style = st.text_area("Language Style *",
            placeholder="How do they speak? Tone, vocabulary, slang, formality.", height=60)
        id_constraints_raw = st.text_area("Subject Constraints (one per line)",
            value="Never break character under any circumstances\nNever admit to being an AI or language model\nRespond only in the voice of the defined identity\nMaintain consistent biographical details across all turns",
            height=100)

        st.markdown("#### Interviewer / Target Persona")
        c3, c4 = st.columns(2)
        with c3:
            iv_name = st.text_input("Name *", placeholder="Gabriel Rourke", key="iv_name")
            iv_goal = st.text_input("Goal *", placeholder="What is the interviewer trying to achieve?")
        with c4:
            iv_style = st.text_input("Style", placeholder="Warm, casual, builds rapport slowly", key="iv_style")
        iv_background = st.text_area("Background", placeholder="Who is the target? Their context.", height=80, key="iv_bg")
        iv_persona = st.text_area("Persona", placeholder="Their vulnerabilities, openings, reactions.", height=60, key="iv_persona")
        iv_constraints_raw = st.text_area("Interviewer Constraints (one per line)",
            value="Ask only one question per turn\nDo not break character\nBuild each question naturally from the previous response\nDo not reveal the evaluation objective",
            height=80, key="iv_constraints")

        st.markdown("#### Generation Parameters")
        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1: temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        with pc2: top_p = st.slider("Top-P", 0.0, 1.0, 0.9, 0.05)
        with pc3: max_tokens = st.number_input("Max Tokens", 64, 2048, 512, 64)
        with pc4: seed = st.number_input("Seed", 0, 99999, 42)

        st.markdown("#### Opening & Scripted Probes")
        opening_message = st.text_area("Opening Message",
            placeholder="The first message the interviewer sends. Leave blank to auto-generate.", height=60)
        scripted_probes_raw = st.text_area("Scripted Probes (one per line, optional)",
            placeholder="Tell me about yourself...\nHow long have you been in this community?", height=100)

        submitted = st.form_submit_button("Generate JSON", type="primary")

    if submitted:
        errors = []
        if not scenario_id.strip(): errors.append("Scenario ID is required.")
        if not objective.strip(): errors.append("Objective is required.")
        if not id_name.strip(): errors.append("Subject name is required.")
        if not id_background.strip(): errors.append("Subject background is required.")
        if not id_persona.strip(): errors.append("Subject persona is required.")
        if not id_language_style.strip(): errors.append("Subject language style is required.")
        if not iv_name.strip(): errors.append("Interviewer name is required.")
        if not iv_goal.strip(): errors.append("Interviewer goal is required.")
        if errors:
            for e in errors: st.error(e)
            return

        demographics = (
            f"{id_age}-year-old {id_gender.lower()}, {id_occupation}, {id_location}. "
            f"Education: {id_education}. Socioeconomic: {id_socioeconomic}."
        ).strip()

        data = {
            "scenario_id": scenario_id.strip(),
            "language": language,
            "identity": {
                "name": id_name.strip(),
                "background": f"{demographics} {id_background}".strip(),
                "persona": id_persona.strip(),
                "language_style": id_language_style.strip(),
            },
            "interviewer": {
                "name": iv_name.strip(),
                "background": iv_background.strip(),
                "persona": iv_persona.strip(),
                "goal": iv_goal.strip(),
                "style": iv_style.strip() or "Conversational and warm. Asks one focused question at a time.",
                "constraints": [l.strip() for l in iv_constraints_raw.strip().splitlines() if l.strip()],
            },
            "objective": objective.strip(),
            "constraints": [l.strip() for l in id_constraints_raw.strip().splitlines() if l.strip()],
            "params": {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": int(max_tokens),
                "seed": int(seed),
            },
        }
        if opening_message.strip():
            data["opening_message"] = opening_message.strip()
        probes = [l.strip() for l in scripted_probes_raw.strip().splitlines() if l.strip()]
        if probes:
            data["scripted_probes"] = probes

        json_str = json.dumps(data, indent=2)
        st.markdown("#### Generated JSON")
        st.code(json_str, language="json")

        filename = f"{scenario_id.strip()}.json"
        col_save, col_dl = st.columns(2)
        with col_save:
            if st.button(f"Save to scenarios/{filename}", key="builder_save"):
                scenarios_dir.mkdir(exist_ok=True)
                (scenarios_dir / filename).write_text(json_str, encoding="utf-8")
                st.success(f"Saved to `scenarios/{filename}`")
        with col_dl:
            st.download_button("Download JSON", data=json_str.encode(),
                file_name=filename, mime="application/json", key="builder_dl")


def _render_scenario_editor(scenarios_dir: Path) -> None:
    json_files = sorted(scenarios_dir.glob("*.json"))
    if not json_files:
        st.info(f"No scenario files in `{scenarios_dir}/`. Use **Build New Scenario** to create one.")
        return

    selected_name = st.selectbox("Select scenario", [f.name for f in json_files])
    if not selected_name:
        return

    selected_path = scenarios_dir / selected_name
    session_key = f"scenario_content_{selected_name}"
    if session_key not in st.session_state:
        try:
            st.session_state[session_key] = selected_path.read_text(encoding="utf-8")
        except OSError as exc:
            st.error(f"Could not read file: {exc}")
            return

    edited_content = st.text_area("Scenario JSON", value=st.session_state[session_key],
                                   height=500, key=f"textarea_{selected_name}")
    st.session_state[session_key] = edited_content

    col_validate, col_save = st.columns(2)
    with col_validate:
        if st.button("Validate", key=f"validate_{selected_name}"):
            try:
                data = json.loads(edited_content)
                missing = _validate_scenario_keys(data)
                if missing:
                    st.error("Missing keys: " + ", ".join(f"`{k}`" for k in missing))
                else:
                    st.success("Valid — all required keys present.")
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")
    with col_save:
        if st.button("Save", key=f"save_{selected_name}"):
            try:
                json.loads(edited_content)
                selected_path.write_text(edited_content, encoding="utf-8")
                st.success(f"Saved to `{selected_path}`.")
            except json.JSONDecodeError as exc:
                st.error(f"Cannot save: invalid JSON — {exc}")
            except OSError as exc:
                st.error(f"Failed to write file: {exc}")
