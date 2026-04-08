"""
Batch experiment runner — executes all model × scenario combinations.
Usage: python batch_run.py --scenarios scenarios/ --models ollama:mistral openai:gpt-4o
"""

import argparse
import traceback
from pathlib import Path
from main import build_model
from runner import ConversationRunner
from evaluation import LLMJudge, export_for_manual_scoring
from utils import load_scenario, save_run, translate_run
from utils.logger import save_error, now_iso, new_run_id
import json


MANIFEST_PATH = Path("logs/manifest.json")


def load_manifest() -> list:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return []


def append_manifest(entry: dict):
    manifest = load_manifest()
    manifest.append(entry)
    MANIFEST_PATH.parent.mkdir(exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def run_experiment(scenario_path: str, model_str: str, interviewer_str: str, judge_str: str = None, eval_target: str = "subject", max_turns: int = 40, retries: int = 2):
    scenario = load_scenario(scenario_path)
    scenario_id = scenario.get("scenario_id", Path(scenario_path).stem)

    for attempt in range(retries + 1):
        try:
            model = build_model(model_str)
            interviewer_model = build_model(interviewer_str)
            runner = ConversationRunner(model, interviewer_model, max_turns=max_turns)
            run_data = runner.run(scenario)

            if scenario.get("language", "english").lower() != "english":
                trans_model = build_model(judge_str or interviewer_str)
                translate_run(run_data, trans_model)

            if judge_str:
                judge_model = build_model(judge_str)
                judge = LLMJudge(judge_model, eval_target=eval_target)
                result = judge.evaluate(run_data)
                run_data["scores"]["llm_judge"] = result

            log_path = save_run(run_data)
            export_for_manual_scoring(run_data)

            append_manifest({
                "run_id": run_data["run_id"],
                "scenario_id": scenario_id,
                "model": model_str,
                "status": "success",
                "log": str(log_path),
                "timestamp": now_iso(),
            })

            print(f"  [OK] {scenario_id} × {model_str} → {log_path}")
            return

        except Exception as e:
            if attempt < retries:
                print(f"  [RETRY {attempt + 1}] {scenario_id} × {model_str}: {e}")
            else:
                err_path = save_error(scenario_id, model_str, traceback.format_exc())
                append_manifest({
                    "run_id": new_run_id(),
                    "scenario_id": scenario_id,
                    "model": model_str,
                    "status": "failed",
                    "error": str(e),
                    "error_log": str(err_path),
                    "timestamp": now_iso(),
                })
                print(f"  [FAIL] {scenario_id} × {model_str}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Batch DTIC Offset evaluation runner")
    parser.add_argument("--scenarios", required=True, help="Path to scenarios directory or single JSON file")
    parser.add_argument("--models", nargs="+", required=True, help="Subject model strings e.g. ollama:mistral openai:gpt-4o")
    parser.add_argument("--interviewer", required=True, help="Interviewer model e.g. openai:gpt-4o")
    parser.add_argument("--judge", default=None, help="Judge model for automated scoring")
    parser.add_argument("--eval-target", default="subject", choices=["subject", "interviewer"],
                        help="Which side to score (default: subject)")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    scenario_path = Path(args.scenarios)
    if scenario_path.is_dir():
        scenario_files = sorted(scenario_path.glob("*.json"))
    else:
        scenario_files = [scenario_path]

    print(f"Scenarios: {len(scenario_files)} | Models: {len(args.models)}")
    print(f"Total runs: {len(scenario_files) * len(args.models)}\n")

    for scenario_file in scenario_files:
        for model_str in args.models:
            run_experiment(
                str(scenario_file),
                model_str,
                interviewer_str=args.interviewer,
                judge_str=args.judge,
                eval_target=args.eval_target,
                max_turns=args.max_turns,
                retries=args.retries,
            )

    print(f"\nManifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
