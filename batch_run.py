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


def run_experiment(scenario_path: str, model_str: str, interviewer_str: str, judge_str: str = None,
                   eval_target: str = "subject", max_turns: int = 40, retries: int = 2,
                   prompt_format: str = "flat"):
    scenario = load_scenario(scenario_path)
    scenario_id = scenario.get("scenario_id", Path(scenario_path).stem)

    for attempt in range(retries + 1):
        try:
            model = build_model(model_str)
            interviewer_model = build_model(interviewer_str)
            runner = ConversationRunner(model, interviewer_model, max_turns=max_turns,
                                        prompt_format=prompt_format)
            run_data = runner.run(scenario)

            if scenario.get("language", "english").lower() != "english":
                trans_model = build_model(judge_str or interviewer_str)
                translate_run(run_data, trans_model)

            if judge_str:
                judge_model = build_model(judge_str)
                judge = LLMJudge(judge_model, eval_target=eval_target, prompt_name="strict")
                result = judge.evaluate(run_data)
                run_data["scores"]["llm_judge"] = result

            log_path = save_run(run_data)
            export_for_manual_scoring(run_data)

            # Sync to Supabase
            try:
                from dashboard.supabase_store import get_store
                store = get_store()
                if store.available:
                    store.save_run(run_data)
            except Exception:
                pass

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
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--interviewer", required=True)
    parser.add_argument("--judge", default=None)
    parser.add_argument("--eval-target", default="subject", choices=["subject", "interviewer"])
    parser.add_argument("--prompt-formats", nargs="+", default=["flat"],
                        choices=["flat", "hierarchical", "xml"],
                        help="One or more prompt formats for A/B testing (default: flat)")
    parser.add_argument("--runs-per-combo", type=int, default=1,
                        help="How many times to run each scenario×model×format combo (default: 1)")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    scenario_path = Path(args.scenarios)
    scenario_files = sorted(scenario_path.glob("*.json")) if scenario_path.is_dir() else [scenario_path]

    total = len(scenario_files) * len(args.models) * len(args.prompt_formats) * args.runs_per_combo
    print(f"Scenarios: {len(scenario_files)} | Models: {len(args.models)} | "
          f"Formats: {args.prompt_formats} | Runs/combo: {args.runs_per_combo}")
    print(f"Total runs: {total}\n")

    for scenario_file in scenario_files:
        for model_str in args.models:
            for fmt in args.prompt_formats:
                for run_n in range(args.runs_per_combo):
                    run_experiment(
                        str(scenario_file), model_str,
                        interviewer_str=args.interviewer,
                        judge_str=args.judge,
                        eval_target=args.eval_target,
                        max_turns=args.max_turns,
                        retries=args.retries,
                        prompt_format=fmt,
                    )

    print(f"\nManifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
