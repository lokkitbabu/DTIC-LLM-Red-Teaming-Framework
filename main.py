"""
Single run entry point.
Usage: python main.py --scenario scenarios/example_scenario.json --model ollama:mistral
"""

import argparse
from dotenv import load_dotenv
from models import OllamaAdapter, OpenAIAdapter, AnthropicAdapter, GrokAdapter, HuggingFaceAdapter, TogetherAdapter
from runner import ConversationRunner
from evaluation import LLMJudge, export_for_manual_scoring
from utils import load_scenario, save_run, translate_run

load_dotenv()


def build_model(model_str: str):
    """Parse 'provider:model_name' string and return the right adapter."""
    if ":" in model_str:
        provider, name = model_str.split(":", 1)
    else:
        provider, name = model_str, model_str

    provider = provider.lower()
    if provider == "ollama":
        return OllamaAdapter(name)
    elif provider == "openai":
        return OpenAIAdapter(name)
    elif provider == "anthropic":
        return AnthropicAdapter(name)
    elif provider == "grok":
        return GrokAdapter(name)
    elif provider == "hf":
        return HuggingFaceAdapter(name)
    elif provider == "together":
        return TogetherAdapter(name)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use ollama, openai, anthropic, grok, hf, or together.")


def main():
    parser = argparse.ArgumentParser(description="Run a single DTIC Offset evaluation scenario")
    parser.add_argument("--scenario", required=True, help="Path to scenario JSON file")
    parser.add_argument("--model", required=True, help="Subject model, e.g. ollama:mistral or openai:gpt-4o")
    parser.add_argument("--interviewer", required=True, help="Interviewer model, e.g. openai:gpt-4o")
    parser.add_argument("--judge", default=None, help="Optional judge model for automated scoring")
    parser.add_argument("--eval-target", default="subject", choices=["subject", "interviewer"],
                        help="Which side to score (default: subject)")
    parser.add_argument("--max-turns", type=int, default=40)
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    model = build_model(args.model)
    interviewer_model = build_model(args.interviewer)

    runner = ConversationRunner(model, interviewer_model, max_turns=args.max_turns)
    if scenario.get("format") == "discord_chat":
        runner = DiscordRunner(model, interviewer_model, max_turns=args.max_turns)
    else:
        runner = ConversationRunner(model, interviewer_model, max_turns=args.max_turns)
    print(f"Running scenario '{scenario.get('scenario_id')}' with {repr(model)}...")

    run_data = runner.run(scenario)

    # Translate all turns to English in one shot if the scenario runs in another language
    if scenario.get("language", "english").lower() != "english":
        trans_model = build_model(args.judge or args.interviewer)
        translate_run(run_data, trans_model)

    log_path = save_run(run_data)
    print(f"Run saved: {log_path}")

    csv_path = export_for_manual_scoring(run_data)
    print(f"Manual scoring sheet: {csv_path}")

    if args.judge:
        judge_model = build_model(args.judge)
        judge = LLMJudge(judge_model, eval_target=args.eval_target)
        print(f"Running LLM judge ({repr(judge_model)})...")
        result = judge.evaluate(run_data)
        run_data["scores"]["llm_judge"] = result
        save_run(run_data, run_data["run_id"])  # overwrite with scores
        print(f"Judge scores: {result['scores']}")

    print("Done.")


if __name__ == "__main__":
    main()
