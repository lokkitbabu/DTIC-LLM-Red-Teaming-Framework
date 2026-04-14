"""
Single run entry point.
Usage: python main.py --scenario scenarios/example_scenario.json --model ollama:mistral --interviewer openai:gpt-4o
"""

import argparse
from dotenv import load_dotenv
from models import OllamaAdapter, OpenAIAdapter, AnthropicAdapter, GrokAdapter, HuggingFaceAdapter, TogetherAdapter, MistralAdapter
from runner import ConversationRunner
from runner.session_memory import SessionMemory
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
    elif provider == "mistral":
        return MistralAdapter(name)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use ollama, openai, anthropic, grok, hf, together, or mistral.")


def main():
    parser = argparse.ArgumentParser(description="Run a single DTIC Offset evaluation scenario")
    parser.add_argument("--scenario", required=True, help="Path to scenario JSON file")
    parser.add_argument("--model", required=True, help="Subject model, e.g. together:meta-llama/Llama-4-Maverick-Instruct-17B-128E")
    parser.add_argument("--interviewer", required=True, help="Interviewer model, e.g. openai:gpt-4o")
    parser.add_argument("--judge", default=None, help="Optional judge model for automated scoring")
    parser.add_argument("--eval-target", default="subject", choices=["subject", "interviewer"],
                        help="Which side to score (default: subject)")
    parser.add_argument("--prompt-format", default="flat", choices=["flat", "hierarchical", "xml"],
                        help="Prompt structure for A/B testing (default: flat)")
    parser.add_argument("--session-group", default=None,
                        help="Session group ID for multi-session continuity runs")
    parser.add_argument("--session-number", type=int, default=1,
                        help="Session number within a group (default: 1)")
    parser.add_argument("--max-turns", type=int, default=40)
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    model = build_model(args.model)
    interviewer_model = build_model(args.interviewer)

    # Load prior session memory if this is a continuation run
    prior_memory = []
    if args.session_group and args.session_number > 1:
        mem_loader = SessionMemory()
        loaded = mem_loader.load_local(args.session_group, args.session_number - 1)
        if loaded:
            prior_memory = loaded.get("facts", [])
            print(f"Loaded {len(prior_memory)} memory facts from session {args.session_number - 1}")

    runner = ConversationRunner(
        model, interviewer_model,
        max_turns=args.max_turns,
        prompt_format=args.prompt_format,
        prior_memory=prior_memory,
    )
    print(f"Running scenario '{scenario.get('scenario_id')}' with {repr(model)} "
          f"[format={args.prompt_format}, session={args.session_number}]...")

    run_data = runner.run(scenario)

    # Tag session metadata onto the run
    if args.session_group:
        run_data["session_group_id"] = args.session_group
        run_data["session_number"] = args.session_number

    if scenario.get("language", "english").lower() != "english":
        trans_model = build_model(args.judge or args.interviewer)
        translate_run(run_data, trans_model)

    log_path = save_run(run_data)
    print(f"Run saved: {log_path}")

    # Sync to Supabase if configured
    try:
        from dashboard.supabase_store import get_store
        store = get_store()
        if store.available:
            store.save_run(run_data)
            print(f"✓ Synced to Supabase — visible in dashboard immediately.")
        else:
            print("  (Supabase not configured — run is local only. Add credentials to .env to auto-sync.)")
    except Exception as e:
        print(f"  (Supabase sync failed: {e})")

    csv_path = export_for_manual_scoring(run_data)
    print(f"Manual scoring sheet: {csv_path}")

    if args.judge:
        judge_model = build_model(args.judge)
        judge = LLMJudge(judge_model, eval_target=args.eval_target)
        print(f"Running LLM judge ({repr(judge_model)})...")
        result = judge.evaluate(run_data)
        run_data["scores"]["llm_judge"] = result
        save_run(run_data, run_data["run_id"])
        try:
            from dashboard.supabase_store import get_store
            get_store().save_run(run_data)
        except Exception:
            pass
        print(f"Judge scores: {result['scores']}")

        # Extract session memory after judged run
        if args.session_group:
            mem = SessionMemory(judge_model)
            memory = mem.extract(run_data)
            mem.save_local(args.session_group, args.session_number, run_data["run_id"], memory)
            try:
                from dashboard.supabase_store import get_store
                get_store().save_session_memory(
                    session_group_id=args.session_group,
                    session_number=args.session_number,
                    run_id=run_data["run_id"],
                    scenario_id=run_data.get("scenario_id", ""),
                    subject_model=repr(model),
                    facts=memory.get("facts", []),
                    raw_summary=memory.get("raw_summary", ""),
                )
            except Exception:
                pass
            print(f"Session memory saved: {len(memory['facts'])} facts extracted.")

    print("Done.")


if __name__ == "__main__":
    main()
