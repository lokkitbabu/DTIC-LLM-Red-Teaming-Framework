"""
sync.py — One-shot or watch mode sync of local run logs to Supabase.

Usage:
    # Sync all existing logs once (backfill)
    python sync.py

    # Watch logs/ and sync new files as they appear
    python sync.py --watch

Set SUPABASE_URL and SUPABASE_KEY in your .env (already pre-filled in .env.example).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LOGS_DIR = Path("logs")
ERRORS_DIR = LOGS_DIR / "errors"
POLL_INTERVAL = 3  # seconds between scans in watch mode


def _load_store():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dashboard.supabase_store import get_store
    store = get_store()
    if not store.available:
        print("✗ Supabase not configured.")
        print("  Copy .env.example to .env and fill in your API keys.")
        print("  SUPABASE_URL and SUPABASE_KEY are pre-filled — just copy them.")
        sys.exit(1)
    return store


def sync_file(path: Path, store, already_synced: set[str]) -> bool:
    """Sync a single log file. Returns True if synced, False if skipped."""
    if path.name in already_synced:
        return False
    if not path.suffix == ".json":
        return False
    if path.name in {"flags.json", "manifest.json"}:
        return False
    try:
        # Skip anything in errors/
        path.relative_to(ERRORS_DIR)
        return False
    except ValueError:
        pass

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "run_id" not in data:
            return False

        run_id = data["run_id"]
        store.save_run(data)
        already_synced.add(path.name)
        print(f"  ✓ {run_id[:8]}…  {data.get('scenario_id', '?')} / {data.get('subject_model', '?')}")
        return True
    except Exception as e:
        print(f"  ✗ {path.name}: {e}")
        return False


def sync_all(store) -> set[str]:
    """Sync all existing log files. Returns set of synced filenames."""
    print(f"Scanning {LOGS_DIR}/…")
    synced: set[str] = set()
    files = sorted(LOGS_DIR.glob("*.json"))
    if not files:
        print("  No log files found.")
        return synced
    for f in files:
        sync_file(f, store, synced)
    print(f"\nDone — {len(synced)} run(s) synced to Supabase.")
    return synced


def watch(store) -> None:
    """Poll logs/ every few seconds and sync new files."""
    print(f"Watching {LOGS_DIR}/ for new run logs (Ctrl+C to stop)…")
    synced: set[str] = set()

    # Initial backfill
    for f in sorted(LOGS_DIR.glob("*.json")):
        sync_file(f, store, synced)
    print(f"Backfill complete — {len(synced)} existing run(s) synced.\n")

    try:
        while True:
            time.sleep(POLL_INTERVAL)
            new = 0
            for f in sorted(LOGS_DIR.glob("*.json")):
                if sync_file(f, store, synced):
                    new += 1
            if new:
                print(f"  {new} new run(s) synced.")
    except KeyboardInterrupt:
        print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Sync local run logs to Supabase")
    parser.add_argument("--watch", action="store_true",
                        help="Watch logs/ and sync new files as they appear")
    args = parser.parse_args()

    store = _load_store()
    print(f"✓ Connected to Supabase\n")

    if args.watch:
        watch(store)
    else:
        sync_all(store)


if __name__ == "__main__":
    main()
