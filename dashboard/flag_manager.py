"""
FlagManager: persists flagged run IDs to logs/flags.json.
Handles all reads and writes to the flags file.
"""

import json
import os
import tempfile
from pathlib import Path


class FlagManager:
    def __init__(self, flags_file: Path = Path("logs/flags.json")):
        self.flags_file = Path(flags_file)

    def load_flags(self) -> list[str]:
        """Read flags_file and return list of flagged run IDs.

        Returns [] if file is missing, malformed JSON, or missing "flagged" key.
        """
        try:
            text = self.flags_file.read_text(encoding="utf-8")
            data = json.loads(text)
            flagged = data.get("flagged", [])
            if not isinstance(flagged, list):
                return []
            return [str(item) for item in flagged]
        except (FileNotFoundError, json.JSONDecodeError, AttributeError):
            return []

    def is_flagged(self, run_id: str) -> bool:
        """Return True if run_id is in the flagged list."""
        return run_id in self.load_flags()

    def toggle_flag(self, run_id: str) -> bool:
        """Add run_id if not flagged, remove if flagged. Persist and return new state."""
        flagged = self.load_flags()
        if run_id in flagged:
            flagged.remove(run_id)
            new_state = False
        else:
            flagged.append(run_id)
            new_state = True
        self.save_flags(flagged)
        return new_state

    def save_flags(self, flagged: list[str]) -> None:
        """Write {"flagged": [...]} to flags_file atomically."""
        self.flags_file.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps({"flagged": flagged}, indent=2)
        # Atomic write: write to temp file then rename
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.flags_file.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp_path, self.flags_file)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
