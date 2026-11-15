"""Shared helpers for the leaderboard app.

Importing this package puts `src/` on `sys.path` first, so every
submodule can `from bbq_eval import bbq_metrics` — the metric definitions and the palette stay
in one place instead of being restated (and left to drift) in the UI.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
