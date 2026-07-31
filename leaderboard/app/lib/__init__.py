"""Shared helpers for the leaderboard app.

Importing this package puts `pipeline/` on `sys.path` first, so every submodule
can `import bbq_metrics` — the metric definitions and the palette stay in one
place instead of being restated (and left to drift) in the UI.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_DIR = REPO_ROOT / "pipeline"

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))
