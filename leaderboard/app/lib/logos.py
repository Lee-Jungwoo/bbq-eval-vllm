"""Per-family brand logos for the ranking table.

The repository ships no logo artwork — vendor logos are trademarks and are not
ours to redistribute. This module is the wiring only: drop an official SVG (or
PNG) into ``app/assets/logos/`` under the name this module expects and it shows
up in the leaderboard on the next rerun. Until then every lookup returns None
and the table falls back to the plain model name, which is what it did before.

See ``app/assets/logos/README.md`` for the expected file names.
"""
from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

LOGO_DIR = Path(__file__).resolve().parent.parent / "assets" / "logos"

#: family label (as produced by bbq_metrics.family_and_size) -> file stem.
#: Families sharing a vendor deliberately share a stem: one openai.svg serves
#: every GPT-OSS run, one qwen.svg serves all three Qwen generations.
LOGO_STEMS = {
    "Gemma": "gemma",
    "GPT-OSS": "openai",
    "Qwen2.5": "qwen",
    "Qwen3": "qwen",
    "Qwen3.5": "qwen",
    "Llama": "meta",
    "Ministral-3": "mistral",
    "Mistral": "mistral",
    "Phi": "microsoft",
}

#: Extensions tried in order. SVG first — it stays sharp in the table.
_EXTENSIONS = (".svg", ".png")

_MIME = {".svg": "image/svg+xml", ".png": "image/png"}


@lru_cache(maxsize=None)
def data_uri(family: str) -> str | None:
    """A data: URI for this family's logo, or None when no file is present.

    Cached on the family name, so a missing file is only looked up once per
    session. Delete the cache (or restart the app) after adding artwork.
    """
    stem = LOGO_STEMS.get(family)
    if stem is None:
        return None
    for ext in _EXTENSIONS:
        path = LOGO_DIR / f"{stem}{ext}"
        if path.is_file():
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{_MIME[ext]};base64,{encoded}"
    return None


def column(families) -> list[str | None]:
    """Logo column for a leaderboard frame: one data URI (or None) per row."""
    return [data_uri(f) for f in families]


def missing() -> dict[str, str]:
    """{family: expected path} for families whose artwork is not installed.

    Used by the app to tell the user what to drop in rather than silently
    rendering a column of blanks.
    """
    gaps = {}
    for family, stem in LOGO_STEMS.items():
        if data_uri(family) is None:
            gaps[family] = str(LOGO_DIR / f"{stem}{_EXTENSIONS[0]}")
    return gaps
