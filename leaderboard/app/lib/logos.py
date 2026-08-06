"""Vendor logos for the leaderboard table.

The images are the vendors' own avatars, fetched once into `app/assets/logos/`
and inlined as data URIs so the table renders with no network access and no
extra HTTP round-trip per row.
"""
from __future__ import annotations

import base64

from functools import lru_cache
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "logos"

#: HF org prefix (`model` column) → asset stem. The org is authoritative; the
#: family name is only a fallback for rows without a qualified model id.
_BY_ORG = {
    "google": "google",
    "meta-llama": "meta",
    "mistralai": "mistral",
    "openai": "openai",
    "qwen": "qwen",
}

#: Lower-cased substring of `family` → asset stem, tried in order.
_BY_FAMILY = (
    ("gemma", "google"),
    ("gpt-oss", "openai"),
    ("llama", "meta"),
    ("ministral", "mistral"),
    ("mistral", "mistral"),
    ("qwen", "qwen"),
)

VENDOR_NAME = {
    "google": "Google",
    "meta": "Meta",
    "mistral": "Mistral AI",
    "openai": "OpenAI",
    "qwen": "Alibaba Qwen",
}


@lru_cache(maxsize=None)
def data_uri(stem: str) -> str:
    """Inline `assets/logos/<stem>.webp`; '' when the asset is missing."""
    path = ASSETS / f"{stem}.webp"
    if not path.exists():
        return ""
    return ("data:image/webp;base64,"
            + base64.b64encode(path.read_bytes()).decode("ascii"))


def vendor(model: str = "", family: str = "") -> str:
    """Asset stem for a run, or '' when the vendor is unknown."""
    org = str(model).split("/")[0].lower()
    if org in _BY_ORG:
        return _BY_ORG[org]
    lowered = str(family).lower()
    for needle, stem in _BY_FAMILY:
        if needle in lowered:
            return stem
    return ""


def logo(model: str = "", family: str = "") -> tuple[str, str]:
    """`(data-uri, vendor label)` for a run. Both '' when unknown."""
    stem = vendor(model, family)
    if not stem:
        return "", ""
    return data_uri(stem), VENDOR_NAME.get(stem, stem.title())
