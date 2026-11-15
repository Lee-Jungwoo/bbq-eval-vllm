#!/usr/bin/env python3
"""Build the unified external BBQ table from raw/ into bbq_external.csv.

Externally-published BBQ numbers only — nothing here comes from src/bbq_eval/.
Everything is normalised onto the scale used by src/bbq_eval/bbq_metrics.py:
s_DIS / s_AMB / accuracy_* are fractions in [-1, 1] / [0, 1], never percent.

Re-run after editing anything in raw/:

    python app/external/build_external.py
"""

import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
OUT = HERE / "bbq_external.csv"

FIELDS = [
    "source_id",
    "comparability_group",
    "vendor",
    "model",
    "accuracy_ambig",
    "accuracy_disambig",
    "accuracy_overall",
    "s_AMB",
    "s_AMB_status",
    "s_DIS",
    "s_DIS_status",
    "p_not_stereotype_ambig",
    "n_eval",
    "setup",
    "source_ref",
]

# HELM model name -> vendor. First pattern that matches wins.
HELM_VENDORS = [
    (r"^(GPT|o[134]\b|o1-|o3-|o4-|gpt-oss|text-|davinci)", "OpenAI"),
    (r"^Claude", "Anthropic"),
    (r"^(Gemini|Gemma|PaLM|Flan|T5)", "Google"),
    (r"^(Llama|Llama-|LLaMA)", "Meta"),
    (r"^(Mistral|Mixtral|Ministral)", "Mistral"),
    (r"^Qwen", "Alibaba"),
    (r"^DeepSeek", "DeepSeek"),
    (r"^Grok", "xAI"),
    (r"^(Command|Cohere)", "Cohere"),
    (r"^(OLMo|OLMoE|Tulu)", "Ai2"),
    (r"^Marin", "Stanford"),
    (r"^(IBM )?Granite", "IBM"),
    (r"^(GLM|ChatGLM)", "Zhipu"),
    (r"^Kimi", "Moonshot"),
    (r"^DBRX", "Databricks"),
    (r"^(Yi|Phi|Jamba|Falcon|Arctic|Solar|Nemotron)", "other"),
]


def helm_vendor(name: str) -> str:
    for pat, vendor in HELM_VENDORS:
        if re.match(pat, name, re.I):
            return vendor
    return ""


def pct(v):
    """Percent string -> fraction, blank-safe."""
    if v is None or v == "":
        return ""
    return round(float(v) / 100.0, 6)


def load_helm():
    rows = []
    with open(RAW / "helm_safety_v1.17.0.csv") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "source_id": "helm_safety_v1.17.0",
                    "comparability_group": "helm_safety_v1.17.0",
                    "vendor": helm_vendor(r["model"]),
                    "model": r["model"],
                    # HELM publishes one pooled accuracy, not a per-condition split.
                    "accuracy_ambig": "",
                    "accuracy_disambig": "",
                    "accuracy_overall": r["bbq_accuracy"],
                    "s_AMB": r["s_AMB_helm"],
                    "s_AMB_status": "degraded",
                    "s_DIS": r["s_DIS_helm"],
                    "s_DIS_status": "artifact",
                    "p_not_stereotype_ambig": "",
                    "n_eval": r["n_eval"],
                    "setup": "0-shot, letter generation, temp 0, 1000 sampled instances",
                    "source_ref": "HELM Safety v1.17.0 groups/bbq.json",
                }
            )
    return rows


# Which OpenAI system card a row came from decides its comparability group,
# because OpenAI changed the harness between cards (GPT-4o moves 0.97/0.72 -> 0.88/0.85).
OPENAI_GROUPS = {
    "o1 SC (2024-12)": "openai_syscard_o1",
    "GPT-4.5 SC (2025-02)": "openai_syscard_gpt45",
    "o3/o4-mini SC (2025-04)": "openai_syscard_o3",
    "GPT-5 SC (2025-08)": "openai_syscard_gpt5",
}


def load_vendor():
    rows = []
    with open(RAW / "vendor_system_cards.csv") as f:
        for r in csv.DictReader(f):
            vendor, src = r["vendor"], r["source"]
            if vendor == "Anthropic":
                source_id = group = "anthropic_syscard"
                setup = "vendor-internal; standard (non-extended) thinking mode"
                s_status = "ok"
            elif vendor == "OpenAI":
                source_id = group = OPENAI_GROUPS[src]
                setup = "vendor-internal"
                s_status = "unavailable"
            else:
                source_id = group = "google_gemma2" if "Gemma 2" in src else "google_gemma1"
                setup = "4-shot, top-1" if "4-shot" in src else "1-shot, top-1"
                s_status = "unavailable"

            rows.append(
                {
                    "source_id": source_id,
                    "comparability_group": group,
                    "vendor": vendor,
                    "model": r["model"],
                    "accuracy_ambig": pct(r["acc_ambig_pct"]),
                    "accuracy_disambig": pct(r["acc_disambig_pct"]),
                    "accuracy_overall": "",
                    "s_AMB": pct(r["s_AMB_pct"]),
                    "s_AMB_status": s_status,
                    "s_DIS": pct(r["s_DIS_pct"]),
                    "s_DIS_status": s_status,
                    "p_not_stereotype_ambig": r["P_not_stereotype_ambig"],
                    "n_eval": "",
                    "setup": setup,
                    "source_ref": src,
                }
            )
    return rows


def main():
    rows = load_vendor() + load_helm()
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    groups = {}
    for r in rows:
        groups[r["comparability_group"]] = groups.get(r["comparability_group"], 0) + 1
    usable = sum(1 for r in rows if r["s_DIS_status"] == "ok")

    print(f"wrote {OUT.relative_to(HERE.parent.parent)}  ({len(rows)} rows)")
    for g, n in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"  {g:26s} {n:3d}")
    print(f"\nrows with a usable bias score (s_DIS_status == 'ok'): {usable}")

    registry = json.loads((HERE / "sources.json").read_text())["sources"]
    missing = groups.keys() - registry.keys()
    if missing:
        raise SystemExit(f"comparability groups absent from sources.json: {sorted(missing)}")


if __name__ == "__main__":
    main()
