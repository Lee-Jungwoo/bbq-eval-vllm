#!/usr/bin/env python3
"""Collect the per-run summaries of both conditions into two small CSVs.

Reads only files that are committed (results/runs*/<run_id>/{meta,summary}.json and
results/leaderboard/exports/leaderboard.csv) and copies values; it computes no new
metric. Writes:

  results/summary_original.csv       one row per model, 3-option condition
  results/summary_forced_choice.csv  one row per model, 2-option condition

Values come from:
  accuracy, s_DIS, s_AMB, s_AMB_unscaled, n_*  summary.json (aggregate*.py)
  B, B_lo, B_hi, rank, eligible (original)      leaderboard.csv (publish.py)
  B, B_lo, B_hi (forced_choice)                 summary.json (aggregate_binary.py)

Usage:
    python scripts/export_summary.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bbq_eval.bbq_metrics import family_and_size  # noqa: E402

RESULTS = REPO_ROOT / "results"


def load_runs(runs_dir: Path):
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        meta = json.loads((summary_path.parent / "meta.json").read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))[meta["run_id"]]
        yield meta, summary


def base_row(meta: dict, overall: dict) -> dict:
    family, size = family_and_size(meta["model"])
    return {
        "model": meta["model"],
        "family": family,
        "params_b": size,
        "run_id": meta["run_id"],
        "model_revision": meta["model_revision"],
        "dtype": meta["dtype"],
        "n": overall["n"],
        "accuracy_disambig": overall["accuracy_disambig"],
        "accuracy_ambig": overall["accuracy_ambig"],
        "s_DIS": overall["s_DIS"],
        "s_AMB": overall["s_AMB"],
        "s_AMB_unscaled": overall["s_AMB_unscaled"],
    }


def write(rows: list[dict], path: Path) -> None:
    rows.sort(key=lambda r: (r["family"], r["params_b"] or 0))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.relative_to(REPO_ROOT)} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    board = {r["run_id"]: r for r in csv.DictReader(
        open(RESULTS / "leaderboard" / "exports" / "leaderboard.csv", encoding="utf-8"))}

    original = []
    for meta, s in load_runs(RESULTS / "runs"):
        o = s["overall"]
        b = board[meta["run_id"]]
        original.append({
            **base_row(meta, o),
            "unknown_rate_ambig": b["unknown_rate_ambig"],
            "n_answered_ambig": o["bias_counts"]["ambig"]["n_non_unknown"],
            "B": b["B"], "B_lo": b["B_lo"], "B_hi": b["B_hi"],
            "eligible": b["eligible"], "rank": b["rank"],
            "mean_prob_mass_captured": s["mean_prob_mass_captured"],
        })
    write(original, RESULTS / "summary_original.csv")

    forced = []
    for meta, s in load_runs(RESULTS / "runs_binary"):
        o = s["overall"]
        forced.append({
            **base_row(meta, o),
            "target_rate_ambig": o["target_rate_ambig"],
            "letter_A_rate": o["letter_A_rate"],
            "n_unmeasurable": o["n_unmeasurable"],
            "B": o["B"], "B_lo": o.get("B_lo"), "B_hi": o.get("B_hi"),
            "mean_prob_mass_captured": s["mean_prob_mass_captured"],
        })
    write(forced, RESULTS / "summary_forced_choice.csv")


if __name__ == "__main__":
    main()
