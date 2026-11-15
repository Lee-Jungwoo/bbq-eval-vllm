#!/usr/bin/env python3
"""Evaluate one model on BBQ under one condition.

Conditions:
  original       3 options (A/B/C), the "unknown" option included (BBQ as published)
                 code: src/bbq_eval/pipeline.py, aggregate.py -> results/runs/
  forced_choice  2 options (A/B), the "unknown" option removed in every item
                 code: src/bbq_eval/pipeline_binary.py, aggregate_binary.py -> results/runs_binary/

Steps (default: all four, in order; this is one iteration of scripts/run_ladder*.sh):
  prepare    build chat-templated, pre-tokenized prompts -> <run_dir>/requests.jsonl
  serve      vLLM first-token logprobs (scripts/serve.py)  -> <run_dir>/responses.jsonl
  collect    logprobs -> prediction and bias fields          -> <run_dir>/records.jsonl
  aggregate  records -> accuracy and bias scores             -> <run_dir>/summary.json

Every step resumes: rerunning skips work that is already on disk.

Examples:
    python scripts/run_eval.py --condition original \
        --serving configs/serving/qwen2.5-3b-instruct.yaml

    # local side only (no GPU): prompts now, inference on the server later
    python scripts/run_eval.py --condition forced_choice \
        --serving configs/serving/qwen3-8b.yaml --steps prepare

    # build and print the first 5 prompts and the serve command; writes nothing
    python scripts/run_eval.py --condition original \
        --serving configs/serving/qwen3-0.6b.yaml --dry-run 5
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

CONDITIONS = {
    "original": {
        "pipeline": "bbq_eval.pipeline",
        "aggregate": "bbq_eval.aggregate",
        "eval_config": REPO_ROOT / "configs" / "eval" / "original.json",
        "runs_dir": REPO_ROOT / "results" / "runs",
        "run_id_prefix": "[pipeline] run_id=",
    },
    "forced_choice": {
        "pipeline": "bbq_eval.pipeline_binary",
        "aggregate": "bbq_eval.aggregate_binary",
        "eval_config": REPO_ROOT / "configs" / "eval" / "forced_choice.json",
        "runs_dir": REPO_ROOT / "results" / "runs_binary",
        "run_id_prefix": "[pipeline_binary] run_id=",
    },
}
STEPS = ["prepare", "serve", "collect", "aggregate"]


def per_model_config(eval_config_path: Path, model: str) -> dict:
    """Same derivation as run_ladder.sh: the condition's config with the model
    swapped in and run_id left null, so the run id is derived per model."""
    config = json.loads(eval_config_path.read_text(encoding="utf-8"))
    config["model"] = model
    config["run_id"] = None
    return config


def pipeline_module(condition: str):
    if condition == "original":
        from bbq_eval import pipeline as mod
    else:
        from bbq_eval import pipeline_binary as mod
    return mod


def derive_run_id(condition: str, config: dict) -> str:
    mod = pipeline_module(condition)
    return mod.derive_run_id(config, mod.resolve_categories(config))


def run(cmd: list[str], capture: bool = False) -> str:
    env = {**os.environ,
           "PYTHONPATH": str(SRC_DIR) + (os.pathsep + os.environ["PYTHONPATH"]
                                         if os.environ.get("PYTHONPATH") else "")}
    print("[run_eval] $ " + " ".join(cmd), flush=True)
    if not capture:
        subprocess.run(cmd, env=env, check=True)
        return ""
    proc = subprocess.run(cmd, env=env, check=True, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    sys.stdout.write(proc.stdout)
    return proc.stdout


def dry_run(condition: str, serving: dict, config: dict, n: int) -> None:
    """In-process prepare for the first n items; nothing is written to disk."""
    from transformers import AutoTokenizer

    from serve import build_command, build_env

    mod = pipeline_module(condition)
    if condition == "original":
        build, letters = mod.load_instances, None
    else:
        build, letters = mod.build_instances, mod.LETTERS

    categories = mod.resolve_categories(config)
    run_id = derive_run_id(condition, config)
    run_dir = CONDITIONS[condition]["runs_dir"] / run_id
    print(f"[dry-run] condition={condition} model={config['model']}")
    print(f"[dry-run] run_id={run_id} (recorded run exists: {(run_dir / 'meta.json').exists()})")

    target_locs = mod.load_target_locs(mod.BBQ_ROOT / "analysis_scripts"
                                       / "additional_metadata.csv")
    instances = build(config, categories, target_locs)
    print(f"[dry-run] {len(instances)} instances in the full run; showing {n}")

    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    label_info = (mod.choose_label_tokens(tokenizer) if letters is None
                  else mod.choose_label_tokens(tokenizer, letters=letters))
    prefill = mod.assistant_prefill_for(config["model"])
    print(f"[dry-run] label tokens: {label_info['label_token_ids']} "
          f"(variant={label_info['variant']}), assistant prefill: {prefill!r}")

    decoded = []
    for inst in instances[:n]:
        ids = mod.encode_chat_prompt(tokenizer, inst["prompt_user_content"], prefill=prefill)
        decoded.append(tokenizer.decode(ids, skip_special_tokens=False))
        print(f"\n--- item {inst['index']}  {inst['category']}  {inst['context_condition']}/"
              f"{inst['question_polarity']}  label={inst['label']}  "
              f"target_loc={inst['target_loc']}  ({len(ids)} tokens)")
        print(decoded[-1])

    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("sample_prompt"):
            same = meta["sample_prompt"] == decoded[0]
            print(f"\n[dry-run] item 0 prompt identical to the recorded run's "
                  f"meta.json sample_prompt: {same}")
        print(f"[dry-run] label tokens identical to the recorded run: "
              f"{meta['label_token_ids'] == label_info['label_token_ids']}")

    cmd = build_command(serving, run_dir / "requests.jsonl", run_dir / "responses.jsonl",
                        serving["revision"])
    env = build_env(serving, None)
    print("\n[dry-run] serve step would run:")
    print("  " + " ".join(f"{k}={v}" for k, v in env.items()) + " " + " ".join(cmd))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--serving", required=True, help="configs/serving/<model>.yaml")
    parser.add_argument("--eval-config", default=None,
                        help="override configs/eval/<condition>.json "
                             "(e.g. configs/eval/original_smoke.json)")
    parser.add_argument("--steps", default=",".join(STEPS),
                        help=f"comma-separated subset of {STEPS}")
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES for the serve step")
    parser.add_argument("--dry-run", type=int, metavar="N", default=None,
                        help="build the first N prompts and print them with the serve "
                             "command; no files written, no GPU needed")
    args = parser.parse_args()

    from serve import load_serving_config

    cond = CONDITIONS[args.condition]
    serving = load_serving_config(args.serving)
    eval_config_path = Path(args.eval_config) if args.eval_config else cond["eval_config"]
    config = per_model_config(eval_config_path, serving["model"])

    if args.dry_run is not None:
        dry_run(args.condition, serving, config, args.dry_run)
        return

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise SystemExit(f"unknown steps {unknown}; choose from {STEPS}")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        config_path = f.name

    # The run id is a pure function of the config and the prompt template, so it is
    # computed here without touching disk; prepare (which writes meta.json) only
    # runs when asked for.
    run_id = derive_run_id(args.condition, config)
    run_dir = cond["runs_dir"] / run_id
    print(f"[run_eval] run_id={run_id}")

    if "prepare" in steps:
        out = run([sys.executable, "-m", cond["pipeline"], "prepare",
                   "--config", config_path], capture=True)
        printed = next((line[len(cond["run_id_prefix"]):].split()[0]
                        for line in out.splitlines()
                        if line.startswith(cond["run_id_prefix"])), None)
        if printed != run_id:
            raise SystemExit(f"prepare reported run id {printed}, expected {run_id}")
    if "serve" in steps:
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "serve.py"),
               "--config", args.serving, "--run-dir", str(run_dir)]
        if args.gpu:
            cmd += ["--gpu", args.gpu]
        run(cmd)
    if "collect" in steps:
        run([sys.executable, "-m", cond["pipeline"], "collect", "--config", config_path])
    if "aggregate" in steps:
        run([sys.executable, "-m", cond["aggregate"], str(run_dir / "records.jsonl"),
             "--out", str(run_dir / "summary.json")])
    print(f"[run_eval] done: {run_dir}")


if __name__ == "__main__":
    main()
