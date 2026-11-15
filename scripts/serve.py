#!/usr/bin/env python3
"""Run vLLM inference for one model from its serving config.

Reads configs/serving/<model>.yaml and launches src/bbq_eval/serve_logprobs.py with
the same flags scripts/run_ladder.sh passes. This is offline batch inference with
the vllm.LLM class (no HTTP server): serve_logprobs.py reads a pre-tokenized
requests.jsonl, calls llm.generate() batch by batch, and appends first-token
logprobs to responses.jsonl. It resumes from whatever responses.jsonl already
holds.

Examples:
    # inputs from a prepared run directory (revision is taken from its meta.json)
    python scripts/serve.py --config configs/serving/qwen2.5-3b-instruct.yaml \
        --run-dir results/runs/Qwen2.5-3B-Instruct_bfloat16_6db43b09

    # on the GPU server, with requests.jsonl copied next to this script
    python scripts/serve.py --config qwen2.5-3b-instruct.yaml \
        --input requests.jsonl --output responses.jsonl --gpu 0

    # print the command and environment without running anything
    python scripts/serve.py --config ... --run-dir ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVE_SCRIPT = REPO_ROOT / "src" / "bbq_eval" / "serve_logprobs.py"


def load_serving_config(path: str | Path) -> dict:
    import yaml  # local import: --help works without PyYAML

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for key in ("model", "revision", "engine", "runner"):
        if key not in config:
            raise SystemExit(f"{path}: missing '{key}'")
    if config["engine"].get("quantization") is not None:
        raise SystemExit(f"{path}: engine.quantization is set, but serve_logprobs.py has "
                         "no quantization flag (no run used one)")
    return config


def build_command(config: dict, input_path: str | Path, output_path: str | Path,
                  revision: str, python: str = sys.executable,
                  serve_script: str | Path = SERVE_SCRIPT) -> list[str]:
    """serve_logprobs.py argv for this config, flag for flag what run_ladder.sh passes.

    --top-logprobs and --max-tokens are passed explicitly; run_ladder.sh leaves them
    at serve_logprobs.py's defaults (50 and 1), which is what the configs hold.
    """
    engine, runner = config["engine"], config["runner"]
    cmd = [
        python, str(serve_script),
        "--input", str(input_path),
        "--output", str(output_path),
        "--dtype", str(engine["dtype"]),
        "--revision", str(revision),
        "--max-model-len", str(engine["max_model_len"]),
        "--batch-size", str(runner["batch_size"]),
        "--tensor-parallel-size", str(engine["tensor_parallel_size"]),
        "--gpu-memory-utilization", str(engine["gpu_memory_utilization"]),
        "--top-logprobs", str(runner["top_logprobs"]),
        "--max-tokens", str(runner["max_tokens"]),
    ]
    if engine.get("max_num_seqs") is not None:
        cmd += ["--max-num-seqs", str(engine["max_num_seqs"])]
    if engine.get("limit_images") is not None:
        cmd += ["--limit-images", str(engine["limit_images"])]
    return cmd


def build_env(config: dict, gpu: str | None) -> dict[str, str]:
    """The two CUDA variables run_ladder.sh exports."""
    devices = gpu if gpu is not None else str(config.get("hardware", {}).get(
        "cuda_visible_devices", "0"))
    return {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": devices}


def resolve_io(config: dict, args: argparse.Namespace) -> tuple[Path, Path, str]:
    if args.run_dir:
        run_dir = Path(args.run_dir)
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            raise SystemExit(f"{meta_path} not found; run the prepare step first")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["model"] != config["model"]:
            raise SystemExit(f"run dir is for {meta['model']}, config is for {config['model']}")
        # Same as run_ladder.sh: the revision prepare resolved and recorded wins.
        revision = meta["model_revision"]
        if revision != config["revision"]:
            print(f"[serve] note: meta.json revision {revision} differs from the config's "
                  f"{config['revision']}; using meta.json's", file=sys.stderr)
        return run_dir / "requests.jsonl", run_dir / "responses.jsonl", revision
    if not (args.input and args.output):
        raise SystemExit("give --run-dir, or both --input and --output")
    return Path(args.input), Path(args.output), args.revision or config["revision"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="configs/serving/<model>.yaml")
    parser.add_argument("--run-dir", help="results/runs[_binary]/<run_id> made by prepare")
    parser.add_argument("--input", help="requests.jsonl (instead of --run-dir)")
    parser.add_argument("--output", help="responses.jsonl (instead of --run-dir)")
    parser.add_argument("--revision", help="override the config's pinned revision "
                                           "(only with --input/--output)")
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1 "
                                      "(default: hardware.cuda_visible_devices)")
    parser.add_argument("--serve-script", default=str(SERVE_SCRIPT),
                        help="path to serve_logprobs.py (default: the one in src/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the command and environment, run nothing")
    args = parser.parse_args()

    config = load_serving_config(args.config)
    input_path, output_path, revision = resolve_io(config, args)
    cmd = build_command(config, input_path, output_path, revision,
                        serve_script=args.serve_script)
    env = build_env(config, args.gpu)

    n_devices = len(env["CUDA_VISIBLE_DEVICES"].split(","))
    if n_devices != config["engine"]["tensor_parallel_size"]:
        print(f"[serve] warning: {n_devices} visible device(s) but tensor_parallel_size="
              f"{config['engine']['tensor_parallel_size']}", file=sys.stderr)

    print("[serve] " + " ".join(f"{k}={v}" for k, v in env.items()) + " "
          + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if args.dry_run:
        return
    sys.exit(subprocess.call(cmd, env={**os.environ, **env}))


if __name__ == "__main__":
    main()
