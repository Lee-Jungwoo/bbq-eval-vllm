#!/usr/bin/env python3
"""vLLM logprob server script — the only file that runs on the lab server.

Pure function over JSONL: knows nothing about BBQ. Reads a requests file whose
first line is a header ({"type": "header", "model": ..., "label_token_ids":
{"A": id, "B": id, "C": id}}) followed by {"index": int, "prompt": str} lines
(prompts are already chat-templated by the local side), and appends
{"index": int, "logprobs": {"A": float|null, "B": ..., "C": ...}} lines to the
output file. Logprobs are log-softmax over the full vocab at the assistant's
first token position, so the local side can compute prob_mass_captured exactly.
A label token that falls outside the top-k logprobs is emitted as null.

With --max-tokens > 1 each line also carries "text" and "finish_reason" (the
free-generation arm). The logprobs field is unchanged and still describes the
first token, so both scoring methods come out of the same forward pass.

Resume: already-present indices in the output file are skipped, and results are
flushed after every batch, so an interrupted run continues where it stopped.

Usage:
    python serve_logprobs.py --input requests.jsonl --output responses.jsonl \
        --dtype bfloat16 --revision <sha-or-branch>
    cat requests.jsonl | python serve_logprobs.py --input - --output responses.jsonl ...
"""
import argparse
import json
import sys


def read_requests(path):
    stream = sys.stdin if path == "-" else open(path, encoding="utf-8")
    header = None
    items = []
    try:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "header":
                header = obj
            else:
                items.append(obj)
    finally:
        if path != "-":
            stream.close()
    if header is None:
        raise SystemExit("requests file has no header line ({'type': 'header', ...})")
    return header, items


def read_done_indices(path):
    done = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["index"])
    except FileNotFoundError:
        pass
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="requests.jsonl path, or - for stdin")
    parser.add_argument("--output", required=True, help="responses.jsonl path (append mode)")
    parser.add_argument("--dtype", required=True, help='e.g. "bfloat16", "float16"')
    parser.add_argument("--revision", default=None, help="HF revision (sha/branch/tag)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-logprobs", type=int, default=50,
                        help="top-k logprobs to request; label tokens outside it become null")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="vLLM scheduler concurrency (default: vLLM's own). The "
                             "sampler warms up with this many dummy requests, each "
                             "holding a vocab-wide logits row, so lower it (64, 32) if "
                             "init OOMs. Unrelated to --batch-size, which only sets how "
                             "many prompts are handed to the engine at a time.")
    parser.add_argument("--max-tokens", type=int, default=1,
                        help="1 = first-token logprobs only; >1 also decodes greedily "
                             "and emits the generated text (free-generation arm)")
    args = parser.parse_args()

    header, items = read_requests(args.input)
    model = header["model"]
    label_token_ids = {k: int(v) for k, v in header["label_token_ids"].items()}

    done = read_done_indices(args.output)
    todo = [it for it in items if it["index"] not in done]
    print(f"[serve_logprobs] {len(items)} prompts, {len(done)} already done, {len(todo)} to run")
    if not todo:
        return

    from vllm import LLM, SamplingParams

    llm_kwargs = {}
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs

    llm = LLM(
        model=model,
        revision=args.revision,
        dtype=args.dtype,
        max_logprobs=args.top_logprobs,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        **llm_kwargs,
    )
    tokenizer = llm.get_tokenizer()
    # max_tokens=1: temperature=1.0 so returned logprobs are the unscaled
    # log-softmax; the sampled token itself is ignored, only the logprob dict is.
    # max_tokens>1: the decoded text is the point, so greedy (temperature=0) for
    # reproducibility. logprobs[0] is still the first-token distribution, which
    # makes the two arms comparable on one forward pass -- but temperature is no
    # longer the identity, so verify the first-token logprobs against an existing
    # max_tokens=1 run before trusting them (see pipeline README).
    sampling = SamplingParams(temperature=0.0 if args.max_tokens > 1 else 1.0,
                              top_p=1.0, max_tokens=args.max_tokens,
                              logprobs=args.top_logprobs)

    with open(args.output, "a", encoding="utf-8") as out:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            # Prompts already contain the chat template (incl. BOS text), so
            # tokenize without special tokens to avoid a double BOS.
            prompts = [
                {"prompt_token_ids": tokenizer.encode(it["prompt"], add_special_tokens=False)}
                for it in batch
            ]
            outputs = llm.generate(prompts, sampling)
            for item, output in zip(batch, outputs):
                completion = output.outputs[0]
                lp = completion.logprobs[0]  # dict: token_id -> Logprob
                record = {
                    "index": item["index"],
                    "logprobs": {
                        letter: (lp[tid].logprob if tid in lp else None)
                        for letter, tid in label_token_ids.items()
                    },
                }
                if args.max_tokens > 1:
                    record["text"] = completion.text
                    record["finish_reason"] = completion.finish_reason
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[serve_logprobs] {min(start + args.batch_size, len(todo))}/{len(todo)} done")


if __name__ == "__main__":
    main()
