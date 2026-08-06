#!/usr/bin/env bash
# Run the whole pipeline (prepare -> logprobs -> collect -> aggregate) for a
# ladder of models, one after another, on a single GPU.
#
# Every step is idempotent, so rerunning the script resumes rather than redoing
# work: prepare keeps a matching requests.jsonl, serve_logprobs.py skips indices
# already in responses.jsonl (and returns before loading the model when nothing
# is left to do), and collect only appends records that are not there yet.
#
# Usage:
#   ./run_ladder.sh                        # default Qwen2.5 size ladder
#   ./run_ladder.sh Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.1-8B-Instruct
#   GPU=0 BATCH_SIZE=256 ./run_ladder.sh   # env overrides
#   GPU=0,1 ./run_ladder.sh Qwen/Qwen2.5-14B-Instruct   # shard over two GPUs
#   LIMIT_IMAGES=0 ./run_ladder.sh mistralai/Ministral-3-3B-Instruct-2512-BF16
#                                          # vision-capable models: see LIMIT_IMAGES below
#
# Long runs: launch under tmux, or
#   nohup ./run_ladder.sh > ladder.out 2>&1 &
set -uo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PIPELINE_DIR"

GPU="${GPU:-1}"
# Tensor parallelism defaults to however many devices GPU lists, so
# `GPU=0,1 ./run_ladder.sh ...` shards across both without a second knob. Set TP
# explicitly to override (e.g. TP=1 to keep two independent single-GPU runs).
N_GPU="$(awk -F',' '{print NF}' <<< "$GPU")"
TP="${TP:-$N_GPU}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# vLLM scheduler concurrency. Empty = vLLM's own default. Its sampler warmup
# allocates a vocab-wide logits row per sequence, which is what OOMs on a big
# model long before the KV cache does -- BATCH_SIZE does not affect it.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
# Vision-capable models (Ministral-3, Mistral-Small-3.x, ...) profile with a
# dummy image of a few thousand tokens at startup, which does not fit
# MAX_MODEL_LEN=1024. LIMIT_IMAGES=0 turns the image path off -- BBQ prompts are
# text-only, so nothing is lost. Empty = vLLM default (text-only models).
LIMIT_IMAGES="${LIMIT_IMAGES:-}"
CONFIG="${CONFIG:-$PIPELINE_DIR/config.json}"

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    # Smallest first: failures surface early and the cheap points land first.
    MODELS=(
        Qwen/Qwen2.5-0.5B-Instruct
        Qwen/Qwen2.5-1.5B-Instruct
        Qwen/Qwen2.5-3B-Instruct
        Qwen/Qwen2.5-7B-Instruct
    )
fi

LOG_DIR="$PIPELINE_DIR/logs/ladder_$(date +%Y%m%d_%H%M%S)"
CONFIG_DIR="$LOG_DIR/configs"
mkdir -p "$CONFIG_DIR"

# vLLM parses CUDA_VISIBLE_DEVICES as plain integers, so a GPU UUID will not
# work here; PCI_BUS_ID makes the index match what nvidia-smi shows.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"

if pgrep -f 'serve_logprobs.py --input' >/dev/null; then
    echo "ERROR: serve_logprobs.py is already running — a second vLLM process on"
    echo "       the same GPU will OOM. Wait for it to finish, then rerun."
    exit 1
fi

json_get() {
    python -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}

echo "[ladder] GPU=$GPU (tensor_parallel=$TP) dtype=$DTYPE batch=$BATCH_SIZE max_model_len=$MAX_MODEL_LEN"
echo "[ladder] models: ${MODELS[*]}"
echo "[ladder] logs:   $LOG_DIR"

RECORDS=()
FAILED=()

for model in "${MODELS[@]}"; do
    short="${model##*/}"
    started=$SECONDS
    echo
    echo "================================ $model ================================"

    # Per-model config: everything from config.json except the model, with
    # run_id left null so pipeline.py derives a separate run dir per model.
    config="$CONFIG_DIR/$short.json"
    if ! python - "$CONFIG" "$model" "$config" <<'PY'
import json, sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
config["model"] = sys.argv[2]
config["run_id"] = None
with open(sys.argv[3], "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
PY
    then
        FAILED+=("$model (config)")
        continue
    fi

    prepare_log="$LOG_DIR/$short.1_prepare.log"
    if ! python pipeline.py prepare --config "$config" 2>&1 | tee "$prepare_log"; then
        FAILED+=("$model (prepare)")
        continue
    fi
    run_id="$(sed -n 's/^\[pipeline\] run_id=//p' "$prepare_log" | head -1)"
    if [ -z "$run_id" ]; then
        FAILED+=("$model (could not parse run_id)")
        continue
    fi
    run_dir="$PIPELINE_DIR/runs/$run_id"

    revision="$(json_get "$run_dir/meta.json" model_revision)"
    n_instances="$(json_get "$run_dir/meta.json" n_instances)"
    echo "[ladder] run_id=$run_id revision=$revision n_instances=$n_instances"

    if ! python serve_logprobs.py \
            --input "$run_dir/requests.jsonl" \
            --output "$run_dir/responses.jsonl" \
            --dtype "$DTYPE" \
            --revision "$revision" \
            --max-model-len "$MAX_MODEL_LEN" \
            --batch-size "$BATCH_SIZE" \
            --tensor-parallel-size "$TP" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            ${MAX_NUM_SEQS:+--max-num-seqs $MAX_NUM_SEQS} \
            ${LIMIT_IMAGES:+--limit-images $LIMIT_IMAGES} \
            2>&1 | tee "$LOG_DIR/$short.2_serve.log"; then
        FAILED+=("$model (serve_logprobs)")
        continue
    fi

    n_responses=$(wc -l < "$run_dir/responses.jsonl")
    if [ "$n_responses" -lt "$n_instances" ]; then
        echo "[ladder] WARNING: only $n_responses/$n_instances responses — collecting"
        echo "[ladder]          the partial run; rerun this script to fill the rest in."
    fi

    if ! python pipeline.py collect --config "$config" 2>&1 | tee "$LOG_DIR/$short.3_collect.log"; then
        FAILED+=("$model (collect)")
        continue
    fi

    python aggregate.py "$run_dir/records.jsonl" --out "$run_dir/summary.json" \
        2>&1 | tee "$LOG_DIR/$short.4_aggregate.log"

    RECORDS+=("$run_dir/records.jsonl")
    echo "[ladder] $short finished in $((SECONDS - started))s → $run_dir"
done

echo
echo "================================ ladder summary ================================"
if [ ${#RECORDS[@]} -gt 0 ]; then
    python aggregate.py "${RECORDS[@]}" --out "$LOG_DIR/ladder_summary.json" \
        2>&1 | tee "$LOG_DIR/ladder_summary.log"

    # Step 5: republish every run into leaderboard/ and self-check the result
    # against the summary.json files aggregate.py just wrote.
    python publish.py --verify 2>&1 | tee "$LOG_DIR/publish.log"
fi
if [ ${#FAILED[@]} -gt 0 ]; then
    echo
    printf 'FAILED: %s\n' "${FAILED[@]}"
    exit 1
fi
echo
echo "[ladder] all ${#RECORDS[@]} models done — $LOG_DIR/ladder_summary.json"
