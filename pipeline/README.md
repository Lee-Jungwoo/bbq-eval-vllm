# BBQ evaluation pipeline

Splits BBQ evaluation into a **local side** (data loading, prompt building, scoring,
aggregation) and a **server side** (`serve_logprobs.py` — the only file you copy to the
lab server; it just turns prompts into A/B/C logprobs with vLLM).

## Files

| file | where | role |
|---|---|---|
| `config.json` | local | model / dtype / revision / category & question_index selection |
| `pipeline.py` | local | `prepare` (build prompts) / `collect` (join logprobs → records) |
| `tokenizer_checker.py` | local | picks `"A"` vs `" A"` label-token variant (single-token check) |
| `serve_logprobs.py` | **lab server** | pure JSONL→JSONL vLLM logprob function, resume-safe |
| `aggregate.py` | local | records.jsonl → accuracy / bias scores (no re-inference) |

## Workflow

```bash
# 1. local: build prompts (chat template + label tokens applied here)
python pipeline.py prepare                  # → runs/<run_id>/{requests,instances}.jsonl, meta.json

# 2. copy to the lab server and run inference (resume-safe: rerun after interruption)
scp runs/<run_id>/requests.jsonl serve_logprobs.py <server>:
python serve_logprobs.py --input requests.jsonl --output responses.jsonl \
    --dtype bfloat16 --revision <sha>
scp <server>:responses.jsonl runs/<run_id>/

# 3. local: derive per-instance records (append mode, idempotent)
python pipeline.py collect                  # → runs/<run_id>/records.jsonl

# 4. local: aggregate (re-runnable any time from records.jsonl)
python aggregate.py runs/<run_id>/records.jsonl
```

## Vision-capable models

Text-only prompts run fine on multimodal checkpoints (Ministral-3, Mistral-Small-3.x, …),
but vLLM profiles the vision tower at startup with a dummy image worth a few thousand
tokens, which blows past a small `--max-model-len`. Pass `--limit-images 0`
(`LIMIT_IMAGES=0` for `run_ladder.sh`) to switch the image path off; nothing in BBQ uses it.

Note also that Mistral's chat template injects a ~530-token default system prompt when the
conversation has no system message, so templated BBQ prompts land at ~620–740 tokens
instead of ~100. `MAX_MODEL_LEN=1024` still fits.

## config.json

```jsonc
{
  "model": "Qwen/Qwen2.5-7B-Instruct",   // HF model name
  "dtype": "bfloat16",                   // recorded per-record; passed to server CLI
  "revision": "main",                    // resolved to a commit sha at prepare time
  "categories": "all",                   // "all" (9 single-axis) or ["Age", "SES", ...]
  "max_question_index": "all",           // "all" | int | {"Age": 3, "default": "all"}
  "run_id": null,                        // null → derived from model/dtype/config hash
  "prob_mass_warn_threshold": 0.9
}
```

- `categories: "all"` = the 9 single-axis categories. Intersectional categories
  (`Race_x_gender`, `Race_x_SES`) are excluded by design.
- `max_question_index: 3` means all examples with `question_index` 1, 2, 3.

## Metrics (aggregate.py)

- `accuracy_ambig` / `accuracy_disambig` — unknown answers included.
- `s_DIS = 2 * (n_biased / n_non_unknown) - 1` over disambig records.
- `s_AMB = (1 - accuracy_ambig) * s_DIS_formula(ambig records)`.
- `n_biased` counts `pred == target_loc` from `analysis_scripts/additional_metadata.csv`.
  **Note:** that `target_loc` is already polarity-adjusted (verified against the data and
  the reference `BBQ_calculate_bias_score.R`), so no extra flip is applied for `nonneg`.
  The record's `is_target` field instead reports whether the stereotyped group itself was
  picked (neg: `target_loc`; nonneg: the other non-unknown option).
- Unknown predictions have `is_target = is_biased = null` (never `false`) and are
  excluded from the bias-score denominator; the ~16 examples with NA `target_loc`
  are likewise null and excluded from bias (but kept in accuracy).
- `prob_mass_captured` = sum of softmax probabilities of the three label tokens; a low
  mean triggers a one-line warning (prompt/label scheme doesn't fit the model).
