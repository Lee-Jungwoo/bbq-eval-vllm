# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

BBQ (Bias Benchmark for QA) — a hand-built dataset of QA question sets that probe social biases across nine social dimensions (plus two intersectional ones) in U.S. English contexts. This is a fork/working copy of the NYU-MLL BBQ dataset (Findings of ACL 2022), being adapted under `HCCL/`. The full paper is `QA_bias_benchmark.pdf`.

There is no build system, package manifest, or test suite. Work here is Python data-generation scripts, R analysis scripts, and Jupyter exploration. The `.python-version` pins the pyenv env name `hccl-bbq`. Core Python deps are `pandas` / `numpy`; R scripts use `tidyverse`, `jsonlite`, `scales`.

## The generation pipeline (how the data is produced)

The benchmark data in `data/*.jsonl` is *generated* from templates + vocabulary — do not hand-edit the jsonl. Two entry-point scripts drive generation:

- `generate_from_template_all_categories.py` — the 9 single-axis categories (Age, Disability_status, Gender_identity, Nationality, Physical_appearance, Race_ethnicity, Religion, SES, Sexual_orientation).
- `generate_from_template_intersectional_cats.py` — the 2 intersectional categories (Race_x_gender, Race_x_SES).

Both import `utils.py` and read from `templates/`, writing one `data/<Category>.jsonl` per category. `utils.py` holds the shared machinery:
- `do_slotting()` — fills `{{NAME1}}`, `{{NAME2}}`, `{{GEN1/2}}`, `{{OCC1/2}}`, `{{WORD1/2}}` slots in a template row and fixes English agreement (a/an, `lesbian man`→`lesbian woman`, `Latino woman`→`Latina woman`).
- `create_templating_dicts()` — for one filled template + name pair, crosses **context_condition** (`ambig`/`disambig`) × **question_polarity** (`neg`/`nonneg`) into exactly **4 output items**, randomizing answer-option order and recording the correct-answer index in `label`.
- `make_dict()` — serializes one item to the output schema.

Key generation concepts to understand before changing anything:
- Each template yields many items by pairing a **target** word (`NAME1`, a stereotyped group) against a **non-target** (`NAME2`), downsampling large vocab lists to ~5. For categories in `need_stereotyping_subset` (Race_ethnicity, Religion, Nationality, Sexual_orientation), and for proper-name / SES-Occupation templates, the pair is also generated **flipped** (target/non-target swapped) so bias direction is balanced.
- The 3rd answer is always an "unknown" option drawn from `unknown_options` — this is the *correct* answer in ambiguous contexts.
- `answer_info` records `[text, group-info]` for each of ans0/ans1/ans2, used later for scoring. `additional_metadata.stereotyped_groups` records the bias targets.

### ⚠️ Path mismatch gotcha
The generation scripts as committed reference paths that **do not match this repo's layout**: they `from BBQ_Full.utils import *` and read `BBQ_full/templates/...`, `BBQ_full/data/...` (an upstream nested-directory layout). In this checkout everything lives at the repo root (`utils.py`, `templates/`, `data/`). To run generation here you must fix the import to `from utils import *` and strip the `BBQ_full/` prefix from the file paths (or run from a parent dir that provides that structure). Confirm the intended layout with the user before "fixing" — the committed paths may reflect how they run it elsewhere.

## Running things

```bash
# Regenerate all single-axis category jsonl files (after fixing paths per above)
python generate_from_template_all_categories.py
# Regenerate intersectional categories
python generate_from_template_intersectional_cats.py
```

Generation is **stochastic** (`random.sample`/`shuffle`/`choice` with no seed) — rerunning changes exact items and ordering. Don't expect byte-identical `data/` output across runs.

## The evaluation pipeline (`bbq_pipeline/`)

Separate from generation. Splits BBQ evaluation into a local side (prompt building, scoring, aggregation) and a server side (`serve_logprobs.py`, the only file copied to the lab GPU server). Five steps:

```bash
cd bbq_pipeline
python pipeline.py prepare        # data/*.jsonl + templates → runs/<run_id>/{requests,instances}.jsonl, meta.json
# … scp requests.jsonl to the server, run serve_logprobs.py, scp responses.jsonl back …
python pipeline.py collect        # → runs/<run_id>/records.jsonl (one row per instance)
python aggregate.py runs/<run_id>/records.jsonl --out runs/<run_id>/summary.json
python publish.py --verify        # all runs → ../leaderboard/{bbq.sqlite,exports/} + the ranking
streamlit run ../leaderboard/app/Home.py
```

`run_ladder.sh` drives all five for a list of models. See `bbq_pipeline/README.md`.

There is a second, **binary arm** (`pipeline_binary.py` / `aggregate_binary.py` /
`run_ladder_binary.sh` / `config_binary.json`) that asks the same items as a forced A/B
choice with the "unknown" option removed from the prompt in *both* context conditions.
Its runs live in `bbq_pipeline/runs_binary/` and are never published: `publish.py` and
`aggregate.py` refuse binary input, `aggregate_binary.py` refuses 3-option input, and the
leaderboard stays 3-option only. `serve_logprobs.py` is shared unchanged. In that arm
ambiguous items have **no correct answer** (`label`/`correct` are null, `accuracy_ambig`
is null) and `s_AMB` is *unscaled* — it is comparable to the 3-option `s_AMB_unscaled`,
not to `s_AMB`. See the "Binary arm" section of `bbq_pipeline/README.md`.

- `bbq_pipeline/bbq_metrics.py` is the **single source of truth for every metric** — `summarize`, `answered_stats`, `polarity_stats`, `wilson`, the margin stats, the palette, and the bootstrap. `publish.py` and the analysis notebooks both import it. Do not restate a metric anywhere else.
- `bbq_pipeline/aggregate.py` is deliberately left alone: it is the oracle that `publish.py --verify` checks against (1,441 exact comparisons). Changing it destroys the check. (Its only edit is an input guard that rejects binary-arm records — no metric logic touched; re-aggregating a run still reproduces its committed `summary.json` byte-for-byte.)
- The three `pipeline*.ipynb` notebooks (Qwen2.5 / Llama / Ministral) are exploratory and still hold their own copies of those functions; `leaderboard/app/pages/2_Scaling_curves.py` reproduces their headline figures from the database.

**Metric gotcha that matters more than any other:** `is_biased` / `is_target` are tri-state — `None` for an unknown prediction and for the ~16 examples with no `target_loc`, never `False`. `fillna(0)` on either counts every abstention as an anti-stereotype pick and drags every bias score downward.

**Interpretation gotcha:** `s_AMB = (1 - accuracy_ambig) * s_AMB_unscaled`, and in an ambiguous context the correct answer *is* "unknown" — so `accuracy_ambig` is essentially the abstention rate and `s_AMB` collapses toward zero for a model that simply declines to answer. Always read `s_AMB_unscaled` / `s_answered` / `n_elig_ambig` alongside it. See `leaderboard/README.md`.

## Leaderboard (`leaderboard/`)

`bbq.sqlite` (gitignored, regenerable) + committed CSV/JSON exports + a Streamlit app. Ranking is `B(m) = |s_DIS| + |s_AMB|` ascending among models passing `accuracy_disambig >= 0.7`, within a comparability group, with a paired cluster-bootstrap CI on B.

Two pages do **not** read `bbq.sqlite`: `3_External_leaderboard` (results published by other people, from `leaderboard/external/`) and `4_Binary_leaderboard` (the binary arm, read straight from `bbq_pipeline/runs_binary/*/summary.json` via `app/lib/binary.py`). The binary board drops `acc_amb` and every abstention diagnostic — they do not exist without an unknown option — and shows `target_rate` / `letter_A_rate` instead. `table.render(..., columns=...)` takes a column subset, which is how both pages render a board that lacks some metrics.

## Data schema (`data/*.jsonl`, one JSON object per line)

`example_id`, `question_index` (template id), `question_polarity` (`neg`/`nonneg`), `context_condition` (`ambig`/`disambig`), `category`, `answer_info` (`{ans0/1/2: [text, group]}`), `additional_metadata` (`subcategory`, `stereotyped_groups`, `version`, `source`), `context`, `question`, `ans0`/`ans1`/`ans2` (the three option strings), `label` (index 0–2 of the correct answer).

## Scoring & analysis (R)

`analysis_scripts/BBQ_calculate_bias_score.R` is the reference implementation of the bias metric. It joins model predictions against `analysis_scripts/additional_metadata.csv` (also mirrored at `supplemental/additional_metadata.csv`), keying on `category` + `example_id`, and uses `target_loc` (index of the bias-target answer) to compute the bias score. Reads model outputs from `results/UnifiedQA/*.jsonl` and `results/RoBERTa_and_DeBERTaV3/df_bbq.csv`. R scripts `setwd()` to their own file location via `rstudioapi` and use `../data`, `../results` relative paths — run them from `analysis_scripts/`.

## Directory map

- `templates/` — source of truth for generation: `new_templates - <Category>.csv` (one per category), `vocabulary.csv` (group words; rows with `Pilot_include == "No"` are dropped), `vocabulary_proper_names.csv` (first/last names with gender + ethnicity metadata).
- `data/` — generated benchmark jsonl (the files you evaluate models against).
- `results/` — published model outputs (UnifiedQA jsonl per category; RoBERTa/DeBERTaV3 csv of logits).
- `analysis_scripts/` — R scoring + metadata.
- `supplemental/` — MTurk validation HITs/scripts/results, name & occupation-prestige source data, `additional_metadata.csv`.
- `bbq_pipeline/` — the evaluation pipeline (above). `bbq_pipeline/runs/<run_id>/` holds per-run artifacts; only `meta.json` and `summary.json` are version-controlled.
- `leaderboard/` — published database, committed exports, and the Streamlit app.
