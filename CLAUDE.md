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

`pipeline.ipynb` is the user's working notebook (loads each `data/*.jsonl` with `pd.read_json(..., lines=True)` and explores structure). `pipeline.py` is currently empty — likely the intended script form of that notebook.

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
