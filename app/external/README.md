# External BBQ results

BBQ numbers **published by other people**, scraped from vendor system cards, tech reports and
the HELM Safety release. Nothing in this directory was measured by this repo's pipeline (`src/bbq_eval/`).

This is deliberately kept out of `results/leaderboard/exports/` and out of `bbq.sqlite`. `publish.py` does not read
it and `--verify` does not check it. It exists to back a *separate* leaderboard.

```
external/
  build_external.py            builds bbq_external.csv from raw/
  bbq_external.csv             the unified table (120 rows)
  sources.json                 provenance registry + what was checked and found absent
  raw/
    helm_safety_v1.17.0.csv    87 models, scraped from HELM's public GCS bucket
    vendor_system_cards.csv    33 rows, hand-transcribed from vendor PDFs
```

Rebuild after editing anything in `raw/`:

```bash
python app/external/build_external.py
```

## The app page

`app/pages/3_External_leaderboard.py` renders this table with the same
`lib/table.py` renderer as the Home board. The buttons under the title switch which source is
shown, one source at a time, because the setups are not comparable. Each source gets only the
columns it actually publishes, and a banner saying what it can and cannot support:

| source | banner |
|---|---|
| `anthropic_syscard` | green: bias score is comparable, B(m) ranking is meaningful |
| `openai_syscard_*` | orange: accuracy only, and the harness changed between cards |
| `google_gemma*` | blue: accuracy only, this is not a bias ranking |
| `helm_safety_v1.17.0` | red: `s_DIS` withheld as an artifact, `s_AMB` is a lower bound |

The page never merges sources; the "전체 출처 한눈에 보기" expander is a lookup table, explicitly
labelled as not a ranking.

Vendor logos come from `app/assets/logos/<stem>.webp`. `anthropic`, `deepseek`, `xai` and `cohere`
have no asset yet, so those rows render the blank placeholder; dropping the file in is all it takes.

## Scale

`s_DIS`, `s_AMB`, `accuracy_*` are on the same scale as `src/bbq_eval/bbq_metrics.py`: fractions,
never percent. Anthropic publishes bias as a percentage (`-1.37%`); `build_external.py` divides
by 100 on the way in, so the CSV holds `-0.0137`.

## What each source actually gives you

| group | n | s_DIS | s_AMB | acc split | notes |
|---|---|---|---|---|---|
| `anthropic_syscard` | 14 | ok | ok | ambig + disambig | the only source with a directly comparable bias score |
| `openai_syscard_*` | 13 | n/a | n/a | ambig + disambig | plus `P(not-stereotype \| not unknown)` on the older cards |
| `google_gemma1/2` | 6 | n/a | n/a | ambig + disambig | 1-shot / 4-shot top-1 accuracy |
| `helm_safety_v1.17.0` | 87 | **artifact** | degraded | pooled only | see below |

The `s_DIS_status` / `s_AMB_status` columns carry this per row. Filter on
`s_DIS_status == "ok"` before ranking anything by `B = |s_DIS| + |s_AMB|`.

## Traps

**1. HELM's `s_DIS` is not a measurement.** All 87 models land in −0.19…−0.24 regardless of
accuracy (0.42 to 0.99), which is the tell. The cause is in HELM's own scenario code:
`bbq_scenario.py` matches the bias target with `group in answer_info_i` (*list-element equality*,
not substring), with ad-hoc `any(g in e for e in ...)` patches for only `Black`, `African American`,
`Latino` and `lowSES`. So `stereotyped_groups=['Asian']` never matches
`answer_info={'ans2': ['Jenny Choi', 'F-Asian']}`.

Replaying that matcher over this repo's `third_party/bbq/data/`:

| category | items with no target found |
|---|---|
| Race_x_gender | 100.0% |
| Race_x_SES | 27.2% |
| Race_ethnicity | 23.3% |
| Gender_identity | 10.4% |
| the other 7 | 0% |

21,192 of 58,492 items (36.2%) get `target_letter = ""`. In HELM's `bbq_metrics.py` the disambiguated
branch (unlike the ambiguous one, which has a `no_non_targets_exist` guard) counts those items in
the denominator while they can never reach the numerator. For a perfect model that gives
`2 × (0.64×0.5 + 0.36×0.25) − 1 ≈ −0.18`, which is the band observed. Use HELM's `accuracy` column;
treat its `s_DIS` as a constant and its `s_AMB` as a lower bound.

**2. HELM scales `s_AMB` by `(1 − accuracy_overall)`, this repo uses `(1 − accuracy_ambig)`.**
Different quantity, same name.

**3. OpenAI changed harness between cards.** GPT-4o is `0.97 ambig / 0.72 disambig` in the o1 card
and `0.88 / 0.85` in the GPT-5 card. Both rows are kept, in different comparability groups. Compare
only within a group.

**4. Anthropic's own numbers drift between cards.** The Opus 4.8 card says prior-model results
"show variance from previous system cards due to routine evaluation updates". They are all in one
group here because Anthropic presents them as comparable, but that is their claim, not a verified one.

**5. Vendor accuracy is not this repo's accuracy.** None of these vendors publish their prompt
format or scoring path, and none of them use the logprob scoring in `src/bbq_eval/serve_logprobs.py`.
Cross-source comparison is indicative at best.

**6. Read `s_AMB` next to `accuracy_ambig`, always.** The usual trap applies here too: in an
ambiguous context the correct answer *is* "unknown", so a model that simply abstains scores near
zero bias. Claude Opus 4.8 has `s_AMB = 0.0007` at `accuracy_ambig = 0.999`, near-total abstention.
Its `accuracy_disambig` of 0.721 is what makes that score meaningful, and it only just clears the
`>= 0.7` eligibility bar used in `docs/leaderboard.md`.

## Checked and absent

Gemma 3 dropped BBQ. Meta publishes no BBQ in any Llama model card. Gemini tech reports have no BBQ
table. BIG-bench `bbq_lite_json` has raw per-model scores but only for BIG-G / GPT-3 / PaLM, and
scores `multiple_choice_grade` rather than a bias score. Details in `sources.json`.
