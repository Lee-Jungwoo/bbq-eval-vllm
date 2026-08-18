"""Read the binary arm's runs — `bbq_pipeline/runs_binary/*/{meta,summary}.json`.

The A/B arm (`bbq_pipeline/pipeline_binary.py`) asks every BBQ item as a forced
choice between the two group answers, with the "unknown" option removed in both
context conditions. Kept strictly apart from `db.py`:

  - it never touches `bbq.sqlite`; `publish.py` refuses binary runs on purpose,
    because a 3-option and a 2-option score do not belong in one ranking;
  - the numbers are read from each run's `summary.json` — written by
    `aggregate_binary.py`, which is the only implementation of the binary
    metrics. Nothing is recomputed here, so the page cannot drift from the
    aggregate.

What the missing unknown option removes from the board:

  `accuracy_ambig`      no correct answer exists in an ambiguous context once
                        "unknown" is gone, so the column is not rendered at all
                        (`aggregate_binary.py` reports it as null).
  abstention diagnostics  `s_AMB_unscaled`, `abstained (amb)`, `s_answered`,
                        `n_elig amb` and the abstention flag all describe how
                        often a model took the unknown option. There is no such
                        option here, so they are gone; `s_AMB` is already the
                        unscaled quantity.

In their place the board shows what a two-way forced choice makes measurable:
`target_rate` (per condition and polarity) and `letter_A_rate`.
"""
from __future__ import annotations

import json

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bbq_metrics as bm

from . import table
from . import theme as t

RUNS_DIR = Path(__file__).resolve().parents[3] / "bbq_pipeline" / "runs_binary"

#: The eligibility gate from `../README.md`, the same 0.70 `publish.py` applies:
#: a bias score is only readable when the model can answer the disambiguated
#: questions at all. It reads the same way here — `accuracy_disambig` keeps its
#: meaning in this arm, since the gold answer there is one of the two groups.
GATE = 0.70

#: Column set for `table.render`. `acc_amb` is absent by construction — see the
#: module docstring. Widths sum with the 58px rank column the way `table.COLUMNS`
#: does; the 6% freed by dropping `acc_amb` goes to the two bias bars.
COLUMNS: tuple[tuple[str, str, bool, str, str, str], ...] = (
    ("#",         "rank",              True,  "58px", "",      "rank"),
    ("Model",     "model_short",       True,  "28%",  "",      "model"),
    ("Bias B(m)", "B",                 True,  "9%",   "",      "bias"),
    ("95% CI",    "B_lo",              True,  "13%",  "",      "ci"),
    ("s_DIS",     "s_DIS",             False, "22%",  "",      "diverge"),
    ("s_AMB",     "s_AMB",             False, "22%",  "",      "diverge"),
    ("acc_dis",   "accuracy_disambig", False, "6%",   "right", "acc_gate"),
)

SORT_OPTIONS: dict[str, tuple[str, bool]] = {
    label: (column, ascending) for label, column, ascending, _, _, _ in COLUMNS}

#: Flat metric columns lifted out of `summary.json`'s `overall` block.
_METRICS = ("B", "B_lo", "B_hi", "s_DIS", "s_AMB", "accuracy_disambig",
            "target_rate_ambig", "target_rate_disambig", "letter_A_rate",
            "letter_A_rate_ambig", "letter_A_rate_disambig",
            "n", "n_ambig", "n_disambig")


def _signature() -> tuple:
    """Cache key: every summary the directory holds, with its mtime."""
    if not RUNS_DIR.exists():
        return ()
    return tuple(sorted((str(p), p.stat().st_mtime)
                        for p in RUNS_DIR.glob("*/summary.json")))


@st.cache_data(show_spinner=False)
def _load(signature: tuple) -> tuple[pd.DataFrame, dict]:
    """`(runs frame, {run_id: per-category frame})`."""
    rows: list[dict] = []
    per_category: dict[str, pd.DataFrame] = {}
    for summary_path in sorted(RUNS_DIR.glob("*/summary.json")):
        run_dir = summary_path.parent
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("arm") != "binary":  # never mix a 3-option run in
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        block = summary.get(meta["run_id"]) or next(iter(summary.values()), {})
        overall = block.get("overall", {})

        model = meta["model"]
        family, params_b = bm.family_and_size(model)
        row = {
            "run_id": meta["run_id"],
            "model": model,
            "model_short": model.split("/")[-1],
            "family": family,
            "params_b": params_b,
            "dtype": meta.get("dtype"),
            "model_revision": meta.get("model_revision"),
            "comparability_key": bm.comparability_key(meta),
            "n_instances": meta.get("n_instances"),
            "n_categories": len(meta.get("categories") or []),
            "mean_prob_mass_captured": block.get("mean_prob_mass_captured"),
            "bootstrap_R": (block.get("bootstrap") or {}).get("replicates"),
            "n_clusters": (block.get("bootstrap") or {}).get("n_clusters"),
        }
        row.update({column: overall.get(column) for column in _METRICS})
        by_polarity = overall.get("target_rate_by_polarity") or {}
        for condition in ("ambig", "disambig"):
            for polarity in ("neg", "nonneg"):
                row[f"target_rate_{condition}_{polarity}"] = \
                    (by_polarity.get(condition) or {}).get(polarity)
        rows.append(row)

        categories = block.get("per_category") or {}
        if categories:
            per_category[meta["run_id"]] = pd.DataFrame([
                {"scope_value": category,
                 **{k: v for k, v in values.items() if not isinstance(v, dict)}}
                for category, values in sorted(categories.items())
            ])

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("model_short").reset_index(drop=True)
    return frame, per_category


def require_runs() -> None:
    """Render a how-to instead of a traceback when the arm has no runs yet."""
    frame, _ = _load(_signature())
    if not frame.empty:
        return
    st.error("아직 binary arm 실행 결과가 없습니다.")
    st.markdown(
        "이 페이지는 `bbq_pipeline/runs_binary/<run_id>/summary.json` 을 읽습니다. "
        "3-옵션 리더보드(**Home**)와 달리 `bbq.sqlite` 를 쓰지 않습니다. "
        "한 모델을 돌리는 전체 절차:")
    st.code(
        "cd bbq_pipeline\n"
        "python pipeline_binary.py prepare --config config_binary.json\n"
        "# runs_binary/<run_id>/requests.jsonl 을 서버로 보내 serve_logprobs.py 실행\n"
        "python pipeline_binary.py collect\n"
        "python aggregate_binary.py runs_binary/<run_id>/records.jsonl \\\n"
        "    --out runs_binary/<run_id>/summary.json\n"
        "# 여러 모델을 한 번에: ./run_ladder_binary.sh <model> ...",
        language="bash")
    st.stop()


def runs() -> pd.DataFrame:
    return _load(_signature())[0]


def per_category(run_id: str) -> pd.DataFrame:
    return _load(_signature())[1].get(run_id, pd.DataFrame())


def groups() -> pd.DataFrame:
    """Comparability groups, largest first — same key as the 3-option board."""
    frame = runs()
    if frame.empty:
        return pd.DataFrame(columns=["comparability_key", "n_runs", "n_instances"])
    grouped = (frame.groupby("comparability_key")
               .agg(n_runs=("run_id", "size"), n_instances=("n_instances", "min"))
               .reset_index()
               .sort_values(["n_runs", "comparability_key"], ascending=[False, True]))
    return grouped


def board(key: str) -> pd.DataFrame:
    """One comparability group, ranked by B ascending among eligible runs."""
    frame = runs()
    frame = frame[frame["comparability_key"] == key].copy()
    accuracy = pd.to_numeric(frame["accuracy_disambig"], errors="coerce")
    frame["eligible"] = (accuracy >= GATE).fillna(False).astype(int)
    frame["eligibility_threshold"] = GATE
    frame = frame.sort_values(["eligible", "B"], ascending=[False, True])
    n_eligible = int((frame["eligible"] == 1).sum())
    frame["rank"] = pd.Series(
        list(range(1, n_eligible + 1)) + [pd.NA] * (len(frame) - n_eligible),
        index=frame.index, dtype="Float64")
    return frame


def render(frame: pd.DataFrame) -> str:
    return table.render(frame, threshold=GATE, columns=COLUMNS)


def bias_dotplot(frame: pd.DataFrame) -> go.Figure:
    """B(m) with its cluster-bootstrap CI, in rank order.

    A separate function from `charts.bias_dotplot` rather than a flag on it: that
    one's hover text is built out of the abstention diagnostics, none of which
    exist in this arm.
    """
    ordered = pd.concat([frame[frame["eligible"] == 1].sort_values("B"),
                        frame[frame["eligible"] == 0].sort_values("B")])
    category_order = ordered["model_short"].tolist()[::-1]

    fig = go.Figure()
    for eligible, colour, label in ((1, t.BIASED, "ranked"),
                                    (0, t.UNKNOWN, "below the accuracy gate")):
        sub = ordered[ordered["eligible"] == eligible]
        if sub.empty:
            continue
        hi = pd.to_numeric(sub["B_hi"], errors="coerce")
        lo = pd.to_numeric(sub["B_lo"], errors="coerce")
        b = pd.to_numeric(sub["B"], errors="coerce")
        fig.add_trace(go.Scatter(
            x=sub["B"], y=sub["model_short"],
            mode="markers", name=label, legendgroup=label,
            error_x=dict(type="data", symmetric=False,
                         array=(hi - b).fillna(0), arrayminus=(b - lo).fillna(0),
                         color=colour, thickness=2, width=6),
            marker=dict(size=11, color=colour,
                        line=dict(width=1.5, color=t.SURFACE)),
            customdata=np.stack([lo, hi, sub["s_DIS"], sub["s_AMB"],
                                 sub["target_rate_ambig"],
                                 sub["letter_A_rate"]], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "B = %{x:.4f}  [%{customdata[0]:.4f}, %{customdata[1]:.4f}]<br>"
                "s_DIS = %{customdata[2]:+.4f} · s_AMB = %{customdata[3]:+.4f} "
                "(unscaled)<br>"
                "고정관념 집단 지목률(모호) = %{customdata[4]:.3f}<br>"
                "A 선택률 = %{customdata[5]:.3f}<extra></extra>"),
        ))
    fig.add_vline(x=0, line=dict(color=t.AXIS, width=1, dash="dash"))
    return t.apply(fig, height=80 + 34 * len(ordered),
                   title="모델별 Bias — binary A/B (95% cluster-bootstrap CI)",
                   xaxis_title="<- 편향이 작음                        편향이 큼 ->",
                   yaxis_title=None,
                   yaxis=dict(showgrid=False, categoryorder="array",
                              categoryarray=category_order))


__all__ = ["COLUMNS", "GATE", "SORT_OPTIONS", "bias_dotplot", "board", "groups",
           "per_category", "render", "require_runs", "runs"]
