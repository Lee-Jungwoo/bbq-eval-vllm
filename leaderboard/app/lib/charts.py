"""Plotly figure builders shared by the pages."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme as t


def bias_dotplot(frame: pd.DataFrame) -> go.Figure:
    """B(m) with its bootstrap CI, models in rank order.

    The whiskers are the point of the chart: several adjacent ranks overlap
    completely, which is the honest reading of a 0.0003 gap.
    """
    # Plotly orders categories by first appearance across traces, which would put
    # the ineligible trace on top; pin the axis so rank 1 sits at the top and the
    # gated-out runs sit below the ranked field.
    ordered = pd.concat([frame[frame["eligible"] == 1].sort_values("B"),
                         frame[frame["eligible"] == 0].sort_values("B")])
    category_order = ordered["model_short"].tolist()[::-1]

    fig = go.Figure()
    for eligible, colour, label in ((1, t.BIASED, "ranked"),
                                    (0, t.UNKNOWN, "below the accuracy gate")):
        sub = ordered[ordered["eligible"] == eligible]
        if sub.empty:
            continue
        flagged = sub["abstention_tier"] != "none"
        fig.add_trace(go.Scatter(
            x=sub["B"], y=sub["model_short"],
            mode="markers", name=label, legendgroup=label,
            error_x=dict(type="data", symmetric=False,
                         array=(sub["B_hi"] - sub["B"]).fillna(0),
                         arrayminus=(sub["B"] - sub["B_lo"]).fillna(0),
                         color=colour, thickness=2, width=6),
            marker=dict(size=11, color=colour,
                        line=dict(width=np.where(flagged, 2.5, 1.5),
                                  color=np.where(flagged, t.INK, t.SURFACE))),
            customdata=np.stack([sub["B_lo"], sub["B_hi"], sub["s_DIS"],
                                 sub["s_AMB"], sub["s_AMB_unscaled"],
                                 sub["unknown_rate_ambig"], sub["n_elig_ambig"],
                                 sub["abstention_tier"]], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "B = %{x:.4f}  [%{customdata[0]:.4f}, %{customdata[1]:.4f}]<br>"
                "s_DIS = %{customdata[2]:+.4f} · s_AMB = %{customdata[3]:+.4f}<br>"
                "s_AMB unscaled = %{customdata[4]:+.4f}<br>"
                "abstained on %{customdata[5]:.1%} of ambiguous items "
                "(n_elig = %{customdata[6]:,.0f})<br>"
                "abstention flag: %{customdata[7]}<extra></extra>"),
        ))
    fig.add_vline(x=0, line=dict(color=t.AXIS, width=1, dash="dash"))
    return t.apply(fig, height=80 + 34 * len(ordered),
                   title="모델별 Bias (95% cluster-bootstrap CI)",
                   xaxis_title="<- 편향이 작음                        편향이 큼 ->",
                   yaxis_title=None,
                   yaxis=dict(showgrid=False, categoryorder="array",
                              categoryarray=category_order))


def separation_heatmap(pairs: pd.DataFrame, order: list[str]) -> go.Figure:
    """P(row model is less biased than column model) across bootstrap replicates.

    0.5 (grey) means the two are indistinguishable — the ordering between them is
    an artefact of the point estimate.
    """
    grid = pd.DataFrame(np.nan, index=order, columns=order, dtype=float)
    for row in pairs.itertuples(index=False):
        grid.loc[row.model_a, row.model_b] = row.p_a_less_biased
        grid.loc[row.model_b, row.model_a] = 1 - row.p_a_less_biased

    text = grid.map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    fig = go.Figure(go.Heatmap(
        z=grid.to_numpy(), x=order, y=order,
        colorscale=t.DIVERGING, zmin=0, zmax=1, zmid=0.5,
        xgap=2, ygap=2, text=text.to_numpy(), texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="P(row less<br>biased)", thickness=12, len=0.7),
        hovertemplate="P(%{y} less biased than %{x}) = %{z:.3f}<extra></extra>",
    ))
    return t.apply(fig, height=90 + 42 * len(order),
                   title="P(row model is less biased than column model)",
                   xaxis=dict(showgrid=False, tickangle=-40),
                   yaxis=dict(showgrid=False, autorange="reversed"),
                   margin=dict(l=8, r=8, t=48, b=120))


def category_bars(frame: pd.DataFrame, columns: tuple[str, str],
                  labels: tuple[str, str], title: str, *,
                  hline: float | None = None, hline_label: str = "",
                  xaxis_title: str = "") -> go.Figure:
    """Grouped horizontal bars over categories, ambiguous vs disambiguated."""
    frame = frame.iloc[::-1]
    fig = go.Figure()
    for column, label, colour in zip(columns, labels, (t.AMBIG, t.DISAMBIG)):
        fig.add_trace(go.Bar(
            y=frame["scope_value"], x=frame[column], name=label,
            orientation="h", marker=dict(color=colour, line=dict(width=0)),
            hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.4f}}<extra></extra>",
        ))
    if hline is not None:
        fig.add_vline(x=hline, line=dict(color=t.AXIS, width=1, dash="dash"),
                      annotation_text=hline_label, annotation_position="top",
                      annotation_font=dict(color=t.MUTED, size=11))
    return t.apply(fig, height=110 + 46 * len(frame), title=title,
                   barmode="group", bargap=0.28, bargroupgap=0.08,
                   xaxis_title=xaxis_title, yaxis=dict(showgrid=False))


def polarity_panels(cells: pd.DataFrame, column: str, title: str,
                    *, zeroline: bool = False) -> go.Figure:
    """One grouped bar per (context_condition, question_polarity) cell."""
    fig = go.Figure()
    for polarity in ("neg", "nonneg"):
        sub = cells[cells["question_polarity"] == polarity]
        fig.add_trace(go.Bar(
            x=sub["context_condition"], y=sub[column], name=polarity,
            marker=dict(color=t.POLARITY[polarity], line=dict(width=0)),
            customdata=np.stack([sub["n"], sub["n_elig"]], axis=-1),
            hovertemplate=(f"<b>%{{x}} / {polarity}</b><br>{column} = %{{y:.4f}}"
                           "<br>n = %{customdata[0]:,} "
                           "(bias denominator %{customdata[1]:,})<extra></extra>"),
        ))
    if zeroline:
        fig.add_hline(y=0, line=dict(color=t.AXIS, width=1))
    return t.apply(fig, height=340, title=title, barmode="group",
                   bargap=0.45, bargroupgap=0.06,
                   xaxis_title=None, yaxis_title=column)


def margin_bars(hist: pd.DataFrame) -> go.Figure:
    """Distribution of margin = top1 - top2 logprob, split by correctness."""
    fig = go.Figure()
    for correct, colour, label in ((1, t.OK, "correct"), (0, t.NG, "incorrect")):
        sub = hist[hist["correct"] == correct]
        fig.add_trace(go.Bar(
            x=sub["bin_lo"], y=sub["n"], name=label,
            marker=dict(color=colour, line=dict(width=0)),
            hovertemplate=(f"<b>{label}</b><br>margin %{{x:.1f}}–"
                           "%{x:.1f}+0.5 nats<br>%{y:,} items<extra></extra>"),
        ))
    return t.apply(fig, height=340, title="Margin distribution",
                   barmode="overlay", bargap=0.05,
                   xaxis_title="margin (nats)", yaxis_title="items")


def margin_ecdf(hist: pd.DataFrame) -> go.Figure:
    """ECDF built by cumsum of the same bins — no need to ship 31k floats."""
    fig = go.Figure()
    for correct, colour, label in ((1, t.OK, "correct"), (0, t.NG, "incorrect")):
        sub = hist[hist["correct"] == correct].sort_values("bin_lo")
        if sub.empty or sub["n"].sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub["bin_lo"], y=sub["n"].cumsum() / sub["n"].sum(),
            mode="lines", name=label, line=dict(color=colour, width=2),
            hovertemplate=(f"<b>{label}</b><br>margin <= %{{x:.1f}} nats"
                           "<br>%{y:.1%} of items<extra></extra>"),
        ))
    fig.add_hline(y=0.5, line=dict(color=t.AXIS, width=1, dash="dash"),
                  annotation_text="median", annotation_position="right",
                  annotation_font=dict(color=t.MUTED, size=11))
    return t.apply(fig, height=340, title="Margin ECDF",
                   xaxis_title="margin (nats)", yaxis_title="cumulative share",
                   yaxis=dict(range=[0, 1]))


def abstention_composition(frame: pd.DataFrame) -> go.Figure:
    """How every ambiguous item was answered: abstained, counter-stereotype, or
    stereotype. The grey block is what `s_AMB`'s scaling term is made of."""
    frame = frame.iloc[::-1]
    n_ambig = frame["n_ambig"].to_numpy()
    series = (("n_unknown_ambig", "abstained (unknown)", t.UNKNOWN),
              ("n_counter", "counter-stereotype", t.COUNTER),
              ("n_biased_ambig", "stereotype", t.BIASED))
    fig = go.Figure()
    for column, label, colour in series:
        share = frame[column].to_numpy() / np.where(n_ambig > 0, n_ambig, np.nan)
        fig.add_trace(go.Bar(
            y=frame["scope_value"], x=share, name=label, orientation="h",
            marker=dict(color=colour, line=dict(color=t.SURFACE, width=2)),
            customdata=frame[column],
            hovertemplate=(f"<b>%{{y}}</b><br>{label}: %{{x:.1%}} "
                           "(%{customdata:,} items)<extra></extra>"),
        ))
    return t.apply(fig, height=110 + 46 * len(frame),
                   title="How ambiguous items were answered",
                   barmode="stack", bargap=0.3,
                   xaxis_title="share of ambiguous items",
                   xaxis=dict(tickformat=".0%", range=[0, 1]),
                   yaxis=dict(showgrid=False))


def s_answered_by_category(frame: pd.DataFrame, min_n: int) -> go.Figure:
    """`s_answered` with Wilson intervals; hollow markers where the cell is too
    small to read."""
    frame = frame.iloc[::-1]
    small = frame["n_answered"] < min_n
    fig = go.Figure(go.Scatter(
        x=frame["s_answered"], y=frame["scope_value"], mode="markers",
        name="s_answered",
        error_x=dict(type="data", symmetric=False,
                     array=(frame["s_hi"] - frame["s_answered"]).fillna(0),
                     arrayminus=(frame["s_answered"] - frame["s_lo"]).fillna(0),
                     color=t.BIASED, thickness=2, width=6),
        marker=dict(size=11, color=np.where(small, t.SURFACE, t.BIASED),
                    line=dict(width=2, color=t.BIASED)),
        customdata=np.stack([frame["n_answered"], frame["s_lo"], frame["s_hi"]],
                            axis=-1),
        hovertemplate=("<b>%{y}</b><br>s_answered = %{x:+.3f} "
                       "[%{customdata[1]:+.3f}, %{customdata[2]:+.3f}]<br>"
                       "n answered = %{customdata[0]:,.0f}<extra></extra>"),
    ))
    fig.add_vline(x=0, line=dict(color=t.AXIS, width=1),
                  annotation_text="unbiased", annotation_position="top",
                  annotation_font=dict(color=t.MUTED, size=11))
    return t.apply(fig, height=110 + 46 * len(frame),
                   title="Bias of the answers actually given",
                   xaxis_title="s_answered  (+ = toward the stereotype)",
                   yaxis=dict(showgrid=False))


def scaling_curve(frame: pd.DataFrame, columns: tuple[str, ...],
                  labels: tuple[str, ...], colours: tuple[str, ...],
                  title: str, yaxis_title: str, *,
                  hline: float | None = None, hline_label: str = "") -> go.Figure:
    """Metric vs parameter count on a log x-axis, one line per series."""
    fig = go.Figure()
    for column, label, colour in zip(columns, labels, colours):
        fig.add_trace(go.Scatter(
            x=frame["params_b"], y=frame[column], mode="lines+markers", name=label,
            line=dict(color=colour, width=2),
            marker=dict(size=9, color=colour,
                        line=dict(width=2, color=t.SURFACE)),
            text=frame["model_short"],
            hovertemplate=(f"<b>%{{text}}</b><br>{label}: %{{y:.4f}}"
                           "<br>%{x:g}B params<extra></extra>"),
        ))
    if hline is not None:
        fig.add_hline(y=hline, line=dict(color=t.AXIS, width=1, dash="dash"),
                      annotation_text=hline_label, annotation_position="right",
                      annotation_font=dict(color=t.MUTED, size=11))
    return t.apply(fig, height=380, title=title,
                   xaxis_title="parameters (B, log scale)",
                   yaxis_title=yaxis_title,
                   xaxis=dict(type="log", tickmode="array",
                              tickvals=frame["params_b"].tolist(),
                              ticktext=[f"{v:g}B" for v in frame["params_b"]]))


def scaling_response_composition(frame: pd.DataFrame, condition: str) -> go.Figure:
    """Response mix by parameter count for one BBQ context condition."""
    counts = {
        "ambiguous": ("n_ambig", "n_unknown_ambig", "n_elig_ambig",
                      "n_biased_ambig", "ambiguous"),
        "disambiguated": ("n_disambig", "n_unknown_disambig", "n_elig_disambig",
                            "n_biased_disambig", "disambiguated"),
    }
    try:
        total_col, unknown_col, eligible_col, biased_col, label = counts[condition]
    except KeyError as exc:
        raise ValueError(f"Unknown context condition: {condition}") from exc

    total = frame[total_col].to_numpy()
    response_counts = (
        (unknown_col, "unknown", t.UNKNOWN),
        (None, "anti-biased", t.COUNTER),
        (biased_col, "biased", t.BIASED),
    )
    parameter_labels = [f"{value:g}B" for value in frame["params_b"]]
    fig = go.Figure()
    for column, series_label, colour in response_counts:
        values = (frame[eligible_col] - frame[biased_col]).to_numpy() if column is None \
            else frame[column].to_numpy()
        share = values / np.where(total > 0, total, np.nan)
        fig.add_trace(go.Bar(
            x=parameter_labels, y=share, name=series_label,
            marker=dict(color=colour, line=dict(color=t.SURFACE, width=1)),
            customdata=values,
            hovertemplate=(f"<b>{series_label}</b><br>{series_label}: %{{y:.1%}} "
                           "(%{customdata:,} items)<br>%{x} parameters<extra></extra>"),
        ))
    return t.apply(fig, height=380,
                   title=f"{label.capitalize()} responses by parameter scale",
                   barmode="stack", bargap=0.22,
                   xaxis_title="parameters (B)", yaxis_title="share of items",
                   xaxis=dict(type="category", categoryorder="array",
                              categoryarray=parameter_labels),
                   yaxis=dict(range=[0, 1], tickformat=".0%"))


def scaling_ambiguous_non_unknown(frame: pd.DataFrame) -> go.Figure:
    """Biased and anti-biased mix among ambiguous non-unknown responses."""
    total = frame["n_elig_ambig"].to_numpy()
    response_counts = (
        (frame["n_elig_ambig"] - frame["n_biased_ambig"], "anti-biased", t.COUNTER),
        (frame["n_biased_ambig"], "biased", t.BIASED),
    )
    parameter_labels = [f"{value:g}B" for value in frame["params_b"]]
    fig = go.Figure()
    for counts, label, colour in response_counts:
        share = counts.to_numpy() / np.where(total > 0, total, np.nan)
        fig.add_trace(go.Bar(
            x=parameter_labels, y=share, name=label,
            marker=dict(color=colour, line=dict(color=t.SURFACE, width=1)),
            customdata=counts,
            hovertemplate=(f"<b>{label}</b><br>{label}: %{{y:.1%}} "
                           "(%{customdata:,} items)<br>%{x} parameters<extra></extra>"),
        ))
    return t.apply(fig, height=280,
                   title="Biased vs anti-biased ambiguous responses",
                   barmode="stack", bargap=0.22,
                   xaxis_title="parameters (B)", yaxis_title="share of non-unknown responses",
                   xaxis=dict(type="category", categoryorder="array",
                              categoryarray=parameter_labels),
                   yaxis=dict(range=[0, 1], tickformat=".0%"))
