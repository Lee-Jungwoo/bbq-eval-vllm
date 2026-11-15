"""Chart theme. Colours come from `bbq_metrics` so the app, the notebooks and the
published figures can never drift apart.

Encoding rules held constant across every figure here:
  - colour follows the entity (ambiguous / disambiguated / abstained), never rank;
  - a legend is present whenever two or more series are drawn;
  - one y-axis per panel — two measures of different scale get two panels;
  - the diverging scale used for pairwise separation has a neutral midpoint.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from bbq_eval import bbq_metrics as bm

AMBIG = bm.AMBIG
DISAMBIG = bm.DISAMBIG
UNKNOWN = bm.UNKNOWN
BIASED = bm.BIASED
COUNTER = bm.COUNTER
OK = bm.OK
NG = bm.NG
POLARITY = bm.POLARITY
GRID = bm.GRID
AXIS = bm.AXIS
MUTED = bm.MUTED
INK = bm.INK

SURFACE = "#ffffff"

#: Re-exported so pages never need to import bbq_metrics themselves.
MIN_N_ANSWERED = bm.MIN_N_ANSWERED

#: Polarity of a comparison. Keeps the meaning colour carries everywhere else in
#: the app: orange = more stereotyped, blue = less. So a high P(row less biased)
#: reads blue, a low one orange, and grey marks the 0.5 midpoint where the
#: bootstrap cannot separate the two models at all.
DIVERGING = [[0.0, BIASED], [0.5, "#eef0f2"], [1.0, COUNTER]]

#: BBQ's nine social dimensions (plus the two intersectional ones) in the words
#: a Korean audience reads. Display only — the database keys stay as they are.
CATEGORY_KO = {
    "Age": "나이",
    "Disability_status": "장애",
    "Gender_identity": "성별",
    "Nationality": "국적",
    "Physical_appearance": "외모",
    "Race_ethnicity": "인종·민족",
    "Religion": "종교",
    "SES": "사회경제적 지위",
    "Sexual_orientation": "성적 지향",
    "Race_x_gender": "인종 × 성별",
    "Race_x_SES": "인종 × 사회경제적 지위",
}


def category_ko(name: str) -> str:
    return CATEGORY_KO.get(name, name)


TIER_LABEL = {
    "none": "—",
    "caution": "주의",
    "severe": "심각",
}

BBQ_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=INK, size=13),
        title=dict(font=dict(size=15), x=0, xanchor="left", y=0.97,
                   yanchor="top"),
        margin=dict(l=8, r=8, t=72, b=8),
        colorway=[AMBIG, DISAMBIG, POLARITY["nonneg"], OK, UNKNOWN],
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=AXIS,
                        font=dict(color=INK, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, title_text=""),
        xaxis=dict(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS,
                   ticks="outside", tickcolor=AXIS, tickfont=dict(color=MUTED),
                   title=dict(font=dict(color=MUTED, size=12))),
        yaxis=dict(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS,
                   ticks="outside", tickcolor=AXIS, tickfont=dict(color=MUTED),
                   title=dict(font=dict(color=MUTED, size=12))),
    )
)
pio.templates["bbq"] = BBQ_TEMPLATE


def apply(fig: go.Figure, *, height: int | None = None, **layout) -> go.Figure:
    fig.update_layout(template="bbq", height=height, **layout)
    return fig


def fmt_interval(lo, hi, digits: int = 3) -> str:
    """'[0.011, 0.056]' — used where the point estimate has its own column."""
    if lo is None or lo != lo or hi is None or hi != hi:
        return "—"
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def fmt_ci(value, lo, hi, digits: int = 4) -> str:
    """'0.0982 [0.074, 0.122]' — the point estimate is meaningless without it."""
    if value is None or value != value:
        return "—"
    if lo is None or lo != lo or hi is None or hi != hi:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} [{lo:.3f}, {hi:.3f}]"
