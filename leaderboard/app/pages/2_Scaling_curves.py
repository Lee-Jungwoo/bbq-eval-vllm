"""Accuracy and bias against model size, within a family.

This is the view the three near-identical `pipeline*.ipynb` notebooks were built
to produce, one family at a time.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import charts, db
from lib import theme as t

st.set_page_config(page_title="Scaling · BBQ", page_icon="📈", layout="wide")
db.require_db()

runs = db.runs()
families = sorted(runs.loc[runs["params_b"].notna(), "family"].unique())
if not families:
    st.warning("No runs with a parsable parameter count.")
    st.stop()

with st.sidebar:
    st.header("Family")
    family = st.selectbox("Model family", families)
db.sidebar_footer()

frame = db.q("""
    SELECT r.model_short, r.params_b, r.run_id, m.*
    FROM runs r JOIN run_metrics m ON m.run_id = r.run_id
    WHERE m.scope = 'overall' AND r.family = ? AND r.params_b IS NOT NULL
    ORDER BY r.params_b
""", (family,))
if len(frame) < 2:
    st.info(f"{family} has fewer than two runs — nothing to plot against size.")
    st.stop()

st.title(f"{family} — accuracy and bias by model size")
st.caption("Accuracy includes unknown predictions. A positive bias score points toward the stereotype.")

left, right = st.columns(2)
with left:
    st.plotly_chart(charts.scaling_curve(
        frame, ("accuracy_ambig", "accuracy_disambig"),
        ("ambiguous", "disambiguated"), (t.AMBIG, t.DISAMBIG),
        "Accuracy", "accuracy",
        hline=1 / 3, hline_label="chance"), width="stretch")
with right:
    st.plotly_chart(charts.scaling_curve(
        frame, ("s_AMB", "s_DIS"), ("s_AMB", "s_DIS"), (t.AMBIG, t.DISAMBIG),
        "Bias score", "bias score",
        hline=0.0, hline_label="unbiased"), width="stretch")

st.plotly_chart(charts.scaling_curve(
    frame, ("s_AMB", "s_answered", "unknown_rate_ambig"),
    ("s_AMB (scaled)", "s_answered (unscaled)", "abstention rate"),
    (t.AMBIG, t.BIASED, t.UNKNOWN),
    "Why s_AMB falls with scale", "value"), width="stretch")

st.info(
    "The two lines pull in opposite directions. `s_AMB` carries a "
    "`(1 − accuracy_ambig)` factor, and `accuracy_ambig` on ambiguous items is "
    "essentially the abstention rate — so as a family scales up and abstains more, "
    "`s_AMB` is damped toward zero even while `s_answered`, the direction the model "
    "leans when it *does* name a group, rises.")

st.dataframe(
    frame[["model_short", "params_b", "accuracy_ambig", "accuracy_disambig",
           "s_AMB", "s_AMB_unscaled", "s_answered", "s_DIS",
           "unknown_rate_ambig", "n_elig_ambig"]]
    .rename(columns={"model_short": "model", "params_b": "params (B)"}),
    hide_index=True, width="stretch")
