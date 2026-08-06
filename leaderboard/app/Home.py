"""BBQ bias leaderboard — ranking page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import charts, db, logos
from lib import theme as t

st.set_page_config(page_title="BBQ bias leaderboard", page_icon="⚖️",
                   layout="wide")
db.require_db()

st.title("BBQ bias leaderboard")

groups = db.comparability_groups()
labels = {row.comparability_key: f"{row.comparability_key} · {row.n_runs} runs · "
                                f"{row.n_instances:,} items"
          for row in groups.itertuples(index=False)}

with st.sidebar:
    st.header("View")
    # Runs are only comparable when they answered the same questions under the
    # same prompt; ranking across groups would be meaningless.
    key = st.selectbox("Comparability group", list(labels),
                       format_func=lambda k: labels[k],
                       help="sha256(prompt_hash | categories | n_instances). "
                            "Only runs sharing a key are ranked against each other.")
    show_ineligible = st.checkbox("Show runs below the accuracy gate", value=True)
    families = st.multiselect("Model family", sorted(db.runs()["family"].unique()),
                              default=None, placeholder="all families")
    sort_by = st.selectbox(
        "Sort by", ["B(m) — official rank", "accuracy_disambig",
                    "s_AMB_unscaled", "unknown_rate_ambig", "s_answered"])
db.sidebar_footer()

board = db.leaderboard(key)
if families:
    board = board[board["family"].isin(families)]
if not show_ineligible:
    board = board[board["eligible"] == 1]
if board.empty:
    st.warning("No runs match the current filters.")
    st.stop()

threshold = float(board["eligibility_threshold"].iloc[0])
n_ranked = int((board["eligible"] == 1).sum())

top = st.columns(5)
top[0].metric("Models ranked", n_ranked,
              delta=f"{len(board) - n_ranked} gated out", delta_color="off")
top[1].metric("Accuracy gate", f"{threshold:.2f}", delta="acc_dis ≥", delta_color="off")
top[2].metric("Items per run", f"{int(board['n_ambig'].iloc[0]) * 2:,}")
top[3].metric("Prompt hash",
              db.provenance(board["run_id"].iloc[0]).get("prompt_hash", "?")[:8])
info = db.publish_info()
top[4].metric("Bootstrap", f"R = {info.get('bootstrap_R', '?')}",
              delta=f"seed {info.get('bootstrap_seed', '?')}", delta_color="off")

# --- the caveat, computed live rather than written down --------------------
ranked = board[board["eligible"] == 1].sort_values("rank")
if not ranked.empty:
    first = ranked.iloc[0]
    rank_by_unscaled = (ranked["s_AMB_unscaled"].abs()
                        .rank(ascending=False).loc[first.name])
    st.warning(
        f"**s_AMB rewards abstention.** `s_AMB = (1 − accuracy_ambig) × s_AMB_unscaled`, "
        f"and in an ambiguous context the correct answer *is* “unknown” — so "
        f"`accuracy_ambig` is essentially the abstention rate and the scaling term "
        f"collapses toward zero for any model that declines to answer.\n\n"
        f"**{first['model_short']}** ranks #1 at B = {first['B']:.4f}, but it abstained on "
        f"**{first['unknown_rate_ambig']:.1%}** of ambiguous items — only "
        f"**{int(first['n_elig_ambig']):,}** of {int(first['n_ambig']):,} reach the bias "
        f"denominator. Among the answers it did give, `s_AMB_unscaled = "
        f"{first['s_AMB_unscaled']:+.3f}`, the **#{int(rank_by_unscaled)} most stereotyped** "
        f"of the {n_ranked} ranked models. Read the diagnostic columns to the right of "
        f"B before reading the rank.")

# --- the table -------------------------------------------------------------
sort_column = {"B(m) — official rank": "B", "accuracy_disambig": "accuracy_disambig",
               "s_AMB_unscaled": "s_AMB_unscaled",
               "unknown_rate_ambig": "unknown_rate_ambig",
               "s_answered": "s_answered"}[sort_by]
board = board.sort_values(["eligible", sort_column],
                          ascending=[False, sort_column == "B"])

view = pd.DataFrame({
    "rank": board["rank"],
    # Empty for every family whose artwork is not installed — see
    # app/assets/logos/README.md. The model name carries the identity either way.
    "": logos.column(board["family"]),
    "model": board["model_short"],
    "B(m)": board["B"],
    "95% CI": [t.fmt_interval(lo, hi) for lo, hi
               in zip(board["B_lo"], board["B_hi"])],
    "s_DIS": board["s_DIS"],
    "s_AMB": board["s_AMB"],
    "acc_dis": board["accuracy_disambig"],
    "acc_amb": board["accuracy_ambig"],
    "◆ s_AMB unscaled": board["s_AMB_unscaled"],
    "◆ abstained (amb)": board["unknown_rate_ambig"],
    "◆ n_elig amb": board["n_elig_ambig"],
    "◆ s_answered": board["s_answered"],
    "◆ flag": board["abstention_tier"].map(t.TIER_LABEL),
})

st.dataframe(
    view, hide_index=True, width="stretch",
    column_config={
        "rank": st.column_config.NumberColumn(width="small", format="%d"),
        "": st.column_config.ImageColumn(width="small", help="Model family"),
        "B(m)": st.column_config.ProgressColumn(
            "B(m) ↓", help="|s_DIS| + |s_AMB|. Lower is less biased.",
            format="%.4f", min_value=0.0, max_value=float(board["B"].max())),
        "s_DIS": st.column_config.NumberColumn(format="%+.4f"),
        "s_AMB": st.column_config.NumberColumn(format="%+.4f"),
        "acc_dis": st.column_config.NumberColumn(format="%.3f"),
        "acc_amb": st.column_config.NumberColumn(format="%.3f"),
        "◆ s_AMB unscaled": st.column_config.NumberColumn(
            format="%+.3f",
            help="s_AMB without the (1 - accuracy_ambig) factor: which way the "
                 "model leans when it does commit to a group."),
        "◆ abstained (amb)": st.column_config.NumberColumn(
            format="percent",
            help="Share of ambiguous items answered 'unknown'. This is what "
                 "damps s_AMB toward zero."),
        "◆ n_elig amb": st.column_config.NumberColumn(
            format="%d", help="Items in the ambiguous bias-score denominator. "
                              "A small value makes s_AMB unstable."),
        "◆ s_answered": st.column_config.NumberColumn(
            format="%+.3f", help="Bias of the ambiguous answers actually given — "
                                 "s_AMB with the abstention scaling removed."),
    })
st.caption("◆ diagnostics — these do not affect the rank. "
           "`flag` marks a model whose ambiguous bias score rests on very few "
           "answers (caution: ≥75% abstention or <10% of items eligible; "
           "severe: ≥90% or <3%).")

gaps = logos.missing()
if gaps:
    st.caption(f"No logo installed for {', '.join(sorted(gaps))} — drop the "
               f"artwork into `app/assets/logos/` (see the README there).")

st.plotly_chart(charts.bias_dotplot(board), width="stretch")

# --- separability ----------------------------------------------------------
pairs = db.pairs(key)
if not pairs.empty:
    order = ranked["model_short"].tolist()
    pairs = pairs[pairs["model_a"].isin(order) & pairs["model_b"].isin(order)]
    n_sep = int(pairs["separated"].sum())
    with st.expander(f"Is the ranking separable?  "
                     f"{n_sep} of {len(pairs)} pairs separated at 95%", expanded=False):
        st.markdown(
            "Each replicate resamples whole 4-item template clusters, and the same "
            "resampling is applied to every model, so the difference between two "
            "models is paired and shared item difficulty cancels out.")
        st.plotly_chart(charts.separation_heatmap(pairs, order),
                        width="stretch")
        unresolved = pairs[pairs["separated"] == 0]
        if not unresolved.empty:
            st.markdown("**Pairs the data cannot order:**")
            st.dataframe(
                pd.DataFrame({
                    "model A": unresolved["model_a"],
                    "model B": unresolved["model_b"],
                    "ΔB": unresolved["delta_B"],
                    "ΔB 95% CI": [f"[{lo:+.4f}, {hi:+.4f}]" for lo, hi
                                  in zip(unresolved["delta_lo"], unresolved["delta_hi"])],
                    "P(A less biased)": unresolved["p_a_less_biased"],
                }), hide_index=True, width="stretch",
                column_config={
                    "ΔB": st.column_config.NumberColumn(format="%+.4f"),
                    "P(A less biased)": st.column_config.NumberColumn(format="%.3f"),
                })
