"""BBQ bias leaderboard — ranking page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import charts, db, table
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
    sort_by = st.selectbox("Sort by", list(table.SORT_OPTIONS))
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

# top = st.columns(5)
# top[0].metric("Models ranked", n_ranked,
#               delta=f"{len(board) - n_ranked} gated out", delta_color="off")
# top[1].metric("Accuracy gate", f"{threshold:.2f}", delta="acc_dis ≥", delta_color="off")
# top[2].metric("Items per run", f"{int(board['n_ambig'].iloc[0]) * 2:,}")
# top[3].metric("Prompt hash",
#               db.provenance(board["run_id"].iloc[0]).get("prompt_hash", "?")[:8])
# info = db.publish_info()
# top[4].metric("Bootstrap", f"R = {info.get('bootstrap_R', '?')}",
#               delta=f"seed {info.get('bootstrap_seed', '?')}", delta_color="off")

# --- the caveat, computed live rather than written down --------------------
ranked = board[board["eligible"] == 1].sort_values("rank")
if not ranked.empty:
    first = ranked.iloc[0]
    rank_by_unscaled = (ranked["s_AMB_unscaled"].abs()
                        .rank(ascending=False).loc[first.name]) # type: ignore
#     st.warning(
#         f"**s_AMB rewards abstention.** `s_AMB = (1 − accuracy_ambig) × s_AMB_unscaled`, "
#         f"and in an ambiguous context the correct answer *is* “unknown” — so "
#         f"`accuracy_ambig` is essentially the abstention rate and the scaling term "
#         f"collapses toward zero for any model that declines to answer.\n\n"
#         f"**{first['model_short']}** ranks #1 at B = {first['B']:.4f}, but it abstained on "
#         f"**{first['unknown_rate_ambig']:.1%}** of ambiguous items — only "
#         f"**{int(first['n_elig_ambig']):,}** of {int(first['n_ambig']):,} reach the bias "
#         f"denominator. Among the answers it did give, `s_AMB_unscaled = "
#         f"{first['s_AMB_unscaled']:+.3f}`, the **#{int(rank_by_unscaled)} most stereotyped** "
#         f"of the {n_ranked} ranked models. Read the diagnostic columns to the right of "
#         f"B before reading the rank.")

# --- the table -------------------------------------------------------------
# Gated runs always sink below the ranked field, whatever the sort key.
sort_column, ascending = table.SORT_OPTIONS[sort_by]
board = board.sort_values(["eligible", sort_column],
                          ascending=[False, ascending])

st.html(table.render(board, threshold=threshold))
st.caption("Bias B(m)은 낮을수록 편향이 작습니다. "
           "막대 색: 주황 = 고정관념 방향, 파랑 = 반(反)고정관념 방향. "
           "정렬 기준은 왼쪽 사이드바 **Sort by**에서 표의 열 이름으로 고를 수 있습니다.")

# The rank is built from the ◆ diagnostics but does not show them; keep them one
# click away rather than in the headline table.
with st.expander("전체 수치 (◆ 진단 열 포함)", expanded=False):
    view = pd.DataFrame({
        "rank": board["rank"],
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
            "rank": st.column_config.NumberColumn(
                width="small", format="%d", alignment="left"),
            "B(m)": st.column_config.NumberColumn(
                "Bias (낮을수록 좋음)", help="|s_DIS| + |s_AMB|. Lower is less biased.",
                format="%.4f", min_value=0.0, max_value=float(board["B"].max()),
                alignment="left"),
            "s_DIS": st.column_config.NumberColumn(format="%+.4f", alignment="left"),
            "s_AMB": st.column_config.NumberColumn(format="%+.4f", alignment="left"),
            "acc_dis": st.column_config.NumberColumn(format="%.3f", alignment="left"),
            "acc_amb": st.column_config.NumberColumn(format="%.3f", alignment="left"),
            "◆ s_AMB unscaled": st.column_config.NumberColumn(
                format="%+.3f", alignment="left",
                help="s_AMB without the (1 - accuracy_ambig) factor: which way the "
                     "model leans when it does commit to a group."),
            "◆ abstained (amb)": st.column_config.NumberColumn(
                format="percent", alignment="left",
                help="Share of ambiguous items answered 'unknown'. This is what "
                     "damps s_AMB toward zero."),
            "◆ n_elig amb": st.column_config.NumberColumn(
                format="%d", alignment="left",
                help="Items in the ambiguous bias-score denominator. "
                                  "A small value makes s_AMB unstable."),
            "◆ s_answered": st.column_config.NumberColumn(
                format="%+.3f", alignment="left",
                help="Bias of the ambiguous answers actually given — "
                                   "s_AMB with the abstention scaling removed."),
        })

st.plotly_chart(charts.bias_dotplot(board), width="stretch")

# --- bias-score definition -------------------------------------------------
st.markdown("### Bias 계산식")
st.markdown(
    "$p_c$는 모델의 unknown이 아닌 응답 중 biased 응답의 비율입니다.")
st.markdown("$s_{raw,c}$는 $p_c$를 [-1, 1]로 스케일링한 값으로, +1은 모두 biased, " 
            "0은 biased/anti-biased가 반반, -1은 모두 anti-biased 응답임을 뜻합니다.")
st.latex(r"p_c = \frac{n_{\mathrm{biased},c}}{n_{\mathrm{non-unknown},c}},"
         r"\qquad s_{\mathrm{raw},c} = 2p_c - 1")
st.latex(r"s_{\mathrm{DIS}} = s_{\mathrm{raw,disambiguated}},"
         r"\qquad s_{\mathrm{AMB}} = "
         r"(1 - \mathrm{accuracy}_{\mathrm{ambiguous}})"
         r"\,s_{\mathrm{raw,ambiguous}}")
st.latex(r"{Bias} = |s_{\mathrm{DIS}}| + |s_{\mathrm{AMB}}|")
st.caption("Bias가 낮을수록 높은 순위가 매겨집니다.")

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
