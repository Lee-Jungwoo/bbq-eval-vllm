"""Binary (A/B) BBQ leaderboard — the arm without an "unknown" option.

Reads `bbq_pipeline/runs_binary/` through `lib/binary.py`. Deliberately a second
board rather than a toggle on Home: dropping the unknown option changes what
`s_AMB` means and deletes `accuracy_ambig` outright, so the two rankings must not
be read as one.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import binary, charts

st.set_page_config(page_title="BBQ binary leaderboard", page_icon="⚖️",
                   layout="wide")
binary.require_runs()

st.title("BBQ bias leaderboard · binary (A/B)")

st.info(
    "이 표의 모든 항목은 **선택지가 두 개**입니다 — 모호한 맥락에서도 "
    "“unknown”(Not enough info / Cannot be determined …) 선택지를 없애고 두 집단 "
    "중 하나를 고르도록 강제했습니다. 그래서 모델은 기권할 수 없습니다.\n\n"
    "**Home(3-옵션)과 순위를 섞지 마세요.** 여기의 `s_AMB` 는 "
    "`(1 − accuracy_ambig)` 스케일링이 없는 값이라, Home 의 `s_AMB` 가 아니라 "
    "`s_AMB unscaled` 와 비교해야 합니다.")

groups = binary.groups()
labels = {row.comparability_key: f"{row.comparability_key} · {row.n_runs} runs · "
                                f"{int(row.n_instances):,} items"
          for row in groups.itertuples(index=False)}

with st.sidebar:
    st.header("View")
    # Same rule as the 3-option board: only runs that answered the same
    # questions under the same prompt may be ranked against each other.
    key = st.selectbox("Comparability group", list(labels),
                       format_func=lambda k: labels[k],
                       help="sha256(prompt_hash | categories | n_instances). "
                            "binary arm 의 prompt_hash 는 3-옵션과 다르므로 "
                            "두 arm 의 런이 같은 그룹에 들어가는 일은 없습니다.")
    show_ineligible = st.checkbox("Show runs below the accuracy gate", value=True)
    families = st.multiselect("Model family", sorted(binary.runs()["family"].unique()),
                              default=None, placeholder="all families")
    sort_by = st.selectbox("Sort by", list(binary.SORT_OPTIONS))

board = binary.board(key)
if families:
    board = board[board["family"].isin(families)]
if not show_ineligible:
    board = board[board["eligible"] == 1]
if board.empty:
    st.warning("No runs match the current filters.")
    st.stop()

n_ranked = int((board["eligible"] == 1).sum())

# Gated runs always sink below the ranked field, whatever the sort key.
sort_column, ascending = binary.SORT_OPTIONS[sort_by]
board = board.sort_values(["eligible", sort_column], ascending=[False, ascending])

st.html(binary.render(board))
st.caption(
    f"Bias B(m)은 낮을수록 편향이 작습니다. 막대 색: 주황 = 고정관념 방향, "
    f"파랑 = 반(反)고정관념 방향. 정확도 게이트는 `accuracy_disambig` ≥ "
    f"{binary.GATE:.2f} 이며 {n_ranked}개 런이 순위에 들어갔습니다. "
    f"`accuracy_ambig` 열은 없습니다 — unknown 을 없애면 모호한 맥락에는 정답이 "
    f"존재하지 않습니다.")

# --- what this arm measures instead of abstention ---------------------------
with st.expander("전체 수치", expanded=False):
    st.markdown(
        "`target_rate` 는 고정관념 집단을 지목한 비율입니다. `neg` 항목에서는 "
        "`(s+1)/2`, `nonneg` 항목에서는 그 거울상 `(1−s)/2` 이므로 고정관념 방향으로 "
        "일관된 모델은 **합산 0.5 근처**에 놓입니다. 합산값이 0.5에서 크게 벗어나면 "
        "질문의 극성과 **무관하게** 특정 집단을 더(또는 덜) 지목한다는 뜻이고, "
        "이는 `s_DIS`/`s_AMB` 로는 보이지 않습니다. `A rate` 는 위치 편향 점검용으로, "
        "BBQ 는 항목마다 선택지 순서를 무작위화하므로 0.5에서 벗어나면 모델 쪽 "
        "성향입니다.")
    view = pd.DataFrame({
        "rank": board["rank"],
        "model": board["model_short"],
        "B": board["B"],
        "B 95% CI": [f"[{lo:.4f}, {hi:.4f}]" if pd.notna(lo) and pd.notna(hi) else "—"
                     for lo, hi in zip(board["B_lo"], board["B_hi"])],
        "s_DIS": board["s_DIS"],
        "s_AMB": board["s_AMB"],
        "acc_dis": board["accuracy_disambig"],
        "target_rate (amb)": board["target_rate_ambig"],
        "target_rate (dis)": board["target_rate_disambig"],
        "target_rate amb·neg": board["target_rate_ambig_neg"],
        "target_rate amb·nonneg": board["target_rate_ambig_nonneg"],
        "A rate": board["letter_A_rate"],
        "prob mass (A+B)": board["mean_prob_mass_captured"],
        "n": board["n"],
    })
    st.dataframe(
        view, hide_index=True, width="stretch",
        column_config={
            "rank": st.column_config.NumberColumn(format="%d", alignment="left"),
            "B": st.column_config.NumberColumn(format="%.4f", alignment="left"),
            "s_DIS": st.column_config.NumberColumn(format="%+.4f", alignment="left"),
            "s_AMB": st.column_config.NumberColumn(
                format="%+.4f", alignment="left",
                help="스케일링 없는 값. Home 의 s_AMB unscaled 와 같은 층위입니다."),
            "acc_dis": st.column_config.NumberColumn(format="%.3f", alignment="left"),
            "target_rate (amb)": st.column_config.NumberColumn(
                format="%.3f", alignment="left",
                help="모호한 맥락에서 고정관념 집단을 지목한 비율."),
            "target_rate (dis)": st.column_config.NumberColumn(
                format="%.3f", alignment="left"),
            "target_rate amb·neg": st.column_config.NumberColumn(
                format="%.3f", alignment="left",
                help="부정 질문(neg)만. (s_AMB + 1) / 2 와 같습니다."),
            "target_rate amb·nonneg": st.column_config.NumberColumn(
                format="%.3f", alignment="left",
                help="긍정 질문(nonneg)만. 거울상이라 neg 와 합쳐지면 0.5 근처가 됩니다."),
            "A rate": st.column_config.NumberColumn(
                format="%.3f", alignment="left",
                help="A 선택률. 0.5 = 위치 선호 없음."),
            "prob mass (A+B)": st.column_config.NumberColumn(
                format="%.3f", alignment="left",
                help="라벨 토큰 두 개에 실린 확률 질량. 3-옵션 런보다 약간 낮은 것이 정상."),
            "n": st.column_config.NumberColumn(format="%d", alignment="left"),
        })

st.plotly_chart(binary.bias_dotplot(board), width="stretch")

# --- bias-score definition, as it actually is in this arm -------------------
st.markdown("### Bias 계산식 (binary arm)")
st.markdown(
    "$p_c$는 각 맥락 조건 $c$에서 biased 응답의 비율입니다. unknown 선택지가 없으므로 "
    "모든 응답이 분모에 들어가고, 제외되는 것은 `target_loc` 메타데이터가 없는 "
    "~16개 항목뿐입니다.")
st.latex(r"p_c = \frac{n_{\mathrm{biased},c}}{n_c},"
         r"\qquad s_{\mathrm{raw},c} = 2p_c - 1")
st.latex(r"s_{\mathrm{DIS}} = s_{\mathrm{raw,disambiguated}},"
         r"\qquad s_{\mathrm{AMB}} = s_{\mathrm{raw,ambiguous}}")
st.latex(r"{Bias} = |s_{\mathrm{DIS}}| + |s_{\mathrm{AMB}}|")
st.caption(
    "3-옵션 arm 의 $s_{AMB} = (1 - \\mathrm{accuracy_{ambiguous}})\\,s_{raw,ambiguous}$ "
    "와 비교하세요. 그 스케일링 항은 “모델이 올바르게 기권한 비율만큼 편향을 "
    "할인”하는 것인데, 기권이 불가능한 이 arm 에서는 정의되지 않으므로 빠집니다.")

# --- per-category view ------------------------------------------------------
with st.expander("카테고리별 점수", expanded=False):
    picked = st.selectbox("Run", board["model_short"].tolist())
    run_id = board.loc[board["model_short"] == picked, "run_id"].iloc[0]
    by_category = binary.per_category(run_id)
    if by_category.empty:
        st.info("이 런의 summary.json 에는 카테고리별 블록이 없습니다.")
    else:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(charts.category_bars(
                by_category, ("s_AMB", "s_DIS"), ("s_AMB", "s_DIS"),
                "Bias score by category  (+ = toward the stereotype)",
                hline=0.0, xaxis_title="bias score"), width="stretch")
        with right:
            st.plotly_chart(charts.category_bars(
                by_category, ("target_rate_ambig", "target_rate_disambig"),
                ("ambiguous", "disambiguated"),
                "고정관념 집단 지목률 by category",
                hline=0.5, hline_label="0.5", xaxis_title="target_rate"),
                width="stretch")
        st.dataframe(
            by_category[[c for c in ("scope_value", "n", "accuracy_disambig",
                                     "s_AMB", "s_DIS", "target_rate_ambig",
                                     "target_rate_disambig", "letter_A_rate")
                         if c in by_category]]
            .rename(columns={"scope_value": "category"}),
            hide_index=True, width="stretch")

st.caption(
    "이 페이지의 숫자는 `bbq_pipeline/runs_binary/*/summary.json` 에서 그대로 "
    "읽어옵니다 — 계산은 `aggregate_binary.py` 가 하고, `bbq.sqlite` 와 "
    "`leaderboard/exports/` 는 3-옵션 arm 전용입니다. 자세한 내용은 "
    "`bbq_pipeline/README.md` 의 “Binary arm” 절을 보세요.")
