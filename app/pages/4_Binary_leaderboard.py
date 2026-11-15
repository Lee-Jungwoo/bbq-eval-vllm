"""Binary (A/B) BBQ leaderboard — the arm without an "unknown" option.

Reads `results/runs_binary/` through `lib/binary.py`. Deliberately a second
board rather than a toggle on Home: dropping the unknown option changes what
`s_AMB` means and deletes `accuracy_ambig` outright, so the two rankings must not
be read as one.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import binary, charts, explain
from lib import theme as t

st.set_page_config(page_title="BBQ binary leaderboard", page_icon="⚖️",
                   layout="wide")
binary.require_runs()

explain.page_header(
    "“모르겠다”를 없애면 무엇이 드러나는가",
    "같은 문항에서 <b>“알 수 없음” 선택지를 지우고</b> 두 집단 중 하나를 반드시 "
    "고르게 했습니다. 이제 어떤 모델도 회피할 수 없습니다.")

groups = binary.groups()
labels = {row.comparability_key: f"{row.comparability_key} · {row.n_runs} runs · "
                                f"{int(row.n_instances):,} items"
          for row in groups.itertuples(index=False)}

# Same rule as the 3-option board: only runs that answered the same questions
# under the same prompt may be ranked against each other — and, as there, it is
# not a choice worth putting in front of a reader.
keys = list(labels)
with st.sidebar:
    with st.expander("표 설정", expanded=False):
        key = keys[0] if len(keys) == 1 else st.selectbox(
            "비교 가능 묶음", keys, format_func=lambda k: labels[k])
        families = st.multiselect("모델 계열",
                                  sorted(binary.runs()["family"].unique()),
                                  default=None, placeholder="전체")
        sort_by = st.selectbox("정렬 기준", list(binary.SORT_OPTIONS))
        show_ineligible = st.checkbox("정답률이 낮아 제외된 모델도 표시", value=True)

board = binary.board(key)
if families:
    board = board[board["family"].isin(families)]
if not show_ineligible:
    board = board[board["eligible"] == 1]
if board.empty:
    st.warning("조건에 맞는 모델이 없습니다.")
    st.stop()

n_ranked = int((board["eligible"] == 1).sum())

# Gated runs always sink below the ranked field, whatever the sort key.
sort_column, ascending = binary.SORT_OPTIONS[sort_by]
board = board.sort_values(["eligible", sort_column], ascending=[False, ascending])

st.html(binary.render(board))
st.caption(
    f"막대 색: **주황 = 고정관념 방향**, **파랑 = 반대 방향**. 기본 실력이 "
    f"{binary.GATE:.0%}에 못 미치는 모델은 순위에서 빼, {n_ranked}개가 순위에 "
    f"들어갔습니다. <strong>“모르겠다” 비율 열은 없습니다</strong> — 선택지를 "
    f"지웠으니 정보가 없는 문항에는 정답 자체가 존재하지 않습니다.",
    unsafe_allow_html=True)

# --- what this arm measures instead of abstention ---------------------------
with st.expander("전체 수치", expanded=False):
    st.markdown(
        "**고정관념 집단을 고른 비율**은 부정 질문(`neg`)과 그 거울상인 긍정 "
        "질문(`nonneg`)이 서로 반대 방향이라, 고정관념 방향으로 일관된 모델은 "
        "둘을 합치면 **0.5 근처**에 놓입니다. 합산값이 0.5에서 크게 벗어나면 질문의 "
        "방향과 **무관하게** 특정 집단을 더(또는 덜) 지목한다는 뜻이고, 이는 편향 "
        "점수로는 보이지 않습니다. **A만 고르는 습관**은 위치 편향 점검용입니다 — "
        "BBQ 는 문항마다 선택지 순서를 섞으므로 0.5에서 벗어나면 모델 쪽 성향입니다.")
    view = pd.DataFrame({
        "순위": board["rank"],
        "모델": board["model_short"],
        "편향 점수": board["B"],
        "95% 신뢰구간": [f"[{lo:.4f}, {hi:.4f}]" if pd.notna(lo) and pd.notna(hi) else "—"
                     for lo, hi in zip(board["B_lo"], board["B_hi"])],
        "정보가 있을 때 편향": board["s_DIS"],
        "정보가 없을 때 편향": board["s_AMB"],
        "기본 실력": board["accuracy_disambig"],
        "고정관념 집단을 고른 비율 (정보 없음)": board["target_rate_ambig"],
        "고정관념 집단을 고른 비율 (정보 있음)": board["target_rate_disambig"],
        "  └ 부정 질문만": board["target_rate_ambig_neg"],
        "  └ 긍정 질문만": board["target_rate_ambig_nonneg"],
        "A만 고르는 습관": board["letter_A_rate"],
        "확률 질량 (A+B)": board["mean_prob_mass_captured"],
        "문항 수": board["n"],
    })
    st.dataframe(
        view, hide_index=True, width="stretch",
        column_config={
            "순위": st.column_config.NumberColumn(format="%d", alignment="left"),
            "편향 점수": st.column_config.NumberColumn(format="%.4f", alignment="left"),
            "정보가 있을 때 편향": st.column_config.NumberColumn(
                format="%+.4f", alignment="left"),
            "정보가 없을 때 편향": st.column_config.NumberColumn(
                format="%+.4f", alignment="left",
                help="회피가 불가능하므로 스케일링이 없습니다. 3지선다 쪽의 "
                     "“굳이 답할 때 편향”과 같은 층위입니다."),
            "기본 실력": st.column_config.NumberColumn(format="%.3f", alignment="left"),
            "고정관념 집단을 고른 비율 (정보 없음)": st.column_config.NumberColumn(
                format="%.3f", alignment="left",
                help="맥락에 단서가 없는데도 고정관념 집단을 지목한 비율."),
            "고정관념 집단을 고른 비율 (정보 있음)": st.column_config.NumberColumn(
                format="%.3f", alignment="left"),
            "  └ 부정 질문만": st.column_config.NumberColumn(
                format="%.3f", alignment="left",
                help="부정적인 고정관념을 묻는 질문만 모은 값."),
            "  └ 긍정 질문만": st.column_config.NumberColumn(
                format="%.3f", alignment="left",
                help="그 거울상. 부정 질문과 합치면 0.5 근처가 되는 것이 정상입니다."),
            "A만 고르는 습관": st.column_config.NumberColumn(
                format="%.3f", alignment="left",
                help="A 선택률. 0.5 = 위치 선호 없음."),
            "확률 질량 (A+B)": st.column_config.NumberColumn(
                format="%.3f", alignment="left",
                help="두 선택지 토큰에 실린 확률. 3지선다보다 약간 낮은 것이 정상."),
            "문항 수": st.column_config.NumberColumn(format="%d", alignment="left"),
        })

st.plotly_chart(binary.bias_dotplot(board), width="stretch")

# --- bias-score definition, as it actually is in this arm -------------------
formula = st.expander("편향 점수는 어떻게 계산하나 (수식)", expanded=False)
formula.markdown(
    "$p_c$는 각 맥락 조건 $c$에서 biased 응답의 비율입니다. unknown 선택지가 없으므로 "
    "모든 응답이 분모에 들어가고, 제외되는 것은 `target_loc` 메타데이터가 없는 "
    "~16개 항목뿐입니다.")
formula.latex(r"p_c = \frac{n_{\mathrm{biased},c}}{n_c},"
              r"\qquad s_{\mathrm{raw},c} = 2p_c - 1")
formula.latex(r"s_{\mathrm{DIS}} = s_{\mathrm{raw,disambiguated}},"
              r"\qquad s_{\mathrm{AMB}} = s_{\mathrm{raw,ambiguous}}")
formula.latex(r"{Bias} = |s_{\mathrm{DIS}}| + |s_{\mathrm{AMB}}|")
formula.caption(
    "3-옵션 arm 의 $s_{AMB} = (1 - \\mathrm{accuracy_{ambiguous}})\\,s_{raw,ambiguous}$ "
    "와 비교하세요. 그 스케일링 항은 “모델이 올바르게 기권한 비율만큼 편향을 "
    "할인”하는 것인데, 기권이 불가능한 이 arm 에서는 정의되지 않으므로 빠집니다.")

# --- per-category view ------------------------------------------------------
with st.expander("속성별 점수", expanded=False):
    picked = st.selectbox("모델", board["model_short"].tolist())
    run_id = board.loc[board["model_short"] == picked, "run_id"].iloc[0]
    by_category = binary.per_category(run_id)
    if not by_category.empty:
        by_category = by_category.assign(
            scope_value=by_category["scope_value"].map(t.category_ko))
    if by_category.empty:
        st.info("이 런에는 속성별 수치가 없습니다.")
    else:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(charts.category_bars(
                by_category, ("s_AMB", "s_DIS"), ("정보 없을 때", "정보 있을 때"),
                "속성별 편향 점수  (+ = 고정관념 방향)",
                hline=0.0, xaxis_title="편향 점수"), width="stretch")
        with right:
            st.plotly_chart(charts.category_bars(
                by_category, ("target_rate_ambig", "target_rate_disambig"),
                ("정보 없음", "정보 있음"),
                "속성별 고정관념 집단을 고른 비율",
                hline=0.5, hline_label="0.5", xaxis_title="비율"),
                width="stretch")
        st.dataframe(
            by_category[[c for c in ("scope_value", "n", "accuracy_disambig",
                                     "s_AMB", "s_DIS", "target_rate_ambig",
                                     "target_rate_disambig", "letter_A_rate")
                         if c in by_category]]
            .rename(columns={
                "scope_value": "속성",
                "n": "문항 수",
                "accuracy_disambig": "기본 실력",
                "s_AMB": "정보가 없을 때 편향",
                "s_DIS": "정보가 있을 때 편향",
                "target_rate_ambig": "고정관념 선택률 (정보 없음)",
                "target_rate_disambig": "고정관념 선택률 (정보 있음)",
                "letter_A_rate": "A만 고르는 습관"}),
            hide_index=True, width="stretch")

st.caption(
    "이 페이지의 숫자는 `results/runs_binary/*/summary.json` 에서 그대로 "
    "읽어옵니다 — 계산은 `aggregate_binary.py` 가 하고, `bbq.sqlite` 와 "
    "`results/leaderboard/exports/` 는 3-옵션 arm 전용입니다. 자세한 내용은 "
    "`docs/pipeline.md` 의 “Binary arm” 절을 보세요.")
