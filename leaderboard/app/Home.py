"""BBQ bias leaderboard — ranking page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import charts, db, explain, table
from lib import theme as t

st.set_page_config(page_title="BBQ bias leaderboard", page_icon="⚖️",
                   layout="wide")
db.require_db()

groups = db.comparability_groups()
keys = groups["comparability_key"].tolist()
labels = {row.comparability_key: f"{row.comparability_key} · {row.n_runs} runs · "
                                f"{row.n_instances:,} items"
          for row in groups.itertuples(index=False)}

# Every control on this page needs BBQ vocabulary to use, and none of them gets
# touched during a talk — so they live one click away, and the sidebar shows the
# page list instead. The comparability group is not a choice a reader should be
# asked to make at all: it only exists so runs made under different prompts
# cannot be ranked together, and there is normally one of them.
with st.sidebar:
    with st.expander("표 설정", expanded=False):
        key = keys[0] if len(keys) == 1 else st.selectbox(
            "비교 가능 묶음", keys, format_func=lambda k: labels[k],
            help="같은 프롬프트로 같은 문항을 푼 런끼리만 순위를 매깁니다.")
        families = st.multiselect("모델 계열", sorted(db.runs()["family"].unique()),
                                  default=None, placeholder="전체")
        sort_by = st.selectbox("정렬 기준", list(table.SORT_OPTIONS))
        show_ineligible = st.checkbox("정답률이 낮아 제외된 모델도 표시", value=True)
        db.sidebar_footer(st)

board = db.leaderboard(key)
if families:
    board = board[board["family"].isin(families)]
if not show_ineligible:
    board = board[board["eligible"] == 1]
if board.empty:
    st.warning("조건에 맞는 모델이 없습니다.")
    st.stop()

threshold = float(board["eligibility_threshold"].iloc[0])
n_ranked = int((board["eligible"] == 1).sum())
n_items = int(board["n_ambig"].iloc[0]) * 2

explain.page_header(
    "모델은 얼마나 고정관념대로 답하는가",
    f"모델 <b>{len(board)}개</b>에게 같은 문항 <b>{n_items:,}개</b>를 풀렸습니다. "
    f"점수가 <b>0에 가까울수록</b> 고정관념에 덜 기울었다는 뜻입니다.")

# The audience has never seen a BBQ item. One real pair explains in three seconds
# why there are two bias columns instead of one.
explain.example_item()

ranked = board[board["eligible"] == 1].sort_values("rank")
first = ranked.iloc[0] if not ranked.empty else None

top = st.columns(3)
top[0].metric("순위에 든 모델", f"{n_ranked}개",
              delta=f"{len(board) - n_ranked}개는 정답률 미달로 제외",
              delta_color="off")
top[1].metric("푼 문항", f"{n_items:,}개",
              delta="정보 있음 / 없음 절반씩", delta_color="off")
if first is not None:
    top[2].metric("1위 모델이 “모르겠다”고 답한 비율",
                  f"{first['unknown_rate_ambig']:.0%}",
                  delta="바로 아래를 보세요", delta_color="off")

# --- the caveat, computed live rather than written down --------------------
# This is the whole talk in one box, and every number in it comes off the frame
# so it cannot go stale when a run is added — including the correlation, which
# is what turns "#1 abstained" into "the ranking is mostly an abstention
# ranking". Spearman without scipy: Pearson on the ranks.
if first is not None and len(ranked) > 2:
    rho = float(ranked["rank"].rank().corr(ranked["unknown_rate_ambig"].rank()))
    talker = ranked.nsmallest(1, "unknown_rate_ambig").iloc[0]
    lined_up = "**거의 회피율 순서대로 줄을 선 셈입니다.** " if rho <= -0.7 else ""
    st.warning(
        f"**1위 {first['model_short']}는 편향 점수 {first['B']:.4f}로 가장 낮습니다. "
        f"그런데 정보가 없는 문항의 {first['unknown_rate_ambig']:.0%}를 “모르겠다”로 "
        f"넘겼습니다.**\n\n"
        f"{int(first['n_ambig']):,}개 중 실제로 누군가를 지목한 건 "
        f"**{int(first['n_elig_ambig']):,}개**뿐 — 점수가 전체 문항의 "
        f"{first['n_elig_ambig'] / first['n_ambig']:.1%} 위에서 계산됐습니다. "
        f"이건 이 모델만의 얘기가 아닙니다: **순위와 회피율의 상관이 {rho:+.2f}**. "
        f"{lined_up}회피를 가장 적게 한 {talker['model_short']}"
        f"({talker['unknown_rate_ambig']:.0%})는 {int(talker['rank'])}위입니다.\n\n"
        f"편향 점수가 “회피한 만큼 편향을 깎아 주는” 계산이기 때문입니다. **답을 안 "
        f"하면 점수가 좋아집니다.** 그래서 뒤에서 “모르겠다”를 아예 없애고 다시 "
        f"재봅니다.")

# --- the table -------------------------------------------------------------
# Gated runs always sink below the ranked field, whatever the sort key.
sort_column, ascending = table.SORT_OPTIONS[sort_by]
board = board.sort_values(["eligible", sort_column],
                          ascending=[False, ascending])

st.html(table.render(board, threshold=threshold))
st.caption(
    "막대 색: **주황 = 고정관념 방향**, **파랑 = 반대 방향**. "
    "**“모르겠다” 비율 열을 편향 점수 옆에 붙여 뒀습니다** — 위쪽 모델들이 왜 "
    "점수가 좋은지가 그 두 칸에 다 있습니다. 회색 줄은 기본 실력이 "
    f"{threshold:.0%}에 못 미쳐 순위에서 뺀 모델로, 문제 자체를 못 푸는 모델의 "
    "편향 점수는 읽을 수 없습니다.")

# The rank is built from the abstention diagnostics but does not show them; keep
# them one click away rather than in the headline table.
with st.expander("전체 수치", expanded=False):
    # `unknown_rate_ambig` == `accuracy_ambig` and `s_answered` ==
    # `s_AMB_unscaled` — identical by construction, since the correct answer in
    # an ambiguous context *is* "unknown". One column each, not two.
    view = pd.DataFrame({
        "순위": board["rank"],
        "모델": board["model_short"],
        "편향 점수": board["B"],
        "95% 신뢰구간": [t.fmt_interval(lo, hi) for lo, hi
                     in zip(board["B_lo"], board["B_hi"])],
        "정보가 있을 때 편향": board["s_DIS"],
        "정보가 없을 때 편향": board["s_AMB"],
        "기본 실력": board["accuracy_disambig"],
        '"모르겠다" 비율': board["accuracy_ambig"],
        "굳이 답할 때 편향": board["s_AMB_unscaled"],
        "답한 문항 수": board["n_elig_ambig"],
        "회피 경고": board["abstention_tier"].map(t.TIER_LABEL),
    })
    st.dataframe(
        view, hide_index=True, width="stretch",
        column_config={
            "순위": st.column_config.NumberColumn(
                width="small", format="%d", alignment="left"),
            "편향 점수": st.column_config.NumberColumn(
                help="|정보가 있을 때 편향| + |정보가 없을 때 편향|. 낮을수록 공정.",
                format="%.4f", min_value=0.0, max_value=float(board["B"].max()),
                alignment="left"),
            "정보가 있을 때 편향": st.column_config.NumberColumn(
                format="%+.4f", alignment="left",
                help="s_DIS — 맥락에 답이 있는 문항. +면 고정관념 쪽으로 틀립니다."),
            "정보가 없을 때 편향": st.column_config.NumberColumn(
                format="%+.4f", alignment="left",
                help="s_AMB — 맥락에 답이 없는 문항. 회피한 만큼 0쪽으로 눌립니다."),
            "기본 실력": st.column_config.NumberColumn(
                format="%.3f", alignment="left",
                help="accuracy_disambig — 답이 있는 문항의 정답률."),
            '"모르겠다" 비율': st.column_config.NumberColumn(
                format="percent", alignment="left",
                help="정보가 없는 문항에서 “알 수 없음”을 고른 비율. 그 문항의 정답이 "
                     "곧 “알 수 없음”이라 정답률과 같은 값이며, 이 값이 클수록 "
                     "“정보가 없을 때 편향”이 0쪽으로 눌립니다."),
            "굳이 답할 때 편향": st.column_config.NumberColumn(
                format="%+.3f", alignment="left",
                help="s_AMB unscaled (= s_answered) — 회피하지 않고 집단을 지목한 "
                     "답만 모아서 본 쏠림."),
            "답한 문항 수": st.column_config.NumberColumn(
                format="%d", alignment="left",
                help="n_elig_ambig — 위 값의 분모. 작을수록 불안정합니다."),
        })

st.plotly_chart(charts.bias_dotplot(board), width="stretch")

# --- bias-score definition -------------------------------------------------
with st.expander("편향 점수는 어떻게 계산하나 (수식)", expanded=False):
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
    with st.expander(f"순위 차이가 우연이 아닌가?  "
                     f"{len(pairs)}쌍 중 {n_sep}쌍만 구분됩니다", expanded=False):
        st.markdown(
            "같은 템플릿에서 나온 4문항 묶음 단위로 다시 뽑기를 반복하되, 모든 "
            "모델에 똑같은 뽑기를 적용합니다. 그래서 두 모델의 차이는 짝지어져 "
            "있고 문항 난이도는 상쇄됩니다.")
        st.plotly_chart(charts.separation_heatmap(pairs, order),
                        width="stretch")
        unresolved = pairs[pairs["separated"] == 0]
        if not unresolved.empty:
            st.markdown("**순서를 가릴 수 없는 쌍:**")
            st.dataframe(
                pd.DataFrame({
                    "모델 A": unresolved["model_a"],
                    "모델 B": unresolved["model_b"],
                    "점수 차": unresolved["delta_B"],
                    "95% 신뢰구간": [f"[{lo:+.4f}, {hi:+.4f}]" for lo, hi
                                 in zip(unresolved["delta_lo"], unresolved["delta_hi"])],
                    "A가 덜 편향일 확률": unresolved["p_a_less_biased"],
                }), hide_index=True, width="stretch",
                column_config={
                    "점수 차": st.column_config.NumberColumn(format="%+.4f"),
                    "A가 덜 편향일 확률": st.column_config.NumberColumn(format="%.3f"),
                })
