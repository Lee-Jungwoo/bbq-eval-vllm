"""3-option vs binary (A/B) — does removing the unknown option reorder the field?

Joins the published 3-option board (`bbq.sqlite`, via `lib/db.py`) to the binary
arm's summaries (`bbq_pipeline/runs_binary/`, via `lib/binary.py`) and ranks both
sides over the same set of models. `lib/compare.py` does the join, the ranking and
the figures; this file is layout and prose.

Kept as its own page rather than a toggle on either board for the same reason the
two boards are separate: the arms measure different things, and the only honest
way to show them together is with the caveat attached.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import binary, compare, db, explain

st.set_page_config(page_title="BBQ 3-opt vs binary", page_icon="⚖️",
                   layout="wide")
db.require_db()
binary.require_runs()

explain.page_header(
    "회피를 막으면 순위가 뒤집힌다",
    "같은 모델을 <b>3지선다</b>(“모르겠다” 가능)와 <b>2지선다</b>(강제 선택)에서 "
    "각각 순위 매겨 나란히 놓았습니다.")

# --- which comparability group on each side --------------------------------
tri_groups = db.comparability_groups()
bin_groups = binary.groups()

with st.sidebar:
    with st.expander("표 설정", expanded=False):
        basis = st.radio(
            "순위 기준", list(compare.RANK_BASES), index=1,
            help="두 arm 의 순위를 무엇으로 매길지. 기본값은 스케일링을 걷어낸 B*.")
        families = st.multiselect(
            "모델 계열", sorted(db.runs()["family"].unique()),
            default=None, placeholder="전체")
        sort_by = st.selectbox("정렬 기준", list(compare.sort_options(basis)))
        show_ineligible = st.checkbox("정답률이 낮아 제외된 모델도 표시", value=True)

        # The two arms never share a key — the binary prompt hash differs — so
        # the group is picked once per side rather than from one list.
        tri_keys = tri_groups["comparability_key"].tolist()
        bin_keys = bin_groups["comparability_key"].tolist()
        tri_key = tri_keys[0] if len(tri_keys) == 1 else st.selectbox(
            "비교 가능 묶음 · 3지선다", tri_keys)
        bin_key = bin_keys[0] if len(bin_keys) == 1 else st.selectbox(
            "비교 가능 묶음 · 2지선다", bin_keys)
        db.sidebar_footer(st)

merged = compare.frame(tri_key, bin_key)
if merged.empty:
    st.warning("두 쪽에 모두 있는 모델이 없습니다.")
    st.stop()

merged = compare.rank(merged, basis)
if families:
    merged = merged[merged["family"].isin(families)]
if not show_ineligible:
    merged = merged[merged["eligible"] == 1]
if merged.empty:
    st.warning("조건에 맞는 모델이 없습니다.")
    st.stop()

_, _, basis_note = compare.RANK_BASES[basis]
n_ranked = int((merged["eligible"] == 1).sum())
moved = merged.loc[merged["eligible"] == 1, "rank_delta"].abs()

st.info(
    f"순위는 양쪽 모두에서 기본 실력 {binary.GATE:.0%}를 넘긴 {n_ranked}개 모델 "
    f"안에서 **이 페이지가 다시 계산**합니다 — 한쪽에만 있는 모델이 아래 순위를 "
    f"통째로 밀어 없는 변동을 만들어내지 않도록 하기 위해서입니다.\n\n"
    f"**순위 기준: {basis}** — {basis_note}")

# --- headline: what moved --------------------------------------------------
top = st.columns(4)
top[0].metric("가장 큰 변동", f"{int(moved.max())} 계단" if len(moved) else "—")
top[1].metric("평균 변동", f"{moved.mean():.1f} 계단" if len(moved) else "—")
top[2].metric("순위가 그대로인 모델", int((moved == 0).sum()))
top[3].metric("나란히 순위 매긴 모델", n_ranked,
              delta=f"{len(merged) - n_ranked}개 제외", delta_color="off")

# --- the table -------------------------------------------------------------
# Gated runs always sink below the ranked field, whatever the sort key — the same
# rule Home and the binary board follow.
sort_column, ascending = compare.sort_options(basis)[sort_by]
merged = merged.sort_values(["eligible", sort_column],
                            ascending=[False, ascending])

st.html(compare.render(merged, basis))
st.caption(
    "**순위 이동** 열을 세로로 훑으세요. 빈 원 = 3지선다 순위, 채워진 원 = 2지선다 "
    f"순위이고, 모든 줄이 똑같은 1–{n_ranked}위 구간을 씁니다. "
    "**주황 = 회피를 못 하게 하니 순위가 떨어진 모델**, 파랑 = 올라간 모델, "
    "회색 = 그대로. 점수 막대 색은 앱 전체와 같습니다 — 주황 = 고정관념 방향, "
    "파랑 = 반대 방향.")

# --- why the two arms disagree ---------------------------------------------
with st.expander("두 순위가 왜 어긋나는가 — 회피율과의 관계", expanded=False):
    st.markdown(
        "3지선다의 “굳이 답할 때 편향”은 **회피하지 않은 답만** 분모로 씁니다. "
        "회피율이 높은 모델일수록 그 분모가 작아지고, 남은 소수의 답은 편향 쪽으로 "
        "치우쳐 있어 편향이 과대추정됩니다. 2지선다에서는 모든 문항이 분모에 "
        "들어가므로 그런 축소가 없습니다.")
    st.plotly_chart(compare.abstention_gap(merged), width="stretch")
    st.caption(
        "y 가 0보다 크면 3지선다가 2지선다보다 편향을 크게 잡았다는 뜻입니다. "
        "오른쪽(회피율이 높은 쪽)으로 갈수록 격차가 벌어지면, 3지선다 순위의 "
        "상당 부분이 편향이 아니라 **회피 습관**을 재고 있다는 신호입니다.")

# --- full numbers -----------------------------------------------------------
with st.expander("전체 수치", expanded=False):
    view = pd.DataFrame({
        "계열": merged["family"],
        "크기 (B)": merged["params_b"],
        "모델": merged["model_short"],
        "순위 · 3지선다": merged["rank_tri"],
        "순위 · 2지선다": merged["rank_bin"],
        "순위 변동": merged["rank_delta"],
        "편향 점수 · 3지선다": merged["tri_B"],
        "편향 점수(보정) · 3지선다": merged["tri_B_star"],
        "편향 점수 · 2지선다": merged["bin_B"],
        "정보 없을 때 편향 · 3지선다": merged["tri_s_AMB"],
        "굳이 답할 때 편향 · 3지선다": merged["tri_s_AMB_unscaled"],
        "정보 없을 때 편향 · 2지선다": merged["bin_s_AMB"],
        "정보 있을 때 편향 · 3지선다": merged["tri_s_DIS"],
        "정보 있을 때 편향 · 2지선다": merged["bin_s_DIS"],
        '"모르겠다" 비율 · 3지선다': merged["tri_accuracy_ambig"],
        "답한 문항 수 · 3지선다": merged["tri_n_elig_ambig"],
        "기본 실력 · 3지선다": merged["tri_accuracy_disambig"],
        "기본 실력 · 2지선다": merged["bin_accuracy_disambig"],
    })
    signed = ("정보 없을 때 편향 · 3지선다", "굳이 답할 때 편향 · 3지선다",
              "정보 없을 때 편향 · 2지선다", "정보 있을 때 편향 · 3지선다",
              "정보 있을 때 편향 · 2지선다")
    st.dataframe(
        view, hide_index=True, width="stretch",
        column_config={
            "크기 (B)": st.column_config.NumberColumn(format="%g", alignment="left"),
            "순위 · 3지선다": st.column_config.NumberColumn(format="%d",
                                                      alignment="left"),
            "순위 · 2지선다": st.column_config.NumberColumn(format="%d",
                                                      alignment="left"),
            "순위 변동": st.column_config.NumberColumn(
                format="%+d", alignment="left",
                help="3지선다 순위 − 2지선다 순위. +면 회피를 막았을 때 올라간 모델."),
            "편향 점수 · 3지선다": st.column_config.NumberColumn(format="%.4f",
                                                          alignment="left"),
            "편향 점수(보정) · 3지선다": st.column_config.NumberColumn(
                format="%.4f", alignment="left",
                help="회피에 따른 스케일링을 걷어낸 값 — 2지선다 점수와 같은 층위."),
            "편향 점수 · 2지선다": st.column_config.NumberColumn(format="%.4f",
                                                          alignment="left"),
            '"모르겠다" 비율 · 3지선다': st.column_config.NumberColumn(
                format="%.3f", alignment="left",
                help="정보가 없는 문항의 정답이 곧 “모르겠다”라 사실상 회피율입니다."),
            "답한 문항 수 · 3지선다": st.column_config.NumberColumn(
                format="%d", alignment="left",
                help="“굳이 답할 때 편향”의 분모. 작을수록 불안정합니다."),
            "기본 실력 · 3지선다": st.column_config.NumberColumn(format="%.3f",
                                                          alignment="left"),
            "기본 실력 · 2지선다": st.column_config.NumberColumn(format="%.3f",
                                                          alignment="left"),
            **{c: st.column_config.NumberColumn(format="%+.4f", alignment="left")
               for c in signed},
        })

st.caption(
    "3지선다 수치는 `leaderboard/bbq.sqlite`, 2지선다 수치는 "
    "`bbq_pipeline/runs_binary/*/summary.json` 에서 그대로 읽어옵니다. 이 페이지는 "
    "두 결과를 붙이고 순위만 다시 매길 뿐, 어떤 지표도 새로 정의하지 않습니다.")
