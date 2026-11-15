"""Accuracy and bias against model size, within a family.

This is the view the three near-identical `pipeline*.ipynb` notebooks were built
to produce, one family at a time. Two charts carry the whole page: the left one
shows that growing a model mostly buys abstention, the right one shows that the
published score falls while the bias among answers actually given does not.
"""
from __future__ import annotations

import streamlit as st

from lib import binary, charts, db, explain
from lib import theme as t

st.set_page_config(page_title="Scaling · BBQ", page_icon="📈", layout="wide")
db.require_db()

runs = db.runs()
# Only families that actually form a ladder, longest first: a family with one
# run has nothing to say about size, and landing on it is an empty first screen.
ladders = (runs.loc[runs["params_b"].notna(), "family"].value_counts()
           .loc[lambda counts: counts >= 2])
families = ladders.index.tolist()
if not families:
    st.warning("크기 비교가 가능한 계열이 없습니다 — 계열마다 런이 두 개는 있어야 합니다.")
    st.stop()

explain.page_header(
    "모델이 커지면 편향이 줄어드나?",
    "같은 계열 안에서 <b>크기만</b> 바꿔 가며 비교합니다.")

# The family switch is the one control this page is for — in the body, as a row
# of buttons, so flipping between families during a talk is one click.
family = st.segmented_control(
    "모델 계열", families, default=families[0],
    format_func=lambda f: f"{f} ({ladders[f]}개)",
    label_visibility="collapsed") or families[0]
with st.sidebar:
    with st.expander("표 설정", expanded=False):
        db.sidebar_footer(st)

frame = db.q("""
    SELECT r.model_short, r.params_b, r.run_id, m.*
    FROM runs r JOIN run_metrics m ON m.run_id = r.run_id
    WHERE m.scope = 'overall' AND r.family = ? AND r.params_b IS NOT NULL
    ORDER BY r.params_b
""", (family,))
if len(frame) < 2:
    st.info(f"{family} 는 런이 두 개 미만이라 크기별로 그릴 것이 없습니다.")
    st.stop()

# Runs below the accuracy gate stay on the ladder — the size story needs them —
# but their bias numbers are not readable, so they are drawn hollow and left out
# of the sentence below.
gated = frame["accuracy_disambig"] < binary.GATE
readable = frame[~gated]

# --- the answer, computed from this family rather than written down ---------
if len(readable) >= 2:
    small, large = readable.iloc[0], readable.iloc[-1]
    B = lambda row: abs(row["s_DIS"]) + abs(row["s_AMB"])
    more_silent = large["accuracy_ambig"] > small["accuracy_ambig"]
    better_score = B(large) < B(small)
    still_biased = large["s_AMB_unscaled"] >= small["s_AMB_unscaled"]
    sentence = (
        f"**{family}**는 {small['params_b']:g}B → {large['params_b']:g}B 로 커지면서 "
        f"“모르겠다”가 **{small['accuracy_ambig']:.0%} → "
        f"{large['accuracy_ambig']:.0%}** 로 늘고, 편향 점수는 "
        f"**{B(small):.3f} → {B(large):.3f}** 이 됩니다. 그런데 **굳이 답할 때의 "
        f"편향**은 **{small['s_AMB_unscaled']:+.3f} → "
        f"{large['s_AMB_unscaled']:+.3f}** 입니다.")
    if more_silent and better_score and still_biased:
        st.warning(sentence + "\n\n**커져서 공정해진 것이 아니라, 커져서 말을 "
                              "아끼게 된 것입니다.** 점수를 좋게 만든 건 회피이고, "
                              "실제로 집단을 지목할 때의 쏠림은 줄지 않았습니다.")
    else:
        st.info(sentence)

# --- the two charts that carry the page ------------------------------------
left, right = st.columns(2)
with left:
    st.plotly_chart(charts.scaling_curve(
        frame, ("accuracy_ambig", "accuracy_disambig"),
        ('"모르겠다" 비율', "기본 실력 (정답률)"), (t.UNKNOWN, t.DISAMBIG),
        "커질수록 말을 아낀다", "비율",
        hline=binary.GATE, hline_label=f"실력 기준선 {binary.GATE:.0%}",
        hollow=gated), width="stretch")
with right:
    st.plotly_chart(charts.scaling_curve(
        frame, ("s_AMB", "s_AMB_unscaled"),
        ("편향 점수 (공식)", "굳이 답할 때 편향"), (t.AMBIG, t.BIASED),
        "점수는 내려가는데, 쏠림은 그대로", "편향",
        hline=0.0, hline_label="편향 없음", hollow=gated), width="stretch")

st.caption(
    "**속이 빈 점**은 기본 실력이 "
    f"{binary.GATE:.0%}에 못 미쳐 편향 점수를 읽을 수 없는 모델입니다 — 선은 "
    "이어 두되 추세를 그 점에 걸지 마세요. 왼쪽 회색 선(“모르겠다” 비율)과 오른쪽 "
    "주황 선(굳이 답할 때 편향)이 **같이 올라가는 것**이 이 페이지의 요지입니다.")

# --- everything else, one click away ---------------------------------------
with st.expander("응답 구성과 전체 수치", expanded=False):
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.scaling_response_composition(frame, "ambiguous"),
                        width="stretch")
        st.plotly_chart(charts.scaling_ambiguous_non_unknown(frame),
                        width="stretch")
    with right:
        st.plotly_chart(charts.scaling_response_composition(frame, "disambiguated"),
                        width="stretch")
    st.dataframe(
        frame[["model_short", "params_b", "accuracy_disambig", "accuracy_ambig",
               "s_DIS", "s_AMB", "s_AMB_unscaled", "n_elig_ambig"]]
        .rename(columns={
            "model_short": "모델",
            "params_b": "크기 (B)",
            "accuracy_disambig": "기본 실력",
            "accuracy_ambig": '"모르겠다" 비율',
            "s_DIS": "정보가 있을 때 편향",
            "s_AMB": "정보가 없을 때 편향",
            "s_AMB_unscaled": "굳이 답할 때 편향",
            "n_elig_ambig": "답한 문항 수"}),
        hide_index=True, width="stretch")
