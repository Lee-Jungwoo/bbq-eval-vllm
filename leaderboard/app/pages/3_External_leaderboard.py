"""External BBQ leaderboard — results published by vendors and by HELM.

Separate from the Home board on purpose. Nothing here was measured by
`bbq_pipeline/`, the sources do not share a setup, and only one of them
publishes a bias score under the same definition this repo uses.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import external

st.set_page_config(page_title="External BBQ leaderboard", page_icon="🌐",
                   layout="wide")
external.require_csv()

st.title("외부 공개 BBQ 결과")

# --- the table switch, directly under the title ----------------------------
boards = external.boards()
options = {external.label(source_id, board): source_id
           for source_id, board in boards.items()}

picked = st.segmented_control(
    "표 선택", list(options), default=next(iter(options)),
    label_visibility="collapsed",
    help="출처마다 프롬프트 형식·shot 수·표본 크기가 달라 서로 비교할 수 없습니다. "
         "한 번에 한 출처만 봅니다.")
if picked is None:  # the control is clearable; keep a board on screen
    picked = next(iter(options))

source_id = options[picked]
board = boards[source_id]

st.caption(external.caption(board))

# --- what this particular source can and cannot tell you --------------------
if board.has_bias:
    st.success(
        "이 표의 `s_DIS` / `s_AMB`는 `bbq_metrics.py`와 **같은 정의**입니다. "
        "Bias B(m)로 순위를 매길 수 있는 유일한 외부 출처입니다. "
        "다만 프롬프트 형식과 채점 경로는 공개되지 않았습니다.")
elif source_id.startswith("helm"):
    st.error(
        "**HELM의 `s_DIS`는 측정값이 아니라 상수 아티팩트라 이 표에서 제외했습니다.** "
        "HELM의 target 매칭이 리스트 원소 완전일치라 BBQ 항목의 36.2%에서 "
        "target을 찾지 못하고, 그 항목들이 분모에만 들어가 모든 모델을 "
        "−0.19…−0.24로 밀어냅니다. 남긴 `s_AMB`도 같은 이유로 0쪽으로 눌린 "
        "하한값입니다. 근거는 `external/README.md`에 있습니다.")
else:
    st.info(
        "이 출처는 **정확도만** 공개합니다. 편향 점수가 없으므로 B(m) 순위를 "
        "매길 수 없고, 위 순위는 `accuracy_disambig` 내림차순일 뿐입니다.")

if source_id.startswith("openai_syscard"):
    st.warning(
        "OpenAI는 카드마다 harness를 바꿨습니다 — GPT-4o가 o1 카드에서는 "
        "0.97 / 0.72, GPT-5 카드에서는 0.88 / 0.85입니다. **카드를 넘나드는 "
        "비교는 하지 마세요.** 그래서 카드별로 표를 나눠 뒀습니다.")

st.html(external.render(board))

# --- abstention caveat, computed live --------------------------------------
if board.has_bias:
    frame = board.frame
    top = frame[frame["eligible"] == 1].sort_values("rank").head(1)
    if not top.empty:
        row = top.iloc[0]
        st.caption(
            f"**{row['model_short']}**가 B = {row['B']:.4f}로 1위지만, "
            f"모호한 맥락의 정답은 “unknown”이라 `accuracy_ambig` = "
            f"{row['accuracy_ambig']:.3f}는 사실상 기권율입니다. 이 점수를 "
            f"의미 있게 만드는 건 `accuracy_disambig` = "
            f"{row['accuracy_disambig']:.3f}이고, 게이트는 {external.GATE}입니다.")

# --- provenance -------------------------------------------------------------
meta = board.meta
with st.expander("출처 · 원본 수치", expanded=False):
    if publisher := meta.get("publisher"):
        st.markdown(f"**{publisher}**")
    for key in ("url", "data_url"):
        if url := meta.get(key):
            st.markdown(f"- {url}")
    if cards := meta.get("cards"):
        st.markdown("**시스템 카드**")
        for name, url in cards.items():
            st.markdown(f"- {name} — {url}")
    if caveat := meta.get("caveat"):
        st.markdown(f"**주의:** {caveat}")

    st.dataframe(
        board.frame[[c for c in ("model_short", "s_DIS", "s_AMB",
                                 "accuracy_disambig", "accuracy_ambig",
                                 "accuracy_overall", "p_not_stereotype_ambig",
                                 "n_eval", "source_ref")
                     if c in board.frame and board.frame[c].notna().any()]],
        hide_index=True, width="stretch")

st.caption(
    "이 페이지의 어떤 숫자도 `bbq_pipeline/`에서 측정한 것이 아닙니다. "
    "직접 돌린 결과는 **Home**에 있고, 두 리더보드는 합치면 안 됩니다 — "
    "이유는 `leaderboard/external/README.md`를 보세요.")

# --- cross-source view, explicitly labelled as not a ranking ---------------
with st.expander("전체 출처 한눈에 보기 (순위 아님)", expanded=False):
    st.markdown(
        "출처가 다르면 세팅이 달라 비교할 수 없습니다. 어떤 모델이 어디에 "
        "실려 있는지 찾는 용도입니다.")
    everything = pd.concat([b.frame.assign(source=source_id)
                            for source_id, b in boards.items()])
    st.dataframe(
        everything[["source", "vendor", "model_short", "s_DIS", "s_AMB",
                    "accuracy_disambig", "accuracy_ambig", "accuracy_overall",
                    "s_DIS_status"]],
        hide_index=True, width="stretch")
