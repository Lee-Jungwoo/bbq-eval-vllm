"""Per-model breakdown: category, polarity, margin, abstention, provenance.

One question carries the page — *which social attribute does this model lean on?*
— so the category chart is the page, and everything that answers a different
question (decisiveness, run provenance) sits in one expander at the bottom.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import charts, db, explain
from lib import theme as t

st.set_page_config(page_title="Model detail · BBQ", page_icon="🔍", layout="wide")
db.require_db()

runs = db.runs()
if runs.empty:
    st.warning("아직 게시된 런이 없습니다.")
    st.stop()

labels = {row.run_id: f"{row.family} · {row.model_short}"
          for row in runs.itertuples(index=False)}
options = list(labels)
# Default to the model that answered the most ambiguous items rather than the
# top-ranked one: rank 1 abstained on ~99% of them, so every per-category bar
# would be a stub. The most talkative model is the one this page can show.
talkative = db.q("""SELECT run_id FROM leaderboard WHERE eligible = 1
                    ORDER BY n_elig_ambig DESC LIMIT 1""")
default = st.query_params.get("run_id")
if default not in labels:
    default = talkative["run_id"].iloc[0] if not talkative.empty else options[0]
index = options.index(default)

explain.page_header(
    "모델 하나를 뜯어보면",
    "같은 모델이라도 <b>어느 사회적 속성</b>에서 편향이 크게 나오는지는 다릅니다.")

run_id = st.selectbox("모델 선택", options, index=index,
                      format_func=lambda r: labels[r], label_visibility="collapsed")
with st.sidebar:
    with st.expander("표 설정", expanded=False):
        db.sidebar_footer(st)
st.query_params["run_id"] = run_id

overall = db.run_metrics(run_id, "overall").iloc[0]
provenance = db.provenance(run_id)

# Compare against the ranked field rather than against nothing.
board = db.leaderboard(provenance["comparability_key"])
peers = board[(board["eligible"] == 1) & (board["run_id"] != run_id)]
row = board[board["run_id"] == run_id]
row = row.iloc[0] if not row.empty else None


def delta_vs_peers(value, column: str) -> str | None:
    if peers.empty or value is None or pd.isna(value):
        return None
    median = peers[column].median()
    if pd.isna(median):
        return None
    return f"다른 모델 중간값 대비 {(value - median) * 100:+.1f}%p"


cols = st.columns(3)
cols[0].metric("기본 실력 (정답률)", f"{overall['accuracy_disambig']:.1%}",
               delta_vs_peers(overall["accuracy_disambig"], "accuracy_disambig"),
               delta_color="off")
cols[1].metric("“모르겠다” 비율", f"{overall['accuracy_ambig']:.1%}",
               delta_vs_peers(overall["accuracy_ambig"], "accuracy_ambig"),
               delta_color="off")
if row is not None:
    cols[2].metric("편향 점수", f"{row['B']:.4f}",
                   f"{int(row['rank'])}위" if pd.notna(row["rank"])
                   else "정답률이 낮아 순위 제외", delta_color="off")

if row is not None and row["abstention_tier"] != "none":
    st.warning(
        f"이 모델은 정보가 없는 문항의 **{row['unknown_rate_ambig']:.1%}**를 "
        f"“모르겠다”로 넘겼습니다. {int(overall['n_ambig']):,}개 중 실제로 누군가를 "
        f"지목한 건 **{int(row['n_elig_ambig']):,}개**뿐이라 아래 속성별 막대는 "
        f"적은 답 위에서 계산됩니다 (경고: "
        f"*{t.TIER_LABEL.get(row['abstention_tier'], row['abstention_tier'])}*).")

by_category = db.run_metrics(run_id, "category")
order = db.category_order(run_id)
by_category = by_category.set_index("scope_value").loc[order].reset_index()
by_category["scope_value"] = by_category["scope_value"].map(t.category_ko)

tabs = st.tabs(["속성별", "질문 유형별"])

with tabs[0]:
    # Both series are undamped — `s_AMB` itself collapses toward zero for an
    # abstaining model, which would draw a flat chart and say nothing about
    # which attribute the model leans on.
    st.plotly_chart(charts.category_bars(
        by_category, ("s_DIS", "s_AMB_unscaled"),
        ("정보가 있을 때", "정보가 없을 때 (답한 것만)"),
        "어느 사회적 속성에서 고정관념 쪽으로 기우나  (오른쪽 = 고정관념)",
        hline=0.0, xaxis_title="편향"), width="stretch")
    smallest = int(by_category["n_elig_ambig"].min())
    st.caption(
        "두 막대 모두 **회피 보정을 걷어낸 값**입니다 — 공식 점수(`정보가 없을 때 "
        "편향`)는 회피한 만큼 0쪽으로 눌려 속성 간 차이가 보이지 않기 때문입니다. "
        f"답이 가장 적은 속성은 {smallest:,}개뿐이라 그 막대는 흔들릴 수 "
        "있습니다 — 신뢰구간은 아래 접힌 칸에 있습니다.")

    with st.expander("정답률 · 신뢰구간 · 전체 수치", expanded=False):
        left, right = st.columns(2)
        with left:
            st.plotly_chart(charts.category_bars(
                by_category, ("accuracy_ambig", "accuracy_disambig"),
                ('"모르겠다" 비율', "기본 실력"), "속성별 정답률",
                hline=1 / 3, hline_label="찍기 (1/3)",
                xaxis_title="비율"), width="stretch")
        with right:
            st.plotly_chart(
                charts.s_answered_by_category(by_category, t.MIN_N_ANSWERED),
                width="stretch")
        st.caption(
            f"오른쪽 막대는 95% 신뢰구간이고, 속이 빈 점은 답이 "
            f"{t.MIN_N_ANSWERED}개 미만이라 읽을 수 없다는 뜻입니다.")
        st.dataframe(
            by_category[["scope_value", "n", "accuracy_disambig", "accuracy_ambig",
                         "s_DIS", "s_AMB", "s_AMB_unscaled", "n_elig_ambig"]]
            .rename(columns={
                "scope_value": "속성",
                "n": "문항 수",
                "accuracy_disambig": "기본 실력",
                "accuracy_ambig": '"모르겠다" 비율',
                "s_DIS": "정보가 있을 때 편향",
                "s_AMB": "정보가 없을 때 편향",
                "s_AMB_unscaled": "굳이 답할 때 편향",
                "n_elig_ambig": "답한 문항 수"}),
            hide_index=True, width="stretch")

with tabs[1]:
    cells = db.cell_metrics(run_id, "overall")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.polarity_panels(
            cells, "accuracy", "맥락 × 질문 유형별 정답률"), width="stretch")
    with right:
        st.plotly_chart(charts.polarity_panels(
            cells, "s", "맥락 × 질문 유형별 편향 점수", zeroline=True), width="stretch")
    st.caption("`neg` 는 부정적인 고정관념에 맞는 집단을 묻는 질문(“누가 휴대폰을 "
               "불편해했나?”), `nonneg` 는 그 거울상(“누가 가르쳐 주고 있었나?”)"
               "입니다. 두 방향의 답이 서로 맞물리는지가 이 탭에서 보입니다.")
    st.dataframe(
        cells[["context_condition", "question_polarity", "n", "accuracy",
               "unknown_rate", "n_elig", "n_biased", "s_raw", "s"]]
        .rename(columns={
            "context_condition": "맥락",
            "question_polarity": "질문 유형",
            "n": "문항 수",
            "accuracy": "정답률",
            "unknown_rate": '"모르겠다" 비율',
            "n_elig": "답한 문항 수",
            "n_biased": "고정관념 쪽 답",
            "s_raw": "편향 (원값)",
            "s": "편향 (공식)"}),
        hide_index=True, width="stretch")

# --- everything that answers a different question --------------------------
with st.expander("기술 상세 — 확신 정도 · 회피 · 실행 정보", expanded=False):
    detail = st.tabs(["확신 정도", "회피", "실행 정보"])

    with detail[0]:
        hist = db.margin_hist(run_id)
        if hist.empty:
            st.info("이 런에는 margin 데이터가 없습니다.")
        else:
            left, right = st.columns(2)
            with left:
                st.plotly_chart(charts.margin_bars(hist), width="stretch")
            with right:
                st.plotly_chart(charts.margin_ecdf(hist), width="stretch")
            st.caption("`margin` 은 1·2위 선택지의 로그확률 차이 — 모델이 얼마나 "
                       "단호하게 골랐는지입니다. 세 선택지 중 둘이 서버의 top-k "
                       "밖으로 밀리면 값이 비어 있습니다.")
            st.dataframe(pd.DataFrame([{
                "p25": overall["margin_p25"], "중앙값": overall["margin_median"],
                "p75": overall["margin_p75"],
                "중앙값 (정답)": overall["margin_median_correct"],
                "중앙값 (오답)": overall["margin_median_incorrect"],
                "값 없음": int(overall["n_margin_null"]),
            }]), hide_index=True, width="stretch")

    with detail[1]:
        st.plotly_chart(charts.abstention_composition(by_category),
                        width="stretch")
        st.caption("정보가 없는 문항을 어떻게 답했는지의 구성입니다 — 회색이 "
                   "“모르겠다”, 주황이 고정관념 쪽, 파랑이 반대 쪽입니다.")

    with detail[2]:
        left, right = st.columns(2)
        with left:
            st.markdown("**모델**")
            st.write({
                "model": provenance["model"],
                "revision": provenance.get("model_revision"),
                "dtype": provenance.get("dtype"),
                "params (B)": provenance.get("params_b"),
                "run_id": run_id,
            })
            st.markdown("**채점 설정**")
            st.write({
                "prompt_hash": provenance["prompt_hash"],
                "comparability_key": provenance["comparability_key"],
                "option_letters": provenance["option_letters"],
                "label_token_ids": provenance["label_token_ids"],
                "label_token_variant": provenance.get("label_token_variant"),
                "max_question_index": provenance["max_question_index"],
            })
        with right:
            st.markdown("**토크나이즈**")
            backend = provenance.get("tokenizer_backend")
            if backend:
                st.write({
                    "tokenizer_backend": backend,
                    "prompt tokens": f"{provenance['prompt_token_len_min']}–"
                                     f"{provenance['prompt_token_len_max']}",
                })
                st.markdown("**프롬프트 예시**")
                st.code(provenance.get("sample_prompt") or "", language="text")
            else:
                # Absent, not empty: these fields were added after this run was made.
                st.info("기록되지 않았습니다 — 이 런은 `tokenizer_backend` / "
                        "`prompt_token_len` / `sample_prompt` 필드가 "
                        "meta.json 에 추가되기 전에 만들어졌습니다.")
            st.markdown("**데이터 품질**")
            st.write({
                "mean prob mass captured": round(overall["mean_prob_mass_captured"], 6),
                "min prob mass captured": round(overall["min_prob_mass_captured"], 6),
                "라벨 로그확률이 빠진 레코드": int(overall["n_null_logprob_records"]),
                "target_loc 메타데이터가 없는 레코드": int(overall["n_no_target_metadata"]),
            })

st.caption(f"{provenance['model']} · {provenance['dtype']} · "
           f"revision `{(provenance.get('model_revision') or '?')[:12]}` · "
           f"문항 {int(overall['n']):,}개")
