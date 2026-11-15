"""Read `leaderboard/external/` — BBQ results published by other people.

Kept strictly apart from `db.py`: nothing here touches `bbq.sqlite`, and none of
these numbers came out of `bbq_pipeline/`. Each source is its own board because
the setups are not comparable — different prompt formats, shot counts, sample
sizes and, for HELM, a different definition of `s_AMB`. See
`leaderboard/external/README.md`.
"""
from __future__ import annotations

import json

from pathlib import Path

import pandas as pd
import streamlit as st

from . import table
from . import theme as t

EXTERNAL_DIR = Path(__file__).resolve().parents[1] / "external"
CSV_PATH = EXTERNAL_DIR / "bbq_external.csv"
SOURCES_PATH = EXTERNAL_DIR / "sources.json"

#: The eligibility gate from `../README.md`, reused so a board with a published
#: `accuracy_disambig` is read under the same rule as the local runs.
GATE = 0.7

#: Column sets for `table.render`. A board only gets the columns its source
#: actually publishes — an accuracy-only card should not show four em-dash
#: columns where the bias scores would be.
COLS_BIAS = (
    ("#",                            "rank",              True,  "58px", "",      "rank"),
    ("모델",                          "model_short",       True,  "30%",  "",      "model"),
    ("편향 점수\n0에 가까울수록 공정",     "B",                 True,  "12%",  "",      "bias"),
    ("정보가 있을 때 편향",             "s_DIS",             False, "20%",  "",      "diverge"),
    ("정보가 없을 때 편향",             "s_AMB",             False, "20%",  "",      "diverge"),
    ("기본 실력\n정답률",               "accuracy_disambig", False, "9%",   "right", "acc_gate"),
    ('"모르겠다" 비율',                 "accuracy_ambig",    False, "9%",   "right", "acc"),
)

COLS_ACC = (
    ("#",                 "rank",              True,  "58px", "",      "rank"),
    ("모델",               "model_short",       True,  "52%",  "",      "model"),
    ('"모르겠다" 비율',      "accuracy_ambig",    False, "16%",  "right", "acc"),
    ("기본 실력\n정답률",    "accuracy_disambig", False, "16%",  "right", "acc"),
)

COLS_ACC_P = COLS_ACC[:2] + (
    ('"모르겠다" 비율',            "accuracy_ambig",         False, "13%", "right", "acc"),
    ("기본 실력\n정답률",          "accuracy_disambig",      False, "13%", "right", "acc"),
    ("고정관념을 피한 비율",        "p_not_stereotype_ambig", True,  "18%", "right", "num"),
)

COLS_HELM = (
    ("#",                  "rank",             True,  "58px", "",      "rank"),
    ("모델",                "model_short",      True,  "42%",  "",      "model"),
    ("정보가 없을 때 편향",   "s_AMB",            False, "34%",  "",      "diverge"),
    ("정답률",              "accuracy_overall", False, "16%",  "right", "acc"),
)


class Board:
    """One source's table, plus how to render and read it."""

    def __init__(self, source_id: str, frame: pd.DataFrame, meta: dict):
        self.source_id = source_id
        self.frame = frame
        self.meta = meta

    @property
    def has_bias(self) -> bool:
        return self.meta.get("s_DIS_status") == "ok"

    @property
    def columns(self):
        if self.has_bias:
            return COLS_BIAS
        if self.source_id.startswith("helm"):
            return COLS_HELM
        if self.frame["p_not_stereotype_ambig"].notna().any():
            return COLS_ACC_P
        return COLS_ACC

    @property
    def ranked_by(self) -> str:
        if self.has_bias:
            return "B(m) = |s_DIS| + |s_AMB|, 오름차순"
        if self.source_id.startswith("helm"):
            return "|s_AMB|, 오름차순 — s_DIS는 신뢰할 수 없어 순위에서 제외"
        return "accuracy_disambig, 내림차순 — 편향 순위가 아닙니다"


@st.cache_data(show_spinner=False)
def _load(mtime: float) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(CSV_PATH)
    sources = json.loads(SOURCES_PATH.read_text())["sources"]
    return frame, sources


def _mtime() -> float:
    return CSV_PATH.stat().st_mtime if CSV_PATH.exists() else 0.0


def require_csv() -> None:
    """Render a build hint instead of a traceback when the table is absent."""
    if CSV_PATH.exists() and SOURCES_PATH.exists():
        return
    st.error("No external table yet.")
    st.markdown("`leaderboard/external/bbq_external.csv` is built from `external/raw/`:")
    st.code("python leaderboard/external/build_external.py", language="bash")
    st.stop()


def boards() -> dict[str, Board]:
    """One `Board` per source, in a stable display order."""
    frame, sources = _load(_mtime())
    order = ["anthropic_syscard", "openai_syscard_gpt5", "openai_syscard_o3",
             "openai_syscard_gpt45", "openai_syscard_o1",
             "google_gemma2", "google_gemma1", "helm_safety_v1.17.0"]
    present = [s for s in order if s in set(frame["source_id"])]
    present += [s for s in frame["source_id"].unique() if s not in present]

    out: dict[str, Board] = {}
    for source_id in present:
        out[source_id] = Board(source_id,
                               _prepare(frame[frame["source_id"] == source_id].copy(),
                                        sources.get(source_id, {})),
                               sources.get(source_id, {}))
    return out


def _prepare(frame: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Shape one source into the frame `table.render` expects."""
    frame = frame.rename(columns={"model": "model_short"})
    frame["chip"] = frame["source_ref"].where(frame["source_ref"].notna(), "")

    if meta.get("s_DIS_status") == "ok":
        frame["B"] = frame["s_DIS"].abs() + frame["s_AMB"].abs()
        # Same gate as the local board: a bias score is only readable when the
        # model can actually answer the disambiguated questions.
        frame["eligible"] = (frame["accuracy_disambig"] >= GATE).astype(int)
        frame = frame.sort_values(["eligible", "B"], ascending=[False, True])
    elif str(meta.get("s_AMB_status")) == "degraded":
        frame["eligible"] = 1
        frame = frame.sort_values("s_AMB", key=lambda s: s.abs())
    else:
        frame["eligible"] = 1
        frame = frame.sort_values("accuracy_disambig", ascending=False)

    ranks = range(1, int((frame["eligible"] == 1).sum()) + 1)
    frame["rank"] = pd.Series(list(ranks) + [pd.NA] * int((frame["eligible"] == 0).sum()),
                              index=frame.index, dtype="Float64")
    return frame


def label(source_id: str, board: Board) -> str:
    """The switch-button caption."""
    short = {
        "anthropic_syscard": "Anthropic 시스템 카드",
        "openai_syscard_gpt5": "OpenAI · GPT-5 카드",
        "openai_syscard_o3": "OpenAI · o3 / o4-mini 카드",
        "openai_syscard_gpt45": "OpenAI · GPT-4.5 카드",
        "openai_syscard_o1": "OpenAI · o1 카드",
        "google_gemma2": "Google · Gemma 2",
        "google_gemma1": "Google · Gemma 1",
        "helm_safety_v1.17.0": "HELM Safety",
    }.get(source_id, source_id)
    return f"{short} ({len(board.frame)})"


def render(board: Board) -> str:
    return table.render(board.frame, threshold=GATE, columns=board.columns)


def caption(board: Board) -> str:
    parts = [f"**정렬:** {board.ranked_by}"]
    if setup := board.meta.get("setup"):
        parts.append(f"**세팅:** {setup}")
    return "  \n".join(parts)


__all__ = ["Board", "boards", "caption", "label", "render", "require_csv",
           "GATE", "t"]
