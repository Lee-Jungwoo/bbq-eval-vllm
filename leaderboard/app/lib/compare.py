"""Put the 3-option board and the binary (A/B) board side by side.

The two arms are deliberately kept apart everywhere else — `publish.py` refuses
binary runs, `bbq.sqlite` is 3-option only, and `lib/binary.py` reads the binary
summaries straight off disk. This module is the one place they meet, and it does
nothing but *join and rank*: every number comes from `db.leaderboard()` or
`binary.board()` unchanged, so the comparison cannot drift from either board.

What makes the comparison legitimate, and what does not:

  `s_AMB` is **not** the same quantity in the two arms. The 3-option arm scales
  it by `(1 - accuracy_ambig)`, and with "unknown" as the gold answer in an
  ambiguous context that factor is essentially the abstention rate — so a model
  that declines to answer is scored as unbiased. The binary arm has no unknown
  option, so its `s_AMB` is already unscaled. The comparable pair is therefore
  3-option `s_AMB_unscaled` ↔ binary `s_AMB`, which is what `RANK_BASES` offers
  alongside the two boards' own published rankings.

Ranks are recomputed here rather than read from either board, because a rank is
only meaningful within a fixed field: a model missing from one arm, or gated out
of it, would otherwise shift every rank below it and manufacture movement that
is not there. Only models eligible in *both* arms are ranked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import binary, db
from . import table
from . import theme as t

#: Ranking basis → (3-option column, binary column, what it means).
#:
#: ``B(m)`` reproduces what the two pages actually publish, so it answers "does
#: the leaderboard reorder?" — but the 3-option side carries the abstention
#: scaling, so part of any movement is abstention, not bias. ``B*`` is the same
#: formula with that scaling removed from the 3-option side, which is the
#: like-for-like ordering. ``s_AMB`` isolates the ambiguous half alone.
RANK_BASES: dict[str, tuple[str, str, str]] = {
    "두 보드가 실제로 게시하는 순위": (
        "B", "B",
        "각 쪽의 공식 순위입니다. 다만 3지선다 점수에는 **회피한 만큼 편향을 깎아 "
        "주는 보정**이 들어 있어, 순위 변동의 일부는 편향이 아니라 회피율 "
        "차이입니다."),
    "회피 보정을 걷어낸 순위 (기본값)": (
        "B_star", "B",
        "3지선다 쪽의 회피 보정만 제거해 두 쪽을 같은 층위로 맞춘 값입니다. "
        "**순위 비교에는 이 기준이 가장 정확합니다.**"),
    "정보가 없는 문항만": (
        "s_AMB_unscaled", "s_AMB",
        "맥락에 단서가 없는 문항의 편향만 떼어 본 순위입니다."),
}

#: Columns carried over from each board, renamed with an arm prefix on the join.
_TRI = ("rank", "eligible", "B", "B_lo", "B_hi", "s_DIS", "s_AMB",
        "s_AMB_unscaled", "accuracy_ambig", "accuracy_disambig",
        "unknown_rate_ambig", "n_elig_ambig", "s_answered")
_BIN = ("rank", "eligible", "B", "B_lo", "B_hi", "s_DIS", "s_AMB",
        "accuracy_disambig", "target_rate_ambig", "letter_A_rate")


def frame(tri_key: str, bin_key: str) -> pd.DataFrame:
    """Inner join of the two boards on `model_short`, one row per model.

    Inner on purpose: a model run in only one arm has nothing to compare and
    would only pad the field.
    """
    tri = db.leaderboard(tri_key)
    bin_ = binary.board(bin_key)
    if tri.empty or bin_.empty:
        return pd.DataFrame()

    # |s_DIS| + |s_AMB unscaled| — the binary arm's B with the 3-option arm's
    # abstention scaling taken back out, so the two sides mean the same thing.
    tri = tri.assign(B_star=tri["s_DIS"].abs() + tri["s_AMB_unscaled"].abs())

    left = tri[["model_short", "model", "family", "params_b", "run_id",
                *_TRI, "B_star"]].rename(
        columns={c: f"tri_{c}" for c in (*_TRI, "B_star", "run_id")})
    right = bin_[["model_short", *_BIN, "run_id"]].rename(
        columns={c: f"bin_{c}" for c in (*_BIN, "run_id")})

    merged = left.merge(right, on="model_short", how="inner")
    # A rank is unreadable on either side unless the model can answer the
    # disambiguated questions in *both* arms.
    merged["eligible"] = (merged["tri_eligible"].astype(int)
                          & merged["bin_eligible"].astype(int))
    return merged


def rank(merged: pd.DataFrame, basis: str) -> pd.DataFrame:
    """Add `rank_tri` / `rank_bin` / `rank_delta` for one ranking basis.

    Ranked ascending by magnitude among the models eligible in both arms; gated
    models get a null rank rather than a position, exactly as the boards do.
    `rank_delta = rank_tri - rank_bin`, so **positive means the model moved up**
    (toward rank 1) when the unknown option was removed.
    """
    tri_column, bin_column, _ = RANK_BASES[basis]
    out = merged.copy()
    eligible = out["eligible"] == 1

    for side, column in (("tri", f"tri_{tri_column}"), ("bin", f"bin_{bin_column}")):
        values = pd.to_numeric(out[column], errors="coerce").abs()
        out[f"rank_{side}"] = (values.where(eligible)
                               .rank(method="min", ascending=True))
        out[f"metric_{side}"] = pd.to_numeric(out[column], errors="coerce")

    out["rank_delta"] = out["rank_tri"] - out["rank_bin"]
    return out


def columns(basis: str) -> tuple[tuple[str, str, bool, str, str, str], ...]:
    """`table.render` column spec for one basis.

    Ordered so a reader can scan it in one pass: the 3-option rank is the spine,
    then the model, then the whole move to the binary rank in a single cell, then
    the metric that produced the move on each side, then the diagnostic that
    explains most of it.
    """
    tri_column, bin_column, _ = RANK_BASES[basis]
    return (
        ("#\n3지선다 기준",  "rank_tri",              True,  "56px", "",      "rank"),
        ("모델",             "model_short",           True,  "23%",  "",      "model"),
        ("순위 이동\n3지선다 → 2지선다", "rank_delta",  False, "23%",  "",      "rankshift"),
        (f"{_label(tri_column)}\n3지선다", f"tri_{tri_column}",
         False, "19%", "", "diverge"),
        (f"{_label(bin_column)}\n2지선다 (강제 선택)", f"bin_{bin_column}",
         False, "19%", "", "diverge"),
        ('"모르겠다" 비율\n3지선다',  "tri_accuracy_ambig",    False, "8%",  "right", "acc"),
        ("기본 실력\n3지선다",       "tri_accuracy_disambig", False, "8%",  "right", "acc_gate"),
        ("기본 실력\n2지선다",       "bin_accuracy_disambig", False, "8%",  "right", "acc_gate"),
    )


def sort_options(basis: str) -> dict[str, tuple[str, bool]]:
    """헤더 라벨 → (정렬할 컬럼, 오름차순). 사이드바의 "정렬 기준" 목록.

    표의 열 정의에서 그대로 만들어 둘이 어긋날 수 없게 하고, 2지선다 순위만
    따로 더한다 — 순위 이동이 한 칸에 들어가면서 자기 열은 없어졌지만 그것으로
    정렬하고 싶은 사람은 여전히 있다.
    """
    return table.sort_options(columns(basis), skip=("model_short",)) | {
        "# 2지선다 기준": ("rank_bin", True)}


def _label(column: str) -> str:
    return {"B": "편향 점수",
            "B_star": "편향 점수 (보정)",
            "s_AMB_unscaled": "굳이 답할 때 편향",
            "s_AMB": "정보 없을 때 편향"}.get(column, column)


def render(merged: pd.DataFrame, basis: str) -> str:
    return table.render(merged, threshold=binary.GATE, columns=columns(basis))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def abstention_gap(merged: pd.DataFrame) -> go.Figure:
    """Why the two arms disagree: the gap against the 3-option abstention rate.

    One series, so no legend — the title names it. `accuracy_ambig` is the
    abstention rate (the gold answer in an ambiguous context *is* "unknown"), and
    the gap is what the 3-option arm's small non-unknown denominator does to
    `s_AMB_unscaled`.
    """
    sub = merged.dropna(subset=["tri_accuracy_ambig", "tri_s_AMB_unscaled",
                                "bin_s_AMB"]).copy()
    sub["gap"] = sub["tri_s_AMB_unscaled"].abs() - sub["bin_s_AMB"].abs()

    fig = go.Figure(go.Scatter(
        x=sub["tri_accuracy_ambig"], y=sub["gap"], mode="markers",
        marker=dict(size=11, color=t.AMBIG,
                    line=dict(width=2, color=t.SURFACE)),
        text=sub["model_short"],
        customdata=np.stack([sub["tri_s_AMB_unscaled"], sub["bin_s_AMB"],
                             sub["tri_n_elig_ambig"]], axis=-1),
        hovertemplate=("<b>%{text}</b><br>회피율 %{x:.1%}<br>"
                       "3지선다 · 굳이 답할 때 편향 = %{customdata[0]:+.4f}<br>"
                       "2지선다 · 정보 없을 때 편향 = %{customdata[1]:+.4f}<br>"
                       "격차 %{y:+.4f}  ·  답한 문항 %{customdata[2]:,.0f}개"
                       "<extra></extra>")))
    fig.add_hline(y=0, line=dict(color=t.AXIS, width=1, dash="dash"))
    return t.apply(
        fig, height=420,
        title="3지선다가 편향을 과대평가하는 정도 vs 회피율",
        xaxis_title="3지선다에서 “모르겠다”를 고른 비율",
        yaxis_title="3지선다 − 2지선다 (편향 크기 차이)",
        xaxis=dict(tickformat=".0%"))


__all__ = ["RANK_BASES", "abstention_gap", "columns", "frame", "rank",
           "render", "sort_options"]
