"""The leaderboard table, rendered as HTML.

`st.dataframe` cannot show a logo next to a text label, and its cell bars are
tied to a single column's range. This renderer keeps the same numbers and the
same colour meanings as the rest of the app — orange = more stereotyped, blue =
less — and lets the bars share the width the table gains from `layout="wide"`.
"""
from __future__ import annotations

import html

import pandas as pd

from . import logos
from . import theme as t

#: Rank badge colours for the podium; everything else gets the neutral badge.
PODIUM = {1: ("#b8860b", "#fdf3d3"), 2: ("#5f6773", "#eceef0"), 3: ("#9a5b2c", "#f7e6d8")}

#: The columns, in display order: header label, the frame column that label
#: sorts on, whether that sort is ascending, its share of the table width, the
#: alignment class, and the cell renderer. The header row, the `<colgroup>`, the
#: body and the sidebar's sort list are all built from this, so they cannot
#: drift apart.
#:
#: A label may carry a second line after `\n`: the column's name on top, how to
#: read it underneath, in small grey type. The sort list keeps only the first
#: line. Labels are the words a reader who has never seen BBQ can follow — the
#: underlying variable names live in `bbq_metrics.py` and in the expanders.
#:
#: Cell renderers: ``rank`` podium badge · ``model`` logo + name + chips ·
#: ``bias`` bold B(m) · ``ci`` the B_lo/B_hi interval · ``diverge`` signed bar
#: centred at zero · ``delta`` signed rank change, arrow + magnitude ·
#: ``rankshift`` `rank_tri`→`rank_bin` on a shared 1..N track ·
#: ``acc`` plain 3dp · ``acc_gate`` 3dp, reddened below the eligibility
#: threshold · ``num`` plain 3dp for anything else.
COLUMNS: tuple[tuple[str, str, bool, str, str, str], ...] = (
    ("#",                            "rank",              True,  "58px", "",      "rank"),
    ("모델",                          "model_short",       True,  "28%",  "",      "model"),
    ("편향 점수\n0에 가까울수록 공정",     "B",                 True,  "11%",  "",      "bias"),
    ('"모르겠다" 비율\n높을수록 점수가 좋아짐', "accuracy_ambig", False, "10%",  "",      "acc"),
    ("정보가 있을 때 편향\n주황 = 고정관념 쪽", "s_DIS",         False, "21%",  "",      "diverge"),
    ("정보가 없을 때 편향\n주황 = 고정관념 쪽", "s_AMB",         False, "21%",  "",      "diverge"),
    ("기본 실력\n정답률",               "accuracy_disambig", False, "9%",   "right", "acc_gate"),
)


def sort_options(columns, *, skip: tuple[str, ...] = ()) -> dict[str, tuple[str, bool]]:
    """헤더 라벨 → (정렬할 컬럼, 오름차순). 사이드바의 "정렬 기준" 목록.

    라벨의 둘째 줄(힌트)은 표에만 쓰고 정렬 목록에서는 뗀다. 첫 줄이 겹치면 —
    비교 페이지처럼 두 arm 이 같은 지표를 나란히 놓을 때 — 힌트를 붙여 구분한다.
    """
    out: dict[str, tuple[str, bool]] = {}
    for label, column, ascending, _, _, _ in columns:
        if column in skip:
            continue
        head, _, hint = label.partition("\n")
        key = head if head not in out else f"{head} · {hint}".strip(" ·")
        out[key] = (column, ascending)
    return out


#: 게시된 순위가 기본값이라 공식 순서를 그대로 재현한다.
SORT_OPTIONS: dict[str, tuple[str, bool]] = sort_options(COLUMNS)

CSS = f"""
<style>
/* the table is the page — let it use the full width of the main column */
[data-testid="stMainBlockContainer"], .block-container {{
    max-width: 100% !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
}}
.bbq-board {{
    width: 100%;
    overflow-x: auto;
    border: 1px solid {t.GRID};
    border-radius: 12px;
    box-shadow: 0 1px 2px rgba(31, 35, 40, .06), 0 6px 18px rgba(31, 35, 40, .04);
    background: {t.SURFACE};
    margin-bottom: 1.25rem;
}}
table.bbq {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 13.5px;
    color: {t.INK};
    font-variant-numeric: tabular-nums;
}}
table.bbq thead th {{
    position: sticky; top: 0; z-index: 2;
    background: #fafbfc;
    border-bottom: 1px solid {t.GRID};
    padding: 11px 14px;
    text-align: left;
    font-size: 11.5px;
    font-weight: 600;
    letter-spacing: .03em;
    color: {t.MUTED};
    white-space: nowrap;
}}
table.bbq thead th .hint {{
    display: block; margin-top: 2px;
    font-size: 10.5px; font-weight: 500; letter-spacing: 0;
    color: {t.AXIS}; text-transform: none;
}}
table.bbq th.right, table.bbq td.right {{ text-align: right; }}
table.bbq tbody td {{
    padding: 9px 14px;
    border-bottom: 1px solid #f0f1f2;
    vertical-align: middle;
    white-space: nowrap;
}}
table.bbq tbody tr:last-child td {{ border-bottom: 0; }}
table.bbq tbody tr:hover td {{ background: #f7f9fc; }}
table.bbq tbody tr.gated td {{ background: #fcfcfd; color: {t.MUTED}; }}
table.bbq tbody tr.gated:hover td {{ background: #f5f6f7; }}
table.bbq tbody tr.gated .name {{ color: {t.MUTED}; font-weight: 500; }}
table.bbq tr.first-gated td {{ border-top: 1px dashed {t.AXIS}; }}

.badge {{
    display: inline-flex; align-items: center; justify-content: center;
    min-width: 26px; height: 24px; padding: 0 7px;
    border-radius: 7px; font-size: 12.5px; font-weight: 700;
    background: #f1f3f5; color: {t.MUTED};
}}
.model {{ display: flex; align-items: center; gap: 10px; }}
.logo {{
    width: 26px; height: 26px; border-radius: 7px; flex: 0 0 26px;
    object-fit: contain; background: #fff; border: 1px solid {t.GRID};
}}
.logo.blank {{ background: #f1f3f5; }}
.name {{ font-weight: 600; letter-spacing: -.01em; }}
.sub {{ font-size: 11px; color: {t.MUTED}; margin-top: 1px; }}
.chip {{
    display: inline-block; padding: 1px 7px; border-radius: 999px;
    font-size: 10.5px; font-weight: 600; letter-spacing: .02em;
    border: 1px solid {t.GRID}; color: {t.MUTED}; background: #fafbfc;
}}
.chip.gate {{ border-color: #f2c7b4; background: #fdf1ec; color: #b5470f; }}

/* rank change between two boards: arrow + magnitude, never colour alone */
.delta {{ font-weight: 700; font-size: 12.5px; }}
.delta.up {{ color: {t.COUNTER}; }}
.delta.down {{ color: {t.BIASED}; }}
.delta.flat {{ color: {t.MUTED}; font-weight: 600; }}

/* one row's move across the whole 1..N field, on a track every row shares */
.rshift {{ display: flex; align-items: center; gap: 9px; min-width: 168px; }}
.rshift .pair {{
    min-width: 62px; font-weight: 600; font-size: 12px; color: {t.MUTED};
}}
.rshift .pair b {{ color: {t.INK}; font-weight: 700; }}
.rtrack {{
    position: relative; flex: 1 1 auto; height: 7px; min-width: 62px;
    border-radius: 999px; background: #eef0f2;
}}
.rseg {{ position: absolute; top: 0; height: 100%; border-radius: 999px; }}
.rdot {{ position: absolute; top: 50%; border-radius: 50%; }}
.rdot.from {{
    width: 11px; height: 11px; margin: -5.5px 0 0 -5.5px;
    background: {t.SURFACE}; border: 2px solid {t.AXIS};
}}
.rdot.to {{
    width: 11px; height: 11px; margin: -5.5px 0 0 -5.5px;
    border: 2px solid {t.SURFACE};
}}
.rshift .delta {{ min-width: 34px; text-align: right; }}

.bias {{ font-weight: 700; font-size: 14.5px; }}
.diverge {{ display: flex; align-items: center; gap: 10px; min-width: 130px; }}
.diverge .num {{ min-width: 54px; font-weight: 600; }}
.dtrack {{
    position: relative; flex: 1 1 auto; height: 7px; min-width: 54px;
    border-radius: 999px; background: #eef0f2;
}}
.dtrack::before {{
    content: ""; position: absolute; left: 50%; top: -2px;
    width: 1px; height: 11px; background: {t.AXIS};
}}
.dfill {{ position: absolute; top: 0; height: 100%; border-radius: 999px; }}
.pos {{ color: {t.BIASED}; }}
.neg {{ color: {t.COUNTER}; }}
.ci {{ color: {t.MUTED}; font-size: 12px; }}
.acc {{ font-weight: 600; }}
.acc.low {{ color: {t.BIASED}; }}
</style>
"""


def _header_cell(label: str, align: str) -> str:
    """`<th>`. 라벨의 둘째 줄은 그 열을 어떻게 읽는지 알려주는 작은 힌트가 된다."""
    head, _, hint = label.partition("\n")
    inner = html.escape(head)
    if hint:
        inner += f'<span class="hint">{html.escape(hint)}</span>'
    return f'<th class="{align}">{inner}</th>'


def _fmt(value, spec: str) -> str:
    if value is None or value != value:
        return "—"
    return format(float(value), spec)


def _diverge(value, vmax: float) -> str:
    """Signed value on a bar centred at zero: right/orange = more stereotyped."""
    if value is None or value != value:
        return '<div class="diverge"><span class="num">—</span></div>'
    value = float(value)
    half = 0.0 if vmax <= 0 else min(abs(value) / vmax, 1.0) * 50
    side = "left:50%" if value >= 0 else "right:50%"
    colour = t.BIASED if value >= 0 else t.COUNTER
    css_class = "pos" if value >= 0 else "neg"
    return (f'<div class="diverge">'
            f'<span class="num {css_class}">{value:+.4f}</span>'
            f'<span class="dtrack"><span class="dfill" style="{side};'
            f'width:{half:.1f}%;background:{colour}"></span></span></div>')


def _delta(value) -> str:
    """Rank change as arrow + magnitude.

    Sign convention is the caller's: positive means the row moved *up* (toward
    rank 1). Colour keeps the app-wide meaning — blue = less stereotyped, orange
    = more — and the arrow repeats it, so direction is never colour alone.
    """
    if value is None or value != value:
        return '<span class="delta flat">—</span>'
    value = int(value)
    if value == 0:
        return '<span class="delta flat">=</span>'
    arrow = "\u25b2" if value > 0 else "\u25bc"
    return f'<span class="delta {"up" if value > 0 else "down"}">{arrow} {abs(value)}</span>'


def _rankshift(row: dict, rank_max: float) -> str:
    """`rank_tri` → `rank_bin` as a move along the shared 1..N field.

    Reads both rank columns directly, the way `ci` reads `B_lo`/`B_hi`: the cell
    is one movement, not one number. Every row's track spans the same 1..N, so
    the bar lengths are comparable down the column — that is the whole point of
    putting it in the table instead of in a chart beside it.
    """
    start, end = row.get("rank_tri"), row.get("rank_bin")
    if any(v is None or v != v for v in (start, end)):
        return '<div class="rshift"><span class="pair">—</span></div>'
    start, end = float(start), float(end)

    span = max(float(rank_max) - 1, 1.0)
    from_pct = min(max((start - 1) / span, 0.0), 1.0) * 100
    to_pct = min(max((end - 1) / span, 0.0), 1.0) * 100
    colour = t.COUNTER if end < start else t.BIASED if end > start else t.UNKNOWN
    left, right = sorted((from_pct, to_pct))

    return (f'<div class="rshift">'
            f'<span class="pair">{start:.0f} &rarr; <b>{end:.0f}</b></span>'
            f'<span class="rtrack">'
            f'<span class="rseg" style="left:{left:.2f}%;width:{right - left:.2f}%;'
            f'background:{colour}"></span>'
            f'<span class="rdot from" style="left:{from_pct:.2f}%"></span>'
            f'<span class="rdot to" style="left:{to_pct:.2f}%;'
            f'background:{colour}"></span>'
            f'</span>{_delta(row.get("rank_delta"))}</div>')


def _model_cell(row) -> str:
    # External rows carry a publisher name instead of an HF org and a family.
    if row.get("vendor"):
        uri, vendor_name = logos.logo_for_vendor(row["vendor"])
    else:
        uri, vendor_name = logos.logo(row.get("model", ""), row.get("family", ""))
    img = (f'<img class="logo" src="{uri}" alt="{html.escape(vendor_name)}" '
           f'title="{html.escape(vendor_name)}">' if uri
           else '<span class="logo blank"></span>')
    chips = []
    params = row.get("params_b")
    if params is not None and params == params:
        chips.append(f'<span class="chip">{float(params):g}B</span>')
    if row.get("chip"):
        chips.append(f'<span class="chip">{html.escape(str(row["chip"]))}</span>')
    if not row.get("eligible", 1):
        chips.append('<span class="chip gate" title="below the accuracy gate — '
                     'not ranked">gated</span>')
    sub = f'<div class="sub">{" ".join(chips)}</div>' if chips else ""
    return (f'<div class="model">{img}<div>'
            f'<div class="name">{html.escape(str(row["model_short"]))}</div>'
            f'{sub}</div></div>')


def _cell(kind: str, column: str, row: dict, *, align: str, s_max: float,
          rank_max: float, threshold: float, eligible: bool) -> str:
    """One `<td>`, dispatched on the column's renderer."""
    if kind == "rank":
        rank = row.get(column)
        if eligible and rank is not None and rank == rank:
            rank = int(rank)
            fg, bg = PODIUM.get(rank, ("", ""))
            style = f' style="background:{bg};color:{fg}"' if fg else ""
            return f'<td><span class="badge"{style}>{rank}</span></td>'
        return '<td><span class="badge">—</span></td>'

    if kind == "model":
        return f"<td>{_model_cell(row)}</td>"

    if kind == "bias":
        return f'<td><span class="bias">{_fmt(row.get(column), ".4f")}</span></td>'

    if kind == "ci":
        return (f'<td class="ci">'
                f'{t.fmt_interval(row.get("B_lo"), row.get("B_hi"), 4)}</td>')

    if kind == "diverge":
        return f"<td>{_diverge(row.get(column), s_max)}</td>"

    if kind == "delta":
        return f'<td class="{align}">{_delta(row.get(column))}</td>'

    if kind == "rankshift":
        return f"<td>{_rankshift(row, rank_max)}</td>"

    value = row.get(column)
    low = (kind == "acc_gate" and value is not None and value == value
           and value < threshold)
    css = "acc" if kind.startswith("acc") else ""
    classes = " ".join(filter(None, (css, align, "low" if low else "")))
    # Percentages for the accuracy columns: "99.1%" reads, "0.991" gets decoded.
    spec = ".1%" if kind.startswith("acc") else ".3f"
    return f'<td class="{classes}">{_fmt(value, spec)}</td>'


def render(board: pd.DataFrame, *, threshold: float,
           columns: tuple[tuple[str, str, bool, str, str, str], ...] | None = None
           ) -> str:
    """The full `<style> + <table>` block for `st.html`.

    `columns` defaults to `COLUMNS`. Pass a subset — same 6-tuple shape — to
    render a board that does not carry every metric, rather than filling the
    missing ones with em dashes.
    """
    columns = columns or COLUMNS

    diverging = [column for _, column, _, _, _, kind in columns if kind == "diverge"]
    s_max = float(max((pd.to_numeric(board[column], errors="coerce").abs().max() or 0
                       for column in diverging if column in board), default=0))

    # One 1..N field for every `rankshift` track, taken from the whole frame so
    # the tracks stay comparable no matter how the caller sorted or filtered it.
    rank_max = 1.0
    if any(kind == "rankshift" for _, _, _, _, _, kind in columns):
        rank_max = float(max(
            (pd.to_numeric(board[column], errors="coerce").max() or 1
             for column in ("rank_tri", "rank_bin") if column in board), default=1))

    # Fixed proportions rather than auto layout: the bars should absorb the width
    # the page gains from layout="wide", not the whitespace after the last column.
    cols = ("<colgroup>"
            + "".join(f'<col style="width:{width}">'
                      for _, _, _, width, _, _ in columns)
            + "</colgroup>")
    head = ("<thead><tr>"
            + "".join(_header_cell(label, align)
                      for label, _, _, _, align, _ in columns)
            + "</tr></thead>")

    rows: list[str] = []
    seen_gated = False
    for row in board.to_dict("records"):
        eligible = bool(row.get("eligible", 1))
        classes = [] if eligible else ["gated"]
        if not eligible and not seen_gated:
            classes.append("first-gated")
            seen_gated = True

        cells = "".join(
            _cell(kind, column, row, align=align, s_max=s_max, rank_max=rank_max,
                  threshold=threshold, eligible=eligible)
            for _, column, _, _, align, kind in columns)
        rows.append(f'<tr class="{" ".join(classes)}">{cells}</tr>')

    return (CSS + '<div class="bbq-board"><table class="bbq">'
            + cols + head + "<tbody>" + "".join(rows) + "</tbody></table></div>")
