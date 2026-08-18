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
#: body and the sidebar's "Sort by" list are all built from this, so they cannot
#: drift apart.
#:
#: Cell renderers: ``rank`` podium badge · ``model`` logo + name + chips ·
#: ``bias`` bold B(m) · ``ci`` the B_lo/B_hi interval · ``diverge`` signed bar
#: centred at zero · ``acc`` plain 3dp · ``acc_gate`` 3dp, reddened below the
#: eligibility threshold · ``num`` plain 3dp for anything else.
COLUMNS: tuple[tuple[str, str, bool, str, str, str], ...] = (
    ("#",         "rank",               True,  "58px", "",      "rank"),
    ("Model",     "model_short",        True,  "28%",  "",      "model"),
    ("Bias B(m)", "B",                  True,  "9%",   "",      "bias"),
    ("95% CI",    "B_lo",               True,  "13%",  "",      "ci"),
    ("s_DIS",     "s_DIS",              False, "19%",  "",      "diverge"),
    ("s_AMB",     "s_AMB",              False, "19%",  "",      "diverge"),
    ("acc_dis",   "accuracy_disambig",  False, "6%",   "right", "acc_gate"),
    ("acc_amb",   "accuracy_ambig",     False, "6%",   "right", "acc"),
)

#: Header label → (frame column, ascending). `"#"` is the published rank, so it
#: is the default and reproduces the official order.
SORT_OPTIONS: dict[str, tuple[str, bool]] = {
    label: (column, ascending) for label, column, ascending, _, _, _ in COLUMNS}

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
          threshold: float, eligible: bool) -> str:
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

    value = row.get(column)
    low = (kind == "acc_gate" and value is not None and value == value
           and value < threshold)
    css = "acc" if kind.startswith("acc") else ""
    classes = " ".join(filter(None, (css, align, "low" if low else "")))
    return f'<td class="{classes}">{_fmt(value, ".3f")}</td>'


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

    # Fixed proportions rather than auto layout: the bars should absorb the width
    # the page gains from layout="wide", not the whitespace after the last column.
    cols = ("<colgroup>"
            + "".join(f'<col style="width:{width}">'
                      for _, _, _, width, _, _ in columns)
            + "</colgroup>")
    head = ("<thead><tr>"
            + "".join(f'<th class="{align}">{html.escape(label)}</th>'
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
            _cell(kind, column, row, align=align, s_max=s_max,
                  threshold=threshold, eligible=eligible)
            for _, column, _, _, align, kind in columns)
        rows.append(f'<tr class="{" ".join(classes)}">{cells}</tr>')

    return (CSS + '<div class="bbq-board"><table class="bbq">'
            + cols + head + "<tbody>" + "".join(rows) + "</tbody></table></div>")
