"""Per-model breakdown: category, polarity, margin, abstention, provenance."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import charts, db
from lib import theme as t

st.set_page_config(page_title="Model detail · BBQ", page_icon="🔍", layout="wide")
db.require_db()

runs = db.runs()
if runs.empty:
    st.warning("No runs published yet.")
    st.stop()

labels = {row.run_id: f"{row.family} · {row.model_short}"
          for row in runs.itertuples(index=False)}
options = list(labels)
# Default to the top-ranked run rather than whatever sorts first — landing on a
# model that failed the accuracy gate is a confusing first impression.
top_ranked = db.q("SELECT run_id FROM leaderboard WHERE rank = 1 LIMIT 1")
default = st.query_params.get("run_id")
if default not in labels:
    default = top_ranked["run_id"].iloc[0] if not top_ranked.empty else options[0]
index = options.index(default)

with st.sidebar:
    st.header("Run")
    run_id = st.selectbox("Model", options, index=index,
                          format_func=lambda r: labels[r])
db.sidebar_footer()
st.query_params["run_id"] = run_id

overall = db.run_metrics(run_id, "overall").iloc[0]
provenance = db.provenance(run_id)
st.title(provenance["model_short"])
st.caption(f"{provenance['model']} · {provenance['dtype']} · "
           f"revision `{(provenance.get('model_revision') or '?')[:12]}` · "
           f"{int(overall['n']):,} items")

# Compare against the ranked field rather than against nothing.
board = db.leaderboard(provenance["comparability_key"])
peers = board[(board["eligible"] == 1) & (board["run_id"] != run_id)]
row = board[board["run_id"] == run_id]
row = row.iloc[0] if not row.empty else None


def delta_vs_peers(value, column: str, invert: bool = False) -> str | None:
    if peers.empty or value is None or pd.isna(value):
        return None
    median = peers[column].median()
    if pd.isna(median):
        return None
    diff = value - median
    return f"{diff:+.3f} vs peer median" if not invert else f"{-diff:+.3f}"


cols = st.columns(5)
cols[0].metric("accuracy_ambig", f"{overall['accuracy_ambig']:.4f}",
               delta_vs_peers(overall["accuracy_ambig"], "accuracy_ambig"),
               delta_color="off")
cols[1].metric("accuracy_disambig", f"{overall['accuracy_disambig']:.4f}",
               delta_vs_peers(overall["accuracy_disambig"], "accuracy_disambig"),
               delta_color="off")
cols[2].metric("s_AMB", f"{overall['s_AMB']:+.4f}",
               f"unscaled {overall['s_AMB_unscaled']:+.3f}", delta_color="off")
cols[3].metric("s_DIS", f"{overall['s_DIS']:+.4f}",
               delta_vs_peers(overall["s_DIS"], "s_DIS"), delta_color="off")
if row is not None:
    cols[4].metric("B(m)", f"{row['B']:.4f}",
                   f"rank {int(row['rank'])}" if pd.notna(row["rank"])
                   else "below the accuracy gate", delta_color="off")

if row is not None and row["abstention_tier"] != "none":
    st.warning(
        f"This model abstained on **{row['unknown_rate_ambig']:.1%}** of ambiguous "
        f"items, leaving only **{int(row['n_elig_ambig']):,}** of "
        f"{int(overall['n_ambig']):,} in the bias denominator "
        f"(flag: *{row['abstention_tier']}*). Its `s_AMB` of "
        f"{overall['s_AMB']:+.4f} is a heavily damped view of "
        f"`s_AMB_unscaled = {overall['s_AMB_unscaled']:+.3f}`.")

by_category = db.run_metrics(run_id, "category")
order = db.category_order(run_id)
by_category = by_category.set_index("scope_value").loc[order].reset_index()

tabs = st.tabs(["By category", "Polarity split", "Margin", "Abstention",
                "Provenance"])

with tabs[0]:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.category_bars(
            by_category, ("accuracy_ambig", "accuracy_disambig"),
            ("ambiguous", "disambiguated"), "Accuracy by category",
            hline=1 / 3, hline_label="chance (1/3)", xaxis_title="accuracy"), width="stretch")
    with right:
        st.plotly_chart(charts.category_bars(
            by_category, ("s_AMB", "s_DIS"), ("s_AMB", "s_DIS"),
            "Bias score by category  (+ = toward the stereotype)",
            hline=0.0, xaxis_title="bias score"), width="stretch")
    st.dataframe(
        by_category[["scope_value", "n", "accuracy_ambig", "accuracy_disambig",
                     "s_AMB", "s_AMB_unscaled", "s_DIS", "n_elig_ambig",
                     "n_elig_disambig"]]
        .rename(columns={"scope_value": "category"}),
        hide_index=True, width="stretch")

with tabs[1]:
    cells = db.cell_metrics(run_id, "overall")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.polarity_panels(
            cells, "accuracy", "Accuracy by context × polarity"), width="stretch")
    with right:
        st.plotly_chart(charts.polarity_panels(
            cells, "s", "Bias score by context × polarity", zeroline=True), width="stretch")
    st.caption("`neg` asks which group fits the negative stereotype, `nonneg` its "
               "complement. `s` follows the headline convention: scaled by "
               "(1 − accuracy) for ambiguous, raw for disambiguated.")
    st.dataframe(cells[["context_condition", "question_polarity", "n", "accuracy",
                        "unknown_rate", "n_elig", "n_biased", "s_raw", "s",
                        "s_lo", "s_hi"]],
                 hide_index=True, width="stretch")

with tabs[2]:
    hist = db.margin_hist(run_id)
    if hist.empty:
        st.info("No margin data for this run.")
    else:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(charts.margin_bars(hist), width="stretch")
        with right:
            st.plotly_chart(charts.margin_ecdf(hist), width="stretch")
        st.caption("`margin` is the gap between the top two label logprobs — how "
                   "decisively the model picked its answer. It is null when two "
                   "of the three label tokens fell outside the server's top-k.")
        st.dataframe(pd.DataFrame([{
            "p25": overall["margin_p25"], "median": overall["margin_median"],
            "p75": overall["margin_p75"],
            "median (correct)": overall["margin_median_correct"],
            "median (incorrect)": overall["margin_median_incorrect"],
            "null margins": int(overall["n_margin_null"]),
        }]), hide_index=True, width="stretch")

with tabs[3]:
    st.plotly_chart(charts.abstention_composition(
        by_category.assign(scope_value=by_category["scope_value"])),
        width="stretch")
    st.plotly_chart(charts.s_answered_by_category(by_category, t.MIN_N_ANSWERED),
                    width="stretch")
    st.caption(
        f"`s_answered` is `s_AMB` **without** the (1 − accuracy_ambig) factor — the "
        f"direction the model leans when it commits to a group. Bars are Wilson 95% "
        f"intervals; a hollow marker means fewer than {t.MIN_N_ANSWERED} answers, "
        f"too few to read.")

with tabs[4]:
    left, right = st.columns(2)
    with left:
        st.markdown("**Model**")
        st.write({
            "model": provenance["model"],
            "revision": provenance.get("model_revision"),
            "dtype": provenance.get("dtype"),
            "params (B)": provenance.get("params_b"),
            "run_id": run_id,
        })
        st.markdown("**Scoring setup**")
        st.write({
            "prompt_hash": provenance["prompt_hash"],
            "comparability_key": provenance["comparability_key"],
            "option_letters": provenance["option_letters"],
            "label_token_ids": provenance["label_token_ids"],
            "label_token_variant": provenance.get("label_token_variant"),
            "max_question_index": provenance["max_question_index"],
        })
    with right:
        st.markdown("**Tokenisation**")
        backend = provenance.get("tokenizer_backend")
        if backend:
            st.write({
                "tokenizer_backend": backend,
                "prompt tokens": f"{provenance['prompt_token_len_min']}–"
                                 f"{provenance['prompt_token_len_max']}",
            })
            st.markdown("**Sample prompt**")
            st.code(provenance.get("sample_prompt") or "", language="text")
        else:
            # Absent, not empty: these fields were added after this run was made.
            st.info("Not recorded — this run predates the `tokenizer_backend` / "
                    "`prompt_token_len` / `sample_prompt` fields in meta.json.")
        st.markdown("**Data quality**")
        st.write({
            "mean prob mass captured": round(overall["mean_prob_mass_captured"], 6),
            "min prob mass captured": round(overall["min_prob_mass_captured"], 6),
            "records missing a label logprob": int(overall["n_null_logprob_records"]),
            "records without target_loc metadata": int(overall["n_no_target_metadata"]),
        })
