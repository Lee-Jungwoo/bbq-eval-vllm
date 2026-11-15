"""Read-only access to `leaderboard/bbq.sqlite`, cached for Streamlit.

The database is a build artifact of `bbq_pipeline/publish.py` and is never written
here. Every query is cached against the file's mtime, so a republish while the
app is running invalidates the cache instead of serving stale rows.
"""
from __future__ import annotations

import json
import sqlite3

from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR.parent
LEADERBOARD_DIR = REPO_ROOT / "results" / "leaderboard"
DB_PATH = LEADERBOARD_DIR / "bbq.sqlite"

# The metric definitions and palette live with the pipeline, so the app can never
# drift from the numbers publish.py wrote.



def _mtime() -> float:
    return DB_PATH.stat().st_mtime if DB_PATH.exists() else 0.0


@st.cache_resource(show_spinner=False)
def _connect(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)


def require_db() -> None:
    """Render a build hint instead of a traceback when the database is absent."""
    if DB_PATH.exists():
        return
    st.error("No published database yet.")
    st.markdown(
        "`leaderboard/bbq.sqlite` is a build artifact and is not committed. "
        "Build it from the run directories:")
    st.code("python bbq_pipeline/publish.py --verify", language="bash")
    st.stop()


@st.cache_data(show_spinner=False)
def query(sql: str, params: tuple = (), _mtime_key: float = 0.0) -> pd.DataFrame:
    return pd.read_sql_query(sql, _connect(str(DB_PATH)), params=params)


def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    return query(sql, params, _mtime_key=_mtime())


# ---------------------------------------------------------------------------
# typed accessors
# ---------------------------------------------------------------------------
def publish_info() -> dict:
    frame = q("SELECT key, value FROM publish_meta")
    return dict(zip(frame["key"], frame["value"]))


def comparability_groups() -> pd.DataFrame:
    return q("""
        SELECT comparability_key,
               COUNT(*)             AS n_runs,
               MIN(n_instances)     AS n_instances,
               MIN(prompt_hash)     AS prompt_hash,
               MIN(categories)      AS categories
        FROM runs GROUP BY comparability_key ORDER BY n_runs DESC
    """)


def leaderboard(key: str) -> pd.DataFrame:
    return q("""
        SELECT l.*, r.model, r.model_short, r.family, r.params_b, r.dtype,
               m.n_ambig, m.n_answered, m.mean_prob_mass_captured
        FROM leaderboard l
        JOIN runs r        ON r.run_id = l.run_id
        JOIN run_metrics m ON m.run_id = l.run_id AND m.scope = 'overall'
        WHERE l.comparability_key = ?
        ORDER BY l.eligible DESC, l.rank, l.B
    """, (key,))


def pairs(key: str) -> pd.DataFrame:
    return q("""
        SELECT p.*, a.model_short AS model_a, b.model_short AS model_b
        FROM leaderboard_pairs p
        JOIN runs a ON a.run_id = p.run_id_a
        JOIN runs b ON b.run_id = p.run_id_b
        WHERE p.comparability_key = ?
    """, (key,))


def runs() -> pd.DataFrame:
    return q("SELECT * FROM runs ORDER BY family, params_b, model_short")


def run_metrics(run_id: str, scope: str = "overall") -> pd.DataFrame:
    return q("SELECT * FROM run_metrics WHERE run_id = ? AND scope = ? "
             "ORDER BY scope_value", (run_id, scope))


def cell_metrics(run_id: str, scope: str = "overall") -> pd.DataFrame:
    return q("SELECT * FROM cell_metrics WHERE run_id = ? AND scope = ? "
             "ORDER BY scope_value, context_condition, question_polarity",
             (run_id, scope))


def margin_hist(run_id: str) -> pd.DataFrame:
    return q("SELECT * FROM margin_hist WHERE run_id = ? ORDER BY correct, bin_lo",
             (run_id,))


def category_order(run_id: str) -> list[str]:
    """Categories by descending item count — the notebooks' panel order."""
    frame = q("SELECT scope_value, n FROM run_metrics WHERE run_id = ? "
              "AND scope = 'category' ORDER BY n DESC, scope_value", (run_id,))
    return frame["scope_value"].tolist()


def provenance(run_id: str) -> dict:
    frame = q("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    if frame.empty:
        return {}
    row = frame.iloc[0].to_dict()
    for field in ("option_letters", "label_token_ids", "categories",
                  "max_question_index"):
        try:
            row[field] = json.loads(row[field])
        except (TypeError, ValueError):
            pass
    return row


def sidebar_footer(container=None) -> None:
    """게시 시각과 부트스트랩 설정. 발표 중에는 읽을 것이 아니므로 기본으로는
    사이드바 맨 아래에, 페이지가 컨테이너를 넘겨주면 그 안에 접어 둔다."""
    info = publish_info()
    container = container or st.sidebar
    container.caption(
        f"published {info.get('published_at', '?')}  \n"
        f"schema v{info.get('schema_version', '?')} · "
        f"bootstrap R={info.get('bootstrap_R', '?')} seed={info.get('bootstrap_seed', '?')}")
