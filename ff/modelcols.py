"""The model columns beside E_pts: E_pts_model, p10, p90.

`data/model_pts.csv` (written by `python -m model.serve`) holds one row per predicted player per week: the model's points
in each configured league's own linear scoring (`pts_<slug>`, composed from the component models), the PPR components
composition (`pts_model_components`), and a PPR floor / ceiling (`p10`, `p90`). This module turns one league's slice of
that file into the three columns the recipes print, and joins them onto any frame that has a `gsis_id`.

  E_pts_model   that league's composed points (1 decimal, like E_pts)
  p10, p90      the PPR floor / ceiling moved by the league's scoring difference: `p + (pts_<slug> - pts_model_components)`.
                A league that scores exactly PPR (no difference) gets the PPR band unchanged. For the half-PPR league and
                the TE-premium league it is the PPR band shifted, with its width left as PPR's: the quantile models are
                fit on PPR points, and a mean does not compose into a quantile.

Display only. E_pts stays authoritative for every recipe, sort and report; nothing here is read by the pipeline. With no
`data/model_pts.csv`, or no row for the frame's week, `attach` returns the frame unchanged (no columns are added), so the
recipes print exactly what they printed before the model existed.

Pure pandas: `ff.ask` runs in question sessions that have no ML libraries installed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

MODEL_COLS = ("E_pts_model", "p10", "p90")
ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = "model_pts.csv"
NOTE = ("E_pts_model / p10 / p90 = the boosted-tree model's points in this league's scoring and its floor / ceiling: "
        "side by side and display only; E_pts stays authoritative while the model is on probation")


def read_model_pts(root: Path = ROOT) -> pd.DataFrame:
    p = Path(root) / "data" / MODEL_FILE
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, low_memory=False, dtype={"gsis_id": str})
    except Exception:          # an unreadable file must never break a recipe
        return pd.DataFrame()


def league_columns(mp: pd.DataFrame, slug: str) -> pd.DataFrame:
    """`gsis_id, E_pts_model, p10, p90` for one league from the rows of ONE (season, week) of model_pts.csv. Empty when the
    league has no composed column."""
    col = f"pts_{slug}"
    need = ["gsis_id", col, "pts_model_components", "p10", "p90"]
    if mp is None or mp.empty or any(c not in mp.columns for c in need):
        return pd.DataFrame(columns=["gsis_id", *MODEL_COLS])
    d = mp[need].dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")
    delta = pd.to_numeric(d[col], errors="coerce") - pd.to_numeric(d["pts_model_components"], errors="coerce")
    tidy = lambda x: (x.round(1) + 0.0).to_numpy()            # + 0.0: a tiny negative must not print as -0.0
    return pd.DataFrame({"gsis_id": d["gsis_id"].astype(str).to_numpy(),
                         "E_pts_model": tidy(pd.to_numeric(d[col], errors="coerce")),
                         "p10": tidy(pd.to_numeric(d["p10"], errors="coerce") + delta),
                         "p90": tidy(pd.to_numeric(d["p90"], errors="coerce") + delta)})


def week_of(df: pd.DataFrame, mp: pd.DataFrame) -> tuple[int, int] | None:
    """(season, week) whose model rows belong next to `df`: the frame's own (most common) season and week when it has them,
    else the newest week in the model file."""
    if mp is None or mp.empty:
        return None
    if {"season", "week"} <= set(df.columns) and df[["season", "week"]].dropna().shape[0]:
        s, w = df[["season", "week"]].dropna().mode().iloc[0]
        return int(s), int(w)
    m = mp[["season", "week"]].dropna().astype(int).drop_duplicates().sort_values(["season", "week"])
    return (int(m.iloc[-1]["season"]), int(m.iloc[-1]["week"])) if len(m) else None


def attach(df: pd.DataFrame, slug: str, root: Path = ROOT) -> pd.DataFrame:
    """`df` with E_pts_model / p10 / p90 joined on gsis_id (NaN where the model has no row for that player). Unchanged, and
    with no new columns, when there is nothing for this league and week."""
    if df is None or df.empty or "gsis_id" not in df.columns:
        return df
    mp = read_model_pts(root)
    key = week_of(df, mp)
    if key is None:
        return df
    rows = mp[(mp["season"] == key[0]) & (mp["week"] == key[1])]
    cols = league_columns(rows, slug)
    if cols.empty:
        return df
    out = df.drop(columns=[c for c in MODEL_COLS if c in df.columns]).copy()
    lookup = cols.set_index("gsis_id")
    gid = out["gsis_id"].astype("object")
    for c in MODEL_COLS:
        out[c] = gid.map(lookup[c]).astype("float64")
    return out
