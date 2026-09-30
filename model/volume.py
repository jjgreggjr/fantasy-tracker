"""Experiment 2: a two-stage model, and how well volume itself can be predicted.

    stage one   LightGBM models for the week-N targets, carries and pass attempts (the inputs of expected fantasy points),
                on the Phase 2 primary features, walk-forward exactly like the point model.
    stage two   points from the stage-one predictions plus shrunken efficiency priors (career / season-to-date catch
                rate, yards and TDs per target and per carry, per-attempt passing rates) and position.

Stage two is trained on HONEST stage-one predictions: for every row of every earlier week, the value the stage-one model
would have produced that week, from a model trained strictly before it (`honest_series`: one walk-forward per season, so
row (season, week) sees only earlier labels). Training stage two on in-sample stage-one fits would teach it to trust
volume estimates that are far better than anything available at prediction time. The cost is that 2021 has no honest
predictions (there is nothing earlier to train on), so stage two trains on 2022+ and the controls are matched to that
window (`flat_2022`).

Also tested: the stage-one predictions as extra columns of the flat model, with and without the lagged-xFP features they
would replace. No same-week value enters any model: the only same-week numbers are the labels being scored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model import features as F
from model import p25run as P
from model import train as T
from model.train import POSITIONS, Spec

# label -> stage-one column, and the lag columns whose mean is the trailing-3 baseline for it
S1 = {"y_tgt": "s1_tgt", "y_car": "s1_car", "y_att": "s1_att"}
S1_COLS = tuple(S1.values())
TRAIL = {"y_tgt": ("targets_l1", "targets_l2", "targets_l3"), "y_car": ("carries_l1", "carries_l2", "carries_l3"),
         "y_att": ("pass_att_l1", "pass_att_l2", "pass_att_l3")}
STD = {"y_tgt": "targets_std_mean", "y_car": "carries_std_mean"}
RELEVANT = {"y_tgt": ("RB", "WR", "TE"), "y_car": ("RB", "QB"), "y_att": ("QB",)}    # where the quantity is real volume
EFF_COLS = tuple(F.FAMILIES["eff"])
XFP_LAGS = ("xfp_l1", "xfp_l2", "xfp_l3", "xfp_std_mean", "prev_season_xfp_pg")
HONEST_SEASONS = (2022, 2023, 2024, 2025)          # 2021 has no earlier data to be honest about

SPECS = {
    "two_stage": Spec("two_stage", only_cols=S1_COLS + EFF_COLS, min_season=2022),
    "flat_2022": Spec("flat_2022", min_season=2022),
    "flat_plus_s1": Spec("flat_plus_s1", extra_cols=S1_COLS),
    "flat_plus_eff": Spec("flat_plus_eff", add_families=("eff",)),
    "flat_plus_s1_eff": Spec("flat_plus_s1_eff", add_families=("eff",), extra_cols=S1_COLS),
    "flat_no_xfp": Spec("flat_no_xfp", drop_cols=XFP_LAGS),
    "flat_no_xfp_plus_s1": Spec("flat_no_xfp_plus_s1", drop_cols=XFP_LAGS, extra_cols=S1_COLS),
}


def honest_series(df: pd.DataFrame, mkey: str, spec: Spec = T.PRIMARY, *, seasons=HONEST_SEASONS, log=None) -> pd.DataFrame:
    """The stage-one predictions of every row of `seasons` (all spine rows of each week, played or not) from a model
    trained strictly before that week. One walk-forward per (target, season) with the per-target tree count."""
    out = pd.DataFrame(np.nan, index=df.index, columns=list(S1_COLS))
    for target, col in S1.items():
        params = P.tree_params(df, mkey, target, spec)
        for season in seasons:
            r = P.wf(df, mkey, spec, season, target=target, params=params, log=log)
            out.loc[r.index, col] = r["pred"].to_numpy()
    return out


def with_stage_one(df: pd.DataFrame, s1: pd.DataFrame) -> pd.DataFrame:
    """The matrix plus the honest stage-one columns. The result is only ever used through the walk-forward mask."""
    return df.join(s1)


def trailing_baseline(df: pd.DataFrame, target: str) -> pd.Series:
    """Mean of the player's last <=3 appearances this season (the same lags every model sees); NaN with no history."""
    return df[list(TRAIL[target])].mean(axis=1, skipna=True)


def quality_frame(df: pd.DataFrame, res: pd.DataFrame, target: str) -> pd.DataFrame:
    """Scored rows for one stage-one target: the head-to-head set (played rows with an earlier game), with the model
    prediction, the trailing-3 mean and (where it exists) the season-to-date mean."""
    e = res[(res["y_played"] == 1) & res["base_trail3_ppr"].notna()].copy()
    e["model"] = e["pred"]
    e["trail3"] = trailing_baseline(df, target).loc[e.index]
    e["std_mean"] = df[STD[target]].loc[e.index] if target in STD else np.nan
    e["y"] = df.loc[e.index, target]
    return e[e["y"].notna() & e["trail3"].notna()]


def _week_sums(e: pd.DataFrame, col: str) -> pd.DataFrame:
    err = (e[col] - e["y"]).abs()
    return pd.DataFrame({"abs": err, "n": 1.0, "week": e["week"]}).groupby("week").sum()


def volume_quality(e: pd.DataFrame, target: str, *, draws: int = 2000, seed: int = T.SEED) -> pd.DataFrame:
    """Per position where the quantity is real volume (targets: RB/WR/TE, carries: RB/QB, attempts: QB), and pooled: n, MAE of the model, of the trailing-3
    mean and of the season-to-date mean, the model minus trailing-3 MAE difference with a 95% week-blocked bootstrap
    interval, RMSE, Pearson correlation and R-squared (against the group's own mean) for the model and the trailing-3 mean."""
    rng = np.random.default_rng(seed)
    weeks = sorted(e["week"].unique())
    idx = rng.integers(0, len(weeks), size=(draws, len(weeks)))
    rows = {}
    groups = {p: (p,) for p in RELEVANT[target]}
    if len(RELEVANT[target]) > 1:
        groups["relevant"] = RELEVANT[target]
    for label, pos in groups.items():
        g = e[e["position"].isin(pos)]
        if not len(g):
            continue
        a, b = _week_sums(g, "model").reindex(weeks).fillna(0.0), _week_sums(g, "trail3").reindex(weeks).fillna(0.0)
        n = a["n"].to_numpy()
        d = (a["abs"].to_numpy()[idx].sum(axis=1) - b["abs"].to_numpy()[idx].sum(axis=1)) / n[idx].sum(axis=1)
        r = {"n": int(len(g)), "y_mean": g["y"].mean(), "MAE_model": a["abs"].sum() / n.sum(),
             "MAE_trail3": b["abs"].sum() / n.sum(),
             "dMAE": (a["abs"].sum() - b["abs"].sum()) / n.sum(), "dMAE_lo": float(np.percentile(d, 2.5)),
             "dMAE_hi": float(np.percentile(d, 97.5)),
             "RMSE_model": float(np.sqrt(((g["model"] - g["y"]) ** 2).mean())),
             "RMSE_trail3": float(np.sqrt(((g["trail3"] - g["y"]) ** 2).mean())),
             "corr_model": float(np.corrcoef(g["model"], g["y"])[0, 1]),
             "corr_trail3": float(np.corrcoef(g["trail3"], g["y"])[0, 1])}
        sst = float(((g["y"] - g["y"].mean()) ** 2).sum())
        r["R2_model"] = 1 - float(((g["model"] - g["y"]) ** 2).sum()) / sst
        r["R2_trail3"] = 1 - float(((g["trail3"] - g["y"]) ** 2).sum()) / sst
        if g["std_mean"].notna().all():
            r["MAE_std_mean"] = float((g["std_mean"] - g["y"]).abs().mean())
        rows[label] = r
    return pd.DataFrame(rows).T
