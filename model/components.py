"""Experiment 3: component models, and composing points under any linear scoring.

One LightGBM per scoring component (targets, receptions, receiving yards and TDs, carries, rushing yards and TDs, pass
attempts / completions / yards / TDs / interceptions, fumbles lost, and a "misc" bucket of two-point conversions and
special-teams TDs), each walk-forward like the point model. A model predicts the conditional MEAN of its component, so any
linear scoring composes exactly: points = sum(weight * predicted component), and the composed value is the conditional mean
of the points. What linear scoring cannot express from means is a threshold (100-yard bonuses, 40-yard TD bonuses): those
need a distribution, not a mean, and are not composed (`divergences` lists them for a league).

`PPR` reproduces nflverse's fantasy_points_ppr from the actual components exactly (a test pins it, so the composition path
cannot drift). The alternate scoring is read from a league file: PPR plus the league's TE premium (`bonus_rec_te`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from model import labels
from model import p25run as P
from model import train as T
from model.train import Spec

REPO = Path(__file__).resolve().parents[1]
MISC = "y_misc_pts"
# modelled components. y_tgt / y_car / y_att are also the stage-one volume models of experiment 2 (same models, same cache).
TARGETS = ("y_tgt", "y_rec", "y_rec_yds", "y_rec_td", "y_car", "y_rush_yds", "y_rush_td", "y_att", "y_cmp", "y_pass_yds",
           "y_pass_td", "y_int", "y_fum_lost", MISC)
# the PPR weight per component (labels.PPR_WEIGHTS with two-point conversions and special-teams TDs folded into MISC)
PPR_WEIGHTS = {**{k: v for k, v in labels.PPR_WEIGHTS.items() if k not in ("y_two_pt", "y_st_td")}, MISC: 1.0}


@dataclass(frozen=True)
class Scoring:
    name: str
    weights: dict
    position_bonus: dict = field(default_factory=dict)     # {(position, component): points per unit}, added on top

    def needs(self) -> set[str]:
        return set(self.weights) | {c for _, c in self.position_bonus}


PPR = Scoring("ppr", PPR_WEIGHTS)


def add_misc(df: pd.DataFrame) -> pd.DataFrame:
    """The misc-points label: two-point conversions (2) and special-teams TDs (6), the components no model of ours has any
    signal on but that PPR counts. NaN exactly where the component labels are NaN (a DNP)."""
    out = df.copy()
    out[MISC] = 2.0 * out["y_two_pt"] + 6.0 * out["y_st_td"]
    return out


def compose(preds: dict[str, pd.Series], position: pd.Series, scoring: Scoring) -> pd.Series:
    """Points under `scoring` from per-component values (predictions, or actuals to check the identity)."""
    missing = scoring.needs() - set(preds)
    if missing:
        raise KeyError(f"scoring {scoring.name!r} needs components {sorted(missing)}")
    total = sum(w * preds[c] for c, w in scoring.weights.items())
    for (pos, comp), bonus in scoring.position_bonus.items():
        total = total + np.where(position == pos, bonus * preds[comp], 0.0)
    return pd.Series(np.asarray(total, dtype="float64"), index=position.index)


def league_scoring(slug: str, *, root: Path = REPO) -> tuple[Scoring, list[str]]:
    """(PPR + the league's TE premium, the ways the league's real scoring differs from it). Reads
    leagues/<slug>/league.json. The premium is `bonus_rec_te`; the base must be full PPR (rec == 1) or it refuses."""
    sc = json.loads((root / "leagues" / slug / "league.json").read_text())["scoring"]
    if sc.get("rec") != 1.0:
        raise ValueError(f"{slug}: rec = {sc.get('rec')}, not full PPR: the TE premium is defined on top of PPR")
    bonus = float(sc.get("bonus_rec_te", 0.0))
    diverge = []
    ppr_keys = {"rec": 1.0, "rec_yd": 0.1, "rec_td": 6.0, "rush_yd": 0.1, "rush_td": 6.0, "pass_yd": 0.04, "pass_td": 4.0,
                "pass_int": -2.0, "fum_lost": -2.0}
    for k, v in ppr_keys.items():
        if sc.get(k) != v:
            diverge.append(f"{k}: league {sc.get(k)} vs PPR {v}")
    thresholds = [k for k, v in sc.items() if v and (k.startswith("bonus_") and k != "bonus_rec_te" or k.endswith(("_40p", "_50p")))]
    if thresholds:
        diverge.append("threshold bonuses (need a distribution, not a mean): " + ", ".join(sorted(thresholds)))
    return Scoring("te_premium", dict(PPR_WEIGHTS), {("TE", "y_rec"): bonus}), diverge


# --------------------------------------------------------------------------- the runs
def component_runs(df: pd.DataFrame, mkey: str, spec: Spec, season: int, *, targets=TARGETS, log=None) -> dict[str, pd.DataFrame]:
    """{component: walk-forward result} on `spec`'s inputs, each with its own early-stopped tree count."""
    out = {}
    for target in targets:
        params = P.tree_params(df, mkey, target, spec)
        out[target] = P.wf(df, mkey, spec, season, target=target, params=params, log=log)
    return out


def composed(runs: dict[str, pd.DataFrame], scoring: Scoring, template: pd.DataFrame) -> pd.DataFrame:
    """A walk-forward-shaped frame (`template` = the primary run: same rows, KEEP columns, y = actual PPR) whose `pred` is
    the composed prediction under `scoring`. `y` stays the PPR label unless the caller replaces it."""
    idx = template.index
    preds = {c: r.loc[idx, "pred"] for c, r in runs.items()}
    out = template.copy()
    out["pred"] = compose(preds, template["position"], scoring).to_numpy()
    return out


def actual_points(df: pd.DataFrame, index: pd.Index, scoring: Scoring) -> pd.Series:
    """Points under `scoring` from the ACTUAL components of `index` rows (NaN for a DNP)."""
    d = df.loc[index]
    return compose({c: d[c] for c in scoring.needs()}, d["position"], scoring)


def component_skill(df: pd.DataFrame, runs: dict[str, pd.DataFrame], *, need_trail: bool = True) -> pd.DataFrame:
    """Per component on the head-to-head rows: n, mean, the model's RMSE, and the R-squared-style skill against a constant per
    position (the scored rows' own mean for that position: a reference with no player information). Skill = 1 - MSE / MSE_ref;
    it says how much of a component's variance any model can explain, which is where composed points gain or lose."""
    rows = {}
    for c, r in runs.items():
        e = r[(r["y_played"] == 1) & (r["base_trail3_ppr"].notna() if need_trail else True)]
        e = e[e["y"].notna()]
        ref = e.groupby("position")["y"].transform("mean")
        mse, mse_ref = float(((e["pred"] - e["y"]) ** 2).mean()), float(((ref - e["y"]) ** 2).mean())
        rows[c] = {"n": len(e), "mean": float(e["y"].mean()), "RMSE": mse ** 0.5, "RMSE_pos_const": mse_ref ** 0.5,
                   "skill_vs_pos_const": 1 - mse / mse_ref}
    return pd.DataFrame(rows).T
