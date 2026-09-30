"""Experiment 4, the cheap wins: an ensemble of the three tied libraries, and quantile recalibration.

Ensemble. The simple average of LightGBM, XGBoost and CatBoost predictions; and weights (non-negative, summing to one)
fit on ONE earlier walk-forward year and scored on the next: 2024 -> 2025 (the plan), and 2023 -> 2024 as the out-of-time
replication (the weights are fit on predictions that were themselves made walk-forward, so nothing in the fit saw the year
it is scored on).

Recalibration. The Phase 2 p10/p90 band covers 78% overall but 74% for quarterbacks. Each target week's band is shifted by
the empirical residual quantiles of the rows before it: for a position, q10 moves by the 10th percentile of (actual - q10)
and q90 by the 90th percentile of (actual - q90), over the previous season's walk-forward predictions plus the current
season's earlier weeks. The default rule touches a position only when its trailing p10-p90 coverage is below the target by
more than a tolerance, so a position that is already calibrated (RB/WR/TE at 78-79%) is left alone, not widened.
`mode="all"` shifts every position and is reported next to it as a diagnostic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model import backtest as B
from model import train as T
from model.train import POSITIONS

TARGET_COVERAGE = 0.80
TOLERANCE = 0.03
MIN_ROWS = 100


# --------------------------------------------------------------------------- ensemble
def fit_weights(preds: pd.DataFrame, y: pd.Series, *, step: float = 0.01) -> dict[str, float]:
    """Weights on the simplex minimising RMSE of the blend, by exhaustive grid (step 0.01: 5,151 points for three models,
    deterministic, ties go to the first grid point). `preds` has one column per model."""
    cols = list(preds.columns)
    if len(cols) != 3:
        raise ValueError("the grid is written for three models")
    P = preds.to_numpy(dtype="float64")
    yv = y.to_numpy(dtype="float64")
    n = int(round(1 / step))
    best, best_w = np.inf, None
    for i in range(n + 1):
        for j in range(n + 1 - i):
            w = np.array([i, j, n - i - j], dtype="float64") / n
            mse = float(((P @ w - yv) ** 2).mean())
            if mse < best - 1e-15:
                best, best_w = mse, w
    return dict(zip(cols, (float(x) for x in best_w)))


def blend(preds: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    return sum(weights[c] * preds[c] for c in preds.columns)


# --------------------------------------------------------------------------- quantile recalibration
def coverage(y, lo, hi) -> float:
    return float(((y >= lo) & (y <= hi)).mean())


def recalibrate(cur: pd.DataFrame, history: pd.DataFrame, *, mode: str = "underage_only", target: float = TARGET_COVERAGE,
                tol: float = TOLERANCE, min_rows: int = MIN_ROWS, positions: tuple = POSITIONS) -> pd.DataFrame:
    """`cur` (one season's walk-forward quantile rows: y, q10, q50, q90, position, week, y_played; sorted quantiles) with
    q10a / q90a added. Week N's shift uses only `history` (a previous season's out-of-sample rows) and `cur` rows of weeks
    < N, never week N or later, and only played rows. The shift is per position; see the module docstring for `mode`.
    `positions` restricts which positions may be touched at all (("QB",) leaves every other band exactly as it was)."""
    if mode not in ("underage_only", "all"):
        raise ValueError(mode)
    out = cur.copy()
    out["q10a"], out["q90a"] = out["q10"], out["q90"]
    hist = history[history["y_played"] == 1]
    for wk in sorted(out["week"].unique()):
        trail = pd.concat([hist, cur[(cur["week"] < wk) & (cur["y_played"] == 1)]])
        rows = out["week"] == wk
        for pos in positions:
            r = trail[trail["position"] == pos]
            if len(r) < min_rows:
                continue
            if mode == "underage_only" and coverage(r["y"], r["q10"], r["q90"]) >= target - tol:
                continue
            lo = float(np.quantile(r["y"] - r["q10"], 0.10))
            hi = float(np.quantile(r["y"] - r["q90"], 0.90))
            m = rows & (out["position"] == pos)
            out.loc[m, "q10a"] = out.loc[m, "q10"] + lo
            out.loc[m, "q90a"] = out.loc[m, "q90"] + hi
    return out


def coverage_table(e: pd.DataFrame, lo: str, hi: str, *, draws: int = 2000, seed: int = T.SEED) -> pd.DataFrame:
    """p10-p90 coverage per position and pooled, with a 95% week-blocked bootstrap interval, the mean band width, and the
    share of actuals below `lo` / above `hi` (targets .10 / .10)."""
    rng = np.random.default_rng(seed)
    weeks = sorted(e["week"].unique())
    idx = rng.integers(0, len(weeks), size=(draws, len(weeks)))
    rows = {}
    for pos in [*POSITIONS, "ALL"]:
        g = e if pos == "ALL" else e[e["position"] == pos]
        hit = ((g["y"] >= g[lo]) & (g["y"] <= g[hi])).astype(float)
        by = pd.DataFrame({"h": hit, "n": 1.0, "week": g["week"]}).groupby("week").sum().reindex(weeks).fillna(0.0)
        boot = by["h"].to_numpy()[idx].sum(axis=1) / np.maximum(by["n"].to_numpy()[idx].sum(axis=1), 1)
        rows[pos] = {"n": len(g), "coverage": float(hit.mean()), "lo95": float(np.percentile(boot, 2.5)),
                     "hi95": float(np.percentile(boot, 97.5)), "below_lo": float((g["y"] < g[lo]).mean()),
                     "above_hi": float((g["y"] > g[hi]).mean()), "mean_width": float((g[hi] - g[lo]).mean())}
    return pd.DataFrame(rows).T


def interval_score(e: pd.DataFrame, lo: str, hi: str, alpha: float = 0.2) -> float:
    """Winkler interval score (lower is better): width plus 2/alpha times the miss distance. Sharpness and coverage in one
    number, so a recalibration that only widens has to pay for it."""
    y, l, u = e["y"].to_numpy(), e[lo].to_numpy(), e[hi].to_numpy()
    return float(np.mean((u - l) + (2 / alpha) * np.maximum(l - y, 0) + (2 / alpha) * np.maximum(y - u, 0)))
