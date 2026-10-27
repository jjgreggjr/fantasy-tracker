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

`mode="underage_by_position"` (2026-10-27, from the week 4-7 miss review: 78.6% inside [p10, p90] but 12.9% above p90 against
8.5% below; RB 15.4% and WR 14.1% above p90, TE 14.5% below p10, QB 89% inside): each TAIL of each position's band is judged on
its own. The tail's miss share (actuals above q90, or below q10) is measured on the trailing out-of-sample rows of THAT position,
pooled over positions when the position has fewer than `min_pos_rows` of them; a tail that misses more than its 10% share by more
than `tol / 2` is STRETCHED multiplicatively about q50 by the factor that would have made it cover (the 90th percentile of
(y - q50) / (q90 - q50) over the trailing rows, for the upper tail); a tail that misses by less than that is SHRUNK the same way
only when `allow_shrink` is on (off by default: a narrower band is a sharper claim and is James's call). Both directions are
bounded to `factor_bounds` (0.7 to 1.6). q50 never moves, and a positive factor keeps q10a <= q50 <= q90a.
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
MODES = ("underage_only", "all", "underage_by_position")
# underage_by_position defaults (serving_config.json "recalibration" may override: allow_shrink, min_pos_rows, factor_bounds)
MIN_POS_ROWS = 200                 # trailing rows a position needs before its own tails are judged; fewer -> pooled
FACTOR_BOUNDS = (0.7, 1.6)         # a tail's half-width is multiplied by a factor inside these bounds, never outside
ALLOW_SHRINK = False


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


def recal_kwargs(section: dict) -> dict:
    """`recalibrate` keyword arguments from serving_config.json's "recalibration" section: the mode, plus the by-position
    options when they are set (absent keys keep the module defaults, so the frozen {"mode": "underage_only"} is unchanged)."""
    kw = {"mode": section["mode"]}
    if "allow_shrink" in section:
        kw["allow_shrink"] = bool(section["allow_shrink"])
    if "min_pos_rows" in section:
        kw["min_pos_rows"] = int(section["min_pos_rows"])
    if "factor_bounds" in section:
        kw["factor_bounds"] = tuple(float(x) for x in section["factor_bounds"])
    return kw


def tail_factors(r: pd.DataFrame, *, tol: float = TOLERANCE, allow_shrink: bool = ALLOW_SHRINK,
                 factor_bounds: tuple = FACTOR_BOUNDS) -> tuple[float, float]:
    """(lower, upper) multiplicative factors on the half-widths q50 - q10 and q90 - q50 from the trailing rows `r`, the
    underage_by_position rule: a tail whose miss share (y below q10 / above q90) is over 10% + tol/2 stretches to the ratio
    quantile that would have covered it; one under 10% - tol/2 shrinks the same way only with `allow_shrink`; otherwise 1.
    Rows with a zero half-width carry no ratio. Each factor is clipped to `factor_bounds`."""
    lo_b, hi_b = factor_bounds
    side_tol = tol / 2
    out = []
    for miss, num, den in ((r["y"] < r["q10"], r["q50"] - r["y"], r["q50"] - r["q10"]),
                           (r["y"] > r["q90"], r["y"] - r["q50"], r["q90"] - r["q50"])):
        share = float(miss.mean())
        ok = den > 0
        if not ok.any():
            out.append(1.0)
            continue
        f = float(np.quantile((num[ok] / den[ok]).to_numpy(dtype="float64"), 0.90))
        if share > 0.10 + side_tol and f > 1.0:
            out.append(min(max(f, 1.0), hi_b))
        elif share < 0.10 - side_tol and f < 1.0 and allow_shrink:
            out.append(max(min(f, 1.0), lo_b))
        else:
            out.append(1.0)
    return out[0], out[1]


def recalibrate(cur: pd.DataFrame, history: pd.DataFrame, *, mode: str = "underage_only", target: float = TARGET_COVERAGE,
                tol: float = TOLERANCE, min_rows: int = MIN_ROWS, positions: tuple = POSITIONS,
                min_pos_rows: int = MIN_POS_ROWS, allow_shrink: bool = ALLOW_SHRINK,
                factor_bounds: tuple = FACTOR_BOUNDS) -> pd.DataFrame:
    """`cur` (one season's walk-forward quantile rows: y, q10, q50, q90, position, week, y_played; sorted quantiles) with
    q10a / q90a added. Week N's shift uses only `history` (a previous season's out-of-sample rows) and `cur` rows of weeks
    < N, never week N or later, and only played rows. The shift is per position; see the module docstring for `mode`.
    `positions` restricts which positions may be touched at all (("QB",) leaves every other band exactly as it was).
    `min_pos_rows`, `allow_shrink` and `factor_bounds` belong to `underage_by_position` only."""
    if mode not in MODES:
        raise ValueError(mode)
    out = cur.copy()
    out["q10a"], out["q90a"] = out["q10"], out["q90"]
    hist = history[history["y_played"] == 1]
    for wk in sorted(out["week"].unique()):
        trail = pd.concat([hist, cur[(cur["week"] < wk) & (cur["y_played"] == 1)]])
        rows = out["week"] == wk
        for pos in positions:
            r = trail[trail["position"] == pos]
            m = rows & (out["position"] == pos)
            if mode == "underage_by_position":
                if len(r) < min_pos_rows:                # too few of his own position: the pooled trailing rows decide
                    r = trail
                if len(r) < min_rows:
                    continue
                f_lo, f_hi = tail_factors(r, tol=tol, allow_shrink=allow_shrink, factor_bounds=factor_bounds)
                out.loc[m, "q10a"] = out.loc[m, "q50"] - f_lo * (out.loc[m, "q50"] - out.loc[m, "q10"])
                out.loc[m, "q90a"] = out.loc[m, "q50"] + f_hi * (out.loc[m, "q90"] - out.loc[m, "q50"])
                continue
            if len(r) < min_rows:
                continue
            if mode == "underage_only" and coverage(r["y"], r["q10"], r["q90"]) >= target - tol:
                continue
            lo = float(np.quantile(r["y"] - r["q10"], 0.10))
            hi = float(np.quantile(r["y"] - r["q90"], 0.90))
            out.loc[m, "q10a"] = out.loc[m, "q10"] + lo
            out.loc[m, "q90a"] = out.loc[m, "q90"] + hi
    return out


def coverage_frame(rows: pd.DataFrame, by: str = "position", *, y: str = "y", lo: str = "q10", hi: str = "q90") -> pd.DataFrame:
    """Per group of `by` (and a pooled ALL row): rows, share inside [lo, hi], share above hi, share below lo. Rows where any of
    the three is missing are dropped. The miss review's calibration table (targets .80 / .10 / .10)."""
    d = rows.dropna(subset=[y, lo, hi])
    keys = [*(k for k in POSITIONS if (d[by] == k).any()), *sorted(set(map(str, d[by])) - set(POSITIONS))] if len(d) else []
    recs = {}
    for k in [*keys, "ALL"]:
        g = d if k == "ALL" else d[d[by] == k]
        yv, lv, hv = g[y].to_numpy(dtype="float64"), g[lo].to_numpy(dtype="float64"), g[hi].to_numpy(dtype="float64")
        n = len(g)
        recs[k] = {"n": n, "inside": float(((yv >= lv) & (yv <= hv)).mean()) if n else np.nan,
                   "above_hi": float((yv > hv).mean()) if n else np.nan, "below_lo": float((yv < lv).mean()) if n else np.nan}
    return pd.DataFrame(recs).T.astype({"n": "int64"})


def coverage_report(hist_rows: pd.DataFrame, by: str = "position", *, y: str = "y", lo: str = "q10", hi: str = "q90",
                    out=print) -> str:
    """The calibration table as markdown (inside / above p90 / below p10 by `by`, then ALL), printed through `out` (None: not
    printed) and returned. `hist_rows` are quantile rows (y, q10, q90, position) or any frame with the named columns."""
    t = coverage_frame(hist_rows, by, y=y, lo=lo, hi=hi)
    L = [f"| {by} | n | inside [{lo}, {hi}] | above {hi} | below {lo} |", "|---|---|---|---|---|"]
    for k, r in t.iterrows():
        L.append(f"| {k} | {int(r['n'])} | {r['inside']:.1%} | {r['above_hi']:.1%} | {r['below_lo']:.1%} |" if r["n"]
                 else f"| {k} | 0 | n/a | n/a | n/a |")
    L.append("")
    L.append("Targets: 80% inside, 10% above, 10% below. A tail over 10% is short (stretch it), one under 10% is slack (shrink only on a say-so).")
    text = "\n".join(L)
    if out is not None:
        out(text)
    return text


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
