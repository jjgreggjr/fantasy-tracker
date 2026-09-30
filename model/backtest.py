"""Walk-forward backtest of 2025, scoring, and the ablation runner.

    model/.venv/bin/python -m model.backtest bakeoff          # LightGBM / XGBoost / CatBoost, primary config
    model/.venv/bin/python -m model.backtest ablate --lib lightgbm
    model/.venv/bin/python -m model.backtest replicate --lib lightgbm   # the same table on the 2024 walk-forward
    model/.venv/bin/python -m model.backtest quantiles
    model/.venv/bin/python -m model.backtest shap --lib lightgbm
    model/.venv/bin/python -m model.backtest all

The rule (PLAN_MODEL.md ground rule 4): to score week N of 2025, train on strictly earlier data, that is
2021-2024 plus 2025 weeks < N, refit every week, never shuffle weeks into folds. `train_mask` is the ONE place that
decides which rows train, and `assert_walk_forward` re-checks the mask against the target week before every fit
(tests/test_backtest.py pins it, and proves that garbage in every row at or after the target week changes no
prediction). Nothing reads a `y_*` column of the target week or later while fitting.

Rows and targets:
  * models train on played rows (`y_played == 1`: a stats row OR offensive snaps, snap-only = 0.0 points), the
    Phase 1 philosophy "points if he plays; availability is a separate layer". `stats_row` (only rows nflverse has a
    stats line for) is the ablation.
  * every model is scored on the SAME rows: 2025 played rows, and for the head-to-head against the trailing
    average only those with at least one earlier game this season (`base_trail3_ppr` not NA).
  * `spine_depth_only == 1` rows (admitted only by the 2025 daily depth-chart snapshot; ~5x as common in 2025 as in
    the training years) are scored in and out.

Every number is a pure function of (matrix, seeds, tuned_params.json): outputs land in model/cache/phase2/
(git-ignored) and a rerun overwrites them with identical values (`shap` refits the winner and asserts it).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from model import train as T
from model.train import POSITIONS, TARGET, Spec

OUT = T.pit.CACHE_DIR / "phase2"
TEST_SEASON = 2025
WEEKS = tuple(range(1, 19))
KEEP = ["player_id", "game_id", "season", "week", "position", "team", "y_played", "y_has_stats_row",
        "spine_depth_only", "is_rookie", "base_trail3_ppr", "base_xfp_sameweek", "prev_season_ppg", "pts_ppr_std_mean"]
STARTABLE_TOP = {"QB": 14, "RB": 30, "WR": 40, "TE": 14}     # who counts as a start/sit-relevant player, by trailing average
QUANTILES = (0.1, 0.5, 0.9)


# --------------------------------------------------------------------------- the walk-forward rule
def train_mask(df: pd.DataFrame, season: int, week: int, spec: Spec = T.PRIMARY, target: str = TARGET) -> np.ndarray:
    """Rows that may train when predicting (season, week): strictly earlier, played, from `spec.min_season` (the first
    matrix season unless a spec says otherwise) and labelled on `target`."""
    earlier = (df["season"] < season) | ((df["season"] == season) & (df["week"] < week))
    m = earlier & (df["season"] >= spec.min_season)
    m &= (df["y_has_stats_row"] == 1) if spec.label_rows == "stats_row" else (df["y_played"] == 1)
    if spec.exclude_depth_only:
        m &= df["spine_depth_only"] == 0
    m &= df[target].notna()
    return m.to_numpy()


def test_mask(df: pd.DataFrame, season: int, week: int) -> np.ndarray:
    return ((df["season"] == season) & (df["week"] == week)).to_numpy()


def assert_walk_forward(df: pd.DataFrame, train: np.ndarray, test: np.ndarray, season: int, week: int) -> None:
    """Raises if any training row is at or after the target week, or the test rows are not exactly that week."""
    tr = df.loc[train, ["season", "week"]]
    if len(tr) == 0:
        raise AssertionError("empty training set")
    key = tr["season"].to_numpy() * 100 + tr["week"].to_numpy()
    if key.max() >= season * 100 + week:
        raise AssertionError(f"look-ahead: a training row is at (season, week) {int(key.max())} >= {season * 100 + week}")
    te = df.loc[test, ["season", "week"]]
    if len(te) == 0 or not ((te["season"] == season) & (te["week"] == week)).all():
        raise AssertionError("test rows are not exactly the target week")
    if (train & test).any():
        raise AssertionError("a row is in both train and test")


def _fit_predict(lib, params, Xtr, ytr, Xte, quantile=None):
    m = T.make_model(lib, params, quantile=quantile)
    m.fit(Xtr, ytr)
    return m, m.predict(Xte)


def walk_forward(df: pd.DataFrame, spec: Spec = T.PRIMARY, lib: str = "lightgbm", params: dict | None = None, *,
                 season: int = TEST_SEASON, weeks=WEEKS, quantile: float | None = None, hook=None,
                 log=None, refit_every: int = 1, target: str = TARGET) -> pd.DataFrame:
    """Predictions for every spine row of each target week (played or not), refitting each week on strictly
    earlier rows. `hook(week, model, X_test, test_index)` is called after each fit (SHAP uses it).
    `refit_every=k` refits only every k-th week (weeks 1, 1+k, ...) and predicts the weeks in between with the last
    fit: a model that is one to k-1 weeks stale, never one that has seen the target week (k=1 is the walk-forward)."""
    params = params if params is not None else T.load_params()["point"][lib]
    dead = T.dead_columns(df)
    cols = T.feature_columns(df, spec, dead)
    X, _ = T.encode(df, cols)
    y = df[target].to_numpy()
    pos = df["position"].to_numpy()
    out = []
    t0 = time.time()
    fitted_at, stale = None, None
    for wk in weeks:
        te = test_mask(df, season, wk)
        if not te.any():
            continue
        if fitted_at is None or (wk - weeks[0]) % refit_every == 0:
            fitted_at = wk
        tr = train_mask(df, season, fitted_at, spec, target)   # strictly earlier than the fit week, hence than `wk`
        assert_walk_forward(df, tr, te, season, wk)
        pred = np.full(int(te.sum()), np.nan)
        if fitted_at != wk and not spec.per_position and stale is not None:
            model, pred = stale[0], stale[0].predict(X[te])
        elif spec.per_position:
            te_pos = pos[te]
            for p in POSITIONS:
                a, b = tr & (pos == p), te_pos == p
                if b.any():
                    _, pred[b] = _fit_predict(lib, params, X[a], y[a], X[te][b], quantile)
            model = None
        else:
            model, pred = _fit_predict(lib, params, X[tr], y[tr], X[te], quantile)
            stale = (model, fitted_at)
        part = df.loc[te, [c for c in KEEP if c in df.columns]].copy()
        part["y"] = df.loc[te, target].to_numpy()
        part["pred"] = pred
        part["n_train"] = int(tr.sum())
        part["fit_week"] = fitted_at
        out.append(part)
        if hook is not None and model is not None:
            hook(wk, model, X[te], df.index[te])
        if log:
            log(f"    {spec.name:24s} {lib:9s} week {wk:2d}  train {int(tr.sum()):6,d}  test {int(te.sum()):4d}  "
                f"{time.time() - t0:6.1f}s")
    res = pd.concat(out)
    res.attrs["features"] = list(X.columns)
    return res


def walk_forward_quantiles(df: pd.DataFrame, spec: Spec = T.PRIMARY, params: dict | None = None, *,
                           season: int = TEST_SEASON, weeks=WEEKS, log=None) -> pd.DataFrame:
    """LightGBM p10/p50/p90, one model per alpha, same rows and features as the point model."""
    doc = T.load_params()
    qp = params or doc["quantile"] or {str(a): {**doc["point"]["lightgbm"]} for a in QUANTILES}
    res = None
    for a in QUANTILES:
        r = walk_forward(df, spec, "lightgbm", qp[str(a)], season=season, weeks=weeks, quantile=a, log=log)
        r = r.rename(columns={"pred": f"q{int(a * 100)}"})
        res = r if res is None else res.assign(**{f"q{int(a * 100)}": r[f"q{int(a * 100)}"]})
    return res


# --------------------------------------------------------------------------- evaluation sets
def eval_set(res: pd.DataFrame, *, need_trail: bool = True, depth_only: bool = True) -> pd.DataFrame:
    """The scored rows: played rows; with a trailing average (the head-to-head set); optionally without the
    depth-chart-only spine rows."""
    e = res[res["y_played"] == 1]
    if need_trail:
        e = e[e["base_trail3_ppr"].notna()]
    if not depth_only:
        e = e[e["spine_depth_only"] == 0]
    return e


# --------------------------------------------------------------------------- metrics
def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return np.nan
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _pairs(y: np.ndarray, p: np.ndarray, sel: np.ndarray | None = None) -> tuple[float, int]:
    """(sum of credit, number of pairs) over same-group pairs with different actuals. Credit 1 if the higher
    prediction scored more, 0.5 for a tied prediction. `sel` restricts to pairs whose members are both selected."""
    n = len(y)
    if n < 2:
        return 0.0, 0
    iu = np.triu_indices(n, 1)
    dy, dp = (y[:, None] - y[None, :])[iu], (p[:, None] - p[None, :])[iu]
    ok = dy != 0
    if sel is not None:
        ok &= (sel[:, None] & sel[None, :])[iu]
    dy, dp = dy[ok], dp[ok]
    credit = float(((np.sign(dy) == np.sign(dp)).sum()) + 0.5 * (dp == 0).sum())
    return credit, int(len(dy))


def weekly_stats(e: pd.DataFrame, pred: str) -> pd.DataFrame:
    """One row per (position, week): n, spearman, pair credit / pairs (all, and startable-only), abs/sq error sums."""
    rows = []
    for (pos, wk), g in e.groupby(["position", "week"], sort=True):
        y, p = g["y"].to_numpy(), g[pred].to_numpy()
        t = g["base_trail3_ppr"].to_numpy()
        top = STARTABLE_TOP[pos]
        rank = pd.Series(-t).rank(method="first").to_numpy()            # 1 = best trailing average
        sel = rank <= top
        c1, n1 = _pairs(y, p)
        c2, n2 = _pairs(y, p, sel)
        err = p - y
        rows.append({"position": pos, "week": wk, "n": len(g), "spearman": _rank_corr(p, y), "credit": c1, "pairs": n1,
                     "credit_start": c2, "pairs_start": n2, "abs": float(np.abs(err).sum()), "sq": float((err ** 2).sum())})
    return pd.DataFrame(rows)


def summarize(ws: pd.DataFrame) -> pd.DataFrame:
    """Per position (plus ALL) from weekly_stats rows."""
    def one(g: pd.DataFrame) -> pd.Series:
        n = g["n"].sum()
        return pd.Series({"n": int(n), "MAE": g["abs"].sum() / n, "RMSE": np.sqrt(g["sq"].sum() / n),
                          "spearman": g["spearman"].mean(), "pick_acc": g["credit"].sum() / max(g["pairs"].sum(), 1),
                          "pick_acc_startable": g["credit_start"].sum() / max(g["pairs_start"].sum(), 1)})
    t = ws.groupby("position").apply(one, include_groups=False).reindex(list(POSITIONS))
    allr = one(ws)
    allr["spearman"] = ws["spearman"].mean()
    t.loc["ALL"] = allr
    return t


def score(e: pd.DataFrame, preds: dict[str, str]) -> dict[str, pd.DataFrame]:
    """{label: per-position summary} for prediction columns `preds` = {label: column} on the identical rows `e`."""
    return {lab: summarize(weekly_stats(e, col)) for lab, col in preds.items()}


def paired_bootstrap(e: pd.DataFrame, a: str, b: str, *, draws: int = 2000, seed: int = T.SEED) -> pd.DataFrame:
    """Week-blocked bootstrap of (a - b): resample the 18 weeks with replacement (players within a week are
    correlated, so rows are never resampled alone). Columns: metric deltas with 95% intervals, per position + ALL.
    Negative dMAE/dRMSE and positive dSpearman mean `a` is better."""
    rng = np.random.default_rng(seed)
    wa, wb = weekly_stats(e, a), weekly_stats(e, b)
    weeks = sorted(e["week"].unique())
    idx = rng.integers(0, len(weeks), size=(draws, len(weeks)))
    out = {}
    for pos in [*[p for p in POSITIONS if p in set(e["position"])], "ALL"]:
        A = wa if pos == "ALL" else wa[wa["position"] == pos]
        B = wb if pos == "ALL" else wb[wb["position"] == pos]
        def by_week(ws, col, agg="sum"):
            return ws.groupby("week")[col].agg(agg).reindex(weeks).fillna(0.0).to_numpy()
        n = by_week(A, "n")
        d = {}
        ma, mb = by_week(A, "abs"), by_week(B, "abs")
        sa, sb = by_week(A, "sq"), by_week(B, "sq")
        # a tie in the resample (a week drawn twice) is fine: sums and counts both double
        N = n[idx].sum(axis=1)
        mae_d = (ma[idx].sum(axis=1) - mb[idx].sum(axis=1)) / N
        rmse_d = np.sqrt(sa[idx].sum(axis=1) / N) - np.sqrt(sb[idx].sum(axis=1) / N)
        spa = A.groupby("week")["spearman"].mean().reindex(weeks).to_numpy()
        spb = B.groupby("week")["spearman"].mean().reindex(weeks).to_numpy()
        sp_d = np.nanmean((spa - spb)[idx], axis=1)
        ca, cb = by_week(A, "credit"), by_week(B, "credit")
        pa = by_week(A, "pairs")
        pk_d = (ca[idx].sum(axis=1) - cb[idx].sum(axis=1)) / pa[idx].sum(axis=1)
        for name, arr, point in (("dMAE", mae_d, (ma.sum() - mb.sum()) / n.sum()),
                                 ("dRMSE", rmse_d, np.sqrt(sa.sum() / n.sum()) - np.sqrt(sb.sum() / n.sum())),
                                 ("dSpearman", sp_d, np.nanmean(spa - spb)),
                                 ("dPick", pk_d, (ca.sum() - cb.sum()) / pa.sum())):
            lo, hi = np.percentile(arr, [2.5, 97.5])
            d[name] = point
            d[name + "_lo"], d[name + "_hi"] = float(lo), float(hi)
        out[pos] = d
    return pd.DataFrame(out).T


# --------------------------------------------------------------------------- quantile scoring
def quantile_calibration(e: pd.DataFrame) -> pd.DataFrame:
    """Empirical coverage per position: share of actuals at or below each quantile (targets .10/.50/.90), the p10-p90
    interval coverage (target .80), mean interval width, pinball loss, and the share of rows whose raw quantiles
    crossed (the scored quantiles are sorted per row; see `sort_quantiles`)."""
    rows = []
    for pos in [*[p for p in POSITIONS if p in set(e["position"])], "ALL"]:
        g = e if pos == "ALL" else e[e["position"] == pos]
        y = g["y"].to_numpy()
        r = {"position": pos, "n": len(g)}
        for a in QUANTILES:
            q = g[f"q{int(a * 100)}"].to_numpy()
            r[f"below_q{int(a * 100)}"] = float((y <= q).mean())
            d = y - q
            r[f"pinball_q{int(a * 100)}"] = float(np.mean(np.maximum(a * d, (a - 1) * d)))
        r["coverage_10_90"] = float(((y >= g["q10"]) & (y <= g["q90"])).mean())
        r["mean_width"] = float((g["q90"] - g["q10"]).mean())
        r["crossed_share"] = float(g["crossed"].mean()) if "crossed" in g else np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index("position")


def sort_quantiles(q: pd.DataFrame) -> pd.DataFrame:
    """Independent quantile models can cross on a row; report the crossing rate, then score the row-sorted values."""
    q = q.copy()
    raw = q[["q10", "q50", "q90"]].to_numpy()
    q["crossed"] = ((raw[:, 0] > raw[:, 1]) | (raw[:, 1] > raw[:, 2])).astype(float)
    q[["q10", "q50", "q90"]] = np.sort(raw, axis=1)
    return q


# --------------------------------------------------------------------------- the experiments
FAMILY_NAMES = ["lags", "prev_season", "td_luck", "role", "injury", "dvp", "vegas", "context", "static", "adp", "college"]
# Joint drops, chosen from the one-at-a-time 2025 table AFTER seeing it: they are reported only next to their 2024
# replication (`replicate`), never as a result on their own.
LEAN = {"lean_A": ("context", "td_luck", "dvp", "college"),
        "lean_B": ("context", "td_luck", "dvp", "college", "static", "injury")}


def specs_for_ablation() -> list[Spec]:
    return ([Spec("stats_row_target", label_rows="stats_row"), Spec("with_season", season=True),
             Spec("with_weather", weather=True),
             Spec("with_roof_structure", weather_cols=("wx_roof_obs", "wx_indoor_obs")),
             Spec("with_temp_wind", weather_cols=("wx_temp_obs", "wx_wind_obs")),
             Spec("train_no_depth_only", exclude_depth_only=True),
             Spec("per_position_models", per_position=True), Spec("drop_week", drop_cols=("week",))]
            + [Spec(f"drop_{f}", drop_families=(f,)) for f in FAMILY_NAMES]
            + [Spec(n, drop_families=fams) for n, fams in LEAN.items()])


def _save(res: pd.DataFrame, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.parquet"
    res.to_parquet(p)
    return p


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(OUT / f"{name}.parquet")


def stage_bakeoff(df, libs=T.LIBS, weeks=WEEKS, force=False, log=print):
    params = T.load_params()["point"]
    for lib in libs:
        f = OUT / f"bakeoff_{lib}.parquet"
        if f.exists() and not force:
            log(f"  {lib}: cached ({f.name})")
            continue
        t0 = time.time()
        res = walk_forward(df, T.PRIMARY, lib, params[lib], weeks=weeks, log=log if weeks != WEEKS else None)
        _save(res, f"bakeoff_{lib}")
        log(f"  {lib}: {len(res):,} rows in {time.time() - t0:,.0f}s")


def stage_ablate(df, lib, specs=None, weeks=WEEKS, force=False, log=print):
    params = T.load_params()["point"][lib]
    for spec in specs or specs_for_ablation():
        f = OUT / f"ablate_{lib}_{spec.name}.parquet"
        if f.exists() and not force:
            log(f"  {spec.name}: cached")
            continue
        t0 = time.time()
        res = walk_forward(df, spec, lib, params, weeks=weeks)
        _save(res, f"ablate_{lib}_{spec.name}")
        log(f"  {spec.name}: {time.time() - t0:,.0f}s")


def stage_replicate(df, lib, season=2024, force=False, log=print):
    """The same ablation table on an earlier walk-forward year (train 2021..season-1 plus `season` weeks < N), the
    out-of-time check on decisions read off 2025: a family only counts as dead weight if it is dead in BOTH years.
    Hyper-parameters were tuned on 2024 (validation), which is a mild, symmetric advantage for every variant."""
    params = T.load_params()["point"][lib]
    for spec in [T.PRIMARY, *specs_for_ablation()]:
        f = OUT / f"rep{season}_{lib}_{spec.name}.parquet"
        if f.exists() and not force:
            log(f"  {season} {spec.name}: cached")
            continue
        t0 = time.time()
        _save(walk_forward(df, spec, lib, params, season=season), f"rep{season}_{lib}_{spec.name}")
        log(f"  {season} {spec.name}: {time.time() - t0:,.0f}s")


def stage_cadence(df, lib="lightgbm", force=False, log=print):
    """How stale can the model be? Refit weekly (the primary), every 4 weeks, or never inside the season (trained on
    2021-2024 only, week 1's fit used for all 18 weeks)."""
    params = T.load_params()["point"][lib]
    for k, name in ((4, "refit_every_4"), (18, "refit_never")):
        f = OUT / f"cadence_{lib}_{name}.parquet"
        if f.exists() and not force:
            log(f"  {name}: cached")
            continue
        _save(walk_forward(df, T.PRIMARY, lib, params, refit_every=k), f"cadence_{lib}_{name}")
        _save(walk_forward(df, T.PRIMARY, lib, params, season=2024, refit_every=k), f"cadence2024_{lib}_{name}")
        log(f"  {name}: done")


def stage_prune(df, lib="lightgbm", share_below=0.0015, force=False, log=print):
    """The pruning test: drop every feature whose OUT-OF-SAMPLE SHAP share on the 2024 walk-forward is under `share_below`,
    then walk 2025 forward with the smaller set. The list comes from 2024, so 2025 is a clean test of it."""
    import json
    from model import explain
    f = OUT / f"prune_{lib}_2025.parquet"
    contrib = pd.read_parquet(OUT / f"shap2024_{lib}.parquet")
    drop = explain.prune_list(contrib, share_below)
    (OUT / "prune_list.json").write_text(json.dumps({"share_below": share_below, "source": f"shap2024_{lib}", "drop": drop}, indent=1))
    log(f"  prune list from 2024 SHAP (share < {share_below:.2%}): {len(drop)} registry columns")
    if f.exists() and (OUT / f"prune_{lib}_noadp_2025.parquet").exists() and not force:
        log("  prune 2025: cached")
        return
    spec = Spec("prune_2024_shap", drop_cols=tuple(drop))
    params = T.load_params()["point"][lib]
    _save(walk_forward(df, spec, lib, params), f"prune_{lib}_2025")
    _save(walk_forward(df, spec, lib, params, season=2024), f"prune_{lib}_2024")
    # the serving candidate for 2026: the pruned set WITHOUT ADP (no preseason 2026 ADP snapshot exists: the fetch returned a
    # 29-player in-season window, see PLAN_MODEL.md), so the live model cannot depend on it
    noadp = Spec("prune_no_adp", drop_cols=tuple(drop), drop_families=("adp",))
    _save(walk_forward(df, noadp, lib, params), f"prune_{lib}_noadp_2025")
    _save(walk_forward(df, noadp, lib, params, season=2024), f"prune_{lib}_noadp_2024")


def stage_quantiles(df, weeks=WEEKS, force=False, log=print):
    f = OUT / "quantiles_lightgbm.parquet"
    if f.exists() and not force:
        log("  quantiles: cached")
        return
    t0 = time.time()
    res = walk_forward_quantiles(df, weeks=weeks)
    _save(res, "quantiles_lightgbm")
    log(f"  quantiles: {len(res):,} rows in {time.time() - t0:,.0f}s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["bakeoff", "ablate", "replicate", "prune", "cadence", "quantiles", "all"])
    ap.add_argument("--lib", default="lightgbm")
    ap.add_argument("--weeks", default="1-18")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--matrix", type=Path, default=T.MATRIX)
    a = ap.parse_args(argv)
    lo, _, hi = a.weeks.partition("-")
    weeks = tuple(range(int(lo), int(hi or lo) + 1))
    df = T.load_matrix(a.matrix)
    if a.stage in ("bakeoff", "all"):
        stage_bakeoff(df, weeks=weeks, force=a.force)
    if a.stage in ("quantiles", "all"):
        stage_quantiles(df, weeks=weeks, force=a.force)
    if a.stage in ("ablate", "all"):
        stage_ablate(df, a.lib, weeks=weeks, force=a.force)
    if a.stage in ("replicate", "all"):
        stage_replicate(df, a.lib, force=a.force)
    if a.stage in ("prune", "all"):
        stage_prune(df, a.lib, force=a.force)
    if a.stage in ("cadence", "all"):
        stage_cadence(df, a.lib, force=a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
