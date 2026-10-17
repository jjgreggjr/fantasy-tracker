"""Per-column predictability screen: every column outside the frozen primary's inputs, ranked in one run.

    python -m model.screen                          # opt-in families, pruned columns, dropped families (weather and season stay out)
    python -m model.screen --families streak,eff    # only these families (a named family is screened even if it is weather)
    python -m model.screen --years 2022-2025 --top 40 --force

What it measures, per column, on the head-to-head rows of `--years` (played, with an earlier game that season: `backtest.eval_set`):

    rho_y      Spearman with actual PPR points within (season, week, position), averaged over those groups.
    rho_resid  the same against the PRIMARY's out-of-sample residual (actual minus the frozen LightGBM primary's walk-forward
               prediction; trial.py's cached runs, the missing years computed once and cached the same way). This is the
               number that matters: a column that tracks what the model already predicts is redundant, one that tracks what
               it misses is new information. t_resid is the mean over groups divided by its standard error.
    gain       SCREEN-GRADE. The column's LightGBM gain share when added ALONE to the primary spec, from ONE fit per year
               (train 2021..y-1, predict year y, y = 2024 and 2025; tuned parameters and tree count, no retree, no weekly
               refit), and dRMSE = that fit minus the same single fit of the primary, on year y's head-to-head rows.
               With colsample_bytree < 1 adding ANY column reshuffles the sampled features, so a few seeded placebo columns
               (pure noise) go through the same fits: their dRMSE spread is the noise floor of this step.
    null       share missing on the screened rows, and on the 2026 head-to-head rows (serving skew: all-NaN live is useless).

Verdicts, first match wins: SKEW (2026 null rate >= 50%), PROMISING (|rho_resid| in the top decile of the screened columns AND
dRMSE < 0 in both years), UNCONFIRMED (top-decile |rho_resid| but the single fits do not both improve), REDUNDANT
(|rho_y| >= 0.10, |rho_resid| below the top decile), NOISE (both low). DEAD = no information in 2021-2024 (`train.dead_columns`).

A screen ranks; it never adopts. The PROMISING set goes through `python -m model.trial` (the full walk-forward and the
Phase 2.5 adoption rule); the report prints that exact command. Opt-in and dropped-family columns can only enter a trial as
their whole family (`+family:`), pruned columns as `+cols:`.

Outputs: model/reports/screen.md (rewritten each run) and model/cache/screen/<matrix_key>.parquet (the raw numbers);
single fits are cached in model/cache/screen/fits_<matrix_key>.parquet so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from model import backtest as B
from model import features as F
from model import train as T
from model import trial as TR

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "screen.md"
OUT = T.pit.CACHE_DIR / "screen"
YEARS = (2022, 2025)
GAIN_YEARS = (2024, 2025)
LIVE = TR.LIVE_SEASON
LIB = "lightgbm"
MIN_GROUP = 8                  # a (season, week, position) group needs this many non-null rows to count
TOP_DECILE = 0.90
RHO_Y_HIGH = 0.10
SKEW_NULL = 0.50
N_PLACEBO = 5
PLACEBO_SEED = 20261016
THREADS = 1                    # `main` sets train.THREADS (every fit, incl. the residual walk-forwards) and OMP_NUM_THREADS to this
VERSION = 1                   # bump when the arithmetic of a fit changes (invalidates the fit cache)
GROUP_KEYS = ["season", "week", "position"]

log = TR.log


# --------------------------------------------------------------------------- which columns
def candidates(df: pd.DataFrame, cfg: dict, families: tuple[str, ...] | None = None) -> pd.DataFrame:
    """One row per registry column to screen: column, family, status (trial.column_status), kind.
    kind: 'candidate' (outside the inputs), 'in' (already an input: reported as a reference, no gain fit), 'dead'.
    Default set: every registry column outside the inputs except weather (post-game) and `season`; with `families`,
    exactly those families' columns (in-input ones included as references)."""
    spec, inputs = TR.serving_spec(cfg), set(cfg["inputs"])
    dead = set(T.dead_columns(df))
    rows = []
    for fam, cols in F.FAMILIES.items():
        if families is not None and fam not in families:
            continue
        for c in cols:
            if c not in df.columns:
                continue
            st = TR.column_status(c, fam, spec, inputs)
            if families is None and (st == "IN" or fam == "weather" or c == "season"):
                continue
            kind = "in" if st == "IN" else ("dead" if c in dead else "candidate")
            rows.append({"column": c, "family": fam, "status": st, "kind": kind})
    return pd.DataFrame(rows, columns=["column", "family", "status", "kind"])


def numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """The column as the model sees it (`train.encode`: categoricals to ordinals / one-hots), position one-hots removed."""
    X, _ = T.encode(df[["position", col]], [col])
    return X[[c for c in X.columns if not c.startswith("pos_")]]


def placebos(index: pd.Index, n: int = N_PLACEBO, seed: int = PLACEBO_SEED) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {f"placebo_{i}": rng.normal(size=len(index)).astype("float32") for i in range(1, n + 1)}


# --------------------------------------------------------------------------- the correlations
def within_spearman(x, y, groups, min_n: int = MIN_GROUP) -> tuple[float, int, float]:
    """(mean Spearman over groups, number of groups used, t = mean / standard error). Rows with x or y missing are dropped
    pairwise; a group needs `min_n` rows and variation in both; ties take average ranks."""
    d = pd.DataFrame({"x": np.asarray(x, dtype="float64"), "y": np.asarray(y, dtype="float64"), "g": np.asarray(groups)})
    d = d.dropna()
    if d.empty:
        return np.nan, 0, np.nan
    g = d.groupby("g", sort=False)
    rx, ry = g["x"].rank(), g["y"].rank()
    dx = rx - rx.groupby(d["g"]).transform("mean")
    dy = ry - ry.groupby(d["g"]).transform("mean")
    agg = pd.DataFrame({"xy": dx * dy, "xx": dx * dx, "yy": dy * dy, "g": d["g"]}).groupby("g").sum()
    n = g.size()
    ok = (n.reindex(agg.index) >= min_n) & (agg["xx"] > 1e-12) & (agg["yy"] > 1e-12)
    r = (agg.loc[ok, "xy"] / np.sqrt(agg.loc[ok, "xx"] * agg.loc[ok, "yy"])).to_numpy()
    k = len(r)
    if k == 0:
        return np.nan, 0, np.nan
    m = float(r.mean())
    se = float(r.std(ddof=1) / np.sqrt(k)) if k > 1 else np.nan
    if k < 2:
        return m, k, np.nan
    return m, k, (m / se if se > 0 else (np.sign(m) * np.inf if m else np.nan))


def group_ids(df: pd.DataFrame) -> np.ndarray:
    return (df["season"].astype(str) + "_" + df["week"].astype(str) + "_" + df["position"].astype(str)).to_numpy()


def null_rate(x) -> float:
    x = np.asarray(x, dtype="float64")
    return float(np.isnan(x).mean()) if len(x) else np.nan


def rho_table(values: dict[str, pd.DataFrame | np.ndarray], df: pd.DataFrame, rows: np.ndarray, resid: np.ndarray,
              live: np.ndarray) -> pd.DataFrame:
    """rho_y / rho_resid / t / null rates for each {name: encoded values over all of df} on the screened `rows` (bool mask);
    `resid` is aligned with df (NaN off the screened rows); `live` masks the 2026 head-to-head rows. A registry column behind
    several encoded columns keeps the encoded part with the largest |rho_resid|."""
    sub = df.loc[rows]
    groups = group_ids(sub)
    y = sub[T.TARGET].to_numpy("float64")
    r = np.asarray(resid, dtype="float64")[rows]
    out = []
    for name, v in values.items():
        parts = v if isinstance(v, pd.DataFrame) else pd.DataFrame({name: np.asarray(v)}, index=df.index)
        best = None
        for enc in parts.columns:
            arr = parts[enc].to_numpy("float64")
            ry, ky, _ = within_spearman(arr[rows], y, groups)
            rr, kr, tr = within_spearman(arr[rows], r, groups)
            rec = {"column": name, "encoded": enc, "rho_y": ry, "rho_resid": rr, "t_resid": tr, "n_groups": kr,
                   "null": null_rate(arr[rows]), "null_2026": null_rate(arr[live]) if live.any() else np.nan}
            if best is None or (np.nan_to_num(abs(rr), nan=-1) > np.nan_to_num(abs(best["rho_resid"]), nan=-1)):
                best = rec
        out.append(best)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- the primary's residual
def primary_residual(df: pd.DataFrame, mkey: str, base, seasons, *, force: bool = False, log=log) -> pd.Series:
    """actual minus the frozen primary's out-of-sample walk-forward prediction, indexed like df (NaN where not scored).
    Uses trial.py's cached walk-forwards (model/cache/trials/<spec>__lightgbm__<season>), computing a missing season once."""
    have = TR.seasons_with_labels(df)
    resid = pd.Series(np.nan, index=df.index)
    for s in seasons:
        weeks = have.get(s, ())
        if not weeks:
            log(f"  {s}: no labelled weeks, no residual")
            continue
        res = TR.run_variant(df, mkey, None, base, LIB, s, weeks, primary=True, force=force, log=log)
        e = B.eval_set(res)
        resid.loc[e.index] = (e["y"] - e["pred"]).to_numpy()
    return resid


# --------------------------------------------------------------------------- the single-fit gain step
def _fit_key(mkey, base_cols, params, year, column) -> str:
    return hashlib.md5(json.dumps([VERSION, mkey, base_cols, params, year, column], default=str).encode()).hexdigest()[:16]


def single_fit(Xtr, ytr, Xte, params) -> tuple[np.ndarray, np.ndarray]:
    """(predictions, gain per column) for one LightGBM fit on `train.THREADS` threads (`main` sets 1: the fit is ~2 s alone,
    while on a shared, oversubscribed machine four OpenMP threads busy-waiting made one fit take minutes)."""
    m = T.make_model(LIB, {**params, "n_jobs": T.THREADS})
    m.fit(Xtr, ytr)
    return m.predict(Xte), m.booster_.feature_importance(importance_type="gain")


def gain_screen(df: pd.DataFrame, mkey: str, base_cols: list[str], extra: dict[str, pd.DataFrame], rows_by_year: dict,
                params: dict, *, fit=single_fit, cache: Path | None = None, force: bool = False, log=log) -> pd.DataFrame:
    """One row per (column, year): gain share, RMSE of the fit with the column, RMSE of the primary's same single fit, dRMSE.
    `extra` = {name: encoded columns over df}; `rows_by_year` = {year: bool mask of that year's head-to-head rows}."""
    book = pd.read_parquet(cache) if (cache is not None and cache.exists() and not force) else pd.DataFrame()
    done = set(book["key"]) if len(book) else set()
    new: list[dict] = []
    Xall, _ = T.encode(df, base_cols)
    y = df[T.TARGET].to_numpy("float64")
    t0, n_fit = time.time(), 0

    def flush():
        nonlocal book, new
        if cache is not None and new:
            book = pd.concat([book, pd.DataFrame(new)], ignore_index=True)
            new = []
            cache.parent.mkdir(parents=True, exist_ok=True)
            book.to_parquet(cache)

    out = []
    for year in GAIN_YEARS:
        tr = B.train_mask(df, year, 1)
        te = rows_by_year.get(year)
        if te is None or not te.any():
            continue
        yte = y[te]
        base_name = "__primary__"
        jobs = [(base_name, None)] + list(extra.items())
        base_rmse = None
        for name, enc in jobs:
            key = _fit_key(mkey, base_cols, params, year, name)
            hit = book[book["key"] == key] if key in done else None
            if hit is not None and len(hit):
                rec = hit.iloc[0].to_dict()
            else:
                Xtr, Xte = Xall[tr], Xall[te]
                if enc is not None:
                    Xtr = pd.concat([Xtr, enc[tr].set_axis(Xtr.index)], axis=1)
                    Xte = pd.concat([Xte, enc[te].set_axis(Xte.index)], axis=1)
                pred, gain = fit(Xtr, y[tr], Xte, params)
                share = float(gain[-enc.shape[1]:].sum() / gain.sum()) if enc is not None and gain.sum() > 0 else np.nan
                rec = {"key": key, "column": name, "year": year, "gain": share,
                       "rmse": float(np.sqrt(np.mean((pred - yte) ** 2)))}
                new.append(rec)
                n_fit += 1
                if n_fit % 10 == 0:
                    flush()
                    log(f"    {n_fit} fits, {time.time() - t0:,.0f}s (last: {year} {name})")
            if name == base_name:
                base_rmse = rec["rmse"]
                continue
            out.append({"column": name, "year": year, "gain": rec["gain"], "rmse": rec["rmse"], "rmse_primary": base_rmse,
                        "drmse": rec["rmse"] - base_rmse})
    flush()
    return pd.DataFrame(out, columns=["column", "year", "gain", "rmse", "rmse_primary", "drmse"])


# --------------------------------------------------------------------------- verdicts
VERDICTS = ("PROMISING", "UNCONFIRMED", "REDUNDANT", "NOISE", "SKEW", "DEAD", "IN (reference)", "PLACEBO")


def assign_verdicts(t: pd.DataFrame, *, top: float = TOP_DECILE, rho_y_high: float = RHO_Y_HIGH,
                    skew: float = SKEW_NULL) -> tuple[pd.Series, float]:
    """(verdict per row, the |rho_resid| top-decile threshold). The threshold is taken over kind == 'candidate' rows only.
    Needs columns kind, rho_y, rho_resid, null_2026 and drmse_<year> for each GAIN_YEARS."""
    cand = t["kind"] == "candidate"
    a = t["rho_resid"].abs()
    thr = float(a[cand].quantile(top)) if a[cand].notna().any() else np.inf
    hi_resid = a >= thr
    better = np.logical_and.reduce([t[f"drmse_{y}"].fillna(np.inf) < 0 for y in GAIN_YEARS])
    v = []
    for i in range(len(t)):
        row = t.iloc[i]
        if row["kind"] == "placebo":
            v.append("PLACEBO")
        elif row["kind"] == "dead":
            v.append("DEAD")
        elif row["kind"] == "in":
            v.append("IN (reference)")
        elif pd.notna(row["null_2026"]) and row["null_2026"] >= skew:
            v.append("SKEW")
        elif bool(hi_resid.iloc[i]) and bool(better[i]):
            v.append("PROMISING")
        elif bool(hi_resid.iloc[i]):
            v.append("UNCONFIRMED")
        elif pd.notna(row["rho_y"]) and abs(row["rho_y"]) >= rho_y_high:
            v.append("REDUNDANT")
        else:
            v.append("NOISE")
    return pd.Series(v, index=t.index), thr


def trial_candidates(rows: pd.DataFrame, base) -> tuple[list[str], list[str]]:
    """(candidate strings for `python -m model.trial`, notes). Pruned columns -> one +cols:; an opt-in or dropped family ->
    +family: (trial.py adds those only whole); weather cannot be served and is left out."""
    add_cols, fams, notes = [], [], []
    for c, f in zip(rows["column"], rows["family"]):
        if c in base.drop_cols:
            add_cols.append(c)
        elif f in T.OPT_IN_FAMILIES or f in base.drop_families:
            if f not in fams:
                fams.append(f)
        else:
            notes.append(f"{c} ({f}) cannot be added by trial.py")
    for f in fams:
        k = int((rows["family"] == f).sum())
        notes.append(f"+family:{f} adds all {len(F.FAMILIES[f])} {f} columns ({k} screened PROMISING): trial.py has no per-column add for it")
    cands = [f"+family:{f}" for f in fams] + ([f"+cols:{','.join(add_cols)}"] if add_cols else [])
    if cands:
        TR.compose(cands, base)             # raises if trial.py would refuse the string
    return cands, notes


def trial_command(cands: list[str]) -> str:
    return "python -m model.trial " + " ".join(f"--candidate {c}" for c in cands) if cands else ""


# --------------------------------------------------------------------------- the report
def _f(x, fmt="{:+.3f}") -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)


def _pct(x) -> str:
    return _f(x, "{:.1%}") if pd.notna(x) else ""


def table_md(t: pd.DataFrame) -> str:
    """Grouped by family (families in order of their best |rho_resid|), rows by |rho_resid| desc."""
    t = t.assign(_a=t["rho_resid"].abs())
    order = t.groupby("family")["_a"].max().sort_values(ascending=False).index
    head = ["family", "column", "rho_resid", "t", "rho_y", "gain 2024", "gain 2025", "dRMSE 2024", "dRMSE 2025",
            "null", "null 2026", "verdict"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for fam in order:
        for _, r in t[t["family"] == fam].sort_values("_a", ascending=False).iterrows():
            lines.append("| " + " | ".join([
                fam, f"`{r['column']}`", _f(r["rho_resid"]), _f(r["t_resid"], "{:+.1f}"), _f(r["rho_y"]),
                _pct(r.get("gain_2024")), _pct(r.get("gain_2025")), _f(r.get("drmse_2024"), "{:+.4f}"),
                _f(r.get("drmse_2025"), "{:+.4f}"), _pct(r["null"]), _pct(r["null_2026"]), r["verdict"]]) + " |")
    return "\n".join(lines)


def reading(t: pd.DataFrame, thr: float) -> str:
    real = t[t["kind"] == "candidate"]
    counts = real["verdict"].value_counts()
    cnt = ", ".join(f"{counts[v]} {v}" for v in VERDICTS if v in counts)
    pl = t[t["kind"] == "placebo"]
    noise = ""
    if len(pl) and pl[[f"drmse_{y}" for y in GAIN_YEARS]].notna().any().any():
        d = pl[[f"drmse_{y}" for y in GAIN_YEARS]].to_numpy().ravel()
        noise = (f" The {len(pl)} placebo columns (pure noise) move the single-fit RMSE by {np.nanmin(d):+.4f} to {np.nanmax(d):+.4f} "
                 f"and their |rho_resid| peaks at {pl['rho_resid'].abs().max():.3f}: a dRMSE or rho inside those bands is not evidence.")
    fam = real.groupby("family")["rho_resid"].apply(lambda s: s.abs().median()).sort_values(ascending=False)
    fam_txt = ", ".join(f"{f} {v:.3f}" for f, v in fam.head(4).items())
    top = real.loc[real["rho_resid"].abs().sort_values(ascending=False).index].head(3)
    top_txt = ", ".join(f"`{c}` {r:+.3f} (rho_y {ry:+.3f})" for c, r, ry in zip(top["column"], top["rho_resid"], top["rho_y"]))
    prom = real[real["verdict"] == "PROMISING"]["column"].tolist()
    return (f"{len(real)} columns screened: {cnt}. The top-decile cut is |rho_resid| >= {thr:.3f}; the strongest residual signals are "
            f"{top_txt}. By family, median |rho_resid|: {fam_txt}.{noise} "
            + (f"PROMISING: {', '.join(f'`{c}`' for c in prom)}; that set still has to pass `model.trial` (full walk-forward, 2025 "
               f"interval below zero, 2024 same sign) before it means anything." if prom else
               "Nothing is PROMISING: no column both tracks the primary's misses (top decile) and lowers the single-fit RMSE in both "
               "years with a usable 2026 feed.")
            + " Spearman is rank-only and within position-week; a column the trees could use through an interaction can show a "
              "small rho, which is why the gain step exists.")


def write_report(t: pd.DataFrame, thr: float, meta: dict, path: Path | None = None, *, top: int = 40) -> Path:
    path = Path(path or REPORT)
    real = t[t["kind"] != "placebo"]
    shown = real.loc[real["rho_resid"].abs().sort_values(ascending=False).index].head(top)
    prom = real[real["verdict"] == "PROMISING"]
    shown = pd.concat([shown, prom[~prom.index.isin(shown.index)]])
    cands, notes = trial_candidates(prom, meta["base"]) if len(prom) else ([], [])
    cmd = trial_command(cands)
    pl = t[t["kind"] == "placebo"]
    lines = [
        "# Predictability screen",
        "",
        f"Run {meta['date']} (`python -m model.screen{meta.get('argv', '')}`), {meta['runtime']}. Matrix key `{meta['mkey']}` "
        f"({meta['shape'][0]:,} rows x {meta['shape'][1]} columns). Screened rows: {meta['n_rows']:,} head-to-head rows of "
        f"{meta['years'][0]}-{meta['years'][1]} (played, with an earlier game that season); 2026 null rates on {meta['n_live']:,} "
        f"2026 head-to-head rows (weeks {meta['live_weeks']}).",
        "",
        "`rho_resid` / `rho_y`: Spearman with the frozen LightGBM primary's out-of-sample residual / with actual PPR, within "
        "(season, week, position), mean over groups; `t` = that mean over its standard error. `gain` / `dRMSE` are **screen-grade**: "
        "one fit per year on 2021..y-1 with the column added alone (tuned parameters, no retree, no weekly refit), gain share of "
        "that column and RMSE minus the primary's same single fit on year y's head-to-head rows. Verdicts: PROMISING (top-decile "
        "|rho_resid|, dRMSE < 0 both years, 2026 null < 50%), UNCONFIRMED (top decile, single fits not both better), REDUNDANT "
        f"(|rho_y| >= {RHO_Y_HIGH}), NOISE, SKEW (2026 null >= 50%), DEAD (no information 2021-2024). The top {top} by "
        "|rho_resid| (plus every PROMISING) are shown, grouped by family; every number is in "
        f"`{TR._rel(meta['out'])}`.",
        "",
        table_md(shown),
        "",
    ]
    if len(pl):
        lines += ["Noise floor (placebo columns, same fits): " + "; ".join(
            f"{c} rho_resid {r:+.3f}, dRMSE {a:+.4f} / {b:+.4f}" for c, r, a, b in
            zip(pl["column"], pl["rho_resid"], pl["drmse_2024"], pl["drmse_2025"])) + ".", ""]
    lines += ["## Reading", "", reading(t, thr), "", "## PROMISING set", ""]
    if cmd:
        lines += ["Columns: " + ", ".join(f"`{c}`" for c in prom["column"]) + ".", "", "```", cmd, "```", ""]
        lines += [f"* {n}" for n in notes] + ([""] if notes else [])
    else:
        lines += ["None.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- the run
def run(df: pd.DataFrame, mkey: str, cfg: dict, *, families=None, years=YEARS, force: bool = False,
        resid: pd.Series | None = None, fit=single_fit, n_placebo: int = N_PLACEBO, cache_dir: Path | None = None,
        log=log) -> tuple[pd.DataFrame, float]:
    """The screen table (one row per screened column plus placebos) and the top-decile threshold."""
    base = TR.serving_spec(cfg)
    cand = candidates(df, cfg, families)
    seasons = tuple(range(years[0], years[1] + 1))
    played_h2h = (df["y_played"] == 1) & df["base_trail3_ppr"].notna() & df[T.TARGET].notna()
    rows = (played_h2h & df["season"].isin(seasons)).to_numpy()
    live = (played_h2h & (df["season"] == LIVE)).to_numpy()
    if resid is None:
        log(f"primary residuals {seasons[0]}-{seasons[-1]} (cached walk-forwards)")
        resid = primary_residual(df, mkey, base, seasons, force=force, log=log)
    rows &= resid.notna().to_numpy()
    values: dict[str, pd.DataFrame | np.ndarray] = {c: numeric(df, c) for c in cand["column"]}
    pl = placebos(df.index, n_placebo)
    log(f"correlations: {len(values)} columns + {len(pl)} placebos on {int(rows.sum()):,} rows")
    t = rho_table({**values, **pl}, df, rows, resid.to_numpy(), live)
    meta_rows = pd.concat([cand, pd.DataFrame({"column": list(pl), "family": "placebo", "status": "placebo", "kind": "placebo"})],
                          ignore_index=True)
    t = meta_rows.merge(t, on="column", how="left")
    # gain step: candidates and placebos only
    base_cols = T.feature_columns(df, base, T.dead_columns(df))
    extra = {c: values[c] for c in cand.loc[cand["kind"] == "candidate", "column"]}
    extra.update({k: pd.DataFrame({k: v}, index=df.index) for k, v in pl.items()})
    by_year = {y: (played_h2h & (df["season"] == y)).to_numpy() for y in GAIN_YEARS}
    params = T.load_params()["point"][LIB]
    log(f"single fits: {len(extra)} columns x {len(GAIN_YEARS)} years (+ the primary)")
    cache = (cache_dir or OUT) / f"fits_{mkey}.parquet"
    g = gain_screen(df, mkey, base_cols, extra, by_year, params, fit=fit, cache=cache, force=force, log=log)
    for y in GAIN_YEARS:
        gy = g[g["year"] == y].set_index("column")
        t[f"gain_{y}"] = t["column"].map(gy["gain"])
        t[f"drmse_{y}"] = t["column"].map(gy["drmse"])
        t[f"rmse_primary_{y}"] = gy["rmse_primary"].iloc[0] if len(gy) else np.nan
    t["verdict"], thr = assign_verdicts(t)
    t = t.assign(_a=t["rho_resid"].abs()).sort_values(["family", "_a"], ascending=[True, False]).drop(columns="_a")
    return t.reset_index(drop=True), thr


def parse_years(s: str) -> tuple[int, int]:
    lo, _, hi = s.partition("-")
    a, b = int(lo), int(hi or lo)
    if a > b or b >= LIVE:
        raise argparse.ArgumentTypeError(f"--years {s}: need lo-hi with hi < {LIVE}")
    return a, b


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--families", default=None, help="comma list: screen only these families (in-input columns shown as references)")
    ap.add_argument("--years", type=parse_years, default=YEARS, help="seasons of the correlation rows (default 2022-2025)")
    ap.add_argument("--top", type=int, default=40, help="rows shown in the report table (PROMISING always shown)")
    ap.add_argument("--matrix", type=Path, default=None)
    ap.add_argument("--force", action="store_true", help="recompute the residual walk-forwards and every single fit")
    ap.add_argument("--report", type=Path, default=None, help=f"default {TR._rel(REPORT)}")
    ap.add_argument("--threads", type=int, default=THREADS, help="LightGBM/OpenMP threads (default 1: the machine is shared)")
    a = ap.parse_args(argv)
    os.environ["OMP_NUM_THREADS"] = str(a.threads)       # before lightgbm is imported (it is imported lazily, at the first fit)
    T.THREADS = a.threads
    fams = None
    if a.families:
        fams = tuple(f.strip() for f in a.families.split(",") if f.strip())
        bad = [f for f in fams if f not in F.FAMILIES]
        if bad:
            ap.error(f"unknown families {bad}; families: {', '.join(F.FAMILIES)}")
    t0 = time.time()
    cfg = TR.load_config()
    df, mkey, src = TR.load_frame(a.matrix)
    log(f"frame {df.shape} from {src}, key {mkey}")
    t, thr = run(df, mkey, cfg, families=fams, years=a.years, force=a.force)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{mkey}.parquet"
    t.to_parquet(out)
    played_h2h = (df["y_played"] == 1) & df["base_trail3_ppr"].notna() & df[T.TARGET].notna()
    live_w = sorted(df.loc[played_h2h & (df["season"] == LIVE), "week"].unique())
    meta = {"date": date.today().isoformat(), "mkey": mkey, "shape": df.shape, "years": a.years, "base": TR.serving_spec(cfg),
            "n_rows": int((played_h2h & df["season"].between(*a.years)).sum()), "n_live": int((played_h2h & (df["season"] == LIVE)).sum()),
            "live_weeks": f"{live_w[0]}-{live_w[-1]}" if live_w else "none", "out": out,
            "runtime": f"{(time.time() - t0) / 60:,.1f} min", "argv": (" " + " ".join(argv if argv is not None else sys.argv[1:])).rstrip()}
    path = write_report(t, thr, meta, a.report, top=a.top)
    show = t[t["kind"] != "placebo"]
    print(table_md(show.loc[show["rho_resid"].abs().sort_values(ascending=False).index].head(15)))
    prom = t[t["verdict"] == "PROMISING"]
    print(f"\nPROMISING ({len(prom)}): {', '.join(prom['column']) or '-'}")
    if len(prom):
        print(trial_command(trial_candidates(prom, TR.serving_spec(cfg))[0]))
    print(f"wrote {TR._rel(path)} and {TR._rel(out)}  ({time.time() - t0:,.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
