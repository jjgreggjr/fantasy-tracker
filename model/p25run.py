"""Shared plumbing for the Phase 2.5 experiments: cached walk-forward runs, per-target tree counts, and the
head-to-head comparison tables.

Everything here is a thin layer over model/backtest.py (the walk-forward, the metrics, the week-blocked bootstrap) so a
Phase 2.5 number is computed exactly like a Phase 2 one: the same 2025 played rows with an earlier game (the "head-to-head"
rows), the same fixed seeds, the same 18-week block bootstrap. Results are cached under model/cache/phase25/ (git-ignored),
keyed by a hash of the matrix, the spec, the parameters and the target, so a rebuilt matrix or a changed configuration
recomputes instead of silently reusing a stale file, and a rerun with nothing changed returns identical bytes.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from model import backtest as B
from model import train as T
from model.train import POSITIONS, Spec

OUT = T.pit.CACHE_DIR / "phase25"
TEST_SEASONS = (2025, 2024)          # the test year first, the out-of-time replication second
KIND = ("RMSE", "MAE", "spearman", "pick_acc")


def matrix_key(df: pd.DataFrame) -> str:
    """A short fingerprint of the design matrix: shape, columns, row identity and the sum of every float column."""
    ident = int(pd.util.hash_pandas_object(df[["player_id", "game_id"]], index=False).sum() % (2 ** 63))
    total = float(np.nansum(df.select_dtypes("float64").to_numpy(dtype="float64")))
    return hashlib.md5(f"{df.shape}|{list(df.columns)}|{ident}|{total!r}".encode()).hexdigest()[:16]


def _key(*parts) -> str:
    return hashlib.md5(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def cached(name: str, key: str, fn, *, force: bool = False) -> pd.DataFrame:
    """`fn()` once per (name, key): parquet plus a `.key` sidecar. A different key recomputes and overwrites."""
    OUT.mkdir(parents=True, exist_ok=True)
    f, k = OUT / f"{name}.parquet", OUT / f"{name}.key"
    if not force and f.exists() and k.exists() and k.read_text() == key:
        return pd.read_parquet(f)
    res = fn()
    res.to_parquet(f)
    k.write_text(key)
    return res


def load_matrix(path: Path | None = None) -> tuple[pd.DataFrame, str]:
    df = T.load_matrix(path or T.MATRIX)
    return df, matrix_key(df)


def wf(df: pd.DataFrame, mkey: str, spec: Spec, season: int, *, target: str = T.TARGET, lib: str = "lightgbm",
       params: dict | None = None, quantile: float | None = None, name: str | None = None, force: bool = False,
       log=None) -> pd.DataFrame:
    """One cached walk-forward: every week of `season`, refit weekly on strictly earlier rows (backtest.walk_forward)."""
    p = params if params is not None else T.load_params()["point"][lib]
    name = name or f"{target}__{spec.name}__{lib}__{season}" + (f"__q{int(quantile * 100)}" if quantile else "")
    key = _key(mkey, repr(spec), p, target, lib, season, quantile)
    t0 = time.time()

    def run():
        return B.walk_forward(df, spec, lib, p, season=season, target=target, quantile=quantile)
    res = cached(name, key, run, force=force)
    if log:
        log(f"    {name}: {len(res):,} rows ({time.time() - t0:,.0f}s)")
    return res


def tree_params(df: pd.DataFrame, mkey: str, target: str, spec: Spec, *, force: bool = False) -> dict:
    """The tuned LightGBM shape with a tree count picked for `target`: early stopping on 2024 after training on 2021-2023
    (the same protocol `train.tune` used for the quantile models, x1.1 because the walk-forward trains on more data).
    2025 is never seen. The 2024 replication therefore has a mild in-sample edge on the tree count (its validation
    year), which is symmetric across every variant compared and is stated wherever 2024 numbers are reported."""
    f = OUT / "tree_counts.json"
    OUT.mkdir(parents=True, exist_ok=True)
    book = json.loads(f.read_text()) if f.exists() else {}
    key = _key(mkey, target, repr(spec))
    base = {k: v for k, v in T.load_params()["point"]["lightgbm"].items() if k != "n_estimators"}
    if force or key not in book:
        dead = T.dead_columns(df)
        cols = T.feature_columns(df, spec, dead)
        played = df[(df["y_played"] == 1) & df[target].notna() & (df["season"] >= spec.min_season)]
        tr = played[played["season"].isin(T.TRAIN_SEASONS_FOR_TUNING)]
        va = played[played["season"] == T.VALID_SEASON]
        Xtr, _ = T.encode(tr, cols)
        Xva, _ = T.encode(va, cols)
        _, n, loss = T._fit_with_early_stopping("lightgbm", base, Xtr, tr[target].to_numpy(), Xva, va[target].to_numpy())
        book[key] = {"target": target, "spec": spec.name, "best_iteration": int(n), "valid_rmse": round(float(loss), 4)}
        f.write_text(json.dumps(book, indent=1, sort_keys=True))
    return {**base, "n_estimators": max(50, int(round(book[key]["best_iteration"] * T.FOLLOW_UP_TREE_FACTOR)))}


# --------------------------------------------------------------------------- head-to-head tables
def h2h(res: dict[str, pd.DataFrame], base: str) -> pd.DataFrame:
    """The scored rows: the base run's head-to-head set (played, with an earlier game) with every run's `pred` attached as
    `pred_<name>` (aligned on the matrix index, which every walk-forward output keeps)."""
    e = B.eval_set(res[base]).copy()
    for n, r in res.items():
        if n != base:
            if not e.index.isin(r.index).all():
                raise ValueError(f"run {n!r} does not cover every scored row of {base!r}")
            e[f"pred_{n}"] = r.loc[e.index, "pred"]
    e[f"pred_{base}"] = res[base].loc[e.index, "pred"]
    return e


def compare(res: dict[str, pd.DataFrame], base: str, *, draws: int = 2000) -> dict[str, pd.DataFrame]:
    """{'scores': one row per run (ALL positions: RMSE, MAE, Spearman, pick accuracy), 'delta': run minus base with 95%
    week-blocked bootstrap intervals (ALL), 'delta_pos': the same per position, 'n': rows}. Negative dRMSE/dMAE and
    positive dSpearman/dPick mean the run beats the base."""
    e = h2h(res, base)
    names = list(res)
    summ = B.score(e, {n: f"pred_{n}" for n in names})
    scores = pd.DataFrame({n: s.loc["ALL", list(KIND)] for n, s in summ.items()}).T
    per_pos = {n: s for n, s in summ.items()}
    delta, delta_pos = {}, {}
    for n in names:
        if n == base:
            continue
        bt = B.paired_bootstrap(e, f"pred_{n}", f"pred_{base}", draws=draws)
        delta[n] = bt.loc["ALL"]
        delta_pos[n] = bt
    return {"scores": scores, "delta": pd.DataFrame(delta).T, "delta_pos": delta_pos, "n": len(e), "per_pos": per_pos,
            "eval": e}


def verdict_rmse(d25: pd.Series, d24: pd.Series) -> str:
    """The adoption rule, mechanically: beats the base on 2025 (pooled dRMSE < 0) AND the same sign on 2024."""
    if d25["dRMSE"] < 0 and d24["dRMSE"] < 0:
        strength = "significant in 2025" if d25["dRMSE_hi"] < 0 else "2025 interval spans zero"
        return f"SHIPS ({strength})"
    return "does not ship"


def verdict_tie(d25: pd.Series, d24: pd.Series) -> str:
    """The component-model rule: a tie within noise ships, so does a win. Fails only if either year's 95% interval for
    dRMSE lies entirely above zero (significantly worse)."""
    if d25["dRMSE_lo"] > 0 or d24["dRMSE_lo"] > 0:
        return "does not ship (significantly worse in " + ("2025" if d25["dRMSE_lo"] > 0 else "2024") + ")"
    if d25["dRMSE_hi"] < 0 and d24["dRMSE_hi"] < 0:
        return "SHIPS (better in both years)"
    return "SHIPS (tie within noise)"
