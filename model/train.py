"""Phase 2 modelling code: the design matrix, the three boosted-tree libraries, and the tuning search.

    model/.venv/bin/python -m model.train --tune          # random search on 2024, writes model/tuned_params.json

Nothing here reads a raw table or a label from the future: it consumes `model/cache/features.parquet` (built
through the leakage gate by model.build_features) and knows only which columns are inputs. The walk-forward
itself (who trains on what) is model/backtest.py.

Design decisions, each with its reason (PLAN_MODEL.md "Phase 2 findings" has the evidence):

  * One model, position as a feature (four one-hot columns), evaluated per position.
  * Inputs = the registered feature families minus the weather family (observed post-game values cannot exist
    at prediction time) minus `season` (a calendar index a tree cannot extrapolate: 2026 is never seen).
    `week` stays. Columns that are all-NA in the 2021-2024 training years are dropped (`dead_columns`).
  * Categoricals are collapsed to ordinals whose meaning cannot drift between seasons: own report status
    none / Questionable / Doubtful-or-Out, practice status none / full / limited / did-not-practice.
    (The mix of Out/Doubtful designations changed in 2025; on played rows Out is ~0% anyway.)
  * Same squared-error objective for all three libraries; the point estimate is the conditional MEAN, which is
    the start/sit currency. LightGBM also fits p10/p50/p90 with the pinball loss.
  * Every library gets the same seed and thread count; LightGBM runs `deterministic=True, force_row_wise=True`.
    A rerun reproduces every prediction (tests/test_backtest.py checks it).
  * Tuning is a small seeded random search that trains on 2021-2023 and validates on 2024 ONLY. 2025 is never
    seen before the walk-forward. Early stopping on the 2024 validation year fixes the tree count; the
    walk-forward then uses that count x 1.1 (it trains on more data), fixed for every weekly refit.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from model import features as F
from model import point_in_time as pit

SEED = 20260929
THREADS = 4
MATRIX = pit.CACHE_DIR / "features.parquet"
PARAMS_PATH = Path(__file__).resolve().parent / "tuned_params.json"
TARGET = "y_points_ppr"
POSITIONS = ("QB", "RB", "WR", "TE")
LIBS = ("lightgbm", "xgboost", "catboost")
FIRST_SEASON = 2021
TRAIN_SEASONS_FOR_TUNING = (2021, 2022, 2023)
VALID_SEASON = 2024
FOLLOW_UP_TREE_FACTOR = 1.1

# Phase 2.5 families exist in the matrix but are inputs only when a Spec asks for them (`add_families`), so the Phase 2
# primary keeps exactly its Phase 2 inputs and its results reproduce.
OPT_IN_FAMILIES = ("pbp_usage", "pbp_team", "pbp_part", "eff")

REPORT_ORD = {"Questionable": 1.0, "Doubtful": 2.0, "Out": 2.0}          # 'Note' and anything unknown -> 0 (no designation)
PRACTICE_ORD = {"Full Participation in Practice": 1.0, "Limited Participation in Practice": 2.0,
                "Did Not Participate In Practice": 3.0}
NOT_INPUTS = ("y_", "base_", "spine_")


def load_matrix(path: Path = MATRIX) -> pd.DataFrame:
    if not Path(path).exists():
        raise SystemExit(f"{path} is missing: run `model/.venv/bin/python -m model.build_features` first")
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- the design matrix
@dataclass(frozen=True)
class Spec:
    """One model configuration. Frozen so it can key a results table."""
    name: str
    weather: bool = False                      # the wx_* family (backtest-only proxy): ablation only
    weather_cols: tuple = ()                   # ...or just these wx_* columns (roof structure is knowable, temp/wind are not)
    season: bool = False                       # `season` as a feature: ablation only
    drop_families: tuple = ()
    add_families: tuple = ()                   # opt-in families (OPT_IN_FAMILIES) added to the inputs
    drop_cols: tuple = ()
    label_rows: str = "played"                 # 'played' (y_played == 1) | 'stats_row' (y_has_stats_row == 1)
    exclude_depth_only: bool = False           # drop spine_depth_only rows from TRAINING
    per_position: bool = False                 # one model per position instead of one with position as a feature
    only_cols: tuple = ()                      # if set, bypass the registry: exactly these matrix columns (+ position one-hots)
    extra_cols: tuple = ()                     # matrix columns outside the registry (e.g. stage-one predictions), appended
    min_season: int = FIRST_SEASON             # first season whose rows may train
    extra: dict = field(default_factory=dict, compare=False, hash=False)


PRIMARY = Spec("primary")


def feature_columns(df: pd.DataFrame, spec: Spec = PRIMARY, dead: tuple = ()) -> list[str]:
    """Registry columns that enter the model for `spec` (categoricals still raw; `encode` expands them)."""
    cols: list[str] = []
    if spec.only_cols:
        cols = [c for c in spec.only_cols if c in df.columns and c not in dead]
    for fam, cs in ([] if spec.only_cols else F.FAMILIES.items()):
        if fam == "weather" and not (spec.weather or spec.weather_cols):
            continue
        if fam in OPT_IN_FAMILIES and fam not in spec.add_families:
            continue
        if fam in spec.drop_families:
            continue
        for c in cs:
            if fam == "weather" and not spec.weather and c not in spec.weather_cols:
                continue
            if c == "season" and not spec.season:
                continue
            if c in spec.drop_cols or c in dead or c not in df.columns or c in cols:
                continue
            cols.append(c)
    cols += [c for c in spec.extra_cols if c in df.columns and c not in cols]
    bad = [c for c in cols if c.startswith(NOT_INPUTS) or c in F.IDENTITY and c not in ("season", "week")]
    if bad:
        raise ValueError(f"not inputs: {bad}")
    return cols


def dead_columns(df: pd.DataFrame) -> tuple[str, ...]:
    """Feature columns with no information in the training years (2021-2024, played rows): all-NA or constant."""
    tr = df[(df["season"] <= VALID_SEASON) & (df["y_played"] == 1)]
    out = []
    for c in F.feature_columns():
        if c not in tr.columns or c == "season":
            continue
        s = tr[c]
        if s.notna().sum() == 0 or s.nunique(dropna=True) <= 1:
            out.append(c)
    return tuple(out)


def encode(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Numeric float32 design matrix and {encoded column: family}. Position is always present as one-hots."""
    X: dict[str, np.ndarray] = {}
    fam: dict[str, str] = {}
    for p in POSITIONS:
        X[f"pos_{p}"] = (df["position"] == p).to_numpy("float32")
        fam[f"pos_{p}"] = "position"
    for c in cols:
        f = F.family_of(c) or "other"
        if c == "inj_report_status":
            X["inj_report_ord"] = df[c].map(REPORT_ORD).fillna(0.0).to_numpy("float32")
            fam["inj_report_ord"] = f
        elif c == "inj_practice_status":
            X["inj_practice_ord"] = df[c].map(PRACTICE_ORD).fillna(0.0).to_numpy("float32")
            fam["inj_practice_ord"] = f
        elif c == "wx_roof_obs":
            r = df[c]
            for level in ("dome", "retractable"):
                X[f"wx_roof_{level}"] = np.where(r.isna(), np.nan, (r == level).astype("float32")).astype("float32")
                fam[f"wx_roof_{level}"] = f
        else:
            X[c] = df[c].to_numpy("float32")
            fam[c] = f
    return pd.DataFrame(X, index=df.index), fam


# --------------------------------------------------------------------------- the libraries
def make_model(lib: str, params: dict, *, quantile: float | None = None, seed: int = SEED):
    """An unfitted sklearn-style regressor. `params` holds the tuned hyper-parameters plus `n_estimators`."""
    p = dict(params)
    if lib == "lightgbm":
        import lightgbm as lgb
        base = dict(objective="regression" if quantile is None else "quantile", random_state=seed, n_jobs=THREADS,
                    deterministic=True, force_row_wise=True, verbose=-1, subsample_freq=1)
        if quantile is not None:
            base["alpha"] = quantile
        return lgb.LGBMRegressor(**{**base, **p})
    if lib == "xgboost":
        import xgboost as xgb
        if quantile is not None:
            raise ValueError("quantile models are LightGBM only")
        return xgb.XGBRegressor(**{**dict(tree_method="hist", objective="reg:squarederror", random_state=seed,
                                          n_jobs=THREADS, verbosity=0), **p})
    if lib == "catboost":
        from catboost import CatBoostRegressor
        if quantile is not None:
            raise ValueError("quantile models are LightGBM only")
        return CatBoostRegressor(**{**dict(loss_function="RMSE", random_seed=seed, thread_count=THREADS, verbose=0,
                                           allow_writing_files=False), **p})
    raise ValueError(lib)


def default_params(lib: str) -> dict:
    """Reasonable untuned starting points (config #0 of the search, and the fallback when no tuning file exists)."""
    return {"lightgbm": dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                             colsample_bytree=0.7, subsample=0.8, reg_lambda=5.0),
            "xgboost": dict(n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=10,
                            colsample_bytree=0.7, subsample=0.8, reg_lambda=5.0),
            "catboost": dict(iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=5.0)}[lib]


def load_params(path: Path = PARAMS_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"point": {lib: default_params(lib) for lib in LIBS}, "quantile": {}, "meta": {"tuned": False}}


# --------------------------------------------------------------------------- tuning
SPACE = {
    "lightgbm": dict(num_leaves=[7, 15, 31, 63], min_child_samples=[20, 50, 100, 200], colsample_bytree=[0.4, 0.6, 0.8, 1.0],
                     subsample=[0.6, 0.8, 1.0], reg_lambda=[0.0, 5.0, 20.0, 50.0], learning_rate=[0.02, 0.03, 0.05]),
    "xgboost": dict(max_depth=[2, 3, 4, 6, 8], min_child_weight=[1, 5, 20, 50], colsample_bytree=[0.4, 0.6, 0.8, 1.0],
                    subsample=[0.6, 0.8, 1.0], reg_lambda=[1.0, 5.0, 20.0, 50.0], learning_rate=[0.02, 0.03, 0.05]),
    "catboost": dict(depth=[3, 4, 6, 8], l2_leaf_reg=[1.0, 3.0, 10.0, 30.0], learning_rate=[0.03, 0.05, 0.08],
                     rsm=[0.5, 0.8, 1.0]),
}
MAX_TREES = {"lightgbm": 1500, "xgboost": 1500, "catboost": 1500}


def _sample_configs(lib: str, n: int, seed: int = SEED) -> list[dict]:
    rng = random.Random(f"{seed}-{lib}")
    cfgs = [dict(default_params(lib))]
    for _ in range(n - 1):
        cfgs.append({k: rng.choice(v) for k, v in SPACE[lib].items()})
    return cfgs


def _fit_with_early_stopping(lib: str, params: dict, Xtr, ytr, Xva, yva, *, quantile: float | None = None):
    """(model, best number of trees, validation loss). The 2024 validation year is the ONLY thing early stopping sees."""
    p = {k: v for k, v in params.items() if k not in ("n_estimators", "iterations")}
    if lib == "lightgbm":
        import lightgbm as lgb
        m = make_model(lib, {**p, "n_estimators": MAX_TREES[lib]}, quantile=quantile)
        metric = "quantile" if quantile is not None else "rmse"
        m.fit(Xtr, ytr, eval_X=Xva, eval_y=yva, eval_metric=metric,
              callbacks=[lgb.early_stopping(100, verbose=False)])
        return m, int(m.best_iteration_), float(m.best_score_["valid_0"][metric])
    if lib == "xgboost":
        m = make_model(lib, {**p, "n_estimators": MAX_TREES[lib], "early_stopping_rounds": 100})
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return m, int(m.best_iteration) + 1, float(m.best_score)
    m = make_model(lib, {**p, "iterations": MAX_TREES[lib], "od_type": "Iter", "od_wait": 100, "use_best_model": True})
    m.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=False)
    return m, int(m.get_best_iteration()) + 1, float(m.get_best_score()["validation"]["RMSE"])


def tune(df: pd.DataFrame, n_configs: int = 10, libs=LIBS, log=print) -> dict:
    """Seeded random search per library on train = 2021-2023, validate = 2024 (played rows). Returns the
    params document (also the content of tuned_params.json)."""
    dead = dead_columns(df)
    cols = feature_columns(df, PRIMARY, dead)
    played = df[df["y_played"] == 1]
    tr = played[played["season"].isin(TRAIN_SEASONS_FOR_TUNING)]
    va = played[played["season"] == VALID_SEASON]
    Xtr, fam = encode(tr, cols)
    Xva, _ = encode(va, cols)
    ytr, yva = tr[TARGET].to_numpy(), va[TARGET].to_numpy()
    doc = load_params() if PARAMS_PATH.exists() else {"point": {}, "quantile": {}, "meta": {}}
    doc["meta"] = {"tuned": True, "train": list(TRAIN_SEASONS_FOR_TUNING), "valid": VALID_SEASON,
                   "n_configs": n_configs, "seed": SEED, "n_train": int(len(tr)), "n_valid": int(len(va)),
                   "features": list(Xtr.columns), "dead_columns_dropped": list(dead), "search_log": {}}
    for lib in libs:
        best = None
        rows = []
        for i, cfg in enumerate(_sample_configs(lib, n_configs)):
            t0 = time.time()
            _, n_trees, loss = _fit_with_early_stopping(lib, cfg, Xtr, ytr, Xva, yva)
            rows.append({"config": i, "valid_rmse": round(loss, 4), "trees": n_trees, "seconds": round(time.time() - t0, 1),
                         **{k: v for k, v in cfg.items() if k not in ("n_estimators", "iterations")}})
            log(f"  {lib:9s} #{i:<2d} rmse {loss:.4f}  trees {n_trees:4d}  {time.time() - t0:5.1f}s  "
                f"{ {k: v for k, v in cfg.items() if k not in ('n_estimators', 'iterations')} }")
            if best is None or loss < best[0]:
                best = (loss, n_trees, cfg)
        loss, n_trees, cfg = best
        key = "iterations" if lib == "catboost" else "n_estimators"
        final = {**{k: v for k, v in cfg.items() if k not in ("n_estimators", "iterations")},
                 key: max(50, int(round(n_trees * FOLLOW_UP_TREE_FACTOR)))}
        doc["point"][lib] = final
        doc["meta"]["search_log"][lib] = rows
        doc["meta"].setdefault("best_valid_rmse", {})[lib] = round(loss, 4)
        log(f"-> {lib}: {final}  (valid rmse {loss:.4f})")
    # quantile LightGBM: the tuned point-model shape, tree count from the pinball loss on 2024 per alpha
    lg = {k: v for k, v in doc["point"]["lightgbm"].items() if k != "n_estimators"} if "lightgbm" in doc["point"] else None
    if lg is not None:
        doc["quantile"] = {}
        for a in (0.1, 0.5, 0.9):
            _, n_trees, loss = _fit_with_early_stopping("lightgbm", lg, Xtr, ytr, Xva, yva, quantile=a)
            doc["quantile"][str(a)] = {**lg, "n_estimators": max(50, int(round(n_trees * FOLLOW_UP_TREE_FACTOR)))}
            log(f"-> quantile {a}: trees {n_trees} (valid pinball {loss:.4f})")
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tune", action="store_true", help="run the random search and write tuned_params.json")
    ap.add_argument("--configs", type=int, default=10)
    ap.add_argument("--libs", default=",".join(LIBS))
    ap.add_argument("--matrix", type=Path, default=MATRIX)
    a = ap.parse_args(argv)
    if not a.tune:
        ap.print_help()
        return 0
    df = load_matrix(a.matrix)
    t0 = time.time()
    doc = tune(df, a.configs, tuple(a.libs.split(",")))
    PARAMS_PATH.write_text(json.dumps(doc, indent=1, default=str))
    print(f"wrote {PARAMS_PATH} in {time.time() - t0:,.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
