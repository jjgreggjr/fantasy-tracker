"""Phase 2.5 experiment runner: volume-first experiments on top of the Phase 2 primary.

    model/.venv/bin/python -m model.build_features        # the matrix with the Phase 2.5 columns (pbp_*, part_*, team_*, eff_*, y_*)
    model/.venv/bin/python -m model.phase25 reproduce     # the primary from the rebuilt matrix equals Phase 2, bit for bit
    model/.venv/bin/python -m model.phase25 exp1          # play-by-play micro-signals added to the flat model
    model/.venv/bin/python -m model.phase25 exp2          # two-stage: volume first, then points
    model/.venv/bin/python -m model.phase25 exp3          # component models composed to PPR and to a TE-premium scoring
    model/.venv/bin/python -m model.phase25 exp4          # three-library ensemble, quantile recalibration
    model/.venv/bin/python -m model.report_phase25        # writes model/reports/phase25_experiments.md from the cached runs

Every experiment is scored twice with the Phase 2 machinery (same head-to-head rows, same seeds, same week-blocked bootstrap):
on the 2025 walk-forward and on the 2024 walk-forward, and the adoption rule is applied to the pair. A variant that gets its
own tree count (`p25run.tree_params`) gets it by early stopping on 2024 after training on 2021-2023, the protocol that
produced the primary's count; nothing ever sees a 2025 label before the 2025 walk-forward scores that week.
Outputs are cached under model/cache/phase25/ (git-ignored); the stages are idempotent.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from model import backtest as B
from model import components as C
from model import ensemble as E
from model import features as F
from model import p25run as P
from model import train as T
from model import volume as V
from model.train import POSITIONS, Spec

PHASE2_CACHE = T.pit.CACHE_DIR / "phase2"
SLUG = "we-can-think-of-something-funny"      # the dynasty league whose TE premium is the alternate scoring


def load(path: Path | None = None) -> tuple[pd.DataFrame, str]:
    """The Phase 2.5 matrix plus the derived misc-points label."""
    df, _ = P.load_matrix(path)
    df = C.add_misc(df)
    return df, P.matrix_key(df)


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- the primary and the reproduction check
def primary(df: pd.DataFrame, mkey: str, season: int, *, lib: str = "lightgbm") -> pd.DataFrame:
    return P.wf(df, mkey, T.PRIMARY, season, lib=lib, name=f"primary__{lib}__{season}")


def stage_reproduce(df: pd.DataFrame, mkey: str) -> dict:
    """The rebuilt matrix must give back Phase 2's predictions exactly: recompute the primary and compare with the cached Phase 2
    walk-forwards (2025: bakeoff_<lib>; 2024: rep2024_lightgbm_primary). Also that the primary's tree count is what the tree-count
    protocol gives, so per-variant counts are comparable to it."""
    out = {}
    for season, lib in ((2025, "lightgbm"), (2025, "xgboost"), (2025, "catboost"), (2024, "lightgbm")):
        old = pd.read_parquet(PHASE2_CACHE / (f"bakeoff_{lib}.parquet" if season == 2025 else "rep2024_lightgbm_primary.parquet"))
        new = primary(df, mkey, season, lib=lib)
        same_idx = old.index.equals(new.index)
        diff = float(np.abs(old["pred"].to_numpy() - new["pred"].to_numpy()).max()) if same_idx else float("nan")
        out[f"{lib}_{season}"] = {"rows": len(new), "same_index": bool(same_idx), "max_abs_pred_diff": diff}
        log(f"  primary {lib} {season}: {len(new):,} rows, same index {same_idx}, max |pred diff| {diff}")
    tp = P.tree_params(df, mkey, T.TARGET, T.PRIMARY)["n_estimators"]
    out["tree_count_protocol"] = {"protocol": tp, "phase2": T.load_params()["point"]["lightgbm"]["n_estimators"]}
    log(f"  tree count by protocol {tp} vs Phase 2 {out['tree_count_protocol']['phase2']}")
    P.OUT.mkdir(parents=True, exist_ok=True)
    (P.OUT / "reproduce.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    return out


# --------------------------------------------------------------------------- experiment 1: play-by-play micro-signals
EXP1 = {
    "pbp_usage": Spec("p25_pbp_usage", add_families=("pbp_usage",)),
    "pbp": Spec("p25_pbp", add_families=("pbp_usage", "pbp_team")),                       # the servable candidate
    "pbp_team": Spec("p25_pbp_team", add_families=("pbp_team",)),                         # diagnostic
    "pbp_part": Spec("p25_pbp_part", add_families=("pbp_usage", "pbp_team", "pbp_part")),  # backtest-only (participation)
    "part_only": Spec("p25_part_only", add_families=("pbp_part",)),                      # diagnostic
    # does play-by-play usage substitute for the lagged xFP columns it overlaps with? (diagnostic pair)
    "no_xfp": Spec("p25_no_xfp", drop_cols=V.XFP_LAGS),
    "no_xfp_pbp": Spec("p25_no_xfp_pbp", drop_cols=V.XFP_LAGS, add_families=("pbp_usage", "pbp_team")),
}


def exp1_runs(df: pd.DataFrame, mkey: str, season: int, *, force: bool = False) -> dict[str, pd.DataFrame]:
    res = {"primary": primary(df, mkey, season)}
    for name, spec in EXP1.items():
        params = P.tree_params(df, mkey, T.TARGET, spec)
        res[name] = P.wf(df, mkey, spec, season, params=params, force=force, log=log)
    return res


def stage_exp1(df, mkey):
    for season in P.TEST_SEASONS:
        exp1_runs(df, mkey, season)


def importance(df: pd.DataFrame, spec: Spec, *, season: int = 2025, week: int = 18, params: dict | None = None) -> pd.DataFrame:
    """LightGBM gain importance of the last walk-forward fit of `season` (trained on everything before `week`)."""
    params = params or T.load_params()["point"]["lightgbm"]
    cols = T.feature_columns(df, spec, T.dead_columns(df))
    X, fam = T.encode(df, cols)
    tr = B.train_mask(df, season, week, spec)
    m = T.make_model("lightgbm", params)
    m.fit(X[tr], df.loc[tr, T.TARGET].to_numpy())
    gain = m.booster_.feature_importance("gain")
    t = pd.DataFrame({"feature": X.columns, "family": [fam[c] for c in X.columns], "gain": gain})
    t["share"] = t["gain"] / t["gain"].sum()
    return t.sort_values("gain", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- experiment 2: two-stage
def exp2_frames(df: pd.DataFrame, mkey: str, *, log=log) -> tuple[pd.DataFrame, str]:
    s1 = V.honest_series(df, mkey, log=log)
    d2 = V.with_stage_one(df, s1)
    return d2, P.matrix_key(d2)


def exp2_runs(d2: pd.DataFrame, k2: str, season: int) -> dict[str, pd.DataFrame]:
    res = {"primary": primary(d2, k2, season)}       # the primary does not read the s1 columns; cached under this frame's key
    for name, spec in V.SPECS.items():
        params = P.tree_params(d2, k2, T.TARGET, spec)
        res[name] = P.wf(d2, k2, spec, season, params=params, log=log)
    return res


def stage_exp2(df, mkey):
    d2, k2 = exp2_frames(df, mkey)
    for season in P.TEST_SEASONS:
        exp2_runs(d2, k2, season)
    for target in V.S1:
        for season in P.TEST_SEASONS:
            P.wf(df, mkey, T.PRIMARY, season, target=target, params=P.tree_params(df, mkey, target, T.PRIMARY), log=log)


# --------------------------------------------------------------------------- experiment 3: component models
COMP_SPECS = {"components": T.PRIMARY, "components_eff": Spec("p25_comp_eff", add_families=("eff",))}


def exp3_runs(df: pd.DataFrame, mkey: str, season: int) -> dict[str, dict[str, pd.DataFrame]]:
    return {name: C.component_runs(df, mkey, spec, season, log=log) for name, spec in COMP_SPECS.items()}


def stage_exp3(df, mkey):
    for season in P.TEST_SEASONS:
        primary(df, mkey, season)
        exp3_runs(df, mkey, season)


def trailing_receptions(df: pd.DataFrame) -> pd.Series:
    """A BASELINE (never an input): the mean of a player's receptions over his previous <=3 appearances this season, from the
    matrix's own earlier rows. Used to give the flat PPR model a TE premium the naive way (PPR + bonus * recent catches)."""
    d = df[df["y_played"] == 1].sort_values(["player_id", "season", "week"], kind="mergesort")
    r = d.groupby(["player_id", "season"])["y_rec"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    return r.reindex(df.index)


# --------------------------------------------------------------------------- experiment 4: cheap wins
def libs_runs(df: pd.DataFrame, mkey: str, season: int) -> dict[str, pd.DataFrame]:
    return {lib: primary(df, mkey, season, lib=lib) for lib in T.LIBS}


def quantile_run(df: pd.DataFrame, mkey: str, season: int) -> pd.DataFrame:
    """LightGBM p10/p50/p90 walk-forward of `season` on the primary inputs, quantiles sorted per row (crossing rate kept)."""
    key = P._key(mkey, "quantiles", season, T.load_params()["quantile"])
    res = P.cached(f"quantiles__{season}", key, lambda: B.sort_quantiles(B.walk_forward_quantiles(df, season=season)))
    return res


def stage_exp4(df, mkey):
    for season in (2023, 2024, 2025):
        libs_runs(df, mkey, season)
        quantile_run(df, mkey, season)


STAGES = {"reproduce": stage_reproduce, "exp1": stage_exp1, "exp2": stage_exp2, "exp3": stage_exp3, "exp4": stage_exp4}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=[*STAGES, "all"])
    ap.add_argument("--matrix", type=Path, default=T.MATRIX)
    a = ap.parse_args(argv)
    df, mkey = load(a.matrix)
    log(f"matrix {df.shape}, key {mkey}")
    t0 = time.time()
    for name in (list(STAGES) if a.stage == "all" else [a.stage]):
        log(f"== {name}")
        STAGES[name](df, mkey)
        log(f"   {name} done, {time.time() - t0:,.0f}s elapsed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
