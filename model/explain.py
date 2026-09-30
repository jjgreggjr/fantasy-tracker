"""SHAP for the winning model: which inputs actually move the prediction.

    model/.venv/bin/python -m model.explain --lib lightgbm

Every SHAP value is OUT OF SAMPLE: the walk-forward refits each week of 2025 on strictly earlier data, and the
contributions are computed for that week's held-out rows with that week's model (TreeExplainer, exact for trees;
additivity is asserted: base value + contributions == the model's prediction). Rows scored: 2025 played rows.
The refit reproduces the bake-off predictions exactly, and this stage asserts it (a rerun reproduces every number).

Outputs (git-ignored, model/cache/phase2/): shap_{lib}.parquet, one row per scored 2025 row, one column per
encoded feature, plus `_base`, `_pred`, `_week`. model/report_phase2.py turns them into tables.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from model import backtest as B
from model import train as T


ENC_TO_REGISTRY = {"inj_report_ord": "inj_report_status", "inj_practice_ord": "inj_practice_status",
                   "wx_roof_dome": "wx_roof_obs", "wx_roof_retractable": "wx_roof_obs"}


def shap_pass(df: pd.DataFrame, lib: str, params: dict | None = None, weeks=B.WEEKS, season: int = B.TEST_SEASON,
              log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(contributions frame, predictions frame) for the walk-forward of `lib` on the primary config."""
    import shap
    parts: list[pd.DataFrame] = []

    def hook(week, model, X_test, idx):
        played = (df.loc[idx, "y_played"] == 1).to_numpy()
        if not played.any():
            return
        Xp = X_test[played]
        ex = shap.TreeExplainer(model)
        sv = np.asarray(ex.shap_values(Xp))
        base = float(np.ravel(ex.expected_value)[0])
        pred = model.predict(Xp)
        gap = float(np.abs(sv.sum(axis=1) + base - pred).max())
        if gap > 1e-3:
            raise AssertionError(f"SHAP is not additive for {lib} week {week}: max gap {gap:.5f}")
        part = pd.DataFrame(sv, columns=list(Xp.columns), index=Xp.index)
        part["_base"], part["_pred"], part["_week"] = base, pred, week
        parts.append(part)

    res = B.walk_forward(df, T.PRIMARY, lib, params, weeks=weeks, season=season, hook=hook)
    return pd.concat(parts), res


def prune_list(contrib: pd.DataFrame, share_below: float = 0.0015) -> list[str]:
    """Registry columns whose out-of-sample SHAP share is under `share_below` of the total (position one-hots are never
    pruned; a registry column behind several encoded columns goes only if all of them qualify)."""
    feats = [c for c in contrib.columns if not c.startswith("_")]
    share = contrib[feats].abs().mean()
    share = share / share.sum()
    by_reg: dict[str, float] = {}
    for enc, v in share.items():
        if enc.startswith("pos_"):
            continue
        reg = ENC_TO_REGISTRY.get(enc, enc)
        by_reg[reg] = max(by_reg.get(reg, 0.0), float(v))
    return sorted(c for c, v in by_reg.items() if v < share_below)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lib", required=True, choices=T.LIBS)
    ap.add_argument("--season", type=int, default=B.TEST_SEASON, help="walk-forward year (2024 feeds the pruning test)")
    ap.add_argument("--matrix", type=Path, default=T.MATRIX)
    a = ap.parse_args(argv)
    df = T.load_matrix(a.matrix)
    t0 = time.time()
    contrib, res = shap_pass(df, a.lib, season=a.season)
    tag = "" if a.season == B.TEST_SEASON else str(a.season)
    cached = B.OUT / (f"bakeoff_{a.lib}.parquet" if a.season == B.TEST_SEASON else f"rep{a.season}_{a.lib}_primary.parquet")
    if cached.exists():
        old = pd.read_parquet(cached)
        np.testing.assert_array_equal(old["pred"].to_numpy(), res["pred"].to_numpy(),
                                      err_msg="a refit did not reproduce the stored predictions")
        print("reproducibility: refit predictions are bit-identical to the stored bake-off run")
    B.OUT.mkdir(parents=True, exist_ok=True)
    contrib.to_parquet(B.OUT / f"shap{tag}_{a.lib}.parquet")
    print(f"{len(contrib):,} rows x {contrib.shape[1] - 3} features in {time.time() - t0:,.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
