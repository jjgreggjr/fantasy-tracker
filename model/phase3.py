"""Phase 3 validation: the SHIPPED design, walked forward through the code that serves.

    python3.12 -m model.phase3 shipped        # walk-forward 2025 and 2024 of exactly what `model.serve` ships; prints the tables

Phase 2.5 measured the three-library average, the component models and the quantile recalibration on the Phase 2 PRIMARY
inputs (114 encoded columns). The work order ships them on Phase 2's 71-column set (the primary minus 39 low-attribution columns
and ADP), which Phase 2 validated only for a single LightGBM. This stage closes that gap with the same machinery and the same
head-to-head rows as Phase 2 / 2.5: every model of the shipped design is walked forward week by week for 2025 (and 2024 as the
out-of-time replication) with `backtest.walk_forward`, the function `model.serve` calls for one week, then scored against the
Phase 2 primary and the trailing-3 baseline with the week-blocked bootstrap.

Nothing here feeds the pipeline; results are cached under model/cache/phase3/ (git-ignored) and the numbers go in PLAN_MODEL.md.
Tree counts for the components are the frozen ones in serving_config.json (early-stopped on 2024 after training 2021-2023, as in
Phase 2.5), so 2024 carries the same mild in-sample edge for every variant, stated wherever 2024 numbers are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from model import backtest as B
from model import components as C
from model import ensemble as E
from model import leaguescore as L
from model import p25run as P
from model import serve as SV
from model import train as T
from model.train import POSITIONS

OUT = T.pit.CACHE_DIR / "phase3"
SEASONS = (2025, 2024)


def log(msg: str) -> None:
    print(msg, flush=True)


def frame() -> pd.DataFrame:
    """2021-2025 matrix rows from the serving history builder (labels attached, the misc label and each non-PPR league's points label
    derived from the component labels)."""
    store = SV.load_store(2026, refresh=False)
    return SV.add_league_targets(C.add_misc(SV.training_frame(store, 2026, 1, jobs=4, log=log)), L.load_leagues(SV.ROOT))


def _cached(name: str, key: str, fn) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    f, k = OUT / f"{name}.parquet", OUT / f"{name}.key"
    if f.exists() and k.exists() and k.read_text() == key:
        return pd.read_parquet(f)
    res = fn()
    res.to_parquet(f)
    k.write_text(key)
    return res


def _key(df: pd.DataFrame, *parts) -> str:
    ident = int(pd.util.hash_pandas_object(df[["player_id", "game_id"]], index=False).sum() % (2 ** 63))
    shape = (len(df), len([c for c in df.columns if not c.startswith("y_pts_")]))        # the league labels are derived, not part of the matrix
    return hashlib.md5(json.dumps([shape, ident, *parts], sort_keys=True, default=str).encode()).hexdigest()[:16]


def walk(df, spec, lib, params, season, *, target=T.TARGET, quantile=None, name=None) -> pd.DataFrame:
    name = name or f"{target}__{lib}__{season}" + (f"__q{int(quantile * 100)}" if quantile else "")
    t0 = time.time()
    res = _cached(name, _key(df, repr(spec), params, target, lib, season, quantile),
                  lambda: B.walk_forward(df, spec, lib, params, season=season, target=target, quantile=quantile))
    log(f"    {name}: {len(res):,} rows ({time.time() - t0:,.0f}s)")
    return res


def runs(df: pd.DataFrame, season: int, cfg: dict) -> dict:
    """Every model of the shipped design, walked forward through `season`."""
    spec, params = SV.spec_from(cfg), T.load_params()
    base = {k: v for k, v in params["point"]["lightgbm"].items() if k != "n_estimators"}
    out = {"flat": {lib: walk(df, spec, lib, params["point"][lib], season) for lib in T.LIBS},
           "comp": {t: walk(df, spec, "lightgbm", {**base, "n_estimators": cfg["components"][t]["n_estimators"]}, season, target=t)
                    for t in C.TARGETS}}
    return out


def quantiles(df: pd.DataFrame, season: int, cfg: dict, target: str = T.TARGET) -> pd.DataFrame:
    """Sorted q10 / q50 / q90 rows of `season`'s walk-forward on the shipped inputs, for PPR points or a league's own points."""
    spec, params = SV.spec_from(cfg), T.load_params()
    qs = {a: walk(df, spec, "lightgbm", params["quantile"][str(a)], season, target=target, quantile=a) for a in B.QUANTILES}
    res = qs[0.1].rename(columns={"pred": "q10"})
    for a in (0.5, 0.9):
        res[f"q{int(a * 100)}"] = qs[a]["pred"]
    return B.sort_quantiles(res)


def with_pred(template: pd.DataFrame, pred) -> pd.DataFrame:
    out = template.copy()
    out["pred"] = np.asarray(pred, dtype="float64")
    return out


def keyed(df: pd.DataFrame) -> pd.Series:
    return df["player_id"].astype(str) + "|" + df["game_id"].astype(str)


def primary_avg(df: pd.DataFrame, template: pd.DataFrame, season: int) -> dict:
    """The Phase 2 primary (114 inputs) from the Phase 2.5 cache, aligned to `template` rows on (player, game)."""
    m = C.add_misc(T.load_matrix())
    mk = keyed(m)
    out = {}
    for lib in T.LIBS:
        f = P.OUT / f"primary__{lib}__{season}.parquet"
        if not f.exists():
            return {}
        r = pd.read_parquet(f)
        out[lib] = pd.Series(r["pred"].to_numpy(), index=mk.loc[r.index].to_numpy())
    tk = keyed(df.loc[template.index]).to_numpy()
    aligned = {lib: s.reindex(tk).to_numpy() for lib, s in out.items()}
    return {"primary_lgbm": with_pred(template, aligned["lightgbm"]),
            "primary_avg": with_pred(template, np.mean([aligned[l] for l in T.LIBS], axis=0))}


def head_to_head(df: pd.DataFrame, season: int, cfg: dict, r: dict) -> dict:
    flat = r["flat"]
    tmpl = flat["lightgbm"]
    avg = np.mean([flat[l]["pred"].to_numpy() for l in T.LIBS], axis=0)
    comp = {t: x["pred"] for t, x in r["comp"].items()}
    ppr = C.compose(comp, tmpl["position"], C.PPR).to_numpy()
    runs_ = {"shipped_avg": with_pred(tmpl, avg), "shipped_lgbm": with_pred(tmpl, flat["lightgbm"]["pred"]),
             "components_ppr": with_pred(tmpl, ppr), "blend_50_50": with_pred(tmpl, 0.5 * avg + 0.5 * ppr),
             "trailing_3": with_pred(tmpl, tmpl["base_trail3_ppr"]), "xfp_oracle": with_pred(tmpl, tmpl["base_xfp_sameweek"])}
    runs_.update(primary_avg(df, tmpl, season))
    return runs_


def shipped(cfg: dict | None = None, *, draws: int = 2000) -> str:
    cfg = cfg or SV.load_config()
    df = frame()
    lines = [f"# Phase 3: the shipped design, walked forward (71 inputs, serving_config.json)\n"]
    cov = {}
    bands = {}
    for season in (2025, 2024, 2023):
        if season != 2023:
            r = runs(df, season, cfg)
            h = head_to_head(df, season, cfg, r)
            cmp = P.compare(h, "shipped_avg", draws=draws)
            lines.append(f"## {season} walk-forward, head-to-head rows ({cmp['n']:,}: played, with an earlier game)\n")
            lines.append(cmp["scores"].round(3).to_string() + "\n")
            lines.append("Deltas against shipped_avg (run minus shipped; negative dRMSE / dMAE and positive dSpearman / dPick mean the run is BETTER), "
                         "95% week-blocked bootstrap:\n")
            lines.append(cmp["delta"].round(3).to_string() + "\n")
            e = cmp["eval"]
            pos = B.score(e, {n: f"pred_{n}" for n in ("shipped_avg", "components_ppr", "blend_50_50", "trailing_3")})
            for n, s in pos.items():
                lines.append(f"per position, {n}:\n{s.round(3).to_string()}\n")
            bands[season] = (r, h)
        q = quantiles(df, season, cfg)
        cov[season] = q
    for season in (2025, 2024):
        r, h = bands[season]
        q, hist = cov[season], cov[season - 1]
        played = q[q["y_played"] == 1]
        rec = E.recalibrate(q, hist, mode=cfg["recalibration"]["mode"])
        e = rec[(rec["y_played"] == 1) & rec["base_trail3_ppr"].notna()]
        raw = E.coverage_table(e, "q10", "q90", draws=draws)
        new = E.coverage_table(e.assign(q10=e["q10a"], q90=e["q90a"]), "q10", "q90", draws=draws)
        lines.append(f"## {season} p10-p90 band, raw then recalibrated ({cfg['recalibration']['mode']}; history = {season - 1} out-of-sample rows "
                     f"plus this season's earlier weeks), head-to-head rows\n")
        lines.append("raw:\n" + raw.round(3).to_string() + "\n\nrecalibrated:\n" + new.round(3).to_string() + "\n")
        lines.append(f"mean band width raw {(e['q90'] - e['q10']).mean():.2f}, recalibrated {(e['q90a'] - e['q10a']).mean():.2f}; "
                     f"interval score raw {E.interval_score(e, 'q10', 'q90'):.3f}, recalibrated {E.interval_score(e, 'q10a', 'q90a'):.3f}; "
                     f"raw quantiles crossed on {played['crossed'].mean():.1%} of played rows; "
                     f"p10 <= p50 <= p90 after recalibration on {((e['q10a'] <= e['q50']) & (e['q50'] <= e['q90a'])).mean():.1%}\n")
        # the non-PPR leagues: each has quantile models fit on its own points; the shifted-PPR-band shortcut is reported to show why
        comp = {t: x["pred"] for t, x in r["comp"].items()}
        tmpl = r["flat"]["lightgbm"]
        for ls in L.load_leagues(SV.ROOT):
            if ls.is_ppr:
                continue
            tgt = SV.league_target(ls.slug)
            ql, qlh = quantiles(df, season, cfg, tgt), quantiles(df, season - 1, cfg, tgt)
            recl = E.recalibrate(ql, qlh, mode=cfg["recalibration"]["mode"])
            el = recl[(recl["y_played"] == 1) & recl["base_trail3_ppr"].notna()]
            rawl = E.coverage_table(el, "q10", "q90", draws=draws)
            newl = E.coverage_table(el.assign(q10=el["q10a"], q90=el["q90a"]), "q10", "q90", draws=draws)
            lines.append(f"## {season} {ls.slug}: p10-p90 band from quantile models fit on that league's own points (head-to-head rows)\n")
            lines.append("raw:\n" + rawl.round(3).to_string() + "\n\nrecalibrated:\n" + newl.round(3).to_string() + "\n")
            idx = el.index
            actual = C.actual_points(df, idx, ls.scoring)
            delta = C.compose({t: s_.loc[idx] for t, s_ in comp.items()}, tmpl.loc[idx, "position"], ls.scoring) \
                - C.compose({t: s_.loc[idx] for t, s_ in comp.items()}, tmpl.loc[idx, "position"], C.PPR)
            ee = e.loc[idx]
            sh = pd.DataFrame({"position": ee["position"], "y": actual, "lo": ee["q10a"] + delta, "hi": ee["q90a"] + delta}).dropna(subset=["y"])
            cov_sh = ((sh["y"] >= sh["lo"]) & (sh["y"] <= sh["hi"])).groupby(sh["position"]).mean()
            lines.append(f"the rejected shortcut, PPR band moved by the change in the mean: coverage "
                         f"{((sh['y'] >= sh['lo']) & (sh['y'] <= sh['hi'])).mean():.3f} (" + ", ".join(f"{p} {cov_sh[p]:.3f}" for p in POSITIONS if p in cov_sh)
                         + ")\n")
            ranks = pd.DataFrame({"position": el["position"], "week": el["week"], "y": actual, "base_trail3_ppr": el["base_trail3_ppr"],
                                  "components": C.compose({t: s_.loc[idx] for t, s_ in comp.items()}, tmpl.loc[idx, "position"], ls.scoring),
                                  "flat_unadjusted": h["shipped_avg"].loc[idx, "pred"],
                                  "flat_adjusted": h["shipped_avg"].loc[idx, "pred"] + delta}).dropna(subset=["y"])
            sc = B.score(ranks, {n: n for n in ("components", "flat_unadjusted", "flat_adjusted")})
            lines.append(f"{season} {ls.slug}: ranking against that league's points\n"
                         + pd.DataFrame({n: s_.loc["ALL", ["RMSE", "MAE", "spearman", "pick_acc"]] for n, s_ in sc.items()}).T.round(3).to_string() + "\n")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["shipped"])
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args(argv)
    text = shipped(draws=a.draws)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shipped_report.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
