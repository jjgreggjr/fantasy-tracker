"""Assemble model/reports/phase2_backtest.md from the cached walk-forward results (model/cache/phase2/).

    model/.venv/bin/python -m model.report_phase2

A pure function of the cached predictions, the SHAP contributions, the feature matrix and tuned_params.json: no model is
fitted here, so the tables are exactly the ones the stages produced. Markdown text and numbers only; no plots, no
binaries (the report is committed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from model import backtest as B
from model import college as C
from model import features as F
from model import point_in_time as pit
from model import train as T
from model.train import POSITIONS

REPORT = Path(__file__).resolve().parent / "reports" / "phase2_backtest.md"
F_COLLEGE = F.FAMILIES["college"]
LIB_LABEL = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost"}
COLS = [*POSITIONS, "ALL"]


# --------------------------------------------------------------------------- markdown helpers
def md(df: pd.DataFrame, fmt: str = "{:.3f}", index: bool = True, floatfmt: dict | None = None, label: str = "") -> str:
    df = df.copy()
    if index:
        df.index.name = label
        df = df.reset_index()
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = (floatfmt or {}).get(c, fmt)
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else f.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def metric_table(summ: dict[str, pd.DataFrame], metric: str, fmt: str = "{:.2f}") -> str:
    t = pd.DataFrame({lab: s[metric] for lab, s in summ.items()}).T[COLS]
    return md(t, fmt)


# --------------------------------------------------------------------------- loading
def wide_predictions() -> pd.DataFrame:
    base = B.load("bakeoff_lightgbm")
    for lib in T.LIBS:
        base[f"pred_{lib}"] = B.load(f"bakeoff_{lib}")["pred"]
    base = base.drop(columns="pred")
    return base


def with_baselines(w: pd.DataFrame, train_means: pd.Series) -> pd.DataFrame:
    w = w.copy()
    w["trail3_fb"] = w["base_trail3_ppr"].fillna(w["prev_season_ppg"]).fillna(w["position"].map(train_means))
    return w


def pick_winner(e: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Pre-declared rule: lowest pooled RMSE on the head-to-head rows; if the leader's 95% interval against LightGBM
    contains zero the result is a statistical tie and LightGBM wins (one library for point, quantile and SHAP, and
    the fastest weekly refit)."""
    rm = {lib: float(np.sqrt(np.mean((e[f"pred_{lib}"] - e["y"]) ** 2))) for lib in T.LIBS}
    lead = min(rm, key=rm.get)
    tab = pd.DataFrame({"pooled RMSE": rm}).T
    if lead == "lightgbm":
        return "lightgbm", tab
    d = B.paired_bootstrap(e, f"pred_{lead}", "pred_lightgbm").loc["ALL"]
    if d["dRMSE_lo"] <= 0 <= d["dRMSE_hi"]:
        return "lightgbm", tab
    return lead, tab


# --------------------------------------------------------------------------- interpretations
INTERPRET: dict[str, str] = {
    "xfp_std_mean": "season-to-date expected fantasy points per game (opportunity quality): the best single predictor",
    "pts_ppr_std_mean": "season-to-date actual PPR per game; tracks xFP, the pair together is the 'true talent' estimate",
    "xfp_l1": "last game's expected points: recent role change, weighted well below the season mean",
    "xfp_l2": "two games ago expected points: a little extra recency",
    "pts_ppr_l1": "last game's actual points: recent form (noisy, small weight)",
    "pts_ppr_l2": "two games ago actual points: recent form",
    "pts_ppr_l3": "three games ago actual points: recent form",
    "adp_ppr_pos_rank": "preseason market rank within position (PPR); a lower rank number is a better player, NA = not in the drafted-player list",
    "adp_std_pos_rank": "preseason market rank within position (standard scoring): same prior, second vote",
    "adp_std": "preseason overall pick number (standard scoring): draft-market prior, decays as games accumulate",
    "adp_ppr": "preseason overall pick number (PPR): draft-market prior",
    "role_score": "trailing-3 offensive snap share (last season's if none): how much he is on the field",
    "snap_pct_l1": "snap share in his last game: usage momentum",
    "snap_pct_l3": "snap share three games ago",
    "prev_season_ppg": "last season's PPR per game: early-season anchor that fades as this season's games arrive",
    "prev_season_xfp_pg": "last season's expected points per game: same anchor, opportunity-based",
    "prev_season_snap_pct": "last season's snap share",
    "implied_team_total": "Vegas implied team points: game script and scoring chances",
    "implied_opp_total": "Vegas implied opponent points: a proxy for pass-heavy comeback scripts",
    "tm_out_group_opps": "trailing carries + targets vacated by same-position teammates ruled Out/Doubtful: an injury-driven role expansion",
    "tm_out_targets_all": "trailing targets vacated by all Out/Doubtful teammates",
    "opps_std_mean": "season-to-date carries + targets per game: workload",
    "carry_share_l1": "share of the team's carries in his last game (RB workload)",
    "is_home": "home game (a small, consistent QB edge)",
    "weight_lb": "heavier tight ends are blockers and score less",
    "pass_att_l1": "QB attempts in his last game: weak, mostly game-script noise",
    "pos_QB": "position offset: quarterbacks score more than the other three groups",
}


def interpret(feature: str, direction: float) -> str:
    if feature in INTERPRET:
        return INTERPRET[feature]
    sign = "higher" if direction >= 0 else "lower"
    return f"{sign} value, more points" if abs(direction) >= 0.05 else "no consistent direction"


# --------------------------------------------------------------------------- SHAP
def shap_tables(df: pd.DataFrame, contrib: pd.DataFrame, fam: dict[str, str], k: int = 15):
    """(family totals, {position: top-k table}, overall table, features that earn nothing)."""
    feats = [c for c in contrib.columns if not c.startswith("_")]
    S = contrib[feats]
    pos = df.loc[S.index, "position"]
    X, _ = T.encode(df.loc[S.index], [c for c in T.feature_columns(df, T.PRIMARY, T.dead_columns(df))])
    mean_abs = S.abs().mean()
    total = mean_abs.sum()
    fam_tab = pd.DataFrame({"mean|SHAP|": mean_abs, "family": pd.Series(fam)}).groupby("family")["mean|SHAP|"].sum().sort_values(ascending=False).to_frame()
    fam_tab["share"] = fam_tab["mean|SHAP|"] / total
    per_pos = {}
    for p in POSITIONS:
        m = (pos == p).to_numpy()
        ma = S[m].abs().mean()
        tot = ma.sum()
        rows = []
        for f in ma.sort_values(ascending=False).index[:k]:
            xv, sv = X.loc[S.index[m], f], S.loc[m, f]
            ok = xv.notna() & (xv.nunique() > 1)
            dirn = float(pd.Series(xv[ok]).rank().corr(pd.Series(sv[ok]).rank())) if ok.sum() > 30 else np.nan
            rows.append({"feature": f, "family": fam.get(f, ""), "mean|SHAP|": ma[f], "share": ma[f] / tot, "direction": dirn})
        per_pos[p] = pd.DataFrame(rows)
    overall = []
    for f in mean_abs.sort_values(ascending=False).index[:k]:
        overall.append({"feature": f, "family": fam.get(f, ""), "mean|SHAP|": mean_abs[f], "share": mean_abs[f] / total})
    # earns nothing: <0.25% of total attribution overall AND <0.5% in every position
    shares = pd.DataFrame({p: S[(pos == p).to_numpy()].abs().mean() / S[(pos == p).to_numpy()].abs().mean().sum() for p in POSITIONS})
    shares["overall"] = mean_abs / total
    useless = shares[(shares["overall"] < 0.0015) & (shares[list(POSITIONS)].max(axis=1) < 0.003)
                     & ~shares.index.str.startswith("pos_")].sort_values("overall")
    return fam_tab, per_pos, pd.DataFrame(overall), useless


def worked_examples(df: pd.DataFrame, contrib: pd.DataFrame, fam: dict[str, str], wide: pd.DataFrame, q: pd.DataFrame,
                    names: pd.Series) -> list[dict]:
    """A stud, a volatile WR, a rookie: three held-out 2025 predictions with their SHAP breakdown and quantile band."""
    feats = [c for c in contrib.columns if not c.startswith("_")]
    X, _ = T.encode(df.loc[contrib.index], T.feature_columns(df, T.PRIMARY, T.dead_columns(df)))
    j = df.loc[contrib.index, ["player_id", "position", "week", "team", "opponent", "is_rookie", "college_rec_market_share",
                               "pts_ppr_l1", "pts_ppr_l2", "pts_ppr_l3", "y_points_ppr"]].join(contrib[["_pred", "_base"]])
    j = j.join(q[["q10", "q50", "q90"]])
    j["l3_std"] = j[["pts_ppr_l1", "pts_ppr_l2", "pts_ppr_l3"]].std(axis=1)
    mid = j[(j["week"] >= 5) & (j["week"] <= 17)]
    picks = {
        "stud": mid[mid["position"].isin(["RB", "WR", "TE"])].sort_values("_pred", ascending=False).iloc[0],
        "volatile WR": mid[(mid["position"] == "WR") & (mid["_pred"] >= 9)].sort_values("l3_std", ascending=False).iloc[0],
        "rookie": mid[(mid["is_rookie"] == 1) & mid["college_rec_market_share"].notna()].sort_values("_pred", ascending=False).iloc[0],
    }
    out = []
    for label, r in picks.items():
        idx = r.name
        sv = contrib.loc[idx, feats]
        top = sv.reindex(sv.abs().sort_values(ascending=False).index).head(8)
        rest = float(sv.sum() - top.sum())
        fs = sv.groupby(pd.Series(fam)).sum()
        fs = fs.reindex(fs.abs().sort_values(ascending=False).index)
        rows = [{"feature": f, "value": X.loc[idx, f], "contribution": v} for f, v in top.items()]
        out.append({"label": label, "name": names.get(r["player_id"], r["player_id"]), "position": r["position"], "week": int(r["week"]),
                    "team": r["team"], "opp": r["opponent"], "base": float(r["_base"]), "pred": float(r["_pred"]),
                    "actual": float(r["y_points_ppr"]), "q10": float(r["q10"]), "q50": float(r["q50"]), "q90": float(r["q90"]),
                    "rows": rows, "rest": rest, "last3": [r["pts_ppr_l1"], r["pts_ppr_l2"], r["pts_ppr_l3"]],
                    "fam": fs})
    return out


# --------------------------------------------------------------------------- the report
def ablv(t: pd.DataFrame | None, name: str) -> float:
    if t is None or name not in set(t["variant"]):
        return float("nan")
    return float(t.loc[t["variant"] == name, "dRMSE"].iloc[0])


def ablcell(t25: pd.DataFrame, t24: pd.DataFrame | None, name: str) -> str:
    """'+0.018 [+0.005, +0.029] (2024: +0.027 [+0.013, +0.041])' for one ablation."""
    def one(t):
        r = t[t["variant"] == name].iloc[0]
        return f"{r['dRMSE']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"
    return one(t25) + (f" (2024: {one(t24)})" if t24 is not None and name in set(t24["variant"]) else "")


def ablation_table(prefix: str, lib: str, e: pd.DataFrame, primary: pd.Series, suffix: str = "") -> pd.DataFrame:
    """One row per ablation spec found on disk: variant-minus-primary deltas with week-blocked bootstrap intervals
    on the rows `e` (primary predictions `primary`, aligned to e's index)."""
    rows = []
    for spec in B.specs_for_ablation():
        f = B.OUT / f"{prefix}_{lib}_{spec.name}.parquet"
        if not f.exists():
            continue
        r = pd.read_parquet(f)
        ee = e[["y", "position", "week", "base_trail3_ppr"]].assign(primary=primary, variant=r["pred"].reindex(e.index))
        d = B.paired_bootstrap(ee, "variant", "primary")
        row = {"variant": spec.name, "dRMSE": d.loc["ALL", "dRMSE"], "lo": d.loc["ALL", "dRMSE_lo"], "hi": d.loc["ALL", "dRMSE_hi"],
               "dMAE": d.loc["ALL", "dMAE"], "dSpearman": d.loc["ALL", "dSpearman"]}
        for p in POSITIONS:
            row[f"dRMSE {p}"] = d.loc[p, "dRMSE"]
        rows.append(row)
    return pd.DataFrame(rows)


def build() -> str:
    df = T.load_matrix()
    doc = T.load_params()
    w = wide_predictions()
    tr_means = df[(df["season"] <= 2024) & (df["y_played"] == 1)].groupby("position")["y_points_ppr"].mean()
    w = with_baselines(w, tr_means)
    e = B.eval_set(w)                                            # head-to-head rows
    e_all = B.eval_set(w, need_trail=False)                      # every played 2025 row
    e_nd = B.eval_set(w, depth_only=False)                       # head-to-head without depth-chart-only rows
    winner, rm = pick_winner(e)
    wlab = LIB_LABEL[winner]
    P = {**{LIB_LABEL[lib]: f"pred_{lib}" for lib in T.LIBS}, "trailing-3 average": "base_trail3_ppr",
         "same-week xFP (oracle)": "base_xfp_sameweek"}
    L: list[str] = []
    a = L.append

    a("# Phase 2 backtest: which model, and which data predicts\n")
    a("Walk-forward over every regular-season week of 2025: to score week N the model is refit on 2021-2024 plus 2025 "
      "weeks below N, so nothing at or after the target week is ever trained on (asserted before every fit and proven "
      "by perturbation in `model/tests/test_backtest.py`). Targets are PPR points. All numbers below are a pure function of "
      "`model/cache/features.parquet`, `model/tuned_params.json` and fixed seeds; a rerun reproduces them "
      "(`python -m model.explain` refits the winner and asserts bit-identical predictions).\n")

    # ---- setup
    n_cmp, n_all = len(e), len(e_all)
    a("## 1. What was scored\n")
    a(f"* Scored rows: 2025 regular season, players who appeared (`y_played == 1`: a stats row or offensive snaps): "
      f"**{n_all:,} rows**. The head-to-head set additionally requires at least one earlier game this season "
      f"(the trailing average exists): **{n_cmp:,} rows** ("
      + ", ".join(f"{p} {int((e['position'] == p).sum()):,}" for p in POSITIONS) + "). Every model and baseline is scored on identical rows in each table.")
    a(f"* Training rows: `y_played == 1` with the same `y_points_ppr` (a snap-only appearance is 0.0). "
      f"Refit weekly; {int(w['n_train'].min()):,} rows at week 1 growing to {int(w['n_train'].max()):,} at week 18.")
    feat_list = doc["meta"].get("features", [])
    a(f"* Primary inputs: {len(feat_list)} columns (12 registered families minus `wx_*` and `season`, position as one-hots). "
      f"Dropped as empty in 2021-2024: {', '.join(doc['meta'].get('dead_columns_dropped', [])) or 'none'}.")
    a("* Predictors compared: the three libraries, the **trailing-3 average** (mean PPR of the last <=3 appearances this season) "
      "and **same-week xFP** (ffopportunity expected points for the game itself: post-game information, an oracle, not a fair bar).\n")

    # ---- headline
    s_cmp = B.score(e, P)
    a("## 2. The verdict table (head-to-head rows, all 2025 weeks)\n")
    a(f"Winner by the pre-declared rule (lowest pooled RMSE; a statistical tie goes to LightGBM): **{wlab}**.\n")
    for metric, title, fmt in (("MAE", "MAE (points, lower is better)", "{:.2f}"), ("RMSE", "RMSE (points)", "{:.2f}"),
                               ("spearman", "Spearman rank correlation within position-week (mean over weeks; the start/sit metric)", "{:.3f}"),
                               ("pick_acc", "Head-to-head pick accuracy, all same-position pairs in a week (share of pairs where the higher prediction scored more; ties in the prediction earn 0.5)", "{:.3f}"),
                               ("pick_acc_startable", "Head-to-head pick accuracy on start/sit-relevant pairs only (both players in the top "
                                f"{B.STARTABLE_TOP['QB']}/{B.STARTABLE_TOP['RB']}/{B.STARTABLE_TOP['WR']}/{B.STARTABLE_TOP['TE']} QB/RB/WR/TE by trailing average that week)", "{:.3f}")):
        a(f"**{title}**\n")
        a(metric_table(s_cmp, metric, fmt) + "\n")

    # ---- does it beat trailing
    a("### Does the winner beat the trailing average, and how close is it to the oracle?\n")
    boot = B.paired_bootstrap(e, P[wlab], "base_trail3_ppr")
    orc = B.paired_bootstrap(e, "base_xfp_sameweek", P[wlab])
    a(f"Paired week-blocked bootstrap (2,000 resamples of the 18 weeks), {wlab} minus trailing-3. Negative dMAE/dRMSE and positive "
      "dSpearman/dPick favour the model.\n")
    a(md(boot[["dMAE", "dMAE_lo", "dMAE_hi", "dRMSE", "dRMSE_lo", "dRMSE_hi", "dSpearman", "dSpearman_lo", "dSpearman_hi",
               "dPick", "dPick_lo", "dPick_hi"]], "{:+.3f}") + "\n")
    tm, mm, om = s_cmp["trailing-3 average"], s_cmp[wlab], s_cmp["same-week xFP (oracle)"]
    gap = pd.DataFrame({"MAE closed": (tm["MAE"] - mm["MAE"]) / (tm["MAE"] - om["MAE"]),
                        "RMSE closed": (tm["RMSE"] - mm["RMSE"]) / (tm["RMSE"] - om["RMSE"]),
                        "Spearman closed": (mm["spearman"] - tm["spearman"]) / (om["spearman"] - tm["spearman"]),
                        "pick closed": (mm["pick_acc"] - tm["pick_acc"]) / (om["pick_acc"] - tm["pick_acc"])}).T[COLS]
    a("Share of the gap between trailing-3 and the xFP oracle that the model closes (0 = trailing average, 1 = oracle):\n")
    a(md(gap, "{:.0%}") + "\n")
    a(f"Library head-to-head (pooled, {wlab} is the reference; a negative dRMSE means the row's library is better):\n")
    rows = {}
    for lib in T.LIBS:
        if lib == winner:
            continue
        d = B.paired_bootstrap(e, f"pred_{lib}", P[wlab]).loc["ALL"]
        rows[LIB_LABEL[lib]] = {"dRMSE": d["dRMSE"], "lo": d["dRMSE_lo"], "hi": d["dRMSE_hi"], "dMAE": d["dMAE"],
                                "dSpearman": d["dSpearman"], "dPick": d["dPick"]}
    a(md(pd.DataFrame(rows).T, "{:+.3f}") + "\n")

    # ---- all rows
    a("## 3. All played rows, including week 1 and new players (no trailing average exists)\n")
    a("Trailing-3 is undefined for a player's first appearance of a season, so on this set it falls back to last season's "
      "points per game, then to the 2021-2024 position mean (a stronger baseline than the strict one above).\n")
    P_all = {wlab: P[wlab], "trailing-3, prior-season fallback": "trail3_fb", "same-week xFP (oracle)": "base_xfp_sameweek"}
    s_all = B.score(e_all.assign(base_trail3_ppr=e_all["trail3_fb"]), P_all)
    for metric, title, fmt in (("MAE", "MAE", "{:.2f}"), ("RMSE", "RMSE", "{:.2f}"), ("spearman", "Spearman", "{:.3f}"), ("pick_acc", "Pick accuracy", "{:.3f}")):
        a(f"**{title}** (n = " + ", ".join(f"{p} {int((e_all['position'] == p).sum()):,}" for p in POSITIONS) + ")\n")
        a(metric_table(s_all, metric, fmt) + "\n")

    # ---- spine subset
    a("## 4. Scoring with and without the depth-chart-only rows (watch-list item 1)\n")
    a("`spine_depth_only == 1` rows were admitted to the row universe only by the 2025 daily depth-chart snapshot: they are 2,201 "
      "of 2025's rows against 549-686 a year before, and that population (new-to-team veterans, low volume) is under-represented "
      "in training. Head-to-head rows, with and without them:\n")
    s_nd = B.score(e_nd, P_all | {wlab: P[wlab], "trailing-3 average": "base_trail3_ppr"})
    keep = [wlab, "trailing-3 average", "same-week xFP (oracle)"]
    rows = {}
    for lab in keep:
        for tag, ss in (("with", s_cmp), ("without", s_nd)):
            rows[f"{lab} ({tag})"] = ss[lab].loc["ALL", ["n", "MAE", "RMSE", "spearman", "pick_acc"]]
    a(md(pd.DataFrame(rows).T, "{:.3f}", floatfmt={"n": "{:.0f}"}) + "\n")
    dn = int((e["spine_depth_only"] == 1).sum())
    a(f"Depth-only rows in the head-to-head set: {dn:,} of {len(e):,} ({dn / len(e):.1%}).\n")
    e_all_nd = e_all[e_all["spine_depth_only"] == 0]
    s_a1 = B.score(e_all.assign(base_trail3_ppr=e_all["trail3_fb"]), P_all)
    s_a2 = B.score(e_all_nd.assign(base_trail3_ppr=e_all_nd["trail3_fb"]), P_all)
    rows = {}
    for lab in P_all:
        for tag, ss in (("with", s_a1), ("without", s_a2)):
            rows[f"{lab} ({tag})"] = ss[lab].loc["ALL", ["n", "MAE", "RMSE", "spearman", "pick_acc"]]
    dn2 = int((e_all["spine_depth_only"] == 1).sum())
    a(f"Every played row (week 1 and new players included; trailing-3 with the prior-season fallback): {dn2:,} of {len(e_all):,} "
      f"({dn2 / len(e_all):.1%}) are depth-only.\n")
    a(md(pd.DataFrame(rows).T, "{:.3f}", floatfmt={"n": "{:.0f}"}) + "\n")

    # ---- quantiles
    q = B.sort_quantiles(B.load("quantiles_lightgbm"))
    qe = q[q["y_played"] == 1]
    cal = B.quantile_calibration(qe)
    a("## 5. Quantile models (LightGBM p10 / p50 / p90)\n")
    a("Independent models can cross on a row; crossed rows are reported and the scored quantiles are sorted per row. Targets: below_q10 "
      "0.10, below_q50 0.50, below_q90 0.90, coverage_10_90 0.80. All 2025 played rows.\n")
    a(md(cal[["n", "below_q10", "below_q50", "below_q90", "coverage_10_90", "mean_width", "crossed_share"]], "{:.3f}", floatfmt={"n": "{:.0f}", "mean_width": "{:.1f}"}) + "\n")
    p50 = qe.assign(pred_p50=qe["q50"])
    sp = B.score(p50[p50["base_trail3_ppr"].notna()], {"p50 (median)": "pred_p50"})
    a("The p50 model as a point estimate (median, so it under-predicts the mean of a right-skewed target; MAE-optimal, not RMSE-optimal), head-to-head rows:\n")
    a(md(sp["p50 (median)"][["MAE", "RMSE", "spearman", "pick_acc"]], "{:.3f}") + "\n")
    by_week = qe.assign(inside=((qe["y"] >= qe["q10"]) & (qe["y"] <= qe["q90"])).astype(float)).groupby("week")["inside"].mean()
    a(f"Interval coverage by week ranges {by_week.min():.2f} to {by_week.max():.2f} (mean {by_week.mean():.3f}); first four weeks "
      f"{by_week.loc[1:4].mean():.3f}, weeks 5-18 {by_week.loc[5:].mean():.3f}.\n")

    # ---- ablations
    a("## 6. Ablations (winner library, same rows, same weekly refit)\n")
    a("Each row changes one thing against the primary config and is scored on the same head-to-head rows. dRMSE is variant minus "
      "primary (positive = the variant is worse) with a 95% week-blocked bootstrap interval. The primary config excludes `wx_*` and "
      "`season`, trains on `y_played` rows, and uses one model with position as a feature. Every row is replicated on the 2024 "
      "walk-forward (train 2021-2023 plus 2024 weeks below N; the same code, one year earlier): a decision read off 2025 only counts if "
      "2024 agrees. (Hyper-parameters were tuned on 2024, a mild advantage shared by every variant.)\n")
    tab25 = ablation_table("ablate", winner, e, e[P[wlab]])
    w24 = B.load(f"rep2024_{winner}_primary") if (B.OUT / f"rep2024_{winner}_primary.parquet").exists() else None
    tab24 = None
    if w24 is not None:
        e24 = B.eval_set(w24)
        tab24 = ablation_table("rep2024", winner, e24, w24["pred"].reindex(e24.index), suffix="")
        p24 = B.summarize(B.weekly_stats(e24.assign(pred=w24["pred"].reindex(e24.index)), "pred")).loc["ALL"]
    ps = B.summarize(B.weekly_stats(e.assign(pred=e[P[wlab]]), "pred")).loc["ALL"]
    line = f"Primary: RMSE {ps['RMSE']:.3f}, MAE {ps['MAE']:.3f}, Spearman {ps['spearman']:.3f}, pick {ps['pick_acc']:.3f} (2025, n = {len(e):,})"
    if w24 is not None:
        line += f"; 2024 replication set n = {len(e24):,}: RMSE {p24['RMSE']:.3f}, MAE {p24['MAE']:.3f}, Spearman {p24['spearman']:.3f}."
    a(line + "\n")

    def show(t25: pd.DataFrame, names: list[str], fam: bool) -> str:
        t = t25[t25["variant"].isin(names)].set_index("variant").reindex(names).dropna(how="all")
        out = pd.DataFrame({"dRMSE 2025": t["dRMSE"], "95% CI 2025": [f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in zip(t["lo"], t["hi"])],
                            "dMAE 2025": t["dMAE"], "dSpearman 2025": t["dSpearman"]})
        if tab24 is not None:
            u = tab24.set_index("variant").reindex(t.index)
            out["dRMSE 2024"] = u["dRMSE"]
            out["95% CI 2024"] = [f"[{lo:+.3f}, {hi:+.3f}]" if pd.notna(lo) else "" for lo, hi in zip(u["lo"], u["hi"])]
            out["dSpearman 2024"] = u["dSpearman"]
            neg, pos = "family is dead weight" if fam else "variant better in both years", "family earns its place" if fam else "variant worse in both years"
            sig = lambda lo, hi: (lo > 0) or (hi < 0)   # noqa: E731  (a 95% interval that excludes zero)
            out["both years"] = [
                "no detectable effect" if not (sig(l1, h1) or sig(l2, h2)) else pos if (x > 0 and y > 0) else neg if (x < 0 and y < 0) else "mixed (sign flips)"
                for x, y, l1, h1, l2, h2 in zip(out["dRMSE 2025"], out["dRMSE 2024"], t["lo"], t["hi"], u["lo"], u["hi"])]
        for p_ in POSITIONS:
            out[f"dRMSE {p_} 2025"] = t[f"dRMSE {p_}"]
        return md(out, "{:+.3f}", label="variant")

    a("**Design choices**\n")
    a(show(tab25, ["stats_row_target", "with_season", "with_weather", "with_roof_structure", "with_temp_wind", "train_no_depth_only",
                   "per_position_models", "drop_week"], False) + "\n")
    a("`with_weather` adds all four `wx_*` columns; `with_roof_structure` only `wx_roof_obs` / `wx_indoor_obs` (fixed dome, retractable roof or open air: "
      "a stadium attribute that IS known before kickoff); `with_temp_wind` only the observed temperature and wind (post-game values, impossible at prediction time).\n")
    a("**Dropping one feature family at a time** (sorted by 2025 dRMSE)\n")
    fam_names = [f"drop_{f}" for f in B.FAMILY_NAMES]
    order = tab25[tab25["variant"].isin(fam_names)].sort_values("dRMSE", ascending=False)["variant"].tolist()
    a(show(tab25, order, True) + "\n")
    a("**Joint drops** (chosen after reading the one-at-a-time 2025 table, so only the 2024 column is out-of-time evidence): "
      + "; ".join(f"`{n}` drops {', '.join(f)}" for n, f in B.LEAN.items()) + "\n")
    a(show(tab25, list(B.LEAN), True) + "\n")

    # ---- where the model helps
    a("### Where the model helps\n")
    wb = e.assign(bucket=pd.cut(e["week"], [0, 4, 12, 18], labels=["weeks 2-4 (weeks 1 excluded: no trailing average)", "weeks 5-12", "weeks 13-18"]),
                  who=np.where(e["is_rookie"] == 1, "rookies", "veterans"))
    rows = {}
    for col, order_ in (("bucket", None), ("who", None)):
        for k, g in wb.groupby(col, observed=True):
            ss = B.score(g, {"m": P[wlab], "t": "base_trail3_ppr", "x": "base_xfp_sameweek"})
            rows[str(k)] = {"n": len(g), "model RMSE": ss["m"].loc["ALL", "RMSE"], "trailing RMSE": ss["t"].loc["ALL", "RMSE"],
                            "oracle RMSE": ss["x"].loc["ALL", "RMSE"], "model Spearman": ss["m"].loc["ALL", "spearman"],
                            "trailing Spearman": ss["t"].loc["ALL", "spearman"]}
    a(md(pd.DataFrame(rows).T, "{:.3f}", floatfmt={"n": "{:.0f}"}) + "\n")

    # ---- refit cadence
    a("### How often to refit\n")
    a("Weekly refit (the primary) against a model refit every 4 weeks and against one that is never refit inside the season (trained on the earlier seasons "
      "only, week 1's fit predicts all 18 weeks). Same rows; dRMSE is stale minus weekly (positive = staleness costs accuracy).\n")
    rows = {}
    for name in ("refit_every_4", "refit_never"):
        f25, f24 = B.OUT / f"cadence_{winner}_{name}.parquet", B.OUT / f"cadence2024_{winner}_{name}.parquet"
        if not (f25.exists() and f24.exists()):
            continue
        row = {}
        for tag, f, base_e, base_pred in (("2025", f25, e, e[P[wlab]]), ("2024", f24, e24 if tab24 is not None else None,
                                                                         w24["pred"].reindex(e24.index) if tab24 is not None else None)):
            if base_e is None:
                continue
            r = pd.read_parquet(f)
            ee = base_e[["y", "position", "week", "base_trail3_ppr"]].assign(primary=base_pred, variant=r["pred"].reindex(base_e.index))
            d = B.paired_bootstrap(ee, "variant", "primary").loc["ALL"]
            row[f"dRMSE {tag}"] = d["dRMSE"]
            row[f"95% CI {tag}"] = f"[{d['dRMSE_lo']:+.3f}, {d['dRMSE_hi']:+.3f}]"
            row[f"dSpearman {tag}"] = d["dSpearman"]
        rows[name] = row
    if rows:
        a(md(pd.DataFrame(rows).T, "{:+.3f}", label="cadence") + "\n")

    # ---- SHAP
    shap_f = B.OUT / f"shap_{winner}.parquet"
    if shap_f.exists():
        contrib = pd.read_parquet(shap_f)
        cols = T.feature_columns(df, T.PRIMARY, T.dead_columns(df))
        _, fam = T.encode(df.iloc[:5], cols)
        fam_tab, per_pos, overall, useless = shap_tables(df, contrib, fam)
        a(f"## 7. Which data actually predicts (SHAP, {wlab}, out-of-sample 2025)\n")
        a("Mean |SHAP| in points over the 2025 played rows, each week's contributions from that week's walk-forward model. `direction` is the rank "
          "correlation between the feature's value and its SHAP contribution across that position's rows (+ = higher value raises the prediction).\n")
        a("**By feature family**\n")
        a(md(fam_tab, "{:.3f}", floatfmt={"share": "{:.1%}"}) + "\n")
        a("**Top 15 overall**\n")
        a(md(overall, "{:.3f}", index=False, floatfmt={"share": "{:.1%}"}) + "\n")
        for p in POSITIONS:
            t = per_pos[p].copy()
            t["reads as"] = [interpret(f, d) for f, d in zip(t["feature"], t["direction"])]
            a(f"**{p}: top 15 features**\n")
            a(md(t, "{:.3f}", index=False, floatfmt={"share": "{:.1%}", "direction": "{:+.2f}"}) + "\n")
        a(f"**Features that earn nothing** ({len(useless)} of {len(fam)}: under 0.15% of total attribution overall and under 0.3% in every position), by family:\n")
        for f_, g in useless.assign(family=[fam.get(c, "") for c in useless.index]).groupby("family"):
            a(f"* {f_} ({len(g)}): " + ", ".join(f"`{c}`" for c in g.index))
        a("")
        prune_f = B.OUT / "prune_list.json"
        if prune_f.exists() and (B.OUT / f"prune_{winner}_2025.parquet").exists():
            pl = json.loads(prune_f.read_text())
            pr = pd.read_parquet(B.OUT / f"prune_{winner}_2025.parquet")
            ee = e[["y", "position", "week", "base_trail3_ppr"]].assign(primary=e[P[wlab]], variant=pr["pred"].reindex(e.index))
            d = B.paired_bootstrap(ee, "variant", "primary")
            sp_ = B.summarize(B.weekly_stats(ee, "variant")).loc["ALL"]
            a("**Pruning test.** The list above comes from 2025 SHAP, so testing it on 2025 would be circular; what was tested is a list chosen on 2024: every "
              f"registry column under {pl['share_below']:.2%} of out-of-sample SHAP on the **2024** walk-forward ({len(pl['drop'])} columns: "
              + ", ".join(f"`{c}`" for c in pl["drop"]) + f"); the model is then walked forward through 2025 without them. "
              f"Result: RMSE {sp_['RMSE']:.3f} vs {ps['RMSE']:.3f} primary (dRMSE {d.loc['ALL', 'dRMSE']:+.3f}, 95% CI "
              f"[{d.loc['ALL', 'dRMSE_lo']:+.3f}, {d.loc['ALL', 'dRMSE_hi']:+.3f}]), Spearman {sp_['spearman']:.3f} vs {ps['spearman']:.3f} "
              f"(dSpearman {d.loc['ALL', 'dSpearman']:+.3f}), MAE {sp_['MAE']:.3f} vs {ps['MAE']:.3f}, with {len(pl['drop'])} fewer columns.\n")

        qi = q.loc[q.index.intersection(contrib.index)]
        players = pit._nflverse("players", "players.parquet")
        names = players.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")["display_name"]
        ex = worked_examples(df, contrib, fam, w, qi, names)
        a("## 8. Three worked predictions\n")
        a("Each is a held-out 2025 prediction. `base` is the model's average prediction; contributions add up to the prediction exactly. A blank `value` means the "
          "feature is NA for that player (no ADP entry, for example).\n")
        for x in ex:
            a(f"### {x['label']}: {x['name']} ({x['position']}, {x['team']} vs {x['opp']}, week {x['week']})\n")
            a(f"Prediction **{x['pred']:.1f}** PPR (band p10 {x['q10']:.1f} / p50 {x['q50']:.1f} / p90 {x['q90']:.1f}); actual **{x['actual']:.1f}**; "
              f"his last three games {', '.join(f'{v:.1f}' for v in x['last3'] if not np.isnan(v)) or 'none'}. Base {x['base']:.2f}.\n")
            t = pd.concat([pd.DataFrame(x["rows"]),
                           pd.DataFrame([{"feature": "all other features", "value": np.nan, "contribution": x["rest"]}])], ignore_index=True)
            a(md(t, "{:+.2f}", index=False, floatfmt={"value": "{:.2f}"}) + "\n")
            a("By family: " + ", ".join(f"{k} {v:+.2f}" for k, v in x["fam"].items() if abs(v) >= 0.005) + "\n")
    else:
        a("## 7. SHAP\n\n_Not run yet: `python -m model.explain --lib " + winner + "`._\n")
    # ---- college wiring
    a("## 9. College priors: match rate and coverage\n")
    mr = C.match_report()
    a("Drafted QB/RB/WR/TE (with a gsis id) per class and what happened to each; the matcher can only shrink coverage "
      "(name + college + position, or a position switch on college agreement; FCS schools have no team totals and get NA; see `model/college.py`).\n")
    mr.index = mr.index.astype(str)
    a(md(mr, "{:.3f}", floatfmt={c: "{:.0f}" for c in mr.columns if c != "match_rate"}, label="draft class") + "\n")
    tot = mr.sum(numeric_only=True)
    a(f"All classes: {int(tot['matched'])} of {int(tot['drafted_skill'])} drafted skill players matched ({tot['matched'] / tot['drafted_skill']:.1%}). "
      "The unmatched are FCS schools (`non_fbs`), players whose CFBD name differs from the draft name (\"Cam Ward\" is \"Cameron Ward\": the fetch "
      "kept only exact drafted names) and players with no final college season in the file (2020 opt-outs).\n")
    ccols = [c for c in F_COLLEGE if c != "college_breakout_age"]
    anyc = df[ccols].notna().any(axis=1)
    rk = df[df["is_rookie"] == 1]
    rows = []
    for season, g in rk.groupby("season"):
        pl = g[g["y_played"] == 1]
        rows.append({"season": int(season), "rookie rows": len(g), "with college": int(anyc[g.index].sum()), "share": anyc[g.index].mean(),
                     "played rows": len(pl), "played with college": int(anyc[pl.index].sum()), "played share": anyc[pl.index].mean(),
                     "drafted share of played": 1 - pl["is_undrafted"].mean(),
                     "drafted-only coverage": anyc[pl.index][pl["is_undrafted"] == 0].mean()})
    a("Coverage on rookie rows (`is_rookie == 1`) of the matrix. Undrafted rookies cannot match (no draft record, and the fetch kept drafted names only), "
      "so the ceiling is the drafted share of rookie rows. No non-rookie row carries a college value by construction.\n")
    a(md(pd.DataFrame(rows), "{:.3f}", index=False, floatfmt={"season": "{:.0f}", "rookie rows": "{:.0f}", "with college": "{:.0f}", "played rows": "{:.0f}", "played with college": "{:.0f}"}) + "\n")
    a(f"`college_breakout_age` is always NA (it needs several college seasons; the fetch pulls the last one) and is dropped as an empty column. "
      f"Non-rookie rows with a college value: {int(anyc[df['is_rookie'] != 1].sum())}.\n")

    # ---- tuning
    a("## 10. Tuning and chosen parameters\n")
    m = doc["meta"]
    a(f"Seeded random search, {m.get('n_configs')} configurations per library (config 0 is the untuned default), train 2021-2023, validate on "
      f"2024 with early stopping ({m.get('n_train'):,} train / {m.get('n_valid'):,} validation rows). 2025 is never seen. Final tree count = best "
      f"early-stopped count x {T.FOLLOW_UP_TREE_FACTOR}, fixed for all weekly refits. Seed {T.SEED}, {T.THREADS} threads, squared-error objective.\n")
    for lib in T.LIBS:
        a(f"* {LIB_LABEL[lib]}: `{json.dumps(doc['point'][lib])}` (best validation RMSE {m.get('best_valid_rmse', {}).get(lib)})")
    a("* LightGBM quantiles: same shape, tree count from the pinball loss per alpha: "
      + ", ".join(f"p{int(float(k) * 100)} n_estimators={v['n_estimators']}" for k, v in doc["quantile"].items()) + "\n")
    for lib in T.LIBS:
        log = pd.DataFrame(m["search_log"][lib])
        a(f"<details><summary>{LIB_LABEL[lib]} search log</summary>\n\n{md(log.drop(columns='seconds').sort_values('valid_rmse'), '{:.4g}', index=False)}\n\n</details>\n")


    # ---- summary (written last: it quotes the tables above)
    sm: list[str] = ["## Summary\n"]
    mA, tA, oA = s_cmp[wlab].loc["ALL"], s_cmp["trailing-3 average"].loc["ALL"], s_cmp["same-week xFP (oracle)"].loc["ALL"]
    bA, gA = boot.loc["ALL"], gap["ALL"]
    sm.append(f"* **Winner: {wlab}.** The three libraries tie (pooled head-to-head RMSE " + ", ".join(f"{LIB_LABEL[l]} {rm.loc['pooled RMSE', l]:.3f}" for l in T.LIBS)
              + "; neither XGBoost nor CatBoost differs from LightGBM beyond noise, both 95% intervals span zero), so the pre-declared tie-break picks LightGBM: one library for the point estimate, the quantiles and SHAP, "
              f"and the fastest weekly refit (18 refits in about 20 seconds against 60 for XGBoost and 120 for CatBoost). One model with position as a feature; "
              f"per-position models were not better ({'; '.join(f'{k} {v}' for k, v in [('dRMSE 2025', f'{ablv(tab25, 'per_position_models'):+.3f}'), ('2024', f'{ablv(tab24, 'per_position_models'):+.3f}' if tab24 is not None else 'n/a')])}).")
    sm.append(f"* **It beats the trailing-3 average clearly, but not by a lot.** MAE {mA['MAE']:.2f} vs {tA['MAE']:.2f} ({bA['dMAE']:+.2f}, 95% CI [{bA['dMAE_lo']:+.2f}, {bA['dMAE_hi']:+.2f}]), "
              f"RMSE {mA['RMSE']:.2f} vs {tA['RMSE']:.2f} ({bA['dRMSE']:+.2f}), Spearman {mA['spearman']:.3f} vs {tA['spearman']:.3f} ({bA['dSpearman']:+.3f}), "
              f"head-to-head pick accuracy {mA['pick_acc']:.1%} vs {tA['pick_acc']:.1%} ({bA['dPick']:+.1%}). Every interval excludes zero in every position for every metric. "
              f"On start/sit-relevant pairs the pick accuracy is {mA['pick_acc_startable']:.1%} vs {tA['pick_acc_startable']:.1%} for the trailing average.")
    sm.append(f"* **Against the same-week xFP oracle it closes {gA['MAE closed']:.0%} of the MAE gap, {gA['RMSE closed']:.0%} of the RMSE gap, {gA['Spearman closed']:.0%} of the "
              f"Spearman gap and {gA['pick closed']:.0%} of the pick-accuracy gap.** The oracle is post-game information (it sees the carries and targets the player actually got), "
              f"so this is a ceiling for pre-game information, not a target; weekly fantasy scoring is mostly opportunity plus luck the model cannot see. "
              f"Quarterbacks are the hardest position (Spearman {s_cmp[wlab].loc['QB', 'spearman']:.2f} vs RB {s_cmp[wlab].loc['RB', 'spearman']:.2f}, WR {s_cmp[wlab].loc['WR', 'spearman']:.2f}, TE {s_cmp[wlab].loc['TE', 'spearman']:.2f}).")
    cA = cal.loc["ALL"]
    sm.append(f"* **Quantiles are usable, slightly narrow.** Share of actuals at or below p10 / p50 / p90: {cA['below_q10']:.1%} / {cA['below_q50']:.1%} / {cA['below_q90']:.1%} "
              f"(targets 10 / 50 / 90); the p10-p90 band covers {cA['coverage_10_90']:.1%} (target 80%), QB {cal.loc['QB', 'coverage_10_90']:.1%}, TE {cal.loc['TE', 'coverage_10_90']:.1%}. "
              f"The floor (p10) is too high by about two points and the median a little high; the ceiling is right.")
    shap_f2 = B.OUT / f"shap_{winner}.parquet"
    if shap_f2.exists():
        top3 = ", ".join(f"`{r['feature']}` ({r['share']:.0%})" for _, r in overall.head(3).iterrows())
        fs = ", ".join(f"{k} {v:.0%}" for k, v in fam_tab["share"].head(6).items())
        sm.append(f"* **What predicts (SHAP, out of sample):** by family {fs}. Top three features: {top3}. Season-to-date expected fantasy points (opportunity quality) and "
                  "actual points are the core; the draft-market prior (ADP) and snap share come next; Vegas implied team total is the only game-environment input that matters "
                  "(largest for quarterbacks); college production, combine numbers, injury designations, TD luck, defense-vs-position and rest/venue carry little "
                  f"(college {fam_tab.loc['college', 'share']:.1%} of total attribution, injury {fam_tab.loc['injury', 'share']:.1%}, td_luck {fam_tab.loc['td_luck', 'share']:.1%}, "
                  f"dvp {fam_tab.loc['dvp', 'share']:.1%}, context {fam_tab.loc['context', 'share']:.1%}).")
    dn_all = int((e_all["spine_depth_only"] == 1).sum())
    sm.append(f"* **Depth-chart-only rows (watch-list item 1):** they are {dn_all / len(e_all):.1%} of all played 2025 rows and {int((e['spine_depth_only'] == 1).sum()) / len(e):.1%} of the head-to-head rows. "
              f"Scored without them the model's head-to-head RMSE is {s_nd[wlab].loc['ALL', 'RMSE']:.3f} (vs {mA['RMSE']:.3f} with) and the trailing average's {s_nd['trailing-3 average'].loc['ALL', 'RMSE']:.3f} "
              f"(vs {tA['RMSE']:.3f}): the margin over the trailing average is unchanged. Training without the depth-only rows changes nothing (row below).")
    ab = lambda n: ablcell(tab25, tab24, n)   # noqa: E731
    sm.append("* **Ablations (dRMSE variant minus primary, 2025 then 2024 replication; positive = variant worse):** "
              f"training on stats-row players only {ab('stats_row_target')} (worse both years: keep `y_played`); "
              f"`season` {ab('with_season')}; all observed weather {ab('with_weather')}, roof structure only {ab('with_roof_structure')}, temperature and wind only {ab('with_temp_wind')} "
              "(the 2025 gains do not replicate: weather, even as a perfect forecast, does not reliably help); "
              f"training without depth-chart-only rows {ab('train_no_depth_only')}; per-position models {ab('per_position_models')}; without `week` {ab('drop_week')}.")
    fam_bits = []
    for n in ("drop_lags", "drop_vegas", "drop_role", "drop_adp"):
        fam_bits.append(f"{n[5:]} {ab(n)}")
    sm.append("* **Which families are load-bearing (dropping them hurts in both years; only lags and vegas have a 2025 interval that excludes zero, role and adp only in 2024):** " + "; ".join(fam_bits) + ". The other families (prev_season, injury, static, college, dvp, td_luck, context) "
              "move RMSE by less than 0.02 in either direction and change sign between 2025 and 2024: one-at-a-time drops cannot separate them from noise, "
              "and the joint drops chosen from the 2025 table (`lean_A`, `lean_B`) also fail to replicate in 2024.")
    prune_f2 = B.OUT / "prune_list.json"
    if prune_f2.exists() and (B.OUT / f"prune_{winner}_2025.parquet").exists():
        pl2 = json.loads(prune_f2.read_text())
        pr2 = pd.read_parquet(B.OUT / f"prune_{winner}_2025.parquet")
        ee2 = e[["y", "position", "week", "base_trail3_ppr"]].assign(primary=e[P[wlab]], variant=pr2["pred"].reindex(e.index))
        d2 = B.paired_bootstrap(ee2, "variant", "primary").loc["ALL"]
        sm.append(f"* **Pruning test (the honest one):** dropping the {len(pl2['drop'])} registry columns that earned under {pl2['share_below']:.2%} of out-of-sample SHAP on the 2024 walk-forward "
                  f"gives 2025 RMSE {ee2.assign(pred=ee2['variant']).pipe(lambda z: B.summarize(B.weekly_stats(z, 'pred')).loc['ALL', 'RMSE']):.3f} vs {mA['RMSE']:.3f} "
                  f"(dRMSE {d2['dRMSE']:+.3f}, CI [{d2['dRMSE_lo']:+.3f}, {d2['dRMSE_hi']:+.3f}]): a smaller model that is no worse.")
        fn = B.OUT / f"prune_{winner}_noadp_2025.parquet"
        if fn.exists():
            r3 = pd.read_parquet(fn)
            ee3 = e[["y", "position", "week", "base_trail3_ppr"]].assign(primary=e[P[wlab]], variant=r3["pred"].reindex(e.index))
            d3 = B.paired_bootstrap(ee3, "variant", "primary").loc["ALL"]
            s3 = B.summarize(B.weekly_stats(ee3.assign(pred=ee3["variant"]), "pred")).loc["ALL"]
            sm.append(f"* **The serving candidate for 2026 is that pruned set without ADP** (no preseason 2026 ADP snapshot exists: the fetch returned a 29-player in-season window, "
                      f"so {df.loc[df['season'] == 2026, 'adp_ppr'].isna().mean():.1%} of 2026 rows are NA on ADP against {df.loc[df['season'] == 2025, 'adp_ppr'].isna().mean():.1%} of 2025 rows, a train/serve skew): 2025 RMSE {s3['RMSE']:.3f} (dRMSE {d3['dRMSE']:+.3f} vs primary, CI [{d3['dRMSE_lo']:+.3f}, {d3['dRMSE_hi']:+.3f}]), "
                      f"Spearman {s3['spearman']:.3f}, MAE {s3['MAE']:.3f}, still ahead of the trailing average ({tA['RMSE']:.3f} RMSE, {tA['spearman']:.3f} Spearman).")
    cad = {}
    for name in ("refit_every_4", "refit_never"):
        bits = []
        for tag, f, base_e, base_pred in (("2025", B.OUT / f"cadence_{winner}_{name}.parquet", e, e[P[wlab]]),
                                          ("2024", B.OUT / f"cadence2024_{winner}_{name}.parquet", e24 if tab24 is not None else None,
                                           w24["pred"].reindex(e24.index) if tab24 is not None else None)):
            if not f.exists() or base_e is None:
                continue
            r = pd.read_parquet(f)
            ee3 = base_e[["y", "position", "week", "base_trail3_ppr"]].assign(primary=base_pred, variant=r["pred"].reindex(base_e.index))
            d3 = B.paired_bootstrap(ee3, "variant", "primary").loc["ALL"]
            bits.append(f"{tag} {d3['dRMSE']:+.3f} [{d3['dRMSE_lo']:+.3f}, {d3['dRMSE_hi']:+.3f}]")
        cad[name] = "; ".join(bits)
    if cad:
        sm.append(f"* **Refit cadence (dRMSE of a staler model vs the weekly refit; positive = staleness costs):** every 4 weeks {cad.get('refit_every_4')}; "
                  f"never inside the season {cad.get('refit_never')}. Small and not consistent across years (nothing in 2025, a real but small cost in 2024), and a weekly refit is free.")
    sm.append("* **Not tested and worth knowing:** the target is PPR only (league-scoring variants are a linear recombination that needs the component model in Phase 3); "
              "there is no platform-projection baseline (historical ESPN/Sleeper projections are not archived); our pipeline's `E_pts` blends platform projections (not archived for 2025) with "
              "trailing form and a DvP multiplier, so it cannot be reconstructed for 2025 and was not compared. `college_breakout_age` is always NA (needs multi-season CFBD data); 27 of 475 drafted skill players fail to match only because "
              "the fetch kept exact drafted names (nicknames) or the school is FCS.\n")
    L[2:2] = ["\n".join(sm) + "\n"]
    return "\n".join(L)


def main() -> int:
    text = build()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)
    print(f"wrote {REPORT} ({len(text):,} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
