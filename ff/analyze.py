"""Rule-based analysis: volume baselines, matchup scores, watchlist, handcuffs.

Nothing here is a point projection. `start_score` is an opportunity score:
a transparent weighted blend of recent volume, snap share and matchup.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SKILL = ["QB", "RB", "WR", "TE"]


def blend_weight(week: int) -> float:
    """Weight given to prior-season data. 0.75 in wk1, decaying to 0 by wk4."""
    return max(0.0, (4 - week) / 4)


def _blend(cur: pd.Series, prior: pd.Series, w: float) -> pd.Series:
    """Blend current and prior season values, tolerating either being absent."""
    c, p = cur.astype(float), prior.astype(float)
    both = c.notna() & p.notna()
    out = c.copy()
    out[both] = w * p[both] + (1 - w) * c[both]
    out[c.isna()] = p[c.isna()]
    return out


def _slope(vals: list[float]) -> float:
    v = [x for x in vals if pd.notna(x)]
    if len(v) < 2:
        return 0.0
    x = np.arange(len(v), dtype=float)
    return float(np.polyfit(x, np.asarray(v, dtype=float), 1)[0])


def player_baselines(pw: pd.DataFrame, preview_week: int, trailing: int,
                     prior_pw: pd.DataFrame | None) -> pd.DataFrame:
    """Per-game volume averages and trends, blended with the prior season."""
    def summarize(df: pd.DataFrame, weeks: int | None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["gsis_id"])
        d = df.sort_values("week")
        if weeks:
            keep = sorted(d.week.unique())[-weeks:]
            d = d[d.week.isin(keep)]
        g = d.groupby("gsis_id")
        res = g.agg(games=("week", "nunique"),
                    avg_opps=("opps", "mean"),
                    avg_targets=("targets", "mean"),
                    avg_carries=("carries", "mean"),
                    avg_snap_pct=("snap_pct", "mean"),
                    avg_fpts=("fpts_ppr", "mean")).reset_index()
        trends = g.apply(lambda x: pd.Series({
            "trend_opps": _slope(x.sort_values("week").opps.tolist()),
            "trend_snap": _slope(x.sort_values("week").snap_pct.tolist()),
        }), include_groups=False).reset_index()
        return res.merge(trends, on="gsis_id", how="left")

    cur = summarize(pw, trailing)
    prior = summarize(prior_pw, None)
    w = blend_weight(preview_week)

    base = cur.merge(prior, on="gsis_id", how="outer", suffixes=("", "_prior"))
    for col in ["avg_opps", "avg_targets", "avg_carries", "avg_snap_pct", "avg_fpts"]:
        pcol = f"{col}_prior"
        if pcol in base.columns:
            base[col] = _blend(base[col], base[pcol], w)
    # trends prefer current-season data; early in the year the only in-season
    # signal available is last season's, so fall back to it explicitly.
    for col in ["trend_opps", "trend_snap"]:
        pcol = f"{col}_prior"
        if pcol in base.columns:
            base[col] = base[col].fillna(base[pcol])
        # Rounded because these are serialised straight into the committed
        # CSVs. polyfit's summation order varies between runs, so identical
        # inputs produced values differing in the last bit (~1e-16) and every
        # run rewrote ~800 rows of pure noise, burying the real changes in the
        # weekly diff. Six decimals is far more precision than a slope of
        # opportunities-per-week carries.
        base[col] = base[col].fillna(0.0).round(6)
    base["trend_season"] = np.where(base.get("games", 0) > 0, "current", "prior")
    base["games"] = base["games"].fillna(0)
    mix = (f"{int((1-w)*100)}% current / {int(w*100)}% prior" if w
           else "current season")
    base["baseline_source"] = np.where(base["games"] > 0, mix, "prior season only")
    keep = ["gsis_id", "games", "avg_opps", "avg_targets", "avg_carries",
            "avg_snap_pct", "avg_fpts", "trend_opps", "trend_snap",
            "trend_season", "baseline_source"]
    return base[keep]


def team_pass_rank(pw: pd.DataFrame, prior_pw: pd.DataFrame | None,
                   preview_week: int) -> pd.DataFrame:
    """Rank teams by pass attempts per game. Rank 1 = throws the most."""
    def per_team(df):
        if df is None or df.empty:
            return pd.DataFrame(columns=["team", "patt"])
        q = df[df.position == "QB"]
        g = q.groupby("team").agg(att=("pass_att", "sum"),
                                  wks=("week", "nunique")).reset_index()
        g["patt"] = g.att / g.wks.replace(0, np.nan)
        return g[["team", "patt"]]

    w = blend_weight(preview_week)
    cur, prior = per_team(pw), per_team(prior_pw)
    m = cur.merge(prior, on="team", how="outer", suffixes=("", "_prior"))
    m["patt"] = _blend(m["patt"], m.get("patt_prior", pd.Series(index=m.index)), w)
    m["team_pass_att_rank"] = m.patt.rank(ascending=False, method="min")
    return m[["team", "patt", "team_pass_att_rank"]]


def dvp_ranks(dvp: pd.DataFrame, preview_week: int, trailing: int,
              prior_dvp: pd.DataFrame | None) -> pd.DataFrame:
    """Per-game production allowed by each defense to each position.

    `dvp_rank` 1 = SOFTEST matchup (allowed the most PPR points to that
    position). This is the opposite of how a 'defensive ranking' usually
    reads, and every report table says so.
    """
    def per_def(df, weeks):
        if df is None or df.empty:
            return pd.DataFrame()
        d = df.sort_values("week")
        if weeks:
            keep = sorted(d.week.unique())[-weeks:]
            d = d[d.week.isin(keep)]
        g = d.groupby(["defense", "position"]).agg(
            g_played=("week", "nunique"),
            fpts=("fpts_ppr_allowed", "sum"),
            tgts=("targets_allowed", "sum"),
            recyds=("rec_yds_allowed", "sum"),
            rectds=("rec_tds_allowed", "sum"),
            car=("carries_allowed", "sum"),
            rushyds=("rush_yds_allowed", "sum"),
            rushtds=("rush_tds_allowed", "sum"),
            passyds=("pass_yds_allowed", "sum"),
            passtds=("pass_tds_allowed", "sum"),
            pts=("points_allowed", "mean"),
        ).reset_index()
        for c in ["fpts", "tgts", "recyds", "rectds", "car", "rushyds",
                  "rushtds", "passyds", "passtds"]:
            g[c] = g[c] / g.g_played.replace(0, np.nan)
        return g

    w = blend_weight(preview_week)
    cur, prior = per_def(dvp, trailing), per_def(prior_dvp, None)
    if cur.empty and prior.empty:
        return pd.DataFrame()
    if cur.empty:
        m = prior.copy()
    elif prior.empty:
        m = cur.copy()
    else:
        m = cur.merge(prior, on=["defense", "position"], how="outer",
                      suffixes=("", "_prior"))
        for c in ["fpts", "tgts", "recyds", "rectds", "car", "rushyds",
                  "rushtds", "passyds", "passtds", "pts"]:
            if f"{c}_prior" in m.columns:
                m[c] = _blend(m[c], m[f"{c}_prior"], w)
    m["dvp_rank"] = m.groupby("position").fpts.rank(ascending=False, method="min")
    cols = ["defense", "position", "dvp_rank", "fpts", "tgts", "recyds",
            "rectds", "car", "rushyds", "rushtds", "passyds", "passtds", "pts"]
    return m[cols].sort_values(["position", "dvp_rank"]).reset_index(drop=True)


def start_sit(roster: pd.DataFrame, players: pd.DataFrame, base: pd.DataFrame,
              dvp: pd.DataFrame, sched: pd.DataFrame, depth: pd.DataFrame,
              week: int, weights: dict) -> pd.DataFrame:
    df = (roster.merge(players.drop(columns=["team", "position", "name"],
                                    errors="ignore"),
                       on="gsis_id", how="left", suffixes=("", "_p"))
                .merge(base, on="gsis_id", how="left"))
    df = df[df.position.isin(SKILL)]
    if df.empty:
        return pd.DataFrame(columns=[
            "name", "position", "team", "opponent", "start_score", "avg_opps",
            "avg_targets", "avg_carries", "avg_snap_pct", "avg_fpts",
            "trend_opps", "dvp_rank", "depth_rank", "total_line", "age",
            "baseline_source", "flag"])

    wk = sched[sched.week == week][["team", "opponent", "home", "spread_line",
                                    "total_line", "gameday"]]
    df = df.merge(wk, on="team", how="left")
    d = dvp.rename(columns={"defense": "opponent"})[["opponent", "position",
                                                     "dvp_rank", "fpts"]]
    df = df.merge(d, on=["opponent", "position"], how="left")
    dep = depth[["gsis_id", "rank", "prev_rank"]].rename(
        columns={"rank": "depth_rank"})
    df = df.merge(dep, on="gsis_id", how="left")

    # normalize opportunity within position against the 95th percentile
    norm = df.groupby("position").avg_opps.transform(
        lambda s: s.quantile(0.95) if s.notna().any() else np.nan)
    df["opps_score"] = (df.avg_opps / norm).clip(0, 1).fillna(0)
    df["snap_score"] = df.avg_snap_pct.clip(0, 1).fillna(0)
    df["dvp_score"] = ((33 - df.dvp_rank) / 32).clip(0, 1).fillna(0.5)
    df["start_score"] = (weights["opps"] * df.opps_score
                         + weights["snap_pct"] * df.snap_score
                         + weights["dvp_rank"] * df.dvp_score).round(3)

    df["flag"] = ""
    # a player with no prior volume (rookie, or never played) has no baseline;
    # scoring him against zero volume would look precise and be meaningless
    df.loc[df.avg_opps.isna(), "flag"] = "NO BASELINE"
    df.loc[df.opponent.eq("BYE") | df.opponent.isna(), "flag"] = "BYE"
    for st, lab in [("Out", "OUT"), ("IR", "IR"), ("Doubtful", "DOUBTFUL"),
                    ("Questionable", "QUESTIONABLE")]:
        df.loc[df.injury_status.eq(st), "flag"] = lab
    df.loc[df.flag.isin(["BYE", "OUT", "IR", "NO BASELINE"]), "start_score"] = np.nan
    return df.sort_values(["position", "start_score"],
                          ascending=[True, False]).reset_index(drop=True)


def watchlist(players: pd.DataFrame, base: pd.DataFrame, depth: pd.DataFrame,
              owned: set, passrank: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    p = players[players.position.isin(["RB", "WR", "TE"])].copy()
    p = p[~p.gsis_id.isin(owned)]
    p = p[p.nfl_status.isin(["ACT", "DEV"])]
    df = (p.merge(base, on="gsis_id", how="left")
           .merge(depth[["gsis_id", "rank", "prev_rank"]].rename(
               columns={"rank": "depth_rank"}), on="gsis_id", how="left")
           .merge(passrank, on="team", how="left"))
    df = df[df.depth_rank.notna()]

    hard = cfg.get("hard_max_age", 27)
    young = (((df.age <= cfg["max_age"]) | (df.years_exp <= cfg["max_years_exp"]))
             & (df.age <= hard))
    in_ranks = df.depth_rank.isin(cfg["depth_ranks"])
    promoted = df.prev_rank.notna() & (df.depth_rank < df.prev_rank)
    ascending = (df.trend_snap >= cfg["min_snap_pct_trend"]) | promoted
    pass_ok = (df.position == "RB") | (
        df.team_pass_att_rank <= cfg["min_team_pass_att_rank"])

    df = df[young & (in_ranks | promoted) & (ascending | in_ranks) & pass_ok].copy()

    reasons = []
    for _, r in df.iterrows():
        bits = []
        if pd.notna(r.depth_rank):
            bits.append(f"{r.position}{int(r.depth_rank)} on depth chart")
        if pd.notna(r.prev_rank) and r.depth_rank < r.prev_rank:
            bits.append(f"up from {int(r.prev_rank)}")
        if pd.notna(r.trend_snap) and r.trend_snap >= cfg["min_snap_pct_trend"]:
            bits.append(f"snap share trending +{r.trend_snap:.0%}/wk")
        if pd.notna(r.team_pass_att_rank) and r.position in ("WR", "TE"):
            bits.append(f"team passes #{int(r.team_pass_att_rank)}")
        if pd.notna(r.age):
            bits.append(f"age {r.age:.0f}")
        reasons.append("; ".join(bits))
    df["why"] = reasons
    return df.sort_values(["trend_snap", "avg_opps"],
                          ascending=False).head(25).reset_index(drop=True)


def handcuffs(my: pd.DataFrame, depth: pd.DataFrame, players: pd.DataFrame,
              owners: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    rbs = my[my.position == "RB"]
    if rbs.empty:
        return pd.DataFrame()
    dep = depth[depth.position == "RB"]
    own = owners.set_index("gsis_id").owner_name.to_dict() if not owners.empty else {}
    binfo = base.set_index("gsis_id") if not base.empty else pd.DataFrame()

    rows = []
    for _, r in rbs.iterrows():
        mine = dep[dep.gsis_id == r.gsis_id]
        my_rank = int(mine["rank"].iloc[0]) if not mine.empty else None
        target = 2 if my_rank == 1 else 1
        mates = dep[(dep.team == r.team) & (dep["rank"] == target)]
        if mates.empty:
            continue
        m = mates.iloc[0]
        pinfo = players[players.gsis_id == m.gsis_id]
        b = binfo.loc[m.gsis_id] if m.gsis_id in binfo.index else None
        rows.append({
            "my_player": r["name"], "my_rank": my_rank, "team": r.team,
            "relation": "handcuff (RB2)" if my_rank == 1 else "starter ahead (RB1)",
            "other_player": m["name"],
            "other_status": (pinfo.injury_status.iloc[0]
                             if not pinfo.empty else pd.NA),
            "other_snap_pct": (b.avg_snap_pct if b is not None else np.nan),
            "other_avg_opps": (b.avg_opps if b is not None else np.nan),
            "owned_by": own.get(m.gsis_id, "FREE AGENT"),
        })
    return pd.DataFrame(rows)


def field_matchups(players: pd.DataFrame, depth: pd.DataFrame, base: pd.DataFrame,
                   dvp: pd.DataFrame, sched: pd.DataFrame, owners: pd.DataFrame,
                   week: int, softest: int = 8) -> pd.DataFrame:
    starters = depth[(depth["rank"] == 1) & (depth.position.isin(["RB", "WR", "TE"]))]
    df = (starters.merge(players[["gsis_id", "age", "injury_status"]],
                         on="gsis_id", how="left")
                  .merge(base, on="gsis_id", how="left")
                  .merge(sched[sched.week == week][["team", "opponent", "total_line"]],
                         on="team", how="left"))
    d = dvp.rename(columns={"defense": "opponent"})[["opponent", "position", "dvp_rank"]]
    df = df.merge(d, on=["opponent", "position"], how="left")
    df = df[df.dvp_rank <= softest]
    own = owners.set_index("gsis_id").owner_name.to_dict() if not owners.empty else {}
    df["owned_by"] = df.gsis_id.map(own).fillna("FREE AGENT")
    return df.sort_values("dvp_rank").reset_index(drop=True)
