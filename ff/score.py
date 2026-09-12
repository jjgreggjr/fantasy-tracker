"""PlayScore: one transparent number per player per week, per league.

Every component is a column that exists in the CSVs, so any score can be
recomputed by hand. Deliberately rule-based; no model, no black box.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import scoring

DEFAULTS = {
    "trail_cap": 0.6,       # most weight trailing data can take from projections
    "trail_games": 4,       # games needed to reach that cap
    "matchup_swing": 0.24,  # total range of the matchup multiplier (±12%)
}


def _conf_label(v: float) -> str:
    if pd.isna(v):
        return ""
    return "High" if v >= 0.7 else ("Med" if v >= 0.4 else "Low")


def compute(roster: pd.DataFrame, players: pd.DataFrame, base: pd.DataFrame,
            proj: pd.DataFrame, dvp: pd.DataFrame, sched: pd.DataFrame,
            depth: pd.DataFrame, points: pd.DataFrame, meta: dict,
            week: int, season: int, cfg: dict | None = None) -> pd.DataFrame:
    """Score a set of players (a roster, or the free-agent pool) for one week."""
    c = {**DEFAULTS, **(cfg or {})}
    if roster is None or roster.empty:
        return pd.DataFrame()

    df = roster.copy()
    keep = [x for x in ["gsis_id", "name", "position", "team", "owner_name",
                        "is_starter", "is_taxi", "is_ir", "sleeper_id",
                        "roster_fetched_at"]
            if x in df.columns]
    df = df[keep]
    df = df[df.position.isin(["QB", "RB", "WR", "TE"])]
    if df.empty:
        return pd.DataFrame()

    pinfo = players[["gsis_id", "age", "years_exp", "injury_status",
                     "nfl_status", "draft_number", "rookie_year"]]
    df = df.merge(pinfo, on="gsis_id", how="left")
    df = df.merge(base, on="gsis_id", how="left")

    # this league's trailing points (last 3 weeks played)
    if points is not None and not points.empty:
        # everything already played: all prior seasons, plus earlier weeks of
        # this one. Filtering on week alone silently dropped the whole prior
        # season in Week 1, which zeroed every trailing figure.
        p = points[(points.season < season)
                   | ((points.season == season) & (points.week < week))]
        p = p.sort_values(["season", "week"])
        recent = (p.groupby("gsis_id").tail(3)
                   .groupby("gsis_id", as_index=False).pts.mean()
                   .rename(columns={"pts": "trail_pts"}))
        df = df.merge(recent, on="gsis_id", how="left")
    else:
        df["trail_pts"] = np.nan

    # projections for this week
    if proj is not None and not proj.empty:
        pw = proj[proj.week == week].drop(columns=["team", "position"],
                                          errors="ignore")
        df = df.merge(pw, on="gsis_id", how="left")
    for col in ["proj_opps", "proj_targets", "proj_carries",
                "proj_carry_share", "proj_target_share"]:
        if col not in df.columns:
            df[col] = np.nan

    # projected points in THIS league's scoring, from the projected stat line
    if "proj_pass_yds" in df.columns:
        line = pd.DataFrame({
            "position": df.position,
            "pass_yds": df.proj_pass_yds, "pass_tds": df.proj_pass_tds,
            "pass_int": df.get("proj_pass_int", 0), "pass_att": df.proj_pass_att,
            "pass_cmp": 0, "rush_yds": df.proj_rush_yds, "carries": df.proj_carries,
            "rush_tds": df.proj_rush_tds, "receptions": df.proj_rec,
            "targets": df.proj_targets, "rec_yds": df.proj_rec_yds,
            "rec_tds": df.proj_rec_tds, "two_pt": 0, "fum": 0, "fum_lost": 0,
            "rush_fd": 0, "rec_fd": 0,
        })
        df["proj_pts_league"] = scoring.score_frame(
            line, meta.get("scoring", {}), f"{meta.get('name')} (proj)")
        df.loc[df.proj_opps.isna() & df.proj_pass_yds.isna(),
               "proj_pts_league"] = np.nan
    else:
        df["proj_pts_league"] = np.nan

    # ---- blend weights ---------------------------------------------------
    games = df.get("games", pd.Series(0, index=df.index)).fillna(0)
    w_trail = (games / c["trail_games"]).clip(0, c["trail_cap"])
    has_proj = df.proj_opps.notna()
    has_trail = df.avg_opps.notna()
    w_trail = np.where(~has_proj, 1.0, np.where(~has_trail, 0.0, w_trail))
    w_proj = 1.0 - w_trail
    df["w_proj"] = np.round(w_proj, 2)

    def blend(pcol, tcol):
        p = pd.to_numeric(df.get(pcol), errors="coerce")
        t = pd.to_numeric(df.get(tcol), errors="coerce")
        return (np.where(has_proj, w_proj * p.fillna(0), 0)
                + np.where(has_trail, w_trail * t.fillna(0), 0))

    df["E_opps"] = np.round(blend("proj_opps", "avg_opps"), 1)
    df["E_targets"] = np.round(blend("proj_targets", "avg_targets"), 1)
    df["E_carries"] = np.round(blend("proj_carries", "avg_carries"), 1)
    df.loc[~has_proj & ~has_trail, ["E_opps", "E_targets", "E_carries"]] = np.nan

    # ---- matchup ---------------------------------------------------------
    wk = sched[sched.week == week][["team", "opponent", "total_line", "home"]]
    df = df.merge(wk, on="team", how="left")
    d = dvp.rename(columns={"defense": "opponent"})[["opponent", "position", "dvp_rank"]]
    df = df.merge(d, on=["opponent", "position"], how="left")
    dvp_pct = ((33 - df.dvp_rank) / 32).clip(0, 1)
    lo = 1 - c["matchup_swing"] / 2
    df["mult"] = (lo + c["matchup_swing"] * dvp_pct.fillna(0.5)).round(3)

    # ---- points ----------------------------------------------------------
    # Renormalize: if one side is missing, the other takes the full weight.
    # Otherwise a player with a projection but no history is scored as though
    # his history were zero, which quietly deflates every rookie and newcomer.
    has_p = df.proj_pts_league.notna()
    has_t = df.trail_pts.notna()
    wp = np.where(has_p, w_proj, 0.0)
    wt = np.where(has_t, w_trail, 0.0)
    tot = wp + wt
    e_pts = np.where(
        tot > 0,
        (wp * df.proj_pts_league.fillna(0) + wt * df.trail_pts.fillna(0))
        / np.where(tot > 0, tot, 1),
        np.nan)
    df["E_pts"] = np.round(e_pts * df.mult, 1)
    # share of the blend that came from the projection, for transparency
    df["pts_mix"] = [
        ("projection only" if not t_ else f"{p_/(p_+t_):.0%} projection")
        if p_ else ("recent only" if t_ else "—")
        for p_, t_ in zip(wp, wt)]

    # ---- depth chart, confidence ----------------------------------------
    dep = depth[["gsis_id", "rank", "prev_rank"]].rename(columns={"rank": "depth_rank"})
    df = df.merge(dep, on="gsis_id", how="left")

    po = pd.to_numeric(df.proj_opps, errors="coerce")
    to = pd.to_numeric(df.avg_opps, errors="coerce")
    denom = pd.concat([po, to], axis=1).max(axis=1).clip(lower=1)
    agree = (1 - (po - to).abs() / denom).clip(0, 1).fillna(0.6)
    role = df.depth_rank.map({1: 1.0, 2: 0.8}).fillna(0.6)
    health = df.injury_status.map(
        {"Questionable": 0.7, "Doubtful": 0.3, "Out": 0.0,
         "IR": 0.0, "PUP": 0.0, "Sus": 0.0}).fillna(1.0)
    df["confidence"] = (agree * role * health).round(2)
    df["conf"] = df.confidence.map(_conf_label)

    # ---- flags -----------------------------------------------------------
    df["flag"] = ""
    df.loc[df.E_opps.isna(), "flag"] = "NO BASELINE"
    df.loc[df.opponent.eq("BYE") | df.opponent.isna(), "flag"] = "BYE"
    for st, lab in [("Questionable", "QUESTIONABLE"), ("Doubtful", "DOUBTFUL"),
                    ("Out", "OUT"), ("IR", "IR"), ("PUP", "PUP")]:
        df.loc[df.injury_status.eq(st), "flag"] = lab
    df.loc[df.flag.isin(["BYE", "OUT", "IR", "PUP", "NO BASELINE"]), "E_pts"] = np.nan

    df["why"] = _why(df)
    return df.sort_values(["position", "E_pts"],
                          ascending=[True, False]).reset_index(drop=True)


def _why(df: pd.DataFrame) -> pd.Series:
    out = []
    for _, r in df.iterrows():
        bits = []
        if pd.notna(r.get("E_opps")):
            src = []
            if pd.notna(r.get("proj_opps")):
                src.append(f"proj {r.proj_opps:.0f}")
            if pd.notna(r.get("avg_opps")):
                src.append(f"recent {r.avg_opps:.0f}")
            bits.append(f"{r.E_opps:.1f} exp opps" +
                        (f" ({', '.join(src)})" if src else ""))
        if pd.notna(r.get("depth_rank")):
            bits.append(f"{r.position}{int(r.depth_rank)}")
        share = r.get("proj_carry_share") if r.position == "RB" else r.get("proj_target_share")
        if pd.notna(share):
            lbl = "of team carries" if r.position == "RB" else "of team targets"
            bits.append(f"{share:.0%} {lbl}")
        if pd.notna(r.get("dvp_rank")) and pd.notna(r.get("opponent")):
            pct = (r.mult - 1) * 100
            bits.append(f"vs {r.opponent} (DvP {int(r.dvp_rank)}, {pct:+.0f}%)")
        if r.get("conf"):
            bits.append(f"{r.conf} confidence")
        if r.get("flag"):
            bits.append(r.flag)
        out.append(" · ".join(bits))
    return pd.Series(out, index=df.index)
