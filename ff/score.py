"""PlayScore: one transparent number per player per week, per league.

Every component is a column that exists in the CSVs, so any score can be
recomputed by hand. Deliberately rule-based; no model, no black box.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import scoring

DEFAULTS = {
    # Most weight trailing data can take from the projection. Set to 0 on 2026-10-15 (was 0.6):
    # over 2026 weeks 2-5 (838 rows) pick accuracy fell monotonically as this rose (0.692 at 0,
    # 0.690 at 0.2, 0.680 at 0.6, 0.648 trailing-only) and when E_pts left Sleeper's number
    # Sleeper was right 56% of the time. Trailing data still carries a player Sleeper has not
    # projected (w_trail is forced to 1 when proj_opps is missing). model/candidates.md has the trail.
    "trail_cap": 0.0,
    "trail_games": 4,       # games needed to reach that cap
    "matchup_swing": 0.24,  # total range of the matchup multiplier (±12%)
    # Touchdown regression (2026-10-27, model/reports/miss_review_2026wk07.md): the biggest attributable
    # E_pts residual was the player's own season TD rate per opportunity (eta² .037: high-TD players
    # over-predicted by 1.6, low-TD under-predicted by 1.8), which the projection takes at face value.
    # The season rate is shrunk toward the position's league-wide rate with a prior worth
    # `td_prior_opps` opportunities (pass attempts for a QB) and the difference, times expected
    # opportunities and the league's TD points, is added to E_pts, clipped to ±`td_adj_clip`.
    "td_regress": True,
    "td_prior_opps": {"QB": 150, "RB": 60, "WR": 60, "TE": 60},
    "td_adj_clip": 3.0,
}

TD_POS_DEFAULT_PTS = {"rush_td": 6.0, "rec_td": 6.0, "pass_td": 4.0}


def td_regression(df: pd.DataFrame, pw: pd.DataFrame | None, season: int, week: int,
                  scoring_rules: dict | None, c: dict) -> pd.DataFrame:
    """`td_rate_raw`, `td_rate_shrunk`, `td_adj` for every row of `df` (needs gsis_id, position, E_opps,
    and proj_pass_att for QBs).

        shrunk = (TDs + prior × pos_rate) / (opps + prior)
        td_adj = clip((shrunk − raw) × E_opps × td_pts, ±td_adj_clip)

    where TDs/opps are the player's season-to-date rush+rec TDs over carries+targets (pass TDs over
    attempts for a QB) from `pw` (weeks before `week` of `season`), pos_rate the same ratio pooled over
    his position, and E_opps the expected opportunities (projected attempts for a QB, else his season
    mean). A player with no season rows, zero opportunities or no E_opps gets td_adj 0 and NaN rates:
    nothing moves for him. Calibration on the 1,019 frozen e_pts rows of 2026 weeks 4-7 (Gooma scoring):
    residual ~ 1.07 x td_adj, MAE 4.73 -> 4.66, so the full correction ships unscaled."""
    out = pd.DataFrame({"td_rate_raw": np.nan, "td_rate_shrunk": np.nan, "td_adj": 0.0},
                       index=df.index)
    if pw is None or pw.empty or df.empty or not c.get("td_regress", True):
        return out
    need = {"season", "week", "gsis_id", "position", "opps", "rec_tds", "rush_tds", "pass_att", "pass_tds"}
    if not need <= set(pw.columns):
        return out
    p = pw[(pd.to_numeric(pw.season, errors="coerce") == season)
           & (pd.to_numeric(pw.week, errors="coerce") < week)].copy()
    if p.empty:
        return out
    num = lambda col: pd.to_numeric(p[col], errors="coerce").fillna(0.0)
    is_qb = p.position.eq("QB")
    p["_tds"] = np.where(is_qb, num("pass_tds"), num("rec_tds") + num("rush_tds"))
    p["_opps"] = np.where(is_qb, num("pass_att"), num("opps"))
    per_player = p.groupby("gsis_id")[["_tds", "_opps"]].sum()
    # a QB's expected attempts when Sleeper has not projected him: his season mean in games he threw in
    thrown = p[is_qb & (p._opps > 0)]
    att_per_game = thrown.groupby("gsis_id")._opps.mean()
    per_pos = p.groupby("position")[["_tds", "_opps"]].sum()
    pos_rate = (per_pos._tds / per_pos._opps.where(per_pos._opps > 0)).to_dict()

    sc = scoring_rules or {}
    pts = lambda k: float(sc.get(k) or TD_POS_DEFAULT_PTS[k])
    skill_td_pts = (pts("rush_td") + pts("rec_td")) / 2.0

    prior_cfg = c.get("td_prior_opps", DEFAULTS["td_prior_opps"])
    prior_of = (lambda pos: float(prior_cfg.get(pos, 60))) if isinstance(prior_cfg, dict) \
        else (lambda pos: float(prior_cfg))

    gid = df.gsis_id.astype(object)
    tds = gid.map(per_player._tds)
    opps = gid.map(per_player._opps)
    prate = df.position.map(pos_rate)
    prior = df.position.map(prior_of).astype(float)
    qb = df.position.eq("QB")
    e_opps = pd.to_numeric(df.get("E_opps"), errors="coerce")
    e_att = pd.to_numeric(df.get("proj_pass_att"), errors="coerce") if "proj_pass_att" in df else \
        pd.Series(np.nan, index=df.index)
    e_att = e_att.fillna(gid.map(att_per_game))
    volume = e_att.where(qb, e_opps)
    td_pts = np.where(qb, pts("pass_td"), skill_td_pts)

    ok = tds.notna() & (opps > 0) & prate.notna() & volume.notna()
    raw = tds / opps.where(opps > 0)
    shrunk = (tds + prior * prate) / (opps + prior)
    adj = ((shrunk - raw) * volume * td_pts).clip(-c["td_adj_clip"], c["td_adj_clip"])
    out["td_rate_raw"] = raw.where(ok).round(4)
    out["td_rate_shrunk"] = shrunk.where(ok).round(4)
    out["td_adj"] = adj.where(ok, 0.0).fillna(0.0).round(2)
    return out


def _conf_label(v: float) -> str:
    if pd.isna(v):
        return ""
    return "High" if v >= 0.7 else ("Med" if v >= 0.4 else "Low")


def compute(roster: pd.DataFrame, players: pd.DataFrame, base: pd.DataFrame,
            proj: pd.DataFrame, dvp: pd.DataFrame, sched: pd.DataFrame,
            depth: pd.DataFrame, points: pd.DataFrame, meta: dict,
            week: int, season: int, cfg: dict | None = None,
            pw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Score a set of players (a roster, or the free-agent pool) for one week.

    `pw` is player_weeks (any seasons); it feeds the touchdown regression (`td_regression`) and is
    optional: without it td_adj is 0 for everyone."""
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
                     "nfl_status", "draft_number", "rookie_year"]
                    + [x for x in ("inactive", "repeat_dnp", "role_change")
                       if x in players.columns]]
    df = df.merge(pinfo, on="gsis_id", how="left")
    df = df.merge(base, on="gsis_id", how="left")
    # An IR activation or practice-squad elevation this week (status.role_changes): his recent weeks
    # were 0-opportunity weeks for a reason that is gone, so trailing volume is not held against him
    # where a projection exists. With no projection the trailing numbers are all he has and stay.
    role_change = (pd.to_numeric(df["role_change"], errors="coerce").fillna(0).astype(int).eq(1)
                   if "role_change" in df.columns else pd.Series(False, index=df.index))

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
        pj = proj[proj.week == week].drop(columns=["team", "position"],
                                          errors="ignore")
        df = df.merge(pj, on="gsis_id", how="left")
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
    has_trail = df.avg_opps.notna() & ~(role_change & has_proj)
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
    has_t = df.trail_pts.notna() & ~(role_change & has_p)
    wp = np.where(has_p, w_proj, 0.0)
    wt = np.where(has_t, w_trail, 0.0)
    tot = wp + wt
    e_pts = np.where(
        tot > 0,
        (wp * df.proj_pts_league.fillna(0) + wt * df.trail_pts.fillna(0))
        / np.where(tot > 0, tot, 1),
        np.nan)
    # touchdown regression: an additive, clipped correction on top of the blend (see td_regression)
    td = td_regression(df, pw, season, week, meta.get("scoring", {}), c)
    df["td_rate_raw"], df["td_rate_shrunk"], df["td_adj"] = td.td_rate_raw, td.td_rate_shrunk, td.td_adj
    df["E_pts"] = np.round(e_pts * df.mult + df.td_adj.to_numpy(), 1)
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
    if "inactive" in df.columns:
        health = health.where(~_inactive_mask(df), 0.0)
    df["confidence"] = (agree * role * health).round(2)
    df["conf"] = df.confidence.map(_conf_label)

    # ---- flags -----------------------------------------------------------
    df = apply_flags(df)

    df["why"] = _why(df)
    return df.sort_values(["position", "E_pts"],
                          ascending=[True, False]).reset_index(drop=True)


def _inactive_mask(df: pd.DataFrame) -> pd.Series:
    """Rows on this week's gameday inactive list (`inactive` == 1, set by the run from ff.status.week_inactives)."""
    if "inactive" not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df["inactive"], errors="coerce").fillna(0).astype(int).eq(1)


def apply_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Set `flag` from baseline, bye and injury status, and blank `E_pts` for
    anyone who cannot play this week."""
    df["flag"] = ""
    df.loc[df.E_opps.isna(), "flag"] = "NO BASELINE"
    bye = df.opponent.eq("BYE") | df.opponent.isna()
    df.loc[bye, "flag"] = "BYE"
    # an activation / practice-squad elevation this week (status.role_changes) is worth a label of
    # its own, but only where nothing else is already said
    df.loc[_role_change_mask(df) & df.flag.eq(""), "flag"] = "ROLE_CHANGE"
    for st, lab in [("Questionable", "QUESTIONABLE"), ("Doubtful", "DOUBTFUL"),
                    ("Out", "OUT"), ("IR", "IR"), ("PUP", "PUP")]:
        df.loc[df.injury_status.eq(st), "flag"] = lab
    # 0 opportunities in each of the last two completed weeks and no Full practice this week
    # (status.recent_dnp): Caleb Williams (2026 week 5) had missed weeks 3-4 and kept 14.8 E_pts.
    # repeat_dnp 1 = capped (play_prob <= 0.25), 2 = also Out/Doubtful or DNP, so zeroed.
    rd = _repeat_dnp_level(df)
    df.loc[rd.ge(1) & ~df.flag.isin(["OUT", "IR", "PUP", "BYE", "NO BASELINE"]),
           "flag"] = "REPEAT_DNP"
    # the gameday inactive list beats any designation: Jalen Coker (2026 week 4) was
    # Questionable until he was inactive and still made the E_pts-optimal lineup.
    df.loc[_inactive_mask(df), "flag"] = "INACTIVE"
    # a bye is a bye whatever the injury tag says: a Questionable player on a
    # bye week once kept his E_pts and the optimizer started him (Aaron Jones,
    # 2026 week 6). Apply the bye zero on the mask, not on the final label.
    df.loc[bye | rd.ge(2)
           | df.flag.isin(["BYE", "OUT", "IR", "PUP", "INACTIVE", "NO BASELINE"]),
           "E_pts"] = np.nan
    return df


def _repeat_dnp_level(df: pd.DataFrame) -> pd.Series:
    """0 / 1 (capped) / 2 (zeroed) from `repeat_dnp` (set by the run from ff.status.build)."""
    if "repeat_dnp" not in df.columns:
        return pd.Series(0, index=df.index)
    return pd.to_numeric(df["repeat_dnp"], errors="coerce").fillna(0).astype(int)


def _role_change_mask(df: pd.DataFrame) -> pd.Series:
    if "role_change" not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df["role_change"], errors="coerce").fillna(0).astype(int).eq(1)


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
        adj = r.get("td_adj")
        if pd.notna(adj) and abs(float(adj)) >= 0.05:
            bits.append(f"TD regress {float(adj):+.1f} "
                        f"(season {r.td_rate_raw:.1%}/opp -> {r.td_rate_shrunk:.1%})")
        if r.get("conf"):
            bits.append(f"{r.conf} confidence")
        if r.get("flag"):
            bits.append(r.flag)
        if pd.notna(r.get("role_change")) and int(r.get("role_change") or 0) == 1:
            bits.append("activated/elevated this week: recent 0-opp weeks not counted")
        out.append(" · ".join(bits))
    return pd.Series(out, index=df.index)
