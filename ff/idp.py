"""Individual defenders (DL / LB / DB) and kickers: points, E_pts and the rows that go beside the skill players.

Deliberately thin. There is no opportunity model for these players (no depth chart, no snap-share
projection, no matchup curve): `E_pts` is Sleeper's projected stat line scored in the league's own
scoring when Sleeper projected him, else his average over his team's last three games in the same
scoring. `E_opps` is the volume behind it, trailing: tackles + assists for a defender, field-goal +
extra-point attempts for a kicker. Only a league that starts a defender slot gets any of this
(`scoring.league_idp_positions`); every other league never sees these rows.

Inputs are the frames the run already holds: `hist` is data/idp_weeks.csv (prior season + this one),
`proj` data/idp_projections.csv, `idp_players` the DL/LB/DB/K player dimension (`build.build_players`
with `positions=IDP_POS`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import score as score_mod
from . import scoring

TRAIL_GAMES = 3
POINT_COLS = ["season", "week", "gsis_id", "name", "position", "pts"]
TRAIL_COLS = ["gsis_id", "trail_pts", "trail_tkl", "trail_sacks", "trail_fg_att", "trail_xpa",
              "trail_opps", "trail_games"]
# projected stat columns that are named exactly like the idp_weeks columns once `proj_` is dropped
_PROJ_STATS = ["tkl_solo", "tkl_ast", "tkl_loss", "sacks", "qb_hits", "ints", "pass_def", "ff",
               "fum_rec", "def_td", "safeties", "blk_kick", "fg_att", "fg_made", "fg_missed",
               "fgm_0_19", "fgm_20_29", "fgm_30_39", "fgm_40_49", "fgm_50p",
               "fgmiss_0_19", "fgmiss_20_29", "fgmiss_30_39", "fgmiss_40_49", "fgmiss_50p",
               "xpa", "xpm", "xpmiss"]


def league_points(hist: pd.DataFrame, scoring_rules: dict, label: str = "") -> pd.DataFrame:
    """Points per player-week in this league's scoring (the IDP counterpart of player_points.csv)."""
    if hist is None or hist.empty:
        return pd.DataFrame(columns=POINT_COLS)
    out = hist[["season", "week", "gsis_id", "name", "position"]].copy()
    out["pts"] = scoring.score_idp(hist, scoring_rules, label)
    return out


def trailing(hist: pd.DataFrame, points: pd.DataFrame, season: int, week: int,
             n: int = TRAIL_GAMES) -> pd.DataFrame:
    """Per player: his average over his team's last `n` completed games before (`season`, `week`).

    nflverse writes a row only for a player who recorded a stat, so a lineman who played and had
    nothing has no row. A game with no row therefore counts as 0 for a player who has a row in an
    earlier game of the window, and a player with no row anywhere in the window gets no trailing
    figure at all (he is not playing, or has just arrived: no baseline, not a false zero). The
    window is the team's games, so a bye week is not one of them."""
    if hist is None or hist.empty:
        return pd.DataFrame(columns=TRAIL_COLS)
    h = hist.copy()
    h["gk"] = pd.to_numeric(h.season, errors="coerce") * 100 + pd.to_numeric(h.week, errors="coerce")
    h = h[h.gk < season * 100 + week].dropna(subset=["gk", "gsis_id"])
    if h.empty:
        return pd.DataFrame(columns=TRAIL_COLS)
    pts = (points[["season", "week", "gsis_id", "pts"]] if points is not None and not points.empty
           else pd.DataFrame(columns=["season", "week", "gsis_id", "pts"]))
    h = h.merge(pts, on=["season", "week", "gsis_id"], how="left")
    num = lambda c: pd.to_numeric(h[c], errors="coerce").fillna(0.0)
    h["pts"] = pd.to_numeric(h.pts, errors="coerce").fillna(0.0)
    h["tkl"] = num("tkl_solo") + num("tkl_ast")
    h["xp_fg"] = num("fg_att") + num("xpa")
    h["opps"] = np.where(h.position.eq("K"), h.xp_fg, h.tkl)
    h["sacks"] = num("sacks")
    h["fg_att"], h["xpa"] = num("fg_att"), num("xpa")

    team_games = h.groupby("team").gk.agg(lambda s: sorted(set(s))).to_dict()
    latest_team = h.sort_values("gk").drop_duplicates("gsis_id", keep="last").set_index("gsis_id").team
    rows = []
    for gid, g in h.groupby("gsis_id"):
        games = team_games.get(latest_team[gid], [])[-n:]
        g = g[g.gk.isin(games)]
        if g.empty:
            continue
        k = len([x for x in games if x >= g.gk.min()])
        rows.append({"gsis_id": gid, "trail_pts": g.pts.sum() / k, "trail_tkl": g.tkl.sum() / k,
                     "trail_sacks": g.sacks.sum() / k, "trail_fg_att": g.fg_att.sum() / k,
                     "trail_xpa": g.xpa.sum() / k, "trail_opps": g.opps.sum() / k,
                     "trail_games": k})
    return pd.DataFrame(rows, columns=TRAIL_COLS)


def _projected(df: pd.DataFrame, scoring_rules: dict, label: str) -> tuple[pd.Series, pd.Series]:
    """(projected points in this league's scoring, projected volume) for rows carrying `proj_*` stats.
    NaN points where Sleeper projected no stat line for him."""
    from .projections import IDP_PRESENCE
    present = [c for c in IDP_PRESENCE if c in df.columns]
    has = df[present].notna().any(axis=1) if present else pd.Series(False, index=df.index)
    line = pd.DataFrame({s: df[f"proj_{s}"] for s in _PROJ_STATS if f"proj_{s}" in df.columns},
                        index=df.index)
    pts = scoring.score_idp(line, scoring_rules, f"{label} (proj)") if len(line.columns) else \
        pd.Series(np.nan, index=df.index)
    num = lambda c: pd.to_numeric(df.get(c), errors="coerce").fillna(0.0)
    vol = np.where(df.position.eq("K"), num("proj_fg_att") + num("proj_xpa"),
                   num("proj_tkl_solo") + num("proj_tkl_ast"))
    return pts.where(has), pd.Series(vol, index=df.index).where(has)


def compute(roster: pd.DataFrame, idp_players: pd.DataFrame, hist: pd.DataFrame, proj: pd.DataFrame,
            points: pd.DataFrame, sched: pd.DataFrame, meta: dict, week: int, season: int,
            positions: list[str]) -> pd.DataFrame:
    """Scored rows for the DL / LB / DB / K players in `roster` (a roster, or the free-agent pool), with
    the same identity and E_pts / E_opps / conf / flag / why columns `score.compute` writes. Columns that
    only mean something for skill players are absent here and come out NaN when the frames are joined."""
    if roster is None or roster.empty or not positions:
        return pd.DataFrame()
    keep = [x for x in ["gsis_id", "name", "position", "team", "owner_name", "is_starter", "is_taxi",
                        "is_ir", "sleeper_id", "roster_fetched_at"] if x in roster.columns]
    df = roster[keep]
    df = df[df.position.isin(positions) & df.gsis_id.notna()].copy()
    if df.empty:
        return pd.DataFrame()

    pinfo = idp_players[["gsis_id", "age", "years_exp", "injury_status", "nfl_status", "draft_number",
                         "rookie_year"] + [x for x in ("inactive", "repeat_dnp", "role_change")
                                           if x in idp_players.columns]]
    df = df.merge(pinfo.drop_duplicates("gsis_id"), on="gsis_id", how="left")

    tr = trailing(hist, points, season, week)
    df = df.merge(tr, on="gsis_id", how="left")

    df["season"], df["week"] = season, week
    if proj is not None and not proj.empty and (proj.week == week).any():
        pj = proj[proj.week == week].drop(columns=["team", "position", "season", "week"], errors="ignore")
        df = df.merge(pj.drop_duplicates("gsis_id"), on="gsis_id", how="left")
        pts, vol = _projected(df, meta.get("scoring", {}), meta.get("name", ""))
        num = lambda c: pd.to_numeric(df.get(c), errors="coerce")
        tkl = (num("proj_tkl_solo").fillna(0) + num("proj_tkl_ast").fillna(0)).where(pts.notna())
        sacks = num("proj_sacks")
        df = df[[c for c in df.columns if not (c.startswith("proj_") and c != "proj_pts_league")]]
        df["proj_pts_league"], df["proj_opps"] = pts.round(2), vol.round(2)
        df["proj_tkl"], df["proj_sacks"] = tkl, sacks.where(pts.notna())
    for c in ("proj_pts_league", "proj_opps", "proj_tkl", "proj_sacks"):
        if c not in df.columns:
            df[c] = np.nan

    has_p = df.proj_pts_league.notna()
    has_t = df.trail_pts.notna()
    df["games"] = df.trail_games
    df["avg_opps"] = df.trail_opps
    df["w_proj"] = has_p.astype(float)
    df["E_opps"] = df.trail_opps.where(has_t, df.proj_opps).round(1)
    df["E_pts"] = np.round(df.proj_pts_league.where(has_p, df.trail_pts), 1)
    df["pts_mix"] = np.where(has_p, "projection only", np.where(has_t, "recent only", "—"))

    wk = sched[sched.week == week][["team", "opponent", "total_line", "home"]]
    df = df.merge(wk, on="team", how="left")

    # no depth chart or matchup model here: confidence is health times how much we know
    health = df.injury_status.map({"Questionable": 0.7, "Doubtful": 0.3, "Out": 0.0, "IR": 0.0,
                                   "PUP": 0.0, "Sus": 0.0}).fillna(1.0)
    if "inactive" in df.columns:
        health = health.where(~score_mod._inactive_mask(df), 0.0)
    know = np.where(has_p & has_t, 0.9, np.where(has_p, 0.75, np.where(has_t, 0.6, 0.0)))
    df["confidence"] = (health * know).round(2)
    df["conf"] = df.confidence.map(score_mod._conf_label)

    df = score_mod.apply_flags(df)
    df["why"] = _why(df)
    return df.sort_values(["position", "E_pts"], ascending=[True, False]).reset_index(drop=True)


def _why(df: pd.DataFrame) -> pd.Series:
    """'IDP: 6.3 tkl/g, 0.7 sacks/g · last 3: 8.4 pts · vs DAL · Med confidence' (K: attempts instead)."""
    out = []
    for _, r in df.iterrows():
        bits = []
        if r.position == "K":
            fga, xpa = r.get("trail_fg_att"), r.get("trail_xpa")
            if pd.notna(fga) and pd.notna(xpa):
                bits.append(f"K: {fga:.1f} FGA + {xpa:.1f} XPA/g")
            elif pd.notna(r.get("proj_opps")):
                bits.append(f"K: {r.proj_opps:.1f} proj FGA + XPA")
        else:
            tkl, sk, tag = r.get("trail_tkl"), r.get("trail_sacks"), ""
            if pd.isna(tkl):
                tkl, sk, tag = r.get("proj_tkl"), r.get("proj_sacks"), " (proj)"
            if pd.notna(tkl):
                bits.append(f"IDP: {tkl:.1f} tkl/g, {0.0 if pd.isna(sk) else sk:.1f} sacks/g{tag}")
        if pd.notna(r.get("proj_pts_league")):
            bits.append(f"proj {r.proj_pts_league:.1f} pts")
        elif pd.notna(r.get("trail_pts")):
            bits.append(f"last {int(r.trail_games)} g avg {r.trail_pts:.1f} pts")
        if pd.notna(r.get("opponent")) and r.get("opponent") != "BYE":
            bits.append(f"vs {r.opponent}")
        if r.get("conf"):
            bits.append(f"{r.conf} confidence")
        if r.get("flag"):
            bits.append(r.flag)
        out.append(" · ".join(bits))
    return pd.Series(out, index=df.index)


def pool(idp_players: pd.DataFrame, owned: set, positions: list[str]) -> pd.DataFrame:
    """Active, unrostered players at the league's IDP / K positions."""
    return idp_players[idp_players.position.isin(positions)
                       & ~idp_players.gsis_id.isin(owned)
                       & idp_players.nfl_status.isin(["ACT", "DEV"])].copy()


def top_of_pool(avail: pd.DataFrame, per_position: int) -> pd.DataFrame:
    """Waiver candidates worth writing down: has a number, best `per_position` by E_pts at each position
    (the whole free-agent pool of defenders is ~1,500 rows and mostly never starts)."""
    if avail.empty:
        return avail
    a = avail[avail.E_pts.notna() | avail.E_opps.notna()]
    return (a.sort_values("E_pts", ascending=False, na_position="last")
             .groupby("position", sort=False).head(per_position))
