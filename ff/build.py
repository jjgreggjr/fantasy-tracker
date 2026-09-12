"""Transform raw nflverse/Sleeper data into the narrow CSVs we keep."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SKILL = ["QB", "RB", "WR", "TE"]
OFFENSE_GRP = "3WR 1TE"

ID_COLS = ["sleeper_id", "espn_id", "pfr_id", "gsis_id"]


def norm_id(v) -> object:
    """Normalize an external id to a clean string.

    Sleeper/ESPN ids are numeric strings. Once a CSV round-trips through
    pandas they come back as floats ("8151.0"), which silently breaks every
    join against the Sleeper API's string ids. Normalize on both sides.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
        return pd.NA
    s = str(v).strip()
    if s in ("", "nan", "<NA>", "None"):
        return pd.NA
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def name_key(name) -> str:
    """Normalized name for fallback matching.

    Sources disagree on suffixes and punctuation: nflverse says
    "Mike Washington Jr." where Sleeper says "Mike Washington", and
    "D.K. Metcalf" vs "DK Metcalf". Strip both.
    """
    if name is None or pd.isna(name):
        return ""
    s = str(name).lower().replace(".", "").replace("'", "").replace("-", " ")
    parts = [p for p in s.split() if p not in _SUFFIXES]
    return " ".join(parts).strip()


def read_data_csv(path: Path) -> pd.DataFrame:
    """Read one of our own CSVs, keeping id columns as strings."""
    df = pd.read_csv(path, low_memory=False)
    for c in ID_COLS:
        if c in df.columns:
            df[c] = df[c].map(norm_id).astype("string")
    return df


def _age(birth: pd.Series, asof: pd.Timestamp) -> pd.Series:
    b = pd.to_datetime(birth, errors="coerce")
    return ((asof - b).dt.days / 365.25).round(1)


def upsert(path: Path, new: pd.DataFrame, keys: list[str]) -> int:
    """Append `new`, replacing any existing rows sharing the same key tuples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, low_memory=False)
        if not new.empty and all(k in old.columns for k in keys):
            idx = pd.MultiIndex.from_frame(new[keys].astype(str))
            oidx = pd.MultiIndex.from_frame(old[keys].astype(str))
            old = old[~oidx.isin(idx)]
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    combined = combined.sort_values(keys).reset_index(drop=True)
    combined.to_csv(path, index=False)
    return len(new)


# --------------------------------------------------------------------------
def build_players(roster: pd.DataFrame, sleeper: dict | None,
                  asof: pd.Timestamp) -> pd.DataFrame:
    r = roster[roster.position.isin(SKILL)].copy()
    r = r.sort_values("week").drop_duplicates("gsis_id", keep="last")
    r = r[r.gsis_id.notna()]

    out = pd.DataFrame({
        "gsis_id": r.gsis_id.map(norm_id).astype("string"),
        "sleeper_id": r.sleeper_id.map(norm_id).astype("string"),
        "espn_id": r.espn_id.map(norm_id).astype("string"),
        "pfr_id": r.pfr_id.map(norm_id).astype("string"),
        "name": r.full_name,
        "team": r.team,
        "position": r.position,
        "age": _age(r.birth_date, asof),
        "years_exp": pd.to_numeric(r.years_exp, errors="coerce"),
        "rookie_year": pd.to_numeric(r.get("rookie_year"), errors="coerce"),
        "draft_number": pd.to_numeric(r.get("draft_number"), errors="coerce"),
        "nfl_status": r.status,
    })
    for c in ("injury_status", "injury_note", "injury_body_part",
              "injury_start_date", "practice_participation",
              "practice_description", "news_updated", "sleeper_depth_order",
              "sleeper_status", "sleeper_active"):
        out[c] = pd.NA

    if sleeper:
        # Capture every status field Sleeper publishes, not just the
        # designation. Practice participation in particular is the best
        # available predictor of whether a player actually suits up.
        sl = pd.DataFrame([
            {"sleeper_id": str(k),
             "s_name": name_key(v.get("full_name")),
             "s_team": v.get("team"),
             "injury_status": v.get("injury_status"),
             "injury_note": v.get("injury_notes"),
             "injury_body_part": v.get("injury_body_part"),
             "injury_start_date": v.get("injury_start_date"),
             "practice_participation": v.get("practice_participation"),
             "practice_description": v.get("practice_description"),
             "news_updated": v.get("news_updated"),
             "sleeper_depth_order": v.get("depth_chart_order"),
             "sleeper_status": v.get("status"),
             "sleeper_active": v.get("active")}
            for k, v in sleeper.items()
            if v.get("position") in SKILL
        ])
        status_cols = ["injury_status", "injury_note", "injury_body_part",
                       "injury_start_date", "practice_participation",
                       "practice_description", "news_updated",
                       "sleeper_depth_order", "sleeper_status",
                       "sleeper_active"]
        out = out.drop(columns=status_cols).merge(
            sl[["sleeper_id"] + status_cols], on="sleeper_id", how="left")
        # fallback: name+team for rows with no sleeper_id
        miss = out.sleeper_id.isna()
        if miss.any():
            key = out.loc[miss, "name"].map(name_key)
            fb = sl.drop_duplicates(["s_name", "s_team"]).set_index(["s_name", "s_team"])
            pairs = list(zip(key, out.loc[miss, "team"]))
            filled = [fb.loc[p].to_dict() if p in fb.index else {} for p in pairs]
            out.loc[miss, "sleeper_id"] = [f.get("sleeper_id", pd.NA) for f in filled]
            for c in status_cols:
                out.loc[miss, c] = [f.get(c, pd.NA) for f in filled]
            log.info("name+team fallback matched %d of %d players missing sleeper_id",
                     sum(bool(f) for f in filled), int(miss.sum()))
    return out.reset_index(drop=True)


def build_player_weeks(stats: pd.DataFrame, snaps: pd.DataFrame | None,
                       players: pd.DataFrame, season: int) -> pd.DataFrame:
    s = stats[(stats.season_type == "REG") & (stats.position.isin(SKILL))].copy()
    if s.empty:
        return s

    num = lambda c: pd.to_numeric(s.get(c), errors="coerce").fillna(0)
    out = pd.DataFrame({
        "season": s.season.astype(int),
        "week": s.week.astype(int),
        "gsis_id": s.player_id,
        "name": s.player_display_name,
        "position": s.position,
        "team": s.team,
        "opponent": s.opponent_team,
        "targets": num("targets"),
        "receptions": num("receptions"),
        "rec_yds": num("receiving_yards"),
        "rec_tds": num("receiving_tds"),
        "target_share": pd.to_numeric(s.get("target_share"), errors="coerce"),
        "air_yds_share": pd.to_numeric(s.get("air_yards_share"), errors="coerce"),
        "wopr": pd.to_numeric(s.get("wopr"), errors="coerce"),
        "carries": num("carries"),
        "rush_yds": num("rushing_yards"),
        "rush_tds": num("rushing_tds"),
        "pass_att": num("attempts"),
        "pass_cmp": num("completions"),
        "pass_yds": num("passing_yards"),
        "pass_tds": num("passing_tds"),
        "pass_int": num("passing_interceptions"),
        # scoring inputs: leagues differ on these, so carry them all
        "two_pt": (num("passing_2pt_conversions") + num("rushing_2pt_conversions")
                   + num("receiving_2pt_conversions")),
        "fum": num("fumbles_total"),
        "fum_lost": num("fumbles_lost_total"),
        "fum_rec": num("fumble_recovery_own") + num("fumble_recovery_opp"),
        "fum_rec_td": num("fumble_recovery_tds"),
        "rush_fd": num("rushing_first_downs"),
        "rec_fd": num("receiving_first_downs"),
        "fpts_ppr": pd.to_numeric(s.get("fantasy_points_ppr"), errors="coerce"),
    })
    out["touches"] = out.carries + out.receptions
    out["opps"] = out.carries + out.targets

    # snap counts join through pfr_id
    out["snaps"] = np.nan
    out["snap_pct"] = np.nan
    if snaps is not None and not snaps.empty:
        xw = players[["gsis_id", "pfr_id"]].dropna().drop_duplicates("gsis_id")
        sn = snaps[snaps.game_type == "REG"][
            ["season", "week", "pfr_player_id", "offense_snaps", "offense_pct"]
        ].drop_duplicates(["season", "week", "pfr_player_id"])
        merged = (out.merge(xw, on="gsis_id", how="left")
                     .merge(sn, left_on=["season", "week", "pfr_id"],
                            right_on=["season", "week", "pfr_player_id"], how="left"))
        out["snaps"] = merged.offense_snaps.values
        out["snap_pct"] = merged.offense_pct.values

    cols = ["season", "week", "gsis_id", "name", "position", "team", "opponent",
            "snaps", "snap_pct", "targets", "receptions", "rec_yds", "rec_tds",
            "rec_fd", "target_share", "air_yds_share", "wopr", "carries",
            "rush_yds", "rush_tds", "rush_fd", "pass_att", "pass_cmp",
            "pass_yds", "pass_tds", "pass_int", "two_pt", "fum", "fum_lost",
            "fum_rec", "fum_rec_td", "touches", "opps", "fpts_ppr"]
    return out[cols].sort_values(["week", "position", "name"]).reset_index(drop=True)


def build_depth_charts(depth: pd.DataFrame, season: int, week: int,
                       prev: pd.DataFrame | None) -> pd.DataFrame:
    d = depth[(depth.pos_grp == OFFENSE_GRP) & (depth.pos_abb.isin(SKILL))].copy()
    if d.empty:
        return d
    d["dt"] = pd.to_datetime(d.dt, errors="coerce", utc=True)
    latest = d.groupby("team").dt.transform("max")
    d = d[d.dt == latest]

    out = pd.DataFrame({
        "season": season,
        "week": week,
        "snapshot_dt": d.dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "team": d.team,
        "position": d.pos_abb,
        "rank": pd.to_numeric(d.pos_rank, errors="coerce"),
        "gsis_id": d.gsis_id,
        "name": d.player_name,
    }).dropna(subset=["rank"])
    out["rank"] = out["rank"].astype(int)
    out = out.drop_duplicates(["team", "position", "rank"])

    out["prev_rank"] = np.nan
    if prev is not None and not prev.empty:
        p = (prev.sort_values("week").drop_duplicates("gsis_id", keep="last")
                 [["gsis_id", "rank"]].rename(columns={"rank": "prev_rank"}))
        out = out.drop(columns=["prev_rank"]).merge(p, on="gsis_id", how="left")
    return out.sort_values(["team", "position", "rank"]).reset_index(drop=True)


def build_schedule(games: pd.DataFrame, season: int) -> pd.DataFrame:
    g = games[(games.season == season) & (games.game_type == "REG")].copy()
    home = pd.DataFrame({
        "season": season, "week": g.week, "team": g.home_team,
        "opponent": g.away_team, "home": 1, "gameday": g.gameday,
        "spread_line": -pd.to_numeric(g.spread_line, errors="coerce"),
        "total_line": pd.to_numeric(g.total_line, errors="coerce"), "roof": g.roof})
    away = pd.DataFrame({
        "season": season, "week": g.week, "team": g.away_team,
        "opponent": g.home_team, "home": 0, "gameday": g.gameday,
        "spread_line": pd.to_numeric(g.spread_line, errors="coerce"),
        "total_line": pd.to_numeric(g.total_line, errors="coerce"), "roof": g.roof})
    sched = pd.concat([home, away], ignore_index=True)

    # explicit BYE rows so "no game this week" is never a silent absence
    teams = sorted(sched.team.unique())
    byes = []
    for wk in sorted(sched.week.unique()):
        playing = set(sched[sched.week == wk].team)
        for t in teams:
            if t not in playing:
                byes.append({"season": season, "week": wk, "team": t,
                             "opponent": "BYE", "home": 0, "gameday": pd.NA,
                             "spread_line": np.nan, "total_line": np.nan,
                             "roof": pd.NA})
    if byes:
        sched = pd.concat([sched, pd.DataFrame(byes)], ignore_index=True)
    return sched.sort_values(["week", "team"]).reset_index(drop=True)


def build_defense_vs_pos(pw: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Defense-vs-position, derived from player weeks. `defense` is the team
    that allowed the production."""
    if pw.empty:
        return pd.DataFrame()
    agg = {"targets": "sum", "rec_yds": "sum", "rec_tds": "sum",
           "carries": "sum", "rush_yds": "sum", "rush_tds": "sum",
           "pass_yds": "sum", "pass_tds": "sum", "fpts_ppr": "sum"}
    by_pos = (pw.groupby(["season", "week", "opponent", "position"], as_index=False)
                .agg(agg).rename(columns={"opponent": "defense"}))
    allpos = (pw.groupby(["season", "week", "opponent"], as_index=False)
                .agg(agg).rename(columns={"opponent": "defense"}))
    allpos["position"] = "ALL"
    out = pd.concat([by_pos, allpos], ignore_index=True)
    out = out.rename(columns={
        "targets": "targets_allowed", "rec_yds": "rec_yds_allowed",
        "rec_tds": "rec_tds_allowed", "carries": "carries_allowed",
        "rush_yds": "rush_yds_allowed", "rush_tds": "rush_tds_allowed",
        "pass_yds": "pass_yds_allowed", "pass_tds": "pass_tds_allowed",
        "fpts_ppr": "fpts_ppr_allowed"})

    # points allowed, from the scoreboard
    g = games[games.game_type == "REG"].copy()
    pa = pd.concat([
        pd.DataFrame({"season": g.season, "week": g.week, "defense": g.home_team,
                      "points_allowed": pd.to_numeric(g.away_score, errors="coerce")}),
        pd.DataFrame({"season": g.season, "week": g.week, "defense": g.away_team,
                      "points_allowed": pd.to_numeric(g.home_score, errors="coerce")}),
    ], ignore_index=True).dropna(subset=["points_allowed"])
    out = out.merge(pa, on=["season", "week", "defense"], how="left")
    return out.sort_values(["season", "week", "defense", "position"]).reset_index(drop=True)


def build_league_rosters(leagues: list[dict], players: pd.DataFrame,
                         season: int, week: int) -> pd.DataFrame:
    """`owner_id` is kept so "my team" is identified by user id, not by a
    display-name string that the user can change at any time."""
    """One row per rostered player per league."""
    rows = []
    px = players.copy()
    px["sleeper_id"] = px.sleeper_id.map(norm_id)
    xw = (px.dropna(subset=["sleeper_id"])
            .drop_duplicates("sleeper_id")
            .set_index("sleeper_id"))
    for lg in leagues:
        meta, rosters, users = lg["league"], lg["rosters"], lg["users"]
        names = {u["user_id"]: (u.get("display_name") or u.get("username") or u["user_id"])
                 for u in users}
        for ros in rosters:
            owner = names.get(ros.get("owner_id"), str(ros.get("owner_id")))
            starters = set(ros.get("starters") or [])
            taxi = set(ros.get("taxi") or [])
            ir = set(ros.get("reserve") or [])
            starters = {norm_id(x) for x in starters}
            taxi = {norm_id(x) for x in taxi}
            ir = {norm_id(x) for x in ir}
            for pid in (ros.get("players") or []):
                pid = norm_id(pid)
                info = xw.loc[pid].to_dict() if pid in xw.index else {}
                rows.append({
                    "season": season, "week": week,
                    "league_id": meta["league_id"], "league_name": meta.get("name"),
                    "roster_id": ros.get("roster_id"),
                    "owner_id": str(ros.get("owner_id")), "owner_name": owner,
                    "sleeper_id": pid,
                    "gsis_id": info.get("gsis_id"),
                    "name": info.get("name"),
                    "position": info.get("position"),
                    "team": info.get("team"),
                    "is_starter": int(pid in starters),
                    "is_taxi": int(pid in taxi),
                    "is_ir": int(pid in ir),
                    "roster_fetched_at": lg.get("fetched_at"),
                })
    return pd.DataFrame(rows)
