"""Player status, and — critically — the status of whoever is ahead of him.

The Charbonnet miss was not a math error. He was listed RB4 because he was on
PUP, and nothing in the pipeline asked *why* the depth chart said what it said.
This module makes that question mandatory: for every player we name, it reports
his own availability and the availability of everyone listed above him.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

OUT_STATES = {"Out", "IR", "PUP", "Sus", "NA", "DNR"}
DOUBTFUL = {"Doubtful"}
SOFT = {"Questionable"}

# The gameday inactive list (nflverse weekly_rosters, `status == "INA"`) is posted about 90 minutes before each
# game's kickoff; Thursday night's is already in Friday's file. It is the league's own word that a player will not
# dress, so it outranks every designation: flag "inactive", play_prob 0, E_pts blanked (flag INACTIVE in score.py).
INACTIVE_LEAD = pd.Timedelta(minutes=90)
INACTIVE = "inactive"

# practice participation -> how much it tells you about Sunday
PRACTICE_WEIGHT = {"DNP": 0.15, "Did Not Participate In Practice": 0.15,
                   "Limited": 0.65, "Limited Participation in Practice": 0.65,
                   "Full": 0.95, "Full Participation in Practice": 0.95}


def _severity(s: str) -> int:
    if s in SOFT:
        return 1
    if s in DOUBTFUL:
        return 2
    return 3                     # Out / IR / PUP / Sus / anything we do not recognise


def designation(row: pd.Series) -> str | None:
    """The game designation to act on. Two independent signals: the nflverse report
    for this week (`report_status`, the league's own Wed-Fri report) and Sleeper's live
    `injury_status` (updated through Sunday morning, so it can know more than the report).
    Either one can be the only one that has spoken; when both have, the more severe wins,
    so neither can talk the other out of a warning."""
    cands = [str(v).strip() for v in (row.get("report_status"), row.get("injury_status"))
             if pd.notna(v) and str(v).strip()]
    return max(cands, key=_severity) if cands else None


def _is_inactive(row: pd.Series) -> bool:
    v = row.get("inactive")
    return bool(pd.notna(v) and v not in (0, False, "0", ""))


def status_flag(row: pd.Series) -> str:
    if _is_inactive(row):                  # gameday inactive list: OUT-equivalent, whatever the designation said
        return INACTIVE
    s = designation(row)
    if s is None:
        return "healthy"
    if s in OUT_STATES:
        return s.lower()
    if s in DOUBTFUL:
        return "doubtful"
    if s in SOFT:
        return "questionable"
    return s.lower()


def play_probability(row: pd.Series) -> float:
    """Rough chance he suits up, from designation plus practice participation.

    Practice is the stronger signal: a Questionable player who practiced fully
    on Friday is a very different bet from one who did not practice at all.
    """
    flag = status_flag(row)
    if flag == INACTIVE:
        return 0.0
    base = {"healthy": 0.97, "questionable": 0.72, "doubtful": 0.28}.get(flag, 0.02)
    pr = row.get("practice_participation")
    if pd.notna(pr) and str(pr) in PRACTICE_WEIGHT:
        # let practice pull the estimate, but never override a hard OUT
        if flag in ("healthy", "questionable", "doubtful"):
            base = 0.4 * base + 0.6 * PRACTICE_WEIGHT[str(pr)]
    return round(float(base), 2)


def _news_age_hours(v) -> float:
    if pd.isna(v):
        return np.nan
    try:
        ts = float(v)
    except (TypeError, ValueError):
        return np.nan
    if ts > 1e12:          # Sleeper sends milliseconds
        ts /= 1000.0
    age = (datetime.now(timezone.utc) - datetime.fromtimestamp(ts, timezone.utc))
    return round(age.total_seconds() / 3600, 1)


def week_reports(injuries: pd.DataFrame | None, week: int | None) -> pd.DataFrame:
    """nflverse injury-report rows issued FOR `week`, one per player.

    The file carries one row per player-week (practice participation, plus the game
    designation once the Friday report is out), keyed by the week the report is for. A
    report for an earlier week is last week's news and is never carried over, so before
    Wednesday this is simply empty. When a file does carry `date_modified` (seasons up
    to 2024) the newest report of the week wins; otherwise the last row does.
    """
    cols = ["gsis_id", "report_status", "practice_status",
            "report_primary_injury", "practice_primary_injury"]
    if injuries is None or injuries.empty or week is None or "week" not in injuries.columns:
        return pd.DataFrame(columns=cols)
    r = injuries[(pd.to_numeric(injuries["week"], errors="coerce") == week)
                 & injuries["gsis_id"].notna()].copy()
    if "date_modified" in r.columns:
        r = r.sort_values("date_modified", kind="stable")
    r = r.drop_duplicates("gsis_id", keep="last")
    for c in cols[1:]:
        if c not in r.columns:
            r[c] = np.nan
        r[c] = r[c].where(r[c].notna() & (r[c].astype(str).str.strip() != ""), np.nan)
    return r[cols].reset_index(drop=True)


def with_reports(players: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    """`players` plus this week's nflverse report: `report_status` (the designation) and
    `practice_participation`. Sleeper's own practice field is null for every player all
    season, so nflverse is the source; Sleeper's value is kept only where nflverse has none.
    `injury_status` (Sleeper, live) is left alone: it is the second signal."""
    p = players.drop(columns=["report_status"], errors="ignore")
    if reports is None or reports.empty:
        p["report_status"] = np.nan
        return p
    r = reports.set_index("gsis_id")
    p["report_status"] = p["gsis_id"].map(r["report_status"])
    nfl_practice = p["gsis_id"].map(r["practice_status"])
    old = p["practice_participation"] if "practice_participation" in p else np.nan
    p["practice_participation"] = nfl_practice.where(nfl_practice.notna(), old)
    return p


INACTIVE_COLS = ["gsis_id", "team", "kickoff", "inactive_known_at"]


def team_kickoffs(games: pd.DataFrame | None, season: int) -> pd.DataFrame:
    """(season, week, team, kickoff UTC) for every scheduled game of `season`, from nflverse's games table
    (gameday + gametime are US Eastern). A row whose time cannot be read gets NaT, never an exception."""
    cols = ["season", "week", "team", "kickoff"]
    if games is None or games.empty or not {"gameday", "gametime", "home_team", "away_team"} <= set(games.columns):
        return pd.DataFrame(columns=cols)
    g = games[pd.to_numeric(games["season"], errors="coerce") == season].copy()
    local = pd.to_datetime(g["gameday"].astype(str) + " " + g["gametime"].astype(str), errors="coerce")
    try:
        g["kickoff"] = (local.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
                        .dt.tz_convert("UTC"))
    except Exception:                      # a malformed schedule must never stop the run: unknown kickoffs
        g["kickoff"] = pd.NaT
    out = pd.concat([g[["season", "week", "home_team", "kickoff"]].rename(columns={"home_team": "team"}),
                     g[["season", "week", "away_team", "kickoff"]].rename(columns={"away_team": "team"})],
                    ignore_index=True)
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    return out[cols].drop_duplicates(["season", "week", "team"], keep="first").reset_index(drop=True)


def _ina_rows(rosters: pd.DataFrame | None, season: int) -> pd.DataFrame:
    """nflverse weekly_rosters rows with `status == "INA"` for `season`. Note: `status_description_abbr` is NOT
    the signal (2026 INA rows carry A01, 'active'; I01 is rare before 2024), so only `status` is read."""
    need = {"status", "week", "gsis_id"}
    if rosters is None or rosters.empty or not need <= set(rosters.columns):
        return pd.DataFrame(columns=["season", "week", "team", "gsis_id"])
    r = rosters[rosters["status"].astype(str).str.strip().str.upper().eq("INA") & rosters["gsis_id"].notna()].copy()
    if "season" in r.columns:
        r = r[pd.to_numeric(r["season"], errors="coerce") == season]
    r["season"] = season
    r["week"] = pd.to_numeric(r["week"], errors="coerce")
    if "team" not in r.columns:
        r["team"] = np.nan
    return r


def _with_kickoff(r: pd.DataFrame, kickoffs: pd.DataFrame | None) -> pd.DataFrame:
    r = r.copy()
    if kickoffs is not None and not kickoffs.empty:
        k = kickoffs.drop_duplicates(["week", "team"]).set_index(["week", "team"])["kickoff"]
        r["kickoff"] = pd.Series(list(zip(r["week"], r["team"])), index=r.index).map(k)
    else:
        r["kickoff"] = pd.NaT
    r["kickoff"] = pd.to_datetime(r["kickoff"], utc=True, errors="coerce")
    known = r["kickoff"] - INACTIVE_LEAD
    r["inactive_known_at"] = known.dt.strftime("%Y-%m-%dT%H:%M:%SZ").where(known.notna(), np.nan)
    return r


def week_inactives(rosters: pd.DataFrame | None, season: int, week: int | None,
                   kickoffs: pd.DataFrame | None = None, now=None) -> pd.DataFrame:
    """Players on the gameday inactive list FOR `week` whose game has not kicked off at `now`.

    Two guards. (1) The row's `week` must be the target week: the season file keeps every earlier week's list, and
    last week's inactive is not this week's (same rule as `week_reports`). (2) The game must not have started: once
    it has, the lineup is locked and the frozen record is the one that stands, so a game under way is not re-flagged.
    A row whose kickoff cannot be found (no schedule row) is kept on the week guard alone. `inactive_known_at` is
    kickoff - 90 min, when nflverse publishes the list (the file carries no per-row timestamp)."""
    if week is None:
        return pd.DataFrame(columns=INACTIVE_COLS)
    r = _ina_rows(rosters, season)
    r = r[r["week"] == week]
    if r.empty:
        return pd.DataFrame(columns=INACTIVE_COLS)
    r = _with_kickoff(r, kickoffs)
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    r = r[r["kickoff"].isna() | (r["kickoff"] > now)]
    return r.drop_duplicates("gsis_id")[INACTIVE_COLS].reset_index(drop=True)


def inactive_archive(rosters: pd.DataFrame | None, season: int,
                     kickoffs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every inactive-list row of the season (all weeks), for data/inactives.csv: the committed record the
    retrospective joins on (gsis_id, week). Not guarded: it is history, not this week's availability."""
    cols = ["season", "week", "game_type", "team", "gsis_id", "name", "position", "inactive_known_at"]
    r = _ina_rows(rosters, season)
    if r.empty:
        return pd.DataFrame(columns=cols)
    r = _with_kickoff(r, kickoffs)
    r["name"] = r["full_name"] if "full_name" in r.columns else np.nan
    for c in ("game_type", "position"):
        if c not in r.columns:
            r[c] = np.nan
    r["week"] = r["week"].astype("Int64")
    return r.drop_duplicates(["week", "gsis_id"])[cols].reset_index(drop=True)


def build(players: pd.DataFrame, depth: pd.DataFrame,
          proj: pd.DataFrame | None = None, week: int | None = None,
          injuries: pd.DataFrame | None = None,
          inactives: pd.DataFrame | None = None) -> pd.DataFrame:
    """One status row per player on a depth chart, including the blocker chain.

    `injuries` is nflverse's injuries_{season} table; its rows for `week` supply the
    practice participation and the game designation (see `week_reports`). `inactives` is
    `week_inactives(...)`: the week's gameday inactive list, already guarded. A player on it
    is flagged `inactive` (play_prob 0) whatever his designation; `injury_status` stays as
    Sleeper reported it."""
    if depth is None or depth.empty:
        return pd.DataFrame()

    p = with_reports(players, week_reports(injuries, week))
    ina = (inactives.drop_duplicates("gsis_id").set_index("gsis_id")
           if inactives is not None and not inactives.empty else None)
    p["inactive"] = p["gsis_id"].isin(ina.index).astype(int) if ina is not None else 0
    p["inactive_known_at"] = (p["gsis_id"].map(ina["inactive_known_at"])
                              if ina is not None and "inactive_known_at" in ina else np.nan)
    p["status_flag"] = p.apply(status_flag, axis=1)
    p["play_prob"] = p.apply(play_probability, axis=1)
    p["news_age_hours"] = p.get("news_updated", pd.Series(index=p.index)).map(_news_age_hours)

    d = depth.merge(
        p[["gsis_id", "status_flag", "report_status", "play_prob", "news_age_hours",
           "injury_body_part", "practice_participation", "sleeper_depth_order",
           "inactive", "inactive_known_at"]],
        on="gsis_id", how="left")
    d["inactive"] = pd.to_numeric(d["inactive"], errors="coerce").fillna(0).astype(int)
    # A player on a depth chart with no match in players.csv has unknown
    # availability, which is not the same as healthy — say so rather than
    # letting NaN propagate into the blocker chain.
    d["status_flag"] = d.status_flag.fillna("unknown")
    d["play_prob"] = pd.to_numeric(d.play_prob, errors="coerce").fillna(0.85)

    # two independent depth charts: ESPN (via nflverse) and Sleeper's own.
    # A big disagreement is the signature of a rank driven by availability
    # rather than by role — exactly the case that misled us.
    sd = pd.to_numeric(d.get("sleeper_depth_order"), errors="coerce")
    d["depth_disagreement"] = (pd.to_numeric(d["rank"], errors="coerce") - sd).abs()

    share = None
    if proj is not None and not proj.empty and week is not None:
        pw = proj[proj.week == week]
        share = pw.set_index("gsis_id")[["proj_carry_share", "proj_target_share"]]

    rows = []
    for (team, pos), g in d.groupby(["team", "position"]):
        g = g.sort_values("rank")
        for _, r in g.iterrows():
            ahead = g[g["rank"] < r["rank"]]
            blockers = [{"name": b["name"], "rank": int(b["rank"]),
                         "status": b.status_flag, "play_prob": b.play_prob,
                         "practice": b.practice_participation}
                        for _, b in ahead.iterrows()]
            # volume held by players ahead of him who probably will not play
            opp_ahead = 0.0
            if share is not None:
                col = "proj_carry_share" if pos == "RB" else "proj_target_share"
                for _, b in ahead.iterrows():
                    if b.gsis_id in share.index and pd.notna(b.play_prob):
                        v = share.loc[b.gsis_id, col]
                        if pd.notna(v):
                            opp_ahead += float(v) * (1 - float(b.play_prob))
            rows.append({
                "gsis_id": r.gsis_id, "name": r["name"], "team": team,
                "position": pos, "espn_rank": r["rank"],
                "sleeper_rank": r.sleeper_depth_order,
                "depth_disagreement": r.depth_disagreement,
                "status_flag": r.status_flag, "report_status": r.report_status,
                "play_prob": r.play_prob,
                "inactive": int(r.inactive),
                "inactive_known_at": r.inactive_known_at,
                "injury_body_part": r.injury_body_part,
                "practice": r.practice_participation,
                "news_age_hours": r.news_age_hours,
                "n_blockers": len(blockers),
                "blockers_out": sum(1 for b in blockers
                                    if b["play_prob"] is not None
                                    and pd.notna(b["play_prob"])
                                    and b["play_prob"] < 0.5),
                "opportunity_ahead": round(opp_ahead, 3),
                "blocked_by": "; ".join(
                    f"{b['name']} ({b['status']}"
                    + (f", {b['practice']}"
                       if b["practice"] is not None and pd.notna(b["practice"])
                       else "")
                    + ")" for b in blockers) or "nobody",
            })
    return pd.DataFrame(rows)


def warnings(st: pd.DataFrame, roster_ids: set | None = None) -> list[str]:
    """Things a human should look at before trusting a recommendation."""
    if st is None or st.empty:
        return []
    s = st if roster_ids is None else st[st.gsis_id.isin(roster_ids)]
    out = []
    dis = s[s.depth_disagreement >= 2]
    for _, r in dis.iterrows():
        out.append(f"{r['name']}: depth charts disagree (ESPN {int(r.espn_rank)} "
                   f"vs Sleeper {int(r.sleeper_rank)}) — his listed rank may "
                   f"reflect availability, not role")
    opp = s[s.opportunity_ahead >= 0.15]
    for _, r in opp.iterrows():
        out.append(f"{r['name']}: {r.opportunity_ahead:.0%} of the volume ahead "
                   f"of him belongs to players unlikely to play ({r.blocked_by})")
    hurt = s[(s.status_flag.isin(["out", "ir", "pup", "doubtful"]))]
    for _, r in hurt.iterrows():
        bp = f" ({r.injury_body_part})" if pd.notna(r.injury_body_part) else ""
        out.append(f"{r['name']}: {r.status_flag.upper()}{bp} — his depth rank "
                   f"of {int(r.espn_rank)} reflects that, not his role")
    for _, r in s[s.status_flag.eq(INACTIVE)].iterrows():
        out.append(f"{r['name']}: INACTIVE on the gameday list ({_inactive_note(r)}) — "
                   "he will not dress; E_pts blanked")
    return out


def _inactive_note(r) -> str:
    when = r.get("inactive_known_at")
    return "posted ~" + str(when) if pd.notna(when) and str(when) else "posted ~90 min before kickoff"


def describe(st: pd.DataFrame, name: str) -> str:
    """One-line status summary for a named player, for use in any answer."""
    if st is None or st.empty:
        return ""
    hit = st[st["name"].str.contains(name, case=False, na=False)]
    if hit.empty:
        return ""
    r = hit.iloc[0]
    bits = [f"{r.status_flag}"]
    if r.get("status_flag") == INACTIVE:
        bits = [f"INACTIVE (gameday inactive list, {_inactive_note(r)}; will not play)"]
    if pd.notna(r.practice):
        bits.append(f"practice: {r.practice}")
    if pd.notna(r.injury_body_part):
        bits.append(str(r.injury_body_part))
    bits.append(f"{r.position}{int(r.espn_rank)}, behind: {r.blocked_by}")
    if r.opportunity_ahead >= 0.10:
        bits.append(f"{r.opportunity_ahead:.0%} of volume ahead of him is in doubt")
    return " · ".join(bits)
