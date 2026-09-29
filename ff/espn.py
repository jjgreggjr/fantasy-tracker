"""ESPN fantasy league support.

ESPN publishes no official API. These endpoints are the ones the browser
itself calls; they work without a key for public leagues and with two session
cookies for private ones. Everything here is wrapped so that any failure logs
a warning and the rest of the pipeline continues without ESPN.

Private leagues: the GitHub Actions workflow writes secrets/espn_cookies.json
from the ESPN_S2 and ESPN_SWID repository secrets at run time:
    {"espn_s2": "<value>", "SWID": "{<value>}"}
Both come from your browser cookies while logged into fantasy.espn.com.
Never paste those values into a chat, a log, or a commit.

Privacy: the repo is public. ESPN identifies league members by their SWID,
which is half of the credential pair above. Nothing derived from
`members[].id` or `teams[].owners` is ever written to a league file; owners
are keyed by team id and labelled with the member's display name.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
UA = {"User-Agent": "Mozilla/5.0 (compatible; ff-volume-tracker/1.0)"}

# ESPN statId -> our scoring key. Unknown ids are logged, never guessed.
# 53 is receptions as ESPN's scoring UI lists it; 58 is targets.
STAT_ID = {
    0: "pass_att", 1: "pass_cmp", 3: "pass_yd", 4: "pass_td", 19: "pass_2pt",
    20: "pass_int", 23: "rush_att", 24: "rush_yd", 25: "rush_td", 26: "rush_2pt",
    42: "rec_yd", 43: "rec_td", 44: "rec_2pt", 53: "rec", 58: "rec_tgt",
    68: "fum", 72: "fum_lost",
}

# ESPN lineupSlotId -> the slot names scoring.SLOT_ELIGIBILITY understands.
# 7 is what ESPN calls OP (offensive player), i.e. a superflex.
SLOT_ID = {
    0: "QB", 1: "TQB", 2: "RB", 3: "WRRB_FLEX", 4: "WR", 5: "REC_FLEX", 6: "TE",
    7: "SUPER_FLEX", 8: "DT", 9: "DE", 10: "LB", 11: "DL", 12: "CB", 13: "S",
    14: "DB", 15: "DP", 16: "DEF", 17: "K", 18: "P", 19: "HC", 20: "BN",
    21: "IR", 23: "FLEX", 24: "ER", 25: "ROOKIE",
}
IDP_SLOTS = {"DT", "DE", "LB", "DL", "CB", "S", "DB", "DP", "ER"}
NON_STARTING = {"BN", "IR"}
POS_ID = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DEF"}

# ESPN proTeamId -> nflverse team abbreviation (0 = free agent, omitted). Used
# only to label the K/DEF roster rows that have no crosswalk entry; matches the
# abbreviations the rest of the pipeline uses (LA not LAR, LV not OAK, JAX, WAS).
PRO_TEAM = {
    1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN",
    8: "DET", 9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV", 14: "LA",
    15: "MIA", 16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ", 21: "PHI",
    22: "ARI", 23: "PIT", 24: "LAC", 25: "SF", 26: "SEA", 27: "TB", 28: "WAS",
    29: "CAR", 30: "JAX", 33: "BAL", 34: "HOU",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_cookies(root: Path) -> dict:
    f = root / "secrets" / "espn_cookies.json"
    if not f.exists():
        return {}
    try:
        c = json.loads(f.read_text(encoding="utf-8"))
        return {k: v for k, v in c.items() if k in ("espn_s2", "SWID") and v}
    except Exception as e:
        log.warning("could not read espn_cookies.json: %s", e)
        return {}


def _get(season: int, league_id: str, views: list[str], cookies: dict,
         scoring_period: int | None = None):
    url = f"{BASE}/seasons/{season}/segments/0/leagues/{league_id}"
    params = [("view", v) for v in views]
    if scoring_period is not None:
        params.append(("scoringPeriodId", scoring_period))
    r = requests.get(url, params=params,
                     headers=UA, cookies=cookies or None, timeout=60)
    if r.status_code in (401, 403):
        raise PermissionError(
            f"ESPN returned {r.status_code} — a private league needs valid "
            "espn_s2 / SWID cookies (ESPN_S2 / ESPN_SWID secrets); 403 with "
            "valid cookies means ESPN rejected the request itself")
    r.raise_for_status()
    return r.json()


def fetch(season: int, league_id: str, root: Path) -> dict | None:
    cookies = load_cookies(root)
    try:
        return _get(season, league_id, ["mSettings", "mRoster", "mTeam"], cookies)
    except Exception as e:
        log.warning("ESPN league %s unavailable: %s", league_id, e)
        return None


def fetch_week(season: int, league_id: str, week: int, root: Path) -> dict | None:
    """One completed scoring period as ESPN recorded it: the lineup each team
    actually set (mRoster) and the matchup scores (mMatchup*), plus settings
    for the matchup-period map. None on any failure, so the caller writes
    nothing rather than something partial."""
    cookies = load_cookies(root)
    try:
        data = _get(season, league_id,
                    ["mSettings", "mRoster", "mMatchup", "mMatchupScore"],
                    cookies, scoring_period=week)
    except Exception as e:
        log.warning("ESPN league %s week %s played lineups unavailable: %s",
                    league_id, week, e)
        return None
    if not isinstance(data, dict):
        log.warning("ESPN league %s week %s: unexpected payload type %s",
                    league_id, week, type(data).__name__)
        return None
    return data


def team_name(t: dict) -> str:
    """ESPN's newer payloads carry `name`; older ones `location` + `nickname`."""
    n = (t.get("name") or "").strip()
    if not n:
        n = f"{t.get('location', '') or ''} {t.get('nickname', '') or ''}".strip()
    return n


def find_my_team(data: dict, entry: dict, cookies: dict):
    """Which `teams[].id` is James's. Team id from config first (stable, not
    secret), then the team owned by the SWID we are logged in as, then a
    name match as the last resort — he can rename the team at any time."""
    teams = data.get("teams") or []
    tid = entry.get("my_team_id")
    if tid is not None:
        for t in teams:
            if str(t.get("id")) == str(tid):
                return t.get("id")
        log.warning("ESPN: my_team_id %s is not among team ids %s",
                    tid, [t.get("id") for t in teams])
    swid = (cookies.get("SWID") or "").strip().lower()
    if swid:
        for t in teams:
            if any(str(o).strip().lower() == swid for o in (t.get("owners") or [])):
                return t.get("id")
    name = (entry.get("my_team_name") or "").strip().lower()
    if name:
        for t in teams:
            if name and name in team_name(t).lower():
                return t.get("id")
    return None


def parse_scoring(data: dict) -> tuple[dict, list]:
    """League scoring in the same keys Sleeper uses, so scoring.score_frame
    applies unchanged. Position overrides on receptions (TE premium and the
    like) become the bonus_rec_<pos> keys the scorer already knows."""
    items = (((data.get("settings") or {}).get("scoringSettings") or {})
             .get("scoringItems") or [])
    scoring, unknown = {}, []
    for it in items:
        sid = it.get("statId")
        pts = float(it.get("points") or 0.0)
        ov = it.get("pointsOverrides") or {}
        key = STAT_ID.get(sid)
        if key is None:
            if pts or ov:
                unknown.append(sid)
            continue
        if pts:
            scoring[key] = pts
        if not ov:
            continue
        if key == "rec":
            for pid_, val in ov.items():
                pos = POS_ID.get(int(pid_))
                if pos in ("TE", "RB", "WR") and val is not None:
                    bonus = round(float(val) - pts, 4)
                    if bonus:
                        scoring[f"bonus_rec_{pos.lower()}"] = bonus
                else:
                    log.info("ESPN reception override for position %s not applied", pid_)
        else:
            log.info("ESPN statId %s (%s) has position overrides %s — not applied",
                     sid, key, ov)
    return scoring, unknown


def parse_meta(data: dict, league_id: str, season: int, entry: dict | None = None,
               cookies: dict | None = None) -> dict:
    entry, cookies = entry or {}, cookies or {}
    s = data.get("settings") or {}
    scoring, unknown = parse_scoring(data)
    if unknown:
        log.info("ESPN scoring statIds not mapped (likely IDP/K/DST/bonuses): %s",
                 sorted(set(unknown))[:30])
    lineup_counts = (s.get("rosterSettings") or {}).get("lineupSlotCounts") or {}
    positions = []
    for sid, n in lineup_counts.items():
        name = SLOT_ID.get(int(sid), f"SLOT{sid}")
        positions += [name] * int(n)
    if any(p.startswith("SLOT") for p in positions):
        log.info("ESPN lineup slot ids not mapped: %s",
                 sorted({p for p in positions if p.startswith("SLOT")}))

    # owners keyed by TEAM id, labelled by display name — never by member id
    members = {str(m.get("id")): (m.get("displayName") or "").strip()
               for m in (data.get("members") or [])}
    owners, team_names = {}, {}
    for t in (data.get("teams") or []):
        tid = str(t.get("id"))
        first = (t.get("owners") or [None])[0]
        owners[tid] = members.get(str(first)) or team_name(t) or f"team {tid}"
        team_names[tid] = team_name(t) or f"team {tid}"

    return {
        "league_id": str(league_id), "platform": "espn",
        "name": s.get("name") or f"ESPN {league_id}",
        "type": entry.get("type") or ("keeper" if s.get("isKeeperLeague") else "redraft"),
        "idp": any(p in IDP_SLOTS for p in positions),
        "scoring": scoring,
        "roster_positions": positions,
        "roster_size": len([p for p in positions if p not in NON_STARTING]),
        "total_slots": len(positions),
        "taxi_slots": 0,
        "reserve_slots": int(lineup_counts.get("21", 0)),
        "owners": owners,
        "team_names": team_names,
        "my_roster_id": find_my_team(data, entry, cookies),
        "my_team_name": entry.get("my_team_name"),
        "season": season,
        "fetched_at": _now(),
    }


def _crosswalk(players: pd.DataFrame) -> pd.DataFrame:
    """The player dimension indexed by ESPN id."""
    from .build import norm_id
    cols = ["gsis_id", "espn_id", "sleeper_id", "name", "position", "team"]
    xw = players[cols].copy()
    xw["espn_id"] = xw.espn_id.map(norm_id)
    return xw.dropna(subset=["espn_id"]).drop_duplicates("espn_id").set_index("espn_id")


def _identity(player: dict, xw: pd.DataFrame, key_all: bool = False) -> dict:
    """Ids and labels for one ESPN player payload, through the crosswalk.

    `key_all` widens the "espn-<id>" fallback from K/DEF to every unmatched
    player. Snapshots must not (see below); played lineups can, because they
    are replaced a whole league-week at a time, so a key that changes once
    nflverse crosswalks the player cannot leave a duplicate behind, and
    dropping an unmatched starter would falsify the lineup.
    """
    from .build import norm_id
    pid = norm_id(player.get("id"))
    info = xw.loc[pid].to_dict() if not pd.isna(pid) and pid in xw.index else {}
    sid = norm_id(info.get("sleeper_id"))
    name = info.get("name")
    position = info.get("position")
    team = info.get("team")
    # K and DEF are never in the skill-only crosswalk (build.SKILL), so
    # without this they have no sleeper_id and run_weekly's upsert drops
    # them. Take their identity from ESPN's own payload and key them as
    # "espn-<id>": that key is stable because K/DEF can never gain a
    # crosswalk entry, and score.compute's position filter keeps them out
    # of the scored roster.csv. Deliberately NOT applied to unmatched skill
    # players in snapshots: their key would change once nflverse crosswalks
    # them (duplicating the row in the upsert log), and a skill position would
    # leak an unscoreable row into the scored CSVs.
    payload_pos = POS_ID.get(player.get("defaultPositionId"))
    if pd.isna(sid) and not pd.isna(pid) and (key_all or payload_pos in ("K", "DEF")):
        sid = f"espn-{pid}"
        position = payload_pos or position
        team = team or PRO_TEAM.get(player.get("proTeamId"))
        name = name or player.get("fullName") or (
            f"{team} DEF" if position == "DEF" and team else None)
    return {"pid": pid, "sleeper_id": sid, "gsis_id": info.get("gsis_id"),
            "name": name, "position": position, "team": team}


def parse_rosters(data: dict, players: pd.DataFrame, season: int,
                  week: int, meta: dict) -> pd.DataFrame:
    """One row per rostered player, in the same shape as the Sleeper rows from
    build.build_league_rosters so every downstream step is platform-blind.
    `sleeper_id` is filled through the player crosswalk because the recipes
    key on it; `owner_id` is the team id, never the member id."""
    xw = _crosswalk(players)
    owners = meta.get("owners", {})

    rows = []
    for t in (data.get("teams") or []):
        tid = t.get("id")
        for e in ((t.get("roster") or {}).get("entries") or []):
            player = (e.get("playerPoolEntry") or {}).get("player") or {}
            slot = SLOT_ID.get(e.get("lineupSlotId"), str(e.get("lineupSlotId")))
            who = _identity(player, xw)
            rows.append({
                "season": season, "week": week,
                "league_id": meta["league_id"], "league_name": meta["name"],
                "roster_id": tid, "owner_id": str(tid),
                "owner_name": owners.get(str(tid), f"team {tid}"),
                "sleeper_id": who["sleeper_id"], "espn_id": who["pid"],
                "gsis_id": who["gsis_id"], "name": who["name"],
                "position": who["position"], "team": who["team"],
                "is_starter": int(slot not in NON_STARTING),
                "is_taxi": 0, "is_ir": int(slot == "IR"),
                "roster_fetched_at": meta.get("fetched_at"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        miss = int(df.gsis_id.isna().sum())
        if miss:
            log.info("ESPN: %d of %d rostered players unmatched to gsis_id",
                     miss, len(df))
    return df


# --------------------------------------------------------------------------
# Played weeks
# --------------------------------------------------------------------------
def _entry_points(entry: dict, week: int) -> float | None:
    """Actual points a roster entry scored in scoring period `week`, in this
    league's scoring. The explicit per-period stat line is unambiguous, so it
    wins over `appliedStatTotal`, which is only as period-specific as the
    request's scoringPeriodId was."""
    pe = entry.get("playerPoolEntry") or {}
    for s in ((pe.get("player") or {}).get("stats") or []):
        if (s.get("scoringPeriodId") == week and s.get("statSourceId") == 0
                and s.get("statSplitTypeId") == 1
                and s.get("appliedTotal") is not None):
            return float(s["appliedTotal"])
    if pe.get("appliedStatTotal") is not None:
        return float(pe["appliedStatTotal"])
    return None


def _side_points(side: dict, week: int) -> float | None:
    by_period = side.get("pointsByScoringPeriod") or {}
    v = by_period.get(str(week))
    if v is None:
        v = side.get("totalPoints")
    return None if v is None else float(v)


def _matchup_periods(data: dict, week: int) -> tuple[set[int], bool]:
    """Matchup period ids that hold scoring period `week`, and whether any of
    them spans several scoring periods (then a matchup's total is not one
    week's score). Defaults to the usual one-week-per-matchup layout."""
    mps = (((data.get("settings") or {}).get("scheduleSettings") or {})
           .get("matchupPeriods") or {})
    ids, multi = set(), False
    for k, v in mps.items():
        try:
            if week in [int(x) for x in v]:
                ids.add(int(k))
                multi = multi or len(v) > 1
        except (TypeError, ValueError):
            continue
    return (ids or {week}), multi


def _outcome(m: dict, home_pts, away_pts) -> tuple[int | None, int | None]:
    """(home won, tied). ESPN's own `winner` when it has decided; the scores
    otherwise (a week can read UNDECIDED until ESPN finalizes it)."""
    w = str(m.get("winner") or "").upper()
    if w == "HOME":
        return 1, 0
    if w == "AWAY":
        return 0, 0
    if w == "TIE":
        return 0, 1
    if home_pts is not None and away_pts is not None:
        return int(home_pts > away_pts), int(home_pts == away_pts)
    return None, None


def parse_played(data: dict, players: pd.DataFrame, season: int, week: int,
                 meta: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(lineups, results) for one completed scoring period, in the columns
    build.PLAYED_COLS / build.RESULT_COLS. Lineup rows are every rostered
    player with the slot he occupied and the points he scored; results are one
    row per team from the matchup schedule. Either frame is empty when ESPN
    returned nothing usable, which the caller treats as "write nothing"."""
    from .build import PLAYED_COLS, RESULT_COLS
    xw = _crosswalk(players)
    owners = meta.get("owners", {})
    lid, lname = meta["league_id"], meta["name"]
    fetched = _now()
    owner = lambda tid: owners.get(str(tid), f"team {tid}")

    rows = []
    for t in (data.get("teams") or []):
        tid = t.get("id")
        for e in ((t.get("roster") or {}).get("entries") or []):
            player = (e.get("playerPoolEntry") or {}).get("player") or {}
            slot = SLOT_ID.get(e.get("lineupSlotId"), str(e.get("lineupSlotId")))
            who = _identity(player, xw, key_all=True)
            rows.append({
                "season": season, "week": week, "league_id": lid,
                "league_name": lname, "roster_id": tid, "owner_id": str(tid),
                "owner_name": owner(tid),
                "sleeper_id": who["sleeper_id"], "gsis_id": who["gsis_id"],
                "espn_id": who["pid"], "name": who["name"],
                "position": who["position"], "team": who["team"],
                "started": int(slot not in NON_STARTING), "slot": slot,
                "points": _entry_points(e, week), "fetched_at": fetched,
            })
    lineups = pd.DataFrame(rows, columns=PLAYED_COLS)
    if not lineups.empty:
        lineups = lineups.sort_values(
            ["roster_id", "started", "points"], ascending=[True, False, False],
            kind="stable").reset_index(drop=True)

    mp_ids, multi = _matchup_periods(data, week)
    res = []
    if multi:
        log.info("ESPN %s week %s sits in a multi-week matchup period; "
                 "no per-week results written", lid, week)
    else:
        for m in (data.get("schedule") or []):
            if m.get("matchupPeriodId") not in mp_ids:
                continue
            home, away = m.get("home") or {}, m.get("away") or {}
            if home.get("teamId") is None or away.get("teamId") is None:
                continue                                   # bye
            hp, ap = _side_points(home, week), _side_points(away, week)
            home_won, tie = _outcome(m, hp, ap)
            away_won = None if home_won is None else int(not home_won and not tie)
            for me, opp, pf, pa, won in ((home, away, hp, ap, home_won),
                                         (away, home, ap, hp, away_won)):
                res.append({
                    "season": season, "week": week, "league_id": lid,
                    "league_name": lname, "roster_id": me["teamId"],
                    "owner_id": str(me["teamId"]), "owner_name": owner(me["teamId"]),
                    "matchup_id": m.get("id"),
                    "opponent_roster_id": opp["teamId"],
                    "opponent_owner_name": owner(opp["teamId"]),
                    "points_for": pf, "points_against": pa,
                    "won": won, "tie": tie, "median_won": None,
                    "fetched_at": fetched,
                })
    results = pd.DataFrame(res, columns=RESULT_COLS)

    if not lineups.empty:
        pts = lineups.points
        if pts.isna().all():
            log.warning("ESPN %s week %s: lineups carried no points at all "
                        "(payload shape changed?)", lid, week)
        if not results.empty:
            tot = (lineups[lineups.started == 1].groupby("roster_id").points.sum())
            pf = results.set_index("roster_id").points_for
            common = tot.index.intersection(pf.index)
            off = int(((tot[common] - pf[common]).abs() > 0.5).sum())
            log.info("ESPN %s week %s: %d entries, %d started; started points "
                     "match the schedule total for %d of %d teams",
                     lid, week, len(lineups), int(lineups.started.sum()),
                     len(common) - off, len(common))
    return lineups, results
