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


def _get(season: int, league_id: str, views: list[str], cookies: dict):
    url = f"{BASE}/seasons/{season}/segments/0/leagues/{league_id}"
    r = requests.get(url, params=[("view", v) for v in views],
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


def parse_rosters(data: dict, players: pd.DataFrame, season: int,
                  week: int, meta: dict) -> pd.DataFrame:
    """One row per rostered player, in the same shape as the Sleeper rows from
    build.build_league_rosters so every downstream step is platform-blind.
    `sleeper_id` is filled through the player crosswalk because the recipes
    key on it; `owner_id` is the team id, never the member id."""
    from .build import norm_id
    cols = ["gsis_id", "espn_id", "sleeper_id", "name", "position", "team"]
    xw = players[cols].copy()
    xw["espn_id"] = xw.espn_id.map(norm_id)
    xw = xw.dropna(subset=["espn_id"]).drop_duplicates("espn_id").set_index("espn_id")
    owners = meta.get("owners", {})

    rows = []
    for t in (data.get("teams") or []):
        tid = t.get("id")
        for e in ((t.get("roster") or {}).get("entries") or []):
            player = (e.get("playerPoolEntry") or {}).get("player") or {}
            pid = norm_id(player.get("id"))
            slot = SLOT_ID.get(e.get("lineupSlotId"), str(e.get("lineupSlotId")))
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
            # players: their key would change once nflverse crosswalks them
            # (duplicating the row in the upsert log), and a skill position would
            # leak an unscoreable row into the scored CSVs.
            payload_pos = POS_ID.get(player.get("defaultPositionId"))
            if pd.isna(sid) and not pd.isna(pid) and payload_pos in ("K", "DEF"):
                sid = f"espn-{pid}"
                position = payload_pos
                team = team or PRO_TEAM.get(player.get("proTeamId"))
                name = name or player.get("fullName") or (
                    f"{team} DEF" if position == "DEF" and team else None)
            rows.append({
                "season": season, "week": week,
                "league_id": meta["league_id"], "league_name": meta["name"],
                "roster_id": tid, "owner_id": str(tid),
                "owner_name": owners.get(str(tid), f"team {tid}"),
                "sleeper_id": sid, "espn_id": pid,
                "gsis_id": info.get("gsis_id"), "name": name,
                "position": position, "team": team,
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
