"""ESPN fantasy league support.

ESPN publishes no official API. These endpoints are the ones the browser
itself calls; they work without a key for public leagues and with two session
cookies for private ones. Everything here is wrapped so that any failure logs
a warning and the rest of the pipeline continues without ESPN.

Private leagues: create secrets/espn_cookies.json on this machine:
    {"espn_s2": "<value>", "SWID": "{<value>}"}
Both come from your browser cookies while logged into fantasy.espn.com.
Never paste those values into a chat.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
UA = {"User-Agent": "Mozilla/5.0 (compatible; ff-volume-tracker/1.0)"}

# ESPN statId -> our scoring key. Unknown ids are logged, never guessed.
STAT_ID = {
    0: "pass_att", 1: "pass_cmp", 3: "pass_yd", 4: "pass_td", 19: "pass_2pt",
    20: "pass_int", 23: "rush_att", 24: "rush_yd", 25: "rush_td", 26: "rush_2pt",
    41: "rec_tgt", 42: "rec_yd", 43: "rec_td", 44: "rec_2pt", 53: "rec",
    58: "rec_tgt", 68: "fum", 72: "fum_lost",
}

SLOT_ID = {0: "QB", 2: "RB", 4: "WR", 6: "TE", 16: "DEF", 17: "K",
           20: "BN", 21: "IR", 23: "FLEX"}
POS_ID = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DEF"}


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
    if r.status_code == 401:
        raise PermissionError(
            "ESPN returned 401 — this league is private and needs valid "
            "espn_s2 / SWID cookies in secrets/espn_cookies.json")
    r.raise_for_status()
    return r.json()


def fetch(season: int, league_id: str, root: Path) -> dict | None:
    cookies = load_cookies(root)
    try:
        return _get(season, league_id, ["mSettings", "mRoster", "mTeam"], cookies)
    except Exception as e:
        log.warning("ESPN league %s unavailable: %s", league_id, e)
        return None


def parse_scoring(data: dict) -> tuple[dict, list]:
    items = (((data.get("settings") or {}).get("scoringSettings") or {})
             .get("scoringItems") or [])
    scoring, unknown = {}, []
    for it in items:
        sid = it.get("statId")
        pts = it.get("points")
        if pts in (None, 0):
            ov = it.get("pointsOverrides") or {}
            pts = next(iter(ov.values()), None) if ov else None
        if not pts:
            continue
        key = STAT_ID.get(sid)
        if key:
            scoring[key] = float(pts)
        else:
            unknown.append(sid)
    return scoring, unknown


def parse_meta(data: dict, league_id: str, season: int) -> dict:
    s = data.get("settings") or {}
    scoring, unknown = parse_scoring(data)
    if unknown:
        log.info("ESPN scoring statIds not mapped (likely IDP/K/DST): %s",
                 sorted(set(unknown))[:20])
    lineup_counts = (s.get("rosterSettings") or {}).get("lineupSlotCounts") or {}
    positions = []
    for sid, n in lineup_counts.items():
        name = SLOT_ID.get(int(sid), f"SLOT{sid}")
        positions += [name] * int(n)
    owners = {}
    for m in (data.get("members") or []):
        owners[str(m.get("id"))] = (m.get("displayName")
                                    or m.get("firstName") or str(m.get("id")))
    keeper = bool((s.get("draftSettings") or {}).get("isTradingEnabled")) and False
    return {
        "league_id": str(league_id), "platform": "espn",
        "name": s.get("name") or f"ESPN {league_id}",
        "type": "keeper" if s.get("isKeeperLeague") else "redraft",
        "idp": any(p.startswith("SLOT") for p in positions),
        "scoring": scoring,
        "roster_positions": positions,
        "roster_size": len([p for p in positions if p not in ("BN", "IR")]),
        "total_slots": len(positions),
        "taxi_slots": 0,
        "reserve_slots": int(lineup_counts.get("21", 0)),
        "owners": owners,
        "my_roster_id": None,
        "season": season,
    }


def parse_rosters(data: dict, players: pd.DataFrame, season: int,
                  week: int, meta: dict, my_team_id=None) -> pd.DataFrame:
    from .build import norm_id
    xw = players[["gsis_id", "espn_id", "name", "position", "team"]].copy()
    xw["espn_id"] = xw.espn_id.map(norm_id)
    xw = xw.dropna(subset=["espn_id"]).drop_duplicates("espn_id").set_index("espn_id")

    rows = []
    for t in (data.get("teams") or []):
        tid = t.get("id")
        owner_ids = t.get("owners") or []
        owner = ", ".join(meta.get("owners", {}).get(str(o), str(o))
                          for o in owner_ids) or f"team {tid}"
        for e in ((t.get("roster") or {}).get("entries") or []):
            pid = norm_id(((e.get("playerPoolEntry") or {}).get("player") or {}).get("id"))
            slot = SLOT_ID.get(e.get("lineupSlotId"), str(e.get("lineupSlotId")))
            info = xw.loc[pid].to_dict() if pid is not pd.NA and pid in xw.index else {}
            rows.append({
                "season": season, "week": week,
                "league_id": meta["league_id"], "league_name": meta["name"],
                "roster_id": tid, "owner_id": str(owner_ids[0]) if owner_ids else "",
                "owner_name": owner, "sleeper_id": pd.NA, "espn_id": pid,
                "gsis_id": info.get("gsis_id"), "name": info.get("name"),
                "position": info.get("position"), "team": info.get("team"),
                "is_starter": int(slot not in ("BN", "IR")),
                "is_taxi": 0, "is_ir": int(slot == "IR"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        miss = int(df.gsis_id.isna().sum())
        if miss:
            log.info("ESPN: %d of %d rostered players unmatched to gsis_id",
                     miss, len(df))
    return df
