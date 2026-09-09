"""Per-league scoring.

Fantasy points are never stored as a fact. We keep raw stat lines and apply
each league's own rules on top, which means points exist retroactively for
every week we have stats for, in every league, and change correctly the
moment a league changes its settings.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Sleeper scoring key -> column in player_weeks.csv. Anything not listed here
# is either out of scope (IDP, DST, kicking) or handled as a bonus below.
STAT_KEYS = {
    "pass_yd": "pass_yds",
    "pass_td": "pass_tds",
    "pass_int": "pass_int",
    "pass_att": "pass_att",
    "pass_cmp": "pass_cmp",
    "pass_2pt": None,          # folded into two_pt
    "rush_yd": "rush_yds",
    "rush_td": "rush_tds",
    "rush_att": "carries",
    "rush_fd": "rush_fd",
    "rush_2pt": None,
    "rec": "receptions",
    "rec_yd": "rec_yds",
    "rec_td": "rec_tds",
    "rec_tgt": "targets",
    "rec_fd": "rec_fd",
    "rec_2pt": None,
    "fum": "fum",
    "fum_lost": "fum_lost",
}

# Per-reception bonuses that apply only to one position.
POS_REC_BONUS = {"bonus_rec_te": "TE", "bonus_rec_rb": "RB", "bonus_rec_wr": "WR"}

# Yardage milestone bonuses: key -> (column, threshold)
YARD_BONUS = {
    "bonus_rush_yd_100": ("rush_yds", 100), "bonus_rush_yd_200": ("rush_yds", 200),
    "bonus_rec_yd_100": ("rec_yds", 100), "bonus_rec_yd_200": ("rec_yds", 200),
    "bonus_pass_yd_300": ("pass_yds", 300), "bonus_pass_yd_400": ("pass_yds", 400),
}

# Keys we knowingly ignore: defensive/IDP, kicking, special teams, team defense.
IGNORE_PREFIXES = ("def_", "idp_", "st_", "pts_allow", "yds_allow", "sack",
                   "int_ret", "fum_ret", "blk_", "safe", "ff", "tkl", "qb_hit")
IGNORE_EXACT = {"fgm", "fga", "xpm", "xpa", "fgmiss", "xpmiss", "rec_yd_bonus",
                "bonus_def_", "pass_sack", "pass_fd"}


def score_frame(pw: pd.DataFrame, scoring: dict, label: str = "") -> pd.Series:
    """Fantasy points for every row of `pw` under one league's `scoring` dict."""
    if pw.empty:
        return pd.Series(dtype=float)
    pts = pd.Series(0.0, index=pw.index)
    used, unhandled = set(), []

    col = lambda c: pd.to_numeric(pw.get(c), errors="coerce").fillna(0.0)

    for key, val in (scoring or {}).items():
        if not val:
            continue
        k = str(key)
        if k.startswith(IGNORE_PREFIXES) or k in IGNORE_EXACT:
            continue

        if k in ("pass_2pt", "rush_2pt", "rec_2pt"):
            # nflverse gives one combined 2pt count; splitting it would be a
            # guess, so credit it once at the pass_2pt rate and skip the rest.
            if k == "pass_2pt":
                pts += col("two_pt") * float(val)
            used.add(k)
            continue

        if k in STAT_KEYS and STAT_KEYS[k]:
            pts += col(STAT_KEYS[k]) * float(val)
            used.add(k)
            continue

        if k in POS_REC_BONUS:
            mask = (pw.position == POS_REC_BONUS[k]).astype(float)
            pts += col("receptions") * mask * float(val)
            used.add(k)
            continue

        if k in YARD_BONUS:
            c, thresh = YARD_BONUS[k]
            pts += (col(c) >= thresh).astype(float) * float(val)
            used.add(k)
            continue

        unhandled.append(k)

    if unhandled:
        log.info("%s: scoring keys not applied (out of scope or unknown): %s",
                 label or "league", ", ".join(sorted(unhandled)[:20]))
    return pts.round(2)


def sleeper_league_meta(league: dict, users: list, rosters: list,
                        my_user_id: str) -> dict:
    """Extract the rules the recipes need from a Sleeper league object."""
    st = league.get("settings") or {}
    type_map = {0: "redraft", 1: "keeper", 2: "dynasty"}
    positions = league.get("roster_positions") or []
    my_rid = None
    for r in rosters:
        if str(r.get("owner_id")) == str(my_user_id):
            my_rid = r.get("roster_id")
    idp = any(p in positions for p in
              ("DL", "LB", "DB", "IDP_FLEX", "DEF_LINE", "LINEBACKER", "DEF_BACK"))
    return {
        "league_id": league["league_id"],
        "platform": "sleeper",
        "name": league.get("name"),
        "type": type_map.get(st.get("type"), "redraft"),
        "idp": idp,
        "scoring": league.get("scoring_settings") or {},
        "roster_positions": positions,
        "roster_size": len([p for p in positions if p != "BN"]) if positions else 0,
        "total_slots": len(positions),
        "taxi_slots": st.get("taxi_slots", 0),
        "reserve_slots": st.get("reserve_slots", 0),
        "owners": {str(u["user_id"]): (u.get("display_name") or u.get("username"))
                   for u in users},
        "my_roster_id": my_rid,
        "my_user_id": str(my_user_id),
    }


def starting_slots(meta: dict) -> list[str]:
    """Startable slots, bench excluded."""
    skip = {"BN", "TAXI", "IR"}
    return [p for p in (meta.get("roster_positions") or []) if p not in skip]


# Which positions may fill which slot.
SLOT_ELIGIBILITY = {
    "QB": {"QB"}, "RB": {"RB"}, "WR": {"WR"}, "TE": {"TE"},
    "FLEX": {"RB", "WR", "TE"},
    "WRRB_FLEX": {"RB", "WR"}, "REC_FLEX": {"WR", "TE"},
    "WRRB_WRT": {"RB", "WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
    "K": set(), "DEF": set(),
}


def slot_allows(slot: str, position: str) -> bool:
    return position in SLOT_ELIGIBILITY.get(slot, set())
