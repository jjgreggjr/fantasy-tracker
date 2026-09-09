"""Trade finding: who wants your player, and what to ask for.

A trade closes when it solves a structural problem for the other owner, not
when a value chart says it is fair. So this module looks for structure first:

  * handcuff  — you hold the direct backup to a starter they own
  * need      — they are thin at a position where you have surplus
  * posture   — an old, deep roster is trying to win now and will pay for
                production; a young one is rebuilding and will pay for age

Then it picks the ask from THEIR BENCH, because an owner will trade a bench
player far more readily than a starter, and it says plainly when you are
asking for more than you are giving.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .build import name_key
from . import policy

STARTABLE = 8.0          # E_pts above which a player is a real starter
CONTEND_AGE = 26.5       # avg age of startable players: above this = win-now


def owner_profiles(all_rosters: pd.DataFrame) -> pd.DataFrame:
    """One row per owner: how good, how old, and where they are thin."""
    if all_rosters is None or all_rosters.empty:
        return pd.DataFrame()
    a = all_rosters.copy()
    a["startable"] = (a.E_pts >= STARTABLE).astype(int)
    rows = []
    for owner, g in a.groupby("owner_name"):
        st = g[g.startable == 1]
        by_pos = st.groupby("position").size().to_dict()
        thin = [p for p in ("RB", "WR", "TE", "QB") if by_pos.get(p, 0) <= 2]
        avg_age = st.age.mean() if not st.empty else np.nan
        rows.append({
            "owner_name": owner,
            "players": len(g),
            "startable": len(st),
            "avg_age_startable": round(avg_age, 1) if pd.notna(avg_age) else np.nan,
            "posture": ("win-now" if pd.notna(avg_age) and avg_age >= CONTEND_AGE
                        else "rebuilding" if pd.notna(avg_age) else "unknown"),
            "thin_at": ",".join(thin),
            **{f"n_{p}": by_pos.get(p, 0) for p in ("QB", "RB", "WR", "TE")},
        })
    return pd.DataFrame(rows).sort_values("startable", ascending=False)


def _interest(player: pd.Series, g: pd.DataFrame, prof: pd.Series,
              depth: pd.DataFrame) -> list[str]:
    """Why this owner specifically would want this player."""
    reasons = []
    dr = player.get("depth_rank")
    if pd.notna(dr) and dr >= 2 and depth is not None and not depth.empty:
        ahead = depth[(depth.team == player.team)
                      & (depth.position == player.position)
                      & (depth["rank"] == dr - 1)]
        if not ahead.empty:
            starter = ahead.iloc[0]["name"]
            if (g["name"].map(name_key) == name_key(starter)).any():
                reasons.append(f"owns {starter}; your player is the direct backup")
    if player.position in str(prof.get("thin_at", "")).split(","):
        reasons.append(f"only {prof.get('n_' + player.position, 0)} startable "
                       f"{player.position}s")
    if prof.get("posture") == "win-now" and pd.notna(player.get("E_pts")):
        old = g[(g.position == player.position) & (g.age >= 28)
                & (g.E_pts >= STARTABLE)]
        if len(old) >= 2:
            reasons.append(f"{len(old)} of their startable {player.position}s "
                           "are 28+, so the room ages out soon")
    return reasons


def shop(player_name: str, roster: pd.DataFrame, all_rosters: pd.DataFrame,
         available: pd.DataFrame, depth: pd.DataFrame, meta: dict,
         my_owner: str, max_owners: int = 3) -> tuple[pd.Series | None, list[dict]]:
    """Who to approach about one of your players, and what to ask for."""
    kv_mine = policy.keep_value(roster, available, meta)
    hit = kv_mine[kv_mine["name"].str.contains(player_name, case=False, na=False)]
    if hit.empty:
        return None, []
    p = hit.iloc[0]

    others = all_rosters[all_rosters.owner_name != my_owner].copy()
    if others.empty:
        return p, []
    kv_all = policy.keep_value(others, available, meta)
    profiles = owner_profiles(all_rosters).set_index("owner_name")

    deals = []
    for owner, g in kv_all.groupby("owner_name"):
        if owner not in profiles.index:
            continue
        prof = profiles.loc[owner]
        reasons = _interest(p, g, prof, depth)
        if not reasons:
            continue
        # ask from their BENCH — far easier to pry loose than a starter
        bench = g[(g.is_starter == 0) & (g.E_pts.notna())].copy()
        if meta.get("type") in ("dynasty", "keeper"):
            bench = bench[(bench.age <= 25) & (bench.has_path == 1)]
        bench = bench.sort_values("keep_value", ascending=False)
        asks = bench.head(3)
        if asks.empty:
            continue
        deals.append({
            "owner": owner, "posture": prof.get("posture"),
            "avg_age": prof.get("avg_age_startable"),
            "reasons": reasons, "asks": asks, "profile": prof,
        })
    deals.sort(key=lambda d: len(d["reasons"]), reverse=True)
    return p, deals[:max_owners]


def pitch(p: pd.Series, deal: dict, ask: pd.Series, meta: dict) -> str:
    """A message you could actually send, built only from facts we verified."""
    lead = deal["reasons"][0]
    body = []
    if "direct backup" in lead:
        starter = lead.split("owns ")[1].split(";")[0]
        body.append(f"You've got {starter} starting. {p['name']} is the "
                    f"{p.position}{int(p.depth_rank)} on {p.team} — if "
                    f"{starter} misses time, he's the one who takes the work.")
    elif "only" in lead:
        body.append(f"You're carrying {deal['profile'].get('n_' + p.position, 0)} "
                    f"startable {p.position}s. {p['name']} gives you another one.")
    else:
        body.append(f"Your {p.position} room is aging — {p['name']} helps you now.")

    body.append(f"I'd take {ask['name']} back. He's on your bench, so you're "
                f"not giving up a starter.")
    if pd.notna(ask.get("age")):
        body.append(f"He's {ask.age:.0f} and I'm building for later; "
                    f"you're closer to winning now.")
    return " ".join(body)


def survey(roster: pd.DataFrame, all_rosters: pd.DataFrame,
           available: pd.DataFrame, depth: pd.DataFrame, meta: dict,
           my_owner: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Your sell candidates, and where you are thin."""
    kv = policy.keep_value(roster, available, meta)
    dynasty = meta.get("type") in ("dynasty", "keeper")
    sell = kv[(kv.E_pts.notna())]
    if dynasty:
        # aging players still producing are the classic sell-high: research
        # puts the RB sell window at 25-26 and WR decline at 29
        cliff = {"RB": 26, "WR": 28, "TE": 29, "QB": 33}
        sell = sell[[a >= cliff.get(pos, 99)
                     for pos, a in zip(sell.position, sell.age.fillna(0))]]
    sell = sell.sort_values("E_pts", ascending=False).head(10)

    prof = owner_profiles(all_rosters)
    mine = prof[prof.owner_name == my_owner]
    return sell, mine
