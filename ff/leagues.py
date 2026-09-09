"""Per-league workspaces.

Shared data (stats, projections, depth charts, defense-vs-position) stays in
data/. Anything that depends on ownership, scoring or roster rules lives here,
one folder per league, so a question about one team never drags in another.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import score as score_mod
from . import upside as upside_mod
from . import scoring, sources

log = logging.getLogger(__name__)


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
    return s or "league"


def write_league(root: Path, meta: dict, my_roster: pd.DataFrame,
                 all_rosters: pd.DataFrame, players: pd.DataFrame,
                 pw_all: pd.DataFrame, base: pd.DataFrame, proj: pd.DataFrame,
                 dvp: pd.DataFrame, sched: pd.DataFrame, depth: pd.DataFrame,
                 week: int, season: int, trending: pd.DataFrame,
                 cfg: dict) -> dict:
    """Write one league's folder and return the scored frames for the report."""
    slug = meta["slug"]
    d = root / "leagues" / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "league.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---- points in this league's scoring, all weeks we have stats for ----
    points = pd.DataFrame()
    if pw_all is not None and not pw_all.empty:
        points = pw_all[["season", "week", "gsis_id", "name", "position"]].copy()
        points["pts"] = scoring.score_frame(pw_all, meta.get("scoring", {}),
                                            meta.get("name", slug))
        points.to_csv(d / "player_points.csv", index=False)

    # historical role/ceiling, computed once and merged into every scored frame
    ups = upside_mod.player_upside(pw_all, points)

    scfg = cfg.get("score", {})
    scored_mine = score_mod.compute(my_roster, players, base, proj, dvp, sched,
                                    depth, points, meta, week, season, scfg)
    if not scored_mine.empty and not ups.empty:
        scored_mine = scored_mine.merge(ups, on="gsis_id", how="left")
    if not scored_mine.empty:
        scored_mine.to_csv(d / "roster.csv", index=False)

    if all_rosters is not None and not all_rosters.empty:
        scored_all = score_mod.compute(all_rosters, players, base, proj, dvp,
                                       sched, depth, points, meta, week, season,
                                       scfg)
        if not scored_all.empty and not ups.empty:
            scored_all = scored_all.merge(ups, on="gsis_id", how="left")
        if not scored_all.empty:
            scored_all.to_csv(d / "all_rosters.csv", index=False)
        owned = set(all_rosters.gsis_id.dropna())
    else:
        scored_all, owned = pd.DataFrame(), set()

    # ---- the waiver wire for THIS league ---------------------------------
    pool = players[players.position.isin(["RB", "WR", "TE", "QB"])
                   & ~players.gsis_id.isin(owned)
                   & players.nfl_status.isin(["ACT", "DEV"])].copy()
    avail = score_mod.compute(pool, players, base, proj, dvp, sched, depth,
                              points, meta, week, season, scfg)
    if not avail.empty and not ups.empty:
        avail = avail.merge(ups, on="gsis_id", how="left")
    if not avail.empty:
        avail = avail[avail.E_pts.notna() | avail.E_opps.notna()]
        if not trending.empty:
            hot = set(trending[trending.kind == "add"].gsis_id.dropna())
            avail["trending_add"] = avail.gsis_id.isin(hot).astype(int)
        else:
            avail["trending_add"] = 0
        avail = avail.sort_values(["position", "E_pts"], ascending=[True, False])
        avail.to_csv(d / "available.csv", index=False)

    # ---- transactions ----------------------------------------------------
    if meta.get("platform") == "sleeper":
        rows = []
        names = players.set_index("sleeper_id").name.to_dict()
        for wk in range(1, week + 1):
            for t in sources.sleeper_transactions(meta["league_id"], wk):
                if t.get("status") != "complete":
                    continue
                who = ", ".join(meta.get("owners", {}).get(str(u), str(u))
                                for u in (t.get("roster_ids") or []))
                for pid in (t.get("adds") or {}):
                    rows.append({"season": season, "week": wk, "type": t.get("type"),
                                 "action": "add", "sleeper_id": str(pid),
                                 "name": names.get(str(pid)), "rosters": who})
                for pid in (t.get("drops") or {}):
                    rows.append({"season": season, "week": wk, "type": t.get("type"),
                                 "action": "drop", "sleeper_id": str(pid),
                                 "name": names.get(str(pid)), "rosters": who})
        if rows:
            pd.DataFrame(rows).to_csv(d / "transactions.csv", index=False)

    return {"meta": meta, "dir": d, "roster": scored_mine,
            "available": avail if not avail.empty else pd.DataFrame(),
            "all_rosters": scored_all, "points": points}


def trending_frame(players: pd.DataFrame) -> pd.DataFrame:
    """League-agnostic hot list, mapped onto our ids."""
    from .build import norm_id
    rows = []
    for kind in ("add", "drop"):
        for item in sources.sleeper_trending(kind):
            rows.append({"kind": kind, "sleeper_id": norm_id(item.get("player_id")),
                         "count": item.get("count")})
    if not rows:
        return pd.DataFrame(columns=["kind", "sleeper_id", "count", "gsis_id"])
    df = pd.DataFrame(rows)
    xw = players[["gsis_id", "sleeper_id", "name", "position", "team"]].copy()
    xw["sleeper_id"] = xw.sleeper_id.map(norm_id)
    return df.merge(xw.dropna(subset=["sleeper_id"]).drop_duplicates("sleeper_id"),
                    on="sleeper_id", how="left")
