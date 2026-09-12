"""Live view of a league: Sleeper's current rosters on top of committed scores.

The pipeline scores every relevant player and commits the result, but it runs
on a schedule. Between runs James moves players, so the committed `is_starter`
drifts from what Sleeper actually shows — and lineup edits cluster right
before games, in the gaps. One unauthenticated GET closes that gap:

    GET https://api.sleeper.app/v1/league/{id}/rosters

gives every roster's `players`, `starters`, `reserve` and `taxi`. Join that
onto the committed scored pool (all_rosters.csv ∪ available.csv, which
carries E_pts / vor / conf / injury for everyone in the league) and you have
the current lineup and the current waiver wire with the last run's numbers.

What this does NOT refresh: projections, injury designations, depth charts.
Those come from the full pipeline. This answers "what is he starting right
now", not "what changed in the NFL since the last run".

Read-only. Never writes to the repo.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import sources
from .build import norm_id


class LiveUnavailable(RuntimeError):
    """Sleeper could not be reached or answered with something unusable."""


def _pool(d: Path) -> pd.DataFrame:
    """Every scored player in this league, keyed by Sleeper id."""
    frames = [pd.read_csv(d / n, low_memory=False)
              for n in ("all_rosters.csv", "available.csv") if (d / n).exists()]
    if not frames:
        return pd.DataFrame()
    pool = pd.concat(frames, ignore_index=True)
    pool["sleeper_id"] = pool.sleeper_id.map(norm_id)
    return (pool.dropna(subset=["sleeper_id"])
                .drop_duplicates("sleeper_id")
                .set_index("sleeper_id", drop=False))


def _ids(xs) -> set:
    """Sleeper pads empty lineup slots with "0"; drop those."""
    return {norm_id(x) for x in (xs or []) if x not in (None, "0", 0)}


def live_league(slug: str, root: Path) -> dict:
    d = root / "leagues" / slug
    if not d.exists():
        raise SystemExit(f"No league '{slug}' in {root / 'leagues'}")
    meta = json.loads((d / "league.json").read_text(encoding="utf-8"))
    league_id = str(meta["league_id"])
    my_uid = str(meta.get("my_user_id") or "")
    my_rid = meta.get("my_roster_id")

    try:
        rosters = sources._get(f"league/{league_id}/rosters")
    except Exception as e:  # network, 4xx/5xx, bad JSON — all the same to a reader
        raise LiveUnavailable(f"could not read live rosters for {slug}: {e}") from e
    if not isinstance(rosters, list) or not rosters:
        raise LiveUnavailable(f"Sleeper returned no rosters for {slug}")
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    pool = _pool(d)
    if pool.empty:
        raise SystemExit(f"{slug} has no scored players yet — run the pipeline first")

    # --- live ownership and lineup flags for every player in the league ----
    owner_of: dict[str, tuple[str, object]] = {}
    starters_of: dict[str, set] = {}
    ir_of: dict[str, set] = {}
    taxi_of: dict[str, set] = {}
    mine = None
    for r in rosters:
        oid = str(r.get("owner_id"))
        rid = r.get("roster_id")
        players = _ids(r.get("players"))
        for pid in players:
            owner_of[pid] = (oid, rid)
        starters_of[oid] = _ids(r.get("starters"))
        ir_of[oid] = _ids(r.get("reserve"))
        taxi_of[oid] = _ids(r.get("taxi"))
        if (my_uid and oid == my_uid) or (my_rid is not None and rid == my_rid):
            mine = r
    if mine is None:
        raise LiveUnavailable(
            f"none of the {len(rosters)} live rosters belongs to user "
            f"{my_uid or '?'} / roster {my_rid} — check league.json")
    my_oid = str(mine.get("owner_id"))
    my_players = _ids(mine.get("players"))

    def _flagged(ids: set) -> pd.DataFrame:
        sub = pool.loc[[i for i in ids if i in pool.index]].copy()
        for col, src in (("is_starter", starters_of), ("is_ir", ir_of),
                         ("is_taxi", taxi_of)):
            sub[col] = [int(i in src.get(owner_of.get(i, ("", None))[0], set()))
                        for i in sub.sleeper_id]
        return sub.reset_index(drop=True)

    mine_live = _flagged(my_players)
    rostered_anywhere = set(owner_of)
    avail_live = (pool[~pool.sleeper_id.isin(rostered_anywhere)]
                  .assign(is_starter=0, is_ir=0, is_taxi=0)
                  .reset_index(drop=True))
    unscored = sorted(my_players - set(pool.index))

    # --- what the committed snapshot thought, for the diff ------------------
    snap = (pd.read_csv(d / "roster.csv", low_memory=False)
            if (d / "roster.csv").exists() else pd.DataFrame())
    snap_st: set = set()
    if not snap.empty and "is_starter" in snap.columns:
        snap_st = set(snap[snap.is_starter == 1].sleeper_id.map(norm_id).dropna())
    live_st = {i for i in starters_of.get(my_oid, set()) if i in pool.index}
    name = dict(zip(pool.sleeper_id, pool.name))
    diff = {
        "started_since_snapshot": sorted(name.get(i, i) for i in live_st - snap_st),
        "benched_since_snapshot": sorted(name.get(i, i) for i in snap_st - live_st),
    }
    return {
        "slug": slug, "league": meta.get("name"), "fetched_at": fetched_at,
        "snapshot_at": meta.get("roster_fetched_at"),
        "mine": mine_live, "available": avail_live,
        "unscored": unscored, "diff": diff,
        "live_starter_ids": live_st,
    }


def render(res: dict, top_adds: int = 8) -> str:
    """Plain-text summary a question session can print verbatim."""
    L = []
    a = L.append
    a(f"Lineup as of {res['fetched_at']} (live from Sleeper) — {res['league']}")
    if res.get("snapshot_at"):
        a(f"committed snapshot was {res['snapshot_at']}")
    d = res["diff"]
    if d["started_since_snapshot"] or d["benched_since_snapshot"]:
        a("changes since the snapshot:")
        if d["started_since_snapshot"]:
            a("  started: " + ", ".join(d["started_since_snapshot"]))
        if d["benched_since_snapshot"]:
            a("  benched: " + ", ".join(d["benched_since_snapshot"]))
    else:
        a("no lineup changes since the snapshot")
    if res["unscored"]:
        a(f"on your roster but not scored — expected for K, DEF and, in IDP "
          f"leagues, defensive players: {', '.join(res['unscored'])}")

    m = res["mine"]
    cols = [c for c in ("name", "position", "team", "injury_status", "E_pts",
                        "conf", "opponent") if c in m.columns]
    st = m[m.is_starter == 1].sort_values("E_pts", ascending=False)
    bn = m[(m.is_starter == 0) & (m.is_ir == 0) & (m.is_taxi == 0)] \
        .sort_values("E_pts", ascending=False)
    a(f"\nCURRENT STARTERS ({len(st)}):")
    a(st[cols].to_string(index=False) if not st.empty else "  (none)")
    a(f"\nBENCH ({len(bn)}), best first:")
    a(bn[cols].head(10).to_string(index=False) if not bn.empty else "  (none)")

    av = res["available"]
    if not av.empty and "E_pts" in av.columns:
        acols = [c for c in cols + ["vor"] if c in av.columns]
        a(f"\nTOP FREE AGENTS right now ({len(av)} available):")
        a(av.nlargest(top_adds, "E_pts")[acols].to_string(index=False))
    return "\n".join(L)
