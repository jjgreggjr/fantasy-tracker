"""Live view of a league: the platform's current rosters on top of committed scores.

The pipeline scores every relevant player and commits the result, but it runs
on a schedule. Between runs James moves players, so the committed `is_starter`
drifts from what the platform actually shows — and lineup edits cluster right
before games, in the gaps. For Sleeper one unauthenticated GET closes that gap:

    GET https://api.sleeper.app/v1/league/{id}/rosters

gives every roster's `players`, `starters`, `reserve` and `taxi`. Join that
onto the committed scored pool (all_rosters.csv ∪ available.csv, which
carries E_pts / vor / conf / injury for everyone in the league) and you have
the current lineup and the current waiver wire with the last run's numbers.

ESPN is different: its API is not reachable from the question sandbox (the
egress policy refuses the host) and a private league needs cookies that only
the pipeline holds. So for an ESPN league this module tries one live read,
and when that fails it returns the committed snapshot with its timestamp and
`live: False`, so the caller can say plainly that the lineup may be stale.

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
    """The platform could not be reached or answered with something unusable."""


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


def _sleeper_rosters(league_id: str) -> list[dict]:
    try:
        rosters = sources._get(f"league/{league_id}/rosters")
    except Exception as e:  # network, 4xx/5xx, bad JSON — all the same to a reader
        raise LiveUnavailable(f"could not read live rosters: {e}") from e
    if not isinstance(rosters, list) or not rosters:
        raise LiveUnavailable("Sleeper returned no rosters")
    return rosters


def _espn_rosters(meta: dict, root: Path) -> list[dict]:
    """ESPN's current rosters reshaped into Sleeper's roster dicts, keyed by
    Sleeper id through the player crosswalk, so the rest of the module is
    platform-blind. Raises LiveUnavailable on any failure, including the
    expected one: the sandbox cannot reach ESPN at all."""
    from . import espn
    pf = root / "data" / "players.csv"
    if not pf.exists():
        raise LiveUnavailable("no data/players.csv for the ESPN id crosswalk")
    try:
        data = espn._get(int(meta.get("season") or datetime.now().year),
                         str(meta["league_id"]), ["mRoster"], espn.load_cookies(root))
    except Exception as e:
        raise LiveUnavailable(f"could not read ESPN live: {e}") from e
    px = pd.read_csv(pf, low_memory=False, usecols=["espn_id", "sleeper_id"])
    px["espn_id"] = px.espn_id.map(norm_id)
    px["sleeper_id"] = px.sleeper_id.map(norm_id)
    xw = (px.dropna(subset=["espn_id", "sleeper_id"])
            .drop_duplicates("espn_id").set_index("espn_id").sleeper_id.to_dict())
    out = []
    for t in (data.get("teams") or []):
        players, starters, reserve = [], [], []
        for e in ((t.get("roster") or {}).get("entries") or []):
            pid = norm_id(((e.get("playerPoolEntry") or {}).get("player") or {}).get("id"))
            sid = xw.get(pid)
            if sid is None:
                continue
            slot = espn.SLOT_ID.get(e.get("lineupSlotId"), "")
            players.append(sid)
            if slot == "IR":
                reserve.append(sid)
            elif slot not in espn.NON_STARTING:
                starters.append(sid)
        out.append({"owner_id": str(t.get("id")), "roster_id": t.get("id"),
                    "players": players, "starters": starters,
                    "reserve": reserve, "taxi": []})
    if not out:
        raise LiveUnavailable("ESPN returned no teams")
    return out


def _snapshot_view(slug: str, meta: dict, d: Path) -> dict:
    """The committed lineup, presented as what it is: a snapshot with a time."""
    snap = (pd.read_csv(d / "roster.csv", low_memory=False)
            if (d / "roster.csv").exists() else pd.DataFrame())
    avail = (pd.read_csv(d / "available.csv", low_memory=False)
             if (d / "available.csv").exists() else pd.DataFrame())
    for col in ("is_starter", "is_ir", "is_taxi"):
        if not snap.empty and col not in snap.columns:
            snap[col] = 0
    if not avail.empty:
        avail = avail.assign(is_starter=0, is_ir=0, is_taxi=0)
    st = set()
    if not snap.empty and "sleeper_id" in snap.columns:
        st = set(snap[snap.is_starter == 1].sleeper_id.map(norm_id).dropna())
    # The per-league roster.csv is scored players only (score.compute keeps just
    # QB/RB/WR/TE), so K, DEF and any unmatched player are absent from it. Surface
    # them the way the Sleeper live read does, from my_roster.csv (which does keep
    # them), so an ESPN snapshot names them instead of leaving them invisible.
    unscored = []
    try:
        mr = pd.read_csv(d.parent.parent / "data" / "my_roster.csv",
                         dtype={"sleeper_id": str})
        mr = mr[mr.league_id.astype(str) == str(meta.get("league_id"))]
        if not mr.empty:
            # season first, then the newest week within it — week numbers reset
            # each season and an ESPN league keeps one league_id across seasons
            mr = mr[mr.season == mr.season.max()]
            mr = mr[mr.week == mr.week.max()]
            for _, x in mr[mr.gsis_id.isna()].iterrows():
                nm = x.get("name")
                if isinstance(nm, str) and nm.strip():
                    tag = ", ".join(str(v) for v in (x.get("position"), x.get("team"))
                                    if isinstance(v, str) and v.strip())
                    unscored.append(f"{nm} ({tag})" if tag else nm)
                elif isinstance(x.get("sleeper_id"), str) and x["sleeper_id"].strip():
                    unscored.append(x["sleeper_id"])
    except Exception:
        # Never let this break the snapshot; an empty list is the old behavior.
        unscored = []
    return {
        "slug": slug, "league": meta.get("name"),
        "platform": meta.get("platform", "sleeper"), "live": False,
        "fetched_at": meta.get("roster_fetched_at"),
        "snapshot_at": meta.get("roster_fetched_at"),
        "mine": snap, "available": avail, "unscored": unscored,
        "diff": {"started_since_snapshot": [], "benched_since_snapshot": []},
        "live_starter_ids": st,
    }


def live_league(slug: str, root: Path) -> dict:
    d = root / "leagues" / slug
    if not d.exists():
        raise SystemExit(f"No league '{slug}' in {root / 'leagues'}")
    meta = json.loads((d / "league.json").read_text(encoding="utf-8"))
    league_id = str(meta["league_id"])
    my_uid = str(meta.get("my_user_id") or "")
    my_rid = meta.get("my_roster_id")
    platform = meta.get("platform") or "sleeper"

    if platform == "espn":
        try:
            rosters = _espn_rosters(meta, root)
        except LiveUnavailable:
            # Expected from the sandbox. Not an error: the snapshot IS the answer,
            # as long as it is labelled with its time.
            return _snapshot_view(slug, meta, d)
    else:
        rosters = _sleeper_rosters(league_id)
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
        "slug": slug, "league": meta.get("name"), "platform": platform,
        "live": True, "fetched_at": fetched_at,
        "snapshot_at": meta.get("roster_fetched_at"),
        "mine": mine_live, "available": avail_live,
        "unscored": unscored, "diff": diff,
        "live_starter_ids": live_st,
    }


def _with_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Add the volume behind E_pts, so points are never shown alone.

    E_pts is derived from opportunity; POLICY.md's whole premise is that the
    opportunity predicts the points. So every table carries both: E_opps
    (expected touches/targets, blended projection + recent), snap share, and
    the position-appropriate share of team volume — targets for WR/TE,
    carries for RB. QBs get pass attempts in the share column.
    """
    out = df.copy()
    if "avg_snap_pct" in out.columns:
        out["snap%"] = (pd.to_numeric(out.avg_snap_pct, errors="coerce") * 100).round(0)
    share = pd.Series(float("nan"), index=out.index)
    pos = out.get("position", pd.Series("", index=out.index))
    if "proj_target_share" in out.columns:
        share = share.mask(pos.isin(["WR", "TE"]),
                           pd.to_numeric(out.proj_target_share, errors="coerce") * 100)
    if "proj_carry_share" in out.columns:
        share = share.mask(pos == "RB",
                           pd.to_numeric(out.proj_carry_share, errors="coerce") * 100)
    if "proj_pass_att" in out.columns:
        share = share.mask(pos == "QB", pd.to_numeric(out.proj_pass_att, errors="coerce"))
    out["share"] = share.round(0)
    return out


VOL_COLS = ("name", "position", "team", "injury_status", "E_pts", "E_opps",
            "snap%", "share", "conf", "opponent")


def render(res: dict, top_adds: int = 8) -> str:
    """Plain-text summary a question session can print verbatim.

    Columns: E_pts and E_opps side by side; snap% = recent snap share;
    share = % of team targets (WR/TE) or carries (RB), pass attempts for QB.
    """
    L = []
    a = L.append
    platform = {"espn": "ESPN"}.get(res.get("platform"), "Sleeper")
    if res.get("live", True):
        a(f"Lineup as of {res['fetched_at']} (live from {platform}) — {res['league']}")
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
    else:
        a(f"Lineup as of {res.get('fetched_at') or 'unknown'} (COMMITTED SNAPSHOT — "
          f"{platform} cannot be read live from here; the lineup may have changed "
          f"since) — {res['league']}")
        a("no live diff available: say 'your lineup as of <that time>', never 'your current lineup'")
    if res["unscored"]:
        a(f"on your roster but not scored — expected for K, DEF and, in IDP "
          f"leagues, defensive players: {', '.join(res['unscored'])}")

    m = _with_volume(res["mine"])
    cols = [c for c in VOL_COLS if c in m.columns]
    a("columns: E_pts = expected points · E_opps = expected touches/targets · "
      "snap% = recent snap share · share = % of team targets (WR/TE) or "
      "carries (RB), pass att (QB)")
    if m.empty:
        a("\n(no scored players on this roster)")
        return "\n".join(L)
    st = m[m.is_starter == 1].sort_values("E_pts", ascending=False)
    bn = m[(m.is_starter == 0) & (m.is_ir == 0) & (m.is_taxi == 0)] \
        .sort_values("E_pts", ascending=False)
    a(f"\nCURRENT STARTERS ({len(st)}):")
    a(st[cols].to_string(index=False) if not st.empty else "  (none)")
    a(f"\nBENCH ({len(bn)}), best first:")
    a(bn[cols].head(10).to_string(index=False) if not bn.empty else "  (none)")

    av = _with_volume(res["available"])
    if not av.empty and "E_pts" in av.columns:
        acols = [c for c in list(VOL_COLS) + ["vor"] if c in av.columns]
        a(f"\nTOP FREE AGENTS right now ({len(av)} available):")
        a(av.nlargest(top_adds, "E_pts")[acols].to_string(index=False))
    return "\n".join(L)
