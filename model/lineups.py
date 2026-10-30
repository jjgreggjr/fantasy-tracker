"""Recorded optimal lineups, frozen slot by slot at kickoff, and what each would have scored.

    python -m model.lineups            # record, then backfill, then score
    python -m model.lineups record     # this week's four optimal lineups per league -> data/lineups_optimal.csv
    python -m model.lineups backfill   # completed weeks nobody recorded live: rebuilt from frozen pre-kickoff predictions
    python -m model.lineups score      # every complete week -> data/lineup_eval.csv (the report section is rendered by model.scoreboard)

record: per league, four lineups from the league's committed roster.csv (after `modelcols.attach`), via `ff.calls.three_lineups`:
  e_pts   the exact optimum by E_pts          model   the exact optimum by E_pts_model          call   the recommendation
  sleeper the exact optimum by Sleeper's projection in the league's scoring (`sleeper_proj_<slug>` of data/model_pts.csv, joined on
          gsis_id for the week); K / IDP / anyone without a projection is filled from E_pts
Each pipeline run re-records the week, but a slot locks at its own player's kickoff, as on Sleeper: a row whose kickoff_utc <= now is
kept exactly as it was recorded; the other slots are re-optimized from a pool that no longer holds the locked players or anyone whose
own game has already kicked off (nobody can be added after lock). So a Sunday run only ever changes slots whose game is still ahead.
The first record of a week (no rows yet) locks nothing it did not itself choose: rows are simply stamped locked when their kickoff has
passed. Kickoff per player: the frozen `kickoff_utc` of data/model_pts.csv (by gsis_id, then by team), and for players with no model
row (K, IDP, anyone unprojected) data/schedule.csv `gameday` at 17:00 UTC, an approximation of the early window; a player with
neither is never locked (None is not a kickoff).

score: a (season, week, league) is complete when data/lineups_played.csv holds rows for that league's id and week (the scoreboard's
own definition). Each recorded lineup is joined to those rows by sleeper_id (Sleeper) or espn_id (ESPN), falling back to the other
ids, and takes the platform's `points`; no played row scores 0 and counts in n_missing. basis `actual` is James's own started lineup
(started == 1 on his roster_id, as `ff.ask played` reads it). offense_points counts QB/RB/WR/TE only, the like-for-like number for the
IDP league (and for `actual`, which also holds K / DEF / IDP starters the recorded lineups do not).

Also per complete league-week: basis `perfect`, the hindsight-best lineup from every player on his roster that week in
lineups_played.csv (started or not; `optimize_lineup` on the platform `points` over `scoreable_slots`), and `efficiency` =
points / perfect points on every row (`offense_efficiency` likewise, the perfect offense lineup built from QB/RB/WR/TE only).

backfill: for every complete league-week with no live-recorded lineup, rebuild the four lineups from (a) his pre-game roster in
data/league_rosters.csv (no IR / taxi, no blank rows) and (b) the predictions data/model_pts.csv froze at serve time, using only rows
with `frozen_before_kickoff` true (the column of data/model_pts.csv; rows from before it existed fall back to served_at < kickoff_utc). A player with no frozen row has no prediction and cannot be picked; K / DEF / IDP have no model row, so
those slots stay empty in backfilled lineups (offense_efficiency is the like-for-like number). `call` is built with the status rule
off (no historical practice / play_prob). Rows carry source = backfill (live rows: live); a live-recorded league-week is never touched.

Never breaks the pipeline: any failure is a WARN row (check `model.lineups`) and nothing is written.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

try:                                                    # pandas missing must be a WARN row from main(), not an import crash
    import pandas as pd
except Exception:                                       # pragma: no cover
    pd = None

ROOT = Path(__file__).resolve().parents[1]
CHECK = "model.lineups"
OPT_FILE, EVAL_FILE = "lineups_optimal.csv", "lineup_eval.csv"
BASES = ("e_pts", "model", "sleeper", "call")
OFFENSE = ("QB", "RB", "WR", "TE")
IDS = ("gsis_id", "sleeper_id", "espn_id")
OPT_COLS = ["season", "week", "league", "basis", "slot", *IDS, "name", "position", "team", "E_pts", "E_pts_model",
            "kickoff_utc", "recorded_at", "locked", "e_pick", "m_pick", "reason", "source"]   # e_pick..reason are the call's own notes
EVAL_COLS = ["season", "week", "league", "basis", "points", "offense_points", "n_slots", "n_missing", "efficiency", "offense_efficiency",
             "source"]
PART = ["season", "week", "league", "basis"]
MIN_COVERAGE = 0.8      # backfill needs frozen (pre-kickoff) rows for at least this share of his offensive players that have ANY row that
                        # week: a serve right at kickoff can overwrite a whole window's frozen rows, and a lineup built from the rest
                        # says nothing about the picker. (Players with no row at all are just unprojected, not a coverage problem.)
SCHEDULE_HOUR = "17:00:00+00:00"


# --------------------------------------------------------------------------- small helpers
def _read(path: Path, **kw):
    return pd.read_csv(path, low_memory=False, **kw) if Path(path).exists() else None


def _read_str(path: Path):
    """A csv with every id column as text (a float sleeper_id would turn '10236' into '10236.0')."""
    return _read(path, dtype={c: str for c in IDS + ("league_id", "reason")}) if Path(path).exists() else None


def _nid(x):
    """An id as clean text, or None for a blank / NaN."""
    if x is None or x != x:
        return None
    s = str(x).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s or None


def _leagues(root: Path):
    for p in sorted((root / "leagues").glob("*/league.json")):
        meta = json.loads(p.read_text(encoding="utf-8"))
        yield meta.get("slug") or p.parent.name, meta, p.parent


def _ts(x):
    return pd.to_datetime(x, utc=True, errors="coerce")


# --------------------------------------------------------------------------- record
def kickoffs(root: Path, roster, season: int, week: int):
    """A Series (index = roster index) of each player's kickoff as a UTC Timestamp, NaT where unknown."""
    mp = _read(root / "data" / "model_pts.csv", dtype={"gsis_id": str})
    by_gsis, by_team = {}, {}
    if mp is not None and len(mp) and "kickoff_utc" in mp.columns:
        w = mp[(mp["season"] == season) & (mp["week"] == week)].copy()
        w["_k"] = _ts(w["kickoff_utc"])
        by_gsis = w.dropna(subset=["gsis_id", "_k"]).drop_duplicates("gsis_id").set_index("gsis_id")["_k"].to_dict()
        by_team = w.dropna(subset=["team", "_k"]).groupby("team")["_k"].min().to_dict()
    sched = _read(root / "data" / "schedule.csv")
    by_sched = {}
    if sched is not None and len(sched):
        s = sched[(sched["season"] == season) & (sched["week"] == week)].dropna(subset=["team", "gameday"])
        by_sched = {t: _ts(f"{g} {SCHEDULE_HOUR}") for t, g in zip(s["team"], s["gameday"])}     # no model row: 17:00 UTC on gameday
    out = []
    for _, r in roster.iterrows():
        g, t = _nid(r.get("gsis_id")), r.get("team")
        k = by_gsis.get(g) if g else None
        if k is None:
            k = by_team.get(t)
        if k is None:
            k = by_sched.get(t)
        out.append(k if k is not None else pd.NaT)
    return pd.Series(out, index=roster.index, dtype="datetime64[ns, UTC]")


def _key(df):
    """One identity per player row: gsis_id, else sleeper_id, else name."""
    def one(r):
        for c in ("gsis_id", "sleeper_id", "espn_id", "name"):
            v = _nid(r.get(c)) if c in df.columns else None
            if v:
                return f"{c}:{v}"
        return None
    return df.apply(one, axis=1) if len(df) else pd.Series([], dtype=object)


def attach_sleeper(roster, slug: str, root: Path, season: int, week: int):
    """`roster` with `sleeper_pts`: Sleeper's projection in the league's scoring for that week (data/model_pts.csv, joined on gsis_id),
    NaN where there is none."""
    mp = _read(root / "data" / "model_pts.csv", dtype={"gsis_id": str})
    col = f"sleeper_proj_{slug}"
    out = roster.copy()
    out["sleeper_pts"] = float("nan")
    if mp is None or mp.empty or col not in mp.columns or "gsis_id" not in out.columns:
        return out
    w = mp[(mp["season"] == season) & (mp["week"] == week)].dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")
    out["sleeper_pts"] = out["gsis_id"].map(w.set_index("gsis_id")[col]).astype("float64")
    return out


def sleeper_lineup(roster, slots: list[str], idp_on: bool):
    """The exact optimum by Sleeper's projection (`sleeper_pts`), a player without one filled from E_pts. Same frame as one
    `three_lineups` entry."""
    from ff import ask, calls
    r = roster.drop_duplicates("name").reset_index(drop=True)
    if "E_pts" not in r.columns:
        r["E_pts"] = float("nan")
    sp = pd.to_numeric(r["sleeper_pts"], errors="coerce") if "sleeper_pts" in r.columns else pd.Series(float("nan"), index=r.index)
    r["sleeper_pts"] = sp.fillna(pd.to_numeric(r["E_pts"], errors="coerce"))
    picks = ask.optimize_lineup(r, slots, "sleeper_pts", idp_on)
    return calls._frame(picks, slots, r.set_index("name", drop=False), [c for c in calls.ID_COLS if c in r.columns])


def build_lineup(basis: str, pool, slots: list[str], idp_on: bool, three_lineups, **kw):
    """One basis's lineup frame (one row per slot) from `pool`."""
    if basis == "sleeper":
        return sleeper_lineup(pool, slots, idp_on)
    return three_lineups(pool, slots, idp_on, **kw)[basis]


def record_league(slug: str, meta: dict, ldir: Path, root: Path, now, existing, three_lineups=None):
    """The three lineups of one league as a DataFrame (OPT_COLS), honoring the rows already recorded for this week in `existing`
    (the whole data/lineups_optimal.csv, or None). Returns None when the league has nothing to record."""
    from ff import ask, modelcols
    if three_lineups is None:
        from ff.calls import three_lineups
    rp = ldir / "roster.csv"
    if not rp.exists():
        return None
    roster = pd.read_csv(rp, low_memory=False, dtype={c: str for c in IDS})
    if roster.empty or not {"season", "week"} <= set(roster.columns):
        return None
    season, week = int(roster["season"].dropna().iloc[0]), int(roster["week"].dropna().iloc[0])
    roster = attach_sleeper(modelcols.attach(roster, slug, root), slug, root, season, week)
    slots, idp_on = ask.scoreable_slots(meta)
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    roster = roster.reset_index(drop=True)
    kick = kickoffs(root, roster, season, week)
    gone = kick.notna() & (kick <= now)                         # his game has kicked off: cannot be added now
    rkey = _key(roster)
    kick_by_key = dict(zip(rkey, kick))
    out = []
    for basis in BASES:
        prev = pd.DataFrame()
        if existing is not None and len(existing):
            m = (existing["season"] == season) & (existing["week"] == week) & (existing["league"] == slug) & (existing["basis"] == basis)
            if "source" in existing.columns:                    # a backfilled week is not a live lock: live rows replace it
                m &= existing["source"].fillna("live") != "backfill"
            prev = existing[m]
        locked = prev[_ts(prev["kickoff_utc"]).le(now).fillna(False)] if len(prev) else prev
        # a slot the league no longer has (settings changed) cannot stay locked
        left, keep = list(slots), []
        for _, r in locked.iterrows():
            if r["slot"] in left:
                left.remove(r["slot"])
                keep.append(r)
        if prev.empty:                                          # nothing recorded yet: the whole roster is the pool, rows are stamped
            pool = roster
        else:
            pool = roster[~gone & ~rkey.isin(set(_key(pd.DataFrame(keep))))] if keep else roster[~gone]
        fresh = pd.DataFrame()
        if left:
            fresh = build_lineup(basis, pool.reset_index(drop=True), left, idp_on, three_lineups).copy()
            fresh["kickoff_utc_ts"] = [kick_by_key.get(k, pd.NaT) for k in _key(fresh)] if len(fresh) else []
        rows, pend = [], {}
        for _, r in fresh.iterrows():
            pend.setdefault(r["slot"], []).append(r)
        kept = {}
        for r in keep:
            kept.setdefault(r["slot"], []).append(r)
        for s in slots:                                          # slots order; a kept (locked) row wins its slot first
            if kept.get(s):
                r = kept[s].pop(0).to_dict()
                r["locked"] = 1
                rows.append(r)
            elif pend.get(s):
                r = pend[s].pop(0).to_dict()
                k = r.pop("kickoff_utc_ts", pd.NaT)
                r["kickoff_utc"] = k.isoformat() if k == k else ""
                r["recorded_at"] = now_iso
                r["locked"] = int(bool(k == k and k <= now))
                rows.append(r)
        if rows:
            df = pd.DataFrame(rows)
            df["season"], df["week"], df["league"], df["basis"] = season, week, slug, basis
            out.append(df)
    if not out:
        return None
    res = pd.concat(out, ignore_index=True)
    res["source"] = "live"
    for c in OPT_COLS:
        if c not in res.columns:
            res[c] = ""
    res = res[OPT_COLS]
    for c in IDS:
        res[c] = res[c].map(_nid)
    return res


def _replace(path: Path, new, partition: list[str]) -> int:
    """`ff.build.replace_partition` semantics (every existing row of a partition present in `new` is replaced, nothing duplicates),
    kept here only so ids stay text and rows keep their slot order instead of being re-sorted."""
    if new.empty:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = _read_str(path)
        pn = set(map(tuple, new[partition].astype(str).to_numpy()))
        keep = [tuple(r) not in pn for r in old[partition].astype(str).to_numpy()]
        new = pd.concat([old[keep], new], ignore_index=True)
    new = new.sort_values(["season", "week", "league", "basis"], kind="stable").reset_index(drop=True)
    new.to_csv(path, index=False)
    return len(new)


def record(root: Path = ROOT, now=None, three_lineups=None, write: bool = True):
    """Record every league. Returns the new rows (all leagues)."""
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    path = root / "data" / OPT_FILE
    existing = _read_str(path)
    parts = []
    for slug, meta, ldir in _leagues(root):
        r = record_league(slug, meta, ldir, root, now, existing, three_lineups)
        if r is not None and len(r):
            parts.append(r)
    if not parts:
        return pd.DataFrame(columns=OPT_COLS)
    new = pd.concat(parts, ignore_index=True)
    if write:
        _replace(path, new, ["season", "week", "league", "basis"])
    return new


# --------------------------------------------------------------------------- backfill
def _complete_weeks(lp, lid: str):
    d = lp[lp["league_id"].astype(str) == lid]
    return sorted({(int(a), int(b)) for a, b in zip(d["season"], d["week"])})


def _has_live(existing, slug: str, season: int, week: int) -> bool:
    if existing is None or not len(existing):
        return False
    d = existing[(existing["season"] == season) & (existing["week"] == week) & (existing["league"] == slug)]
    if d.empty:
        return False
    src = d["source"].fillna("live") if "source" in d.columns else pd.Series("live", index=d.index)
    return bool(src.ne("backfill").any())


def backfill_league(slug: str, meta: dict, root: Path, lp, snap, mp, existing, now, three_lineups=None, skipped=None):
    """Backfilled rows (OPT_COLS frames) for every complete week of one league that has no live-recorded lineup."""
    from ff import ask, modelcols
    if three_lineups is None:
        from ff.calls import three_lineups
    skipped = [] if skipped is None else skipped
    lid = str(meta["league_id"])
    rid = _my_roster_id(meta, root, lid)
    if rid is None:
        return []
    slots, idp_on = ask.scoreable_slots(meta)
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = []
    for season, week in _complete_weeks(lp, lid):
        if _has_live(existing, slug, season, week):
            continue                                                  # recorded live: never overwritten
        team = snap[(snap["league_id"].astype(str) == lid) & (snap["season"] == season) & (snap["week"] == week)
                    & (pd.to_numeric(snap["roster_id"], errors="coerce") == rid)]
        team = team[team["name"].notna() & team["position"].notna()]
        for c in ("is_ir", "is_taxi"):
            if c in team.columns:
                team = team[pd.to_numeric(team[c], errors="coerce").fillna(0) == 0]
        wk = mp[(mp["season"] == season) & (mp["week"] == week)].copy()
        if team.empty or wk.empty:
            continue
        any_row = set(wk["gsis_id"].dropna())
        from model.serve import frozen_mask
        wk = wk[frozen_mask(wk)].dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")   # frozen rows only
        roster = team[[c for c in ("gsis_id", "sleeper_id", "espn_id", "name", "position", "team") if c in team.columns]].copy()
        for c in IDS:
            if c not in roster.columns:
                roster[c] = None
            roster[c] = roster[c].map(_nid)
        roster = roster.drop_duplicates(subset=["name", "position", "team"]).reset_index(drop=True)
        by = wk.set_index("gsis_id")

        def num(col):
            if col not in by.columns:
                return pd.Series(float("nan"), index=roster.index)
            return roster["gsis_id"].map(pd.to_numeric(by[col], errors="coerce")).astype("float64")

        roster["E_pts"] = num(f"e_pts_{slug}")
        roster["sleeper_pts"] = num(f"sleeper_proj_{slug}")
        have = roster["position"].isin(OFFENSE) & roster["gsis_id"].isin(any_row)
        cov = float((roster.loc[have, "gsis_id"].isin(set(wk["gsis_id"]))).mean()) if have.any() else 1.0
        if cov < MIN_COVERAGE:
            skipped.append((slug, season, week, round(cov, 2)))
            continue
        roster["season"], roster["week"] = season, week
        mc = modelcols.league_columns(wk, slug)
        if len(mc):
            roster = roster.merge(mc, on="gsis_id", how="left")
        else:
            for c in modelcols.MODEL_COLS:
                roster[c] = float("nan")
        roster = roster.reset_index(drop=True)
        kick = kickoffs(root, roster, season, week)
        kick_by_key = dict(zip(_key(roster), kick))
        for basis in BASES:
            kw = {"status": pd.DataFrame()} if basis == "call" else {}       # no historical practice / play_prob: the status rule is off
            fr = build_lineup(basis, roster, slots, idp_on, three_lineups, **kw).copy()
            fr = fr[fr["name"].astype(str).ne("")].reset_index(drop=True)
            if fr.empty:
                continue
            ks = [kick_by_key.get(k, pd.NaT) for k in _key(fr)]
            fr["kickoff_utc"] = [k.isoformat() if k == k else "" for k in ks]
            fr["recorded_at"], fr["locked"], fr["source"] = now_iso, 1, "backfill"
            fr["season"], fr["week"], fr["league"], fr["basis"] = season, week, slug, basis
            for c in OPT_COLS:
                if c not in fr.columns:
                    fr[c] = ""
            fr = fr[OPT_COLS].copy()
            for c in IDS:
                fr[c] = fr[c].map(_nid)
            out.append(fr)
    return out


def backfill(root: Path = ROOT, now=None, three_lineups=None, write: bool = True, skipped=None):
    """Build the four lineups for completed league-weeks nobody recorded live. Returns the new rows; league-weeks left out for thin
    frozen coverage (< MIN_COVERAGE of his offensive players) are appended to `skipped` as (league, season, week, coverage)."""
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    path = root / "data" / OPT_FILE
    existing = _read_str(path)
    lp = _read(root / "data" / "lineups_played.csv", dtype={c: str for c in IDS + ("league_id",)})
    snap = _read(root / "data" / "league_rosters.csv", dtype={c: str for c in IDS + ("league_id",)})
    mp = _read(root / "data" / "model_pts.csv", dtype={"gsis_id": str})
    parts = []
    if lp is not None and len(lp) and snap is not None and mp is not None and len(mp) and "served_at" in mp.columns:
        for slug, meta, _ in _leagues(root):
            parts += backfill_league(slug, meta, root, lp, snap, mp, existing, now, three_lineups, skipped)
    if not parts:
        return pd.DataFrame(columns=OPT_COLS)
    new = pd.concat(parts, ignore_index=True)
    if write:
        _replace(path, new, PART)
    return new


# --------------------------------------------------------------------------- score
def _my_roster_id(meta: dict, root: Path, lid: str):
    rid = meta.get("my_roster_id")
    if rid is None:
        mine = _read(root / "data" / "my_roster.csv", dtype={"league_id": str})
        if mine is not None and len(mine):
            mine = mine[mine["league_id"].astype(str) == lid]
            rid = mine["roster_id"].iloc[-1] if len(mine) else None
    return None if rid is None else int(rid)


def _lookup(played, platform: str):
    """{id kind: {id: (points, position)}} over one league-week's played rows."""
    order = ("espn_id", "sleeper_id", "gsis_id") if platform == "espn" else ("sleeper_id", "espn_id", "gsis_id")
    look = {}
    for c in order:
        if c in played.columns:
            d = {}
            for i, p, pos in zip(played[c].map(_nid), pd.to_numeric(played["points"], errors="coerce").fillna(0.0), played["position"]):
                if i:
                    d[i] = (float(p), pos)
            look[c] = d
    return order, look


def perfect_lineup(team, slots: list[str], idp_on: bool, offense_only: bool = False):
    """(total points, slots filled) of the hindsight-best lineup from `team` (his rows of lineups_played: every player on the
    roster, started or not) by the platform `points`, over `slots`. `offense_only` builds it from QB/RB/WR/TE alone."""
    from ff import ask
    t = team.copy()
    t["points"] = pd.to_numeric(t["points"], errors="coerce").fillna(0.0)
    if offense_only:
        t = t[t["position"].isin(OFFENSE)]
    if t.empty:
        return 0.0, 0
    pool = pd.DataFrame({"name": [f"{n}#{i}" for i, n in enumerate(t["name"].astype(str) if "name" in t.columns else [""] * len(t))], "position": t["position"].to_numpy(),
                         "team": "", "E_pts": t["points"].to_numpy(), "pts": t["points"].to_numpy()})
    picks = ask.optimize_lineup(pool, slots, "pts", idp_on)
    return round(float(sum(p["E_pts"] for p in picks)), 2), len(picks)


def _ratio(a, b):
    return round(a / b, 4) if b and b > 0 else float("nan")


def score(root: Path = ROOT, write: bool = True):
    """Rows of lineup_eval.csv for every complete (season, week, league) in lineups_optimal.csv."""
    from ff import ask
    opt = _read_str(root / "data" / OPT_FILE)
    lp = _read(root / "data" / "lineups_played.csv", dtype={c: str for c in IDS + ("league_id",)})
    empty = pd.DataFrame(columns=EVAL_COLS)
    if opt is None or opt.empty or lp is None or lp.empty:
        return empty
    if "source" not in opt.columns:
        opt["source"] = "live"
    rows = []
    for slug, meta, _ in _leagues(root):
        lid, platform = str(meta["league_id"]), str(meta.get("platform") or "sleeper").lower()
        rid = _my_roster_id(meta, root, lid)
        slots, idp_on = ask.scoreable_slots(meta)
        mine = opt[opt["league"] == slug]
        for (season, week), grp in mine.groupby(["season", "week"]):
            played = lp[(lp["league_id"].astype(str) == lid) & (lp["season"] == season) & (lp["week"] == week)]
            if played.empty:
                continue                                          # the week is not complete for this league yet
            src = "backfill" if (grp["source"].fillna("live") == "backfill").any() else "live"
            order, look = _lookup(played, platform)
            team = played[pd.to_numeric(played["roster_id"], errors="coerce") == rid] if rid is not None else played.iloc[0:0]
            perf = perf_off = None
            if len(team):
                perf, n_perf = perfect_lineup(team, slots, idp_on)
                perf_off, _ = perfect_lineup(team, slots, idp_on, offense_only=True)
                rows.append({"season": int(season), "week": int(week), "league": slug, "basis": "perfect", "points": perf,
                             "offense_points": perf_off, "n_slots": n_perf, "n_missing": 0,
                             "efficiency": 1.0 if perf > 0 else float("nan"),
                             "offense_efficiency": 1.0 if perf_off > 0 else float("nan"), "source": src})

            def row(basis, tot, off, n, miss):
                return {"season": int(season), "week": int(week), "league": slug, "basis": basis, "points": round(tot, 2),
                        "offense_points": round(off, 2), "n_slots": n, "n_missing": miss,
                        "efficiency": _ratio(tot, perf) if perf is not None else float("nan"),
                        "offense_efficiency": _ratio(off, perf_off) if perf_off is not None else float("nan"), "source": src}

            for basis in BASES:
                g = grp[grp["basis"] == basis]
                if g.empty:
                    continue
                tot = off = 0.0
                miss = 0
                for _, r in g.iterrows():
                    hit = next((look[c][_nid(r.get(c))] for c in order if c in look and _nid(r.get(c)) in look[c]), None)
                    if hit is None:
                        miss += 1
                        continue
                    tot += hit[0]
                    off += hit[0] if r.get("position") in OFFENSE else 0.0
                rows.append(row(basis, tot, off, len(g), miss))
            if rid is not None:
                st = played[(pd.to_numeric(played["roster_id"], errors="coerce") == rid) & (pd.to_numeric(played["started"], errors="coerce") == 1)]
                if len(st):
                    pts = pd.to_numeric(st["points"], errors="coerce").fillna(0.0)
                    rows.append(row("actual", float(pts.sum()), float(pts[st["position"].isin(OFFENSE)].sum()), len(st), 0))
    res = pd.DataFrame(rows, columns=EVAL_COLS)
    if write and len(res):
        from ff import build
        build.replace_partition(root / "data" / EVAL_FILE, res, ["season", "week", "league"], ["season", "week", "league", "basis"])
    return res


# --------------------------------------------------------------------------- the report section
TITLE = "## Lineups: which number should pick my lineup"
ORDER = BASES + ("actual", "perfect")
PICKERS = BASES                                                     # who can win a league-week


def load_eval(root: Path = ROOT):
    ev = _read(root / "data" / EVAL_FILE)
    if ev is None or ev.empty:
        return None
    ev = ev.copy()
    for c in ("efficiency", "offense_efficiency"):
        if c not in ev.columns:
            ev[c] = float("nan")
    if "source" not in ev.columns:
        ev["source"] = "live"
    ev["source"] = ev["source"].fillna("live")
    ev["wk"] = ev["season"].astype(int).astype(str) + "|" + ev["week"].astype(int).astype(str) + "|" + ev["league"]
    return ev


def pooled_stats(ev) -> dict:
    """{basis: dict(points, offense, eff, off_eff, n, wins, vs, vs_off)} pooled over every scored league-week. Efficiency is the
    sum of a basis's points over the sum of the perfect points of the same league-weeks."""
    perf = ev[ev["basis"] == "perfect"].drop_duplicates("wk").set_index("wk")
    act = ev[ev["basis"] == "actual"].drop_duplicates("wk").set_index("wk")
    pivot = ev.pivot_table(index="wk", columns="basis", values="points", aggfunc="sum")
    full = [k for k in pivot.index if all(b in pivot.columns and pivot.loc[k, b] == pivot.loc[k, b] for b in PICKERS)]
    wins = {b: 0 for b in PICKERS}
    for k in full:
        top = max(pivot.loc[k, b] for b in PICKERS)
        for b in PICKERS:
            wins[b] += int(pivot.loc[k, b] == top)
    out = {}
    for b in ORDER:
        d = ev[ev["basis"] == b]
        if d.empty:
            continue
        p, a = perf.reindex(d["wk"]), act.reindex(d["wk"])
        have, ok = p["points"].notna().to_numpy(), a["points"].notna().to_numpy()
        pt, of = d["points"].to_numpy(), d["offense_points"].to_numpy()
        out[b] = {"points": float(pt.sum()), "offense": float(of.sum()), "n": len(d),
                  "eff": _ratio(float(pt[have].sum()), float(p["points"].to_numpy()[have].sum())) if have.any() else float("nan"),
                  "off_eff": _ratio(float(of[have].sum()), float(p["offense_points"].to_numpy()[have].sum())) if have.any() else float("nan"),
                  "wins": wins.get(b),
                  "vs": None if b == "actual" or not ok.any() else float(pt[ok].sum() - a["points"].to_numpy()[ok].sum()),
                  "vs_off": None if b == "actual" or not ok.any() else float(of[ok].sum() - a["offense_points"].to_numpy()[ok].sum())}
    return out


def _pct(x) -> str:
    return "n/a" if x is None or x != x else f"{100 * x:.1f}%"


def efficiency_lines(root: Path = ROOT) -> list[str]:
    """Summary-block lines for the lineup scoring (pooled efficiency per basis)."""
    ev = load_eval(root)
    if ev is None:
        return ["Lineup efficiency (points / hindsight-best lineup): no weeks scored yet."]
    st = pooled_stats(ev)
    n = max(v["n"] for v in st.values())
    cell = lambda key: ", ".join(f"{b} {_pct(st[b][key])}" for b in ORDER if b in st and b != "perfect")
    return [f"Lineup efficiency (points / hindsight-best lineup, pooled over {n} league-weeks): {cell('eff')}.",
            f"Offense-only efficiency (QB/RB/WR/TE, the like-for-like number): {cell('off_eff')}.",
            "Lineup wins (highest total of e_pts / model / sleeper / call): " + ", ".join(f"{b} {st[b]['wins']}" for b in PICKERS if b in st) + "."]


def pooled_table(ev) -> str:
    st = pooled_stats(ev)
    L = ["| lineup | total points | offense points | efficiency | offense efficiency | league-weeks | wins | points vs actual |",
         "|---|---|---|---|---|---|---|---|"]
    for b in ORDER:
        if b not in st:
            continue
        v = st[b]
        L.append(f"| {b} | {v['points']:.1f} | {v['offense']:.1f} | {_pct(v['eff'])} | {_pct(v['off_eff'])} | {v['n']} | "
                 f"{'-' if v['wins'] is None else v['wins']} | {'-' if v['vs'] is None else format(v['vs'], '+.1f')} |")
    return "\n".join(L)


def league_table(ev, slug: str) -> str:
    """One league: a row per week, each basis's total and efficiency, the winner, then season-to-date efficiency per basis."""
    d = ev[ev["league"] == slug]
    if d.empty:
        return f"No scored weeks for {slug}."
    L = ["| season / week | " + " | ".join(ORDER) + " | winner |", "|---|" + "---|" * (len(ORDER) + 1)]
    for (s, w), g in d.groupby(["season", "week"]):
        t = g.drop_duplicates("basis").set_index("basis")
        cell = lambda b: (f"{t.loc[b, 'points']:.1f} ({_pct(t.loc[b, 'efficiency'])})" if b in t.index else "-")
        win = "-"
        if all(b in t.index for b in PICKERS):
            top = max(t.loc[b, "points"] for b in PICKERS)
            win = " / ".join(b for b in PICKERS if t.loc[b, "points"] == top)
        mark = " (b)" if (g["source"] == "backfill").any() else ""
        L.append(f"| {int(s)} wk{int(w):02d}{mark} | " + " | ".join(cell(b) for b in ORDER) + f" | {win} |")
    st = pooled_stats(d)
    L += ["", "Selected team efficiency, season to date (points / hindsight-best lineup): " +
          ", ".join(f"{b} {_pct(st[b]['eff'])} (offense {_pct(st[b]['off_eff'])})" for b in ORDER if b in st and b != "perfect") + "."]
    return "\n".join(L)


def section(root: Path = ROOT) -> str:
    """The scoreboard section, a pure function of data/lineup_eval.csv (no clock)."""
    ev = load_eval(root)
    L = [TITLE, ""]
    if ev is None:
        return "\n".join(L + ["No weeks scored yet: a week counts once it is in `data/lineups_played.csv` and a lineup was recorded for it "
                              "before kickoff.", ""])
    nb = int(((ev["basis"] == "perfect") & (ev["source"] == "backfill")).sum())
    L += ["Each week the pipeline records four lineups per league: optimal by E_pts, by the model, by Sleeper's projection, and the call "
          "(frozen slot by slot at kickoff), then scores them with the platform's real points next to the lineup actually started and "
          "the `perfect` lineup (the best his roster could have started, in hindsight). Efficiency = points / perfect points. A win is the "
          "highest total of e_pts / model / sleeper / call in a league-week (a tie counts for each). `actual` and `perfect` also count "
          "K / DEF / IDP slots the recorded lineups may leave empty, so compare lineups on the offense columns (QB/RB/WR/TE). A recorded "
          "player with no played row scores 0.", "", "### All leagues pooled", "", pooled_table(ev), ""]
    for slug in sorted(ev["league"].unique()):
        L += [f"### {slug}", "", league_table(ev, slug), ""]
    if nb:
        L += [f"(b) = backfilled week ({nb} league-weeks): built after the fact from the pre-game roster snapshot and the predictions frozen "
              "before kickoff in `data/model_pts.csv`. Backfilled `call` lineups had no status rule (no historical practice / play_prob), "
              "and K / DEF / IDP slots have no frozen projection, so they are empty there. A league-week where a serve at kickoff overwrote the "
              "frozen rows (under 80% of his projected offensive players still frozen) is not backfilled.", ""]
    try:
        skipped = []
        backfill(root, three_lineups=None, write=False, skipped=skipped)
        for slug, season, week, cov in sorted(skipped):
            L += [f"Frozen coverage: {slug} {season} week {week} skipped, only {cov:.0%} of its projected offensive players have a "
                  f"pre-kickoff prediction (need {MIN_COVERAGE:.0%}).", ""]
    except Exception:                                      # the report must render without it
        pass
    return "\n".join(L)


# --------------------------------------------------------------------------- entry point
def _do_record() -> str:
    new = record(ROOT)
    return f"model.lineups: recorded {len(new)} rows {new.groupby('league').size().to_dict() if len(new) else {}}"


def thin_warns(skipped, root: Path | None = None) -> list[str]:
    """WARN texts for league-weeks backfill skipped for thin frozen coverage, minus those already in logs/runs.csv (a skipped week
    stays skipped, and the same row on every pipeline run would only be noise)."""
    try:
        seen = ((root or ROOT) / "logs" / "runs.csv").read_text(encoding="utf-8")
    except OSError:
        seen = ""
    msgs = [f"lineups backfill skipped {slug} {season} week {week}: frozen coverage {cov:.0%} of his projected offensive players "
            f"(need {MIN_COVERAGE:.0%}); a serve at or after kickoff overwrote the pre-kickoff rows" for slug, season, week, cov in skipped]
    return [m for m in msgs if m not in seen]


def _do_backfill() -> str:
    skipped = []
    new = backfill(ROOT, skipped=skipped)
    if skipped:
        from model.serve import log_warns
        log_warns(ROOT, thin_warns(skipped), check=CHECK)
    n = new.drop_duplicates(["season", "week", "league"]).groupby("league").size().to_dict() if len(new) else {}
    thin = f"; skipped for thin frozen coverage {skipped}" if skipped else ""
    return f"model.lineups: backfilled {len(new)} rows, league-weeks {n}{thin}"


def _do_score() -> str:
    return f"model.lineups: scored {len(score(ROOT))} lineup-weeks"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", nargs="?", choices=("record", "backfill", "score"), help="default: both")
    a = ap.parse_args(argv)
    from model.serve import log_warns
    for name, fn in (("record", _do_record), ("backfill", _do_backfill), ("score", _do_score)):          # each step on its own: a failed record still lets score run
        if a.cmd in (None, name):
            try:
                print(fn())
            except Exception as e:                                # never the pipeline's problem
                msg = f"{type(e).__name__}: {e}"[:400]
                print(f"model.lineups {name} FAILED, outputs untouched: {msg}", file=sys.stderr)
                traceback.print_exc()
                log_warns(ROOT, [f"lineups {name} failed, nothing written: {msg}"], check=CHECK)
    return 0


if __name__ == "__main__":
    sys.exit(main())
