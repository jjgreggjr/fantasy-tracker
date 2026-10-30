"""Recorded optimal lineups, frozen slot by slot at kickoff, and what each would have scored.

    python -m model.lineups            # record, then score
    python -m model.lineups record     # this week's three optimal lineups per league -> data/lineups_optimal.csv
    python -m model.lineups score      # every complete week -> data/lineup_eval.csv (the report section is rendered by model.scoreboard)

record: per league, three lineups from the league's committed roster.csv (after `modelcols.attach`), via `ff.calls.three_lineups`:
  e_pts   the exact optimum by E_pts          model   the exact optimum by E_pts_model          call   the recommendation
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
BASES = ("e_pts", "model", "call")
OFFENSE = ("QB", "RB", "WR", "TE")
IDS = ("gsis_id", "sleeper_id", "espn_id")
OPT_COLS = ["season", "week", "league", "basis", "slot", *IDS, "name", "position", "team", "E_pts", "E_pts_model",
            "kickoff_utc", "recorded_at", "locked", "e_pick", "m_pick", "reason"]       # the last three are the call's own notes
EVAL_COLS = ["season", "week", "league", "basis", "points", "offense_points", "n_slots", "n_missing"]
PART = ["season", "week", "league", "basis"]
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
    roster = modelcols.attach(roster, slug, root)
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
            fresh = three_lineups(pool.reset_index(drop=True), left, idp_on)[basis].copy()
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


def score(root: Path = ROOT, write: bool = True):
    """Rows of lineup_eval.csv for every complete (season, week, league) in lineups_optimal.csv."""
    opt = _read_str(root / "data" / OPT_FILE)
    lp = _read(root / "data" / "lineups_played.csv", dtype={c: str for c in IDS + ("league_id",)})
    empty = pd.DataFrame(columns=EVAL_COLS)
    if opt is None or opt.empty or lp is None or lp.empty:
        return empty
    rows = []
    for slug, meta, _ in _leagues(root):
        lid, platform = str(meta["league_id"]), str(meta.get("platform") or "sleeper").lower()
        rid = _my_roster_id(meta, root, lid)
        mine = opt[opt["league"] == slug]
        for (season, week), grp in mine.groupby(["season", "week"]):
            played = lp[(lp["league_id"].astype(str) == lid) & (lp["season"] == season) & (lp["week"] == week)]
            if played.empty:
                continue                                          # the week is not complete for this league yet
            order, look = _lookup(played, platform)
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
                rows.append({"season": int(season), "week": int(week), "league": slug, "basis": basis, "points": round(tot, 2),
                             "offense_points": round(off, 2), "n_slots": len(g), "n_missing": miss})
            if rid is not None:
                st = played[(pd.to_numeric(played["roster_id"], errors="coerce") == rid) & (pd.to_numeric(played["started"], errors="coerce") == 1)]
                if len(st):
                    pts = pd.to_numeric(st["points"], errors="coerce").fillna(0.0)
                    rows.append({"season": int(season), "week": int(week), "league": slug, "basis": "actual",
                                 "points": round(float(pts.sum()), 2),
                                 "offense_points": round(float(pts[st["position"].isin(OFFENSE)].sum()), 2),
                                 "n_slots": len(st), "n_missing": 0})
    res = pd.DataFrame(rows, columns=EVAL_COLS)
    if write and len(res):
        from ff import build
        build.replace_partition(root / "data" / EVAL_FILE, res, ["season", "week", "league"], ["season", "week", "league", "basis"])
    return res


# --------------------------------------------------------------------------- the report section
TITLE = "## Lineups: which optimizer would have scored more"
ORDER = BASES + ("actual",)


def section(root: Path = ROOT) -> str:
    """The scoreboard section, a pure function of data/lineup_eval.csv (no clock)."""
    ev = _read(root / "data" / EVAL_FILE)
    L = [TITLE, ""]
    if ev is None or ev.empty:
        return "\n".join(L + ["No weeks scored yet: a week counts once it is in `data/lineups_played.csv` and a lineup was recorded for it "
                              "before kickoff.", ""])
    ev = ev.copy()
    ev["wk"] = ev["season"].astype(int).astype(str) + "|" + ev["week"].astype(int).astype(str) + "|" + ev["league"]
    pivot = ev.pivot_table(index="wk", columns="basis", values="points", aggfunc="sum")
    full = [k for k in pivot.index if all(b in pivot.columns and pivot.loc[k, b] == pivot.loc[k, b] for b in BASES)]
    wins = {b: 0 for b in BASES}
    for k in full:
        top = max(pivot.loc[k, b] for b in BASES)
        for b in BASES:
            wins[b] += int(pivot.loc[k, b] == top)
    act = ev[ev["basis"] == "actual"].set_index("wk")
    L += ["Each week the pipeline records three optimal lineups per league (by E_pts, by the model, and the call), frozen slot by slot at "
          "kickoff, then scores them with the platform's real points next to the lineup actually started. Wins count league-weeks where "
          "that lineup had the highest total of the three (a tie counts for each). `actual` also holds K / DEF / IDP starters the "
          "recorded lineups do not, so compare on the offense column (QB/RB/WR/TE only). A recorded player with no played row scores 0.",
          "", "### All leagues pooled", "",
          "| lineup | total points | offense points | league-weeks | wins | total vs actual | offense vs actual |", "|---|---|---|---|---|---|---|"]
    for b in ORDER:
        d = ev[ev["basis"] == b]
        if d.empty:
            continue
        a = act.reindex(d["wk"])
        vs = lambda col: ("-" if b == "actual" or a[col].isna().all() else f"{d[col].sum() - a[col].sum():+.1f}")
        L.append(f"| {b} | {d['points'].sum():.1f} | {d['offense_points'].sum():.1f} | {len(d)} | "
                 f"{wins[b] if b in wins else '-'} | {vs('points')} | {vs('offense_points')} |")
    L.append("")
    for slug in sorted(ev["league"].unique()):
        d = ev[ev["league"] == slug]
        L += [f"### {slug}", "", "| season / week | e_pts | model | call | actual | best of the three |", "|---|---|---|---|---|---|"]
        for (s, w), g in d.groupby(["season", "week"]):
            t = g.set_index("basis")["points"]
            cell = lambda b: f"{t[b]:.1f}" if b in t.index else "-"
            win = "-"
            if all(b in t.index for b in BASES):
                top = max(t[b] for b in BASES)
                win = " / ".join(b for b in BASES if t[b] == top)
            L.append(f"| {int(s)} wk{int(w):02d} | " + " | ".join(cell(b) for b in ORDER) + f" | {win} |")
        L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- entry point
def _do_record() -> str:
    new = record(ROOT)
    return f"model.lineups: recorded {len(new)} rows {new.groupby('league').size().to_dict() if len(new) else {}}"


def _do_score() -> str:
    return f"model.lineups: scored {len(score(ROOT))} lineup-weeks"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", nargs="?", choices=("record", "score"), help="default: both")
    a = ap.parse_args(argv)
    from model.serve import log_warns
    for name, fn in (("record", _do_record), ("score", _do_score)):          # each step on its own: a failed record still lets score run
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
