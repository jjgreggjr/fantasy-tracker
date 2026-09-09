"""Data integrity checks and missed-run recovery.

Two separate worries:

  1. Did the run happen? Windows handles most of this (`-StartWhenAvailable`
     makes a task fire as soon as the machine is back), and anything sourced
     from a full-season file self-heals on the next run regardless.
  2. Is the data *right*? A run that "succeeds" while a source quietly returns
     an empty or stale file is worse than one that fails loudly.

`verify()` answers the second. `missing_weeks()` and the backfill helpers
answer what is left of the first.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

OK, WARN, FAIL = "OK", "WARN", "FAIL"


def _r(sev, check, detail):
    return {"severity": sev, "check": check, "detail": detail}


def verify(data_dir: Path, season: int, week: int) -> list[dict]:
    """Structural checks on everything the pipeline just wrote."""
    out = []
    rd = lambda n: (pd.read_csv(data_dir / n, low_memory=False)
                    if (data_dir / n).exists() else None)

    # --- depth charts: every team, every position, a starter ---------------
    d = rd("depth_charts.csv")
    if d is None or d.empty:
        out.append(_r(FAIL, "depth_charts", "missing"))
    else:
        cur = d[d.week == week]
        teams = cur.team.nunique()
        out.append(_r(OK if teams == 32 else FAIL, "depth_charts.teams",
                      f"{teams}/32 teams for week {week}"))
        missing = []
        for pos in ("QB", "RB", "WR", "TE"):
            have = cur[(cur.position == pos) & (cur["rank"] == 1)].team.nunique()
            if have < 32:
                missing.append(f"{pos} {have}/32")
        out.append(_r(OK if not missing else WARN, "depth_charts.starters",
                      "every team has a starter at each position" if not missing
                      else "missing rank-1: " + ", ".join(missing)))

    # --- schedule ---------------------------------------------------------
    s = rd("schedule.csv")
    if s is None or s.empty:
        out.append(_r(FAIL, "schedule", "missing"))
    else:
        n = len(s[s.week == week])
        out.append(_r(OK if n == 32 else FAIL, "schedule.week",
                      f"{n}/32 team-rows for week {week}"))

    # --- player weeks: duplicates and continuity ---------------------------
    pw = rd("player_weeks.csv")
    if pw is None or pw.empty:
        out.append(_r(WARN if week <= 1 else FAIL, "player_weeks",
                      "no rows yet" if week <= 1 else "missing after week 1"))
    else:
        cur = pw[pw.season == season]
        dupes = cur.duplicated(["season", "week", "gsis_id"]).sum()
        out.append(_r(OK if dupes == 0 else FAIL, "player_weeks.duplicates",
                      f"{dupes} duplicate player-weeks"))
        weeks = sorted(cur.week.unique())
        gaps = [w for w in range(1, max(weeks) + 1) if w not in weeks] if weeks else []
        out.append(_r(OK if not gaps else FAIL, "player_weeks.continuity",
                      f"weeks {weeks}" if not gaps
                      else f"MISSING weeks {gaps} (have {weeks})"))
        if weeks:
            per = cur.groupby("week").size()
            thin = per[per < 200]
            out.append(_r(OK if thin.empty else WARN, "player_weeks.volume",
                          f"{per.min()}-{per.max()} rows/week"
                          if thin.empty else
                          f"suspiciously few rows in week(s) {list(thin.index)}"))
            miss_snap = cur[cur.week == max(weeks)].snap_pct.isna().mean()
            out.append(_r(OK if miss_snap < 0.5 else WARN, "player_weeks.snaps",
                          f"{miss_snap:.0%} of week {max(weeks)} missing snap counts"))

    # --- players / id crosswalk -------------------------------------------
    p = rd("players.csv")
    if p is None or p.empty:
        out.append(_r(FAIL, "players", "missing"))
    else:
        act = p[p.nfl_status == "ACT"]
        rate = 1 - act.sleeper_id.isna().mean() if len(act) else 0
        out.append(_r(OK if rate >= 0.95 else WARN, "players.id_match",
                      f"{rate:.1%} of active skill players matched to Sleeper"))

    # --- projections ------------------------------------------------------
    pr = rd("projections.csv")
    if pr is None or pr.empty:
        out.append(_r(WARN, "projections", "none for any week"))
    else:
        cur = pr[(pr.season == season) & (pr.week == week)]
        out.append(_r(OK if len(cur) >= 200 else WARN, "projections.week",
                      f"{len(cur)} players projected for week {week}"))
        if not cur.empty:
            shares = cur.groupby("team").proj_carry_share.sum()
            bad = shares[(shares < 0.9) | (shares > 1.1)]
            out.append(_r(OK if bad.empty else WARN, "projections.shares",
                          "carry shares sum to ~1 per team" if bad.empty
                          else f"{len(bad)} teams with shares off: {list(bad.index)[:5]}"))

    # --- status -----------------------------------------------------------
    st = rd("status.csv")
    if st is None or st.empty:
        out.append(_r(WARN, "status", "missing — no availability checking"))
    else:
        out.append(_r(OK, "status.rows", f"{len(st)} players"))
        prac = st.practice.notna().mean() if "practice" in st.columns else 0
        out.append(_r(OK if prac > 0 else WARN, "status.practice",
                      f"{prac:.0%} have practice reports"
                      + (" (normal before Wednesday)" if prac == 0 else "")))

    # --- freshness --------------------------------------------------------
    now = datetime.now(timezone.utc)
    for name in ("players.csv", "depth_charts.csv", "status.csv"):
        f = data_dir / name
        if f.exists():
            age = (now.timestamp() - f.stat().st_mtime) / 3600
            out.append(_r(OK if age < 48 else WARN, f"freshness.{name}",
                          f"{age:.1f}h old"))
    return out


def missing_weeks(data_dir: Path, season: int, through_week: int) -> list[int]:
    """Regular-season weeks with no player rows but which should have them."""
    f = data_dir / "player_weeks.csv"
    if not f.exists():
        return list(range(1, through_week + 1))
    pw = pd.read_csv(f, low_memory=False)
    have = set(pw[pw.season == season].week.unique())
    return [w for w in range(1, through_week + 1) if w not in have]


def backfill_depth_charts(raw_depth: pd.DataFrame, season: int, weeks: list[int],
                          schedule: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct depth charts for weeks we missed.

    nflverse keeps every snapshot it has ever taken, each stamped with a
    timestamp, so a week missed because the machine was off can be rebuilt
    from the snapshot that was current on that week's first game day.
    """
    if raw_depth is None or raw_depth.empty or not weeks:
        return pd.DataFrame()
    from .build import build_depth_charts
    d = raw_depth.copy()
    d["dt"] = pd.to_datetime(d.dt, errors="coerce", utc=True)
    frames = []
    for w in weeks:
        games = schedule[(schedule.season == season) & (schedule.week == w)]
        if games.empty or games.gameday.isna().all():
            continue
        cutoff = pd.to_datetime(games.gameday.dropna().min(), utc=True)
        past = d[d.dt <= cutoff]
        if past.empty:
            continue
        frames.append(build_depth_charts(past, season, w, None))
        log.info("backfilled depth chart for week %d (snapshot <= %s)",
                 w, cutoff.date())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def log_run(logs_dir: Path, season: int, week: int, results: list[dict],
            note: str = "") -> Path:
    """Append this run's outcome so gaps are visible after the fact."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    f = logs_dir / "runs.csv"
    fails = sum(1 for r in results if r["severity"] == FAIL)
    warns = sum(1 for r in results if r["severity"] == WARN)
    row = pd.DataFrame([{
        "ran_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season": season, "week": week,
        "status": "FAIL" if fails else ("WARN" if warns else "OK"),
        "fails": fails, "warns": warns,
        "detail": "; ".join(f"{r['check']}: {r['detail']}" for r in results
                            if r["severity"] != OK)[:500],
        "note": note,
    }])
    if f.exists():
        row = pd.concat([pd.read_csv(f), row], ignore_index=True)
    row.to_csv(f, index=False)
    return f


def last_run_gap(logs_dir: Path) -> float | None:
    """Hours since the last recorded run, or None if never run."""
    f = logs_dir / "runs.csv"
    if not f.exists():
        return None
    r = pd.read_csv(f)
    if r.empty:
        return None
    last = pd.to_datetime(r.ran_at.iloc[-1], utc=True, errors="coerce")
    if pd.isna(last):
        return None
    return round((datetime.now(timezone.utc) - last).total_seconds() / 3600, 1)
