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


MAX_WEEK = 18   # NFL regular season length; Sleeper has no matchups past it


def completed_weeks(current_week: int) -> list[int]:
    """Weeks fully played: strictly below the current NFL week."""
    return list(range(1, min(current_week, MAX_WEEK + 1)))


UNKNOWN = object()      # "nobody told verify() what nflverse has" (e.g. --verify-only)


def verify(data_dir: Path, season: int, week: int,
           league_ids: list | None = None,
           injury_rows=UNKNOWN) -> list[dict]:
    """Structural checks on everything the pipeline just wrote. `league_ids`
    are the configured leagues that must have every completed week recorded
    in lineups_played.csv and matchup_results.csv. `injury_rows` is how many
    nflverse injury-report rows exist for `week` (None: the file could not be
    loaded at all); it decides whether an empty practice column is a problem."""
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
        # depth_charts.csv is upserted by key, so rows for a team that stopped
        # appearing in the source are not removed — they survive from the
        # previous run and keep the 32-team count above satisfied. Compare each
        # team's snapshot against the newest one to surface that.
        if "snapshot_dt" in cur.columns and not cur.empty:
            newest = cur.snapshot_dt.max()
            stale = sorted(cur.groupby("team").snapshot_dt.max()
                           .loc[lambda x: x < newest].index)
            out.append(_r(OK if not stale else WARN, "depth_charts.snapshot",
                          "all teams from the current snapshot" if not stale
                          else (f"{len(stale)} team(s) carried over from an "
                                f"older snapshot, so the team count above is "
                                f"not evidence they were published this run: "
                                f"{', '.join(stale)}")))
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
            # A team on its bye has no game to share carries in; its projection rows
            # (if Sleeper ships any) sum to nothing, which is not a data problem.
            byes = bye_teams(s, season, week)
            shares = cur[~cur.team.isin(byes)].groupby("team").proj_carry_share.sum()
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
        out.append(_practice_check(st, week, injury_rows))

    # --- played lineups: every completed week, every configured league ------
    done = completed_weeks(week)
    if league_ids and done:
        for name in ("lineups_played.csv", "matchup_results.csv"):
            f = rd(name)
            have = set()
            if f is not None and not f.empty:
                cur = f[f.season == season]
                have = set(zip(cur.league_id.astype(str), cur.week.astype(int)))
            gaps = {str(lid): [w for w in done if (str(lid), w) not in have]
                    for lid in league_ids}
            gaps = {k: v for k, v in gaps.items() if v}
            out.append(_r(OK if not gaps else WARN, f"{name[:-4]}.coverage",
                          f"weeks {done[0]}-{done[-1]} recorded for every league"
                          if not gaps else "missing " + ", ".join(
                              f"league {k} wk{v}" for k, v in gaps.items())))

    # --- freshness --------------------------------------------------------
    now = datetime.now(timezone.utc)
    for name in ("players.csv", "depth_charts.csv", "status.csv"):
        f = data_dir / name
        if f.exists():
            age = (now.timestamp() - f.stat().st_mtime) / 3600
            out.append(_r(OK if age < 48 else WARN, f"freshness.{name}",
                          f"{age:.1f}h old"))
    return out


def bye_teams(schedule: pd.DataFrame | None, season: int, week: int) -> set:
    """Teams with no game in `week`. schedule.csv carries an explicit BYE row for each (build.build_schedule);
    a team with no row at all that week counts too."""
    if schedule is None or schedule.empty:
        return set()
    sea = schedule[schedule.season == season]
    wk = sea[sea.week == week]
    if wk.empty:
        return set()
    explicit = set(wk[wk.opponent.astype(str).str.upper() == "BYE"].team)
    return explicit | (set(sea.team) - set(wk.team))


def _practice_check(st: pd.DataFrame, week: int, injury_rows) -> dict:
    """status.csv practice coverage, judged against what nflverse actually has.

    The practice column comes from nflverse's injury report for the week. An empty
    column is only a problem when nflverse HAS rows for the week and none of them
    reached a status row (an id/column/join break). Before Wednesday nflverse has no
    rows for the week yet, and that is normal."""
    prac = st.practice.notna().mean() if "practice" in st.columns else 0
    n = int(st.practice.notna().sum()) if "practice" in st.columns else 0
    if prac > 0:
        return _r(OK, "status.practice", f"{prac:.0%} have practice reports ({n} players)")
    if injury_rows is UNKNOWN:      # caller could not say: keep the old, cautious reading
        return _r(WARN, "status.practice",
                  f"{prac:.0%} have practice reports (normal before Wednesday)")
    if injury_rows is None:
        # The file is part of the season from week 1 on; missing later means it broke.
        if week <= 1:
            return _r(OK, "status.practice",
                      "0% have practice reports (nflverse injuries not published yet; "
                      "normal before Wednesday)")
        return _r(WARN, "status.practice",
                  "0% have practice reports: nflverse injuries file could not be loaded")
    if injury_rows == 0:
        return _r(OK, "status.practice",
                  f"0% have practice reports (nflverse has none for week {week} yet; "
                  "normal before Wednesday)")
    return _r(WARN, "status.practice",
              f"0% have practice reports but nflverse has {injury_rows} report rows "
              f"for week {week}: none matched a status row")


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


DETAIL_CAP = 500


def _append_run(logs_dir: Path, row: dict) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    f = logs_dir / "runs.csv"
    df = pd.DataFrame([row])
    if f.exists():
        df = pd.concat([pd.read_csv(f), df], ignore_index=True)
    df.to_csv(f, index=False)
    return f


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_run(logs_dir: Path, season: int, week: int, results: list[dict],
            note: str = "") -> Path:
    """Append this run's outcome so gaps are visible after the fact."""
    fails = sum(1 for r in results if r["severity"] == FAIL)
    warns = sum(1 for r in results if r["severity"] == WARN)
    return _append_run(logs_dir, {
        "ran_at": _now_iso(),
        "season": season, "week": week,
        "status": "FAIL" if fails else ("WARN" if warns else "OK"),
        "fails": fails, "warns": warns,
        "detail": "; ".join(f"{r['check']}: {r['detail']}" for r in results
                            if r["severity"] != OK)[:DETAIL_CAP],
        "note": note,
    })


def log_crash(logs_dir: Path, season: int, week: int, exc: BaseException) -> Path:
    """A run that died before it could call log_run still leaves a row: FAIL, with the
    exception's class and first line. Without it the committed log shows a clean week
    around a crash and only the Actions UI knows."""
    first = (str(exc).strip().splitlines() or [""])[0]
    return _append_run(logs_dir, {
        "ran_at": _now_iso(),
        "season": season, "week": week,
        "status": "FAIL", "fails": 1, "warns": 0,
        "detail": f"{type(exc).__name__}: {first}"[:DETAIL_CAP],
        "note": "crash",
    })


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
