"""Entrypoint: python -m ff.run_weekly [--season 2026] [--week N] [--skip-sleeper]"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import (analyze, build, espn, leagues, projections, report, score,
               scoring, sources, status as status_mod, verify as verify_mod)

ROOT = Path(__file__).resolve().parent.parent
DATA, RAW, REPORTS, LOGS = (ROOT / "data", ROOT / "raw",
                            ROOT / "reports", ROOT / "logs")
CONFIG = ROOT / "config.json"
log = logging.getLogger("ff")


def setup_logging(today: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOGS / f"run_{today}.log", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])


def _my_rows(lr: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Rows belonging to James. Match on Sleeper user_id; fall back to the
    display name only if the id is unavailable."""
    if lr is None or lr.empty:
        return pd.DataFrame()
    uid = str(cfg.get("sleeper_user_id") or "")
    if uid and "owner_id" in lr.columns:
        hit = lr[lr.owner_id.astype(str) == uid]
        if not hit.empty:
            return hit
    name = (cfg.get("sleeper_display_name")
            or cfg.get("sleeper_username", "")).lstrip("@").lower()
    if not name:
        return pd.DataFrame()
    return lr[lr.owner_name.astype(str).str.lower() == name]


def load_config() -> dict:
    if not CONFIG.exists():
        raise SystemExit(f"Missing {CONFIG}. Copy config.example.json and edit it.")
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def save_config(cfg: dict) -> None:
    CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def resolve_leagues(cfg: dict, season: int, warnings: list) -> list[dict]:
    """Fetch each configured league. On first run, discover and store them all."""
    username = cfg.get("sleeper_username", "").strip()
    if not username:
        warnings.append("No sleeper_username in config.json; league sections skipped.")
        return []
    try:
        uid, found = sources.sleeper_leagues(username, season)
    except Exception as e:
        warnings.append(f"Sleeper unreachable ({e}); league sections skipped. "
                        "Run this on a machine that can reach api.sleeper.app.")
        return []
    cfg["sleeper_user_id"] = uid

    configured = cfg.get("leagues") or []
    if not configured:
        configured = [{"league_id": l["league_id"], "name": l.get("name"),
                       "type": l.get("settings", {}).get("type")} for l in found]
        cfg["leagues"] = configured
        save_config(cfg)
        log.info("discovered %d leagues for %s", len(configured), username)

    out = []
    for entry in configured:
        try:
            detail = sources.sleeper_league_detail(entry["league_id"])
            detail["league"].setdefault("name", entry.get("name"))
            out.append(detail)
        except Exception as e:
            warnings.append(f"League {entry.get('name') or entry['league_id']} "
                            f"failed to load: {e}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int, help="week to preview")
    ap.add_argument("--skip-sleeper", action="store_true")
    ap.add_argument("--verify-only", action="store_true",
                    help="run integrity checks against existing data, change nothing")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any integrity check FAILs (after writing everything) "
                         "— lets a scheduler mark the run failed and alert")
    args = ap.parse_args(argv)

    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    setup_logging(today)
    cfg = load_config()
    warnings: list[str] = []

    if args.verify_only:
        season = args.season or cfg.get("season") or now.year
        week = args.week or 1
        res = verify_mod.verify(DATA, season, week)
        for r in res:
            print(f"  [{r['severity']:4s}] {r['check']:28s} {r['detail']}")
        gap = verify_mod.last_run_gap(LOGS)
        print(f"\nlast recorded run: "
              + (f"{gap}h ago" if gap is not None else "never"))
        return 1 if any(r["severity"] == "FAIL" for r in res) else 0

    season = args.season or cfg.get("season") or now.year
    preview_week = args.week
    if preview_week is None:
        try:
            st = sources.sleeper_state()
            season = args.season or int(st.get("season", season))
            preview_week = int(st.get("week") or 1)
        except Exception as e:
            warnings.append(f"Could not read Sleeper state ({e}); defaulting to week 1. "
                            "Pass --week to be explicit.")
            preview_week = 1
    stats_week = preview_week - 1 if preview_week > 1 else None

    log.info("season=%s preview_week=%s stats_week=%s", season, preview_week, stats_week)

    # ---- raw data -------------------------------------------------------
    nv = sources.load_nflverse(season, RAW / "nflverse", with_prior=True)
    for key, label in [("stats", f"stats_player_week_{season}.csv"),
                       ("snaps", f"snap_counts_{season}.csv")]:
        if nv.get(key) is None:
            warnings.append(f"{label} not published yet — using prior season only.")

    sleeper_players = None
    if not args.skip_sleeper:
        try:
            sleeper_players = sources.sleeper_players(RAW / "sleeper", today)
        except Exception as e:
            warnings.append(f"Sleeper player dictionary unavailable ({e}); "
                            "injury status will be blank.")

    # ---- dimensions -----------------------------------------------------
    players = build.build_players(nv["roster"], sleeper_players, pd.Timestamp(now.date()))
    players.to_csv(DATA / "players.csv", index=False)

    sched = build.build_schedule(nv["games"], season)
    sched.to_csv(DATA / "schedule.csv", index=False)

    # ---- facts ----------------------------------------------------------
    pw_new = pd.DataFrame()
    if nv.get("stats") is not None:
        pw_new = build.build_player_weeks(nv["stats"], nv.get("snaps"), players, season)
        if not pw_new.empty:
            build.upsert(DATA / "player_weeks.csv", pw_new, ["season", "week", "gsis_id"])
            missing = pw_new.snap_pct.isna().mean()
            if missing > 0.5:
                warnings.append(
                    f"Snap counts missing for {missing:.0%} of player-weeks "
                    "(PFR often lags to Tuesday afternoon). Re-run to backfill.")

    prior_pw = pd.DataFrame()
    if nv.get("stats_prior") is not None:
        prior_players = build.build_players(
            nv.get("roster_prior", nv["roster"]), None, pd.Timestamp(now.date()))
        prior_pw = build.build_player_weeks(
            nv["stats_prior"], nv.get("snaps_prior"), prior_players, season - 1)

    pw_all = (build.read_data_csv(DATA / "player_weeks.csv")
              if (DATA / "player_weeks.csv").exists() else pd.DataFrame())
    pw_season = pw_all[pw_all.season == season] if not pw_all.empty else pd.DataFrame()

    prev_depth = (build.read_data_csv(DATA / "depth_charts.csv")
                  if (DATA / "depth_charts.csv").exists() else None)
    depth = build.build_depth_charts(nv["depth"], season, preview_week, prev_depth)
    build.upsert(DATA / "depth_charts.csv", depth, ["season", "week", "team",
                                                    "position", "rank"])

    # If the machine was off, weeks can be missing. Stats self-heal because
    # nflverse ships the whole season in one file; depth charts do not, so
    # rebuild them from the snapshot that was current that week.
    gaps = verify_mod.missing_weeks(DATA, season, max(stats_week or 0, 0))
    if gaps:
        warnings.append(f"No player rows for week(s) {gaps} — likely missed runs. "
                        "Stats self-heal from the season file; depth charts "
                        "backfilled from historical snapshots.")
        bf = verify_mod.backfill_depth_charts(nv["depth"], season, gaps, sched)
        if not bf.empty:
            build.upsert(DATA / "depth_charts.csv", bf,
                         ["season", "week", "team", "position", "rank"])
            log.info("backfilled depth charts for %d week(s)", len(gaps))

    dvp = build.build_defense_vs_pos(pw_season, nv["games"])
    if not dvp.empty:
        dvp.to_csv(DATA / "defense_vs_pos.csv", index=False)
    prior_dvp = build.build_defense_vs_pos(prior_pw, nv["games"])

    # ---- leagues --------------------------------------------------------
    leagues_raw = [] if args.skip_sleeper else resolve_leagues(cfg, season, warnings)
    lr = build.build_league_rosters(leagues_raw, players, season, preview_week)
    if not lr.empty:
        build.upsert(DATA / "league_rosters.csv", lr,
                     ["season", "week", "league_id", "sleeper_id"])
        mine = _my_rows(lr, cfg)
        if mine.empty:
            warnings.append(
                "Could not identify your teams in the league rosters "
                f"(user_id={cfg.get('sleeper_user_id')}). Start/sit skipped.")
        else:
            build.upsert(DATA / "my_roster.csv", mine,
                         ["season", "week", "league_id", "sleeper_id"])

    # ---- analysis -------------------------------------------------------
    trailing = cfg["startsit"]["trailing_weeks"]
    base = analyze.player_baselines(pw_season, preview_week, trailing, prior_pw)
    passrank = analyze.team_pass_rank(pw_season, prior_pw, preview_week)
    ranks = analyze.dvp_ranks(dvp, preview_week, 4, prior_dvp)

    w = analyze.blend_weight(preview_week)
    if pw_season.empty:
        baseline_note = f"{season-1} only (no {season} games played yet)"
    elif w:
        baseline_note = f"{int((1-w)*100)}% {season} / {int(w*100)}% {season-1}"
    else:
        baseline_note = f"{season} only"

    # ---- projections ----------------------------------------------------
    proj = pd.DataFrame()
    if not args.skip_sleeper:
        proj = projections.build(season, preview_week, players,
                                 RAW / "sleeper", today)
        if proj.empty:
            warnings.append("No projections for this week; scores fall back to "
                            "trailing volume only.")
        else:
            build.upsert(DATA / "projections.csv", proj,
                         ["season", "week", "gsis_id", "source"])
            log.info("projections: %d players", len(proj))

    # ---- trending (league-agnostic hot list) -----------------------------
    trending = pd.DataFrame()
    if not args.skip_sleeper:
        trending = leagues.trending_frame(players)
        if not trending.empty:
            trending.to_csv(DATA / "trending.csv", index=False)

    # ---- status: availability, and who is ahead of whom ------------------
    st = status_mod.build(players, depth, proj, preview_week)
    if not st.empty:
        st.to_csv(DATA / "status.csv", index=False)
        flagged = st[(st.depth_disagreement >= 2)
                     | (st.opportunity_ahead >= 0.15)]
        log.info("status: %d players, %d flagged for review", len(st), len(flagged))

    # ---- per-league workspaces ------------------------------------------
    pw_hist = pw_all if not pw_all.empty else prior_pw
    if not prior_pw.empty and not pw_all.empty:
        pw_hist = pd.concat([prior_pw, pw_all], ignore_index=True)
    elif prior_pw is not None and not prior_pw.empty and pw_all.empty:
        pw_hist = prior_pw

    league_ctx = []
    slugs_cfg = {str(e.get("league_id")): e for e in (cfg.get("leagues") or [])}
    for lg in leagues_raw:
        L = lg["league"]
        meta = scoring.sleeper_league_meta(L, lg["users"], lg["rosters"],
                                           cfg.get("sleeper_user_id"))
        meta["slug"] = (slugs_cfg.get(str(meta["league_id"]), {}).get("slug")
                        or leagues.slugify(meta["name"]))
        slugs_cfg.setdefault(str(meta["league_id"]), {})["slug"] = meta["slug"]
        sub = lr[lr.league_id == meta["league_id"]] if not lr.empty else pd.DataFrame()
        mine = _my_rows(sub, cfg)
        ctx = leagues.write_league(ROOT, meta, mine, sub, players, pw_hist, base,
                                   proj, ranks, sched, depth, preview_week,
                                   season, trending, cfg)
        ctx["watchlist"] = analyze.watchlist(
            players, base, depth,
            set(sub.gsis_id.dropna()) if not sub.empty else set(),
            passrank, cfg["watchlist"])
        ctx["status_warnings"] = status_mod.warnings(
            st, set(mine.gsis_id.dropna()) if not mine.empty else set())
        league_ctx.append(ctx)

    # ---- optional ESPN league -------------------------------------------
    for e in (cfg.get("espn_leagues") or []):
        data = espn.fetch(season, e["league_id"], ROOT)
        if not data:
            warnings.append(f"ESPN league {e['league_id']} could not be loaded "
                            "(private leagues need secrets/espn_cookies.json).")
            continue
        meta = espn.parse_meta(data, e["league_id"], season)
        meta["slug"] = e.get("slug") or leagues.slugify(meta["name"])
        rosters = espn.parse_rosters(data, players, season, preview_week, meta)
        my_name = e.get("my_team_name")
        mine = (rosters[rosters.owner_name.str.contains(my_name, case=False, na=False)]
                if my_name else pd.DataFrame())
        if mine.empty and my_name:
            warnings.append(f"ESPN: no team matching '{my_name}'. Owners seen: "
                            + ", ".join(sorted(rosters.owner_name.unique())[:8]))
        ctx = leagues.write_league(ROOT, meta, mine, rosters, players, pw_hist,
                                   base, proj, ranks, sched, depth,
                                   preview_week, season, trending, cfg)
        ctx["watchlist"] = analyze.watchlist(
            players, base, depth, set(rosters.gsis_id.dropna()),
            passrank, cfg["watchlist"])
        ctx["status_warnings"] = status_mod.warnings(
            st, set(mine.gsis_id.dropna()) if not mine.empty else set())
        league_ctx.append(ctx)

    cfg["leagues"] = [{"league_id": k, **v} for k, v in slugs_cfg.items()]
    save_config(cfg)

    if not league_ctx:
        league_ctx = [{
            "meta": {"name": "(no league data — Sleeper not reachable)", "slug": "none"},
            "dir": REPORTS, "roster": pd.DataFrame(), "available": pd.DataFrame(),
            "all_rosters": pd.DataFrame(), "points": pd.DataFrame(),
            "watchlist": analyze.watchlist(players, base, depth, set(), passrank,
                                           cfg["watchlist"]),
        }]

    freshness = {
        "depth chart snapshot": (depth.snapshot_dt.max() if not depth.empty else "n/a"),
        f"{season} player weeks": (f"weeks {sorted(pw_season.week.unique())}"
                                   if not pw_season.empty else "none yet"),
        f"{season-1} baseline weeks": (f"{prior_pw.week.nunique()} weeks"
                                       if not prior_pw.empty else "none"),
        "players in dimension": len(players),
        "Sleeper players cache": today if sleeper_players else "unavailable",
        "projections": (f"{len(proj)} players (Sleeper, {proj.pulled_at.iloc[0]})"
                        if not proj.empty else "none"),
        "status rows": (f"{len(st)} players; "
                        f"{int((st.practice.notna()).sum())} with practice reports"
                        if not st.empty else "none"),
    }

    ctx = {"season": season, "preview_week": preview_week, "stats_week": stats_week,
           "now": now.strftime("%Y-%m-%d %H:%M UTC"), "baseline_note": baseline_note,
           "freshness": freshness, "warnings": warnings,
           "leagues": league_ctx, "dvp": ranks, "trending": trending}

    checks = verify_mod.verify(DATA, season, preview_week)
    verify_mod.log_run(LOGS, season, preview_week, checks)
    bad = [c for c in checks if c["severity"] != "OK"]
    for c in bad:
        log.warning("verify %s: %s — %s", c["severity"], c["check"], c["detail"])
    ctx["checks"] = checks

    REPORTS.mkdir(parents=True, exist_ok=True)
    for lg in league_ctx:                      # one report per league
        if lg.get("dir") and lg["dir"] != REPORTS:
            (lg["dir"] / "report.md").write_text(
                report.render_league(lg, ctx), encoding="utf-8")
    md = report.render_digest(ctx)
    path = REPORTS / f"{season}_wk{preview_week:02d}.md"
    path.write_text(md, encoding="utf-8")
    (REPORTS / "latest.md").write_text(md, encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"season {season}  preview week {preview_week}  stats through "
          f"{stats_week or 'n/a'}")
    print(f"players={len(players)}  player_weeks(season)={len(pw_season)}  "
          f"depth_rows={len(depth)}  dvp_rows={len(dvp)}")
    print(f"leagues={len(league_ctx)}  rostered_players={len(lr)}  "
          f"projections={len(proj)}")
    for lg in league_ctx:
        if lg.get("dir") and lg["dir"] != REPORTS:
            print(f"  - {lg['meta']['name']}: roster={len(lg['roster'])} "
                  f"available={len(lg['available'])} -> {lg['dir'].name}/")
    fails = sum(1 for c in checks if c["severity"] == "FAIL")
    warns = sum(1 for c in checks if c["severity"] == "WARN")
    print(f"warnings={len(warnings)}  integrity: "
          + ("all checks passed" if not (fails or warns)
             else f"{fails} FAIL, {warns} WARN — see logs/runs.csv"))
    print(f"report -> {path}")
    print("=" * 60)
    if args.strict and fails:
        log.error("--strict: %d integrity check(s) FAILED; exiting 1", fails)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
