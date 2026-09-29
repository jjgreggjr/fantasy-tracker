"""Re-runnable source probe: is each PLAN_MODEL.md source reachable from THIS machine, and what
does it look like. Phase 0 ran it from the Claude sandbox; the three hosts that are blocked there
(FFC, CFBD, Open-Meteo) are meant to be re-probed from GitHub Actions with the same script.

    model/.venv/bin/python -m model.probe_sources
    CFBD_KEY=... model/.venv/bin/python -m model.probe_sources     # authenticated CFBD probe

Prints; never writes to the repo. Failures are findings, not crashes.
"""
from __future__ import annotations

import os

import pandas as pd
import requests

from model import point_in_time as pit

NFL = pit.NFLVERSE


def attempt(label: str, url: str, **kw):
    try:
        r = requests.get(url, timeout=60, **kw)
        return r
    except requests.RequestException as e:
        print(f"[UNREACHABLE] {label}: {type(e).__name__}: {str(e)[-160:]}")
        return None


def parquet(label: str, url: str, dest: str):
    try:
        df = pd.read_parquet(pit._fetch(url, pit.CACHE_DIR / dest))
    except Exception as e:                                    # noqa: BLE001 (a finding, not a crash)
        print(f"[FAIL] {label}: {type(e).__name__}: {str(e)[:160]}")
        return None
    print(f"[OK] {label}: {len(df):,} rows x {df.shape[1]} cols")
    return df


def main() -> None:
    print("== nflverse release assets ==")
    for y in (2024, 2025):
        parquet(f"legacy player_stats_{y} (nfl_data_py's weekly source)", f"{NFL}/player_stats/player_stats_{y}.parquet",
                f"nflverse/player_stats_{y}.parquet")
    for y in (2024, 2025, 2026):
        parquet(f"stats_player_week_{y}", f"{NFL}/stats_player/stats_player_week_{y}.parquet",
                f"nflverse/stats_player_week_{y}.parquet")
    g = parquet("schedules games.parquet", f"{NFL}/schedules/games.parquet", "nflverse/games.parquet")
    if g is not None:
        g = g[g["season"] == 2024]
        print("   2024 non-null:", {c: round(float(g[c].notna().mean()), 3) for c in
                                   ("gameday", "gametime", "roof", "temp", "wind", "spread_line", "total_line")})
        print("   outdoor-game temp/wind non-null:",
              round(float(g.loc[g["roof"] == "outdoors", "temp"].notna().mean()), 3))
    parquet("snap_counts_2024", f"{NFL}/snap_counts/snap_counts_2024.parquet", "nflverse/snap_counts_2024.parquet")
    for y in (2024, 2025):
        d = parquet(f"injuries_{y}", f"{NFL}/injuries/injuries_{y}.parquet", f"nflverse/injuries_{y}.parquet")
        if d is not None:
            print("   has date_modified:", "date_modified" in d.columns)
    for y in (2024, 2025):
        d = parquet(f"depth_charts_{y}", f"{NFL}/depth_charts/depth_charts_{y}.parquet", f"nflverse/depth_charts_{y}.parquet")
        if d is not None:
            print("   has snapshot dt:", "dt" in d.columns, "| has week:", "week" in d.columns)
    parquet("combine", f"{NFL}/combine/combine.parquet", "nflverse/combine.parquet")
    parquet("draft_picks", f"{NFL}/draft_picks/draft_picks.parquet", "nflverse/draft_picks.parquet")
    parquet("players", f"{NFL}/players/players.parquet", "nflverse/players.parquet")

    print("\n== ffopportunity ==")
    x = parquet("ep_weekly_2024", f"{pit.FFOPP}/ep_weekly_2024.parquet", "ffopportunity/ep_weekly_2024.parquet")
    if x is not None:
        print("   xFP cols:", [c for c in x.columns if c.endswith("fantasy_points_exp")])

    print("\n== FantasyFootballCalculator ADP ==")
    r = attempt("FFC", "https://fantasyfootballcalculator.com/api/v1/adp/ppr?teams=12&year=2024")
    if r is not None:
        print(f"[HTTP {r.status_code}] content-type={r.headers.get('content-type')}")
        try:
            j = r.json()
            print("   top-level keys:", list(j)[:10], "| meta:", j.get("meta"))
            pl = j.get("players") or []
            print("   players:", len(pl), "| first:", pl[0] if pl else None)
        except ValueError:
            print("   body:", r.text[:200])

    print("\n== CollegeFootballData ==")
    key = os.environ.get("CFBD_KEY")
    r = attempt("CFBD", "https://api.collegefootballdata.com/player/usage?year=2024&position=WR",
                headers={"Authorization": f"Bearer {key}"} if key else {})
    if r is not None:
        print(f"[HTTP {r.status_code}] key={'set' if key else 'absent'} body: {r.text[:200]}")

    print("\n== Open-Meteo archive ==")
    r = attempt("Open-Meteo", "https://archive-api.open-meteo.com/v1/archive",
                params={"latitude": 39.0489, "longitude": -94.4839, "start_date": "2024-09-05",
                        "end_date": "2024-09-06", "hourly": "temperature_2m,wind_speed_10m", "timezone": "UTC"})
    if r is not None:
        print(f"[HTTP {r.status_code}]")
        try:
            j = r.json()
            print("   keys:", list(j), "| hourly keys:", list(j.get("hourly", {})), "| n:", len(j.get("hourly", {}).get("time", [])))
        except ValueError:
            print("   body:", r.text[:200])


if __name__ == "__main__":
    main()
