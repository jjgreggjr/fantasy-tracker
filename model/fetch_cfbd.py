"""Fetch final-college-season stat lines for NFL draft classes from CollegeFootballData.

    CFBD_KEY=... python -m model.fetch_cfbd --seasons 2020-2025 [--out model/data/cfbd]

`--seasons` are COLLEGE seasons: the class drafted in year Y played its last college season in Y-1
(2021-2026 draft classes -> 2020-2025). Per season it writes, under model/data/cfbd/:

  player_stats_{season}.csv  CFBD /stats/player/season rows (passing, rushing, receiving) kept only for
                             players whose normalised name matches a QB/RB/WR/TE drafted the next April
                             (over-inclusive on purpose: name is the only join key we have; the loader will
                             disambiguate by college), with fetched_at
  team_stats_{season}.csv    CFBD /stats/season rows for every team (the denominators for market share)
  fetch_log.csv              one row per request: HTTP status, rows, and the exact error text

The key comes ONLY from the environment (a GitHub Actions secret; never a workflow input, never in the
repo). It is sent as `Authorization: Bearer`, never printed, never put in a URL, and never written to the
log: error text is `type(e).__name__` plus the exception message, which carries the URL but not headers.
No key -> every request is logged as skipped and the process still exits 0, like fetch_adp.

CFBD's response schema was never observed from a machine that could reach it (the sandbox is blocked), so
rows are stored as received (`pd.json_normalize`) and nothing here interprets stat names; model/college.py
does, and refuses to guess if the names it needs are absent.
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from model.adp import norm_name
from model.fetch_adp import parse_seasons

API = "https://api.collegefootballdata.com"
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "cfbd"
CATEGORIES = ("passing", "rushing", "receiving")
RETRY_STATUS = {429, 500, 502, 503, 504}
SKILL = {"QB", "RB", "WR", "TE"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_json(path: str, params: dict, key: str | None, retries: int = 4, sleep=time.sleep):
    """(payload|None, log dict). Retries connection errors and 429/5xx only."""
    log = {"path": path, "params": "&".join(f"{k}={v}" for k, v in sorted(params.items())), "http_status": "",
           "rows": 0, "ok": False, "error": "", "fetched_at": _now()}
    if not key:
        log["error"] = "CFBD_KEY not set: request skipped"
        return None, log
    for attempt in range(retries):
        try:
            r = requests.get(f"{API}{path}", params=params, timeout=90,
                             headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
        except requests.RequestException as e:
            log["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            sleep(2 ** attempt)
            continue
        log["http_status"] = r.status_code
        if r.status_code in RETRY_STATUS:
            log["error"] = f"HTTP {r.status_code}"
            sleep(2 ** attempt)
            continue
        if r.status_code != 200:
            log["error"] = f"HTTP {r.status_code}: {r.text[:200]!r}"
            return None, log
        try:
            payload = r.json()
        except ValueError:
            log["error"] = f"HTTP 200 but not JSON: {r.text[:200]!r}"
            return None, log
        log.update(ok=True, error="", rows=len(payload) if isinstance(payload, list) else 1)
        return payload, log
    return None, log


def draft_class_names(draft_year: int) -> set[str]:
    """Normalised names of the QB/RB/WR/TE drafted in `draft_year` (nflverse draft_picks release asset)."""
    from model import point_in_time as pit
    dp = pit._nflverse("draft_picks", "draft_picks.parquet")
    dp = dp[(dp["season"] == draft_year) & dp["position"].isin(SKILL)]
    return {norm_name(n) for n in dp["pfr_player_name"].dropna()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", default="2020-2025", help="college seasons (draft year - 1)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    key = os.environ.get("CFBD_KEY") or None
    a.out.mkdir(parents=True, exist_ok=True)
    logs, wrote = [], 0
    for season in parse_seasons(a.seasons):
        names = draft_class_names(season + 1)
        frames = []
        for cat in CATEGORIES:
            payload, log = get_json("/stats/player/season", {"year": season, "category": cat}, key)
            if payload:
                df = pd.json_normalize(payload)
                if "player" in df.columns:
                    df = df[df["player"].map(norm_name).isin(names)]
                log["rows_kept"] = len(df)
                frames.append(df)
            logs.append(log)
            print(f"{season} player {cat:9s} ok={log['ok']!s:5s} rows={log['rows']} kept={log.get('rows_kept', '')} {log['error']}")
            time.sleep(0.5)
        if frames:
            out = pd.concat(frames, ignore_index=True)
            out["fetched_at"] = _now()
            out.to_csv(a.out / f"player_stats_{season}.csv", index=False)
            wrote += 1
        payload, log = get_json("/stats/season", {"year": season}, key)
        if payload:
            t = pd.json_normalize(payload)
            t["fetched_at"] = _now()
            t.to_csv(a.out / f"team_stats_{season}.csv", index=False)
            wrote += 1
        logs.append(log)
        print(f"{season} team stats ok={log['ok']!s:5s} rows={log['rows']} {log['error']}")
        time.sleep(0.5)
    pd.DataFrame(logs).to_csv(a.out / "fetch_log.csv", index=False)
    print(f"{wrote} CSVs written to {a.out} (key {'present' if key else 'ABSENT'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
