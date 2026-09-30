"""Fetch FantasyFootballCalculator ADP (one CSV per season x format) and log exactly what happened.

    python -m model.fetch_adp --seasons 2021-2026 --formats ppr,standard [--out model/data/adp]

fantasyfootballcalculator.com is blocked by the Claude sandbox's egress policy (403 on the proxy
tunnel), so this runs in GitHub Actions (.github/workflows/model_fetch.yml). The FFC response schema
was NEVER observed in Phase 0, so the parser is schema-tolerant: it flattens whatever `players` rows
come back, keeps every `meta` field as a `meta_*` column (the drafts date window matters: see
model/adp.py), and stamps `fetched_at`. Every attempt, success or not, lands in `fetch_log.csv`
with the exact exception/HTTP status, so a blocked host is a recorded finding, not a silent gap.

Never raises for a network problem: exit code 0 always, so the workflow reaches its commit step and
the failure text gets committed. Only local programming errors (bad CLI args) exit non-zero.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

HOSTS = ("https://fantasyfootballcalculator.com", "https://www.fantasyfootballcalculator.com")
UA = "fantasy-tracker-model/1.0 (+https://github.com/jjgreggjr/fantasy-tracker)"
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "adp"
RETRY_STATUS = {429, 500, 502, 503, 504}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_seasons(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def players_frame(payload: dict, season: int, fmt: str, url: str, fetched_at: str) -> pd.DataFrame:
    """Flatten an FFC response. Every key of every player row is kept; `meta` fields become
    `meta_<key>` columns repeated on each row. Returns an empty frame when there are no players."""
    players = payload.get("players") or []
    df = pd.json_normalize(players) if players else pd.DataFrame()
    if df.empty:
        return df
    meta = payload.get("meta") or {}
    for k, v in meta.items():
        df[f"meta_{k}"] = json.dumps(v) if isinstance(v, (dict, list)) else v
    df.insert(0, "season", season)
    df.insert(1, "format", fmt)
    df["fetched_at"] = fetched_at
    df["source_url"] = url
    return df


def fetch_one(season: int, fmt: str, teams: int = 12, retries: int = 4, sleep=time.sleep):
    """(frame|None, log_row). Tries each host; retries only on connection errors and 429/5xx."""
    last_err, last_status, last_url = "", "", ""
    for host in HOSTS:
        url = f"{host}/api/v1/adp/{fmt}"
        params = {"teams": teams, "year": season, "position": "all"}
        for attempt in range(retries):
            last_url = f"{url}?teams={teams}&year={season}&position=all"
            try:
                r = requests.get(url, params=params, timeout=45, headers={"User-Agent": UA, "Accept": "application/json"})
            except requests.RequestException as e:
                last_err, last_status = f"{type(e).__name__}: {str(e)[:300]}", ""
                sleep(2 ** attempt)          # network errors can be transient: 1, 2, 4, 8 s
                continue
            last_status = r.status_code
            if r.status_code in RETRY_STATUS:
                last_err = f"HTTP {r.status_code}"
                sleep(2 ** attempt)
                continue
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}: {r.text[:200]!r}"
                break                        # a real refusal: try the other host, do not hammer
            fetched_at = _now()
            try:
                payload = r.json()
            except ValueError:
                last_err = f"HTTP 200 but not JSON: {r.text[:200]!r}"
                break
            df = players_frame(payload, season, fmt, last_url, fetched_at)
            meta = payload.get("meta") or {}
            sample = (payload.get("players") or [None])[0]
            log = {"season": season, "format": fmt, "url": last_url, "http_status": 200, "ok": len(df) > 0,
                   "n_players": len(df), "error": "" if len(df) else f"no players; status={payload.get('status')!r}",
                   "fetched_at": fetched_at, "top_level_keys": json.dumps(list(payload)),
                   "meta": json.dumps(meta), "player_keys": json.dumps(list(sample) if isinstance(sample, dict) else [])}
            return (df if len(df) else None), log
    return None, {"season": season, "format": fmt, "url": last_url, "http_status": last_status, "ok": False,
                  "n_players": 0, "error": last_err, "fetched_at": _now(), "top_level_keys": "", "meta": "",
                  "player_keys": ""}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", default="2021-2026")
    ap.add_argument("--formats", default="ppr,standard")
    ap.add_argument("--teams", type=int, default=12)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    logs, ok = [], 0
    for season in parse_seasons(a.seasons):
        for fmt in [f.strip() for f in a.formats.split(",") if f.strip()]:
            df, log = fetch_one(season, fmt, a.teams)
            logs.append(log)
            if df is not None:
                df.to_csv(a.out / f"adp_{fmt}_{season}.csv", index=False)
                ok += 1
            print(f"{season} {fmt:9s} ok={log['ok']!s:5s} n={log['n_players']:4d} status={log['http_status']} {log['error']}")
            time.sleep(1.0)          # be polite: 12 requests total
    pd.DataFrame(logs).to_csv(a.out / "fetch_log.csv", index=False)
    print(f"{ok}/{len(logs)} season-format files written to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
