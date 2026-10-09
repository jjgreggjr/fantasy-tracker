"""Download and cache raw data from nflverse (GitHub releases) and Sleeper."""
from __future__ import annotations

import json
from datetime import datetime, timezone
import logging
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
SLEEPER = "https://api.sleeper.app/v1"
UA = {"User-Agent": "ff-volume-tracker/1.0 (personal use)"}
TIMEOUT = 120


# --------------------------------------------------------------------------
# nflverse
# --------------------------------------------------------------------------
# nflverse has renamed a published asset before (schedules/games.csv became games.csv.gz overnight and
# killed the Oct 6 run). The same table is published under several encodings, so when the name we ask
# for is gone we try the siblings before giving up. pandas reads all three by extension.
_VARIANT_SUFFIXES = (".csv", ".csv.gz", ".parquet")


def asset_variants(filename: str) -> list[str]:
    """`filename` first, then the same asset under the other encodings (.csv, .csv.gz, .parquet)."""
    for suf in sorted(_VARIANT_SUFFIXES, key=len, reverse=True):     # ".csv.gz" before ".csv"
        if filename.endswith(suf):
            stem = filename[: -len(suf)]
            break
    else:
        return [filename]
    return [filename] + [stem + s for s in _VARIANT_SUFFIXES if stem + s != filename]


def _read_asset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)        # needs pyarrow; the model's install has it, the base pipeline's does not
    return pd.read_csv(path, low_memory=False)


def nflverse_csv(release: str, filename: str, raw_dir: Path,
                 required: bool = True) -> pd.DataFrame | None:
    """Download an nflverse table, caching by ETag. Returns None when it is not published.

    A 404 is normal early in a season for files that only exist once games
    have been played (stats_player_week, snap_counts). It is also what an
    upstream rename looks like, so a 404 sends us to the same asset under its
    other extensions (.csv -> .csv.gz -> .parquet) before we conclude it is
    missing. Which variant served is logged; anything but the one asked for is
    a WARNING so a rename shows up in the run log instead of as a dead week.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    variants = asset_variants(filename)
    for name in variants:
        try:
            df = _fetch_one(release, name, raw_dir)
        except requests.RequestException as e:
            # the network, not the file: another extension will not help
            if required:
                raise
            log.warning("%s unreachable and no cache: %s", filename, e)
            return None
        except ImportError as e:            # a .parquet variant on an install without pyarrow
            log.warning("%s unreadable here (%s); trying the next encoding", name, e)
            continue
        if df is None:                      # 404: try the next encoding
            continue
        if name != filename:
            log.warning("%s is not published; %s served instead (upstream renamed or "
                        "re-encoded the asset)", filename, name)
        return df

    msg = f"{filename} not published yet (404; also tried {', '.join(variants[1:])})"
    if required:
        raise FileNotFoundError(msg)
    log.warning(msg)
    return None


def _fetch_one(release: str, filename: str, raw_dir: Path) -> pd.DataFrame | None:
    """One concrete asset name. None on 404. A dead network falls back to the
    cached copy of this name, or raises requests.RequestException if there is none."""
    dest = raw_dir / filename
    etag_file = dest.with_suffix(dest.suffix + ".etag")

    headers = dict(UA)
    if dest.exists() and etag_file.exists():
        headers["If-None-Match"] = etag_file.read_text().strip()

    url = f"{NFLVERSE}/{release}/{filename}"
    try:
        r = requests.get(url, headers=headers, timeout=TIMEOUT, stream=True)
    except requests.RequestException as e:
        if dest.exists():
            log.warning("%s unreachable (%s); using cached copy", filename, e)
            return _read_asset(dest)
        raise

    if r.status_code == 304 and dest.exists():
        log.info("%s unchanged (cached)", filename)
        return _read_asset(dest)

    if r.status_code == 404:
        return None

    r.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    tmp.replace(dest)
    if r.headers.get("ETag"):
        etag_file.write_text(r.headers["ETag"])
    log.info("%s downloaded (%.1f MB)", filename, dest.stat().st_size / 1e6)
    return _read_asset(dest)


def load_nflverse(season: int, raw_dir: Path, with_prior: bool = True) -> dict:
    """Fetch every nflverse file the pipeline needs.

    stats/snaps for the current season are optional: they do not exist until
    the season's first games have been played. injuries is optional too: it
    gains a week's rows only when that week's first report is issued.
    """
    out = {
        # schedules/games.csv is gone upstream (404 on 2026-10-07, which crashed the
        # weekly run); games.csv.gz has the same 46 columns and pandas reads it by extension.
        "games": nflverse_csv("schedules", "games.csv.gz", raw_dir),
        "roster": nflverse_csv("rosters", f"roster_{season}.csv", raw_dir),
        "depth": nflverse_csv("depth_charts", f"depth_charts_{season}.csv", raw_dir),
        "stats": nflverse_csv("stats_player", f"stats_player_week_{season}.csv",
                              raw_dir, required=False),
        "snaps": nflverse_csv("snap_counts", f"snap_counts_{season}.csv",
                              raw_dir, required=False),
        # Weekly injury reports (practice participation + game designation), one row per
        # player-week, appearing as the week's reports are issued (Wednesday on).
        "injuries": nflverse_csv("injuries", f"injuries_{season}.csv",
                                 raw_dir, required=False),
    }
    if with_prior:
        p = season - 1
        out["stats_prior"] = nflverse_csv(
            "stats_player", f"stats_player_week_{p}.csv", raw_dir, required=False)
        out["snaps_prior"] = nflverse_csv(
            "snap_counts", f"snap_counts_{p}.csv", raw_dir, required=False)
        out["roster_prior"] = nflverse_csv(
            "rosters", f"roster_{p}.csv", raw_dir, required=False)
    return out


# --------------------------------------------------------------------------
# Sleeper
# --------------------------------------------------------------------------
class SleeperError(RuntimeError):
    pass


def _get(path: str):
    url = f"{SLEEPER}/{path}"
    r = requests.get(url, headers=UA, timeout=TIMEOUT)
    if r.status_code == 404:
        raise SleeperError(f"Sleeper 404 for {path}")
    r.raise_for_status()
    return r.json()


def sleeper_state() -> dict:
    return _get("state/nfl")


def sleeper_players(raw_dir: Path, today: str) -> dict:
    """~5 MB player dictionary. Sleeper asks this be pulled at most daily."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / f"players_{today}.json"
    if cache.exists():
        log.info("using cached Sleeper players for %s", today)
        return json.loads(cache.read_text(encoding="utf-8"))
    data = _get("players/nfl")
    cache.write_text(json.dumps(data), encoding="utf-8")
    for old in sorted(raw_dir.glob("players_*.json"))[:-3]:
        old.unlink()  # keep only the 3 most recent daily snapshots
    return data


def sleeper_leagues(username: str, season: int) -> tuple[str, list[dict]]:
    user = _get(f"user/{username.lstrip('@')}")
    if not user or "user_id" not in user:
        raise SleeperError(f"Sleeper user '{username}' not found")
    uid = user["user_id"]
    return uid, _get(f"user/{uid}/leagues/nfl/{season}")


def sleeper_transactions(league_id: str, week: int) -> list:
    """Adds, drops, waiver claims and trades for one week ("round")."""
    try:
        return _get(f"league/{league_id}/transactions/{week}")
    except Exception as e:
        log.warning("transactions wk%s for %s unavailable: %s", week, league_id, e)
        return []


def sleeper_matchups(league_id: str, week: int) -> list:
    """Each roster's starters, players and points for one week. Frozen once the
    week is played. For the week in progress it is only the current lineup
    with zero points, so ask for completed weeks only."""
    return _get(f"league/{league_id}/matchups/{week}")


def sleeper_trending(kind: str = "add", hours: int = 24, limit: int = 50) -> list:
    """Most-added or most-dropped players across all of Sleeper."""
    try:
        return _get(f"players/nfl/trending/{kind}?lookback_hours={hours}&limit={limit}")
    except Exception as e:
        log.warning("trending/%s unavailable: %s", kind, e)
        return []


def sleeper_league_detail(league_id: str) -> dict:
    out = {
        "league": _get(f"league/{league_id}"),
        "rosters": _get(f"league/{league_id}/rosters"),
        "users": _get(f"league/{league_id}/users"),
    }
    # When the lineup was actually read. Nothing else records this; the
    # `pulled_at` that reaches roster.csv is the projections fetch time.
    out["fetched_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return out
