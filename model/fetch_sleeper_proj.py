"""Sleeper's historical weekly projections (company = Rotowire) -> a gated `sleeper_proj` table for the opt-in `market` family.

    python -m model.fetch_sleeper_proj                        # 2021-2026, weeks 1-18, cached per season-week
    python -m model.fetch_sleeper_proj --seasons 2025 --weeks 3-5 --force

Source: `GET https://api.sleeper.app/projections/nfl/{season}/{week}?season_type=regular&position[]=QB&...` (unauthenticated,
undocumented; every row says `company: rotowire`). Rows with a null `stats.pts_ppr` are dropped (Sleeper lists every rostered
player; most have no projection). One parquet per season-week under model/cache/sleeper_proj/ (git-ignored, immutable until
deleted or --force). A week is fetched only once its last game has kicked off + 4h, so a cached week is the archive's final
version and a rerun never freezes a mid-week snapshot.

known_at = THAT GAME'S KICKOFF - 1 HOUR (the leakage decision). Sleeper keeps one version per (player, week), the last one, and
publishes no publish time: `last_modified` / `updated_at` are null before 2023 and afterwards a batch stamp written AFTER the games
(the archive is re-saved post-week), so they say when the archive was written, not when the number was knowable. The number is a
pre-game projection that Rotowire revises up to inactives (~90 min before kickoff); kickoff - 1h is the last moment that final
pre-game version can be assumed to exist, and the row passes the gate for its own game (strict <) and every later one.

Use the VALUE, never its PRESENCE. The archive drops players who were scratched late (ruled out after the projection was zeroed
or pulled), so "has a projection row" is partly a post-hoc P(plays) signal. The family therefore has no has-a-projection flag;
missing rows are NaN like any other missing value. (A tree model can still split on NaN-ness; the trial scores played rows only,
where that channel carries little. Do not add a coverage/indicator column.)

`pos_rank` is computed here, within (season, week, position) over every projected player of that week, and carried on each
player's own row (stamped at his own kickoff - 1h). For a Thursday game that rank uses the archive's final numbers for Sunday's
players, which were revised after Thursday: declared as the table's proxy. Projections for the whole week are published early in
the week, so the rank (coarse) moves little.

Identity: Sleeper ids -> gsis ids, from three maps in priority order: data/players.csv (the pipeline's own crosswalk, current
players only), the Sleeper player dump's `gsis_id` (`/v1/players/nfl`), and DynastyProcess' db_playerids.csv. nflverse
players.parquet has no sleeper_id column. The map is written once to model/cache/sleeper_proj/id_map.parquet by the fetcher; the
loader reads only cached files (no network). Unmatched rows are counted in `RawTable.dropped`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from model import point_in_time as pit
from model.point_in_time import KNOWN_AT, RawTable

URL = ("https://api.sleeper.app/projections/nfl/{season}/{week}?season_type=regular&position[]=QB&position[]=RB"
       "&position[]=WR&position[]=TE&order_by=pts_half_ppr")
PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
DP_IDS_URL = "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv"
CACHE = pit.CACHE_DIR / "sleeper_proj"
REPO_PLAYERS = Path(__file__).resolve().parents[1] / "data" / "players.csv"
SEASONS = range(2021, 2027)
WEEKS = range(1, 19)
SLEEP = 0.3
KNOWN_LEAD = pd.Timedelta(hours=1)
SETTLED = pd.Timedelta(hours=4)             # a week is cached only once its last kickoff is this far in the past
STAT_COLS = ["pts_ppr", "pts_half_ppr", "pts_std", "pass_att", "pass_yd", "pass_td", "rush_att", "rush_yd", "rush_td",
             "rec", "rec_tgt", "rec_yd", "rec_td"]
RAW_COLS = ["season", "week", "sleeper_id", "position", "team", "opponent", "company", "last_modified", "updated_at", *STAT_COLS]
SLEEPER_TEAM_TO_NFLVERSE = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "OAK": "LV", "SD": "LAC"}
PROXY = ("pos_rank ranks the week's FINAL archived projections, so a Thursday row's rank uses Sunday players' numbers as revised "
         "after Thursday; the value itself is the last pre-game version, stamped kickoff - 1h (no publish time exists)")


# --------------------------------------------------------------------------- parsing (pure, tested on a fixture)
def _ms(x) -> pd.Timestamp:
    return pd.NaT if x is None or (isinstance(x, float) and np.isnan(x)) else pd.Timestamp(int(x), unit="ms", tz="UTC")


def parse_week(records: list[dict], season: int, week: int) -> pd.DataFrame:
    """One API response -> RAW_COLS rows, only those with a non-null stats.pts_ppr. `team` is the row's top-level team (the
    team he was projected FOR that week); `player.team` is his current team and is ignored. Position comes from the player
    block (Sleeper carries none at the row level)."""
    rows = []
    for r in records or []:
        st = r.get("stats") or {}
        if st.get("pts_ppr") is None:
            continue
        pl = r.get("player") or {}
        rows.append({"season": int(season), "week": int(week), "sleeper_id": str(r.get("player_id")),
                     "position": pl.get("position"), "team": r.get("team"), "opponent": r.get("opponent"),
                     "company": r.get("company"), "last_modified": _ms(r.get("last_modified")),
                     "updated_at": _ms(r.get("updated_at")),
                     **{c: (float(st[c]) if st.get(c) is not None else np.nan) for c in STAT_COLS}})
    df = pd.DataFrame(rows, columns=RAW_COLS)
    for c in ("last_modified", "updated_at"):
        df[c] = pd.to_datetime(df[c], utc=True).astype("datetime64[ns, UTC]")
    for c in STAT_COLS:
        df[c] = df[c].astype("float64")
    df["season"] = df["season"].astype("int64")
    df["week"] = df["week"].astype("int64")
    return df


# --------------------------------------------------------------------------- id map
def build_id_map(repo_players: pd.DataFrame | None, sleeper_dump: dict | None, dp_ids: pd.DataFrame | None) -> pd.DataFrame:
    """[sleeper_id, gsis_id, id_source], one row per sleeper_id; earlier sources win."""
    parts = []
    if repo_players is not None and len(repo_players):
        parts.append(repo_players[["sleeper_id", "gsis_id"]].assign(id_source="data/players.csv"))
    if sleeper_dump:
        parts.append(pd.DataFrame([(str(k), (v or {}).get("gsis_id")) for k, v in sleeper_dump.items()],
                                  columns=["sleeper_id", "gsis_id"]).assign(id_source="sleeper_players"))
    if dp_ids is not None and len(dp_ids):
        parts.append(dp_ids[["sleeper_id", "gsis_id"]].assign(id_source="dynastyprocess"))
    if not parts:
        return pd.DataFrame(columns=["sleeper_id", "gsis_id", "id_source"])
    m = pd.concat(parts, ignore_index=True)
    m["sleeper_id"] = m["sleeper_id"].astype("string").str.strip()
    m["gsis_id"] = m["gsis_id"].astype("string").str.strip()
    m = m[m["sleeper_id"].notna() & m["gsis_id"].str.match(r"^00-\d{7}$").fillna(False)]
    m = m[m["sleeper_id"].str.fullmatch(r"\d+").fillna(False)]
    return m.drop_duplicates("sleeper_id", keep="first").astype(str).reset_index(drop=True)


def fetch_id_map(force: bool = False) -> pd.DataFrame:
    dest = CACHE / "id_map.parquet"
    if dest.exists() and not force:
        return pd.read_parquet(dest)
    repo = pd.read_csv(REPO_PLAYERS, dtype=str) if REPO_PLAYERS.exists() else None
    dump = requests.get(PLAYERS_URL, timeout=120).json()
    from io import StringIO
    r = requests.get(DP_IDS_URL, timeout=120)
    dp = pd.read_csv(StringIO(r.text), dtype=str) if r.ok else None
    m = build_id_map(repo, dump, dp)
    dest.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(dest, index=False)
    return m


# --------------------------------------------------------------------------- fetching
def _path(season: int, week: int) -> Path:
    return CACHE / f"proj_{season}_{week:02d}.parquet"


def _settled_weeks(now: pd.Timestamp) -> set[tuple[int, int]]:
    g = pit._games()
    g = g[g["game_type"] == "REG"]
    last = g.groupby(["season", "week"])["kickoff"].max()
    return {k for k, ko in last.items() if ko + SETTLED < now}


def fetch(seasons: Iterable[int] = SEASONS, weeks: Iterable[int] = WEEKS, *, force: bool = False,
          session: requests.Session | None = None) -> dict:
    s = session or requests.Session()
    settled = _settled_weeks(pd.Timestamp.now(tz="UTC"))
    log = {"fetched": 0, "cached": 0, "not_settled": 0, "rows": {}}
    for season in seasons:
        for week in weeks:
            dest = _path(season, week)
            if (season, week) not in settled:
                log["not_settled"] += 1
                continue
            if dest.exists() and not force:
                log["cached"] += 1
                continue
            r = s.get(URL.format(season=season, week=week), timeout=60)
            r.raise_for_status()
            df = parse_week(r.json(), season, week)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".parquet.part")
            df.to_parquet(tmp, index=False)
            tmp.replace(dest)
            log["fetched"] += 1
            log["rows"][(season, week)] = len(df)
            time.sleep(SLEEP)
    return log


# --------------------------------------------------------------------------- the table
def attach_known_at(raw: pd.DataFrame, games: pd.DataFrame, id_map: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """raw rows -> gated rows: gsis `player_id`, nflverse team codes, the game's id and kickoff, known_at = kickoff - 1h,
    pos_rank within (season, week, position). Rows with no gsis id or no REG fixture for (season, week, team) are dropped and
    counted. `games` is nflverse's schedule with a UTC `kickoff` (point_in_time._games())."""
    df = raw.copy()
    n0 = len(df)
    df["team"] = df["team"].replace(SLEEPER_TEAM_TO_NFLVERSE)
    df["opponent"] = df["opponent"].replace(SLEEPER_TEAM_TO_NFLVERSE)
    df = df[df["position"].isin(pit.SKILL_POSITIONS)]
    n_pos = n0 - len(df)
    # rank among every projected player of the position that week (before the id/fixture filters: the market's own view)
    df["pos_rank"] = df.groupby(["season", "week", "position"])["pts_ppr"].rank(method="min", ascending=False)
    ids = id_map.drop_duplicates("sleeper_id").set_index("sleeper_id")["gsis_id"]
    df["player_id"] = df["sleeper_id"].astype(str).map(ids)
    n_noid = int(df["player_id"].isna().sum())
    df = df[df["player_id"].notna()]
    g = games[games["game_type"] == "REG"]
    tw = pd.concat([g[["season", "week", "home_team", "game_id", "kickoff"]].rename(columns={"home_team": "team"}),
                    g[["season", "week", "away_team", "game_id", "kickoff"]].rename(columns={"away_team": "team"})])
    df = df.merge(tw, on=["season", "week", "team"], how="left")
    n_nofx = int(df["kickoff"].isna().sum())
    df = df[df["kickoff"].notna()].copy()
    # one row per (player, week); a duplicate would be a Sleeper id collision after mapping: keep the higher projection
    n_dup = int(df.duplicated(["player_id", "season", "week"]).sum())
    df = df.sort_values(["player_id", "season", "week", "pts_ppr"], ascending=[True, True, True, False]) \
           .drop_duplicates(["player_id", "season", "week"])
    df[KNOWN_AT] = pd.to_datetime(df["kickoff"] - KNOWN_LEAD, utc=True).astype("datetime64[ns, UTC]")
    df = df.sort_values([KNOWN_AT, "player_id"], kind="mergesort").reset_index(drop=True)
    return df, {"not_skill_position": n_pos, "no_gsis_id": n_noid, "no_reg_fixture": n_nofx, "duplicate_after_mapping": n_dup}


def _raw(seasons: Iterable[int]) -> pd.DataFrame:
    frames = [pd.read_parquet(f) for y in seasons for f in sorted(CACHE.glob(f"proj_{y}_*.parquet"))]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RAW_COLS)


def load_sleeper_proj(seasons: Iterable[int], *, with_dropped: bool = False):
    """The cached projections for `seasons` with `player_id` (gsis) and `known_at` = kickoff - 1h. Reads only the cache."""
    raw = _raw(list(seasons))
    idp = CACHE / "id_map.parquet"
    id_map = pd.read_parquet(idp) if idp.exists() else pd.DataFrame(columns=["sleeper_id", "gsis_id", "id_source"])
    df, dropped = attach_known_at(raw, pit._games(), id_map)
    return (df, dropped) if with_dropped else df


def load_sleeper_proj_table(seasons: Iterable[int]) -> RawTable | None:
    """The RawTable for the store, or None when nothing is cached (the market columns are then NaN)."""
    df, dropped = load_sleeper_proj(seasons, with_dropped=True)
    if df.empty:
        return None
    return RawTable("sleeper_proj", df, "game kickoff - 1h (last pre-game version; Sleeper's stamps are post-game batch writes)",
                    proxy=PROXY, dropped=dropped)


# --------------------------------------------------------------------------- CLI
def _parse(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return sorted(set(out))


def match_report(seasons: Iterable[int]) -> pd.DataFrame:
    raw = _raw(list(seasons))
    raw = raw[raw["position"].isin(pit.SKILL_POSITIONS)]
    m = pd.read_parquet(CACHE / "id_map.parquet")
    src = m.set_index("sleeper_id")["id_source"]
    raw = raw.assign(src=raw["sleeper_id"].astype(str).map(src))
    out = raw.groupby("season").agg(rows=("sleeper_id", "size"), weeks=("week", "nunique"),
                                    matched=("src", lambda s: s.notna().mean()))
    for s in ("data/players.csv", "sleeper_players", "dynastyprocess"):
        out[s] = raw.groupby("season")["src"].apply(lambda x, s=s: int((x == s).sum()))
    pts = raw.assign(m=raw["src"].notna()).groupby("season").apply(
        lambda g: (g["pts_ppr"] * g["m"]).sum() / g["pts_ppr"].sum(), include_groups=False)
    out["matched_pts_weighted"] = pts
    return out.round(4)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", default="2021-2026")
    ap.add_argument("--weeks", default="1-18")
    ap.add_argument("--force", action="store_true", help="refetch cached weeks (and the id map)")
    a = ap.parse_args(argv)
    seasons = _parse(a.seasons)
    fetch_id_map(force=a.force)
    log = fetch(seasons, _parse(a.weeks), force=a.force)
    print(f"fetched {log['fetched']}, cached {log['cached']}, not yet settled {log['not_settled']}")
    print(match_report(seasons).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
