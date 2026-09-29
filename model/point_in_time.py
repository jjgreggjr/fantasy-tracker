"""Raw tables with explicit availability timestamps, and the one gate to them.

The leakage rule (PLAN_MODEL.md): every feature for a (player, game) row must be
knowable strictly before that game's kickoff. Enforced structurally:

  1. Every raw table below carries a `known_at` column: the earliest moment the
     row's information could have been in our hands. The loaders are the only
     place `known_at` is derived, and each derivation is written next to the
     table (and surfaced by `RawStore.describe()`).
  2. `as_of_join(store, table, kickoff_ts, **keys)` is the only path from a raw
     table to feature code. It returns rows with `known_at < kickoff_ts`
     (strictly), filtered on equality keys. `RawStore` keeps its frames private;
     `features.py` receives a store and can do nothing with it except call the
     gate. Anything else is a visible bypass (and `tests/test_no_leakage.py`
     fails the build on one).
  3. `RawStore.truncated_before(ts)` builds a physical copy of every table cut to
     `known_at < ts`. The audit rebuilds features from it and demands identical
     output, so a feature that reaches past the gate (or a gate bug) shows up as
     a diff, not as an unexplained backtest score.

Read `PLAN_MODEL.md` "Phase 0 findings" for the per-source evidence behind each
rule. Loaders read nflverse / ffopportunity release assets directly (not via
nfl_data_py; see model/requirements.txt) and cache them under model/cache/, which
is git-ignored and immutable until deleted (a rerun reproduces every number).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

KNOWN_AT = "known_at"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
FFOPP = "https://github.com/ffverse/ffopportunity/releases/download/latest-data"

# ---- availability policy: the only tunables, each justified in PLAN_MODEL.md ----
# Season-static data (draft capital, combine, player bio) counts as known one week
# before the opener, so it is never withheld from the opener itself.
SEASON_START_LEAD = timedelta(days=7)
# nflverse spread_line/total_line are closing lines; no close timestamp is published.
LINES_CLOSE_LEAD = timedelta(minutes=60)
# v1 weather PROXY: observed (post-game) temp/wind stand in for a perfect forecast
# issued a day ahead. Declared, not hidden: RawTable.proxy carries the caveat.
WEATHER_PROXY_LEAD = timedelta(hours=24)
# nflverse injuries >= 2025 dropped `date_modified`. In 2024 the report landed a
# median 47h (p1 = 23.9h) before kickoff, so a row with no timestamp is treated as
# known 24h before kickoff: later than reality, hence conservative.
INJURY_DERIVED_LEAD = timedelta(hours=24)
# Playoff pairings are not knowable at season start; treat as known 24h ahead.
NON_REG_FIXTURE_LEAD = timedelta(hours=24)

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
# Offense labels shared by both depth-chart formats (drops defense, KR/PR, specialists).
OFFENSE_DEPTH_POS = ("QB", "RB", "HB", "FB", "F", "WR", "TE", "LT", "LG", "C", "RG", "RT")

# Static tables are whitelisted column-by-column: the raw files carry career
# outcomes (w_av, car_av, games, ...) and mutable "current" fields (status,
# latest_team, years_of_experience) that would leak the future into a past row.
PLAYER_COLS = ["gsis_id", "pfr_id", "espn_id", "display_name", "birth_date", "position",
               "height", "weight", "college_name", "rookie_season", "draft_year",
               "draft_round", "draft_pick", "draft_team"]
DRAFT_COLS = ["season", "round", "pick", "team", "gsis_id", "pfr_player_id",
              "pfr_player_name", "position", "age", "college"]
COMBINE_COLS = ["season", "pfr_id", "player_name", "pos", "school", "ht", "wt", "forty",
                "bench", "vertical", "broad_jump", "cone", "shuttle"]


# --------------------------------------------------------------------------- tables
@dataclass(frozen=True)
class RawTable:
    """A frame plus the contract that makes it safe to gate."""
    name: str
    df: pd.DataFrame
    known_at_rule: str
    proxy: str | None = None          # non-None: known_at is a documented approximation
    dropped: dict = field(default_factory=dict)   # {reason: n_rows} removed by the loader

    def __post_init__(self):
        if KNOWN_AT not in self.df.columns:
            raise ValueError(f"{self.name}: no `{KNOWN_AT}` column")
        col = self.df[KNOWN_AT]
        if str(col.dtype) != "datetime64[ns, UTC]":
            raise ValueError(f"{self.name}: known_at must be datetime64[ns, UTC], got {col.dtype}")
        if col.isna().any():
            raise ValueError(f"{self.name}: {int(col.isna().sum())} rows have no known_at; "
                             "the loader must drop them explicitly (and count them in `dropped`)")

    def with_df(self, df: pd.DataFrame) -> "RawTable":
        return RawTable(self.name, df, self.known_at_rule, self.proxy, dict(self.dropped))


class RawStore:
    """All raw tables for one build. Frames are private on purpose: the way in is
    `as_of_join`. (`_tables` is reachable by design for tests and target lookup;
    feature code touching it is a bug the AST test catches.)"""

    def __init__(self, tables: Iterable[RawTable]):
        self._tables: dict[str, RawTable] = {t.name: t for t in tables}

    def names(self) -> list[str]:
        return list(self._tables)

    def describe(self) -> pd.DataFrame:
        rows = []
        for t in self._tables.values():
            rows.append({"table": t.name, "rows": len(t.df),
                         "known_at_min": t.df[KNOWN_AT].min(), "known_at_max": t.df[KNOWN_AT].max(),
                         "rule": t.known_at_rule, "proxy": t.proxy or "",
                         "dropped": dict(t.dropped)})
        return pd.DataFrame(rows)

    def truncated_before(self, ts) -> "RawStore":
        """Physical copy of every table cut to known_at < ts (the audit's oracle)."""
        ts = _utc(ts)
        return RawStore(t.with_df(t.df[t.df[KNOWN_AT] < ts].copy()) for t in self._tables.values())

    def with_frame(self, name: str, df: pd.DataFrame) -> "RawStore":
        """Copy of the store with one table's frame replaced. For canary/audit tests."""
        tables = dict(self._tables)
        tables[name] = tables[name].with_df(df)
        return RawStore(tables.values())


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError(f"naive timestamp {ts!r}: the gate only compares tz-aware instants")
    return ts.tz_convert("UTC")


def _ns_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).astype("datetime64[ns, UTC]")


# --------------------------------------------------------------------------- the gate
def as_of_join(store: RawStore, table: str, kickoff_ts, *, trace: list | None = None,
               **keys) -> pd.DataFrame:
    """THE gate. Rows of `table` with known_at < kickoff_ts (strict), restricted by
    equality (scalar) or membership (list/tuple/set) on `keys`, oldest known first.

    Returns a copy: mutate it freely. If `trace` is a list, one audit record per
    call is appended (table, rows, max known_at, game_ids) so tests can prove what a
    feature row actually consumed.
    """
    kickoff = _utc(kickoff_ts)
    df = store._tables[table].df
    mask = df[KNOWN_AT] < kickoff
    for col, val in keys.items():
        if col not in df.columns:
            raise KeyError(f"{table} has no column {col!r}")
        if val is None:
            raise ValueError(f"as_of_join({table}, {col}=None): refusing an ambiguous key")
        mask &= df[col].isin(list(val)) if isinstance(val, (list, tuple, set, frozenset)) else df[col] == val
    out = df[mask].sort_values(KNOWN_AT, kind="mergesort").reset_index(drop=True)
    if trace is not None:
        trace.append({"table": table, "rows": len(out), "kickoff": kickoff,
                      "max_known_at": out[KNOWN_AT].max() if len(out) else None,
                      "game_ids": set(out["game_id"]) if "game_id" in out.columns else set()})
    return out


@dataclass(frozen=True)
class TargetRow:
    """Identity of the row we build features for. Carries no target-week stats."""
    player_id: str
    season: int
    week: int
    team: str
    opponent: str
    position: str
    game_id: str
    kickoff: pd.Timestamp     # UTC
    is_home: bool


def make_target(store: RawStore, player_id: str, season: int, week: int) -> TargetRow:
    """Spine lookup: who is this row and when is kickoff. Reads identity only (team and
    position from the target week's own stats or injury row, kickoff from the fixture).
    It never returns a stat, so it is not a feature path, but it is the one function
    allowed to look at target-week rows."""
    pg = store._tables["player_games"].df
    hit = pg[(pg["player_id"] == player_id) & (pg["season"] == season) & (pg["week"] == week)]
    if len(hit):
        team, pos = hit.iloc[0]["team"], hit.iloc[0]["position"]
    else:                                  # e.g. ruled Out: no stats row, but an injury row
        inj = store._tables["injuries"].df
        hit = inj[(inj["player_id"] == player_id) & (inj["season"] == season) & (inj["week"] == week)]
        if not len(hit):
            raise LookupError(f"no stats or injury row for {player_id} {season} wk{week}")
        team, pos = hit.iloc[0]["team"], hit.iloc[0]["position"]
    fx = store._tables["fixtures"].df
    g = fx[(fx["season"] == season) & (fx["week"] == week)
           & ((fx["home_team"] == team) | (fx["away_team"] == team))]
    if len(g) != 1:
        raise LookupError(f"{team} {season} wk{week}: {len(g)} fixtures")
    g = g.iloc[0]
    home = g["home_team"] == team
    return TargetRow(player_id, int(season), int(week), team,
                     g["away_team"] if home else g["home_team"], pos, g["game_id"],
                     g["kickoff"], bool(home))


# --------------------------------------------------------------------------- loaders
def _fetch(url: str, dest: Path, retries: int = 4) -> Path:
    """Download once, keep forever (delete model/cache/ to refresh)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(r.content)
            tmp.replace(dest)
            return dest
        except requests.RequestException as e:
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"could not fetch {url}: {last}")


def _nflverse(tag: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(_fetch(f"{NFLVERSE}/{tag}/{name}", CACHE_DIR / "nflverse" / name))


_GAMES: pd.DataFrame | None = None


def _games() -> pd.DataFrame:
    """nflverse schedule with a UTC `kickoff`. gameday/gametime are US Eastern
    (Thu 2024-09-05 20:20 = KC opener = 00:20Z on the 6th); tz_localize raises on any
    ambiguous or nonexistent local time rather than guessing. 1999 has no gametime."""
    global _GAMES
    if _GAMES is None:
        g = _nflverse("schedules", "games.parquet")
        g = g[g["season"] >= 2000].copy()
        local = pd.to_datetime(g["gameday"] + " " + g["gametime"])
        g["kickoff"] = local.dt.tz_localize("America/New_York").dt.tz_convert("UTC") \
                            .astype("datetime64[ns, UTC]")
        _GAMES = g.reset_index(drop=True)
    return _GAMES.copy()


def _season_starts() -> pd.Series:
    g = _games()
    return g[g["game_type"] == "REG"].groupby("season")["kickoff"].min() - SEASON_START_LEAD


def _kickoffs() -> pd.Series:
    g = _games()
    return g.set_index("game_id")["kickoff"]


def load_schedule_tables(seasons: Iterable[int]) -> list[RawTable]:
    """One nflverse games row is four different kinds of information with four
    different availability times, so it is four tables."""
    seasons = list(seasons)
    g = _games()
    g = g[g["season"].isin(seasons)].reset_index(drop=True)
    starts = _season_starts()
    reg = g["game_type"] == "REG"

    fx = g[["game_id", "season", "week", "game_type", "kickoff", "home_team", "away_team",
            "location", "roof", "surface", "stadium_id", "div_game", "home_rest",
            "away_rest"]].copy()
    fx[KNOWN_AT] = fx["kickoff"] - NON_REG_FIXTURE_LEAD
    fx.loc[reg, KNOWN_AT] = fx.loc[reg, "season"].map(starts)
    fx[KNOWN_AT] = _ns_utc(fx[KNOWN_AT])

    has_line = g["spread_line"].notna() | g["total_line"].notna()
    ln = g.loc[has_line, ["game_id", "season", "week", "spread_line", "total_line",
                          "home_moneyline", "away_moneyline"]].copy()
    ln[KNOWN_AT] = _ns_utc(g.loc[has_line, "kickoff"] - LINES_CLOSE_LEAD)

    wx = g[["game_id", "season", "week", "roof", "temp", "wind"]].copy()
    wx[KNOWN_AT] = _ns_utc(g["kickoff"] - WEATHER_PROXY_LEAD)

    done = g["home_score"].notna() & g["away_score"].notna()
    h = g.loc[done, ["game_id", "season", "week", "game_type", "kickoff", "home_team", "away_team",
                     "home_score", "away_score"]]
    long = pd.concat([
        h.rename(columns={"home_team": "team", "away_team": "opponent",
                          "home_score": "points_for", "away_score": "points_against"}).assign(is_home=True),
        h.rename(columns={"away_team": "team", "home_team": "opponent",
                          "away_score": "points_for", "home_score": "points_against"}).assign(is_home=False),
    ], ignore_index=True)
    long[KNOWN_AT] = _ns_utc(long["kickoff"])
    long = long.drop(columns="kickoff").sort_values([KNOWN_AT, "game_id", "team"], kind="mergesort") \
               .reset_index(drop=True)

    return [
        RawTable("fixtures", fx, "REG: season start (first kickoff - 7d); playoffs: kickoff - 24h"),
        RawTable("lines", ln, "kickoff - 60min (closing line; nflverse publishes no close time)"),
        RawTable("weather_obs", wx, "kickoff - 24h",
                 proxy="OBSERVED post-game temp/wind used as a perfect day-ahead forecast (v1)"),
        RawTable("game_results", long, "game kickoff (plan: game rows); own-game row is excluded by strict <",
                 dropped={"not_completed": int((~done).sum())}),
    ]


def load_player_games(seasons: Iterable[int]) -> RawTable:
    keep = ["player_id", "player_display_name", "position", "season", "week", "season_type",
            "game_id", "team", "opponent_team", "completions", "attempts", "passing_yards",
            "passing_tds", "passing_interceptions", "carries", "rushing_yards", "rushing_tds",
            "targets", "receptions", "receiving_yards", "receiving_tds", "target_share",
            "air_yards_share", "wopr", "fantasy_points", "fantasy_points_ppr"]
    frames = [_nflverse("stats_player", f"stats_player_week_{y}.parquet")[keep] for y in seasons]
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    df = df[df["position"].isin(SKILL_POSITIONS)].copy()
    df[KNOWN_AT] = _ns_utc(df["game_id"].map(_kickoffs()))   # RawTable raises if a game_id is unmatched
    return RawTable("player_games", df.reset_index(drop=True), "game kickoff (plan: game rows)",
                    dropped={"non_skill_position": n0 - len(df)})


def _pfr_to_gsis() -> pd.Series:
    p = _nflverse("players", "players.parquet")[["gsis_id", "pfr_id"]].dropna()
    return p.drop_duplicates("pfr_id").set_index("pfr_id")["gsis_id"]


def load_snap_counts(seasons: Iterable[int]) -> RawTable:
    df = pd.concat([_nflverse("snap_counts", f"snap_counts_{y}.parquet") for y in seasons],
                   ignore_index=True)
    df["player_id"] = df["pfr_player_id"].map(_pfr_to_gsis())
    n_unmapped = int(df["player_id"].isna().sum())
    df = df[df["player_id"].notna()].copy()
    df[KNOWN_AT] = _ns_utc(df["game_id"].map(_kickoffs()))
    df = df[["player_id", "season", "week", "game_id", "team", "position", "offense_snaps",
             "offense_pct", "defense_snaps", "st_snaps", KNOWN_AT]]
    return RawTable("snap_counts", df.reset_index(drop=True), "game kickoff (plan: game rows)",
                    dropped={"pfr_id_not_in_players_table": n_unmapped})


def load_xfp(seasons: Iterable[int]) -> RawTable:
    """ffopportunity expected fantasy points. Week N's xFP is computed FROM week N, so
    for a week-N target only rows from earlier games pass the gate (own game: known_at
    == kickoff, excluded). Same-week xFP is a baseline to beat, never an input."""
    keep = ["season", "week", "game_id", "posteam", "player_id", "full_name", "position",
            "total_fantasy_points_exp", "pass_fantasy_points_exp", "rec_fantasy_points_exp",
            "rush_fantasy_points_exp", "total_fantasy_points"]
    df = pd.concat([pd.read_parquet(_fetch(f"{FFOPP}/ep_weekly_{y}.parquet",
                                           CACHE_DIR / "ffopportunity" / f"ep_weekly_{y}.parquet"))[keep]
                    for y in seasons], ignore_index=True)
    n_noid = int(df["player_id"].isna().sum())
    df = df[df["player_id"].notna()].copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    df = df.rename(columns={"posteam": "team"})
    df[KNOWN_AT] = _ns_utc(df["game_id"].map(_kickoffs()))
    return RawTable("xfp", df.reset_index(drop=True), "game kickoff (lag-only feature)",
                    dropped={"null_player_id": n_noid})


def load_injuries(seasons: Iterable[int]) -> RawTable:
    """One row per (player, week): the week's final designation. <=2024 carries
    `date_modified`; >=2025 does not, so known_at is derived and flagged."""
    ko = _games()
    team_wk = pd.concat([ko[["season", "week", "home_team", "kickoff"]].rename(columns={"home_team": "team"}),
                         ko[["season", "week", "away_team", "kickoff"]].rename(columns={"away_team": "team"})])
    team_wk = team_wk.set_index(["season", "week", "team"])["kickoff"]
    frames = []
    for y in seasons:
        d = _nflverse("injuries", f"injuries_{y}.parquet").copy()
        kick = pd.Series(list(zip(d["season"], d["week"], d["team"])), index=d.index).map(team_wk)
        if "date_modified" in d.columns:
            d[KNOWN_AT] = _ns_utc(d["date_modified"])
            d["known_at_source"] = "reported"
        else:
            d[KNOWN_AT] = _ns_utc(kick - INJURY_DERIVED_LEAD)
            d["known_at_source"] = "derived_kickoff_minus_24h"
        frames.append(d)
    df = pd.concat(frames, ignore_index=True).rename(columns={"gsis_id": "player_id"})
    for c in ("report_status", "practice_status", "report_primary_injury", "practice_primary_injury"):
        df[c] = df[c].where(df[c].notna() & (df[c].astype(str).str.strip() != ""), None)
        df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
    n_bad = int(df[KNOWN_AT].isna().sum())
    df = df[df[KNOWN_AT].notna()]
    df = df[["player_id", "season", "week", "game_type", "team", "position", "report_status",
             "report_primary_injury", "practice_status", "practice_primary_injury",
             KNOWN_AT, "known_at_source"]]
    return RawTable("injuries", df.reset_index(drop=True),
                    "date_modified when present (<=2024); else kickoff - 24h (>=2025, conservative)",
                    dropped={"no_timestamp_derivable": n_bad})


def load_depth_charts(seasons: Iterable[int]) -> RawTable:
    """Two formats. <=2024: weekly, no snapshot time -> known_at = the team's kickoff that
    week (so week N's chart is unusable for game N, usable for N+1; bye teams get the
    league's last kickoff that week). >=2025: ESPN-style snapshots with a real `dt`."""
    g = _games()
    tw = pd.concat([g[["season", "week", "home_team", "kickoff"]].rename(columns={"home_team": "team"}),
                    g[["season", "week", "away_team", "kickoff"]].rename(columns={"away_team": "team"})])
    team_ko = tw.set_index(["season", "week", "team"])["kickoff"]
    wk_last = g.groupby(["season", "week"])["kickoff"].max()
    frames, n_sbbye = [], 0
    for y in seasons:
        d = _nflverse("depth_charts", f"depth_charts_{y}.parquet")
        if "dt" in d.columns:
            d = d[d["pos_abb"].isin(OFFENSE_DEPTH_POS)]
            out = pd.DataFrame({"player_id": d["gsis_id"], "team": d["team"], "season": y,
                                "week": pd.NA, "pos": d["pos_abb"], "rank": d["pos_rank"].astype(int),
                                KNOWN_AT: _ns_utc(d["dt"]), "known_at_source": "snapshot_dt"})
        else:
            n_sbbye += int(d["week"].isna().sum())          # 'SBBYE' snapshots have no week
            d = d[(d["formation"] == "Offense") & d["week"].notna()]
            keys = list(zip(d["season"], d["week"].astype(int), d["club_code"]))
            k = pd.Series(keys, index=d.index).map(team_ko)
            bye = k.isna()
            k[bye] = pd.Series(list(zip(d["season"], d["week"].astype(int))), index=d.index)[bye].map(wk_last)
            out = pd.DataFrame({"player_id": d["gsis_id"], "team": d["club_code"], "season": d["season"],
                                "week": d["week"].astype(int),
                                "pos": d["depth_position"].str.strip().replace("", None),
                                "rank": pd.to_numeric(d["depth_team"]).astype(int),
                                KNOWN_AT: _ns_utc(k), "known_at_source": "derived_team_kickoff"})
        frames.append(out)
    df = pd.concat(frames, ignore_index=True)
    n_bad = int(df[KNOWN_AT].isna().sum())
    return RawTable("depth_charts", df[df[KNOWN_AT].notna()].reset_index(drop=True),
                    "<=2024: team kickoff that week (derived); >=2025: snapshot dt",
                    dropped={"no_timestamp_derivable": n_bad, "sbbye_snapshot_no_week": n_sbbye})


def load_static_tables() -> list[RawTable]:
    """Season-static data (player bio, draft capital, combine): known_at = the start of the
    season the fact belongs to, minus SEASON_START_LEAD. Column-whitelisted (see top)."""
    starts = _season_starts()
    pl = _nflverse("players", "players.parquet")[PLAYER_COLS].copy()
    pl = pl[pl["gsis_id"].notna()]
    start = pl["draft_year"].fillna(pl["rookie_season"]).map(starts)   # drafted year, else rookie year
    n_no_year = int(start.isna().sum())
    pl = pl[start.notna()].copy()
    pl[KNOWN_AT] = _ns_utc(start[start.notna()])
    pl = pl.rename(columns={"gsis_id": "player_id"})
    pl["birth_date"] = pd.to_datetime(pl["birth_date"])

    dp = _nflverse("draft_picks", "draft_picks.parquet")[DRAFT_COLS].copy()
    n_dp = int(dp["gsis_id"].isna().sum())
    dp = dp[dp["gsis_id"].notna() & dp["season"].isin(starts.index)].rename(columns={"gsis_id": "player_id"})
    dp[KNOWN_AT] = _ns_utc(dp["season"].map(starts))

    cb = _nflverse("combine", "combine.parquet")[COMBINE_COLS].copy()
    cb["player_id"] = cb["pfr_id"].map(_pfr_to_gsis())
    n_cb = int(cb["player_id"].isna().sum())
    cb = cb[cb["player_id"].notna() & cb["season"].isin(starts.index)].copy()
    cb[KNOWN_AT] = _ns_utc(cb["season"].map(starts))

    rule = "season start (first REG kickoff - 7d) of the draft/rookie/combine season"
    return [RawTable("players_static", pl.reset_index(drop=True), rule,
                     dropped={"no_draft_or_rookie_year": n_no_year}),
            RawTable("draft_picks", dp.reset_index(drop=True), rule, dropped={"no_gsis_id": n_dp}),
            RawTable("combine", cb.reset_index(drop=True), rule, dropped={"pfr_id_not_mapped": n_cb})]


def load_store(seasons: Iterable[int] = (2024,)) -> RawStore:
    """Every raw table, each with known_at. Season-scoped except the static tables."""
    seasons = list(seasons)
    tables = [*load_schedule_tables(seasons), load_player_games(seasons), load_snap_counts(seasons),
              load_xfp(seasons), load_injuries(seasons), load_depth_charts(seasons),
              *load_static_tables()]
    return RawStore(tables)
