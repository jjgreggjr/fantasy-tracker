"""The gameday inactive list as a LABEL (`y_inactive`), 2021 on, from nflverse `weekly_rosters/roster_weekly_{season}.parquet`.

    from model import inactives as INA
    store = INA.with_inactives(store, range(2021, 2027))     # adds the `inactives` RawTable
    df = INA.attach_inactive_label(store, df)                 # + y_inactive (0/1) on (player_id, season, week)

What the file is. One row per player-week of every team's roster; `status == "INA"` marks the 46-man gameday inactives (about 200
players a week league-wide, every position). nflverse posts each game's list about 90 minutes before its kickoff (Thursday night's is
already in Friday's file). `status_description_abbr` is NOT the signal: 2026 INA rows carry `A01` ("active"); `I01` is a minority
before 2024. There is no per-row timestamp, so `known_at` = that team's kickoff - 90 min (INACTIVE_LEAD), derived like the >= 2025
injury rows.

`y_inactive` versus `y_played` (measured on the 2021-2026 spine, 49,859 rows, 2026-10-16):
  * INA is a strict subset of did-not-play: 0 rows are INA and played. 5,274 of the 15,661 DNP rows (34%) are INA.
  * The other DNPs are RES (IR / PUP / NFI, 3,658), ACT but no snap (3,505: mostly backup QBs, the "dressed, did not play" case
    no list can catch), DEV (practice squad, 1,995), no roster row (851), CUT / RET / EXE (375).
  So `y_inactive` is the "ruled out on the day" part of P(plays), with a real timestamp; `y_played` stays the training mask.

Rules (why this lives here and not in labels.py / point_in_time.load_store):
  * It is a LABEL and, at most, a Sunday-only feature. Its known_at is kickoff - 90 min, so it can never pass the gate for a
    Friday serve of a Sunday game; a model that read it as an input would see it only on the pre-lock run. It is in NO training
    family, and the primary's 71 inputs (model/serving_config.json) are unchanged.
  * Its operational use is the status layer's: the pipeline marks the week's INA players `inactive` (ff.status.week_inactives,
    E_pts blanked) and the serve withholds their model rows (model.serve.on_inactive_list).
  * labels.py and point_in_time.py are hashed into the serve's history-matrix cache key (serve.CODE_FILES) and the Actions cache
    key: touching them forces a full 2021-2025 matrix rebuild (~12 min on 4 cores) inside a model step capped at 10 minutes. This
    module is opt-in for that reason; nothing in the weekly path imports it.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Iterable

import numpy as np
import pandas as pd

from model import point_in_time as pit

INACTIVE_LEAD = timedelta(minutes=90)          # nflverse posts each game's list ~90 min before its kickoff
FIRST_SEASON = 2021
TABLE = "inactives"
RULE = ("team kickoff - 90 min (gameday inactive list; no per-row timestamp). A LABEL and a Sunday pre-lock signal only: "
        "never available to a Friday serve")


def inactive_rows(raw: pd.DataFrame, kickoffs: pd.Series) -> tuple[pd.DataFrame, int]:
    """(INA rows with known_at, rows dropped because their team-week has no kickoff). `kickoffs` is indexed (season, week, team)."""
    d = raw[raw["status"].astype(str).str.strip().str.upper().eq("INA") & raw["gsis_id"].notna()].copy()
    d = d.rename(columns={"gsis_id": "player_id"})
    d["season"] = pd.to_numeric(d["season"], errors="coerce").astype("int64")
    d["week"] = pd.to_numeric(d["week"], errors="coerce").astype("int64")
    kick = pd.Series(list(zip(d["season"], d["week"], d["team"])), index=d.index).map(kickoffs)
    d[pit.KNOWN_AT] = pd.to_datetime(kick, utc=True) - INACTIVE_LEAD
    n_bad = int(d[pit.KNOWN_AT].isna().sum())
    d = d[d[pit.KNOWN_AT].notna()].copy()
    d[pit.KNOWN_AT] = d[pit.KNOWN_AT].astype("datetime64[ns, UTC]")
    d["kickoff"] = d[pit.KNOWN_AT] + INACTIVE_LEAD
    keep = ["player_id", "season", "week", "game_type", "team", "position", "status", "status_description_abbr", "kickoff", pit.KNOWN_AT]
    for c in keep:
        if c not in d.columns:
            d[c] = np.nan
    return d[keep].drop_duplicates(["player_id", "season", "week"]).reset_index(drop=True), n_bad


def _team_kickoffs() -> pd.Series:
    g = pit._games()
    tw = pd.concat([g[["season", "week", "home_team", "kickoff"]].rename(columns={"home_team": "team"}),
                    g[["season", "week", "away_team", "kickoff"]].rename(columns={"away_team": "team"})])
    return tw.drop_duplicates(["season", "week", "team"]).set_index(["season", "week", "team"])["kickoff"]


def load_inactives(seasons: Iterable[int]) -> pit.RawTable:
    """The `inactives` RawTable for `seasons` (2021 on: what the spine covers), read through point_in_time's nflverse cache."""
    seasons = [y for y in seasons if y >= FIRST_SEASON]
    raw = pd.concat([pit._nflverse("weekly_rosters", f"roster_weekly_{y}.parquet") for y in seasons], ignore_index=True)
    df, n_bad = inactive_rows(raw, _team_kickoffs())
    return pit.RawTable(TABLE, df, RULE, dropped={"no_team_kickoff": n_bad})


def with_inactives(store: pit.RawStore, seasons: Iterable[int], table: pit.RawTable | None = None) -> pit.RawStore:
    """A copy of `store` with the `inactives` table added (a store is immutable). `table` overrides the download (tests)."""
    t = table if table is not None else load_inactives(seasons)
    return pit.RawStore([*store._tables.values(), t])


def y_inactive(df: pd.DataFrame, inactives: pd.DataFrame) -> np.ndarray:
    """int8 per row of `df` (player_id, season, week): 1 when he was on that week's gameday inactive list."""
    keys = set(zip(inactives["player_id"].astype(str), inactives["season"].astype(int), inactives["week"].astype(int)))
    return np.fromiter(((str(p), int(s), int(w)) in keys for p, s, w in zip(df["player_id"], df["season"], df["week"])),
                       dtype=bool, count=len(df)).astype("int8")


def attach_inactive_label(store: pit.RawStore, df: pd.DataFrame) -> pd.DataFrame:
    """`df` plus `y_inactive`. Like labels.attach_labels it reads target-week rows, so it is a label: never call it from features."""
    if not store.has(TABLE):
        raise LookupError("the store has no `inactives` table: build it with with_inactives(store, seasons)")
    out = df.copy()
    out["y_inactive"] = y_inactive(out, store._tables[TABLE].df)
    return out


def known_before(store: pit.RawStore, kickoff_ts, **keys) -> pd.DataFrame:
    """Inactive rows with known_at < `kickoff_ts` (the strict as-of gate). Gated at a game's kickoff this INCLUDES that game's own list
    (known at kickoff - 90 min): it models a serve made after the list is posted, i.e. the Sunday pre-lock run only. A Friday serve of a
    Sunday game must gate at its own clock, where the list does not exist yet; that is why this is never a Friday feature."""
    return pit.as_of_join(store, TABLE, kickoff_ts, **keys)
