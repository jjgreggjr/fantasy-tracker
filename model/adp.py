"""FantasyFootballCalculator ADP -> a gated `adp` table (only when fetched CSVs exist).

The CSVs come from `model/fetch_adp.py`, run in GitHub Actions because the sandbox cannot reach FFC.
If `model/data/adp/` holds no `adp_*_*.csv`, `load_adp_table()` returns None, the store has no `adp`
table, and the ADP feature columns are NA. Nothing is invented.

known_at (the leakage decision): the plan says season start. FFC serves the CURRENT aggregate for a
past year, so a fetched season can include drafts held after that season's opener. When the response
carried a drafts window (`meta_end_date`), known_at = max(season start, that end date): a later stamp
only hides ADP from early weeks, it can never leak. With no window in the CSV, known_at = season start
and the residual risk (post-opener drafts in the average) is documented in PLAN_MODEL.md.

Identity: FFC gives names, not gsis ids. We match on normalised name + position against the nflverse
players table (identity only: `rookie_season` / `last_season` pick among same-name players and never
become a feature value). Unmatched rows are counted in `RawTable.dropped`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from model import point_in_time as pit
from model.point_in_time import KNOWN_AT, RawTable

ADP_DIR = Path(__file__).resolve().parent / "data" / "adp"
FORMATS = {"ppr": "ppr", "standard": "std"}          # FFC format -> column suffix
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.I)
_POS = {"QB", "RB", "WR", "TE"}


def norm_name(s: str) -> str:
    s = _SUFFIX.sub("", str(s).lower())
    return re.sub(r"[^a-z]", "", s)


def _read_csvs(adp_dir: Path) -> pd.DataFrame | None:
    files = sorted(adp_dir.glob("adp_*_*.csv"))
    if not files:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def build_adp_frame(raw: pd.DataFrame, players: pd.DataFrame, season_starts: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Raw FFC rows -> [player_id, season, format, adp, pos_rank, fetched_at, known_at]."""
    need = {"season", "format", "name", "position", "adp"}
    if not need <= set(raw.columns):
        raise ValueError(f"FFC CSV schema changed: need {sorted(need)}, got {sorted(raw.columns)}")
    df = raw[raw["position"].isin(_POS)].copy()
    df["adp"] = pd.to_numeric(df["adp"], errors="coerce")
    df = df[df["adp"].notna()]
    df["key"] = df["name"].map(norm_name)

    pl = players[players["position"].isin(_POS) & players["gsis_id"].notna()].copy()
    pl["key"] = pl["display_name"].map(norm_name)
    ids: dict[tuple, str] = {}
    for (season, key, pos), _ in df.groupby(["season", "key", "position"]):
        c = pl[(pl["key"] == key) & (pl["position"] == pos) & (pl["rookie_season"] <= season)
               & (pl["last_season"] >= season - 1)]
        if len(c):
            ids[(season, key, pos)] = c.sort_values("rookie_season").iloc[-1]["gsis_id"]
    df["player_id"] = [ids.get((s, k, p)) for s, k, p in zip(df["season"], df["key"], df["position"])]
    n_unmatched = int(df["player_id"].isna().sum())
    df = df[df["player_id"].notna()].copy()
    df = df.sort_values(["season", "format", "player_id", "adp"]).drop_duplicates(["season", "format", "player_id"])
    df["pos_rank"] = df.groupby(["season", "format", "position"])["adp"].rank(method="first")

    start = df["season"].map(season_starts)
    if "meta_end_date" in df.columns:
        end = pd.to_datetime(df["meta_end_date"], utc=True, errors="coerce")
        start = pd.concat([start, end], axis=1).max(axis=1)
    df[KNOWN_AT] = pd.to_datetime(start, utc=True).astype("datetime64[ns, UTC]")
    keep = ["player_id", "season", "format", "position", "adp", "pos_rank", "fetched_at", KNOWN_AT]
    return df[keep].reset_index(drop=True), {"name_not_matched": n_unmatched}


def load_adp_table(adp_dir: Path = ADP_DIR) -> RawTable | None:
    raw = _read_csvs(adp_dir)
    if raw is None or raw.empty:
        return None
    players = pit._nflverse("players", "players.parquet")[["gsis_id", "display_name", "position", "rookie_season",
                                                           "last_season"]]
    df, dropped = build_adp_frame(raw, players, pit._season_starts())
    if df.empty:
        return None
    return RawTable("adp", df, "season start (or the drafts-window end if the fetch carried one)",
                    proxy="FFC serves the current aggregate for a past year: drafts after the opener may be in the average",
                    dropped=dropped)
