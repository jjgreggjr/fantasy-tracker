"""FantasyPros weekly Expert Consensus Rankings (the DynastyProcess archive) -> model/cache/ecr/ecr_weekly.parquet.

    python -m model.fetch_ecr [--force] [--first-season 2021]

Source (GPL-3, DynastyProcess, scraped by their GitHub Action every Friday 00:03 UTC):
  files/db_fpecr.parquet    every FantasyPros ranking page they scrape, one row per (page, player, scrape_date)
  files/db_playerids.csv    the id crosswalk (fantasypros_id -> gsis_id and a dozen others)

Both are downloaded once into model/cache/ecr/ (git-ignored; --force re-downloads). Read off the data, not assumed:

  ecr_type  pages                                    what it is
  wp        ppr-rb / ppr-wr / ppr-te / qb (+ K/DST/IDP)  weekly PPR rankings BY POSITION: `ecr` is the experts' mean
                                                     positional rank (Josh Allen 1.0 on qb.php)
  wsf       ppr-superflex                             weekly PPR superflex OVERALL (QB/RB/WR/TE in one list)
  wo        idp (since 2021; ppr-flex only in 2019-20)  weekly overall IDP: no offense rows after 2020, so unused

Kept: wp and wsf rows of QB/RB/WR/TE. Week assignment: a scrape belongs to the (season, week) of the first game
kickoff STRICTLY after its known_at (a Friday scrape -> that week's Sunday; it can land after the Thursday game of the
same week, which is why the gate, not this file, decides which scrape a given game sees). A scrape more than
MAX_LEAD before that kickoff (offseason) is dropped and counted.

known_at = scrape_date at 00:00 UTC (the real scrape day; the cron fires 00:03 UTC Friday, so it is a lower bound by
minutes: on an EDT Thursday-night game, 00:15 UTC, the Friday scrape may have run just after kickoff; the
rankings themselves carry no in-game information, and this is flagged as the table's proxy).

Per player-week only the scrapes that are the LATEST before some kickoff of that week are kept (normally exactly one).
`ecr_pos_rank` is the ordinal rank of `ecr` within (scrape, position) over the whole published page, before any id
matching, so an unmatched player still occupies his rank.

Identity: FantasyPros id -> gsis_id via db_playerids; then FP's own yahoo / cbs / sportsdata ids against the same
crosswalk. No name matching (a wrong match is worse than NA). The match rate per season x position is printed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from model import point_in_time as pit
from model.point_in_time import KNOWN_AT, RawTable

BASE = "https://raw.githubusercontent.com/dynastyprocess/data/master/files"
ECR_DIR = pit.CACHE_DIR / "ecr"
RAW_ECR, RAW_IDS, OUT = "db_fpecr.parquet", "db_playerids.csv", "ecr_weekly.parquet"
POSITIONS = ("QB", "RB", "WR", "TE")
WEEKLY_TYPES = ("wp", "wsf")
MAX_LEAD = pd.Timedelta(days=8)
FIRST_SEASON = 2021
# FantasyPros column -> db_playerids column, tried in order after fantasypros_id
ALT_IDS = (("yahoo_id", "yahoo_id"), ("cbs_id", "cbs_id"), ("sportsdata_id", "sportradar_id"))


def download(force: bool = False, ecr_dir: Path = ECR_DIR) -> tuple[Path, Path]:
    out = []
    for name in (RAW_ECR, RAW_IDS):
        dest = ecr_dir / name
        if force and dest.exists():
            dest.unlink()
        out.append(pit._fetch(f"{BASE}/{name}", dest))
    return out[0], out[1]


def _key(s: pd.Series) -> pd.Series:
    """Ids arrive as int, float or str depending on the file: compare as canonical strings, NaN stays NaN."""
    s = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return s.where(s.notna() & (s != "") & (s.str.lower() != "nan"))


def assign_weeks(scrapes: pd.Series, games: pd.DataFrame, max_lead: pd.Timedelta = MAX_LEAD) -> pd.DataFrame:
    """For each distinct known_at: season, week of the first kickoff strictly after it (NaN if none within max_lead)."""
    g = games[["season", "week", "kickoff"]].sort_values("kickoff", kind="mergesort").reset_index(drop=True)
    k = pd.DataFrame({KNOWN_AT: pd.Series(scrapes.unique()).sort_values().astype("datetime64[ns, UTC]")})
    m = pd.merge_asof(k, g.rename(columns={"kickoff": "next_kickoff"}), left_on=KNOWN_AT, right_on="next_kickoff",
                      direction="forward", allow_exact_matches=False)
    far = m["next_kickoff"].isna() | (m["next_kickoff"] - m[KNOWN_AT] > max_lead)
    m.loc[far, ["season", "week"]] = pd.NA
    return m


def latest_per_slot(df: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Keep a player's scrape for a week only if it is the latest one before at least one kickoff of that week."""
    keep = []
    slots = games.groupby(["season", "week"])["kickoff"].unique()
    for (season, week), sub in df.groupby(["season", "week"], sort=False):
        for ko in slots.get((season, week), []):
            before = sub[sub[KNOWN_AT] < ko]
            if len(before):
                newest = before.groupby("fp_id")[KNOWN_AT].transform("max").eq(before[KNOWN_AT])
                keep.append(set(before.index[newest.to_numpy()]))
    return df.loc[sorted(set().union(*keep))] if keep else df.iloc[0:0]


def map_ids(df: pd.DataFrame, ids: pd.DataFrame) -> pd.Series:
    """gsis_id per row: fantasypros_id first, then FP's own alternative ids. Only crosswalk keys that are unique win."""
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    tries = (("fp_id", "fantasypros_id"),) + ALT_IDS
    for mine, theirs in tries:
        if mine not in df.columns or theirs not in ids.columns:
            continue
        x = ids[[theirs, "gsis_id"]].copy()
        x[theirs], x["gsis_id"] = _key(x[theirs]), _key(x["gsis_id"])
        x = x.dropna().drop_duplicates()
        x = x[~x[theirs].duplicated(keep=False)]                    # an ambiguous crosswalk key is no match
        lookup = x.set_index(theirs)["gsis_id"]
        miss = out.isna()
        out[miss] = _key(df.loc[miss, mine]).map(lookup)
    return out


def build(raw: pd.DataFrame, ids: pd.DataFrame, games: pd.DataFrame, first_season: int = FIRST_SEASON
          ) -> tuple[pd.DataFrame, dict]:
    """Raw archive rows -> one row per (gsis player, scrape) with the wp (positional) and wsf (overall) values."""
    need = {"id", "pos", "ecr", "sd", "best", "worst", "ecr_type", "scrape_date"}
    if not need <= set(raw.columns):
        raise ValueError(f"db_fpecr schema changed: need {sorted(need)}, got {sorted(raw.columns)}")
    w = raw[raw["ecr_type"].isin(WEEKLY_TYPES) & raw["pos"].isin(POSITIONS)].copy()
    w["fp_id"] = _key(w["id"])
    w[KNOWN_AT] = pd.to_datetime(w["scrape_date"], utc=True, format="ISO8601").astype("datetime64[ns, UTC]")
    w = w[w["fp_id"].notna() & w["ecr"].notna()]
    w = w.sort_values([KNOWN_AT, "ecr_type", "pos", "ecr", "fp_id"], kind="mergesort")
    w = w.drop_duplicates(["ecr_type", KNOWN_AT, "fp_id"])          # a player on two pages of one type: best rank wins

    wp = w[w["ecr_type"] == "wp"].copy()
    wp["ecr_pos_rank"] = wp.groupby([KNOWN_AT, "pos"])["ecr"].rank(method="min")
    wp = wp.rename(columns={"ecr": "ecr_pos_ecr", "sd": "ecr_sd", "best": "ecr_best", "worst": "ecr_worst",
                            "player_owned_avg": "ecr_owned_pct", "pos": "position"})
    sf = w[w["ecr_type"] == "wsf"][["fp_id", KNOWN_AT, "ecr"]].rename(columns={"ecr": "ecr_sf_rank"})
    df = wp.merge(sf, on=["fp_id", KNOWN_AT], how="left")
    if "ecr_owned_pct" not in df.columns:
        df["ecr_owned_pct"] = float("nan")

    wk = assign_weeks(df[KNOWN_AT], games)
    df = df.merge(wk[[KNOWN_AT, "season", "week"]], on=KNOWN_AT, how="left")
    dropped = {"offseason_scrape": int(df["season"].isna().sum())}
    df = df[df["season"].notna()].astype({"season": "int64", "week": "int64"})
    dropped["before_first_season"] = int((df["season"] < first_season).sum())
    df = df[df["season"] >= first_season]
    n = len(df)
    df = latest_per_slot(df, games)
    dropped["superseded_scrape"] = n - len(df)

    df["player_id"] = map_ids(df, ids)
    keep = ["player_id", "fp_id", "player", "position", "team", "season", "week", "scrape_date", "ecr_sf_rank",
            "ecr_pos_ecr", "ecr_pos_rank", "ecr_sd", "ecr_best", "ecr_worst", "ecr_owned_pct", KNOWN_AT]
    df = df[[c for c in keep if c in df.columns]].sort_values([KNOWN_AT, "position", "ecr_pos_rank", "fp_id"],
                                                               kind="mergesort").reset_index(drop=True)
    return df, dropped


def match_report(df: pd.DataFrame) -> pd.DataFrame:
    t = df.assign(m=df["player_id"].notna()).groupby(["season", "position"])["m"].agg(["size", "mean"])
    t = t.unstack("position")
    rate = t["mean"].round(3)
    rate.columns = [f"{c}_match" for c in rate.columns]
    rows = df.groupby("season").agg(rows=("fp_id", "size"), matched=("player_id", lambda s: s.notna().mean().round(3)),
                                    weeks=("week", "nunique"), scrapes=("scrape_date", "nunique"))
    return rows.join(rate)


TABLE_COLS = ["player_id", "season", "week", "position", "scrape_date", "ecr_sf_rank", "ecr_pos_ecr", "ecr_pos_rank",
              "ecr_sd", "ecr_best", "ecr_worst", "ecr_owned_pct", KNOWN_AT]
KNOWN_AT_RULE = "scrape_date 00:00 UTC (DynastyProcess scrapes FantasyPros Fridays 00:03 UTC)"
PROXY = ("the scrape runs minutes after 00:00 UTC on scrape_date; an EDT Thursday-night kickoff (00:15 UTC) can precede it. "
         "Rankings carry no in-game information")


def table_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The fetched frame -> the gated table: matched rows only, one row per (player, scrape)."""
    dropped = {"no_gsis_id": int(df["player_id"].isna().sum())}
    df = df[df["player_id"].notna()].sort_values([KNOWN_AT, "ecr_pos_rank"], kind="mergesort")
    n = len(df)
    df = df.drop_duplicates(["player_id", KNOWN_AT])
    dropped["two_fp_ids_one_player"] = n - len(df)
    out = df[TABLE_COLS].reset_index(drop=True)
    out[KNOWN_AT] = pd.to_datetime(out[KNOWN_AT], utc=True).astype("datetime64[ns, UTC]")
    return out, dropped


def load_ecr_table(path: Path | None = None) -> RawTable | None:
    """The `ecr` RawTable from the fetched file, or None when `python -m model.fetch_ecr` has not run."""
    path = path or ECR_DIR / OUT
    if not path.exists():
        return None
    df, dropped = table_frame(pd.read_parquet(path))
    if df.empty:
        return None
    return RawTable("ecr", df, KNOWN_AT_RULE, proxy=PROXY, dropped=dropped)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-download both files")
    ap.add_argument("--first-season", type=int, default=FIRST_SEASON)
    a = ap.parse_args(argv)
    f_ecr, f_ids = download(a.force)
    raw = pd.read_parquet(f_ecr)
    ids = pd.read_csv(f_ids, low_memory=False, dtype=str)
    games = pit._games()
    df, dropped = build(raw, ids, games, a.first_season)
    out = ECR_DIR / OUT
    df.to_parquet(out, index=False)
    print(match_report(df).to_string())
    print(f"dropped: {dropped}")
    print(f"wrote {out} ({len(df):,} rows, {df['player_id'].notna().sum():,} with a gsis_id)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
