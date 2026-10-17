"""The opt-in weekly charting families `ngs`, `pfr` and `ftn` (model/point_in_time.py `load_ngs_tables` / `load_pfr_tables` /
`load_ftn_tables`, model/features.py `_charting_features`).

    python -m unittest model.tests.test_charting_families

No network: the loaders read synthetic release files written into a temporary cache (requests.get is patched to fail), the
features read a hand-built store. Covered:

  * the known_at rule per table: every row of (season, week) at 23:59:59Z on the first refresh weekday (FTN Wed, PFR Thu,
    NGS Fri) strictly after the day the week's last game ended; FTN's in-season `date_pulled` delays it, a stale re-pull does not.
    Consequence pinned: a week-N game sees weeks <= N-1 at most, and a Thursday game cannot see last week's NGS
  * the loaders: week-0 / postseason / unmapped rows dropped and counted, NGS team codes, the pfr -> gsis map, the FTN join
    to play-by-play and its per-player-game counts, FTN 2021 (a 404) leaves no rows
  * trailing math on a synthetic store: l1 = newest published game, l3 = the last three, a non-qualifying NGS game is NaN,
    an uncharted-for-him FTN game is zero, PFR per-carry rates are ratios of sums; 2021 has NaN ftn columns
  * canaries: rows stamped at / after kickoff move nothing; the same rows one second before kickoff move l1 to their exact
    values (so the first test cannot pass vacuously); each table is gated on its own
  * the registry: opt-in, the frozen 71 serving inputs untouched, constants pinned equal across the two modules
"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from model import features as F
from model import point_in_time as pit
from model import train as T
from model import trial as TR
from model.features import NAN, RESULT_LAG, _charting_features, _history, fingerprint
from model.point_in_time import KNOWN_AT, RawStore, RawTable, TargetRow

FAMS = ("ngs", "pfr", "ftn")
NEW_COLS = [c for f in FAMS for c in F.FAMILIES[f]]
SEC = pd.Timedelta(seconds=1)


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-12


# ================================================================== loaders on synthetic release files
# (season, week, gameday, gametime ET, away, home): Thursday, Sunday, Monday night each week.
SCHEDULE = []
for season, thu in ((2021, "2021-09-09"), (2022, "2022-09-08")):
    for w in (1, 2, 3):
        d0 = pd.Timestamp(thu) + pd.Timedelta(days=7 * (w - 1))
        SCHEDULE += [(season, w, str(d0.date()), "20:15", "CCC", "DDD"),
                     (season, w, str((d0 + pd.Timedelta(days=3)).date()), "13:00", "AAA", "BBB"),
                     (season, w, str((d0 + pd.Timedelta(days=4)).date()), "20:15", "EEE", "LA")]


def gid(season: int, week: int, away: str, home: str) -> str:
    return f"{season}_{week:02d}_{away}_{home}"


def _games_frame() -> pd.DataFrame:
    return pd.DataFrame([{"season": s, "week": w, "gameday": d, "gametime": t, "game_type": "REG", "away_team": a, "home_team": h,
                          "game_id": gid(s, w, a, h)} for s, w, d, t, a, h in SCHEDULE])


def _ngs(kind: str, rows: list[dict]) -> pd.DataFrame:
    base = {"season_type": "REG", "player_gsis_id": None, "team_abbr": None}
    df = pd.DataFrame([{**base, **{s: np.nan for s in pit.NGS_STATS[kind]}, **r} for r in rows])
    df["season"] = df["season"].astype("int32")
    df["week"] = df["week"].astype("int32")
    return df


def _write_release_cache(root: Path) -> None:
    nv = root / "nflverse"
    nv.mkdir(parents=True)
    _games_frame().to_parquet(nv / "games.parquet")
    pd.DataFrame({"gsis_id": ["00-W1", "00-R1", "00-Q1", "00-W2"], "pfr_id": ["W1pf", "R1pf", "Q1pf", None]}).to_parquet(nv / "players.parquet")
    # ---- NGS: one row per kind in 2022 week 1, a LAR row (-> LA), a week-0 season aggregate and a postseason row
    _ngs("receiving", [
        {"season": 2022, "week": 1, "player_gsis_id": "00-W1", "team_abbr": "BBB", "avg_separation": 3.0, "catch_percentage": 70.0},
        {"season": 2022, "week": 1, "player_gsis_id": "00-W2", "team_abbr": "LAR", "avg_separation": 2.0},
        {"season": 2022, "week": 0, "player_gsis_id": "00-W1", "team_abbr": "BBB", "avg_separation": 9.0},
        {"season": 2022, "week": 19, "season_type": "POST", "player_gsis_id": "00-W1", "team_abbr": "BBB", "avg_separation": 9.0},
        {"season": 2022, "week": 2, "player_gsis_id": "00-W1", "team_abbr": "DDD", "avg_separation": 4.0}]
    ).to_parquet(nv / "ngs_receiving.parquet")
    _ngs("rushing", [{"season": 2022, "week": 1, "player_gsis_id": "00-W1", "team_abbr": "BBB", "efficiency": 4.5}]
         ).to_parquet(nv / "ngs_rushing.parquet")
    _ngs("passing", [{"season": 2022, "week": 1, "player_gsis_id": "00-Q1", "team_abbr": "BBB", "aggressiveness": 20.0}]
         ).to_parquet(nv / "ngs_passing.parquet")
    # ---- PFR 2022: W1 receiving, R1 rushing, Q1 passing, an unmapped id, a postseason row
    g1 = gid(2022, 1, "AAA", "BBB")
    common = {"season": 2022, "week": 1, "game_type": "REG", "game_id": g1, "team": "BBB"}
    pd.DataFrame([{**common, "pfr_player_id": "W1pf", "receiving_drop": 1.0, "receiving_drop_pct": 0.25, "receiving_broken_tackles": 2.0,
                   "receiving_rat": 101.0},
                  {**common, "pfr_player_id": "NOPE", "receiving_drop": 0.0, "receiving_drop_pct": 0.0, "receiving_broken_tackles": 0.0,
                   "receiving_rat": 50.0},
                  {**common, "game_type": "WC", "week": 19, "pfr_player_id": "W1pf", "receiving_drop": 9.0, "receiving_drop_pct": 0.9,
                   "receiving_broken_tackles": 9.0, "receiving_rat": 9.0}]).to_parquet(nv / "advstats_week_rec_2022.parquet")
    pd.DataFrame([{**common, "pfr_player_id": "R1pf", "carries": 10.0, "rushing_yards_before_contact": 20.0,
                   "rushing_yards_after_contact": 15.0, "rushing_broken_tackles": 2.0}]).to_parquet(nv / "advstats_week_rush_2022.parquet")
    pd.DataFrame([{**common, "pfr_player_id": "Q1pf", "times_pressured_pct": 0.3, "times_blitzed": 5.0, "times_hurried": 4.0,
                   "times_hit": 3.0, "times_sacked": 2.0, "passing_drop_pct": 0.05, "passing_bad_throw_pct": 0.1}]
                 ).to_parquet(nv / "advstats_week_pass_2022.parquet")
    # ---- FTN 2022 (2021 is a remembered 404) and the play-by-play it joins to
    (nv / "ftn_charting_2021.parquet.absent").write_bytes(b"")
    g2 = gid(2022, 2, "AAA", "BBB")
    pbp = pd.DataFrame([
        # game_id, play_id, play_type, pass_attempt, rush_attempt, qb_dropback, two_point, receiver, rusher, passer
        (g1, 1, "pass", 1, 0, 1, 0, "00-W1", None, "00-Q1"),
        (g1, 2, "pass", 1, 0, 1, 0, "00-W2", None, "00-Q1"),
        (g1, 3, "run", 0, 1, 0, 0, None, "00-R1", None),
        (g1, 4, "run", 0, 1, 0, 0, None, "00-R1", None),
        (g1, 5, "run", 0, 1, 0, 0, None, "00-R1", None),
        (g1, 6, "pass", 1, 0, 1, 1, "00-W1", None, "00-Q1"),          # two-point try: never counted
        (g1, 7, "pass", 1, 0, 1, 0, "00-W1", None, "00-Q1"),          # not charted by FTN: dropped by the inner join
        (g2, 1, "pass", 1, 0, 1, 0, "00-W1", None, "00-Q1"),
    ], columns=["game_id", "play_id", "play_type", "pass_attempt", "rush_attempt", "qb_dropback", "two_point_attempt",
                "receiver_player_id", "rusher_player_id", "passer_player_id"])
    pbp["season"], pbp["season_type"], pbp["posteam"] = 2022, "REG", "BBB"
    pbp["week"] = [1] * 7 + [2]
    pbp.to_parquet(nv / "play_by_play_2022.parquet")
    flags = ["is_motion", "is_play_action", "is_screen_pass", "is_rpo", "is_qb_out_of_pocket", "is_interception_worthy",
             "is_catchable_ball", "is_contested_ball", "is_drop"]

    def ftn(game, play, week, pulled, box=0, blitz=0, rushers=0, **on):
        return {"nflverse_game_id": game, "nflverse_play_id": play, "season": 2022, "week": week, "n_defense_box": box,
                "n_blitzers": blitz, "n_pass_rushers": rushers, "date_pulled": pd.Timestamp(pulled, tz="UTC"),
                **{f: bool(on.get(f, False)) for f in flags}}
    stale = "2024-09-06 23:50"                                       # a bulk re-pull two years later: not an in-season stamp
    pd.DataFrame([ftn(g1, 1, 1, stale, box=6, blitz=1, rushers=5, is_motion=1, is_catchable_ball=1),
                  ftn(g1, 2, 1, stale, box=7, rushers=4, is_play_action=1, is_contested_ball=1, is_catchable_ball=1, is_drop=1,
                      is_qb_out_of_pocket=1),
                  ftn(g1, 3, 1, stale, box=6), ftn(g1, 4, 1, stale, box=8), ftn(g1, 5, 1, stale, box=0),
                  ftn(g1, 6, 1, stale, is_motion=1),
                  ftn(g2, 1, 2, "2022-09-24 12:00", rushers=4)]).to_parquet(nv / "ftn_charting_2022.parquet")


class Loaders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        _write_release_cache(root)
        cls._patches = [mock.patch.object(pit, "CACHE_DIR", root), mock.patch.object(pit, "_GAMES", None),
                        mock.patch("model.point_in_time.requests.get", side_effect=AssertionError("no network in tests"))]
        for p in cls._patches:
            p.start()
        cls.ngs = {t.name: t for t in pit.load_ngs_tables([2021, 2022])}
        cls.pfr = {t.name: t for t in pit.load_pfr_tables([2022])}
        cls.ftn = {t.name: t for t in pit.load_ftn_tables([2021, 2022], {"00-W1", "00-W2", "00-R1", "00-Q1"})}
        cls.kick = pit._kickoffs()

    @classmethod
    def tearDownClass(cls):
        for p in cls._patches:
            p.stop()
        cls._tmp.cleanup()

    # ---- the rule
    def test_the_release_stamps_are_the_end_of_the_first_refresh_weekday_after_the_week_ended(self):
        # 2022 week 1 ends with Monday night (Mon 20:15 ET = Tue 00:15Z, game over Tue 04:15Z)
        self.assertEqual(pit._charting_release_stamps(2)[(2022, 1)], ts("2022-09-14 23:59:59"))      # FTN: Wednesday
        self.assertEqual(pit._charting_release_stamps(3)[(2022, 1)], ts("2022-09-15 23:59:59"))      # PFR: Thursday
        self.assertEqual(pit._charting_release_stamps(4)[(2022, 1)], ts("2022-09-16 23:59:59"))      # NGS: Friday
        # never the day the week ended: a week that ends ON the refresh weekday waits a full week
        self.assertEqual(pit._charting_release_stamps(1)[(2022, 1)], ts("2022-09-20 23:59:59"))      # a Tuesday refresh

    def test_every_table_stamps_every_row_of_a_week_with_its_sources_rule(self):
        for tabs, wd in ((self.ngs, 4), (self.pfr, 3)):
            stamps = pit._charting_release_stamps(wd)
            for t in tabs.values():
                want = pd.Series(list(zip(t.df["season"], t.df["week"]))).map(stamps)
                self.assertTrue((t.df[KNOWN_AT].reset_index(drop=True) == want).all(), t.name)
                self.assertEqual(str(t.df[KNOWN_AT].dtype), "datetime64[ns, UTC]")
                self.assertIn("first", t.known_at_rule)

    def test_ftn_takes_an_in_season_date_pulled_when_later_and_ignores_a_stale_re_pull(self):
        for t in self.ftn.values():
            k = dict(zip(t.df["week"], t.df[KNOWN_AT]))
            self.assertEqual(k[1], ts("2022-09-14 23:59:59"), "week 1: the 2024 re-pull is not an in-season stamp; the rule holds")
            self.assertEqual(k[2], ts("2022-09-24 12:00"), "week 2: pulled Saturday, later than Wednesday's rule: the pull wins")

    def test_a_week_n_game_never_sees_week_n_and_thursday_night_cannot_see_last_weeks_ngs(self):
        for tabs in (self.ngs, self.pfr, self.ftn):
            for t in tabs.values():
                for g, ko in self.kick.items():
                    s, w = int(g[:4]), int(g[5:7])
                    seen = t.df[(t.df[KNOWN_AT] < ko) & (t.df["season"] == s)]
                    self.assertTrue((seen["week"] < w).all(), f"{t.name} leaks week >= {w} into {g}")
        tnf, sun = self.kick[gid(2022, 2, "CCC", "DDD")], self.kick[gid(2022, 2, "AAA", "BBB")]
        wk1 = lambda tab: tab.df[tab.df["week"] == 1][KNOWN_AT].max()                   # noqa: E731
        self.assertGreater(wk1(self.ngs["ngs"]), tnf, "Friday NGS refresh: week 1 is not out by Thursday night of week 2")
        self.assertLess(wk1(self.ngs["ngs"]), sun)
        self.assertLess(wk1(self.pfr["pfr"]), tnf, "Thursday PFR refresh lands before the 00:15Z Friday kickoff")
        self.assertLess(wk1(self.ftn["ftn"]), tnf)

    # ---- what the loaders keep
    def test_ngs_drops_season_aggregates_and_postseason_and_maps_lar_to_la(self):
        d = self.ngs["ngs"].df
        self.assertEqual(set(d["week"]), {1, 2})
        self.assertNotIn(9.0, set(d["avg_separation"]))
        self.assertEqual(self.ngs["ngs"].dropped["receiving_week0_post_or_no_id"], 2)
        w2 = d[d["player_id"] == "00-W2"].iloc[0]
        self.assertEqual((w2["team"], w2["game_id"]), ("LA", gid(2022, 1, "EEE", "LA")))
        w1 = d[(d["player_id"] == "00-W1") & (d["week"] == 1)].iloc[0]          # receiving and rushing merged into one row
        self.assertEqual((w1["avg_separation"], w1["catch_percentage"], w1["efficiency"]), (3.0, 70.0, 4.5))
        self.assertTrue(math.isnan(w1["avg_time_to_throw"]))
        cov = self.ngs["ngs_games"].df
        self.assertEqual(set(cov["game_id"]), {gid(2022, 1, "AAA", "BBB"), gid(2022, 1, "EEE", "LA"), gid(2022, 2, "CCC", "DDD")},
                         "coverage = games with a qualifier, both teams; a game nobody qualified in is not 'published'")
        self.assertEqual(len(cov), 6)

    def test_pfr_ids_map_to_gsis_through_the_players_table_and_unmapped_ids_are_counted(self):
        d = self.pfr["pfr"].df
        self.assertEqual(set(d["player_id"]), {"00-W1", "00-R1", "00-Q1"})
        self.assertEqual(self.pfr["pfr"].dropped["pfr_id_not_in_players_table"], 1)
        self.assertEqual(self.pfr["pfr"].dropped["rec_postseason"], 1)
        r = d.set_index("player_id")
        self.assertEqual((r.loc["00-W1", "drops"], r.loc["00-W1", "rat"]), (1.0, 101.0))
        self.assertEqual((r.loc["00-R1", "rush_att"], r.loc["00-R1", "rush_ybc"], r.loc["00-R1", "brk_tkl"]), (10.0, 20.0, 2.0))
        self.assertEqual(r.loc["00-Q1", "pressure_pct"], 0.3)
        self.assertTrue(math.isnan(r.loc["00-Q1", "drops"]))
        self.assertEqual(set(self.pfr["pfr_games"].df["team"]), {"AAA", "BBB"})
        # the map is the one snap_counts and combine use
        self.assertEqual(pit._pfr_to_gsis().to_dict(), {"W1pf": "00-W1", "R1pf": "00-R1", "Q1pf": "00-Q1"})

    def test_ftn_counts_per_player_game_from_the_join_to_play_by_play(self):
        d = self.ftn["ftn"].df
        g1 = d[d["week"] == 1].set_index("player_id")
        w1, w2, r1, q1 = (g1.loc[p] for p in ("00-W1", "00-W2", "00-R1", "00-Q1"))
        self.assertEqual((w1["tgt"], w1["tgt_motion"], w1["tgt_catchable"], w1["drops"]), (1.0, 1.0, 1.0, 0.0),
                         "the two-point try and the uncharted play are not counted")
        self.assertEqual((w2["tgt"], w2["tgt_pa"], w2["tgt_contested"], w2["drops"]), (1.0, 1.0, 1.0, 1.0))
        self.assertEqual((r1["car"], r1["car_box"], r1["car_light"], r1["box_sum"]), (3.0, 2.0, 1.0, 14.0),
                         "box 0 = not charted; light = 1..6")
        self.assertEqual((q1["db"], q1["db_blitz"], q1["db_oop"], q1["db_pa"], q1["db_rush_n"], q1["rushers_sum"]),
                         (2.0, 1.0, 1.0, 1.0, 2.0, 9.0))
        tm = self.ftn["ftn_team"].df.set_index("week")
        self.assertEqual((tm.loc[1, "T_tgt"], tm.loc[1, "T_tgt_motion"], tm.loc[1, "T_tgt_pa"]), (2.0, 1.0, 1.0))

    def test_ftn_2021_is_not_published_so_it_has_no_rows_and_no_request_is_made(self):
        for t in self.ftn.values():
            self.assertEqual(set(t.df["season"]), {2022})
            self.assertEqual(t.dropped["season_2021_not_published"], 1)

    def test_the_serving_path_loads_none_of_it(self):
        """serve.load_store passes pbp=False; `charting` follows pbp unless set, so the weekly run downloads nothing new."""
        tiny = RawTable("player_games", pd.DataFrame({"player_id": ["x"], KNOWN_AT: pd.Series([ts("2022-01-01")]).astype("datetime64[ns, UTC]")}), "t")
        sn = RawTable("snap_counts", pd.DataFrame({"player_id": ["x"], "position": ["WR"],
                                                   KNOWN_AT: pd.Series([ts("2022-01-01")]).astype("datetime64[ns, UTC]")}), "t")
        with mock.patch.multiple(pit, load_player_games=mock.DEFAULT, load_snap_counts=mock.DEFAULT, load_schedule_tables=mock.DEFAULT,
                                 load_xfp=mock.DEFAULT, load_injuries=mock.DEFAULT, load_depth_charts=mock.DEFAULT,
                                 load_static_tables=mock.DEFAULT, load_career_pre_cutoff=mock.DEFAULT, load_ngs_tables=mock.DEFAULT,
                                 load_pfr_tables=mock.DEFAULT, load_ftn_tables=mock.DEFAULT) as m:
            m["load_player_games"].return_value, m["load_snap_counts"].return_value = tiny, sn
            for k in ("load_schedule_tables", "load_static_tables", "load_ngs_tables", "load_pfr_tables", "load_ftn_tables"):
                m[k].return_value = []
            for k in ("load_xfp", "load_injuries", "load_depth_charts", "load_career_pre_cutoff"):
                m[k].return_value = RawTable(k, tiny.df, "t")
            pit.load_store([2022], adp=False, college=False, pbp=False)
            self.assertFalse(m["load_ngs_tables"].called or m["load_pfr_tables"].called or m["load_ftn_tables"].called)
            pit.load_store([2022], adp=False, college=False, pbp=False, charting=True)
            self.assertTrue(m["load_ngs_tables"].called and m["load_pfr_tables"].called and m["load_ftn_tables"].called)


# ================================================================== features on a synthetic store
KO = ts("2022-10-09 17:00")                                     # target: 2022 week 5, Sunday
PID, TEAM = "P1", "KC"


def wk_ko(week: int, season: int = 2022) -> pd.Timestamp:
    return KO - pd.Timedelta(days=7 * (5 - week)) - pd.Timedelta(days=364 * (2022 - season))


def g(week: int, season: int = 2022) -> str:
    return f"{season}_{week:02d}_LV_KC"


def _col(xs) -> pd.Series:
    return pd.Series(list(xs)).astype("datetime64[ns, UTC]")


def _hist_tables(weeks: list[tuple[int, int]]) -> list[RawTable]:
    """player_games / snap_counts / xfp for appearances in (season, week), each stamped at its kickoff (the loaders' rule)."""
    rows = [{"player_id": PID, "game_id": g(w, s), "season": s, "week": w, "season_type": "REG", "team": TEAM, "position": "WR",
             "fantasy_points_ppr": 10.0, "carries": 1.0, "targets": 5.0, "attempts": 0.0, "target_share": 0.2, "carry_share": 0.05,
             "receptions": 3.0, "receiving_yards": 30.0, "receiving_tds": 0.0, "rushing_yards": 2.0, "rushing_tds": 0.0,
             "passing_yards": 0.0, "passing_tds": 0.0, "passing_interceptions": 0.0, "completions": 0.0, KNOWN_AT: wk_ko(w, s)}
            for s, w in weeks]
    pg = pd.DataFrame(rows)
    pg[KNOWN_AT] = _col(pg[KNOWN_AT])
    sn = pg[["player_id", "game_id", "season", "week", "team", "position", KNOWN_AT]].assign(game_type="REG", offense_snaps=40, offense_pct=0.7)
    xf = pg[["player_id", "game_id", KNOWN_AT]].assign(total_fantasy_points_exp=9.0, rush_touchdown=0.0, rush_touchdown_exp=0.0,
                                                      rec_touchdown=0.0, rec_touchdown_exp=0.0, pass_touchdown=0.0, pass_touchdown_exp=0.0)
    return [RawTable("player_games", pg, "t"), RawTable("snap_counts", sn, "t"), RawTable("xfp", xf, "t")]


def _cov(name: str, games: list[tuple[str, int, int, pd.Timestamp]], **extra) -> RawTable:
    df = pd.DataFrame([{"game_id": gi, "season": s, "week": w, "team": TEAM, KNOWN_AT: k, **extra} for gi, s, w, k in games])
    df[KNOWN_AT] = _col(df[KNOWN_AT])
    return RawTable(name, df, "t")


def _rows(name: str, rows: list[dict]) -> RawTable:
    df = pd.DataFrame([{"player_id": PID, "team": TEAM, **r} for r in rows])
    df[KNOWN_AT] = _col(df[KNOWN_AT])
    return RawTable(name, df, "t")


def pub(week: int, season: int = 2022) -> pd.Timestamp:
    """A release stamp: the Wednesday after the game (any time between the game and the target works for these tests)."""
    return wk_ko(week, season) + pd.Timedelta(days=3)


NGS_V = {1: {"avg_separation": 2.0, "efficiency": 4.0}, 3: {"avg_separation": 3.0, "efficiency": 6.0}}      # week 2: did not qualify
PFR_V = {1: dict(drops=1.0, drop_pct=0.2, rat=90.0, rush_att=4.0, rush_ybc=8.0, rush_yac=4.0, brk_tkl=0.0, blitz=np.nan),
         2: dict(drops=0.0, drop_pct=0.0, rat=110.0, rush_att=2.0, rush_ybc=2.0, rush_yac=6.0, brk_tkl=1.0, blitz=np.nan),
         3: dict(drops=2.0, drop_pct=0.4, rat=60.0, rush_att=0.0, rush_ybc=0.0, rush_yac=0.0, brk_tkl=1.0, blitz=np.nan)}
FTN_V = {1: dict(tgt=5.0, tgt_motion=2.0, tgt_catchable=4.0, tgt_contested=1.0, drops=1.0, car=2.0, car_box=2.0, car_light=1.0, box_sum=13.0),
         3: dict(tgt=3.0, tgt_motion=1.0, tgt_catchable=3.0, tgt_contested=0.0, drops=0.0, car=0.0)}
FTN_TEAM = {1: 10.0, 2: 8.0, 3: 6.0}                             # T_tgt_motion per game


def charting_store(season: int = 2022, *, ftn: bool = True) -> RawStore:
    """Appearances weeks 1-4; the releases have published weeks 1-3 before kickoff, week 4 only after it."""
    s = season
    games = [(g(w, s), s, w, pub(w, s) if w < 4 else KO + pd.Timedelta(days=1)) for w in (1, 2, 3, 4)]
    blank_pfr = {c: np.nan for c in F.PFR_RAW}
    blank_ngs = {c: np.nan for c in F.NGS_STATS}
    tabs = [*_hist_tables([(s, w) for w in (1, 2, 3, 4)]),
            _cov("ngs_games", games), _cov("pfr_games", games),
            _rows("ngs", [{"game_id": g(w, s), "season": s, "week": w, **blank_ngs, **NGS_V[w], KNOWN_AT: pub(w, s)} for w in (1, 3)]
                  + [{"game_id": g(4, s), "season": s, "week": 4, **blank_ngs, "avg_separation": 99.0, KNOWN_AT: KO + pd.Timedelta(days=1)}]),
            _rows("pfr", [{"game_id": g(w, s), "season": s, "week": w, **blank_pfr, **PFR_V[w], KNOWN_AT: pub(w, s)} for w in (1, 2, 3)])]
    if ftn:
        blank = {c: 0.0 for c in F.FTN_C}
        tabs += [_cov("ftn_team", [x for x in games], T_tgt=20.0, T_tgt_pa=0.0, T_tgt_rpo=0.0, T_tgt_screen=0.0, T_tgt_motion=0.0),
                 _rows("ftn", [{"game_id": g(w, s), "season": s, "week": w, **blank, **FTN_V[w], KNOWN_AT: pub(w, s)} for w in (1, 3)])]
        df = tabs[-2].df
        df["T_tgt_motion"] = [FTN_TEAM.get(w, 0.0) for w in df["week"]]
        tabs[-2] = RawTable("ftn_team", df, "t")
    return RawStore(tabs)


def target(season: int = 2022, week: int = 5) -> TargetRow:
    return TargetRow(PID, season, week, TEAM, "LV", "WR", g(week, season), wk_ko(week, season), True, ("usage",))


def feats(store: RawStore, t: TargetRow | None = None) -> dict:
    t = t or target()
    return _charting_features(store, t, _history(store, PID, t, None), None)


class TrailingMath(unittest.TestCase):
    def test_every_column_is_produced(self):
        f = feats(charting_store())
        self.assertEqual(set(f), set(NEW_COLS))

    def test_ngs_l1_is_the_newest_published_game_and_l3_skips_a_non_qualifying_one(self):
        f = feats(charting_store())
        self.assertEqual(f["ngs_avg_separation_l1"], 3.0, "week 4 has ended but NGS has not published it: week 3 is the newest")
        self.assertEqual(f["ngs_avg_separation_l3"], 2.5, "weeks 1-3; week 2 did not qualify (NaN) and is skipped by the mean")
        self.assertEqual(f["ngs_efficiency_l3"], 5.0)
        self.assertTrue(math.isnan(f["ngs_avg_time_to_throw_l1"]))

    def test_a_non_qualifying_newest_game_is_nan_at_l1(self):
        st = charting_store()
        ngs = st._tables["ngs"].df
        f = feats(st.with_frame("ngs", ngs[ngs["week"] != 3]))
        self.assertTrue(math.isnan(f["ngs_avg_separation_l1"]))
        self.assertEqual(f["ngs_avg_separation_l3"], 2.0)

    def test_pfr_per_game_means_and_per_carry_ratios_of_sums(self):
        f = feats(charting_store())
        self.assertEqual((f["pfr_drops_l1"], f["pfr_rat_l1"]), (2.0, 60.0))
        self.assertTrue(same(f["pfr_drops_l3"], 1.0))
        self.assertTrue(same(f["pfr_rat_l3"], (90.0 + 110.0 + 60.0) / 3))
        self.assertTrue(same(f["pfr_drop_pct_l3"], 0.2))
        self.assertTrue(same(f["pfr_ybc_att_l3"], (8.0 + 2.0 + 0.0) / (4.0 + 2.0 + 0.0)))
        self.assertTrue(same(f["pfr_yac_att_l3"], 10.0 / 6.0))
        self.assertTrue(same(f["pfr_att_br_l3"], 6.0 / 2.0))
        self.assertTrue(math.isnan(f["pfr_ybc_att_l1"]), "no carries in the newest game: no rate")
        self.assertTrue(same(f["pfr_brk_tkl_l3"], 2.0 / 3))
        self.assertTrue(math.isnan(f["pfr_blitz_l3"]))

    def test_ftn_ratios_of_sums_with_an_uncharted_for_him_game_counting_zero(self):
        f = feats(charting_store())
        self.assertTrue(same(f["ftn_tgt_share_motion_l3"], (2.0 + 0.0 + 1.0) / (10.0 + 8.0 + 6.0)),
                        "week 2 was charted, he has no row: zero motion targets, but the team's 8 still count")
        self.assertTrue(same(f["ftn_catchable_rate_l3"], 7.0 / 8.0))
        self.assertTrue(same(f["ftn_contested_rate_l3"], 1.0 / 8.0))
        self.assertTrue(same(f["ftn_drops_pg_l3"], 1.0 / 3))
        self.assertTrue(same(f["ftn_light_box_car_share_l3"], 0.5))
        self.assertTrue(same(f["ftn_avg_box_car_l3"], 6.5))
        self.assertTrue(math.isnan(f["ftn_tgt_share_pa_l3"]), "the team had no play-action targets: nothing to divide by")
        self.assertTrue(math.isnan(f["ftn_qb_blitz_rate_l3"]), "a receiver with no dropbacks")

    def test_the_window_is_the_last_three_published_games(self):
        st = charting_store()
        later = {}
        for name in ("ngs_games", "pfr_games", "ftn_team", "ngs", "pfr", "ftn"):     # publish week 4 too, with known values
            df = st._tables[name].df.copy()
            df.loc[df["week"] == 4, KNOWN_AT] = KO - SEC
            later[name] = df
        ftn = later["ftn"]
        row = {**{c: 0.0 for c in F.FTN_C}, "player_id": PID, "team": TEAM, "game_id": g(4), "season": 2022, "week": 4,
               "tgt": 4.0, "tgt_catchable": 1.0, KNOWN_AT: KO - SEC}
        later["ftn"] = pd.concat([ftn, pd.DataFrame([row]).astype({KNOWN_AT: "datetime64[ns, UTC]"})], ignore_index=True)
        for name, df in later.items():
            st = st.with_frame(name, df)
        f = feats(st)
        self.assertEqual(f["ngs_avg_separation_l1"], 99.0)
        self.assertTrue(same(f["ngs_avg_separation_l3"], (3.0 + 99.0) / 2), "weeks 2-4; week 1 has left the window")
        self.assertTrue(math.isnan(f["pfr_drops_l1"]), "week 4 published, no PFR line for him: NaN")
        self.assertTrue(same(f["ftn_catchable_rate_l3"], (0.0 + 3.0 + 1.0) / (0.0 + 3.0 + 4.0)))

    def test_week_one_and_a_store_without_the_tables_are_all_nan(self):
        self.assertTrue(all(math.isnan(v) for v in feats(charting_store(), target(week=1)).values()))
        bare = RawStore(_hist_tables([(2022, w) for w in (1, 2, 3, 4)]))
        self.assertTrue(all(math.isnan(v) for v in feats(bare).values()))

    def test_2021_has_nan_ftn_columns_because_ftn_was_never_published_for_it(self):
        st = charting_store(2021, ftn=False)        # what load_ftn_tables yields for 2021: no ftn / ftn_team rows at all
        f = feats(st, target(2021))
        self.assertTrue(all(math.isnan(f[c]) for c in F.FAMILIES["ftn"]))
        self.assertEqual(f["ngs_avg_separation_l1"], 3.0, "NGS and PFR exist for 2021")
        self.assertEqual(f["pfr_rat_l1"], 60.0)


# ================================================================== canaries
def _canary(store: RawStore, stamp: pd.Timestamp, *, hist_stamp: pd.Timestamp | None = None, only: str | None = None,
            late: pd.Timestamp | None = None) -> RawStore:
    """A monster game for him in week 4.5 (game id CANARY): its stats rows stamped `hist_stamp` (default: the stamp), every
    charting row stamped `stamp`; with `only`, that one table's canary rows are stamped `late` instead."""
    gc = "2022_04_CANARY"
    hs = hist_stamp if hist_stamp is not None else stamp
    out = store
    for t in _hist_tables([(2022, 4)]):
        df = t.df.assign(game_id=gc, **{KNOWN_AT: hs}).astype({KNOWN_AT: "datetime64[ns, UTC]"})
        out = out.with_frame(t.name, pd.concat([out._tables[t.name].df, df], ignore_index=True))
    rows = {"ngs_games": {}, "pfr_games": {}, "ftn_team": {c: 50.0 for c in F.FTN_T},
            "ngs": {**{c: 77.0 for c in F.NGS_STATS}, "player_id": PID},
            "pfr": {**{c: 7.0 for c in F.PFR_RAW}, "player_id": PID},
            "ftn": {**{c: 5.0 for c in F.FTN_C}, "player_id": PID}}
    for name, extra in rows.items():
        k = late if only == name else stamp
        r = pd.DataFrame([{"game_id": gc, "season": 2022, "week": 4, "team": TEAM, **extra, KNOWN_AT: k}])
        r[KNOWN_AT] = _col(r[KNOWN_AT])
        out = out.with_frame(name, pd.concat([out._tables[name].df, r], ignore_index=True))
    return out


class Canaries(unittest.TestCase):
    def test_nothing_stamped_at_or_after_kickoff_changes_any_column(self):
        base = feats(charting_store())
        for stamp in (KO, KO + pd.Timedelta(hours=1), KO + pd.Timedelta(days=7)):
            dirty = feats(_canary(charting_store(), stamp))
            self.assertEqual(fingerprint(dirty), fingerprint(base), str(stamp))
        # the game has ended (stats visible at kickoff - 4h) but no release has published it: still nothing moves
        dirty = feats(_canary(charting_store(), KO, hist_stamp=KO - RESULT_LAG - pd.Timedelta(hours=1)))
        self.assertEqual(fingerprint(dirty), fingerprint(base))

    def test_control_one_second_before_kickoff_moves_l1_to_the_canarys_exact_values(self):
        live = feats(_canary(charting_store(), KO - SEC, hist_stamp=KO - RESULT_LAG - pd.Timedelta(hours=1)))
        for s in F.NGS_STATS:
            self.assertEqual(live[f"ngs_{s}_l1"], 77.0, s)
        self.assertEqual(live["pfr_drops_l1"], 7.0)
        self.assertEqual(live["pfr_ybc_att_l1"], 1.0)
        self.assertTrue(same(live["ftn_tgt_share_motion_l3"], (0.0 + 1.0 + 5.0) / (8.0 + 6.0 + 50.0)))
        moved = [c for c in NEW_COLS if fingerprint({c: live[c]}) == fingerprint({c: feats(charting_store())[c]})]
        self.assertLess(len(moved), len(NEW_COLS) // 4, f"most columns must move: unchanged {moved}")

    def test_each_table_is_gated_on_its_own(self):
        """Coverage and player rows are separate tables: a leak in one would be masked by the other being gated. Stamp the canary
        right in every table but one, which is stamped after kickoff, and that table's rows stay invisible."""
        hs, late = KO - RESULT_LAG - pd.Timedelta(hours=1), KO + pd.Timedelta(hours=1)
        base = feats(charting_store())
        f = feats(_canary(charting_store(), KO - SEC, hist_stamp=hs, only="ngs", late=late))
        self.assertTrue(math.isnan(f["ngs_avg_separation_l1"]), "the game is published, his NGS row is not visible: not qualified")
        f = feats(_canary(charting_store(), KO - SEC, hist_stamp=hs, only="ngs_games", late=late))
        self.assertEqual([c for c in F.FAMILIES["ngs"] if fingerprint({c: f[c]}) != fingerprint({c: base[c]})], [],
                         "his row is there but the game is not published: the window ignores it")
        f = feats(_canary(charting_store(), KO - SEC, hist_stamp=hs, only="pfr", late=late))
        self.assertTrue(math.isnan(f["pfr_drops_l1"]))
        f = feats(_canary(charting_store(), KO - SEC, hist_stamp=hs, only="pfr_games", late=late))
        self.assertEqual(f["pfr_drops_l1"], base["pfr_drops_l1"])
        f = feats(_canary(charting_store(), KO - SEC, hist_stamp=hs, only="ftn", late=late))
        self.assertTrue(same(f["ftn_tgt_share_motion_l3"], 1.0 / 64.0), "charted game, his row invisible: zero for him")
        f = feats(_canary(charting_store(), KO - SEC, hist_stamp=hs, only="ftn_team", late=late))
        self.assertEqual(fingerprint({c: f[c] for c in F.FAMILIES["ftn"]}), fingerprint({c: base[c] for c in F.FAMILIES["ftn"]}))

    def test_a_release_with_nothing_published_yet_is_an_empty_gate_result_not_an_error(self):
        """A Thursday-night week-2 row of a store truncated at its kickoff holds no NGS row at all (the Friday refresh has not
        happened): the indexed gate must answer an empty table with an empty frame (it used to raise IndexError)."""
        st = charting_store()
        empty = st.with_frame("ngs_games", st._tables["ngs_games"].df.iloc[:0])
        self.assertEqual(len(pit.as_of_join(empty, "ngs_games", KO, team=TEAM)), 0)
        self.assertTrue(math.isnan(feats(empty)["ngs_avg_separation_l1"]))
        self.assertEqual(feats(empty)["pfr_rat_l1"], 60.0)

    def test_the_target_games_own_rows_never_enter(self):
        base = feats(charting_store())
        st = charting_store()
        for name, extra in (("ngs", {c: 55.0 for c in F.NGS_STATS}), ("pfr", {c: 55.0 for c in F.PFR_RAW}),
                            ("ftn", {c: 55.0 for c in F.FTN_C})):
            r = pd.DataFrame([{"player_id": PID, "team": TEAM, "game_id": g(5), "season": 2022, "week": 5, **extra, KNOWN_AT: KO}])
            r[KNOWN_AT] = _col(r[KNOWN_AT])
            st = st.with_frame(name, pd.concat([st._tables[name].df, r], ignore_index=True))
        self.assertEqual(fingerprint(feats(st)), fingerprint(base))


# ================================================================== registry
class Registry(unittest.TestCase):
    def test_registered_opt_in_and_the_frozen_serving_inputs_are_unchanged(self):
        self.assertEqual(tuple(len(F.FAMILIES[f]) for f in FAMS), (26, 30, 14))
        self.assertEqual(len(NEW_COLS), len(set(NEW_COLS)))
        for f in FAMS:
            self.assertIn(f, T.OPT_IN_FAMILIES)
            for c in F.FAMILIES[f]:
                self.assertEqual(F.family_of(c), f)
                self.assertFalse(c.startswith(("y_", "base_", "spine_")))
        df = pd.DataFrame({c: [0.0, 1.0] for c in F.feature_columns()} | {"position": ["QB", "RB"],
                          "inj_report_status": [None, None], "inj_practice_status": [None, None], "wx_roof_obs": [None, None]})
        prim = T.feature_columns(df, T.PRIMARY)
        self.assertEqual(set(prim) & set(NEW_COLS), set())
        cfg = TR.load_config()
        X, _ = T.encode(df, T.feature_columns(df, TR.serving_spec(cfg), dead=("college_breakout_age",)))
        self.assertEqual(set(X.columns), set(cfg["inputs"]))
        self.assertEqual(len(cfg["inputs"]), 71)
        for f in FAMS:
            self.assertEqual(set(T.feature_columns(df, T.Spec("x", add_families=(f,)))) - set(prim), set(F.FAMILIES[f]))
            self.assertEqual(TR.compose([f"+family:{f}"], TR.serving_spec(cfg)).spec.add_families, (f,))

    def test_constants_repeated_in_features_equal_the_loaders(self):
        self.assertEqual(F.NGS_STATS, tuple(s for v in pit.NGS_STATS.values() for s in v))
        self.assertEqual(F.PFR_RAW, tuple(c for m in pit.PFR_COLUMNS.values() for c in m.values()))
        self.assertEqual(F.FTN_C, pit.FTN_COUNTS)
        self.assertEqual(F.FTN_T, pit.FTN_TEAM_COUNTS)
        self.assertTrue(set(F.PFR_STATS) - set(F.PFR_RATIOS) <= set(F.PFR_RAW))
        for num, den in F.FTN_SPEC.values():
            self.assertIn(num, F.FTN_C)
            self.assertTrue(den is None or den in F.FTN_C or den in F.FTN_T)


if __name__ == "__main__":
    unittest.main()
