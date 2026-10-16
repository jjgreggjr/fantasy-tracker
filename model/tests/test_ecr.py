"""The opt-in `ecr` family: model/fetch_ecr.py (week assignment from scrape_date, latest scrape before each kickoff, known_at,
id mapping) and features._ecr_features (week-matched, through the gate; a scrape at or after kickoff never enters).

    python -m unittest model.tests.test_ecr

No network: the archive, the crosswalk and the schedule are built by hand.
"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from model import features, train as T, trial as TR
from model import fetch_ecr as E
from model.point_in_time import KNOWN_AT, RawStore, RawTable, TargetRow

COLS = features.FAMILIES["ecr"]
UTC = "UTC"


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=UTC)


# 2024 weeks 6 and 7: Thursday 00:15Z (EDT), Sunday 17:00Z, Monday 00:15Z. Week 7 the same, one week later.
GAMES = pd.DataFrame({
    "season": 2024, "week": [6, 6, 6, 7, 7, 7],
    "game_id": ["g6thu", "g6sun", "g6mon", "g7thu", "g7sun", "g7mon"],
    "kickoff": pd.to_datetime(["2024-10-11 00:15", "2024-10-13 17:00", "2024-10-15 00:15",
                               "2024-10-18 00:15", "2024-10-20 17:00", "2024-10-22 00:15"], utc=True)})


def raw_row(fp_id, pos, ecr, date, typ="wp", **kw) -> dict:
    r = {"id": fp_id, "player": f"P{fp_id}", "pos": pos, "team": "KC", "ecr": ecr, "sd": 1.0, "best": ecr - 1,
         "worst": ecr + 2, "player_owned_avg": 90.0, "ecr_type": typ, "scrape_date": date,
         "yahoo_id": None, "cbs_id": None, "sportsdata_id": None}
    r.update(kw)
    return r


IDS = pd.DataFrame({"fantasypros_id": ["1", "2", "3", None, "9", "9"],
                    "gsis_id": ["00-1", "00-2", "00-3", "00-4", "00-9a", "00-9b"],
                    "yahoo_id": [None, None, None, "777", None, None],
                    "cbs_id": None, "sportradar_id": None})


class WeekAssignment(unittest.TestCase):
    def test_a_friday_scrape_belongs_to_the_week_whose_next_kickoff_follows_it(self):
        k = pd.Series(pd.to_datetime(["2024-10-11", "2024-10-18", "2024-10-14"], utc=True).astype("datetime64[ns, UTC]"))
        m = E.assign_weeks(k, GAMES).set_index(KNOWN_AT)
        self.assertEqual(m.loc[ts("2024-10-11"), "week"], 6)        # Fri 00:00Z, before the 00:15Z Thursday game: week 6
        self.assertEqual(m.loc[ts("2024-10-18"), "week"], 7)
        self.assertEqual(m.loc[ts("2024-10-14"), "week"], 6)        # Monday before MNF: still week 6

    def test_strictly_after_and_the_offseason_gap(self):
        k = pd.Series(pd.to_datetime(["2024-10-13 17:00", "2024-09-01", "2024-11-01"], utc=True, format="ISO8601").astype("datetime64[ns, UTC]"))
        m = E.assign_weeks(k, GAMES).set_index(KNOWN_AT)
        self.assertEqual(m.loc[ts("2024-10-13 17:00"), "week"], 6)  # exactly at Sunday kickoff: the next one (MNF) decides
        self.assertTrue(pd.isna(m.loc[ts("2024-09-01"), "week"]))   # > 8 days before the next kickoff: dropped
        self.assertTrue(pd.isna(m.loc[ts("2024-11-01"), "week"]))   # nothing after it


class Build(unittest.TestCase):
    def frame(self, rows) -> tuple[pd.DataFrame, dict]:
        return E.build(pd.DataFrame(rows), IDS, GAMES, first_season=2024)

    def test_known_at_is_the_scrape_date_and_ranks_come_from_the_whole_page(self):
        rows = [raw_row(5, "WR", 1.0, "2024-10-11"),                  # unmatched, still ranks first
                raw_row(1, "WR", 2.5, "2024-10-11"), raw_row(2, "WR", 2.5, "2024-10-11"),
                raw_row(3, "QB", 1.2, "2024-10-11"),
                raw_row(1, "WR", 14.0, "2024-10-11", typ="wsf"), raw_row(3, "QB", 2.0, "2024-10-11", typ="wsf"),
                raw_row(3, "QB", 1.0, "2024-10-11", typ="ro"),          # not a weekly type
                raw_row(8, "K", 1.0, "2024-10-11")]                     # not a skill position
        df, _ = self.frame(rows)
        self.assertEqual(len(df), 4)
        self.assertTrue((df[KNOWN_AT] == ts("2024-10-11")).all())
        self.assertEqual(str(df[KNOWN_AT].dtype), "datetime64[ns, UTC]")
        self.assertTrue(((df["season"] == 2024) & (df["week"] == 6)).all())
        r = df.set_index("fp_id")
        self.assertEqual((r.loc["5", "ecr_pos_rank"], r.loc["1", "ecr_pos_rank"], r.loc["2", "ecr_pos_rank"]), (1.0, 2.0, 2.0))
        self.assertEqual(r.loc["3", "ecr_pos_rank"], 1.0)
        self.assertEqual((r.loc["1", "ecr_sf_rank"], r.loc["3", "ecr_sf_rank"]), (14.0, 2.0))
        self.assertTrue(math.isnan(r.loc["2", "ecr_sf_rank"]))

    def test_id_mapping_fp_id_then_alternative_ids_and_never_an_ambiguous_key(self):
        rows = [raw_row(1, "WR", 1.0, "2024-10-11"), raw_row(4, "WR", 2.0, "2024-10-11", yahoo_id=777),
                raw_row(9, "WR", 3.0, "2024-10-11"), raw_row(6, "WR", 4.0, "2024-10-11")]
        df, _ = self.frame(rows)
        got = dict(zip(df["fp_id"], df["player_id"]))
        self.assertEqual(got["1"], "00-1")
        self.assertEqual(got["4"], "00-4")                              # no FP id in the crosswalk: Yahoo id matched
        self.assertTrue(pd.isna(got["9"]))                              # FP id 9 maps to two gsis ids: no match
        self.assertTrue(pd.isna(got["6"]))
        tab, dropped = E.table_frame(df)
        self.assertEqual(set(tab["player_id"]), {"00-1", "00-4"})
        self.assertEqual(dropped["no_gsis_id"], 2)

    def test_latest_scrape_before_each_kickoff_per_player_week(self):
        # week 7: Tue and Wed scrapes before the Thursday game, a Fri scrape after it (before Sunday and Monday)
        rows = [raw_row(1, "WR", 9.0, "2024-10-15 12:00"), raw_row(1, "WR", 8.0, "2024-10-16"),
                raw_row(1, "WR", 7.0, "2024-10-18 06:00"),
                raw_row(2, "WR", 9.0, "2024-10-16"),                    # player 2 is not on the Friday page
                raw_row(3, "QB", 5.0, "2024-10-16")]
        rows.append(raw_row(3, "QB", 4.0, "2024-10-18 06:00"))
        df, dropped = self.frame(rows)
        p1 = df[df["fp_id"] == "1"]
        self.assertEqual(sorted(p1["ecr_pos_ecr"]), [7.0, 8.0])         # Tue superseded by Wed for every kickoff
        self.assertEqual(dropped["superseded_scrape"], 1)
        self.assertEqual(list(df[df["fp_id"] == "2"][KNOWN_AT]), [ts("2024-10-16")])   # his newest stays
        self.assertTrue((df["week"] == 7).all())


class Loader(unittest.TestCase):
    def test_the_table_loads_from_the_fetched_file_and_is_absent_without_it(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ecr_weekly.parquet"
            self.assertIsNone(E.load_ecr_table(p))
            df, _ = E.build(pd.DataFrame([raw_row(1, "WR", 1.0, "2024-10-11")]), IDS, GAMES, first_season=2024)
            df.to_parquet(p, index=False)
            t = E.load_ecr_table(p)
            self.assertEqual(t.name, "ecr")
            self.assertTrue(t.proxy)
            self.assertEqual(list(t.df[KNOWN_AT]), [ts("2024-10-11")])


# ------------------------------------------------------------------ the feature, through the gate
def ecr_row(week: int, known: str, pos_rank: float, *, sf=10.0, sd=1.5, best=2.0, worst=9.0, owned=80.0,
            season: int = 2024, pid: str = "P1") -> dict:
    return {"player_id": pid, "season": season, "week": week, "position": "WR", "scrape_date": known[:10],
            "ecr_sf_rank": sf, "ecr_pos_ecr": pos_rank, "ecr_pos_rank": pos_rank, "ecr_sd": sd, "ecr_best": best,
            "ecr_worst": worst, "ecr_owned_pct": owned, KNOWN_AT: ts(known)}


def store(rows: list[dict]) -> RawStore:
    df = pd.DataFrame(rows)
    df[KNOWN_AT] = pd.to_datetime(df[KNOWN_AT], utc=True).astype("datetime64[ns, UTC]")
    return RawStore([RawTable("ecr", df, "test")])


def target(week: int, kickoff: str, season: int = 2024) -> TargetRow:
    return TargetRow("P1", season, week, "KC", "LV", "WR", f"g{week}", ts(kickoff), True, ("usage",))


SUN7 = "2024-10-20 17:00"


class Feature(unittest.TestCase):
    BASE = [ecr_row(6, "2024-10-11", 12.0), ecr_row(7, "2024-10-18", 8.0, sf=20.0)]

    def test_values_and_the_week_over_week_delta(self):
        f = features._ecr_features(store(self.BASE), target(7, SUN7), None)
        self.assertEqual(set(f), set(COLS))
        self.assertEqual((f["ecr_rank"], f["ecr_pos_rank"], f["ecr_sd"]), (20.0, 8.0, 1.5))
        self.assertEqual((f["ecr_best"], f["ecr_worst"], f["ecr_spread"], f["ecr_owned_pct"]), (2.0, 9.0, 7.0, 80.0))
        self.assertEqual(f["ecr_delta_l1"], 8.0 - 12.0)

    def test_no_table_or_no_row_this_week_is_all_nan(self):
        empty = RawStore([])
        self.assertTrue(all(math.isnan(v) for v in features._ecr_features(empty, target(7, SUN7), None).values()))
        f = features._ecr_features(store(self.BASE[:1]), target(7, SUN7), None)     # only last week's row: not used
        self.assertTrue(all(math.isnan(v) for v in f.values()))
        f = features._ecr_features(store(self.BASE[1:]), target(7, SUN7), None)     # no week 6 row: no delta
        self.assertEqual(f["ecr_pos_rank"], 8.0)
        self.assertTrue(math.isnan(f["ecr_delta_l1"]))

    def test_a_scrape_at_or_after_kickoff_never_enters(self):
        base = features._ecr_features(store(self.BASE), target(7, SUN7), None)
        for when in (SUN7, "2024-10-20 17:00:01", "2024-10-21"):        # at kickoff, a second after, the next day
            dirty = features._ecr_features(store(self.BASE + [ecr_row(7, when, 1.0, sf=1.0)]), target(7, SUN7), None)
            self.assertEqual(features.fingerprint(dirty), features.fingerprint(base), when)
        inside = features._ecr_features(store(self.BASE + [ecr_row(7, "2024-10-20 16:59:59", 1.0)]), target(7, SUN7), None)
        self.assertEqual(inside["ecr_pos_rank"], 1.0)                    # one second before: the newest wins
        # a later week's row known before kickoff (mis-stamped) still cannot stand in for this week
        f = features._ecr_features(store(self.BASE + [ecr_row(8, "2024-10-19", 1.0)]), target(7, SUN7), None)
        self.assertEqual(f["ecr_pos_rank"], 8.0)

    def test_a_thursday_game_before_the_friday_scrape_does_not_get_last_weeks_ranking(self):
        late = [ecr_row(6, "2024-10-11", 12.0), ecr_row(7, "2024-10-18 06:00", 8.0)]
        f = features._ecr_features(store(late), target(7, "2024-10-18 00:15"), None)
        self.assertTrue(all(math.isnan(v) for v in f.values()))

    def test_the_gate_is_the_only_path(self):
        import inspect
        src = inspect.getsource(features._ecr_features)
        self.assertIn('as_of_join(store, "ecr", t.kickoff', src)
        self.assertNotIn("_tables", src)
        full = inspect.getsource(features.build_features)
        self.assertIn("_ecr_features(store, t, trace)", full)


class Registry(unittest.TestCase):
    def test_ecr_is_opt_in_and_the_primary_and_serving_inputs_are_unchanged(self):
        self.assertIn("ecr", T.OPT_IN_FAMILIES)
        self.assertEqual(len(COLS), 8)
        for c in COLS:
            self.assertEqual(features.family_of(c), "ecr")
        df = pd.DataFrame({c: [0.0, 1.0] for c in features.feature_columns()} | {"position": ["QB", "RB"],
                          "inj_report_status": [None, None], "inj_practice_status": [None, None], "wx_roof_obs": [None, None]})
        prim = T.feature_columns(df, T.PRIMARY)
        self.assertEqual(set(prim) & set(COLS), set())
        cfg = TR.load_config()
        serving = T.feature_columns(df, TR.serving_spec(cfg), dead=("college_breakout_age",))
        X, _ = T.encode(df, serving)
        self.assertEqual(set(X.columns), set(cfg["inputs"]))
        self.assertEqual(len(cfg["inputs"]), 71)
        added = T.feature_columns(df, T.Spec("x", add_families=("ecr",)))
        self.assertEqual(set(added) - set(prim), set(COLS))
        self.assertRegex(TR.list_families(cfg), r"ecr  \[0/8 not in serving, opt-in\]")


if __name__ == "__main__":
    unittest.main()
