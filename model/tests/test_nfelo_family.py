"""The opt-in `nfelo` family (model/nfelo.py loaders, model/features.py `_nfelo_features`): team-code mapping, the known_at rules of
every table, the line-movement timestamp guard, the trailing WEPA / film-margin math, leakage (the target game's own row, a next-week
rating, a post-kickoff scrape never enter), the conditional fetch, and the registry contract.

    python -m unittest model.tests.test_nfelo_family

No network: fetch() is driven by a fake session; every table is built by hand.
"""
from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import requests

from model import features, nfelo as N, train as T, trial as TR
from model.features import NAN, RESULT_LAG, _nfelo_features
from model.point_in_time import KNOWN_AT, LINES_CLOSE_LEAD, RawStore, RawTable, TargetRow

COLS = features.FAMILIES["nfelo"]
KO = pd.Timestamp("2025-10-19 17:00", tz="UTC")                 # target: 2025 week 7, LV at KC (KC home)
WEEK_FIRST = pd.Timestamp("2025-10-17 00:15", tz="UTC")         # Thursday game of week 7


def tgt(home: bool = True) -> TargetRow:
    team, opp = ("KC", "LV") if home else ("LV", "KC")
    return TargetRow("P1", 2025, 7, team, opp, "WR", "2025_07_LV_KC", KO, home, ("usage",))


def ts(*xs) -> pd.Series:
    return pd.to_datetime(list(xs), utc=True).astype("datetime64[ns, UTC]")


def same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-9


# ------------------------------------------------------------------ team codes and loaders (pure frame builders)
class TeamCodes(unittest.TestCase):
    def test_codes(self):
        self.assertEqual(N.team_code("LAR", 2021), "LA")
        self.assertEqual(N.team_code("OAK", 2024), "LV")
        self.assertEqual(N.team_code("OAK", 2019), "OAK")              # nflverse still says OAK in 2019
        self.assertEqual(N.team_code("LAC", 2024), "LAC")
        self.assertEqual(N.team_code("WAS", 2021), "WAS")
        self.assertEqual(N.team_code("KC", 2025), "KC")

    def test_game_ids(self):
        self.assertEqual(N.game_id("2026_06_BUF_OAK"), "2026_06_BUF_LV")
        self.assertEqual(N.game_id("2021_01_CHI_LAR"), "2021_01_CHI_LA")
        self.assertEqual(N.game_id("2019_01_DEN_OAK"), "2019_01_DEN_OAK")
        self.assertIsNone(N.game_id("garbage"))


def _games() -> pd.DataFrame:
    return pd.DataFrame({"game_id": ["2025_07_PIT_CIN", "2025_07_LV_KC", "2025_08_KC_WAS"], "season": 2025, "week": [7, 7, 8],
                         "game_type": "REG", "kickoff": ts(WEEK_FIRST, KO, "2025-10-28 00:15")})


class Loaders(unittest.TestCase):
    def setUp(self):
        g = _games()
        self.first, self.ko = N.week_first_kickoffs(g), g.set_index("game_id")["kickoff"]

    def test_units_keep_pre_only_map_codes_and_stamp_first_kickoff_minus_an_hour(self):
        raw = pd.DataFrame({"season": [2025, 2025, 2025], "week": [7, 7, 9], "team": ["KC", "OAK", "KC"]})
        for c in N.UNIT_COLS:
            raw[c] = 1.0
        for c in ("pass_off_value_post", "total_value_post"):
            raw[c] = 99.0
        df, dropped = N.units_frame(raw, self.first)
        self.assertFalse(any(c.endswith("_post") for c in df.columns), "post-game values must never be loaded")
        self.assertEqual(sorted(df["team"]), ["KC", "LV"])
        self.assertTrue((df[KNOWN_AT] == WEEK_FIRST - pd.Timedelta(hours=1)).all())
        self.assertEqual(dropped, {"week_not_in_schedule": 1})          # week 9 is not in this schedule
        self.assertEqual(str(df[KNOWN_AT].dtype), "datetime64[ns, UTC]")

    def test_qb_rows_are_per_team_with_mapped_game_ids(self):
        raw = pd.DataFrame({"season": [2025], "week": [7.0], "game_type": ["REG"], "game_id": ["2025_07_OAK_KC"],
                            "team1": ["KC"], "team2": ["OAK"], "qb1": ["Patrick Mahomes"], "qb2": ["Geno Smith"],
                            "qb1_value_pre": [230.0], "qb2_value_pre": [150.0]})
        df, dropped = N.qb_frame(raw, self.first, self.ko)
        self.assertEqual(dict(zip(df["team"], df["qb_value_pre"])), {"KC": 230.0, "LV": 150.0})
        self.assertEqual(set(df["game_id"]), {"2025_07_LV_KC"})
        self.assertEqual(dropped["game_not_in_schedule"], 0)

    def test_per_game_tables_are_stamped_at_kickoff(self):
        raw = pd.DataFrame({"game_id": ["2025_07_OAK_KC", "2025_07_OAK_KC", "2030_01_X_Y"], "season": [2025, 2025, 2030],
                            "team": ["KC", "OAK", "X"], "opponent": ["OAK", "KC", "Y"], "wepa_net": [1.0, -1.0, 0.0]})
        df, dropped = N.per_game_frame(raw, ["opponent", "wepa_net"], self.ko)
        self.assertEqual(list(df["team"]), ["KC", "LV"])
        self.assertTrue((df[KNOWN_AT] == KO).all())
        self.assertEqual(dropped, {"game_not_in_schedule": 1})

    def test_lines_use_their_own_timestamps_and_legacy_rows_the_close_rule(self):
        raw = pd.DataFrame({
            "game_id": ["2025_07_OAK_KC", "2025_07_PIT_CIN"], "season": 2025, "week": 7, "home_team": ["KC", "CIN"],
            "home_spread_open": [-3.0, -6.0], "home_spread_open_timestamp": ["2025-10-13 00:07:14-07:00", None],
            "home_spread_last": [-4.5, -5.5], "home_spread_last_timestamp": ["2025-10-19 09:51:00-07:00", None],
            "total_line_open": [45.0, None], "total_line_open_timestamp": ["2025-10-13 00:07:14-07:00", None],
            "total_line_last": [47.0, 44.0], "total_line_last_timestamp": ["2025-10-19 09:51:00-07:00", None],
            "home_spread_tickets_pct": [0.7, None], "home_spread_pct_timestamp": ["2025-10-19 09:51:00-07:00", None]})
        df, _ = N.lines_frame(raw, self.ko)
        kc = df[df["game_id"] == "2025_07_LV_KC"].set_index("kind")
        self.assertEqual(kc.loc["spread_last", KNOWN_AT], pd.Timestamp("2025-10-19 16:51", tz="UTC"))
        self.assertTrue(bool(kc.loc["spread_open", "timestamped"]))
        cin = df[df["game_id"] == "2025_07_PIT_CIN"].set_index("kind")
        self.assertEqual(set(cin.index), {"spread_open", "spread_last", "total_last"})      # absent values are not rows
        self.assertTrue((cin[KNOWN_AT] == WEEK_FIRST - LINES_CLOSE_LEAD).all())
        self.assertFalse(cin["timestamped"].any())


# ------------------------------------------------------------------ features on a synthetic store
def _weekly(rows, cols) -> pd.DataFrame:
    """rows: (season, week, team, known_at, *values)."""
    df = pd.DataFrame([r[:3] + r[4:] for r in rows], columns=["season", "week", "team", *cols])
    df[KNOWN_AT] = ts(*[r[3] for r in rows])
    return df


STAMP = WEEK_FIRST - pd.Timedelta(hours=1)
NEXT = pd.Timestamp("2025-10-24 00:15", tz="UTC")             # week 8's stamp: after the target kickoff


def _units():
    cols = N.UNIT_COLS
    return _weekly([(2025, 7, "KC", STAMP, 3.0, 0.5, 1.0, 0.2, 3.5, 1.2, 4.7),
                    (2025, 7, "LV", STAMP, -1.0, 0.1, -2.0, -0.4, -0.9, -2.4, -3.3),
                    (2025, 8, "KC", NEXT, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0),          # next week: never
                    (2025, 6, "KC", STAMP - pd.Timedelta(days=7), 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0)], cols)


def _elo():
    return _weekly([(2025, 7, "KC", STAMP, 1600.0, 0.0), (2025, 7, "LV", STAMP, 1450.0, -30.0),
                    (2025, 8, "KC", NEXT, 1700.0, 0.0)], ["elo", "qb_adj"])


def _qb():
    df = pd.DataFrame({"season": 2025, "week": 7, "game_id": "2025_07_LV_KC", "team": ["KC", "LV"], "qb": ["A", "B"],
                       "qb_value_pre": [230.0, 150.0]})
    df[KNOWN_AT] = ts(STAMP, STAMP)
    return df


def _wk(w: int) -> pd.Timestamp:
    return KO - pd.Timedelta(days=7 * (7 - w))


def _wepa(extra=()):
    rows = [("g1", 2025, "KC", _wk(1), 2.0), ("g2", 2025, "KC", _wk(2), -1.0), ("g3", 2025, "KC", _wk(3), 5.0),
            ("p1", 2024, "KC", _wk(1) - pd.Timedelta(days=200), 50.0),              # last season: not season-to-date
            ("h1", 2025, "LV", _wk(1), -3.0), ("h2", 2025, "LV", _wk(2), -5.0),
            ("2025_07_LV_KC", 2025, "KC", KO, 40.0), ("2025_07_LV_KC", 2025, "LV", KO, -40.0), *extra]
    df = pd.DataFrame([(r[0], r[1], r[2], r[4]) for r in rows], columns=["game_id", "season", "team", "wepa_net"])
    df[KNOWN_AT] = ts(*[r[3] for r in rows])
    return df


def _film(games):
    df = pd.DataFrame([(g[0], g[1], g[2], g[4]) for g in games], columns=["game_id", "season", "team", "film_margin_predictive"])
    df[KNOWN_AT] = ts(*[g[3] for g in games])
    return df


FILM = [("p17", 2024, "KC", _wk(1) - pd.Timedelta(days=60), 10.0), ("p18", 2024, "KC", _wk(1) - pd.Timedelta(days=53), 20.0),
        ("f1", 2025, "KC", _wk(1), 1.0), ("f2", 2025, "KC", _wk(2), 2.0), ("f3", 2025, "KC", _wk(3), 3.0),
        ("2025_07_LV_KC", 2025, "KC", KO, 99.0)]


def _results(gids_kicks):
    df = pd.DataFrame({"game_id": [g for g, _ in gids_kicks], "season": [2024 if g.startswith("p") else 2025 for g, _ in gids_kicks],
                       "week": 1, "game_type": "REG", "team": "KC", "opponent": "X", "points_for": 1, "points_against": 0,
                       "is_home": True})
    df[KNOWN_AT] = ts(*[k for _, k in gids_kicks])
    return df


def _lines(last_ts, *, tickets=0.7):
    rows = [("spread_open", -3.0, KO - pd.Timedelta(days=6)), ("spread_last", -4.5, last_ts),
            ("total_open", 45.0, KO - pd.Timedelta(days=6)), ("total_last", 47.0, last_ts), ("tickets", tickets, last_ts)]
    df = pd.DataFrame({"game_id": "2025_07_LV_KC", "season": 2025, "week": 7, "home_team": "KC",
                       "kind": [r[0] for r in rows], "value": [r[1] for r in rows], "timestamped": True})
    df[KNOWN_AT] = ts(*[r[2] for r in rows])
    return df.sort_values(KNOWN_AT, kind="mergesort").reset_index(drop=True)


def store(*, last_ts=KO - pd.Timedelta(minutes=9), wepa_extra=(), film=FILM, results=None) -> RawStore:
    results = results if results is not None else [(g[0], g[3]) for g in film if g[0] != "2025_07_LV_KC"]
    return RawStore([RawTable("nf_units", _units(), "t"), RawTable("nf_elo", _elo(), "t"), RawTable("nf_qb", _qb(), "t"),
                     RawTable("nf_wepa", _wepa(wepa_extra), "t"), RawTable("nf_film", _film(film), "t"),
                     RawTable("nf_lines", _lines(last_ts), "t"), RawTable("game_results", _results(results), "t")])


class Features(unittest.TestCase):
    def test_every_column_hand_computed_home_view(self):
        f = _nfelo_features(store(), tgt(), None)
        self.assertEqual(set(f), set(COLS))
        self.assertEqual(len(COLS), 21)
        self.assertEqual((f["nf_tm_pass_off"], f["nf_tm_rush_off"], f["nf_tm_total"]), (3.0, 0.5, 4.7))
        self.assertEqual((f["nf_opp_pass_def"], f["nf_opp_rush_def"], f["nf_opp_total"]), (-2.0, -0.4, -3.3))
        self.assertEqual((f["nf_pass_edge"], f["nf_rush_edge"]), (3.0 - -2.0, 0.5 - -0.4))
        self.assertEqual((f["nf_elo"], f["nf_opp_elo"], f["nf_elo_dif"], f["nf_qb_adj"], f["nf_opp_qb_adj"]),
                         (1600.0, 1450.0, 150.0, 0.0, -30.0))
        self.assertEqual((f["nf_qb_value"], f["nf_opp_qb_value"]), (230.0, 150.0))
        self.assertTrue(same(f["nf_wepa_net"], (2.0 - 1.0 + 5.0) / 3))          # never last season, never the game itself
        self.assertTrue(same(f["nf_opp_wepa_net"], -4.0))
        self.assertTrue(same(f["nf_film_margin_l4"], (20.0 + 1.0 + 2.0 + 3.0) / 4))   # last 4 reaches into last season
        self.assertEqual(f["nf_spread_move"], -1.5)                            # home view: -3 -> -4.5, KC more favoured
        self.assertEqual(f["nf_total_move"], 2.0)
        self.assertEqual(f["nf_tickets_pct"], 0.7)

    def test_away_view_flips_the_spread_and_tickets(self):
        f = _nfelo_features(store(), tgt(home=False), None)
        self.assertEqual(f["nf_spread_move"], 1.5)
        self.assertTrue(same(f["nf_tickets_pct"], 0.3))
        self.assertEqual(f["nf_total_move"], 2.0)
        self.assertEqual((f["nf_qb_value"], f["nf_opp_qb_value"]), (150.0, 230.0))
        self.assertEqual((f["nf_tm_pass_off"], f["nf_opp_pass_def"]), (-1.0, 1.0))

    def test_a_last_line_scraped_at_or_after_kickoff_falls_back_to_the_open(self):
        for late in (KO, KO + pd.Timedelta(minutes=1)):
            f = _nfelo_features(store(last_ts=late), tgt(), None)
            self.assertEqual(f["nf_spread_move"], 0.0)
            self.assertEqual(f["nf_total_move"], 0.0)
            self.assertTrue(math.isnan(f["nf_tickets_pct"]), "the tickets snapshot carries the same late timestamp")
        f = _nfelo_features(store(last_ts=KO - pd.Timedelta(nanoseconds=1)), tgt(), None)
        self.assertEqual(f["nf_spread_move"], -1.5)

    def test_wepa_result_cutoff_and_own_game(self):
        base = _nfelo_features(store(), tgt(), None)
        at_cut = _nfelo_features(store(wepa_extra=[("g6", 2025, "KC", KO - RESULT_LAG, 100.0)]), tgt(), None)
        self.assertEqual(features.fingerprint(at_cut), features.fingerprint(base))
        inside = _nfelo_features(store(wepa_extra=[("g6", 2025, "KC", KO - RESULT_LAG - pd.Timedelta(seconds=1), 100.0)]), tgt(), None)
        self.assertTrue(same(inside["nf_wepa_net"], (2.0 - 1.0 + 5.0 + 100.0) / 4))

    def test_film_staleness_guard(self):
        # the team played g4 (a completed game the film source does not have yet): the window would be stale -> NaN
        res = [(g[0], g[3]) for g in FILM if g[0] != "2025_07_LV_KC"] + [("f4", _wk(4))]
        f = _nfelo_features(store(results=res), tgt(), None)
        self.assertTrue(math.isnan(f["nf_film_margin_l4"]))

    def test_no_future_or_own_row_enters_any_column(self):
        """Canaries: a copy of every table truncated strictly before kickoff gives identical features, and the dirty rows exist."""
        s = store()
        clean = s.truncated_before(KO)
        self.assertEqual(features.fingerprint(_nfelo_features(clean, tgt(), None)),
                         features.fingerprint(_nfelo_features(s, tgt(), None)))
        self.assertIn(99.0, set(s._tables["nf_units"].df["pass_off_value_pre"]))
        self.assertIn("2025_07_LV_KC", set(s._tables["nf_wepa"].df["game_id"]))

    def test_an_empty_store_gives_nan(self):
        f = _nfelo_features(RawStore([]), tgt(), None)
        self.assertTrue(all(math.isnan(f[c]) for c in COLS))


# ------------------------------------------------------------------ the conditional fetch (no network)
class _Resp:
    def __init__(self, status, content=b"", headers=None):
        self.status_code, self.content, self.headers = status, content, headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class Fetch(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.env = mock.patch.dict(os.environ, {"NFELO_OFFLINE": "0"})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_download_then_304_then_network_failure(self):
        calls = []

        class S:
            def __init__(self, resp):
                self.resp = resp

            def get(self, url, headers=None, timeout=None):
                calls.append(headers)
                if isinstance(self.resp, Exception):
                    raise self.resp
                return self.resp

        p = N.fetch("units.csv", cache_dir=self.dir, session=S(_Resp(200, b"a,b\n1,2\n", {"ETag": '"x1"'})))
        self.assertEqual(p.read_bytes(), b"a,b\n1,2\n")
        p = N.fetch("units.csv", cache_dir=self.dir, session=S(_Resp(304)))
        self.assertEqual(calls[-1], {"If-None-Match": '"x1"'})
        self.assertEqual(p.read_bytes(), b"a,b\n1,2\n")
        p = N.fetch("units.csv", cache_dir=self.dir, session=S(requests.ConnectionError("down")))
        self.assertEqual(p.read_bytes(), b"a,b\n1,2\n")
        self.assertIsNone(N.fetch("elo.csv", cache_dir=self.dir, session=S(requests.ConnectionError("down"))))

    def test_offline_never_calls(self):
        class Boom:
            def get(self, *a, **k):
                raise AssertionError("network used")
        with mock.patch.dict(os.environ, {"NFELO_OFFLINE": "1"}):
            self.assertIsNone(N.fetch("lines.csv", cache_dir=self.dir, session=Boom()))


# ------------------------------------------------------------------ registry
class Registry(unittest.TestCase):
    def test_registered_opt_in_and_the_primary_and_serving_inputs_are_unchanged(self):
        self.assertIn("nfelo", T.OPT_IN_FAMILIES)
        for c in COLS:
            self.assertEqual(features.family_of(c), "nfelo")
        df = pd.DataFrame({c: [0.0, 1.0] for c in features.feature_columns()} | {"position": ["QB", "RB"],
                          "inj_report_status": [None, None], "inj_practice_status": [None, None], "wx_roof_obs": [None, None]})
        prim = T.feature_columns(df, T.PRIMARY)
        self.assertEqual(set(prim) & set(COLS), set())
        cfg = TR.load_config()
        X, _ = T.encode(df, T.feature_columns(df, TR.serving_spec(cfg), dead=("college_breakout_age",)))
        self.assertEqual(set(X.columns), set(cfg["inputs"]))
        self.assertFalse(set(COLS) & set(cfg["inputs"]), "nfelo is not adopted: none of its columns are served")
        self.assertNotIn("nfelo", TR.serving_spec(cfg).add_families)
        both = T.feature_columns(df, T.Spec("x", add_families=("nfelo", "pepa")))
        self.assertEqual(set(both) - set(prim), set(COLS) | set(features.FAMILIES["pepa"]))

    def test_coaches_is_not_loaded(self):
        self.assertNotIn("coaches.csv", N.SOURCES)


if __name__ == "__main__":
    unittest.main()
