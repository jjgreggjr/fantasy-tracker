"""The opt-in `pepa` family (model/features.py `_pepa_features`): hand-computed windows on a synthetic history, the ratio and
per-opportunity definitions, how NaN EPA / CPOE from the weekly stats enter `Hist`, the registry contract (the frozen primary's
inputs are untouched) and leakage on a synthetic store (the target game's own row never enters).

    python -m unittest model.tests.test_pepa_family

No network, no real matrix.
"""
from __future__ import annotations

import math
import unittest

import pandas as pd

from model import features, train as T, trial as TR
from model.features import (ATT, CAR, CPOE, EPA_C, EPA_P, EPA_R, N_VALUES, NAN, PASSY, PAY, PFD, RAY, REC, RECY, RFD, RYAC,
                            SACK, TGT, UFD, Hist, RESULT_LAG, _history, _pepa_features)
from model.point_in_time import KNOWN_AT, PEPA_STATS_COLS, RawStore, RawTable, TargetRow

COLS = features.FAMILIES["pepa"]
KO = pd.Timestamp("2025-10-19 17:00", tz="UTC")


def target(week: int = 7, season: int = 2025) -> TargetRow:
    return TargetRow("P1", season, week, "KC", "LV", "WR", f"{season}_{week:02d}_LV_KC", KO, True, ("usage",))


def row(gid: str, week: int, season: int = 2025, **vals) -> tuple:
    v = [0.0] * N_VALUES
    v[CPOE] = NAN
    for k, x in vals.items():
        v[globals()[k]] = x
    return (gid, season, week, True, True, "KC", *v)


def same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-9


class HandComputed(unittest.TestCase):
    # a QB-ish line so every column is exercised: four games this season, one last season (must not count)
    G = [row("p", 17, season=2024, EPA_P=99.0, ATT=99.0, TGT=0.0),
         row("g1", 1, EPA_P=4.0, EPA_R=1.0, EPA_C=0.0, ATT=30.0, SACK=2.0, CAR=4.0, TGT=0.0, CPOE=2.0, PASSY=250.0, PAY=200.0,
             PFD=12.0, UFD=1.0),
         row("g2", 2, EPA_P=-6.0, EPA_R=0.5, ATT=40.0, SACK=4.0, CAR=2.0, CPOE=-3.0, PASSY=200.0, PAY=300.0, PFD=9.0),
         row("g3", 3, EPA_P=10.0, EPA_R=-1.0, EPA_C=2.0, ATT=20.0, SACK=0.0, CAR=6.0, TGT=1.0, REC=1.0, RECY=15.0, RAY=10.0,
             RYAC=5.0, RFD=1.0, CPOE=5.0, PASSY=220.0, PAY=100.0, PFD=11.0),
         row("g4", 5, EPA_P=2.0, ATT=10.0, SACK=1.0, CAR=0.0, CPOE=NAN, PASSY=60.0, PAY=40.0, PFD=3.0)]

    def test_windows_counts_ratios_and_rates(self):
        f = _pepa_features(Hist(self.G), target())
        self.assertEqual(set(f), set(COLS))
        self.assertEqual(len(COLS), 44)
        # l1 = g4 only; l3 = g2, g3, g4; std = g1..g4 (never last season's game)
        self.assertEqual(f["pepa_pass_epa_l1"], 2.0)
        self.assertTrue(same(f["pepa_pass_epa_l3"], (-6.0 + 10.0 + 2.0) / 3))
        self.assertTrue(same(f["pepa_pass_epa_std"], (4.0 - 6.0 + 10.0 + 2.0) / 4))
        self.assertTrue(same(f["pepa_sacks_std"], 7.0 / 4))
        self.assertTrue(same(f["pepa_pass_fd_l3"], (9.0 + 11.0 + 3.0) / 3))
        self.assertTrue(same(f["pepa_rush_fd_std"], 0.25))
        # cpoe: attempt-weighted over games that have one (g4's NaN carries no weight)
        self.assertTrue(math.isnan(f["pepa_cpoe_l1"]))
        self.assertTrue(same(f["pepa_cpoe_l3"], (40 * -3.0 + 20 * 5.0) / 60))
        self.assertTrue(same(f["pepa_cpoe_std"], (30 * 2.0 + 40 * -3.0 + 20 * 5.0) / 90))
        # pacr / racr: ratio of sums
        self.assertTrue(same(f["pepa_pacr_l3"], (200.0 + 220.0 + 60.0) / (300.0 + 100.0 + 40.0)))
        self.assertTrue(same(f["pepa_racr_std"], 15.0 / 10.0))
        self.assertTrue(math.isnan(f["pepa_racr_l1"]), "no air yards: NaN, not 0")
        # per-opportunity rates over the last three
        self.assertTrue(same(f["pepa_epa_per_opp_l3"], ((-6 + 0.5) + (10 - 1 + 2) + 2) / ((40 + 4 + 2) + (20 + 0 + 6 + 1) + (10 + 1))))
        self.assertTrue(same(f["pepa_pass_epa_per_db_l3"], 6.0 / (70 + 5)))
        self.assertTrue(same(f["pepa_rec_epa_per_tgt_l3"], 2.0))
        self.assertTrue(same(f["pepa_rush_epa_per_car_l3"], -0.5 / 8))
        self.assertTrue(same(f["pepa_yac_per_rec_l3"], 5.0))

    def test_week_one_is_all_nan(self):
        f = _pepa_features(Hist(self.G[:1]), target(week=1))
        self.assertTrue(all(math.isnan(f[c]) for c in COLS))

    def test_fewer_than_three_games(self):
        f = _pepa_features(Hist(self.G[:2]), target(week=2))
        for n in ("pass_epa", "sacks", "cpoe", "pacr"):
            self.assertTrue(same(f[f"pepa_{n}_l1"], f[f"pepa_{n}_l3"]), n)
            self.assertTrue(same(f[f"pepa_{n}_l1"], f[f"pepa_{n}_std"]), n)


class Registry(unittest.TestCase):
    def test_registered_opt_in_and_the_primary_and_serving_inputs_are_unchanged(self):
        self.assertIn("pepa", T.OPT_IN_FAMILIES)
        self.assertEqual(len(set(COLS)), len(COLS))
        for c in COLS:
            self.assertEqual(features.family_of(c), "pepa")
        df = pd.DataFrame({c: [0.0, 1.0] for c in features.feature_columns()} | {"position": ["QB", "RB"],
                          "inj_report_status": [None, None], "inj_practice_status": [None, None], "wx_roof_obs": [None, None]})
        prim = T.feature_columns(df, T.PRIMARY)
        self.assertEqual(set(prim) & set(COLS), set())
        cfg = TR.load_config()
        X, _ = T.encode(df, T.feature_columns(df, TR.serving_spec(cfg), dead=("college_breakout_age",)))
        self.assertEqual(set(X.columns), set(cfg["inputs"]))
        self.assertFalse(set(COLS) & set(cfg["inputs"]), "pepa is not adopted: none of its columns are served")
        self.assertNotIn("pepa", TR.serving_spec(cfg).add_families)
        self.assertEqual(set(T.feature_columns(df, T.Spec("x", add_families=("pepa",)))) - set(prim), set(COLS))

    def test_no_duplicate_of_an_existing_feature(self):
        """aDOT, WOPR and air-yards share already exist (pbp_usage); the eff family has the yards / TD / catch rates."""
        others = {c for f, cs in features.FAMILIES.items() if f != "pepa" for c in cs}
        self.assertFalse(any("adot" in c or "wopr" in c or "ay_share" in c for c in COLS))
        self.assertEqual(set(COLS) & others, set())
        self.assertIn("pbp_adot_t3", others)


# ------------------------------------------------------------------ through the real history path
def _ts(xs) -> pd.Series:
    return pd.to_datetime(list(xs), utc=True).astype("datetime64[ns, UTC]")


def _store(games, *, pepa_cols: bool = True) -> RawStore:
    """games: (game_id, week, kickoff, passing_epa, targets, receiving_epa)."""
    n = len(games)
    pg = pd.DataFrame({
        "player_id": "P1", "game_id": [g[0] for g in games], "season": 2025, "week": [g[1] for g in games],
        "season_type": "REG", "team": "KC", "position": "WR", "fantasy_points_ppr": 10.0, "carries": 0.0,
        "targets": [g[4] for g in games], "attempts": 0.0, "target_share": 0.2, "carry_share": 0.0, "receptions": 3.0,
        "receiving_yards": 40.0, "receiving_tds": 0.0, "rushing_yards": 0.0, "rushing_tds": 0.0, "passing_yards": 0.0,
        "passing_tds": 0.0, "passing_interceptions": 0.0, "completions": 0.0, KNOWN_AT: _ts(g[2] for g in games)})
    if pepa_cols:
        for c in PEPA_STATS_COLS:
            pg[c] = 1.0
        pg["passing_epa"] = [g[3] for g in games]
        pg["receiving_epa"] = [g[5] for g in games]
        pg["rushing_epa"] = NAN
        pg["passing_cpoe"] = NAN
    sn = pd.DataFrame({"player_id": "P1", "game_id": [g[0] for g in games], "season": 2025, "week": [g[1] for g in games],
                       "game_type": "REG", "team": "KC", "position": "WR", "offense_snaps": 40, "offense_pct": 0.7,
                       KNOWN_AT: _ts(g[2] for g in games)})
    xf = pd.DataFrame({"player_id": "P1", "game_id": [g[0] for g in games], "total_fantasy_points_exp": 9.0,
                       "rush_touchdown": 0.0, "rush_touchdown_exp": 0.0, "rec_touchdown": 0.0, "rec_touchdown_exp": 0.0,
                       "pass_touchdown": 0.0, "pass_touchdown_exp": 0.0, KNOWN_AT: _ts(g[2] for g in games)})
    assert n == len(pg)
    return RawStore([RawTable("player_games", pg, "test"), RawTable("snap_counts", sn, "test"), RawTable("xfp", xf, "test")])


def _wk(week: int) -> pd.Timestamp:
    return KO - pd.Timedelta(days=7 * (7 - week))


class ThroughTheHistory(unittest.TestCase):
    PAST = [("g1", 1, _wk(1), NAN, 5.0, 2.0), ("g2", 2, _wk(2), NAN, 4.0, -1.0), ("g3", 3, _wk(3), NAN, 6.0, 3.0)]
    OWN = ("2025_07_LV_KC", 7, KO, NAN, 20.0, 40.0)
    LATE = ("g6", 6, KO - RESULT_LAG, NAN, 20.0, 40.0)                 # exactly at the result cutoff: hidden

    def feats(self, store, t=None):
        t = t or target()
        return _pepa_features(_history(store, "P1", t, None), t)

    def test_nan_epa_on_a_stats_row_is_zero_and_nan_cpoe_stays_nan(self):
        h = _history(_store(self.PAST), "P1", target(), None)
        self.assertEqual(h.v[:, EPA_P].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(h.v[:, EPA_R].tolist(), [0.0, 0.0, 0.0])
        self.assertTrue(all(math.isnan(x) for x in h.v[:, CPOE]))
        f = self.feats(_store(self.PAST))
        self.assertTrue(same(f["pepa_rec_epa_l3"], 4.0 / 3))
        self.assertTrue(same(f["pepa_rec_epa_per_tgt_l3"], 4.0 / 15))
        self.assertEqual(f["pepa_rec_epa_l1"], 3.0)

    def test_own_game_and_post_cutoff_rows_never_enter(self):
        base = self.feats(_store(self.PAST))
        for extra in (self.OWN, self.LATE, ("g8", 8, KO + pd.Timedelta(days=7), NAN, 20.0, 40.0)):
            self.assertEqual(features.fingerprint(self.feats(_store(self.PAST + [extra]))), features.fingerprint(base), extra[0])
        inside = self.feats(_store(self.PAST + [("g6", 6, KO - RESULT_LAG - pd.Timedelta(nanoseconds=1), NAN, 20.0, 40.0)]))
        self.assertEqual(inside["pepa_rec_epa_l1"], 40.0)

    def test_a_table_without_the_new_columns_gives_nan_not_an_error(self):
        f = self.feats(_store(self.PAST, pepa_cols=False))
        self.assertTrue(math.isnan(f["pepa_rec_epa_l1"]))
        self.assertTrue(math.isnan(f["pepa_rec_epa_per_tgt_l3"]))

    def test_the_family_reads_no_table(self):
        import inspect
        src = inspect.getsource(_pepa_features)
        self.assertNotIn("as_of_join", src)
        self.assertNotIn("store", src)


if __name__ == "__main__":
    unittest.main()
