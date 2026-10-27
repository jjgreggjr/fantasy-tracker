"""The opt-in `shape` family (model/features.py `_shape_features`): hand-computed values on a synthetic history, the cross-season
window, NaN safety, determinism, the registry contract (the frozen serving inputs are untouched), and the builder wiring.

    python -m unittest model.tests.test_shape_family

No network, no real matrix: everything here is built by hand.
"""
from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from model import features, train as T, trial as TR
from model.features import (ATT, CSH, NAN, N_VALUES, OPPS, PASSTD, PTS, RECTD, RUSHTD, TSH, Hist, SHAPE_TD_K, SHAPE_TD_PRIOR,
                            _shape_features)
from model.point_in_time import TargetRow

COLS = features.FAMILIES["shape"]
KO = pd.Timestamp("2025-10-19 17:00", tz="UTC")


def row(gid: str, season: int, week: int, pts: float, opps: float, *, tsh: float = NAN, csh: float = NAN, rush_td: float = 0.0,
        rec_td: float = 0.0, pass_td: float = 0.0, att: float = 0.0, reg: bool = True) -> tuple:
    v = [0.0] * N_VALUES
    v[PTS], v[OPPS], v[TSH], v[CSH], v[RUSHTD], v[RECTD], v[PASSTD], v[ATT] = pts, opps, tsh, csh, rush_td, rec_td, pass_td, att
    return (gid, season, week, reg, True, "KC", *v)


def target(season: int = 2025, week: int = 7, position: str = "WR") -> TargetRow:
    return TargetRow("P1", season, week, "KC", "LV", position, f"{season}_{week:02d}_LV_KC", KO, True, ("usage",))


def same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-9


def pstd(xs) -> float:
    return float(np.std(np.array(xs, dtype="float64")))


class HandComputed(unittest.TestCase):
    # this season, weeks 1-6: (pts, opps, tsh, csh, rec_td)
    CUR = [row("g1", 2025, 1, 10.0, 6.0, tsh=0.20, csh=0.00, rec_td=0.0), row("g2", 2025, 2, 30.0, 8.0, tsh=0.30, csh=0.05, rec_td=2.0),
           row("g3", 2025, 3, 8.0, 7.0, tsh=0.22, csh=0.00, rec_td=0.0), row("g4", 2025, 4, 12.0, 9.0, tsh=0.25, csh=0.02, rec_td=0.0),
           row("g5", 2025, 5, 26.0, 10.0, tsh=0.28, csh=0.00, rec_td=1.0), row("g6", 2025, 6, 9.0, 12.0, tsh=0.18, csh=0.00, rec_td=0.0)]
    # last season (REG), plus a POST game that must never count
    PREV = [row("p1", 2024, 16, 14.0, 7.0, tsh=0.21, csh=0.01, rec_td=1.0), row("p2", 2024, 17, 11.0, 5.0, tsh=0.19, csh=0.00),
            row("p3", 2024, 18, 13.0, 6.0, tsh=0.24, csh=0.00), row("p4", 2024, 19, 50.0, 20.0, tsh=0.9, csh=0.5, rec_td=4.0, reg=False)]

    def test_every_column_against_the_definitions(self):
        f = _shape_features(Hist(self.PREV + self.CUR), target())
        self.assertEqual(set(f), set(COLS))
        self.assertTrue(same(f["shape_tgt_share_std_l6"], pstd([0.20, 0.30, 0.22, 0.25, 0.28, 0.18])))      # the last 6 are this season's
        self.assertTrue(same(f["shape_car_share_std_l6"], pstd([0.00, 0.05, 0.00, 0.02, 0.00, 0.00])))
        last8 = [11.0, 13.0, 10.0, 30.0, 8.0, 12.0, 26.0, 9.0]                                                # p2, p3, g1..g6 (never p4)
        mean8 = sum(last8) / 8
        self.assertTrue(same(f["shape_pts_cv_l8"], pstd(last8) / mean8))
        self.assertTrue(same(f["shape_boom_rate_l8"], sum(x >= 1.5 * mean8 for x in last8) / 8))             # 30 and 26 of 119/8 = 14.875
        self.assertTrue(same(f["shape_td_pts_share_l8"], 6.0 * 3.0 / sum(last8)))                           # g2 two, g5 one; p1 is outside
        self.assertTrue(same(f["shape_td_per_opp_std"], (3.0 + 60.0 * SHAPE_TD_PRIOR["WR"]) / (52.0 + 60.0)))  # season to date only
        # opps over the last 4: 7, 9, 10, 12 -> slope of the least-squares line
        self.assertTrue(same(f["shape_opps_trend_l4"], float(np.polyfit([0, 1, 2, 3], [7.0, 9.0, 10.0, 12.0], 1)[0])))

    def test_a_quarterback_uses_pass_touchdowns_per_attempt_and_the_150_prior(self):
        cur = [row(f"q{i}", 2025, i, 18.0, 3.0, pass_td=2.0, att=35.0) for i in range(1, 5)]
        f = _shape_features(Hist(cur), target(week=5, position="QB"))
        self.assertTrue(same(f["shape_td_per_opp_std"], (8.0 + 150.0 * SHAPE_TD_PRIOR["QB"]) / (140.0 + 150.0)))
        self.assertEqual(SHAPE_TD_K["QB"], 150.0)
        self.assertTrue(same(f["shape_td_pts_share_l8"], 4.0 * 8.0 / 72.0))                                 # 4 points a passing score
        self.assertEqual(f["shape_opps_trend_l4"], 0.0)                                                      # 38 every game
        self.assertEqual(f["shape_boom_rate_l8"], 0.0)
        self.assertEqual(f["shape_pts_cv_l8"], 0.0)

    def test_no_history_is_all_nan_and_one_game_leaves_the_spreads_nan(self):
        f = _shape_features(Hist([]), target(week=1))
        self.assertTrue(all(math.isnan(f[c]) for c in COLS))
        f = _shape_features(Hist([row("a", 2025, 1, 12.0, 5.0, tsh=0.2, rec_td=1.0)]), target(week=2))
        for c in ("shape_tgt_share_std_l6", "shape_car_share_std_l6", "shape_opps_trend_l4", "shape_pts_cv_l8"):
            self.assertTrue(math.isnan(f[c]), c)                                                             # one game is not a spread
        self.assertEqual(f["shape_boom_rate_l8"], 0.0)                                                       # 12 < 1.5 * 12
        self.assertTrue(same(f["shape_td_pts_share_l8"], 6.0 / 12.0))
        self.assertTrue(same(f["shape_td_per_opp_std"], (1.0 + 60.0 * SHAPE_TD_PRIOR["WR"]) / 65.0))

    def test_nan_shares_and_non_positive_points_are_safe(self):
        cur = [row("a", 2025, 1, 0.0, 0.0), row("b", 2025, 2, -1.0, 0.0), row("c", 2025, 3, 1.0, 0.0, tsh=0.1)]
        f = _shape_features(Hist(cur), target(week=4))
        self.assertTrue(math.isnan(f["shape_pts_cv_l8"]), "mean of 0, -1, 1 is 0: no ratio")
        self.assertTrue(math.isnan(f["shape_boom_rate_l8"]))
        self.assertTrue(math.isnan(f["shape_td_pts_share_l8"]))
        self.assertTrue(math.isnan(f["shape_td_per_opp_std"]), "no opportunity this season: the prior alone is not a feature")
        self.assertTrue(math.isnan(f["shape_tgt_share_std_l6"]), "one known share is not a spread")
        self.assertEqual(f["shape_opps_trend_l4"], 0.0)
        for c in COLS:
            self.assertIsInstance(f[c], float)

    def test_week_one_reads_last_seasons_regular_season_only_and_season_to_date_is_nan(self):
        f = _shape_features(Hist(self.PREV), target(week=1))
        self.assertTrue(same(f["shape_tgt_share_std_l6"], pstd([0.21, 0.19, 0.24])))
        self.assertTrue(same(f["shape_td_pts_share_l8"], 6.0 / 38.0))
        self.assertTrue(math.isnan(f["shape_td_per_opp_std"]))
        older = [row("z", 2023, 18, 99.0, 50.0, tsh=0.9, rec_td=5.0)]
        g = _shape_features(Hist(older + self.PREV), target(week=1))
        self.assertEqual(features.fingerprint(g), features.fingerprint(f), "a 2023 game is not in the window")

    def test_deterministic(self):
        a = _shape_features(Hist(self.PREV + self.CUR), target())
        b = _shape_features(Hist(self.PREV + self.CUR), target())
        self.assertEqual(features.fingerprint(a), features.fingerprint(b))


class Registry(unittest.TestCase):
    def test_shape_is_registered_opt_in_and_the_serving_inputs_are_unchanged(self):
        self.assertIn("shape", features.FAMILIES)
        self.assertIn("shape", T.OPT_IN_FAMILIES)
        self.assertEqual(len(COLS), 7)
        self.assertEqual(len(set(COLS)), 7)
        for c in COLS:
            self.assertEqual(features.family_of(c), "shape")
            self.assertTrue(c.startswith("shape_"))
        df = pd.DataFrame({c: [0.0, 1.0] for c in features.feature_columns()} | {"position": ["QB", "RB"],
                          "inj_report_status": [None, None], "inj_practice_status": [None, None], "wx_roof_obs": [None, None]})
        self.assertEqual(set(T.feature_columns(df, T.PRIMARY)) & set(COLS), set(), "the Phase 2 primary keeps its inputs")
        cfg = TR.load_config()
        serving = T.feature_columns(df, TR.serving_spec(cfg), dead=("college_breakout_age",))
        X, _ = T.encode(df, serving)
        self.assertEqual(set(X.columns), set(cfg["inputs"]), "the frozen serving inputs must not move")
        self.assertEqual(set(X.columns) & set(COLS), set())
        added = T.feature_columns(df, T.Spec("x", add_families=("shape",)))
        self.assertEqual(set(added) - set(T.feature_columns(df, T.PRIMARY)), set(COLS))

    def test_trial_accepts_the_candidate_and_lists_it_as_opt_in(self):
        base = TR.serving_spec(TR.load_config())
        v = TR.compose(["+family:shape"], base)
        self.assertIn("shape", v.spec.add_families)
        self.assertTrue(v.changes_inputs(base))
        text = TR.list_families(TR.load_config())
        self.assertRegex(text, r"shape  \[0/7 not in serving, opt-in\]")
        self.assertRegex(text, r"shape_pts_cv_l8\s+out: opt-in family \(\+family:shape\)")

    def test_build_features_wires_the_family_from_hist_only(self):
        import ast, inspect
        src = inspect.getsource(_shape_features)
        self.assertNotIn("as_of_join", src)
        self.assertNotIn("store", ast.dump(ast.parse(src)))
        full = inspect.getsource(features.build_features)
        self.assertLess(full.index("_streak_features(h, t)"), full.index("_shape_features(h, t)"))
        self.assertLess(full.index("_shape_features(h, t)"), full.index("_role_features(ctx, h, t)"))


if __name__ == "__main__":
    unittest.main()
