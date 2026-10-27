"""E_pts touchdown regression (ff.score.td_regression): the player's season TD rate per opportunity is shrunk toward
his position's rate and the difference, times expected opportunities and TD points, is added to E_pts.

    python -m unittest tests.test_td_regress

Motivation: model/reports/miss_review_2026wk07.md, residual attribution (eta² .037 on TD rate per opp: E_pts
over-predicted high-TD-rate players by 1.6 and under-predicted low-TD-rate players by 1.8).
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ff import score

SCORING = {"rush_td": 6, "rec_td": 6, "pass_td": 4}
CFG = {**score.DEFAULTS, "td_prior_opps": {"QB": 150, "RB": 60, "WR": 60, "TE": 60}, "td_adj_clip": 3.0}


def weeks(rows) -> pd.DataFrame:
    """player_weeks rows: (week, gsis_id, position, opps, rec_tds, rush_tds, pass_att, pass_tds)."""
    return pd.DataFrame([{"season": 2026, "week": w, "gsis_id": g, "position": pos, "opps": o,
                          "rec_tds": rtd, "rush_tds": rushtd, "pass_att": att, "pass_tds": ptd, "snaps": 50}
                         for w, g, pos, o, rtd, rushtd, att, ptd in rows])


def frame(rows) -> pd.DataFrame:
    """scored rows: (gsis_id, position, E_opps, proj_pass_att)."""
    return pd.DataFrame([{"gsis_id": g, "position": pos, "E_opps": e, "proj_pass_att": att}
                         for g, pos, e, att in rows])


class ShrinksTowardThePrior(unittest.TestCase):
    def setUp(self):
        # league-wide WR rate: 10 TDs on 200 opps = 5%. hot: 4 TDs on 40 opps (10%); cold: 0 on 40 (0%);
        # filler makes up the rest of the position pool.
        self.pw = weeks([(1, "hot", "WR", 20, 2, 0, 0, 0), (2, "hot", "WR", 20, 2, 0, 0, 0),
                         (1, "cold", "WR", 20, 0, 0, 0, 0), (2, "cold", "WR", 20, 0, 0, 0, 0),
                         (1, "fill", "WR", 60, 3, 0, 0, 0), (2, "fill", "WR", 60, 3, 0, 0, 0)])

    def test_hot_player_is_pulled_down_and_cold_player_up(self):
        out = score.td_regression(frame([("hot", "WR", 10, np.nan), ("cold", "WR", 10, np.nan)]),
                                  self.pw, 2026, 3, SCORING, CFG)
        hot, cold = out.iloc[0], out.iloc[1]
        self.assertAlmostEqual(hot.td_rate_raw, 0.10, places=4)
        self.assertAlmostEqual(hot.td_rate_shrunk, (4 + 60 * 0.05) / (40 + 60), places=4)   # 0.07
        self.assertAlmostEqual(hot.td_adj, round((0.07 - 0.10) * 10 * 6, 2), places=2)      # -1.8
        self.assertAlmostEqual(cold.td_rate_shrunk, (0 + 60 * 0.05) / 100, places=4)         # 0.03
        self.assertAlmostEqual(cold.td_adj, round(0.03 * 10 * 6, 2), places=2)              # +1.8

    def test_more_opportunities_shrink_less(self):
        # the same 10% rate on 400 opps barely moves
        big = weeks([(w, "vet", "WR", 200, 20, 0, 0, 0) for w in (1, 2)])
        pw = pd.concat([self.pw, big], ignore_index=True)
        out = score.td_regression(frame([("vet", "WR", 10, np.nan), ("hot", "WR", 10, np.nan)]),
                                  pw, 2026, 3, SCORING, CFG)
        self.assertLess(abs(out.td_adj.iloc[0]), abs(out.td_adj.iloc[1]))

    def test_clip_bounds(self):
        # 10 TDs on 20 opps (50%) against a 5% pool: the raw correction would be far beyond 3 points
        pw = pd.concat([self.pw, weeks([(1, "fluke", "WR", 20, 10, 0, 0, 0)])], ignore_index=True)
        out = score.td_regression(frame([("fluke", "WR", 15, np.nan)]), pw, 2026, 2, SCORING, CFG)
        self.assertEqual(out.td_adj.iloc[0], -3.0)
        out = score.td_regression(frame([("fluke", "WR", 15, np.nan)]), pw, 2026, 2, SCORING,
                                  {**CFG, "td_adj_clip": 1.0})
        self.assertEqual(out.td_adj.iloc[0], -1.0)

    def test_qb_uses_pass_attempts_and_pass_td_points(self):
        pw = weeks([(1, "qb_hot", "QB", 3, 0, 0, 30, 4), (2, "qb_hot", "QB", 3, 0, 0, 30, 4),
                    (1, "qb_avg", "QB", 3, 0, 0, 60, 2), (2, "qb_avg", "QB", 3, 0, 0, 60, 2)])
        # pool: 12 TDs on 180 attempts = 6.67%; qb_hot 8/60 = 13.3%
        out = score.td_regression(frame([("qb_hot", "QB", 3, 35)]), pw, 2026, 3, SCORING, CFG)
        raw, shrunk = 8 / 60, (8 + 150 * (12 / 180)) / (60 + 150)
        self.assertAlmostEqual(out.td_rate_raw.iloc[0], round(raw, 4), places=4)
        self.assertAlmostEqual(out.td_adj.iloc[0], round(max(-3.0, (shrunk - raw) * 35 * 4), 2), places=2)
        # no projected attempts: his season mean (30) stands in
        out = score.td_regression(frame([("qb_hot", "QB", 3, np.nan)]), pw, 2026, 3, SCORING, CFG)
        self.assertAlmostEqual(out.td_adj.iloc[0], round(max(-3.0, (shrunk - raw) * 30 * 4), 2), places=2)

    def test_deterministic(self):
        a = score.td_regression(frame([("hot", "WR", 10, np.nan)]), self.pw, 2026, 3, SCORING, CFG)
        b = score.td_regression(frame([("hot", "WR", 10, np.nan)]), self.pw.sample(frac=1, random_state=7),
                                2026, 3, SCORING, CFG)
        pd.testing.assert_frame_equal(a, b)


class NanSafe(unittest.TestCase):
    def setUp(self):
        self.pw = weeks([(1, "hot", "WR", 20, 2, 0, 0, 0), (1, "fill", "WR", 80, 3, 0, 0, 0)])

    def test_missing_data_moves_nothing(self):
        out = score.td_regression(frame([("rookie", "WR", 10, np.nan),     # no season rows
                                         ("hot", "WR", np.nan, np.nan),    # no expected opps
                                         ("qb", "QB", 3, np.nan),          # QB who has never thrown
                                         ("hot", "WR", 10, np.nan)]),      # the one real case
                                  self.pw, 2026, 2, SCORING, CFG)
        self.assertEqual(list(out.td_adj.iloc[:3]), [0.0, 0.0, 0.0])
        self.assertTrue(out.td_rate_raw.iloc[:3].isna().all())
        self.assertNotEqual(out.td_adj.iloc[3], 0.0)

    def test_zero_opportunities_is_not_a_rate(self):
        pw = pd.concat([self.pw, weeks([(1, "bench", "WR", 0, 0, 0, 0, 0)])], ignore_index=True)
        out = score.td_regression(frame([("bench", "WR", 5, np.nan)]), pw, 2026, 2, SCORING, CFG)
        self.assertEqual(out.td_adj.iloc[0], 0.0)
        self.assertTrue(np.isnan(out.td_rate_raw.iloc[0]))

    def test_no_history_or_switch_off(self):
        f = frame([("hot", "WR", 10, np.nan)])
        for pw in (None, pd.DataFrame(), self.pw[self.pw.season == 2025]):
            self.assertEqual(score.td_regression(f, pw, 2026, 2, SCORING, CFG).td_adj.iloc[0], 0.0)
        self.assertEqual(score.td_regression(f, self.pw, 2026, 2, SCORING,
                                             {**CFG, "td_regress": False}).td_adj.iloc[0], 0.0)

    def test_only_weeks_before_the_target_count(self):
        later = weeks([(5, "hot", "WR", 20, 5, 0, 0, 0)])
        pw = pd.concat([self.pw, later], ignore_index=True)
        a = score.td_regression(frame([("hot", "WR", 10, np.nan)]), self.pw, 2026, 2, SCORING, CFG)
        b = score.td_regression(frame([("hot", "WR", 10, np.nan)]), pw, 2026, 2, SCORING, CFG)
        pd.testing.assert_frame_equal(a, b)


if __name__ == "__main__":
    unittest.main()
