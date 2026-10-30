"""trail_cap stays at the adopted default, and trailing points do not move the projection.

    python -m unittest tests.test_trail_blend

No network and none of the repo's data files. Week 8 2026 rosters shipped with a stale `trail_cap` 0.6 in
config.json overriding the adopted 0.0, so raw trailing points (a 3-game average full of TD noise) got 50-60%
of the weight even though expected opportunities barely moved. Keon Coleman came out 12.7 instead of ~8.1,
Rome Odunze 7.7 instead of ~9.1.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ff import score

WEEK = 8
ROOT = Path(__file__).resolve().parent.parent
PINNED = ("trail_cap", "trail_games", "matchup_swing")


class ConfigPinsAdoptedTrailCap(unittest.TestCase):
    """A stale 0.6 override shipped week 8 2026 rosters with trailing points at 50-60% weight."""

    def check(self, name):
        cfg = json.loads((ROOT / name).read_text())
        for key in PINNED:
            with self.subTest(file=name, key=key):
                self.assertEqual(cfg["score"][key], score.DEFAULTS[key])

    def test_config_json(self):
        self.check("config.json")

    def test_config_example_json(self):
        self.check("config.example.json")


def build(cfg):
    """Two WRs shaped like Coleman (BUF vs BAL) and Odunze (CHI vs SEA); returns score.compute's rows by gsis_id."""
    ids = ["coleman", "odunze"]
    roster = pd.DataFrame({"gsis_id": ids, "name": ["Keon Coleman", "Rome Odunze"], "position": ["WR", "WR"],
                           "team": ["BUF", "CHI"]})
    players = pd.DataFrame({"gsis_id": ids, "age": [22, 23], "years_exp": [2, 2], "injury_status": [None, None],
                            "nfl_status": ["ACT", "ACT"], "draft_number": [33, 9], "rookie_year": [2024, 2024]})
    base = pd.DataFrame({"gsis_id": ids, "avg_opps": [4.0, 6.0], "avg_targets": [4.0, 6.0],
                         "avg_carries": [0.0, 0.0], "games": [2, 3]})
    proj = pd.DataFrame({
        "week": [WEEK, WEEK], "gsis_id": ids,
        "proj_pass_att": [np.nan, np.nan], "proj_pass_yds": [0.0, 0.0], "proj_pass_tds": [0.0, 0.0],
        "proj_pass_int": [0.0, 0.0],
        "proj_carries": [0.0, 0.0], "proj_rush_yds": [0.0, 0.0], "proj_rush_tds": [0.0, 0.0],
        "proj_targets": [3.86, 6.43], "proj_rec": [2.68, 3.56], "proj_rec_yds": [39.47, 53.5],
        "proj_rec_tds": [0.18, 0.21], "proj_opps": [3.9, 6.44],
        "proj_carry_share": [0.0, 0.0], "proj_target_share": [0.15, 0.22]})
    sched = pd.DataFrame({"week": [WEEK, WEEK], "team": ["BUF", "CHI"], "opponent": ["BAL", "SEA"],
                          "total_line": [46.0, 44.0], "home": [1, 0]})
    dvp = pd.DataFrame({"defense": ["BAL", "SEA"], "position": ["WR", "WR"], "dvp_rank": [10, 31]})
    depth = pd.DataFrame({"gsis_id": ids, "rank": [3, 1], "prev_rank": [3, 1]})
    points = pd.DataFrame({"season": 2026, "week": [6, 7, 5, 6, 7],
                           "gsis_id": ["coleman", "coleman", "odunze", "odunze", "odunze"],
                           "pts": [20.0, 12.8, 7.6, 7.6, 7.6]})
    meta = {"name": "t", "scoring": {"rec": 1, "rec_yd": 0.1, "rec_td": 6}}
    out = score.compute(roster, players, base, proj, dvp, sched, depth, points, meta, WEEK, 2026,
                        cfg=cfg, pw=None)
    return out.set_index("gsis_id")


class ProjectionIsNotMovedByTrailingPoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default = build(None)
        cls.stale = build({"trail_cap": 0.6})

    def test_trailing_points_are_what_the_test_says(self):
        self.assertAlmostEqual(self.default.loc["coleman", "trail_pts"], 16.4)
        self.assertAlmostEqual(self.default.loc["odunze", "trail_pts"], 7.6)
        self.assertEqual(self.default.loc["coleman", "mult"], 1.052)
        self.assertEqual(self.default.loc["odunze", "mult"], 0.895)

    def test_default_cap_scores_the_projection_alone(self):
        for g in ("coleman", "odunze"):
            r = self.default.loc[g]
            with self.subTest(player=g):
                self.assertEqual(r.w_proj, 1.0)
                self.assertEqual(r.pts_mix, "projection only")
                self.assertEqual(r.E_pts, np.round(r.proj_pts_league * r.mult, 1))

    def test_a_stale_0_6_cap_is_the_documented_bad_outcome(self):
        for g, w in (("coleman", 0.5), ("odunze", 0.4)):
            r = self.stale.loc[g]
            with self.subTest(player=g):
                self.assertEqual(r.w_proj, w)
                self.assertEqual(r.E_pts, np.round((w * r.proj_pts_league + (1 - w) * r.trail_pts) * r.mult, 1))
        c, o = self.stale.loc["coleman"], self.stale.loc["odunze"]
        self.assertGreater(c.E_pts, c.proj_pts_league * 1.30)
        self.assertLess(o.E_pts, o.proj_pts_league * 0.85)

    def test_the_gap_is_the_points_blend_not_opportunity(self):
        for g in ("coleman", "odunze"):
            with self.subTest(player=g):
                a, b = self.default.loc[g], self.stale.loc[g]
                self.assertLess(abs(a.E_opps - b.E_opps), 0.3)
                self.assertLess(abs(a.E_opps - a.proj_opps), 0.3)
                self.assertLess(abs(b.E_opps - a.proj_opps), 0.3)


if __name__ == "__main__":
    unittest.main()
