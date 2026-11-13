"""A bye week zeroes E_pts whatever the injury tag says.

    python -m unittest tests.test_flags
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ff import score


def frame(**rows) -> pd.DataFrame:
    return pd.DataFrame({
        "name": list(rows),
        "E_opps": [r.get("E_opps", 10.0) for r in rows.values()],
        "opponent": [r.get("opponent", "KC") for r in rows.values()],
        "injury_status": [r.get("injury_status") for r in rows.values()],
        "E_pts": [12.0] * len(rows),
    })


class ByeBeatsInjuryTag(unittest.TestCase):
    def test_questionable_on_a_bye_has_no_points(self):
        # Aaron Jones, 2026 week 6: MIN on bye, tagged Questionable, kept 9.9 E_pts
        df = score.apply_flags(frame(
            jones={"opponent": "BYE", "injury_status": "Questionable"},
            burrow={"opponent": "BYE"},
            hockenson={"opponent": "BYE", "injury_status": "Out"},
            healthy={}, questionable={"injury_status": "Questionable"},
            rookie={"E_opps": np.nan}))
        pts = dict(zip(df.name, df.E_pts))
        flags = dict(zip(df.name, df.flag))
        for who in ("jones", "burrow", "hockenson", "rookie"):
            self.assertTrue(np.isnan(pts[who]), who)
        self.assertEqual(pts["healthy"], 12.0)
        self.assertEqual(pts["questionable"], 12.0)
        self.assertEqual(flags["burrow"], "BYE")
        self.assertEqual(flags["questionable"], "QUESTIONABLE")

    def test_missing_opponent_is_a_bye(self):
        df = score.apply_flags(frame(nobody={"opponent": None, "injury_status": "Doubtful"}))
        self.assertTrue(np.isnan(df.E_pts.iloc[0]))



class BlankedWeekColumns(unittest.TestCase):
    """The flags that blank E_pts blank this week's volume and projection with it; keep_pts is untouched."""

    def roster(self, bad: dict, **extra):
        df = frame(bad=bad, healthy={})
        df["E_targets"], df["E_carries"] = 5.0, 3.0
        df["proj_pts_league"], df["proj_opps"], df["trail_pts"] = 13.7, 7.0, 9.0
        df["keep_pts"] = [10.0, 11.0]
        for k, v in extra.items():
            df[k] = [v, 0]
        return score.apply_flags(df).set_index("name")

    def check(self, d):
        for c in ("E_pts", "E_opps", "E_targets", "E_carries", "proj_pts_league"):
            self.assertTrue(np.isnan(d.loc["bad", c]), c)
            self.assertFalse(np.isnan(d.loc["healthy", c]), c)
        self.assertEqual(d.loc["healthy", "E_opps"], 10.0)
        self.assertEqual(d.loc["healthy", "proj_pts_league"], 13.7)
        self.assertEqual(list(d.keep_pts), [10.0, 11.0])
        self.assertEqual(d.loc["bad", "proj_opps"], 7.0)      # policy.has_path reads it
        self.assertEqual(d.loc["bad", "trail_pts"], 9.0)      # upside.keep_points fallback

    def test_out(self):
        d = self.roster({"injury_status": "Out"})
        self.assertEqual(d.loc["bad", "flag"], "OUT")
        self.check(d)

    def test_bye(self):
        d = self.roster({"opponent": "BYE"})
        self.assertEqual(d.loc["bad", "flag"], "BYE")
        self.check(d)

    def test_repeat_dnp_2(self):
        d = self.roster({}, repeat_dnp=2)
        self.assertEqual(d.loc["bad", "flag"], "REPEAT_DNP")
        self.check(d)


class TrailingWeightIsOff(unittest.TestCase):
    """trail_cap is 0: E_pts rests on the projection whenever one exists, and falls back to
    trailing volume only when Sleeper has not projected the player."""

    def test_projection_takes_all_the_weight(self):
        games = pd.Series([4.0, 4.0, 0.0])
        has_proj = pd.Series([True, False, True])
        has_trail = pd.Series([True, True, False])
        w = (games / score.DEFAULTS["trail_games"]).clip(0, score.DEFAULTS["trail_cap"])
        w = np.where(~has_proj, 1.0, np.where(~has_trail, 0.0, w))
        self.assertEqual(list(w), [0.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
