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


if __name__ == "__main__":
    unittest.main()
