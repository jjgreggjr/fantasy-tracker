"""QB1-change flag in the status layer (ff.status.qb_changes / build / warnings / describe).

    python -m unittest tests.test_qb_change

The Michael Mayer case: LV changed starting QB in 2026 week 10 and nothing in the status layer said so.
"""
from __future__ import annotations

import unittest

import pandas as pd

from ff import status


def dc(rows) -> pd.DataFrame:
    """Depth rows: (week, team, position, rank, gsis_id, name[, snapshot_dt])."""
    return pd.DataFrame([{"season": 2026, "week": r[0], "team": r[1], "position": r[2], "rank": r[3],
                          "gsis_id": r[4], "name": r[5], "snapshot_dt": r[6] if len(r) > 6 else "2026-01-01T00:00:00Z"}
                         for r in rows])


DEPTH = dc([(9, "LV", "QB", 1, "cousins", "Kirk Cousins"), (9, "LV", "QB", 2, "mendoza", "Fernando Mendoza"),
            (10, "LV", "QB", 1, "mendoza", "Fernando Mendoza"), (10, "LV", "QB", 2, "cousins", "Kirk Cousins"),
            (9, "KC", "QB", 1, "pm", "Patrick Mahomes"), (10, "KC", "QB", 1, "pm", "Patrick Mahomes")])


class QbChanges(unittest.TestCase):
    def test_changed_flagged_unchanged_not(self):
        out = status.qb_changes(DEPTH, 2026, 10)
        self.assertEqual(list(out.columns), status.QB_CHANGE_COLS)
        self.assertEqual(out.team.tolist(), ["LV"])
        r = out.iloc[0]
        self.assertEqual((r.qb_now, r.qb_prev, r.qb_now_name, r.qb_prev_name),
                         ("mendoza", "cousins", "Fernando Mendoza", "Kirk Cousins"))

    def test_no_prior_week_is_not_flagged(self):
        self.assertTrue(status.qb_changes(DEPTH, 2026, 9).empty)
        only = dc([(10, "LV", "QB", 1, "mendoza", "Fernando Mendoza")])
        self.assertTrue(status.qb_changes(only, 2026, 10).empty)
        self.assertTrue(status.qb_changes(None, 2026, 10).empty)

    def test_compares_to_the_latest_earlier_week_present(self):
        d = dc([(8, "LV", "QB", 1, "cousins", "Kirk Cousins"), (10, "LV", "QB", 1, "mendoza", "Fernando Mendoza")])
        self.assertEqual(status.qb_changes(d, 2026, 10).qb_prev.tolist(), ["cousins"])

    def test_latest_snapshot_in_a_week_wins(self):
        d = dc([(9, "LV", "QB", 1, "cousins", "Kirk Cousins", "2026-11-01T00:00:00Z"),
                (10, "LV", "QB", 1, "cousins", "Kirk Cousins", "2026-11-09T00:00:00Z"),
                (10, "LV", "QB", 1, "mendoza", "Fernando Mendoza", "2026-11-11T00:00:00Z")])
        self.assertEqual(status.qb_changes(d, 2026, 10).qb_now.tolist(), ["mendoza"])
        # the newer snapshot restores the starter: no change
        d.loc[d.snapshot_dt == "2026-11-11T00:00:00Z", "gsis_id"] = "cousins"
        self.assertTrue(status.qb_changes(d, 2026, 10).empty)
        # an old snapshot in the earlier week does not count as its QB1 either
        d = dc([(9, "LV", "QB", 1, "mendoza", "Fernando Mendoza", "2026-11-01T00:00:00Z"),
                (9, "LV", "QB", 1, "cousins", "Kirk Cousins", "2026-11-02T00:00:00Z"),
                (10, "LV", "QB", 1, "cousins", "Kirk Cousins")])
        self.assertTrue(status.qb_changes(d, 2026, 10).empty)


class BuildCarriesIt(unittest.TestCase):
    def setUp(self):
        ids = {"mendoza": "QB", "cousins": "QB", "mayer": "TE", "meyers": "WR", "jeanty": "RB",
               "pm": "QB", "kelce": "TE"}
        teams = {"pm": "KC", "kelce": "KC"}
        n = len(ids)
        self.players = pd.DataFrame({
            "gsis_id": list(ids), "name": list(ids), "team": [teams.get(i, "LV") for i in ids],
            "position": list(ids.values()), "injury_status": None, "injury_body_part": None,
            "nfl_status": "ACT", "practice_participation": None, "news_updated": None,
            "sleeper_depth_order": [1] * n})
        rows = [("LV", "QB", 1, "mendoza", "Fernando Mendoza"), ("LV", "QB", 2, "cousins", "Kirk Cousins"),
                ("LV", "TE", 1, "mayer", "Michael Mayer"), ("LV", "WR", 1, "meyers", "Jakobi Meyers"),
                ("LV", "RB", 1, "jeanty", "Ashton Jeanty"),
                ("KC", "QB", 1, "pm", "Patrick Mahomes"), ("KC", "TE", 1, "kelce", "Travis Kelce")]
        self.depth = pd.DataFrame([{"gsis_id": g, "name": n, "team": t, "position": p, "rank": k, "prev_rank": k}
                                   for t, p, k, g, n in rows])
        self.qb = status.qb_changes(DEPTH, 2026, 10)

    def _st(self, qb):
        return status.build(self.players, self.depth, None, 10, None, None, None, None, qb).set_index("gsis_id")

    def test_pass_catchers_flagged_qb_and_other_teams_not(self):
        st = self._st(self.qb)
        for g in ("mayer", "meyers", "jeanty"):
            self.assertEqual(st.loc[g, "qb_change"], 1, g)
            self.assertEqual(st.loc[g, "qb_change_note"], "new QB1: Fernando Mendoza (was Kirk Cousins)")
        for g in ("mendoza", "cousins", "pm", "kelce"):
            self.assertEqual(st.loc[g, "qb_change"], 0, g)
            self.assertTrue(pd.isna(st.loc[g, "qb_change_note"]), g)

    def test_informational_only(self):
        a, b = self._st(self.qb), self._st(None)
        self.assertEqual(a.loc["mayer", "play_prob"], b.loc["mayer", "play_prob"])
        self.assertEqual(a.loc["mayer", "status_flag"], b.loc["mayer", "status_flag"])
        self.assertEqual(b.qb_change.sum(), 0)           # no qb frame: columns exist, nobody flagged

    def test_warning_and_describe(self):
        st = self._st(self.qb).reset_index()
        self.assertTrue(any("Michael Mayer" in w and "new QB1" in w for w in status.warnings(st)))
        self.assertIn("QB CHANGE", status.describe(st, "Mayer"))
        self.assertNotIn("QB CHANGE", status.describe(st, "Kelce"))


if __name__ == "__main__":
    unittest.main()
