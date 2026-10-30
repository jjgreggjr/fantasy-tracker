"""The call: three lineups, the model's K/IDP fill, and the three exceptions to "E_pts decides". No network."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ff import ask, calls


def roster(rows):
    """rows: (name, pos, E_pts, E_pts_model, E_opps, extras dict)"""
    recs = []
    for n, pos, e, m, o, *x in rows:
        d = {"name": n, "position": pos, "E_pts": e, "E_pts_model": m, "E_opps": o, "team": "T", "gsis_id": n,
             "sleeper_id": n, "p10": np.nan, "p90": np.nan}
        d.update(x[0] if x else {})
        recs.append(d)
    return pd.DataFrame(recs)


NOSTATUS = pd.DataFrame()


def run(rows, slots=("RB", "FLEX"), idp=False):
    return calls.three_lineups(roster(rows), list(slots), idp, status=NOSTATUS)


def picks(df):
    return list(df["name"])


class ThreeLineups(unittest.TestCase):
    def test_contract_columns_and_slot_order(self):
        out = run([("A", "RB", 10, 10, 10), ("B", "WR", 9, 9, 9), ("C", "RB", 8, 8, 8)], slots=("FLEX", "RB"))
        self.assertEqual(set(out), {"e_pts", "model", "call"})
        for k, df in out.items():
            self.assertEqual(list(df["slot"]), ["FLEX", "RB"])
            for c in ("slot", "name", "position", "team", "E_pts", "E_pts_model", "p10", "p90", "gsis_id", "sleeper_id"):
                self.assertIn(c, df.columns)
        for c in ("e_pick", "m_pick", "reason"):
            self.assertIn(c, out["call"].columns)

    def test_no_model_columns_means_all_three_equal(self):
        r = roster([("A", "RB", 10, np.nan, 10), ("B", "RB", 9, np.nan, 9)]).drop(columns=["E_pts_model"])
        out = calls.three_lineups(r, ["RB", "RB"], False, status=NOSTATUS)
        self.assertEqual(picks(out["model"]), picks(out["e_pts"]))
        self.assertEqual(picks(out["call"]), picks(out["e_pts"]))
        self.assertEqual(set(out["call"]["reason"]), {""})

    def test_same_players_no_reason(self):
        out = run([("A", "RB", 10, 11, 10), ("B", "RB", 9, 10, 9)], slots=("RB", "RB"))
        self.assertEqual(set(out["call"]["reason"]), {""})
        self.assertEqual(set(out["call"]["e_pick"]), {""})


class Rules(unittest.TestCase):
    def test_default_keeps_the_e_pts_pick(self):
        # E_pts gap 4.1, model gap 0.9: E_pts wins
        out = run([("Kincaid", "TE", 12.8, 11.0, 5.2), ("Pitts", "TE", 8.7, 11.9, 4.2)], slots=("TE",))
        self.assertEqual(picks(out["model"]), ["Pitts"])
        self.assertEqual(picks(out["call"]), ["Kincaid"])
        self.assertIn("E_pts +4.1 Kincaid", out["call"].reason[0])
        self.assertEqual((out["call"].e_pick[0], out["call"].m_pick[0]), ("Kincaid", "Pitts"))

    def test_near_tie_goes_to_the_model(self):
        out = run([("Boston", "WR", 10.2, 10.0, 5.4), ("Judkins", "RB", 9.9, 14.0, 18.3),
                   ("Hall", "RB", 13.0, 13.0, 15)], slots=("RB", "FLEX"))
        self.assertIn("Judkins", picks(out["call"]))
        self.assertNotIn("Boston", picks(out["call"]))
        self.assertIn("E_pts tie", " ".join(out["call"].reason))

    def test_near_tie_and_model_tie_goes_to_volume(self):
        out = run([("A", "WR", 10.0, 9.0, 5.0), ("B", "WR", 9.6, 9.3, 9.0)], slots=("WR",))
        self.assertEqual(picks(out["call"]), ["B"])

    def test_model_override_needs_volume(self):
        rows = [("A", "WR", 12.5, 9.0, 6.1), ("B", "WR", 9.9, 13.9, 18.3)]
        self.assertEqual(picks(run(rows, slots=("WR",))["call"]), ["B"])      # gap 2.6, model 4.9, more opps
        rows[1] = ("B", "WR", 9.9, 13.9, 3.0)                                  # fewer opps: no override
        self.assertEqual(picks(run(rows, slots=("WR",))["call"]), ["A"])

    def test_model_gap_must_be_1_5x(self):
        out = run([("A", "WR", 12.0, 9.0, 6.0), ("B", "WR", 9.0, 12.0, 18.0)], slots=("WR",))   # 3.0 vs 3.0
        self.assertEqual(picks(out["call"]), ["A"])

    def test_status_beats_model_and_tie(self):
        hurt = {"injury_status": "Questionable", "practice": "Did Not Participate In Practice"}
        ok = {"practice": "Limited Participation in Practice"}
        out = run([("Jones", "RB", 8.9, 10.8, 12.0, hurt), ("Kincaid", "TE", 9.0, 7.4, 4.0, ok)], slots=("FLEX",))
        self.assertEqual(picks(out["call"]), ["Kincaid"])
        self.assertIn("Jones did not practice; Kincaid did (limited)", out["call"].reason[0])

    def test_status_ignored_past_the_gap(self):
        hurt = {"injury_status": "Questionable", "practice": "Did Not Participate In Practice"}
        out = run([("Coker", "WR", 16.8, 9.8, 8.0), ("Jones", "RB", 10.0, 12.0, 12.1, hurt)], slots=("FLEX",))
        self.assertEqual(picks(out["call"]), ["Coker"])

    def test_repeat_dnp_and_low_play_prob_count_as_status(self):
        for extra in ({"flag": "REPEAT_DNP"}, {"play_prob": 0.4}):
            out = run([("Jones", "RB", 9.0, 12.0, 12.0, extra), ("Kincaid", "TE", 9.5, 7.0, 4.0)], slots=("FLEX",))
            self.assertEqual(picks(out["call"]), ["Kincaid"])

    def test_status_read_from_status_table(self):
        st = pd.DataFrame({"gsis_id": ["Jones", "Kincaid"], "practice": ["Did Not Participate In Practice", None],
                           "play_prob": [0.9, 0.9]})
        r = roster([("Jones", "RB", 9.0, 12.0, 12.0, {"injury_status": "Questionable"}), ("Kincaid", "TE", 9.5, 7.0, 4.0)])
        out = calls.three_lineups(r, ["FLEX"], False, status=st)
        self.assertEqual(picks(out["call"]), ["Kincaid"])

    def test_swap_that_cannot_be_slotted_is_dropped(self):
        # the model's QB cannot play a WR slot: the E_pts WR stays
        r = roster([("W", "WR", 10.0, 5.0, 5.0), ("Q", "QB", 9.9, 20.0, 30.0)])
        out = calls.three_lineups(r, ["WR"], False, status=NOSTATUS)
        self.assertEqual(picks(out["model"]), ["W"])


class KIdpFill(unittest.TestCase):
    def setUp(self):
        self.out = run([("Q", "QB", 20, 18, 30), ("Kick", "K", 8, np.nan, 0), ("Lb", "LB", 7, np.nan, 0),
                        ("Wr", "WR", 5, np.nan, 5)], slots=("QB", "K", "LB", "WR"), idp=True)

    def test_k_and_idp_borrow_e_pts_in_the_model_lineup(self):
        m = self.out["model"].set_index("slot")
        self.assertEqual(m.loc["K", "name"], "Kick")
        self.assertEqual(m.loc["LB", "E_pts_model"], 7)
        self.assertEqual(sorted(self.out["model"].attrs["borrowed"]), ["Kick", "Lb"])

    def test_skill_player_without_a_model_number_stays_out(self):
        self.assertEqual(self.out["model"].set_index("slot").loc["WR", "name"], "")
        self.assertEqual(self.out["e_pts"].set_index("slot").loc["WR", "name"], "Wr")

    def test_no_diff_from_the_borrowed_slots(self):
        c = self.out["call"].set_index("slot")
        self.assertEqual((c.loc["K", "e_pick"], c.loc["LB", "m_pick"]), ("", ""))


class Render(unittest.TestCase):
    def test_table_marks_in_and_prints_net(self):
        r = roster([("A", "WR", 12.5, 9.0, 6.1), ("B", "WR", 9.9, 13.9, 18.3)])
        out = calls.three_lineups(r, ["WR"], False, status=NOSTATUS)
        start = out["e_pts"].copy()
        start.attrs.update(three=out, all_slots=["WR", "DEF"])
        txt = ask.call_table(start, {"A"}, ["KC"])
        self.assertIn("B IN 9.9 / 13.9", txt)
        self.assertIn("not scored (KC)", txt)
        self.assertIn("Net vs current: IN B; OUT A; call differs from E_pts at WR (B over A)", txt)
        self.assertIn("(call lineup)", txt)


if __name__ == "__main__":
    unittest.main()
