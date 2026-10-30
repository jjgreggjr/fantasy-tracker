"""Exact lineup optimizer and the side-by-side render. No network, no scipy."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ff import ask


def pool(rows):
    cols = ["name", "position", "E_pts"] + (["E_pts_model"] if len(rows[0]) == 4 else [])
    df = pd.DataFrame(rows, columns=cols)
    df["team"], df["opponent"], df["conf"], df["why"] = "T", "O", "High", ""
    df["gsis_id"] = df["name"]
    return df


def total(picks, col="E_pts"):
    return sum(x[col] for x in picks)


class Optimizer(unittest.TestCase):
    def test_exact_beats_left_to_right_greedy(self):
        # FLEX is listed before RB: filling slots in order gives FLEX the RB and leaves RB empty (10).
        p = pool([("A", "RB", 10.0), ("B", "WR", 9.0)])
        got = ask.optimize_lineup(p, ["FLEX", "RB"], "E_pts", False)
        self.assertEqual(total(got), 19.0)
        self.assertEqual([(x["slot"], x["name"]) for x in got], [("FLEX", "B"), ("RB", "A")])

    def test_flex_takes_the_best_leftover(self):
        p = pool([("RB1", "RB", 12.0), ("RB2", "RB", 10.0), ("WR1", "WR", 11.0),
                  ("WR2", "WR", 9.0), ("RB3", "RB", 9.5)])
        got = ask.optimize_lineup(p, ["RB", "WR", "FLEX"], "E_pts", False)
        self.assertEqual(total(got), 12.0 + 11.0 + 10.0)
        self.assertEqual(len({x["name"] for x in got}), 3)

    def test_superflex_does_not_eat_the_only_qb(self):
        p = pool([("QB1", "QB", 20.0), ("RB1", "RB", 15.0), ("WR1", "WR", 14.0)])
        got = ask.optimize_lineup(p, ["SUPER_FLEX", "QB", "RB", "WR"], "E_pts", False)
        self.assertEqual(total(got), 49.0)
        self.assertEqual({x["slot"]: x["name"] for x in got}["QB"], "QB1")

    def test_nan_scores_are_ineligible(self):
        p = pool([("A", "RB", np.nan), ("B", "RB", 5.0)])
        got = ask.optimize_lineup(p, ["RB", "RB"], "E_pts", False)
        self.assertEqual([x["name"] for x in got], ["B"])

    def test_unfillable_slot_left_out_and_order_independent(self):
        p = pool([("A", "RB", 5.0), ("B", "RB", 5.0)])
        r1 = ask.optimize_lineup(p, ["RB", "WR", "RB"], "E_pts", False)
        r2 = ask.optimize_lineup(p.sample(frac=1, random_state=1), ["RB", "WR", "RB"], "E_pts", False)
        self.assertEqual([(x["slot"], x["name"]) for x in r1], [(x["slot"], x["name"]) for x in r2])
        self.assertEqual([x["slot"] for x in r1], ["RB", "RB"])


class SideBySide(unittest.TestCase):
    def setUp(self):
        self.p = pool([("A", "RB", 10.0, 6.0), ("B", "RB", 9.0, 12.0), ("C", "WR", 8.0, 8.0)])
        self.e = ask.optimize_lineup(self.p, ["RB", "WR"], "E_pts", False)
        self.m = ask.optimize_lineup(self.p, ["RB", "WR"], "E_pts_model", False)

    def test_optima_differ_when_the_model_disagrees(self):
        self.assertEqual(self.e[0]["name"], "A")
        self.assertEqual(self.m[0]["name"], "B")

    def test_render_shows_totals_and_differences(self):
        out = ask.side_by_side(pd.DataFrame(self.e), pd.DataFrame(self.m))
        self.assertIn("A (10.0)", out)
        self.assertIn("B (12.0)", out)
        self.assertIn("!=", out)
        self.assertIn("same", out)
        self.assertIn("18.0 (E_pts lineup) vs 17.0 (model lineup)", out)
        self.assertIn("E_pts_model: 14.0 vs 20.0", out)
        self.assertIn("E_pts only: A | model only: B", out)

    def test_empty_model_renders_nothing(self):
        self.assertEqual(ask.side_by_side(pd.DataFrame(self.e), pd.DataFrame()), "")


if __name__ == "__main__":
    unittest.main()
