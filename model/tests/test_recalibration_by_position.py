"""`underage_by_position` quantile recalibration (model/ensemble.py) and the coverage report, on synthetic residual histories.

    python -m unittest model.tests.test_recalibration_by_position

What is proven: a position whose upper tail is short (RB: actuals above q90 far more than 10%) gets its p90 stretched and its
p10 left alone; an over-covered position (QB: band twice as wide as it should be) is shrunk ONLY with `allow_shrink`; a position
with fewer than `min_pos_rows` trailing rows falls back to the pooled factor; both directions respect `factor_bounds`; q50 never
moves and the order p10 <= p50 <= p90 holds; `underage_only` is byte-for-byte unchanged; `recal_kwargs` reads the config
section; `coverage_report` prints the miss review's table.
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from model import ensemble as E

# position -> (true residual sd, sd the band was built for): RB's band is too narrow above, QB's too wide, WR/TE calibrated
SD = {"QB": (3.0, 6.0), "RB": (3.0, 3.0), "WR": (3.0, 3.0), "TE": (3.0, 3.0)}
Z90 = 1.2816


def q_frame(seed: int, *, weeks: int = 18, per_pos: int = 40, positions=("QB", "RB", "WR", "TE"), rb_skew: float = 4.0) -> pd.DataFrame:
    """Walk-forward quantile rows: a symmetric band of the stated sd about a true mean; RB actuals get an upper-tail bump
    (`rb_skew` added to a fifth of the rows) so only the RB UPPER tail is short."""
    rng = np.random.default_rng(seed)
    rows = []
    for wk in range(1, weeks + 1):
        for pos in positions:
            sd_true, sd_band = SD[pos]
            for i in range(per_pos):
                mu = rng.normal(12, 3)
                y = mu + rng.normal(0, sd_true)
                if pos == "RB" and i % 5 == 0:
                    y += rb_skew
                rows.append((wk, pos, y, mu - Z90 * sd_band, mu, mu + Z90 * sd_band, 1))
    return pd.DataFrame(rows, columns=["week", "position", "y", "q10", "q50", "q90", "y_played"])


def tail_shares(e: pd.DataFrame, lo: str, hi: str) -> tuple[float, float]:
    return float((e["y"] < e[lo]).mean()), float((e["y"] > e[hi]).mean())


class ByPosition(unittest.TestCase):
    def setUp(self):
        self.hist, self.cur = q_frame(1), q_frame(2)

    def test_a_short_upper_tail_is_stretched_and_the_other_tail_and_q50_are_untouched(self):
        out = E.recalibrate(self.cur, self.hist, mode="underage_by_position")
        rb = out[out["position"] == "RB"]
        below_raw, above_raw = tail_shares(rb, "q10", "q90")
        below_new, above_new = tail_shares(rb, "q10a", "q90a")
        self.assertGreater(above_raw, 0.15, "control: the synthetic RB upper tail is short")
        self.assertLess(above_new, above_raw - 0.03)
        self.assertLess(abs(above_new - 0.10), abs(above_raw - 0.10) / 2)
        self.assertTrue((rb["q90a"] > rb["q90"]).all())
        np.testing.assert_array_equal(rb["q10a"].to_numpy(), rb["q10"].to_numpy())                   # the lower tail was fine
        self.assertAlmostEqual(below_new, below_raw)
        np.testing.assert_array_equal(out["q50"].to_numpy(), self.cur["q50"].to_numpy())
        self.assertTrue(((out["q10a"] <= out["q50"]) & (out["q50"] <= out["q90a"])).all())
        # the stretch is multiplicative about q50: one factor per (week, position)
        f = (rb["q90a"] - rb["q50"]) / (rb["q90"] - rb["q50"])
        self.assertLessEqual(f.groupby(rb["week"]).std().max(), 1e-9)
        self.assertTrue(((f >= 1.0) & (f <= E.FACTOR_BOUNDS[1])).all())

    def test_calibrated_positions_are_left_exactly_alone(self):
        out = E.recalibrate(self.cur, self.hist, mode="underage_by_position")
        for pos in ("WR", "TE"):
            g = out[out["position"] == pos]
            np.testing.assert_array_equal(g["q10a"].to_numpy(), g["q10"].to_numpy(), err_msg=pos)
            np.testing.assert_array_equal(g["q90a"].to_numpy(), g["q90"].to_numpy(), err_msg=pos)

    def test_an_over_covered_band_is_shrunk_only_when_allowed_and_only_within_the_bounds(self):
        off = E.recalibrate(self.cur, self.hist, mode="underage_by_position")
        qb = off[off["position"] == "QB"]
        self.assertGreater(E.coverage(qb["y"], qb["q10"], qb["q90"]), 0.95, "control: the synthetic QB band is twice too wide")
        np.testing.assert_array_equal(qb["q10a"].to_numpy(), qb["q10"].to_numpy())
        np.testing.assert_array_equal(qb["q90a"].to_numpy(), qb["q90"].to_numpy())
        on = E.recalibrate(self.cur, self.hist, mode="underage_by_position", allow_shrink=True)
        qb = on[on["position"] == "QB"]
        self.assertTrue((qb["q90a"] < qb["q90"]).all() and (qb["q10a"] > qb["q10"]).all())
        f_hi = (qb["q90a"] - qb["q50"]) / (qb["q90"] - qb["q50"])
        f_lo = (qb["q50"] - qb["q10a"]) / (qb["q50"] - qb["q10"])
        self.assertTrue(np.allclose(f_hi, E.FACTOR_BOUNDS[0]) and np.allclose(f_lo, E.FACTOR_BOUNDS[0]),
                        "a band twice too wide wants 0.5, the floor is 0.7")
        cov = E.coverage(qb["y"], qb["q10a"], qb["q90a"])
        self.assertLess(cov, E.coverage(qb["y"], qb["q10"], qb["q90"]))
        self.assertGreater(cov, 0.80)
        # a looser floor lets it go further, and the RB stretch is the same with or without shrink
        loose = E.recalibrate(self.cur, self.hist, mode="underage_by_position", allow_shrink=True, factor_bounds=(0.4, 1.6))
        qb2 = loose[loose["position"] == "QB"]
        self.assertLess(float(qb2["q90a"].mean()), float(qb["q90a"].mean()))
        rb_on, rb_off = on[on["position"] == "RB"], off[off["position"] == "RB"]
        np.testing.assert_array_equal(rb_on["q90a"].to_numpy(), rb_off["q90a"].to_numpy())

    def test_the_upper_bound_caps_a_stretch(self):
        hist, cur = q_frame(1, rb_skew=40.0), q_frame(2, rb_skew=40.0)
        out = E.recalibrate(cur, hist, mode="underage_by_position")
        rb = out[out["position"] == "RB"]
        f = (rb["q90a"] - rb["q50"]) / (rb["q90"] - rb["q50"])
        self.assertTrue(np.allclose(f, E.FACTOR_BOUNDS[1]))

    def test_a_position_with_too_few_rows_takes_the_pooled_factor(self):
        # no QB (its wide band would hide the RB bump in the pool); 60 TE history rows, fewer than min_pos_rows=200 until week 8 of cur
        pos = ("RB", "WR", "TE")
        full, cur = q_frame(1, positions=pos), q_frame(2, positions=pos)
        hist = pd.concat([full[full["position"] != "TE"], full[full["position"] == "TE"].iloc[:60]])
        pooled = E.recalibrate(cur, hist, mode="underage_by_position", min_pos_rows=10 ** 6, positions=pos)   # everything pooled
        own = E.recalibrate(cur, hist, mode="underage_by_position", min_pos_rows=200, positions=pos)
        wk = lambda d, pos_, w=1: d[(d["week"] == w) & (d["position"] == pos_)]                  # noqa: E731
        np.testing.assert_array_equal(wk(own, "TE")["q90a"].to_numpy(), wk(pooled, "TE")["q90a"].to_numpy())
        self.assertFalse(np.array_equal(wk(own, "RB")["q90a"].to_numpy(), wk(pooled, "RB")["q90a"].to_numpy()),
                         "RB has its own 720 rows and its own, larger, factor")
        pooled_above = float((hist["y"] > hist["q90"]).mean())
        self.assertGreater(pooled_above, 0.10 + E.TOLERANCE / 2, "control: the pooled upper tail is short because of RB")
        self.assertTrue((wk(own, "TE")["q90a"] > wk(own, "TE")["q90"]).all(), "so the pooled factor stretches TE too")
        # once TE has 200 trailing rows of its own (60 + 40 a week: week 5 of cur), its own calibrated tail is left alone
        late = wk(own, "TE", 10)
        np.testing.assert_array_equal(late["q90a"].to_numpy(), late["q90"].to_numpy())
        # and below min_rows overall nothing moves at all
        few = E.recalibrate(cur.loc[cur["week"] == 1], hist.iloc[:50], mode="underage_by_position")
        np.testing.assert_array_equal(few["q90a"].to_numpy(), few["q90"].to_numpy())

    def test_week_n_is_adjusted_from_rows_before_it_only(self):
        clean = E.recalibrate(self.cur, self.hist, mode="underage_by_position")
        dirty = self.cur.copy()
        dirty.loc[dirty["week"] >= 3, "y"] = 987654.0
        d = E.recalibrate(dirty, self.hist, mode="underage_by_position")
        m = (self.cur["week"] <= 3).to_numpy()
        for col in ("q10a", "q90a"):
            np.testing.assert_array_equal(clean.loc[m, col].to_numpy(), d.loc[m, col].to_numpy())
        self.assertFalse(np.array_equal(clean.loc[clean["week"] == 8, "q90a"].to_numpy(), d.loc[d["week"] == 8, "q90a"].to_numpy()))

    def test_underage_only_is_unchanged_and_the_positions_argument_still_confines(self):
        a = E.recalibrate(self.cur, self.hist)
        b = E.recalibrate(self.cur, self.hist, mode="underage_only", allow_shrink=True, min_pos_rows=1)
        pd.testing.assert_frame_equal(a, b)
        qb_only = E.recalibrate(self.cur, self.hist, mode="underage_by_position", allow_shrink=True, positions=("QB",))
        rb = qb_only[qb_only["position"] == "RB"]
        np.testing.assert_array_equal(rb["q90a"].to_numpy(), rb["q90"].to_numpy())
        with self.assertRaises(ValueError):
            E.recalibrate(self.cur, self.hist, mode="bogus")

    def test_recal_kwargs_reads_the_config_section(self):
        self.assertEqual(E.recal_kwargs({"mode": "underage_only"}), {"mode": "underage_only"})
        self.assertEqual(E.recal_kwargs({"mode": "underage_by_position", "allow_shrink": True, "min_pos_rows": 150, "factor_bounds": [0.8, 1.5]}),
                         {"mode": "underage_by_position", "allow_shrink": True, "min_pos_rows": 150, "factor_bounds": (0.8, 1.5)})
        import json
        from pathlib import Path
        cfg = json.loads((Path(E.__file__).with_name("serving_config.json")).read_text())
        self.assertEqual(cfg["recalibration"], {"mode": "underage_only"}, "the live mode is a frozen serving decision")
        self.assertIn("underage_by_position", cfg["note"])
        for key in ("allow_shrink", "min_pos_rows", "factor_bounds"):
            self.assertIn(key, cfg["note"])


class CoverageReport(unittest.TestCase):
    def test_the_table_matches_hand_values_and_prints(self):
        rows = pd.DataFrame({"position": ["QB", "QB", "RB", "RB", "RB", "TE"], "y": [5.0, 20.0, 1.0, 3.0, 9.0, np.nan],
                             "q10": [4.0, 4.0, 2.0, 2.0, 2.0, 0.0], "q90": [10.0, 10.0, 6.0, 6.0, 6.0, 1.0]})
        t = E.coverage_frame(rows)
        self.assertEqual(list(t.index), ["QB", "RB", "ALL"])                                        # the NaN TE row is dropped
        self.assertEqual(t.loc["QB", "n"], 2)
        self.assertEqual((t.loc["QB", "inside"], t.loc["QB", "above_hi"], t.loc["QB", "below_lo"]), (0.5, 0.5, 0.0))
        self.assertEqual((t.loc["RB", "inside"], t.loc["RB", "above_hi"], t.loc["RB", "below_lo"]), (1 / 3, 1 / 3, 1 / 3))
        self.assertEqual((t.loc["ALL", "n"], t.loc["ALL", "inside"]), (5, 0.4))
        buf = io.StringIO()
        with redirect_stdout(buf):
            text = E.coverage_report(rows, "position")
        self.assertEqual(buf.getvalue().strip(), text.strip())
        self.assertIn("| RB | 3 | 33.3% | 33.3% | 33.3% |", text)
        self.assertIn("| ALL | 5 | 40.0% | 40.0% | 20.0% |", text)
        quiet = E.coverage_report(rows.rename(columns={"q10": "p10", "q90": "p90"}), lo="p10", hi="p90", out=None)
        self.assertIn("| QB | 2 | 50.0% | 50.0% | 0.0% |", quiet)


if __name__ == "__main__":
    unittest.main()
