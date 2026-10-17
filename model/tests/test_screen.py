"""model/screen.py: the within-group Spearman, rho_y / rho_resid on a frame with known answers, the verdict rules, null rates,
the trial command, the gain step's bookkeeping (model fit mocked) and the report.

    python -m unittest model.tests.test_screen

Offline, no real matrix, no boosted-tree library.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from model import screen as S
from model import train as T
from model import trial as TR
from model.tests.test_backtest import synth

CFG = TR.load_config()
BASE = TR.serving_spec(CFG)


def frame(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """synth() with a known model: prediction = 0.8 * pts_ppr_std_mean, residual = y - prediction.
    Returns (df, residual, screened rows 2022-2025, 2026 rows)."""
    df = synth(seed)
    pred = 0.8 * df["pts_ppr_std_mean"].to_numpy()
    resid = df[T.TARGET].to_numpy() - pred
    h2h = ((df["y_played"] == 1) & df["base_trail3_ppr"].notna()).to_numpy()
    rows = h2h & df["season"].between(2022, 2025).to_numpy()
    live = h2h & (df["season"] == 2026).to_numpy()
    resid = np.where(rows, resid, np.nan)
    return df, resid, rows, live


class Spearman(unittest.TestCase):
    def test_perfect_monotone_and_reversed(self):
        g = np.repeat(["a", "b", "c"], 10)
        x = np.tile(np.arange(10.0), 3)
        self.assertAlmostEqual(S.within_spearman(x, x ** 3, g)[0], 1.0)
        self.assertAlmostEqual(S.within_spearman(x, -x, g)[0], -1.0)
        self.assertEqual(S.within_spearman(x, x, g)[1], 3)

    def test_matches_scipy_style_rank_pearson_per_group(self):
        rng = np.random.default_rng(1)
        x, y = rng.normal(size=60), rng.normal(size=60)
        x[::7] = np.round(x[::7])                                     # some ties
        g = np.repeat(["a", "b", "c"], 20)
        want = np.mean([pd.Series(x[g == k]).corr(pd.Series(y[g == k]), method="spearman") for k in "abc"])
        self.assertAlmostEqual(S.within_spearman(x, y, g)[0], want, places=10)

    def test_nan_pairwise_small_and_constant_groups_skipped(self):
        g = np.array(["a"] * 10 + ["b"] * 10 + ["c"] * 3)
        x = np.r_[np.arange(10.0), np.full(10, 5.0), np.arange(3.0)]  # b constant, c too small
        y = np.r_[np.arange(10.0), np.arange(10.0), np.arange(3.0)]
        x[0] = np.nan
        m, k, _ = S.within_spearman(x, y, g)
        self.assertEqual(k, 1)
        self.assertAlmostEqual(m, 1.0)
        self.assertTrue(np.isnan(S.within_spearman(np.full(5, np.nan), np.arange(5.0), ["a"] * 5)[0]))


class Rhos(unittest.TestCase):
    def test_redundant_copy_vs_informative_residual_copy(self):
        df, resid, rows, live = frame()
        rng = np.random.default_rng(3)
        values = {
            "redundant": 2.5 * df["pts_ppr_std_mean"].to_numpy() + 7.0,      # a linear copy of a primary input
            "informative": np.where(rows, resid, rng.normal(size=len(df))),  # a copy of the residual
            "noise": rng.normal(size=len(df)),
        }
        t = S.rho_table(values, df, rows, resid, live).set_index("column")
        self.assertGreater(t.loc["redundant", "rho_y"], 0.4)
        self.assertLess(abs(t.loc["redundant", "rho_resid"]), 0.15)
        self.assertAlmostEqual(t.loc["informative", "rho_resid"], 1.0, places=6)
        self.assertGreater(t.loc["informative", "rho_y"], 0.4)              # the residual is part of y
        self.assertLess(abs(t.loc["noise", "rho_y"]), 0.1)
        self.assertLess(abs(t.loc["noise", "rho_resid"]), 0.1)
        self.assertGreater(abs(t.loc["informative", "t_resid"]), 50)

    def test_categorical_is_screened_encoded(self):
        df, resid, rows, live = frame()
        enc = S.numeric(df, "inj_report_status")
        self.assertEqual(list(enc.columns), ["inj_report_ord"])
        self.assertEqual(S.null_rate(enc["inj_report_ord"]), 0.0)         # 'no designation' is 0, not missing
        two = S.numeric(df, "wx_roof_obs")
        self.assertEqual(list(two.columns), ["wx_roof_dome", "wx_roof_retractable"])
        t = S.rho_table({"wx_roof_obs": two}, df, rows, resid, live)
        self.assertEqual(len(t), 1)
        self.assertIn(t.loc[0, "encoded"], two.columns)


class NullRates(unittest.TestCase):
    def test_overall_and_live(self):
        df, resid, rows, live = frame()
        x = np.ones(len(df))
        x[(df["season"] == 2026).to_numpy()] = np.nan                    # all-NaN live
        x[np.flatnonzero(rows)[::4]] = np.nan                            # a quarter missing on the screened rows
        t = S.rho_table({"x": x}, df, rows, resid, live)
        self.assertAlmostEqual(t.loc[0, "null"], np.isnan(x[rows]).mean())
        self.assertAlmostEqual(t.loc[0, "null"], 0.25, delta=0.01)
        self.assertEqual(t.loc[0, "null_2026"], 1.0)
        self.assertTrue(np.isnan(S.null_rate([])))
        no_live = np.zeros(len(df), bool)
        self.assertTrue(np.isnan(S.rho_table({"x": x}, df, rows, resid, no_live).loc[0, "null_2026"]))


def vt(rows) -> pd.DataFrame:
    cols = ["column", "kind", "rho_y", "rho_resid", "null_2026", "drmse_2024", "drmse_2025"]
    return pd.DataFrame(rows, columns=cols)


class Verdicts(unittest.TestCase):
    def table(self):
        # 30 candidates: the 90th percentile falls between the 27th and 28th, so exactly the 3 large |rho_resid| are top decile
        filler = [(f"f{i}", "candidate", 0.01, 0.001 * i, 0.0, 0.001, 0.001) for i in range(25)]   # low everything
        return vt(filler + [
            ("promising", "candidate", 0.05, 0.30, 0.10, -0.01, -0.002),
            ("unconfirmed", "candidate", 0.05, -0.35, 0.10, -0.01, +0.002),       # sign flips across years
            ("skewed", "candidate", 0.05, 0.40, 0.50, -0.01, -0.01),               # exactly 50% null live
            ("redundant", "candidate", -0.30, 0.01, 0.0, 0.0, 0.0),
            ("noise", "candidate", 0.02, 0.0, 0.0, -0.1, -0.1),
            ("placebo_1", "placebo", 0.0, 0.9, 0.0, -0.1, -0.1),                   # never counts toward the decile
            ("dead", "dead", np.nan, np.nan, 1.0, np.nan, np.nan),
            ("ref", "in", 0.5, 0.0, 0.0, np.nan, np.nan),
        ])

    def test_rules(self):
        t = self.table()
        v, thr = S.assign_verdicts(t)
        got = dict(zip(t["column"], v))
        self.assertEqual(got["promising"], "PROMISING")
        self.assertEqual(got["unconfirmed"], "UNCONFIRMED")
        self.assertEqual(got["skewed"], "SKEW")
        self.assertEqual(got["redundant"], "REDUNDANT")
        self.assertEqual(got["noise"], "NOISE")
        self.assertEqual(got["placebo_1"], "PLACEBO")
        self.assertEqual(got["dead"], "DEAD")
        self.assertEqual(got["ref"], "IN (reference)")
        self.assertTrue(all(got[f"f{i}"] == "NOISE" for i in range(25)))
        cand = t[t["kind"] == "candidate"]["rho_resid"].abs()
        self.assertAlmostEqual(thr, cand.quantile(0.9))
        self.assertLess(thr, 0.9)                                                  # the placebo's 0.9 was not counted

    def test_missing_gain_is_not_better_and_live_nan_is_not_skew(self):
        t = self.table()
        t.loc[t["column"] == "promising", "drmse_2025"] = np.nan
        t.loc[t["column"] == "skewed", "null_2026"] = np.nan
        v = dict(zip(t["column"], S.assign_verdicts(t)[0]))
        self.assertEqual(v["promising"], "UNCONFIRMED")
        self.assertEqual(v["skewed"], "PROMISING")                                # no 2026 rows: the skew rule cannot fire


class TrialCommand(unittest.TestCase):
    def test_pruned_go_as_cols_opt_in_as_family_and_compose(self):
        pruned = list(BASE.drop_cols[:2])
        rows = pd.DataFrame({"column": pruned + ["resid_xfp_l1", "over_xfp_streak", "adp_ppr"],
                             "family": [T.F.family_of(c) for c in pruned] + ["streak", "streak", "adp"]})
        cands, notes = S.trial_candidates(rows, BASE)
        self.assertEqual(cands, ["+family:streak", "+family:adp", f"+cols:{','.join(pruned)}"])
        TR.compose(cands, BASE)
        cmd = S.trial_command(cands)
        self.assertTrue(cmd.startswith("python -m model.trial --candidate +family:streak"))
        self.assertTrue(any("2 screened PROMISING" in n for n in notes))
        self.assertEqual(S.trial_command([]), "")

    def test_weather_is_never_a_trial(self):
        cands, notes = S.trial_candidates(pd.DataFrame({"column": ["wx_temp_obs"], "family": ["weather"]}), BASE)
        self.assertEqual(cands, [])
        self.assertIn("cannot be added", notes[0])


class Candidates(unittest.TestCase):
    def test_default_set_excludes_inputs_weather_and_season(self):
        df = synth()
        df["resid_xfp_l1"] = 1.0
        df["tm_out_group_n"] = np.arange(len(df)) % 3
        c = S.candidates(df, CFG).set_index("column")
        self.assertNotIn("pts_ppr_std_mean", c.index)                         # an input
        self.assertNotIn("wx_temp_obs", c.index)
        self.assertNotIn("season", c.index)
        self.assertEqual(c.loc["resid_xfp_l1", "family"], "streak")
        self.assertEqual(c.loc["resid_xfp_l1", "kind"], "dead")               # constant in 2021-2024
        self.assertEqual(c.loc["tm_out_group_n", "kind"], "candidate")
        self.assertEqual(c.loc["college_breakout_age", "kind"], "dead")

    def test_named_families_include_references_and_weather(self):
        c = S.candidates(synth(), CFG, ("lags", "weather")).set_index("column")
        self.assertEqual(c.loc["pts_ppr_std_mean", "kind"], "in")
        self.assertIn("wx_temp_obs", c.index)
        self.assertEqual(set(c["family"]), {"lags", "weather"})


class GainStep(unittest.TestCase):
    def test_bookkeeping_with_a_mocked_fit_and_the_cache(self):
        df, resid, rows, live = frame()
        base_cols = ["pts_ppr_std_mean", "opps_l1"]
        calls = []

        def fake_fit(Xtr, ytr, Xte, params):
            calls.append(list(Xtr.columns))
            # the prediction uses the informative column when present: better RMSE; gain all on the last column
            pred = 0.8 * Xte["pts_ppr_std_mean"].to_numpy() + (0.1 * Xte["opps_l1"].to_numpy() if "good" in Xte else 0)
            gain = np.ones(Xtr.shape[1])
            return pred, gain

        extra = {"good": pd.DataFrame({"good": df["opps_l1"].to_numpy()}, index=df.index),
                 "bad": pd.DataFrame({"bad": np.zeros(len(df))}, index=df.index)}
        h2h = ((df["y_played"] == 1) & df["base_trail3_ppr"].notna())
        by_year = {y: (h2h & (df["season"] == y)).to_numpy() for y in S.GAIN_YEARS}
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / "fits.parquet"
            g = S.gain_screen(df, "k", base_cols, extra, by_year, {}, fit=fake_fit, cache=cache, log=lambda m: None)
            self.assertEqual(len(calls), 2 * 3)                               # primary + 2 columns, 2 years
            self.assertTrue(all(c[-1] in ("good", "bad", "opps_l1") for c in calls))
            gi = g.set_index(["column", "year"])
            for y in S.GAIN_YEARS:
                self.assertLess(gi.loc[("good", y), "drmse"], 0)
                self.assertAlmostEqual(gi.loc[("bad", y), "drmse"], 0.0)
                self.assertAlmostEqual(gi.loc[("good", y), "gain"], 1 / 7)  # 4 position one-hots + 2 base + 1
            n = len(calls)
            g2 = S.gain_screen(df, "k", base_cols, extra, by_year, {}, fit=fake_fit, cache=cache, log=lambda m: None)
            self.assertEqual(len(calls), n)                                   # all from the cache
            pd.testing.assert_frame_equal(g, g2)


class Report(unittest.TestCase):
    def test_writes_table_reading_and_command(self):
        df, resid, rows, live = frame()
        pruned = BASE.drop_cols[0]
        t = vt([(pruned, "candidate", 0.05, 0.30, 0.0, -0.01, -0.01),
                ("resid_xfp_l1", "candidate", 0.2, 0.02, 0.0, 0.01, 0.01),
                ("part_pass_snap_share_t3", "candidate", 0.1, 0.25, 0.95, -0.01, -0.01),
                *[(f"x{i}", "candidate", 0.0, 0.001 * i, 0.0, 0.0, 0.0) for i in range(10)],
                ("placebo_1", "placebo", 0.0, 0.004, 0.0, 0.0003, -0.0002)])
        t["family"] = [T.F.family_of(pruned), "streak", "pbp_part", *["static"] * 10, "placebo"]
        for c in ("t_resid", "null", "gain_2024", "gain_2025"):
            t[c] = 0.01
        t["verdict"], thr = S.assign_verdicts(t)
        meta = {"date": "2026-10-16", "mkey": "abc123", "shape": (100, 10), "years": (2022, 2025), "base": BASE,
                "n_rows": 1234, "n_live": 56, "live_weeks": "1-5", "out": Path("/tmp/x.parquet"), "runtime": "1.0 min"}
        with tempfile.TemporaryDirectory() as d:
            p = S.write_report(t, thr, meta, Path(d) / "sub" / "screen.md", top=5)
            text = p.read_text()
        self.assertIn("abc123", text)
        self.assertIn("1,234", text)
        self.assertIn("screen-grade", text)
        self.assertIn(f"python -m model.trial --candidate +cols:{pruned}", text)
        self.assertIn("## Reading", text)
        self.assertIn("Noise floor", text)
        self.assertIn("SKEW", text)                                           # part_ column: 95% null live
        body = text.split("## Reading")[0]
        self.assertNotIn("| placebo |", body)                                 # placebos are not table rows
        self.assertEqual(body.count("| `x"), 2)                               # top 5 by |rho_resid|: 3 named + x9, x8


if __name__ == "__main__":
    unittest.main()
