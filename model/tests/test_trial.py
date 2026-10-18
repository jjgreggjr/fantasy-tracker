"""model/trial.py: candidate parsing and composition, the verdict rule, recency weights, and a mocked end-to-end run.

    python -m unittest model.tests.test_trial

No network, no real matrix, no boosted-tree library: the walk-forward is mocked (or its model fit is), so the suite runs
wherever pandas does.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from model import backtest as B
from model import train as T
from model import trial as TR
from model.tests.test_backtest import TINY, synth

TINY_DOC = {"point": {lib: dict(TINY[lib]) for lib in T.LIBS}, "quantile": {}, "meta": {}}
CFG = TR.load_config()
BASE = TR.serving_spec(CFG)
# the previous 71-input primary (no add_families): the family add/drop rules are checked against it, the live spec is pinned below
BASE71 = TR.serving_spec(TR.load_config(Path(__file__).parent / "fixtures" / "serving_config_71.json"))


def d(rmse, lo, hi, pick=0.0, plo=-0.01, phi=0.01) -> pd.Series:
    return pd.Series({"dRMSE": rmse, "dRMSE_lo": lo, "dRMSE_hi": hi, "dPick": pick, "dPick_lo": plo, "dPick_hi": phi})


# ------------------------------------------------------------------ parsing and composition
class Parsing(unittest.TestCase):
    def test_each_kind_parses(self):
        c = TR.parse_candidate("+family:pbp_usage")
        self.assertEqual((c.kind, c.family), ("add_family", "pbp_usage"))
        c = TR.parse_candidate("-family:dvp")
        self.assertEqual((c.kind, c.family), ("drop_family", "dvp"))
        c = TR.parse_candidate("-cols:dvp_ppr_l2, is_home")
        self.assertEqual((c.kind, c.cols), ("drop_cols", ("dvp_ppr_l2", "is_home")))
        c = TR.parse_candidate("+cols:tm_out_group_n")
        self.assertEqual((c.kind, c.cols), ("add_cols", ("tm_out_group_n",)))
        c = TR.parse_candidate("param:lightgbm.reg_lambda=50")
        self.assertEqual((c.kind, c.lib, c.key, c.value), ("param", "lightgbm", "reg_lambda", 50))
        self.assertIsInstance(c.value, int)
        self.assertEqual(TR.parse_candidate("param:xgboost.colsample_bytree=0.6").value, 0.6)
        self.assertIs(TR.parse_candidate("param:catboost.boosting_type=Plain").value.__class__, str)
        c = TR.parse_candidate("recent:4")
        self.assertEqual((c.kind, c.n, c.factor), ("recent", 4, TR.RECENT_FACTOR))
        self.assertEqual(TR.parse_candidate("recent:6x2.5").factor, 2.5)

    def test_bad_candidates_raise_with_the_text(self):
        for bad in ("recent", "recent:0", "recent:abc", "+family:nope", "-cols:not_a_column", "param:lightgbm.x", "param:sklearn.a=1",
                    "wat:1", "", "+family:"):
            with self.assertRaises(ValueError, msg=bad):
                TR.parse_candidate(bad)

    def test_compose_builds_one_variant_from_several_flags(self):
        v = TR.compose(["recent:4", "param:lightgbm.num_leaves=31", "-cols:dvp_ppr_l2", "+cols:week"], BASE)
        self.assertEqual(v.recent, (4, TR.RECENT_FACTOR))
        self.assertEqual(v.params, {"lightgbm": {"num_leaves": 31}})
        self.assertIn("dvp_ppr_l2", v.spec.drop_cols)
        self.assertNotIn("week", v.spec.drop_cols)                       # un-pruned
        self.assertEqual(v.text, "recent:4 param:lightgbm.num_leaves=31 -cols:dvp_ppr_l2 +cols:week")
        self.assertTrue(v.changes_inputs(BASE))
        self.assertTrue(v.spec.name.startswith("trial_"))
        self.assertEqual(v.spec.name, TR.compose(["recent:4", "param:lightgbm.num_leaves=31", "-cols:dvp_ppr_l2", "+cols:week"], BASE).spec.name)
        self.assertEqual(v.lib_params("lightgbm")["num_leaves"], 31)
        self.assertEqual(v.lib_params("lightgbm")["n_estimators"], T.load_params()["point"]["lightgbm"]["n_estimators"])

    def test_families_add_and_drop_against_the_serving_spec(self):
        v = TR.compose(["+family:pbp_usage"], BASE71)                      # opt-in: added
        self.assertEqual(v.spec.add_families, ("pbp_usage",))
        v = TR.compose(["+family:adp"], BASE71)                            # dropped by the serving spec: un-dropped
        self.assertNotIn("adp", v.spec.drop_families)
        v = TR.compose(["-family:dvp"], BASE71)
        self.assertIn("dvp", v.spec.drop_families)
        v = TR.compose(["+family:pbp_usage", "-family:pbp_usage"], BASE71)  # added then dropped: gone again
        self.assertEqual(v.spec.add_families, ())
        for contradiction in (["+family:role"],            # already in
                              ["+family:weather"],         # unservable
                              ["-family:adp"],             # already out
                              ["-family:pbp_team"],        # opt-in, never in
                              ["+cols:pts_ppr_l1"],        # not pruned
                              ["-cols:week"],              # already dropped
                              ["recent:4", "recent:2"], ["param:lightgbm.a=1", "param:lightgbm.a=2"],
                              ["+family:pbp_usage", "+family:pbp_usage"], []):
            with self.assertRaises(ValueError, msg=contradiction):
                TR.compose(contradiction, BASE71)

    def test_families_against_the_live_spec_which_already_adds_market_and_ecr(self):
        self.assertEqual(BASE.add_families, ("market", "ecr"))
        self.assertEqual(TR.compose(["+family:pbp_usage"], BASE).spec.add_families, ("market", "ecr", "pbp_usage"))
        for fam in ("ecr", "market"):
            with self.assertRaises(ValueError):
                TR.compose([f"+family:{fam}"], BASE)
        self.assertEqual(TR.compose(["-family:ecr"], BASE).spec.add_families, ("market",))

    def test_a_recent_only_variant_keeps_the_primary_inputs(self):
        v = TR.compose(["recent:4"], BASE)
        self.assertFalse(v.changes_inputs(BASE))
        self.assertEqual(v.spec.drop_cols, BASE.drop_cols)


class Listing(unittest.TestCase):
    def test_list_marks_serving_inputs_and_explains_the_rest(self):
        text = TR.list_families(CFG)
        self.assertIn("* pts_ppr_l1", text)
        self.assertIn("week", text)
        self.assertRegex(text, r"  week\s+out: pruned \(\+cols:week to restore\)")
        self.assertRegex(text, r"adp_ppr\s+out: family dropped \(adp\)")
        self.assertRegex(text, r"pbp_usage  \[0/\d+ not in serving, opt-in\]")
        self.assertRegex(text, r"wx_temp_obs\s+out: weather")
        self.assertRegex(text, r"\n\s+season\s+out: season excluded")
        self.assertRegex(text, r"inj_report_status\s+out: pruned")      # the encoded name is what the inputs hold
        self.assertIn("* inj_weeks_since_report", text)


# ------------------------------------------------------------------ the verdict
class Verdict(unittest.TestCase):
    def test_the_four_branches(self):
        self.assertEqual(TR.verdict(d(-0.012, -0.020, -0.003), d(-0.004, -0.012, 0.005)), "SHIPS")
        self.assertEqual(TR.verdict(d(-0.008, -0.024, 0.007), d(-0.005, -0.017, 0.006)), "DIRECTION ONLY")
        self.assertEqual(TR.verdict(d(-0.008, -0.024, 0.007), d(0.004, -0.009, 0.016)), "DOES NOT SHIP")    # sign flips
        self.assertEqual(TR.verdict(d(0.067, 0.031, 0.101), d(0.088, 0.047, 0.127)), "DOES NOT SHIP")       # worse
        self.assertEqual(TR.verdict(d(-0.012, -0.020, -0.003), d(0.001, -0.01, 0.01)), "DOES NOT SHIP")     # 2025 significant, 2024 flips
        # the component rule
        self.assertEqual(TR.verdict(d(-0.008, -0.029, 0.013), d(0.006, -0.020, 0.034), rule="tie"), "TIE (components)")
        self.assertEqual(TR.verdict(d(-0.012, -0.020, -0.003), d(-0.004, -0.012, 0.005), rule="tie"), "SHIPS")
        self.assertEqual(TR.verdict(d(0.023, 0.006, 0.040), d(0.005, -0.01, 0.02), rule="tie"), "DOES NOT SHIP")
        self.assertEqual(TR.verdict(d(-0.01, -0.02, 0.0), d(0.03, 0.01, 0.05), rule="tie"), "DOES NOT SHIP")
        for v in ("SHIPS", "DIRECTION ONLY", "DOES NOT SHIP", "TIE (components)"):
            self.assertIn(v, TR.VERDICTS)
        with self.assertRaises(ValueError):
            TR.verdict(d(0, 0, 0), d(0, 0, 0), rule="other")

    def test_verdict_of_needs_both_judged_years(self):
        self.assertTrue(TR.verdict_of({"compare": {2025: None, 2024: None}, "weeks": {}}).startswith("NO VERDICT"))


# ------------------------------------------------------------------ recency weights
class RecentWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synth()

    def test_the_weight_vector(self):
        df = self.df
        tr = B.train_mask(df, 2026, 2, BASE)                          # training rows for the week-2 fit of 2026
        w = TR.recent_weights(df, tr, 2026, 2, 4, 3.0)
        key = df["season"] * 100 + df["week"]
        recent = key.isin([202601, 202518, 202517, 202516]).to_numpy()   # the 4 most recent (season, week) keys, across the boundary
        self.assertEqual(len(w), len(df))
        np.testing.assert_array_equal(w[tr & recent], 3.0)
        np.testing.assert_array_equal(w[tr & ~recent], 1.0)
        np.testing.assert_array_equal(w[~tr], 1.0)                    # unplayed / target-week rows: untouched, never trained on
        self.assertTrue((tr & recent).any() and (tr & ~recent).any())
        # n larger than the training history: every training row is recent
        self.assertTrue((TR.recent_weights(df, tr, 2026, 2, 10_000, 2.0)[tr] == 2.0).all())
        # the default factor
        self.assertEqual(TR.recent_weights(df, tr, 2026, 2, 4)[tr & recent][0], TR.RECENT_FACTOR)

    def test_walk_forward_hands_the_training_rows_weights_to_the_fit(self):
        df = self.df
        seen = {}

        def fake_fit(lib, params, Xtr, ytr, Xte, quantile=None, w=None):
            seen[len(seen)] = (len(Xtr), w)
            return None, np.zeros(len(Xte))
        fn = TR.weight_fn_for(TR.compose(["recent:4"], BASE))
        with mock.patch.object(B, "_fit_predict", side_effect=fake_fit):
            B.walk_forward(df, BASE, "lightgbm", TINY["lightgbm"], season=2026, weeks=(1, 2), weight_fn=fn)
        self.assertEqual(len(seen), 2)
        n_tr, w = seen[1]
        tr = B.train_mask(df, 2026, 2, BASE)
        self.assertEqual(n_tr, int(tr.sum()))
        np.testing.assert_array_equal(w, TR.recent_weights(df, tr, 2026, 2, 4)[tr])
        self.assertEqual(set(np.unique(w)), {1.0, TR.RECENT_FACTOR})
        # without a weight_fn nothing is passed (the shipped path is untouched)
        seen.clear()
        with mock.patch.object(B, "_fit_predict", side_effect=fake_fit):
            B.walk_forward(df, BASE, "lightgbm", TINY["lightgbm"], season=2026, weeks=(1,))
        self.assertIsNone(seen[0][1])

    def test_a_wrong_length_weight_vector_is_refused(self):
        with mock.patch.object(B, "_fit_predict", return_value=(None, np.zeros(1))):
            with self.assertRaises(ValueError):
                B.walk_forward(self.df, BASE, "lightgbm", TINY["lightgbm"], season=2026, weeks=(1,),
                               weight_fn=lambda df, tr, s, w: np.ones(3))


# ------------------------------------------------------------------ end to end, walk-forward mocked
def fake_walk_forward(better: float):
    """A walk-forward stand-in: predictions = actual + noise; the variant (recognised by its spec name) gets `better` x the noise."""
    def wf(df, spec, lib, params, *, season, weeks, weight_fn=None, **kw):
        rng = np.random.default_rng(season * 7 + (1 if spec.name.startswith("trial_") else 0))
        te = df["season"].eq(season) & df["week"].isin(weeks)
        part = df.loc[te, [c for c in B.KEEP if c in df.columns]].copy()
        part["y"] = df.loc[te, T.TARGET].to_numpy()
        y = part["y"].fillna(0).to_numpy()
        scale = better if spec.name.startswith("trial_") else 1.0
        part["pred"] = y + rng.normal(0, 3.0 * scale, len(part))
        part["n_train"], part["fit_week"] = 0, part["week"]
        return part
    return wf


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synth()

    def run_trial(self, cands, better, **kw):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(TR, "OUT", Path(tmp)), \
                mock.patch.object(T, "load_params", return_value=TINY_DOC), \
                mock.patch.object(TR, "tree_count", return_value=50), \
                mock.patch.object(B, "walk_forward", side_effect=fake_walk_forward(better)):
            v = TR.compose(cands, BASE)
            res = TR.trial(self.df, "k", v, BASE, draws=200, log=lambda *_: None, **kw)
            return v, res

    def test_seasons_with_labels_sees_the_live_weeks(self):
        have = TR.seasons_with_labels(self.df)
        self.assertEqual(have[2026], (1, 2, 3))
        self.assertEqual(have[2025], tuple(range(1, 19)))

    def test_a_clearly_better_variant_ships_and_the_row_is_appended(self):
        v, res = self.run_trial(["recent:4"], 0.5)
        self.assertEqual(res["weeks"][2026], (1, 2, 3))
        self.assertEqual(TR.verdict_of(res), "SHIPS")
        row = TR.table_row(res, v, "lightgbm", "rmse", today="2026-10-15")
        self.assertEqual(row["candidate"], "`recent:4`")
        self.assertRegex(row["2025 dRMSE"], r"^-\d\.\d{3} \[-\d\.\d{3}, -\d\.\d{3}\]$")
        self.assertIn("(3 wk, n=", row["2026 dRMSE"])
        self.assertEqual(row["verdict"], "SHIPS")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "trials.md"
            TR.append_report(row, p)
            TR.append_report(row, p)
            text = p.read_text()
            self.assertTrue(text.startswith("# Trials"))
            self.assertEqual(text.count("| 2026-10-15 | `recent:4` |"), 2)
            self.assertEqual(text.count("| date |"), 1)
        lines = TR.per_position_lines(res)
        self.assertTrue(any(l.startswith("  2025 by position") for l in lines))
        steps = TR.adoption_steps(v, CFG, "lightgbm")
        self.assertTrue(any("python -m model.serve --freeze" in s for s in steps))
        self.assertTrue(any("weight_fn" in s for s in steps))
        self.assertTrue(any("--lib average" in s for s in steps))

    def test_a_clearly_worse_variant_does_not_ship(self):
        v, res = self.run_trial(["-cols:dvp_ppr_l4"], 1.6)
        self.assertEqual(TR.verdict_of(res), "DOES NOT SHIP")
        self.assertEqual(TR.verdict_of(res, rule="tie"), "DOES NOT SHIP")
        steps = TR.adoption_steps(v, CFG, "average")
        self.assertTrue(any("serving_config.json" in s and "drop_cols" in s for s in steps))
        self.assertFalse(any("--lib average" in s for s in steps))

    def test_the_live_year_can_be_skipped_or_absent(self):
        v, res = self.run_trial(["recent:4"], 0.5, seasons=(2025, 2024))
        self.assertNotIn(2026, res["compare"])
        row = TR.table_row(res, v, "lightgbm", "rmse")
        self.assertEqual(row["2026 dRMSE"], "n/a")
        df = self.df[self.df["season"] < 2026]
        with mock.patch.object(TR, "OUT", Path(tempfile.mkdtemp())), mock.patch.object(T, "load_params", return_value=TINY_DOC), \
                mock.patch.object(B, "walk_forward", side_effect=fake_walk_forward(0.5)):
            res = TR.trial(df, "k", v, BASE, draws=50, log=lambda *_: None)
        self.assertIsNone(res["compare"][2026])
        self.assertEqual(res["weeks"][2026], ())

    def test_params_for_retrees_only_when_inputs_or_shape_change(self):
        with mock.patch.object(T, "load_params", return_value=TINY_DOC), mock.patch.object(TR, "tree_count", return_value=77) as tc:
            v = TR.compose(["recent:4"], BASE)
            self.assertEqual(TR.params_for(self.df, "k", v, BASE, "lightgbm")["n_estimators"], TINY["lightgbm"]["n_estimators"])
            tc.assert_not_called()
            v = TR.compose(["param:lightgbm.num_leaves=31"], BASE)
            p = TR.params_for(self.df, "k", v, BASE, "lightgbm")
            self.assertEqual((p["num_leaves"], p["n_estimators"]), (31, 77))
            v = TR.compose(["-cols:dvp_ppr_l4"], BASE)
            self.assertEqual(TR.params_for(self.df, "k", v, BASE, "xgboost")["n_estimators"], 77)
            self.assertEqual(TR.params_for(self.df, "k", v, BASE, "xgboost", retree=False)["n_estimators"], TINY["xgboost"]["n_estimators"])
            v = TR.compose(["param:lightgbm.n_estimators=300"], BASE)      # an explicit count is taken as given
            self.assertEqual(TR.params_for(self.df, "k", v, BASE, "lightgbm")["n_estimators"], 300)


if __name__ == "__main__":
    unittest.main()
