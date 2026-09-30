"""Phase 2: the walk-forward is the leakage gate of the modelling stage.

    model/.venv/bin/python -m unittest model.tests.test_backtest

The pipeline gate (as_of_join) keeps FEATURES pre-kickoff. These tests keep the FITTING honest: no training row
is at or after the target week (mask, runtime assertion, and a perturbation proof that garbage in every later row
changes no prediction), reruns reproduce every number for all three libraries, and the metric code is checked
against hand-computed answers. Synthetic frames (fast, no downloads); the real-matrix checks skip only if the
git-ignored matrix has not been built.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from model import backtest as B
from model import features as F
from model import train as T

TINY = {"lightgbm": dict(n_estimators=15, learning_rate=0.1, num_leaves=7, min_child_samples=5),
        "xgboost": dict(n_estimators=15, learning_rate=0.1, max_depth=3),
        "catboost": dict(iterations=15, learning_rate=0.1, depth=3)}


def synth(seed: int = 0) -> pd.DataFrame:
    """A matrix-shaped frame: 2021-2025 (18 weeks) plus 2026 weeks 1-3, 80 players a week."""
    rng = np.random.default_rng(seed)
    rows = []
    for season, weeks in [(s, 18) for s in range(2021, 2026)] + [(2026, 3)]:
        for week in range(1, weeks + 1):
            for pid in range(80):
                rows.append((f"p{pid}", f"{season}_{week:02d}_{pid % 20}", season, week, T.POSITIONS[pid % 4], f"T{pid % 20}"))
    df = pd.DataFrame(rows, columns=["player_id", "game_id", "season", "week", "position", "team"])
    n = len(df)
    df["pts_ppr_std_mean"] = rng.normal(8, 3, n)
    df["opps_l1"] = rng.integers(0, 25, n).astype(float)
    df["team_spread"] = rng.normal(0, 6, n)
    df["is_home"] = rng.integers(0, 2, n)
    df["dvp_ppr_l4"] = rng.normal(20, 5, n)
    df["wx_temp_obs"] = rng.normal(60, 15, n)
    df["wx_roof_obs"] = rng.choice(["dome", "outdoors", "retractable"], n)
    df["college_breakout_age"] = np.nan                       # an all-NA registry column
    df["inj_report_status"] = rng.choice([None, "Questionable", "Doubtful", "Out", "Note"], n, p=[.8, .12, .03, .03, .02])
    df["inj_practice_status"] = rng.choice([None, "Full Participation in Practice", "Limited Participation in Practice"], n)
    played = rng.random(n) < 0.85
    df["y_played"] = played.astype("int8")
    df["y_has_stats_row"] = (played & (rng.random(n) < 0.9)).astype("int8")
    df["y_points_ppr"] = np.where(played, 0.8 * df["pts_ppr_std_mean"] + 0.1 * df["opps_l1"] + rng.normal(0, 3, n), np.nan)
    df["spine_depth_only"] = (rng.random(n) < 0.05).astype("int8")
    df["is_rookie"] = 0
    df["base_trail3_ppr"] = df["pts_ppr_std_mean"] + rng.normal(0, 1, n)
    df["base_xfp_sameweek"] = df["y_points_ppr"] + rng.normal(0, 1, n)
    df["prev_season_ppg"] = rng.normal(8, 3, n)
    return df


def key(df: pd.DataFrame) -> np.ndarray:
    return (df["season"] * 100 + df["week"]).to_numpy()


class TheMaskIsStrictlyEarlier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synth()

    def test_no_training_row_is_at_or_after_the_target_week_for_every_2025_week(self):
        for week in B.WEEKS:
            for spec in (T.PRIMARY, T.Spec("s", label_rows="stats_row"), T.Spec("d", exclude_depth_only=True)):
                tr, te = B.train_mask(self.df, 2025, week, spec), B.test_mask(self.df, 2025, week)
                B.assert_walk_forward(self.df, tr, te, 2025, week)
                self.assertLess(key(self.df)[tr].max(), 2025 * 100 + week)
                self.assertTrue((self.df.loc[te, "season"] == 2025).all() and (self.df.loc[te, "week"] == week).all())

    def test_it_takes_all_earlier_played_rows_and_nothing_else(self):
        d = self.df
        for week in (1, 2, 9, 18):
            tr = B.train_mask(d, 2025, week)
            want = (key(d) < 2025 * 100 + week) & (d["y_played"] == 1).to_numpy() & d["y_points_ppr"].notna().to_numpy()
            np.testing.assert_array_equal(tr, want)
        self.assertEqual(int(B.train_mask(d, 2025, 1).sum()), int(((d["season"] < 2025) & (d["y_played"] == 1)).sum()))
        # 2026 rows can never train a 2025 week, whatever the week
        self.assertFalse(B.train_mask(d, 2025, 18)[(d["season"] == 2026).to_numpy()].any())
        # the target week itself and later weeks are out even though they are labelled
        self.assertFalse(B.train_mask(d, 2025, 5)[((d["season"] == 2025) & (d["week"] >= 5)).to_numpy()].any())

    def test_label_choice_and_depth_only_filters(self):
        d = self.df
        tr_played = B.train_mask(d, 2025, 10)
        tr_stats = B.train_mask(d, 2025, 10, T.Spec("s", label_rows="stats_row"))
        self.assertTrue((tr_stats <= tr_played).all() and tr_stats.sum() < tr_played.sum())
        self.assertTrue((d.loc[tr_stats, "y_has_stats_row"] == 1).all())
        nd = B.train_mask(d, 2025, 10, T.Spec("d", exclude_depth_only=True))
        self.assertTrue((d.loc[nd, "spine_depth_only"] == 0).all())

    def test_the_runtime_assertion_catches_lookahead_and_bad_test_sets(self):
        d = self.df
        te = B.test_mask(d, 2025, 5)
        for bad_key in (2025 * 100 + 5, 2025 * 100 + 6, 2026 * 100 + 1):        # target week, later week, next season
            tr = B.train_mask(d, 2025, 5) | (key(d) == bad_key)
            with self.assertRaises(AssertionError):
                B.assert_walk_forward(d, tr, te, 2025, 5)
        with self.assertRaises(AssertionError):
            B.assert_walk_forward(d, B.train_mask(d, 2025, 5), B.test_mask(d, 2025, 6), 2025, 5)   # test rows are the wrong week
        with self.assertRaises(AssertionError):
            B.assert_walk_forward(d, np.zeros(len(d), bool), te, 2025, 5)                           # nothing to train on


class GarbageInTheFutureChangesNothing(unittest.TestCase):
    """The perturbation proof, for every library: every label at or after the target week and every feature of every
    row after it is replaced with garbage; the target week's predictions must be bit-identical. A control shows the
    same test bites: garbage in the PREVIOUS week's labels does move the prediction."""

    @classmethod
    def setUpClass(cls):
        cls.df = synth()

    def dirty(self, week: int) -> pd.DataFrame:
        d = self.df.copy()
        k = key(d)
        ge, gt = k >= 2025 * 100 + week, k > 2025 * 100 + week
        for c in ("y_points_ppr", "base_xfp_sameweek"):
            d.loc[ge, c] = 987654.0
        d.loc[ge, "y_played"] = 1
        d.loc[ge, "y_has_stats_row"] = 1
        for c in ("pts_ppr_std_mean", "opps_l1", "team_spread", "dvp_ppr_l4", "wx_temp_obs", "is_home"):
            d.loc[gt, c] = -555.0
        d.loc[gt, "week"] = 99                    # even the calendar column of later rows
        return d

    def test_predictions_are_identical_for_every_library(self):
        for lib in T.LIBS:
            for week in (1, 3, 12):
                clean = B.walk_forward(self.df, T.PRIMARY, lib, TINY[lib], weeks=[week])
                dirty = B.walk_forward(self.dirty(week), T.PRIMARY, lib, TINY[lib], weeks=[week])
                np.testing.assert_array_equal(clean["pred"].to_numpy(), dirty["pred"].to_numpy(), f"{lib} wk{week}")

    def test_control_earlier_labels_do_move_the_prediction(self):
        d = self.df.copy()
        prev = (d["season"] == 2025) & (d["week"] == 11)
        d.loc[prev, "y_points_ppr"] = 500.0
        a = B.walk_forward(self.df, T.PRIMARY, "lightgbm", TINY["lightgbm"], weeks=[12])["pred"].to_numpy()
        b = B.walk_forward(d, T.PRIMARY, "lightgbm", TINY["lightgbm"], weeks=[12])["pred"].to_numpy()
        self.assertFalse(np.array_equal(a, b))

    def test_quantile_models_obey_the_same_rule(self):
        d = self.dirty(6)
        for a in B.QUANTILES:
            x = B.walk_forward(self.df, T.PRIMARY, "lightgbm", TINY["lightgbm"], weeks=[6], quantile=a)["pred"].to_numpy()
            y = B.walk_forward(d, T.PRIMARY, "lightgbm", TINY["lightgbm"], weeks=[6], quantile=a)["pred"].to_numpy()
            np.testing.assert_array_equal(x, y)


class StaleModelsNeverSeeTheTargetWeek(unittest.TestCase):
    """refit_every=k: weeks between refits are predicted by an older model, which is never MORE informed than the weekly one."""

    def test_fit_week_is_never_after_the_target_week_and_follows_the_schedule(self):
        df = synth()
        r = B.walk_forward(df, T.PRIMARY, "lightgbm", TINY["lightgbm"], weeks=range(1, 10), refit_every=4)
        self.assertTrue((r["fit_week"] <= r["week"]).all())
        self.assertEqual(sorted(set(zip(r["week"], r["fit_week"]))),
                         [(1, 1), (2, 1), (3, 1), (4, 1), (5, 5), (6, 5), (7, 5), (8, 5), (9, 9)])
        weekly = B.walk_forward(df, T.PRIMARY, "lightgbm", TINY["lightgbm"], weeks=range(1, 10))
        same = r["week"].isin([1, 5, 9]).to_numpy()
        np.testing.assert_array_equal(r["pred"].to_numpy()[same], weekly["pred"].to_numpy()[same])   # a fit week IS the weekly fit
        self.assertFalse(np.array_equal(r["pred"].to_numpy()[~same], weekly["pred"].to_numpy()[~same]))

    def test_labels_from_the_target_weeks_cannot_reach_a_stale_model(self):
        df = synth()
        d = df.copy()
        ge = key(d) >= 2025 * 100 + 1
        d.loc[ge, ["y_points_ppr", "base_xfp_sameweek"]] = 987654.0        # every 2025 label garbage; features untouched
        a = B.walk_forward(df, T.PRIMARY, "lightgbm", TINY["lightgbm"], weeks=[1, 2, 3, 4], refit_every=99)
        b = B.walk_forward(d, T.PRIMARY, "lightgbm", TINY["lightgbm"], weeks=[1, 2, 3, 4], refit_every=99)
        np.testing.assert_array_equal(a["pred"].to_numpy(), b["pred"].to_numpy())
        self.assertEqual(set(a["fit_week"]), {1})


class Reproducibility(unittest.TestCase):
    def test_a_rerun_reproduces_every_prediction_for_all_three_libraries_and_the_quantile_models(self):
        df = synth(1)
        for lib in T.LIBS:
            a = B.walk_forward(df, T.PRIMARY, lib, TINY[lib], weeks=[4, 5])
            b = B.walk_forward(df, T.PRIMARY, lib, TINY[lib], weeks=[4, 5])
            np.testing.assert_array_equal(a["pred"].to_numpy(), b["pred"].to_numpy(), lib)
            self.assertFalse(a["pred"].isna().any())
        q1 = B.walk_forward(df, T.PRIMARY, "lightgbm", TINY["lightgbm"], weeks=[4], quantile=0.9)
        q2 = B.walk_forward(df, T.PRIMARY, "lightgbm", TINY["lightgbm"], weeks=[4], quantile=0.9)
        np.testing.assert_array_equal(q1["pred"].to_numpy(), q2["pred"].to_numpy())

    def test_the_seed_is_real(self):
        df = synth(1)
        p = dict(TINY["lightgbm"], subsample=0.5, colsample_bytree=0.5)
        a = T.make_model("lightgbm", p, seed=1)
        b = T.make_model("lightgbm", p, seed=2)
        X, _ = T.encode(df, T.feature_columns(df))
        tr = df["y_played"] == 1
        a.fit(X[tr], df.loc[tr, T.TARGET])
        b.fit(X[tr], df.loc[tr, T.TARGET])
        self.assertFalse(np.array_equal(a.predict(X[:200]), b.predict(X[:200])))


class TheDesignMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synth()

    def test_primary_excludes_weather_season_labels_baselines_and_dead_columns(self):
        dead = T.dead_columns(self.df)
        self.assertIn("college_breakout_age", dead)
        cols = T.feature_columns(self.df, T.PRIMARY, dead)
        X, fam = T.encode(self.df, cols)
        self.assertFalse([c for c in X.columns if c.startswith("wx_")])
        self.assertNotIn("season", X.columns)
        self.assertIn("week", X.columns)
        self.assertNotIn("college_breakout_age", X.columns)
        self.assertFalse([c for c in X.columns if c.startswith(T.NOT_INPUTS)])
        self.assertEqual(sorted(c for c in X.columns if c.startswith("pos_")), [f"pos_{p}" for p in sorted(T.POSITIONS)])

    def test_ablation_specs_add_exactly_what_they_name(self):
        base = set(T.encode(self.df, T.feature_columns(self.df, T.PRIMARY, T.dead_columns(self.df)))[0].columns)
        wx = set(T.encode(self.df, T.feature_columns(self.df, T.Spec("w", weather=True), T.dead_columns(self.df)))[0].columns)
        self.assertEqual(wx - base, {"wx_temp_obs", "wx_roof_dome", "wx_roof_retractable"})
        se = set(T.encode(self.df, T.feature_columns(self.df, T.Spec("s", season=True), T.dead_columns(self.df)))[0].columns)
        self.assertEqual(se - base, {"season"})
        nd = set(T.encode(self.df, T.feature_columns(self.df, T.Spec("d", drop_families=("dvp",)), T.dead_columns(self.df)))[0].columns)
        self.assertEqual(base - nd, {"dvp_ppr_l4"})

    def test_injury_categoricals_collapse_to_ordinals_that_cannot_drift(self):
        df = pd.DataFrame({"position": ["QB"] * 6,
                           "inj_report_status": [None, "Questionable", "Doubtful", "Out", "Note", "Something New"],
                           "inj_practice_status": [None, "Full Participation in Practice", "Limited Participation in Practice",
                                                   "Did Not Participate In Practice", "", "x"]})
        X, _ = T.encode(df, ["inj_report_status", "inj_practice_status"])
        self.assertEqual(X["inj_report_ord"].tolist(), [0, 1, 2, 2, 0, 0])
        self.assertEqual(X["inj_practice_ord"].tolist(), [0, 1, 2, 3, 0, 0])

    def test_a_label_or_baseline_column_can_never_be_an_input(self):
        df = self.df.assign(y_points_ppr_lag=1.0)
        with self.assertRaises(ValueError):
            spec = T.Spec("x")
            orig = F.FAMILIES["lags"]
            try:
                F.FAMILIES["lags"] = orig + ["base_trail3_ppr"]
                T.feature_columns(df, spec)
            finally:
                F.FAMILIES["lags"] = orig


class Metrics(unittest.TestCase):
    def frame(self, y, p, pos="WR", week=1, trail=None):
        n = len(y)
        return pd.DataFrame({"position": pos, "week": week, "y": np.asarray(y, float), "p": np.asarray(p, float),
                             "base_trail3_ppr": np.asarray(trail if trail is not None else y, float)})

    def test_pair_accuracy_and_spearman_on_known_answers(self):
        y = [1, 2, 3, 4]
        perfect = B.weekly_stats(self.frame(y, [10, 20, 30, 40]), "p").iloc[0]
        self.assertEqual((perfect["spearman"], perfect["credit"], perfect["pairs"]), (1.0, 6.0, 6))
        rev = B.weekly_stats(self.frame(y, [4, 3, 2, 1]), "p").iloc[0]
        self.assertEqual((rev["spearman"], rev["credit"]), (-1.0, 0.0))
        tie = B.weekly_stats(self.frame(y, [5, 5, 5, 5]), "p").iloc[0]
        self.assertEqual(tie["credit"], 3.0)                                  # a tied prediction earns half a point per pair
        self.assertTrue(np.isnan(tie["spearman"]))
        # pairs with equal actuals are not pairs
        eq = B.weekly_stats(self.frame([1, 1, 3, 3], [1, 2, 3, 4]), "p").iloc[0]
        self.assertEqual(eq["pairs"], 4)
        # one wrong pair out of six
        one = B.weekly_stats(self.frame(y, [10, 30, 20, 40]), "p").iloc[0]
        self.assertEqual(one["credit"], 5.0)

    def test_pairs_never_cross_positions_or_weeks(self):
        a = self.frame([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], pos="WR", week=1)
        b = self.frame([8, 7, 6, 5, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6, 7, 8], pos="RB", week=1)
        c = self.frame([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], pos="WR", week=2)
        ws = B.weekly_stats(pd.concat([a, b, c]), "p")
        self.assertEqual(len(ws), 3)
        self.assertEqual(int(ws["pairs"].sum()), 3 * 28)
        s = B.summarize(ws)
        self.assertEqual(s.loc["WR", "pick_acc"], 1.0)
        self.assertEqual(s.loc["RB", "pick_acc"], 0.0)

    def test_mae_rmse_and_the_startable_subset(self):
        f = self.frame([0, 10, 20, 30], [1, 8, 25, 30], trail=[0, 10, 20, 30])
        s = B.summarize(B.weekly_stats(f, "p"))
        self.assertAlmostEqual(s.loc["WR", "MAE"], (1 + 2 + 5 + 0) / 4)
        self.assertAlmostEqual(s.loc["WR", "RMSE"], np.sqrt((1 + 4 + 25 + 0) / 4))
        top = B.STARTABLE_TOP["WR"]
        n = top + 5                                                            # 5 players below the startable cut
        y = np.arange(n, dtype=float)
        g = self.frame(y, -y, trail=y)                                         # a predictor that is exactly backwards
        ws = B.weekly_stats(g, "p").iloc[0]
        self.assertEqual(ws["pairs_start"], top * (top - 1) // 2)              # only pairs inside the top-K by trailing average
        self.assertEqual(ws["credit_start"], 0.0)

    def test_bootstrap_of_identical_predictors_is_zero_and_of_a_clear_winner_excludes_zero(self):
        rng = np.random.default_rng(3)
        rows = []
        for wk in range(1, 19):
            y = rng.normal(8, 5, 60)
            rows.append(pd.DataFrame({"position": "WR", "week": wk, "y": y, "good": y + rng.normal(0, 1, 60),
                                      "bad": y + rng.normal(0, 6, 60), "same": y + 0.0, "base_trail3_ppr": y}))
        e = pd.concat(rows)
        z = B.paired_bootstrap(e, "good", "good", draws=200)
        self.assertEqual(list(z.index), ["WR", "ALL"])
        self.assertTrue((z[["dMAE", "dRMSE", "dSpearman", "dPick"]].abs() < 1e-12).all().all())
        r = B.paired_bootstrap(e, "good", "bad", draws=400)
        self.assertLess(r.loc["WR", "dMAE_hi"], 0)
        self.assertLess(r.loc["WR", "dRMSE_hi"], 0)
        self.assertGreater(r.loc["WR", "dSpearman_lo"], 0)
        again = B.paired_bootstrap(e, "good", "bad", draws=400)
        pd.testing.assert_frame_equal(r, again)                                  # seeded

    def test_quantile_coverage_crossing_and_sorting(self):
        y = np.arange(100, dtype=float)
        q = pd.DataFrame({"position": "QB", "y": y, "q10": y + 10, "q50": y, "q90": y - 10})       # fully crossed
        q = B.sort_quantiles(q)
        self.assertEqual(q["crossed"].mean(), 1.0)
        self.assertTrue((q["q10"] <= q["q50"]).all() and (q["q50"] <= q["q90"]).all())
        cal = B.quantile_calibration(q)
        self.assertEqual(cal.loc["QB", "below_q50"], 1.0)                        # sorted q50 == y: at-or-below is always true
        ok = pd.DataFrame({"position": "QB", "y": y, "q10": np.percentile(y, 10) + 0 * y, "q50": np.percentile(y, 50) + 0 * y,
                           "q90": np.percentile(y, 90) + 0 * y})
        cal = B.quantile_calibration(ok)
        self.assertAlmostEqual(cal.loc["QB", "below_q10"], 0.10, delta=0.02)
        self.assertAlmostEqual(cal.loc["QB", "below_q90"], 0.90, delta=0.02)
        self.assertAlmostEqual(cal.loc["QB", "coverage_10_90"], 0.80, delta=0.03)


@unittest.skipUnless(T.MATRIX.exists(), "the feature matrix has not been built (model.build_features)")
class TheRealMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = T.load_matrix()

    def test_every_2025_week_trains_only_on_earlier_rows_in_the_real_matrix(self):
        d = self.df
        for week in B.WEEKS:
            tr, te = B.train_mask(d, 2025, week), B.test_mask(d, 2025, week)
            B.assert_walk_forward(d, tr, te, 2025, week)
            self.assertEqual(int(d.loc[tr, "season"].min()), 2021)
            self.assertFalse((d.loc[tr, "season"] > 2025).any())
        self.assertEqual(int(B.train_mask(d, 2025, 1).sum()), int(((d["season"] <= 2024) & (d["y_played"] == 1)).sum()))

    def test_the_primary_inputs_in_the_real_matrix(self):
        d = self.df
        dead = T.dead_columns(d)
        self.assertEqual(dead, ("college_breakout_age",))             # the only column with nothing in it
        X, fam = T.encode(d, T.feature_columns(d, T.PRIMARY, dead))
        self.assertFalse([c for c in X.columns if c.startswith(("wx_", "y_", "base_", "spine_"))])
        self.assertNotIn("season", X.columns)
        self.assertEqual(set(fam.values()) - {"position"}, set(F.FAMILIES) - {"weather"} - set(T.OPT_IN_FAMILIES))


if __name__ == "__main__":
    unittest.main()
