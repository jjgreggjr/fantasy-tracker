"""Phase 2.5: the modelling code around the experiments keeps the same discipline as Phase 2.

    model/.venv/bin/python -m unittest model.tests.test_phase25_models

Synthetic frames (fast, no downloads) for the walk-forward mechanics, the two-stage honest series, the ensemble weights and the
quantile recalibration; the composition identities run against the real matrix and league file (skipped only if the git-ignored
matrix has not been built). What is proven:

  * stage-one predictions are honest: garbage in every 2025 label from week 9 on changes no stage-one value for 2022-2024 or for
    2025 weeks <= 9 (bit-identical) and does change later weeks (control), so stage two never trains on a fit that saw its week
  * the ensemble weights and the recalibration shifts use only rows before the week they are applied to
  * composed points equal nflverse PPR from the actual components, and a TE premium adds the league's bonus to TE catches only
  * the adoption rules, the volume-quality table and the coverage table give the hand-computed answers
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from model import backtest as B
from model import components as C
from model import ensemble as E
from model import p25run as P
from model import train as T
from model import phase25 as PH
from model import report_phase25 as R
from model import volume as V
from model.tests.test_backtest import TINY, synth

SLUG = "we-can-think-of-something-funny"
TINY_DOC = {"point": {"lightgbm": {**TINY["lightgbm"]}}, "quantile": {}, "meta": {}}


def synth25(seed: int = 0) -> pd.DataFrame:
    """The Phase 2 synthetic matrix plus volume labels/lag columns and the misc label."""
    df = synth(seed)
    rng = np.random.default_rng(seed + 1)
    n = len(df)
    df["targets_l1"], df["targets_l2"], df["targets_l3"] = (rng.poisson(5, n).astype(float) for _ in range(3))
    df["carries_l1"], df["carries_l2"], df["carries_l3"] = (rng.poisson(6, n).astype(float) for _ in range(3))
    df["pass_att_l1"], df["pass_att_l2"], df["pass_att_l3"] = (rng.poisson(12, n).astype(float) for _ in range(3))
    df["targets_std_mean"] = df[["targets_l1", "targets_l2", "targets_l3"]].mean(axis=1)
    df["carries_std_mean"] = df[["carries_l1", "carries_l2", "carries_l3"]].mean(axis=1)
    played = df["y_played"] == 1
    for lab, base in (("y_tgt", "targets_std_mean"), ("y_car", "carries_std_mean"), ("y_att", "pass_att_l1")):
        df[lab] = np.where(played, 0.7 * df[base] + 0.2 * df["opps_l1"] + rng.normal(0, 1.5, n), np.nan)
    return df


class Params:
    """Patch the tuned-parameter file and the cache directory: tiny models, nothing written under model/cache."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [mock.patch.object(P, "OUT", Path(self.tmp.name)),
                        mock.patch.object(T, "load_params", return_value=TINY_DOC)]
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *a):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()


# ------------------------------------------------------------------ walk-forward generalisation
class WalkForwardTargetsAndSpecs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synth25()

    def test_the_mask_uses_the_named_target_and_min_season(self):
        d = self.df
        tr = B.train_mask(d, 2025, 10, T.PRIMARY, "y_tgt")
        self.assertTrue(d.loc[tr, "y_tgt"].notna().all())
        self.assertLess((d["season"] * 100 + d["week"])[tr].max(), 2025 * 100 + 10)
        late = B.train_mask(d, 2025, 10, T.Spec("late", min_season=2023), "y_tgt")
        self.assertTrue((d.loc[late, "season"] >= 2023).all())
        self.assertLess(late.sum(), tr.sum())
        self.assertEqual(int((B.train_mask(d, 2025, 10) != B.train_mask(d, 2025, 10, T.PRIMARY, T.TARGET)).sum()), 0)

    def test_only_cols_and_extra_cols_pick_exactly_the_named_columns(self):
        d = self.df.copy()
        d["s1_tgt"] = 1.0
        cols = T.feature_columns(d, T.Spec("o", only_cols=("opps_l1", "s1_tgt", "not_there")))
        self.assertEqual(cols, ["opps_l1", "s1_tgt"])
        plus = T.feature_columns(d, T.Spec("p", extra_cols=("s1_tgt",)))
        self.assertEqual(plus[-1], "s1_tgt")
        self.assertEqual(plus[:-1], T.feature_columns(d, T.PRIMARY))
        with self.assertRaises(ValueError):
            T.feature_columns(d.assign(y_bad=1.0), T.Spec("bad", extra_cols=("y_bad",)))
        with self.assertRaises(ValueError):
            T.feature_columns(d, T.Spec("bad", only_cols=("base_trail3_ppr",)))

    def test_garbage_in_the_target_labels_of_the_target_week_and_later_changes_no_prediction_of_that_week(self):
        d = self.df
        clean = B.walk_forward(d, T.PRIMARY, "lightgbm", TINY["lightgbm"], target="y_tgt", weeks=(9, 10))
        dirty_d = d.copy()
        late = ((d["season"] == 2025) & (d["week"] >= 9)).to_numpy()
        dirty_d.loc[late, "y_tgt"] = 987654.0
        dirty = B.walk_forward(dirty_d, T.PRIMARY, "lightgbm", TINY["lightgbm"], target="y_tgt", weeks=(9, 10))
        w9 = clean["week"] == 9
        np.testing.assert_array_equal(clean.loc[w9, "pred"].to_numpy(), dirty.loc[w9, "pred"].to_numpy())
        self.assertGreater(float(np.abs(clean.loc[~w9, "pred"].to_numpy() - dirty.loc[~w9, "pred"].to_numpy()).max()), 1.0,
                           "control: week 10 trains on week 9's (garbage) labels and must move")


# ------------------------------------------------------------------ stage one is honest
class HonestStageOne(unittest.TestCase):
    def test_garbage_in_2025_labels_from_week_nine_leaves_every_earlier_stage_one_value_bit_identical(self):
        d = synth25()
        with Params():
            clean = V.honest_series(d, "k1", seasons=(2024, 2025))
        dirty_d = d.copy()
        late = ((d["season"] == 2025) & (d["week"] >= 9)).to_numpy()
        for lab in V.S1:
            dirty_d.loc[late, lab] = 987654.0
        with Params():
            dirty = V.honest_series(dirty_d, "k2", seasons=(2024, 2025))
        early = ~((d["season"] == 2025) & (d["week"] >= 10)).to_numpy() & (d["season"] >= 2024).to_numpy()
        for col in V.S1_COLS:
            np.testing.assert_array_equal(clean.loc[early, col].to_numpy(), dirty.loc[early, col].to_numpy(), err_msg=col)
            later = ((d["season"] == 2025) & (d["week"] >= 10)).to_numpy()
            self.assertGreater(float(np.nanmax(np.abs(clean.loc[later, col].to_numpy() - dirty.loc[later, col].to_numpy()))), 1.0, col)
        self.assertTrue(clean.loc[(d["season"] < 2024).to_numpy()].isna().all().all(), "seasons outside the request stay NaN")
        self.assertTrue(clean.loc[(d["season"] >= 2024).to_numpy() & (d["season"] < 2026).to_numpy()].notna().all().all())

    def test_stage_two_never_trains_on_the_2021_rows_that_have_no_honest_prediction(self):
        spec = V.SPECS["two_stage"]
        self.assertEqual(spec.min_season, 2022)
        d = synth25()
        m = B.train_mask(d, 2025, 5, spec)
        self.assertTrue((d.loc[m, "season"] >= 2022).all())
        self.assertEqual(set(spec.only_cols), set(V.S1_COLS) | set(V.EFF_COLS))

    def test_the_trailing_baseline_is_the_mean_of_the_last_three_appearances(self):
        d = pd.DataFrame({"targets_l1": [3.0, np.nan, 1.0], "targets_l2": [5.0, np.nan, np.nan], "targets_l3": [7.0, np.nan, np.nan]})
        np.testing.assert_allclose(V.trailing_baseline(d, "y_tgt").to_numpy(), [5.0, np.nan, 1.0])

    def test_volume_quality_matches_a_hand_computation(self):
        weeks = np.repeat([1, 2], 4)
        e = pd.DataFrame({"week": weeks, "position": ["WR"] * 8, "y": [2, 4, 6, 8, 1, 3, 5, 7.0],
                          "model": [3, 4, 5, 8, 1, 2, 5, 9.0], "trail3": [2, 6, 6, 5, 4, 3, 5, 7.0]})
        e["std_mean"] = e["trail3"]
        q = V.volume_quality(e, "y_tgt", draws=200)
        self.assertEqual(int(q.loc["WR", "n"]), 8)
        self.assertAlmostEqual(q.loc["WR", "MAE_model"], (1 + 0 + 1 + 0 + 0 + 1 + 0 + 2) / 8)
        self.assertAlmostEqual(q.loc["WR", "MAE_trail3"], (0 + 2 + 0 + 3 + 3 + 0 + 0 + 0) / 8)
        self.assertAlmostEqual(q.loc["WR", "dMAE"], q.loc["WR", "MAE_model"] - q.loc["WR", "MAE_trail3"])
        self.assertLessEqual(q.loc["WR", "dMAE_lo"], q.loc["WR", "dMAE"] + 1e-12)
        self.assertGreaterEqual(q.loc["WR", "dMAE_hi"], q.loc["WR", "dMAE"] - 1e-12)
        self.assertEqual(list(q.index), ["WR", "relevant"])      # only positions where targets are real volume, then the pool
        again = V.volume_quality(e, "y_tgt", draws=200)
        pd.testing.assert_frame_equal(q, again)                        # fixed seed: a rerun reproduces the interval


# ------------------------------------------------------------------ composition
class Composition(unittest.TestCase):
    def test_compose_is_the_weighted_sum_and_refuses_missing_components(self):
        pos = pd.Series(["QB", "TE", "WR"])
        preds = {c: pd.Series([1.0, 2.0, 3.0]) for c in C.PPR.needs()}
        got = C.compose(preds, pos, C.PPR)
        self.assertAlmostEqual(got[1], 2.0 * sum(C.PPR_WEIGHTS.values()))
        del preds["y_rec"]
        with self.assertRaises(KeyError):
            C.compose(preds, pos, C.PPR)

    def test_a_te_premium_adds_the_bonus_to_tight_end_catches_only(self):
        pos = pd.Series(["QB", "TE", "WR", "TE"])
        preds = {c: pd.Series([0.0, 0.0, 0.0, 0.0]) for c in C.PPR.needs()}
        preds["y_rec"] = pd.Series([4.0, 4.0, 4.0, 6.0])
        te = C.Scoring("te", C.PPR_WEIGHTS, {("TE", "y_rec"): 0.5})
        base, prem = C.compose(preds, pos, C.PPR), C.compose(preds, pos, te)
        np.testing.assert_allclose((prem - base).to_numpy(), [0.0, 2.0, 0.0, 3.0])
        half = C.Scoring("half", {**C.PPR_WEIGHTS, "y_rec": 0.5})
        np.testing.assert_allclose(C.compose(preds, pos, half).to_numpy(), (base - 0.5 * preds["y_rec"]).to_numpy())

    def test_the_league_scoring_is_read_from_the_league_file_and_its_divergences_are_named(self):
        sc, diverge = C.league_scoring(SLUG)
        self.assertEqual(sc.position_bonus, {("TE", "y_rec"): 0.5})
        self.assertEqual(sc.weights, C.PPR_WEIGHTS)
        text = " ".join(diverge)
        self.assertIn("pass_int: league -1.0 vs PPR -2.0", text)
        self.assertIn("bonus_rec_yd_100", text)
        self.assertIn("rec_td_40p", text)
        raw = json.loads((C.REPO / "leagues" / SLUG / "league.json").read_text())["scoring"]
        self.assertEqual((raw["rec"], raw["bonus_rec_te"]), (1.0, 0.5))          # the file the alternate scoring stands on

    def test_a_league_that_is_not_full_ppr_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "leagues" / "x").mkdir(parents=True)
            (Path(d) / "leagues" / "x" / "league.json").write_text(json.dumps({"scoring": {"rec": 0.5, "bonus_rec_te": 0.5}}))
            with self.assertRaises(ValueError):
                C.league_scoring("x", root=Path(d))

    def test_the_misc_label_is_two_point_and_special_teams_points_and_nan_for_a_dnp(self):
        d = pd.DataFrame({"y_two_pt": [1.0, 0.0, np.nan], "y_st_td": [0.0, 1.0, np.nan]})
        np.testing.assert_allclose(C.add_misc(d)[C.MISC].to_numpy(), [2.0, 6.0, np.nan])

    def test_actual_components_rebuild_nflverse_ppr_on_the_real_matrix(self):
        if not T.MATRIX.exists():
            self.skipTest("model/cache/features.parquet has not been built")
        cols = ["position", "y_played", "y_points_ppr", *[c for c in C.TARGETS if c != C.MISC], "y_two_pt", "y_st_td"]
        try:
            df = pd.read_parquet(T.MATRIX, columns=cols)
        except Exception:
            self.skipTest("the matrix predates the Phase 2.5 labels: rebuild it")
        df = C.add_misc(df)
        played = df[df["y_played"] == 1]
        got = C.actual_points(df, played.index, C.PPR)
        self.assertLess(float((got - played["y_points_ppr"]).abs().max()), 1e-9)
        self.assertGreater(len(played), 30000)
        sc, _ = C.league_scoring(SLUG)
        te = C.actual_points(df, played.index, sc)
        diff = (te - played["y_points_ppr"])
        np.testing.assert_allclose(diff[played["position"] != "TE"].to_numpy(), 0.0, atol=1e-9)
        np.testing.assert_allclose(diff[played["position"] == "TE"].to_numpy(), 0.5 * played.loc[played["position"] == "TE", "y_rec"].to_numpy(), atol=1e-9)


# ------------------------------------------------------------------ ensemble
class Ensemble(unittest.TestCase):
    def test_weights_sit_on_the_simplex_and_follow_the_better_model(self):
        rng = np.random.default_rng(0)
        y = pd.Series(rng.normal(10, 4, 2000))
        preds = pd.DataFrame({"a": y + rng.normal(0, 1, 2000), "b": y + rng.normal(0, 3, 2000), "c": y + rng.normal(0, 3, 2000)})
        w = E.fit_weights(preds, y)
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertTrue(all(v >= 0 for v in w.values()))
        self.assertGreater(w["a"], 0.6)
        self.assertLess(float(np.sqrt(((E.blend(preds, w) - y) ** 2).mean())), float(np.sqrt(((preds["a"] - y) ** 2).mean())) + 1e-12)

    def test_identical_models_give_the_first_grid_point_and_a_perfect_one_gets_all_the_weight(self):
        y = pd.Series([1.0, 2.0, 3.0, 4.0])
        w = E.fit_weights(pd.DataFrame({"a": y, "b": y + 5, "c": y + 7}), y)
        self.assertEqual(w, {"a": 1.0, "b": 0.0, "c": 0.0})
        same = E.fit_weights(pd.DataFrame({"a": y, "b": y, "c": y}), y)
        self.assertAlmostEqual(sum(same.values()), 1.0)
        with self.assertRaises(ValueError):
            E.fit_weights(pd.DataFrame({"a": y, "b": y}), y)

    def test_weights_fit_on_one_year_are_scored_on_the_next_year_only(self):
        """The structure of experiment 4: nothing about the scored year enters the fit (asserted by construction: the fit
        function sees only the frame it is given), and the same weights applied to another frame blend it linearly."""
        rng = np.random.default_rng(3)
        fit_y = pd.Series(rng.normal(size=300))
        fit_p = pd.DataFrame({k: fit_y + rng.normal(0, s, 300) for k, s in (("a", 1), ("b", 2), ("c", 2))})
        w = E.fit_weights(fit_p, fit_y)
        other = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
        np.testing.assert_allclose(E.blend(other, w).to_numpy(), [w["a"] + 3 * w["b"] + 5 * w["c"], 2 * w["a"] + 4 * w["b"] + 6 * w["c"]])


# ------------------------------------------------------------------ quantile recalibration
def q_frame(seed: int = 0, weeks: int = 18, per_week: int = 120) -> pd.DataFrame:
    """QB rows whose band is too narrow (noise sd 6 against a band built for sd 3) and RB rows that are calibrated."""
    rng = np.random.default_rng(seed)
    rows = []
    for wk in range(1, weeks + 1):
        for i in range(per_week):
            pos = "QB" if i % 2 == 0 else "RB"
            sd_true, sd_band = (6.0, 3.0) if pos == "QB" else (3.0, 3.0)
            mu = rng.normal(12, 3)
            y = mu + rng.normal(0, sd_true)
            rows.append((wk, pos, y, mu - 1.2816 * sd_band, mu, mu + 1.2816 * sd_band, 1))
    return pd.DataFrame(rows, columns=["week", "position", "y", "q10", "q50", "q90", "y_played"])


class Recalibration(unittest.TestCase):
    def test_a_narrow_position_is_widened_towards_the_target_and_a_calibrated_one_is_left_exactly_alone(self):
        hist, cur = q_frame(1), q_frame(2)
        out = E.recalibrate(cur, hist)
        qb, rb = out[out["position"] == "QB"], out[out["position"] == "RB"]
        before = E.coverage(qb["y"], qb["q10"], qb["q90"])
        after = E.coverage(qb["y"], qb["q10a"], qb["q90a"])
        self.assertLess(before, 0.62)
        self.assertGreater(after, 0.76)
        self.assertLess(abs(after - 0.80), abs(before - 0.80) / 2)
        np.testing.assert_array_equal(rb["q10a"].to_numpy(), rb["q10"].to_numpy())
        np.testing.assert_array_equal(rb["q90a"].to_numpy(), rb["q90"].to_numpy())
        all_mode = E.recalibrate(cur, hist, mode="all")
        self.assertGreater(float((all_mode.loc[all_mode["position"] == "RB", "q90a"] != all_mode.loc[all_mode["position"] == "RB", "q90"]).mean()), 0.9)

    def test_the_positions_argument_confines_the_shift_to_the_named_positions(self):
        hist, cur = q_frame(1), q_frame(2)
        both = E.recalibrate(cur, hist, mode="all")
        qb_only = E.recalibrate(cur, hist, mode="all", positions=("QB",))
        rb = cur["position"] == "RB"
        np.testing.assert_array_equal(qb_only.loc[rb, "q90a"].to_numpy(), cur.loc[rb, "q90"].to_numpy())
        self.assertGreater(float((both.loc[rb, "q90a"] != cur.loc[rb, "q90"]).mean()), 0.9)
        qb = ~rb
        np.testing.assert_array_equal(qb_only.loc[qb, "q90a"].to_numpy(), both.loc[qb, "q90a"].to_numpy())

    def test_week_n_is_adjusted_from_rows_before_it_only(self):
        hist, cur = q_frame(1), q_frame(2)
        clean = E.recalibrate(cur, hist)
        dirty = cur.copy()
        dirty.loc[dirty["week"] >= 3, "y"] = 987654.0
        d = E.recalibrate(dirty, hist)
        for col in ("q10a", "q90a"):
            m = (cur["week"] <= 3).to_numpy()
            np.testing.assert_array_equal(clean.loc[m, col].to_numpy(), d.loc[m, col].to_numpy(), err_msg="garbage in weeks >= 3 changed week <= 3")
        self.assertGreater(float((clean.loc[clean["week"] == 8, "q90a"] - d.loc[d["week"] == 8, "q90a"]).abs().max()), 100.0,
                           "control: week 8 does read the earlier (garbage) residuals")

    def test_history_only_counts_played_rows_and_a_short_history_changes_nothing(self):
        hist, cur = q_frame(1), q_frame(2)
        unplayed = hist.assign(y_played=0)
        out = E.recalibrate(cur.loc[cur["week"] == 1], unplayed)
        np.testing.assert_array_equal(out["q90a"].to_numpy(), out["q90"].to_numpy())
        few = E.recalibrate(cur.loc[cur["week"] == 1], hist.iloc[:10])
        np.testing.assert_array_equal(few["q10a"].to_numpy(), few["q10"].to_numpy())
        with self.assertRaises(ValueError):
            E.recalibrate(cur, hist, mode="bogus")

    def test_coverage_table_and_interval_score_match_hand_values(self):
        e = pd.DataFrame({"week": [1, 1, 2, 2], "position": ["QB", "QB", "RB", "RB"], "y": [5.0, 20.0, 1.0, 3.0],
                          "lo": [4.0, 4.0, 2.0, 2.0], "hi": [10.0, 10.0, 6.0, 6.0]})
        t = E.coverage_table(e, "lo", "hi", draws=100)
        self.assertEqual(t.loc["QB", "coverage"], 0.5)
        self.assertEqual(t.loc["RB", "coverage"], 0.5)
        self.assertEqual(t.loc["ALL", "coverage"], 0.5)
        self.assertEqual(t.loc["QB", "above_hi"], 0.5)
        self.assertEqual(t.loc["RB", "below_lo"], 0.5)
        self.assertAlmostEqual(t.loc["QB", "mean_width"], 6.0)
        self.assertAlmostEqual(E.interval_score(e, "lo", "hi"), np.mean([6, 6 + 10 * 10, 4 + 10 * 1, 4]))
        pd.testing.assert_frame_equal(t, E.coverage_table(e, "lo", "hi", draws=100))


# ------------------------------------------------------------------ comparison machinery and the rules
class CompareAndVerdicts(unittest.TestCase):
    def runs(self):
        d = synth25()
        with Params():
            a = B.walk_forward(d, T.PRIMARY, "lightgbm", TINY["lightgbm"], weeks=(3, 4, 5, 6))
        b = a.copy()
        b["pred"] = a["pred"] + 5.0
        return {"base": a, "same": a.copy(), "shifted": b}

    def test_identical_runs_have_zero_deltas_and_a_shifted_run_is_worse_by_the_shift(self):
        r = P.compare(self.runs(), "base", draws=200)
        self.assertEqual(float(r["delta"].loc["same", "dRMSE"]), 0.0)
        self.assertGreater(float(r["delta"].loc["shifted", "dRMSE"]), 0.0)
        self.assertGreater(float(r["delta"].loc["shifted", "dMAE"]), 0.0)
        self.assertEqual(list(r["scores"].index), ["base", "same", "shifted"])
        self.assertEqual(list(r["scores"].columns), list(P.KIND))
        r2 = P.compare(self.runs(), "base", draws=200)
        pd.testing.assert_frame_equal(r["delta"], r2["delta"])

    def test_a_run_that_does_not_cover_the_scored_rows_is_an_error_not_a_silent_subset(self):
        runs = self.runs()
        runs["short"] = runs["base"].iloc[:50]
        with self.assertRaises(ValueError):
            P.compare(runs, "base", draws=50)

    def test_the_adoption_rule(self):
        row = lambda d, lo, hi: pd.Series({"dRMSE": d, "dRMSE_lo": lo, "dRMSE_hi": hi})     # noqa: E731
        self.assertTrue(P.verdict_rmse(row(-0.05, -0.1, -0.01), row(-0.02, -0.06, 0.02)).startswith("SHIPS (significant"))
        self.assertIn("spans zero", P.verdict_rmse(row(-0.05, -0.1, 0.01), row(-0.02, -0.06, 0.02)))
        self.assertEqual(P.verdict_rmse(row(-0.05, -0.1, -0.01), row(+0.01, -0.03, 0.05)), "does not ship")
        self.assertEqual(P.verdict_rmse(row(+0.01, -0.03, 0.05), row(-0.02, -0.06, 0.02)), "does not ship")

    def test_the_component_model_rule_ships_ties_and_wins_and_refuses_only_a_significant_loss(self):
        row = lambda d, lo, hi: pd.Series({"dRMSE": d, "dRMSE_lo": lo, "dRMSE_hi": hi})     # noqa: E731
        self.assertEqual(P.verdict_tie(row(0.01, -0.03, 0.05), row(0.02, -0.02, 0.06)), "SHIPS (tie within noise)")
        self.assertEqual(P.verdict_tie(row(-0.05, -0.1, -0.01), row(-0.04, -0.09, -0.01)), "SHIPS (better in both years)")
        self.assertIn("2025", P.verdict_tie(row(0.08, 0.02, 0.14), row(0.0, -0.03, 0.03)))
        self.assertIn("2024", P.verdict_tie(row(0.0, -0.03, 0.03), row(0.08, 0.02, 0.14)))

    def test_cached_runs_are_keyed_and_a_changed_key_recomputes(self):
        with Params():
            calls = []
            f = lambda: (calls.append(1), pd.DataFrame({"a": [1.0]}))[1]        # noqa: E731
            P.cached("x", "k1", f)
            P.cached("x", "k1", f)
            self.assertEqual(len(calls), 1)
            P.cached("x", "k2", f)
            self.assertEqual(len(calls), 2)
            P.cached("x", "k2", f, force=True)
            self.assertEqual(len(calls), 3)

    def test_matrix_key_changes_with_any_value_column_or_row(self):
        d = synth25()
        k = P.matrix_key(d)
        self.assertEqual(k, P.matrix_key(d.copy()))
        e = d.copy()
        e.loc[3, "opps_l1"] += 1.0
        self.assertNotEqual(k, P.matrix_key(e))
        self.assertNotEqual(k, P.matrix_key(d.drop(columns=["dvp_ppr_l4"])))
        self.assertNotEqual(k, P.matrix_key(d.iloc[1:]))

    def test_tree_counts_use_2024_validation_only_and_are_cached_per_target(self):
        d = synth25()
        with Params():
            n1 = P.tree_params(d, "k", "y_tgt", T.PRIMARY)
            n2 = P.tree_params(d, "k", "y_tgt", T.PRIMARY)
            self.assertEqual(n1, n2)
            self.assertGreaterEqual(n1["n_estimators"], 50)
            book = json.loads((P.OUT / "tree_counts.json").read_text())
            self.assertEqual(len(book), 1)
            # 2025 labels cannot reach the tree count: garbage there changes nothing
            dirty = d.copy()
            dirty.loc[dirty["season"] == 2025, "y_tgt"] = 987654.0
            self.assertEqual(P.tree_params(dirty, "k2", "y_tgt", T.PRIMARY), n1)


# ------------------------------------------------------------------ runner and report helpers
class RunnerAndReportHelpers(unittest.TestCase):
    def test_trailing_receptions_reads_only_earlier_appearances_of_the_same_season(self):
        df = pd.DataFrame({"player_id": ["a"] * 6 + ["b"] * 2, "season": [2024] * 4 + [2025] * 2 + [2025] * 2,
                           "week": [1, 2, 3, 4, 1, 2, 1, 2], "y_played": [1, 1, 0, 1, 1, 1, 1, 1],
                           "y_rec": [4.0, 6.0, np.nan, 8.0, 10.0, 2.0, 1.0, 3.0]})
        r = PH.trailing_receptions(df)
        self.assertTrue(np.isnan(r.iloc[0]))                       # first game of a season: no history
        self.assertEqual(r.iloc[1], 4.0)                           # week 2 sees week 1 only
        self.assertTrue(np.isnan(r.iloc[2]))                       # a DNP row has no baseline (it is not scored)
        self.assertEqual(r.iloc[3], 5.0)                           # week 4 sees weeks 1-2 (week 3 was a DNP): mean(4, 6)
        self.assertTrue(np.isnan(r.iloc[4]))                       # a new season starts empty for the same player
        self.assertEqual(r.iloc[5], 10.0)
        self.assertEqual(r.iloc[7], 1.0)                           # another player is not mixed in
        longer = pd.DataFrame({"player_id": ["a"] * 5, "season": 2025, "week": range(1, 6), "y_played": 1,
                               "y_rec": [1.0, 2.0, 3.0, 4.0, 100.0]})
        self.assertEqual(PH.trailing_receptions(longer).iloc[4], 3.0)      # mean of the previous three (2, 3, 4), never the row itself

    def test_every_variant_has_its_own_name_and_only_opt_in_families(self):
        specs = [*PH.EXP1.values(), *V.SPECS.values(), *PH.COMP_SPECS.values()]
        names = [sp.name for sp in specs if sp is not T.PRIMARY]
        self.assertEqual(len(names), len(set(names)))
        for sp in specs:
            self.assertTrue(set(sp.add_families) <= set(T.OPT_IN_FAMILIES), sp.name)

    def test_report_helpers_format_intervals_and_keep_the_sign(self):
        self.assertEqual(R.pm(-0.008, -0.024, 0.007), "-0.008 [-0.024, +0.007]")
        self.assertEqual(R.pm(0.0084, 0.0035, 0.0139, "{:+.4f}"), "+0.0084 [+0.0035, +0.0139]")
        c = {"delta": pd.DataFrame({"dRMSE": [-0.5], "dRMSE_lo": [-0.7], "dRMSE_hi": [-0.1]}, index=["x"])}
        self.assertEqual(R.d(c, "x"), "-0.500 [-0.700, -0.100]")

    def test_the_summary_verdict_rows_are_built_from_the_two_years_of_deltas(self):
        mk = lambda d, lo, hi: pd.DataFrame({"dRMSE": [d], "dRMSE_lo": [lo], "dRMSE_hi": [hi]}, index=["v"])       # noqa: E731
        comp = {2025: {"delta": mk(-0.01, -0.02, 0.0)}, 2024: {"delta": mk(0.02, -0.01, 0.05)}}
        row = R.verdict_row("x", "v", comp)
        self.assertEqual(row["verdict"], "does not ship")
        self.assertIn("-0.010", row["2025 result"])
        self.assertEqual(R.verdict_row("x", "v", comp, rule="tie")["verdict"], "SHIPS (tie within noise)")


if __name__ == "__main__":
    unittest.main()
