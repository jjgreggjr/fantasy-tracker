"""The opt-in `streak` family (model/features.py `_streak_features`): hand-computed values on a synthetic history, the sign/cap
convention, the cross-season carry, NaN xFP handling, the registry contract (the frozen primary's inputs are untouched), and a
leakage check on a synthetic store (the target game's own row, stamped at kickoff, never enters; the result cutoff is pinned).

    python -m unittest model.tests.test_streak

No network, no real matrix: everything here is built by hand.
"""
from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from model import features, train as T, trial as TR
from model.features import NAN, N_VALUES, OPPS, PTS, XFP, Hist, RESULT_LAG, _history, _streak_features
from model.point_in_time import KNOWN_AT, RawStore, RawTable, TargetRow

COLS = features.FAMILIES["streak"]
KO = pd.Timestamp("2025-10-19 17:00", tz="UTC")           # the target kickoff (2025 week 7)


def row(gid: str, season: int, week: int, pts: float, xfp: float, opps: float, *, reg: bool = True) -> tuple:
    """One Hist row: identity, then the N_VALUES value columns with pts / xfp / opps set and everything else 0."""
    v = [0.0] * N_VALUES
    v[PTS], v[XFP], v[OPPS] = pts, xfp, opps
    return (gid, season, week, reg, True, "KC", *v)


def target(season: int = 2025, week: int = 7) -> TargetRow:
    return TargetRow("P1", season, week, "KC", "LV", "WR", f"{season}_{week:02d}_LV_KC", KO, True, ("usage",))


def hist(rows: list[tuple]) -> Hist:
    return Hist(rows)


def same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-12


# ------------------------------------------------------------------ hand-computed values
class HandComputed(unittest.TestCase):
    # this season: weeks 1-6, (pts, xfp, opps). Residuals: +2, -1, +3, +1, +2, +4 -> the last four are all positive.
    CUR = [row("g1", 2025, 1, 12.0, 10.0, 8.0), row("g2", 2025, 2, 9.0, 10.0, 6.0), row("g3", 2025, 3, 15.0, 12.0, 10.0),
           row("g4", 2025, 4, 11.0, 10.0, 7.0), row("g5", 2025, 5, 14.0, 12.0, 9.0), row("g6", 2025, 6, 20.0, 16.0, 12.0)]
    # last season (REG): ppg 10; the one POST game must not count
    PREV = [row("p1", 2024, 16, 8.0, 9.0, 5.0), row("p2", 2024, 17, 12.0, 11.0, 7.0), row("p3", 2024, 18, 10.0, 10.0, 6.0),
            row("p4", 2024, 19, 40.0, 30.0, 20.0, reg=False)]

    def test_every_column_against_the_definitions(self):
        f = _streak_features(hist(self.PREV + self.CUR), target())
        self.assertEqual(set(f), set(COLS))
        self.assertEqual((f["resid_xfp_l1"], f["resid_xfp_l2"], f["resid_xfp_l3"]), (4.0, 2.0, 1.0))
        self.assertEqual(f["over_xfp_streak"], 4.0)                         # +4, +2, +1, +3 then -1 ends it
        self.assertEqual(f["resid_xfp_sum_l4"], 4.0 + 2.0 + 1.0 + 3.0)
        # volume: opps minus the trailing-3 mean entering each game (the window starts with last season's REG games)
        #   p1 5 (no history: NaN) | p2 7 - 5 = +2 | p3 6 - 6 = 0 | g1 8 - 6 = +2 | g2 6 - 7 = -1 | g3 10 - 20/3 = +3.33
        #   g4 7 - 8 = -1 | g5 9 - 23/3 = +1.33 | g6 12 - 26/3 = +3.33
        self.assertTrue(same(f["resid_opps_l1"], 12.0 - 26.0 / 3))
        self.assertEqual(f["over_trail_streak"], 2.0)                       # g6 +, g5 +, g4 - ends it
        self.assertTrue(same(f["pts_vs_prev_season_l2"], (14.0 + 20.0) / 2 - 10.0))

    def test_a_negative_run_is_a_negative_count_and_the_cap_is_six(self):
        under = [row(f"u{i}", 2025, i, 5.0, 10.0, 4.0) for i in range(1, 9)]           # eight straight under xFP, flat volume
        f = _streak_features(hist(under), target(week=9))
        self.assertEqual(f["over_xfp_streak"], -6.0)
        self.assertEqual(f["resid_xfp_sum_l4"], -20.0)
        self.assertEqual(f["over_trail_streak"], 0.0)                       # opps == trailing mean: a 0 is neither over nor under
        over = [row(f"o{i}", 2025, i, 10.0 + i, 10.0, float(i)) for i in range(1, 9)]  # over xFP and rising volume every week
        f = _streak_features(hist(over), target(week=9))
        self.assertEqual((f["over_xfp_streak"], f["over_trail_streak"]), (6.0, 6.0))
        self.assertEqual(features.STREAK_CAP, 6)

    def test_a_zero_residual_ends_a_run_and_a_zero_newest_residual_is_no_run(self):
        f = _streak_features(hist([row("a", 2025, 1, 12.0, 10.0, 5.0), row("b", 2025, 2, 10.0, 10.0, 5.0),
                                   row("c", 2025, 3, 13.0, 10.0, 5.0)]), target(week=4))
        self.assertEqual(f["over_xfp_streak"], 1.0)
        f = _streak_features(hist([row("a", 2025, 1, 12.0, 10.0, 5.0), row("b", 2025, 2, 10.0, 10.0, 5.0)]), target(week=3))
        self.assertEqual(f["over_xfp_streak"], 0.0)

    def test_no_history_at_all(self):
        f = _streak_features(hist([]), target(week=1))
        self.assertEqual((f["over_xfp_streak"], f["over_trail_streak"]), (0.0, 0.0))
        for c in COLS:
            if c not in ("over_xfp_streak", "over_trail_streak"):
                self.assertTrue(math.isnan(f[c]), c)

    def test_one_game_of_history(self):
        f = _streak_features(hist([row("a", 2025, 1, 12.0, 10.0, 5.0)]), target(week=2))
        self.assertEqual((f["resid_xfp_l1"], f["over_xfp_streak"], f["resid_xfp_sum_l4"]), (2.0, 1.0, 2.0))
        self.assertTrue(math.isnan(f["resid_xfp_l2"]) and math.isnan(f["resid_xfp_l3"]))
        self.assertTrue(math.isnan(f["resid_opps_l1"]), "no game before it: no trailing mean to compare with")
        self.assertEqual(f["over_trail_streak"], 0.0)
        self.assertTrue(math.isnan(f["pts_vs_prev_season_l2"]), "no prior season")


# ------------------------------------------------------------------ cross-season carry and NaN xFP
class CrossSeasonAndNaN(unittest.TestCase):
    PREV = HandComputed.PREV

    def test_week_one_sees_last_seasons_regular_season_games_only(self):
        f = _streak_features(hist(self.PREV), target(week=1))
        self.assertEqual((f["resid_xfp_l1"], f["resid_xfp_l2"], f["resid_xfp_l3"]), (0.0, 1.0, -1.0))   # p3, p2, p1: never p4 (POST)
        self.assertEqual(f["over_xfp_streak"], 0.0)                         # newest residual is exactly 0
        self.assertEqual(f["resid_xfp_sum_l4"], 0.0)
        self.assertTrue(same(f["resid_opps_l1"], 6.0 - 6.0))
        self.assertTrue(same(f["pts_vs_prev_season_l2"], 11.0 - 10.0))

    def test_week_three_window_reaches_back_into_last_season(self):
        cur = [row("g1", 2025, 1, 12.0, 10.0, 8.0), row("g2", 2025, 2, 13.0, 10.0, 9.0)]
        f = _streak_features(hist(self.PREV + cur), target(week=3))
        self.assertEqual(f["resid_xfp_l3"], 0.0)                            # p3
        self.assertEqual(f["resid_xfp_sum_l4"], 1.0 + 0.0 + 2.0 + 3.0)      # p2, p3, g1, g2
        self.assertEqual(f["over_xfp_streak"], 2.0)                         # g2, g1; p3's 0 ends it
        self.assertTrue(same(f["resid_opps_l1"], 9.0 - (7.0 + 6.0 + 8.0) / 3))
        self.assertTrue(same(f["pts_vs_prev_season_l2"], 12.5 - 10.0))
        # the previous season's games are last season's only: a 2023 game is not in the window
        older = [row("z", 2023, 18, 99.0, 1.0, 50.0)]
        g = _streak_features(hist(older + self.PREV + cur), target(week=3))
        self.assertEqual({c: g[c] for c in COLS if not math.isnan(g[c])}, {c: f[c] for c in COLS if not math.isnan(f[c])})

    def test_a_game_without_xfp_has_a_nan_residual_and_ends_a_run_but_counts_for_volume(self):
        cur = [row("g1", 2025, 1, 12.0, 10.0, 8.0), row("g2", 2025, 2, 15.0, NAN, 9.0), row("g3", 2025, 3, 13.0, 10.0, 10.0)]
        f = _streak_features(hist(cur), target(week=4))
        self.assertEqual(f["resid_xfp_l1"], 3.0)
        self.assertTrue(math.isnan(f["resid_xfp_l2"]))
        self.assertEqual(f["resid_xfp_l3"], 2.0)
        self.assertEqual(f["over_xfp_streak"], 1.0)                         # g3 only: the NaN does not extend it
        self.assertEqual(f["resid_xfp_sum_l4"], 5.0)                        # NaN-ignoring sum of the residuals that exist
        self.assertEqual(f["over_trail_streak"], 2.0)                       # volume ignores xFP: 8 (no mean), 9 > 8, 10 > 8.5
        self.assertTrue(same(f["resid_opps_l1"], 10.0 - 8.5))
        # newest game without xFP: no run
        f = _streak_features(hist(cur[:2]), target(week=3))
        self.assertEqual(f["over_xfp_streak"], 0.0)
        self.assertTrue(math.isnan(f["resid_xfp_l1"]))
        # every game without xFP: the sum is NaN, not 0
        f = _streak_features(hist([row("g1", 2025, 1, 12.0, NAN, 8.0)]), target(week=2))
        self.assertTrue(math.isnan(f["resid_xfp_sum_l4"]))


# ------------------------------------------------------------------ the registry and the frozen primary
class Registry(unittest.TestCase):
    def test_streak_is_registered_opt_in_and_the_primary_and_serving_inputs_are_unchanged(self):
        self.assertIn("streak", features.FAMILIES)
        self.assertIn("streak", T.OPT_IN_FAMILIES)
        self.assertEqual(len(COLS), 8)
        self.assertEqual(len(set(COLS)), 8)
        for c in COLS:
            self.assertEqual(features.family_of(c), "streak")
            self.assertFalse(c.startswith(("y_", "base_", "spine_")))
        df = pd.DataFrame({c: [0.0, 1.0] for c in features.feature_columns()} | {"position": ["QB", "RB"],
                          "inj_report_status": [None, None], "inj_practice_status": [None, None], "wx_roof_obs": [None, None]})
        prim = T.feature_columns(df, T.PRIMARY)
        self.assertEqual(set(prim) & set(COLS), set(), "the Phase 2 primary must keep exactly its Phase 2 inputs")
        cfg = TR.load_config()
        serving = T.feature_columns(df, TR.serving_spec(cfg), dead=("college_breakout_age",))   # always NA in the real matrix
        X, _ = T.encode(df, serving)
        self.assertEqual(set(X.columns), set(cfg["inputs"]), "the frozen serving inputs must not move")
        self.assertEqual(len(cfg["inputs"]), 71)
        added = T.feature_columns(df, T.Spec("x", add_families=("streak",)))
        self.assertEqual(set(added) - set(prim), set(COLS))

    def test_trial_accepts_the_candidate_and_lists_it_as_opt_in(self):
        v = TR.compose(["+family:streak"], TR.serving_spec(TR.load_config()))
        self.assertEqual(v.spec.add_families, ("streak",))
        self.assertTrue(v.changes_inputs(TR.serving_spec(TR.load_config())))
        text = TR.list_families(TR.load_config())
        self.assertRegex(text, r"streak  \[0/8 not in serving, opt-in\]")
        self.assertRegex(text, r"over_xfp_streak\s+out: opt-in family \(\+family:streak\)")


# ------------------------------------------------------------------ leakage on a synthetic store
def _ts(xs) -> pd.Series:
    return pd.to_datetime(list(xs), utc=True).astype("datetime64[ns, UTC]")


def _store(games: list[tuple[str, int, int, pd.Timestamp, float, float, float]], *, xfp_missing: set[str] = frozenset()) -> RawStore:
    """A store with just the three tables `_history` reads: (game_id, season, week, kickoff, pts, xfp, opps) per game, every
    row stamped at its game's kickoff (the loaders' rule for result tables)."""
    pg = pd.DataFrame({
        "player_id": "P1", "game_id": [g[0] for g in games], "season": [g[1] for g in games], "week": [g[2] for g in games],
        "season_type": "REG", "team": "KC", "position": "WR", "fantasy_points_ppr": [g[4] for g in games],
        "carries": [g[6] / 2 for g in games], "targets": [g[6] / 2 for g in games], "attempts": 0.0,
        "target_share": 0.2, "carry_share": 0.1, "receptions": 3.0, "receiving_yards": 40.0, "receiving_tds": 0.0,
        "rushing_yards": 10.0, "rushing_tds": 0.0, "passing_yards": 0.0, "passing_tds": 0.0, "passing_interceptions": 0.0,
        "completions": 0.0, KNOWN_AT: _ts(g[3] for g in games)})
    sn = pd.DataFrame({"player_id": "P1", "game_id": [g[0] for g in games], "season": [g[1] for g in games],
                       "week": [g[2] for g in games], "game_type": "REG", "team": "KC", "position": "WR",
                       "offense_snaps": 40, "offense_pct": 0.7, KNOWN_AT: _ts(g[3] for g in games)})
    xg = [g for g in games if g[0] not in xfp_missing]
    xf = pd.DataFrame({"player_id": "P1", "game_id": [g[0] for g in xg], "total_fantasy_points_exp": [g[5] for g in xg],
                       "rush_touchdown": 0.0, "rush_touchdown_exp": 0.1, "rec_touchdown": 0.0, "rec_touchdown_exp": 0.2,
                       "pass_touchdown": 0.0, "pass_touchdown_exp": 0.0, KNOWN_AT: _ts(g[3] for g in xg)})
    return RawStore([RawTable("player_games", pg, "test"), RawTable("snap_counts", sn, "test"), RawTable("xfp", xf, "test")])


def _wk(week: int) -> pd.Timestamp:
    return KO - pd.Timedelta(days=7 * (7 - week))


class Leakage(unittest.TestCase):
    PAST = [("2025_01", 2025, 1, _wk(1), 12.0, 10.0, 8.0), ("2025_02", 2025, 2, _wk(2), 9.0, 10.0, 6.0),
            ("2025_03", 2025, 3, _wk(3), 15.0, 12.0, 10.0)]
    OWN = ("2025_07_LV_KC", 2025, 7, KO, 44.0, 10.0, 30.0)                              # the target game itself, stamped at kickoff
    FUTURE = ("2025_08", 2025, 8, KO + pd.Timedelta(days=7), 44.0, 10.0, 30.0)
    AFTER = ("2025_06x", 2025, 6, KO + pd.Timedelta(hours=1), 44.0, 10.0, 30.0)       # an earlier week stamped after kickoff

    def feats(self, store: RawStore, t: TargetRow = target()) -> dict:
        return _streak_features(_history(store, "P1", t, None), t)

    def test_the_target_games_own_row_a_future_row_and_a_post_kickoff_row_never_enter(self):
        base = self.feats(_store(self.PAST))
        self.assertEqual(base["resid_xfp_l1"], 3.0)
        self.assertEqual(base["over_xfp_streak"], 1.0)
        for extra in (self.OWN, self.FUTURE, self.AFTER):
            dirty = self.feats(_store(self.PAST + [extra]))
            self.assertEqual(features.fingerprint(dirty), features.fingerprint(base), extra[0])
        # and the full builder path reads the same history: the row IS in the raw table
        self.assertIn(self.OWN[0], set(_store(self.PAST + [self.OWN])._tables["player_games"].df["game_id"]))

    def test_the_result_cutoff_is_kickoff_minus_four_hours_strictly(self):
        base = self.feats(_store(self.PAST))
        at = self.feats(_store(self.PAST + [("2025_06", 2025, 6, KO - RESULT_LAG, 44.0, 10.0, 30.0)]))
        self.assertEqual(features.fingerprint(at), features.fingerprint(base), "a game stamped exactly at kickoff - 4h is hidden")
        inside = self.feats(_store(self.PAST + [("2025_06", 2025, 6, KO - RESULT_LAG - pd.Timedelta(nanoseconds=1), 44.0, 10.0, 30.0)]))
        self.assertEqual(inside["resid_xfp_l1"], 34.0)                      # 1 ns inside: read
        self.assertEqual(inside["over_xfp_streak"], 2.0)
        self.assertTrue(same(inside["resid_opps_l1"], 30.0 - (8.0 + 6.0 + 10.0) / 3))

    def test_a_game_the_xfp_table_lacks_is_a_nan_residual_through_the_real_history_path(self):
        f = self.feats(_store(self.PAST, xfp_missing={"2025_03"}))
        self.assertTrue(math.isnan(f["resid_xfp_l1"]))                     # he had opportunities, so no xFP row means unknown, not 0
        self.assertEqual(f["over_xfp_streak"], 0.0)
        self.assertEqual(f["resid_xfp_l2"], -1.0)
        self.assertTrue(same(f["resid_opps_l1"], 10.0 - 7.0))              # volume does not depend on xFP

    def test_build_features_wires_the_family_right_after_the_lag_features(self):
        """The builder is read-only on `Hist`: the family adds no gate call, so the trace of a full build is unchanged in which
        tables it reads. Checked structurally (no new `as_of_join` inside the builder) since a full store is not built here."""
        import ast, inspect
        src = inspect.getsource(_streak_features)
        self.assertNotIn("as_of_join", src)
        self.assertNotIn("store", ast.dump(ast.parse(src)))
        full = inspect.getsource(features.build_features)
        self.assertLess(full.index("_lag_and_history_features(h, t)"), full.index("_streak_features(h, t)"))
        self.assertLess(full.index("_streak_features(h, t)"), full.index("_role_features(ctx, h, t)"))


if __name__ == "__main__":
    unittest.main()
