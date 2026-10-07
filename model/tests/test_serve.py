"""Phase 3: the serving path is the training path, pointed at a game that has not been played.

    python3.12 -m unittest model.tests.test_serve

Part one is synthetic and fast: league scoring composition (cross-checked against the pipeline's own `ff.scoring`), who gets
no row, the frozen record, the roster.csv columns (every existing byte untouched), and the promise that a failure is a WARN
row and nothing else. Part two runs on REAL data (the nflverse cache the other suites use; the git-ignored Phase 2.5 matrix
is the independent record the serving rows are compared with, and the checks skip only if it has not been built):

  * serving rows carry the training schema, and exactly Phase 2's 71 inputs
  * nothing a serving row reads is stamped at or after its kickoff: a trace of every gate call, and a truncation audit
    that rebuilds sampled rows of the upcoming week from a store physically cut before their kickoff
  * NO TRAIN/SERVE SKEW: the serving row builder, pointed at COMPLETED weeks, gives the matrix's rows bit for bit, the
    history builder gives the matrix's 2021-2025 rows bit for bit, and a serve of a completed week reproduces the backtest's
    predictions for that week to the last bit: the flat models, every component model, the quantile bands and their
    recalibration
  * the week is chosen from the schedule's real kickoffs and a completed week is never rewritten
"""
from __future__ import annotations

import argparse
import csv
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from ff import scoring as ff_scoring
from ff import status as ff_status
from model import audit, backtest as B, components as C, ensemble as E, features as F, leaguescore as L, serve as SV, train as T
from model import build_features as BF
from model import point_in_time as pit

REPO = Path(__file__).resolve().parents[2]
NOW = pd.Timestamp("2026-09-30 18:00", tz="UTC")              # the Wednesday before week 4
_C: dict = {}


def store() -> pit.RawStore:
    if "store" not in _C:
        _C["store"] = pit.load_store(range(2020, 2027), adp=False, college=False, pbp=False)
    return _C["store"]


def matrix() -> pd.DataFrame:
    """The Phase 2.5 matrix (with the misc label), built by an earlier process: the independent record. Skip if not built."""
    if "matrix" not in _C:
        if not Path(T.MATRIX).exists():
            raise unittest.SkipTest(f"{T.MATRIX} not built: run `python3.12 -m model.build_features`")
        _C["matrix"] = C.add_misc(T.load_matrix())
    return _C["matrix"]


def cfg() -> dict:
    return SV.load_config()


def tgs_of(season: int, week: int) -> list:
    return [g for g in pit.team_games(store(), [season], completed_only=False) if g.week == week]


def key(df: pd.DataFrame) -> pd.Series:
    return df["player_id"].astype(str) + "|" + df["game_id"].astype(str)


def same(a: pd.Series, b: pd.Series) -> bool:
    """Bit-identical values, NaN equal to NaN, None equal to None."""
    if a.dtype == object or b.dtype == object:
        return a.astype(object).where(a.notna(), None).tolist() == b.astype(object).where(b.notna(), None).tolist()
    return bool(np.array_equal(a.to_numpy("float64"), b.to_numpy("float64"), equal_nan=True))


# =========================================================================================== league scoring
PPR_DICT = {"pass_yd": 0.04, "pass_td": 4.0, "pass_int": -2.0, "rush_yd": 0.1, "rush_td": 6.0, "rec": 1.0, "rec_yd": 0.1,
            "rec_td": 6.0, "fum_lost": -2.0, "pass_2pt": 2.0, "rush_2pt": 2.0, "rec_2pt": 2.0}


class LeagueComposition(unittest.TestCase):
    def test_a_ppr_dict_composes_to_ppr(self):
        ls = L.from_scoring_dict("x", "X", "sleeper", PPR_DICT)
        self.assertEqual(ls.scoring.weights, C.PPR.weights)
        self.assertTrue(ls.is_ppr)
        self.assertEqual(ls.ignored_thresholds, {})

    def test_the_four_real_leagues_compose_as_documented(self):
        by = {ls.slug: ls for ls in L.load_leagues(REPO)}
        self.assertEqual(sorted(by), ["gooma-s-family-league", "james-gregg-espn", "we-can-think-of-something-funny", "where-you-at"])
        for slug in ("gooma-s-family-league", "james-gregg-espn"):
            self.assertTrue(by[slug].is_ppr, slug)
            self.assertEqual(by[slug].ignored_thresholds, {}, slug)
        dyn = by["we-can-think-of-something-funny"]
        self.assertEqual(dyn.scoring.weights, {**C.PPR.weights, "y_int": -1.0})
        self.assertEqual(dyn.scoring.position_bonus, {("TE", "y_rec"): 0.5})
        self.assertEqual(sorted(dyn.ignored_thresholds),
                         ["bonus_pass_yd_300", "bonus_pass_yd_400", "bonus_rec_yd_100", "bonus_rec_yd_200", "bonus_rush_yd_100",
                          "bonus_rush_yd_200", "pass_td_40p", "pass_td_50p", "rec_td_40p", "rec_td_50p", "rush_td_40p", "rush_td_50p"])
        idp = by["where-you-at"]
        self.assertEqual(idp.scoring.weights, {**C.PPR.weights, "y_rec": 0.5})
        self.assertEqual(idp.ignored_thresholds, {})                        # IDP / kicking / defence never score on skill positions
        self.assertFalse(idp.is_ppr or dyn.is_ppr)
        for ls in by.values():
            self.assertEqual(ls.misc_weight, 1.0)
            self.assertTrue(set(ls.scoring.needs()) <= set(C.TARGETS))

    def test_composition_matches_the_pipelines_own_scorer_where_no_threshold_fires(self):
        """`ff.scoring.score_frame` turns actual stat lines into a league's points; composing the same lines through
        leaguescore must agree wherever no yardage / long-TD bonus applies (a mean cannot express those)."""
        rng = np.random.default_rng(3)
        n = 400
        pos = rng.choice(list(T.POSITIONS), n)
        stat = pd.DataFrame({"position": pos, "pass_yds": rng.integers(0, 290, n).astype(float), "pass_tds": rng.integers(0, 4, n).astype(float),
                             "pass_int": rng.integers(0, 3, n).astype(float), "rush_yds": rng.integers(0, 95, n).astype(float),
                             "rush_tds": rng.integers(0, 2, n).astype(float), "receptions": rng.integers(0, 12, n).astype(float),
                             "rec_yds": rng.integers(0, 95, n).astype(float), "rec_tds": rng.integers(0, 2, n).astype(float),
                             "fum_lost": rng.integers(0, 2, n).astype(float), "two_pt": rng.integers(0, 2, n).astype(float)})
        comps = {"y_pass_yds": stat.pass_yds, "y_pass_td": stat.pass_tds, "y_int": stat.pass_int, "y_rush_yds": stat.rush_yds,
                 "y_rush_td": stat.rush_tds, "y_rec": stat.receptions, "y_rec_yds": stat.rec_yds, "y_rec_td": stat.rec_tds,
                 "y_fum_lost": stat.fum_lost, C.MISC: 2.0 * stat.two_pt}
        for ls in L.load_leagues(REPO):
            d = __import__("json").loads((REPO / "leagues" / ls.slug / "league.json").read_text())
            want = ff_scoring.score_frame(stat, d["scoring"], ls.slug)
            got = C.compose(comps, stat["position"], ls.scoring).round(2)
            pd.testing.assert_series_equal(got, want, check_names=False, atol=1e-9, obj=ls.slug)

    def test_unmodelled_nonzero_keys_are_listed_and_defence_kicking_and_idp_are_not(self):
        sc = {**PPR_DICT, "rec_fd": 0.5, "bonus_fd_wr": 1.0, "pass_sack": -1.0, "bonus_rec_rb": 0.5, "idp_tkl": 1.0, "fgm_40_49": 4.0,
              "def_td": 6.0, "pts_allow_0": 10.0, "xpm": 1.0, "kr_td": 6.0, "st_td": 6.0, "fum_rec": 2.0, "sack": 1.0, "ff": 1.0,
              "bonus_def_int_td_50p": 2.0, "bonus_rec_yd_100": 0.0, "rush_fd": 0.0}
        ls = L.from_scoring_dict("x", "X", "sleeper", sc)
        self.assertEqual(ls.ignored_thresholds, {"bonus_fd_wr": 1.0, "pass_sack": -1.0, "rec_fd": 0.5})
        self.assertEqual(ls.scoring.position_bonus, {("RB", "y_rec"): 0.5})

    def test_stat_keys_beyond_ppr_become_weights(self):
        ls = L.from_scoring_dict("x", "X", "sleeper", {**PPR_DICT, "rec_tgt": 0.25, "pass_att": -0.1, "pass_cmp": 0.5, "rush_att": 0.2})
        w = ls.scoring.weights
        self.assertEqual((w["y_tgt"], w["y_att"], w["y_cmp"], w["y_car"]), (0.25, -0.1, 0.5, 0.2))
        self.assertFalse(ls.is_ppr)

    def test_the_misc_bucket_follows_the_two_point_value_and_is_off_when_they_disagree(self):
        half = L.from_scoring_dict("x", "X", "s", {**PPR_DICT, "pass_2pt": 1.0, "rush_2pt": 1.0, "rec_2pt": 1.0})
        self.assertEqual(half.misc_weight, 0.5)
        odd = L.from_scoring_dict("x", "X", "s", {**PPR_DICT, "pass_2pt": 2.0, "rush_2pt": 1.0, "rec_2pt": 2.0})
        self.assertEqual(odd.misc_weight, 0.0)
        self.assertFalse(L.from_scoring_dict("x", "X", "s", {k: v for k, v in PPR_DICT.items() if not k.endswith("2pt")}).misc_weight)

    def test_a_hand_composed_quarterback_and_tight_end(self):
        dyn = next(ls for ls in L.load_leagues(REPO) if ls.slug == "we-can-think-of-something-funny")
        comps = {c: pd.Series([0.0, 0.0]) for c in C.TARGETS}
        comps.update({"y_pass_yds": pd.Series([250.0, 0.0]), "y_pass_td": pd.Series([2.0, 0.0]), "y_int": pd.Series([1.0, 0.0]),
                      "y_rush_yds": pd.Series([20.0, 0.0]), "y_rush_td": pd.Series([0.2, 0.0]), "y_rec": pd.Series([0.0, 5.0]),
                      "y_rec_yds": pd.Series([0.0, 60.0]), "y_rec_td": pd.Series([0.0, 0.4])})
        pts = C.compose(comps, pd.Series(["QB", "TE"]), dyn.scoring)
        self.assertAlmostEqual(pts[0], 250 * 0.04 + 2 * 4 - 1 + 20 * 0.1 + 0.2 * 6)            # interception is -1 here
        self.assertAlmostEqual(pts[1], 5 * 1.0 + 0.5 * 5 + 60 * 0.1 + 0.4 * 6)                 # +0.5 per TE catch
        ppr = C.compose(comps, pd.Series(["QB", "TE"]), C.PPR)
        self.assertAlmostEqual(ppr[1], 5 + 6 + 2.4)


# =========================================================================================== the serving design
class LeagueTargets(unittest.TestCase):
    def test_a_points_label_for_each_non_ppr_league_from_the_actual_components_and_nan_where_nothing_was_played(self):
        leagues = L.load_leagues(REPO)
        df = pd.DataFrame({"position": ["TE", "WR", "QB", "RB"], **{c: [np.nan, 4.0, 1.0, 0.0] for c in C.TARGETS}})
        df.loc[0, "y_rec"] = np.nan
        out = SV.add_league_targets(df, leagues)
        added = sorted(c for c in out.columns if c.startswith("y_pts_"))
        self.assertEqual(added, ["y_pts_we-can-think-of-something-funny", "y_pts_where-you-at"])      # PPR leagues need no label of their own
        ppr_like = sum(w * 4.0 for w in C.PPR.weights.values())              # every component at 4.0 under full PPR
        half = out["y_pts_where-you-at"]
        self.assertAlmostEqual(half[1], ppr_like - 0.5 * 4.0)                # half PPR: each of the 4.0 receptions is worth 0.5 less
        self.assertTrue(np.isnan(half[0]))                                   # nothing known for that row
        dyn = out["y_pts_we-can-think-of-something-funny"]
        self.assertAlmostEqual(dyn[1], ppr_like + 4.0)                       # a WR: no TE premium, but 4 interceptions at -1 instead of -2
        te = SV.add_league_targets(df.assign(position="TE"), leagues)["y_pts_we-can-think-of-something-funny"]
        self.assertAlmostEqual(te[1] - dyn[1], 0.5 * 4.0)                    # the same stat line as a tight end earns 0.5 more per catch
        self.assertEqual(len(df.columns), len(C.TARGETS) + 1)                # the input frame is not modified


class ShippedDesign(unittest.TestCase):
    def test_the_inputs_are_phase_2s_71_columns_and_nothing_new(self):
        c = cfg()
        self.assertEqual(len(c["inputs"]), 71)
        self.assertEqual(len(c["spec"]["drop_cols"]), 39)
        self.assertEqual(c["spec"]["drop_families"], ["adp"])
        bad = [i for i in c["inputs"] if i.startswith(("wx_", "adp_", "college_", "pbp_", "part_", "team_plays", "eff_"))
               or i in ("season", "week")]
        self.assertEqual(bad, [])
        cols = [x for x in F.feature_columns() if F.family_of(x) not in T.OPT_IN_FAMILIES + ("weather",)]
        frame = pd.DataFrame({x: [0.0] for x in cols} | {"position": ["QB"], "inj_report_status": ["Out"], "inj_practice_status": [None]})
        chosen = T.feature_columns(frame, SV.spec_from(c), dead=("college_breakout_age",))
        X, _ = T.encode(frame, chosen)
        self.assertEqual(list(X.columns), c["inputs"])

    def test_the_frozen_prune_list_is_the_one_phase_2_wrote(self):
        p = REPO / "model" / "cache" / "phase2" / "prune_list.json"
        if not p.exists():
            self.skipTest("model/cache/phase2/prune_list.json not present")
        import json
        self.assertEqual(json.loads(p.read_text())["drop"], cfg()["spec"]["drop_cols"])

    def test_every_component_has_a_frozen_tree_count(self):
        c = cfg()["components"]
        self.assertEqual(sorted(c), sorted(C.TARGETS))
        self.assertTrue(all(v["n_estimators"] >= 50 for v in c.values()))

    def test_flat_quantile_and_recalibration_settings_are_the_shipped_ones(self):
        self.assertEqual(cfg()["recalibration"], {"mode": "underage_only"})
        self.assertEqual(T.LIBS, ("lightgbm", "xgboost", "catboost"))
        p = T.load_params()
        self.assertEqual(sorted(p["quantile"]), ["0.1", "0.5", "0.9"])


# =========================================================================================== who gets no row
class Availability(unittest.TestCase):
    def test_the_status_layer_owns_availability_and_doubtful_and_questionable_keep_a_row(self):
        ids = ["ok", "sl_out", "nfl_out", "ir", "pup", "sus", "cut", "res", "exe", "ret", "doubtful", "quest", "unknown", "sl_quest"]
        frame = pd.DataFrame({"player_id": ids, "inj_report_status": [None, None, "Out", None, None, None, None, None, None, None,
                                                                      "Doubtful", "Questionable", None, None]})
        frame["inj_weeks_since_report"] = [None, None, 0, None, None, None, None, None, None, None, 0, 0, None, None]     # this week's reports
        players = pd.DataFrame({"gsis_id": ids[:-2] + ["sl_quest"],
                                "injury_status": [None, "Out", None, "IR", "PUP", "Sus", None, None, None, None, None, None, "Questionable"],
                                "nfl_status": ["ACT", "ACT", "ACT", "RES", "RES", "ACT", "CUT", "RES", "EXE", "RET", "ACT", "ACT", "ACT"]})
        withheld, why = SV.availability(frame, players)
        got = dict(zip(ids, withheld))
        self.assertEqual([i for i, w in got.items() if w],
                         ["sl_out", "nfl_out", "ir", "pup", "sus", "cut", "res", "exe", "ret"])
        self.assertEqual(why[frame.player_id == "nfl_out"].iloc[0], "nflverse report: Out")
        self.assertEqual(why[frame.player_id == "sl_out"].iloc[0], "status layer: Out")
        self.assertEqual(why[frame.player_id == "cut"].iloc[0], "status layer: nfl_status CUT")
        self.assertFalse(any(got[i] for i in ("ok", "doubtful", "quest", "unknown", "sl_quest")))

    def test_last_weeks_report_does_not_withhold_and_is_not_shown_as_this_weeks(self):
        """On a Wednesday the newest report this season can be last week's final 'Out'. Sleeper's live status speaks for this week."""
        frame = pd.DataFrame({"player_id": ["back", "stale_out", "fresh_out", "none"],
                              "inj_report_status": ["Out", "Doubtful", "Out", None],
                              "inj_weeks_since_report": [1, 2, 0, None]})
        players = pd.DataFrame({"gsis_id": ["back"], "injury_status": ["Questionable"], "nfl_status": ["ACT"]})
        withheld, _ = SV.availability(frame, players)
        self.assertEqual(withheld.tolist(), [False, False, True, False])
        self.assertEqual(SV.current_report(frame).tolist(), [None, None, "Out", None])
        # ...but last week's Out does not hide a player the status layer still calls Out
        players = pd.DataFrame({"gsis_id": ["back"], "injury_status": ["Out"], "nfl_status": ["ACT"]})
        self.assertTrue(SV.availability(frame, players)[0].iloc[0])

    def test_every_state_the_pipelines_status_layer_calls_out_is_withheld(self):
        for state in ff_status.OUT_STATES:
            frame = pd.DataFrame({"player_id": ["p"], "inj_report_status": [None], "inj_weeks_since_report": [None]})
            players = pd.DataFrame({"gsis_id": ["p"], "injury_status": [state], "nfl_status": ["ACT"]})
            self.assertTrue(SV.availability(frame, players)[0].all(), state)

    def test_without_a_players_table_only_this_weeks_nflverse_report_withholds(self):
        frame = pd.DataFrame({"player_id": ["a", "b", "c"], "inj_report_status": ["Out", "Questionable", "Out"],
                              "inj_weeks_since_report": [0, 0, 1]})
        self.assertEqual(SV.availability(frame, None)[0].tolist(), [True, False, False])


# =========================================================================================== the frozen record
def mp_rows(ids, pts, kick, week=4, season=2026, **extra) -> pd.DataFrame:
    return pd.DataFrame({"season": season, "week": week, "game_id": [f"g{k[-5:]}" for k in kick], "kickoff_utc": kick,
                         "gsis_id": ids, "pts_model": pts, **extra})


class FrozenRecord(unittest.TestCase):
    TNF, SUN = "2026-10-02T00:15:00+00:00", "2026-10-04T17:00:00+00:00"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "model_pts.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def read(self) -> pd.DataFrame:
        return pd.read_csv(self.path, dtype={"gsis_id": str}).set_index("gsis_id")

    def test_rows_of_games_under_way_stay_exactly_as_written_and_the_rest_are_replaced(self):
        SV.write_model_pts(self.path, mp_rows(["A", "B", "C"], [10.0, 5.0, 7.0], [self.TNF, self.SUN, self.SUN]), 2026, 4,
                           pd.Timestamp("2026-09-30", tz="UTC"))
        SV.write_model_pts(self.path, mp_rows(["X"], [1.0], [self.SUN], week=3), 2026, 3, pd.Timestamp("2026-09-23", tz="UTC"))
        # Friday evening: Thursday's game has kicked off. The serve only predicts games that have not: B changes, D is new, C is gone.
        later = mp_rows(["B", "D"], [6.0, 8.0], [self.SUN, self.SUN])
        SV.write_model_pts(self.path, later, 2026, 4, pd.Timestamp("2026-10-02T22:00:00Z"))
        d = self.read()
        self.assertEqual(d.loc["A", "pts_model"], 10.0)                        # Thursday's row is the frozen one
        self.assertEqual(d.loc["B", "pts_model"], 6.0)
        self.assertEqual(d.loc["D", "pts_model"], 8.0)
        self.assertNotIn("C", d.index)
        self.assertEqual(d.loc["X", "pts_model"], 1.0)                         # another week untouched

    def test_a_week_whose_games_have_all_kicked_off_is_not_changed_by_a_later_run(self):
        SV.write_model_pts(self.path, mp_rows(["A", "B"], [10.0, 5.0], [self.TNF, self.SUN]), 2026, 4, pd.Timestamp("2026-09-30", tz="UTC"))
        before = self.path.read_bytes()
        SV.write_model_pts(self.path, mp_rows([], [], [], week=4).astype({"gsis_id": object}), 2026, 4, pd.Timestamp("2026-10-06", tz="UTC"))
        self.assertEqual(self.path.read_bytes(), before)

    def test_a_frozen_players_row_wins_over_a_new_row_for_the_same_player(self):
        SV.write_model_pts(self.path, mp_rows(["A"], [10.0], [self.TNF]), 2026, 4, pd.Timestamp("2026-09-30", tz="UTC"))
        SV.write_model_pts(self.path, mp_rows(["A"], [99.0], [self.TNF]), 2026, 4, pd.Timestamp("2026-10-03", tz="UTC"))
        self.assertEqual(self.read().loc["A", "pts_model"], 10.0)

    def test_a_rerun_is_byte_identical_and_leaves_no_temp_file(self):
        now = pd.Timestamp("2026-09-30", tz="UTC")
        rows = mp_rows(["A", "B"], [10.0, 5.5], [self.TNF, self.SUN], p10=[1.0, 2.0])
        SV.write_model_pts(self.path, rows, 2026, 4, now)
        first = self.path.read_bytes()
        SV.write_model_pts(self.path, rows, 2026, 4, now)
        self.assertEqual(self.path.read_bytes(), first)
        self.assertEqual(sorted(p.name for p in self.path.parent.iterdir()), ["model_pts.csv"])

    def test_it_goes_through_the_pipelines_replace_partition(self):
        from ff import build as ffb
        with mock.patch.object(ffb, "replace_partition", wraps=ffb.replace_partition) as rp:
            SV.write_model_pts(self.path, mp_rows(["A"], [1.0], [self.SUN]), 2026, 4, pd.Timestamp("2026-09-30", tz="UTC"))
        self.assertEqual(rp.call_args.args[2:], (["season", "week"], ["season", "week", "gsis_id"]))

    def test_columns_added_later_do_not_break_old_rows(self):
        now = pd.Timestamp("2026-09-30", tz="UTC")
        SV.write_model_pts(self.path, mp_rows(["A"], [1.0], [self.SUN], week=3), 2026, 3, now)
        SV.write_model_pts(self.path, mp_rows(["B"], [2.0], [self.SUN], p10=[0.5]), 2026, 4, now)
        d = self.read()
        self.assertTrue(np.isnan(d.loc["A", "p10"]) and d.loc["B", "p10"] == 0.5)


# =========================================================================================== roster.csv columns
def fake_table(path: Path, every: int = 2) -> pd.DataFrame:
    g = pd.read_csv(path, dtype={"gsis_id": str}, usecols=["gsis_id"])["gsis_id"]
    t = pd.DataFrame({"gsis_id": g, "E_pts_model": np.arange(len(g)) * 0.37 + 1, "p10": np.arange(len(g)) * 0.11, "p90": np.arange(len(g)) * 0.9 + 9})
    t.loc[::every, ["E_pts_model", "p10", "p90"]] = np.nan                     # the model has no row for every other player
    return t.round(1)


class RosterColumns(unittest.TestCase):
    def test_every_existing_byte_of_every_real_roster_csv_is_untouched(self):
        """Dropping the three new columns from the output gives back the file: compared as text, column by column, after the same
        drop from the input (a committed roster.csv that has already been through the model step carries them itself)."""
        names = ("E_pts_model", "p10", "p90")

        def serialise(rows):
            buf = io.StringIO()
            csv.writer(buf, lineterminator="\n").writerows(rows)
            return buf.getvalue()

        def without(rows):
            drop = [i for i, h in enumerate(rows[0]) if h in names]
            return [[c for i, c in enumerate(r) if i not in drop] for r in rows]
        for p in sorted((REPO / "leagues").glob("*/roster.csv")):
            original = p.read_text(encoding="utf-8")
            text = SV.roster_text(p, fake_table(p))
            rows = list(csv.reader(io.StringIO(text)))
            self.assertEqual(rows[0][-3:], list(names), p.parent.name)
            self.assertEqual(serialise([r[:-3] for r in rows]), serialise(without(list(csv.reader(io.StringIO(original))))), p.parent.name)
            if not any(n in original.splitlines()[0].split(",") for n in names):
                self.assertEqual(serialise([r[:-3] for r in rows]), original, p.parent.name)    # the pre-model file, byte for byte

    def test_the_columns_hold_the_joined_values_and_blanks_where_the_model_has_no_row(self):
        p = REPO / "leagues" / "where-you-at" / "roster.csv"
        t = fake_table(p)
        rows = list(csv.DictReader(io.StringIO(SV.roster_text(p, t))))
        gsis = {r["gsis_id"]: r for r in rows}
        for r in t.itertuples(index=False):
            got = gsis[r.gsis_id]["E_pts_model"]
            self.assertEqual(got, "" if r.E_pts_model != r.E_pts_model else repr(float(r.E_pts_model)))

    def test_running_twice_replaces_the_columns_instead_of_stacking_them(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "roster.csv"
            p.write_text((REPO / "leagues" / "where-you-at" / "roster.csv").read_text(encoding="utf-8"), encoding="utf-8")
            t = fake_table(p)
            first = SV.roster_text(p, t)
            p.write_text(first, encoding="utf-8", newline="")
            self.assertEqual(SV.roster_text(p, t), first)
            self.assertEqual(list(csv.reader(io.StringIO(first)))[0].count("E_pts_model"), 1)

    def test_a_roster_from_another_week_is_left_alone_with_a_warning(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "leagues" / "where-you-at").mkdir(parents=True)
            src = REPO / "leagues" / "where-you-at" / "roster.csv"
            (root / "leagues" / "where-you-at" / "roster.csv").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            ls = [x for x in L.load_leagues(REPO) if x.slug == "where-you-at"]
            frame = pd.DataFrame({"gsis_id": ["x"], "pts_where-you-at": [1.0], "pts_model_components": [1.0], "p10": [0.0], "p90": [2.0]})
            w: list = []
            wk = int(pd.read_csv(src, usecols=["week"])["week"].iloc[0])
            self.assertEqual(SV.roster_texts(root, frame, ls, 2026, wk + 1, w), {})
            self.assertEqual(len(w), 1)
            self.assertIn("not the served week", w[0])
            self.assertEqual(len(SV.roster_texts(root, frame, ls, 2026, wk, w)), 1)

    def test_a_bad_roster_file_raises_before_anything_at_all_is_written(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            leagues = [x for x in L.load_leagues(REPO) if x.slug in ("gooma-s-family-league", "where-you-at")]
            for ls in leagues:
                (root / "leagues" / ls.slug).mkdir(parents=True)
                shutil_copy(REPO / "leagues" / ls.slug / "roster.csv", root / "leagues" / ls.slug / "roster.csv")
            bad = root / "leagues" / "where-you-at" / "roster.csv"
            head = pd.read_csv(bad, nrows=2)
            head.drop(columns=["gsis_id"]).to_csv(bad, index=False)              # keeps season/week, loses the join key
            good = root / "leagues" / "gooma-s-family-league" / "roster.csv"
            good_before = good.read_bytes()
            wk = int(pd.read_csv(good, usecols=["week"])["week"].iloc[0])
            frame = pd.DataFrame({"season": 2026, "week": wk, "gsis_id": ["x"], "game_id": "g", "kickoff_utc": "2026-10-04T17:00:00+00:00",
                                  "pts_model": [1.0], "pts_model_components": [1.0], "p10": [0.0], "p90": [2.0],
                                  **{f"pts_{x.slug}": [1.0] for x in leagues}})
            out = root / "data" / "model_pts.csv"
            with self.assertRaises(Exception):
                SV.write_outputs(root, out, frame, leagues, 2026, wk, pd.Timestamp("2026-09-30", tz="UTC"), [])
            self.assertFalse(out.exists())
            self.assertEqual(good.read_bytes(), good_before)


class FridayServe(unittest.TestCase):
    """The Friday run: Thursday night's game has kicked off, so the serve predicts only the later games. The roster.csv model columns
    (and the summary) must still carry the Thursday players, whose frozen rows model_pts.csv keeps."""
    TNF, SUN = "2026-10-02T00:15:00+00:00", "2026-10-04T17:00:00+00:00"
    FRIDAY = pd.Timestamp("2026-10-02T22:00:00Z")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ls = [x for x in L.load_leagues(REPO) if x.slug == "where-you-at"]
        (self.root / "leagues" / "where-you-at").mkdir(parents=True)
        src = REPO / "leagues" / "where-you-at" / "roster.csv"
        r = pd.read_csv(src, low_memory=False, dtype={"gsis_id": str})
        r.drop(columns=[c for c in ("E_pts_model", "p10", "p90") if c in r.columns]).to_csv(self.root / "leagues" / "where-you-at" / "roster.csv", index=False)
        self.week = int(r["week"].iloc[0])
        self.ids = r["gsis_id"].dropna().tolist()[:3]                          # [0] plays Thursday night, [1] and [2] on Sunday
        self.out = self.root / "data" / "model_pts.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self, ids, kicks, pts):
        return pd.DataFrame({"season": 2026, "week": self.week, "game_id": "g", "kickoff_utc": kicks, "gsis_id": ids, "pts_model_components": pts,
                             "pts_where-you-at": pts, "p10": 1.0, "p90": 30.0, "p10_where-you-at": 2.0, "p90_where-you-at": 31.0})

    def roster(self):
        return pd.read_csv(self.root / "leagues" / "where-you-at" / "roster.csv", dtype={"gsis_id": str}).set_index("gsis_id")

    def test_a_thursday_player_keeps_his_roster_columns_on_the_friday_run(self):
        a, b, c = self.ids
        SV.write_outputs(self.root, self.out, self.rows([a, b, c], [self.TNF, self.SUN, self.SUN], [10.0, 11.0, 12.0]), self.ls, 2026, self.week,
                         pd.Timestamp("2026-09-30T18:00:00Z"), [])               # Wednesday: everyone predicted
        self.assertEqual(self.roster().loc[[a, b, c], "E_pts_model"].tolist(), [10.0, 11.0, 12.0])
        # Friday: Thursday's game is under way, so only the Sunday players are predicted (b's number moves, c is now ruled Out: no row)
        n, done = SV.write_outputs(self.root, self.out, self.rows([b], [self.SUN], [15.0]), self.ls, 2026, self.week, self.FRIDAY, [])
        self.assertEqual(done, ["where-you-at"])
        r = self.roster()
        self.assertEqual(r.loc[a, "E_pts_model"], 10.0)                        # the frozen Thursday row still shows
        self.assertEqual((r.loc[a, "p10"], r.loc[a, "p90"]), (2.0, 31.0))
        self.assertEqual(r.loc[b, "E_pts_model"], 15.0)                        # refreshed
        self.assertTrue(pd.isna(r.loc[c, "E_pts_model"]))                      # no row anywhere: blank
        mp = pd.read_csv(self.out, dtype={"gsis_id": str}).set_index("gsis_id")
        self.assertEqual(sorted(mp.index), sorted([a, b]))
        self.assertEqual(n, 2)

    def test_the_summary_counts_the_frozen_players_too(self):
        a, b, _ = self.ids
        SV.write_model_pts(self.out, self.rows([a, b], [self.TNF, self.SUN], [10.0, 11.0]), 2026, self.week, pd.Timestamp("2026-09-30T18:00:00Z"))
        merged = SV.merged_week(self.out, self.rows([b], [self.SUN], [15.0]), 2026, self.week, self.FRIDAY)
        self.assertEqual(sorted(merged["gsis_id"]), sorted([a, b]))
        self.assertEqual(merged.set_index("gsis_id").loc[a, "pts_model_components"], 10.0)
        self.assertEqual(merged.set_index("gsis_id").loc[b, "pts_model_components"], 15.0)


# The 2026 week-5 slate as scheduled on 2026-10-07 (nflverse): Thursday night, an early international game, eight 1 pm games, the late games,
# Sunday night and Monday night. Hard-coded: the serve's rule is about kickoff TIMES, and a later flex must not move this test.
WEEK5_KICKS = {"2026_05_TB_DAL": "2026-10-09T00:15:00+00:00", "2026_05_PHI_JAX": "2026-10-11T13:30:00+00:00",
               **{f"2026_05_{g}": "2026-10-11T17:00:00+00:00" for g in ("IND_PIT", "CIN_MIA", "CHI_GB", "LV_NE", "CLE_NYJ", "NYG_WAS", "MIN_NO", "HOU_TEN")},
               "2026_05_DEN_LAC": "2026-10-11T20:05:00+00:00", "2026_05_SF_SEA": "2026-10-11T20:25:00+00:00", "2026_05_DET_ARI": "2026-10-11T20:25:00+00:00",
               "2026_05_BAL_ATL": "2026-10-12T00:20:00+00:00", "2026_05_BUF_LA": "2026-10-13T00:15:00+00:00"}
WED, FRI, SUN_RUN = (pd.Timestamp(x) for x in ("2026-10-07T14:07:00Z", "2026-10-09T22:11:00Z", "2026-10-11T15:52:00Z"))     # the workflow's runs


class SundayServe(unittest.TestCase):
    """Phase 4: a third weekly serve, at 15:52 UTC on Sunday, an hour before the 1 pm slate locks. What the serve does at that instant is the
    same rule as on Wednesday and Friday (rows of a game that has kicked off are frozen, every other game is re-predicted), so the evidence
    is a three-serve week on the real week-5 kickoffs: Thursday night is frozen from Friday, the 13:30 UTC international game from Sunday, and
    everything from the 1 pm slate to Monday night is refreshed on Sunday."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "model_pts.csv"
        self.games = list(WEEK5_KICKS)

    def tearDown(self):
        self.tmp.cleanup()

    def serve(self, now, pts: float, drop=()):
        """What `serve.run` hands the writer at `now`: two players for every game that has NOT kicked off, all predicting `pts`."""
        upcoming = [g for g in self.games if pd.Timestamp(WEEK5_KICKS[g]) > now]
        rows = pd.DataFrame([{"season": 2026, "week": 5, "game_id": g, "kickoff_utc": WEEK5_KICKS[g], "gsis_id": f"{g}:{i}", "pts_model": pts,
                              "served_at": now.strftime("%Y-%m-%dT%H:%M:%SZ")} for g in upcoming for i in (1, 2)],
                            columns=["season", "week", "game_id", "kickoff_utc", "gsis_id", "pts_model", "served_at"])
        rows = rows[~rows["gsis_id"].isin(drop)]
        SV.write_model_pts(self.path, rows, 2026, 5, now)
        return len(upcoming)

    def read(self) -> pd.DataFrame:
        return pd.read_csv(self.path, dtype={"gsis_id": str})

    def lines(self, game_id: str) -> list[str]:
        return [ln for ln in self.path.read_text().splitlines() if f",{game_id}," in ln]

    def test_thursday_is_frozen_from_friday_the_international_game_from_sunday_and_the_rest_is_refreshed_with_no_duplicates(self):
        SV.write_model_pts(self.path, mp_rows(["W4"], [9.0], ["2026-10-04T17:00:00+00:00"], week=4), 2026, 4, pd.Timestamp("2026-09-30T18:00:00Z"))
        self.assertEqual(self.serve(WED, 1.0), 15)
        wed_tnf = self.lines("2026_05_TB_DAL")
        self.assertEqual(self.serve(FRI, 2.0), 14)                                          # Thursday night is under way: not re-predicted
        fri_intl = self.lines("2026_05_PHI_JAX")
        self.assertEqual(self.lines("2026_05_TB_DAL"), wed_tnf)
        self.assertEqual(self.serve(SUN_RUN, 3.0), 13)                                      # Sunday 15:52: Thursday night AND the 13:30 game are under way
        d = self.read()
        w5 = d[d["week"] == 5]
        self.assertEqual(len(w5), 30)                                                       # 15 games x 2 players: nothing doubled, nothing lost
        self.assertFalse(w5.duplicated(["season", "week", "gsis_id"]).any())
        self.assertEqual(self.lines("2026_05_TB_DAL"), wed_tnf)                             # Thursday: the Wednesday row, byte for byte, after three serves
        self.assertEqual(self.lines("2026_05_PHI_JAX"), fri_intl)                           # the international game: Friday's row, frozen on Sunday
        by_game = w5.groupby("game_id")["pts_model"].agg(lambda x: set(x))
        self.assertEqual(by_game["2026_05_TB_DAL"], {1.0})
        self.assertEqual(by_game["2026_05_PHI_JAX"], {2.0})
        refreshed = [g for g in self.games if g not in ("2026_05_TB_DAL", "2026_05_PHI_JAX")]
        self.assertEqual(len(refreshed), 13)
        self.assertTrue(all(by_game[g] == {3.0} for g in refreshed))                        # 1 pm slate through Monday night: Sunday's numbers
        self.assertEqual(d.loc[d["week"] == 4, "pts_model"].tolist(), [9.0])                # another week is never touched

    def test_a_player_ruled_out_sunday_morning_loses_his_unplayed_row_and_a_player_in_a_game_under_way_keeps_his(self):
        self.serve(WED, 1.0)
        self.serve(FRI, 2.0)
        # Sunday morning: one 1 pm player and one Thursday-night player are both missing from the new predictions (ruled Out, or withheld)
        self.serve(SUN_RUN, 3.0, drop=("2026_05_IND_PIT:1", "2026_05_TB_DAL:1"))
        ids = set(self.read()["gsis_id"])
        self.assertNotIn("2026_05_IND_PIT:1", ids)                                          # his game has not kicked off: the refresh removes him
        self.assertIn("2026_05_IND_PIT:2", ids)
        self.assertIn("2026_05_TB_DAL:1", ids)                                              # his game has: the frozen row stays, the scoreboard judges it

    def test_a_late_sunday_run_after_the_early_slate_freezes_it_too_and_a_run_after_monday_night_changes_nothing(self):
        self.serve(WED, 1.0)
        self.serve(FRI, 2.0)
        self.serve(SUN_RUN, 3.0)
        before = self.path.read_bytes()
        late = pd.Timestamp("2026-10-11T17:30:00Z")                                         # a delayed Sunday run: the 1 pm games are under way
        self.assertEqual(self.serve(late, 4.0), 5)                                          # 20:05, 2 x 20:25, Sunday night, Monday night
        d = self.read()
        by_game = d[d["week"] == 5].groupby("game_id")["pts_model"].agg(lambda x: set(x))
        self.assertEqual(by_game["2026_05_IND_PIT"], {3.0})                                 # frozen at the 15:52 numbers
        self.assertEqual(by_game["2026_05_DEN_LAC"], {4.0})
        end = pd.Timestamp("2026-10-14T18:37:00Z")
        before = self.path.read_bytes()
        self.assertEqual(self.serve(end, 5.0), 0)                                           # Tuesday's run: nothing left to predict this week
        self.assertEqual(self.path.read_bytes(), before)

    def test_the_roster_columns_of_a_player_in_a_frozen_game_persist_through_the_sunday_run(self):
        root = Path(self.tmp.name)
        ls = [x for x in L.load_leagues(REPO) if x.slug == "where-you-at"]
        (root / "leagues" / "where-you-at").mkdir(parents=True)
        r = pd.read_csv(REPO / "leagues" / "where-you-at" / "roster.csv", low_memory=False, dtype={"gsis_id": str})
        r.drop(columns=[c for c in ("E_pts_model", "p10", "p90") if c in r.columns]).to_csv(root / "leagues" / "where-you-at" / "roster.csv", index=False)
        week = int(r["week"].iloc[0])
        tnf, intl, one_pm = r["gsis_id"].dropna().tolist()[:3]
        games = {tnf: "2026_05_TB_DAL", intl: "2026_05_PHI_JAX", one_pm: "2026_05_IND_PIT"}

        def rows(now, pts):
            keep = [i for i in games if pd.Timestamp(WEEK5_KICKS[games[i]]) > now]
            return pd.DataFrame({"season": 2026, "week": week, "game_id": [games[i] for i in keep], "kickoff_utc": [WEEK5_KICKS[games[i]] for i in keep],
                                 "gsis_id": keep, "pts_model_components": pts, "pts_where-you-at": pts, "p10": pts - 1, "p90": pts + 9,
                                 "p10_where-you-at": pts - 2, "p90_where-you-at": pts + 8})
        out = root / "data" / "model_pts.csv"
        for now, pts in ((WED, 10.0), (FRI, 11.0), (SUN_RUN, 12.0)):
            SV.write_outputs(root, out, rows(now, pts), ls, 2026, week, now, [])
        ro = pd.read_csv(root / "leagues" / "where-you-at" / "roster.csv", dtype={"gsis_id": str}).set_index("gsis_id")
        self.assertEqual(ro.loc[tnf, "E_pts_model"], 10.0)                                  # Wednesday's number: frozen at Thursday's kickoff
        self.assertEqual(ro.loc[intl, "E_pts_model"], 11.0)                                 # Friday's: frozen at the 13:30 kickoff
        self.assertEqual(ro.loc[one_pm, "E_pts_model"], 12.0)                               # Sunday's: refreshed an hour before the lock
        self.assertEqual((ro.loc[tnf, "p10"], ro.loc[tnf, "p90"]), (8.0, 18.0))              # this league's own band (non-PPR scoring)

    def test_the_run_log_gets_one_more_row_per_serve_and_a_third_serve_of_a_week_replaces_nothing(self):
        from ff import verify
        root = Path(self.tmp.name)
        (root / "logs").mkdir()
        (root / "logs" / "runs.csv").write_text("ran_at,season,week,status,fails,warns,detail,note\n2026-10-06T22:47:17Z,2026,5,WARN,0,1,status.practice: x,\n")
        for i in range(3):
            SV.log_warns(root, [f"serve {i}: no Sleeper projections yet"], check=SV.SERVE_CHECK)
        rows = list(csv.DictReader(open(root / "logs" / "runs.csv", encoding="utf-8")))
        self.assertEqual(len(rows), 4)                                                      # appended, never replaced
        self.assertEqual({(r["season"], r["week"]) for r in rows}, {("2026", "5")})         # each next to the pipeline run it belongs to
        self.assertEqual([r["detail"].split(":")[1].strip() for r in rows[1:]], ["serve 0", "serve 1", "serve 2"])
        self.assertIsNotNone(verify.last_run_gap(root / "logs"))


class FrozenInputs(unittest.TestCase):
    """serving_config.json freezes the 71 inputs Phase 3 validated; a data-dependent change in which columns are dead must not serve a
    different model."""

    def frame(self, **override) -> pd.DataFrame:
        cols = [x for x in F.feature_columns() if F.family_of(x) not in T.OPT_IN_FAMILIES + ("weather",)]
        rows = 40
        d = {x: np.linspace(0.0, 1.0, rows) + i for i, x in enumerate(cols)}
        d.update(position=(["QB", "RB", "WR", "TE"] * 10), inj_report_status=[None] * rows, inj_practice_status=[None] * rows,
                 season=[2021 + i % 4 for i in range(rows)], y_played=[1] * rows)
        d["college_breakout_age"] = np.nan                                    # the one column that is dead in the real data too
        d.update(override)
        return pd.DataFrame(d)

    def test_the_live_inputs_equal_the_frozen_ones(self):
        self.assertEqual(SV.check_inputs(self.frame(), cfg()), cfg()["inputs"])

    def test_an_input_that_turns_dead_in_the_data_is_refused_not_silently_dropped(self):
        col = next(c for c in cfg()["inputs"] if c.startswith("dvp_"))
        with self.assertRaises(SV.ServingSpecMismatch) as cm:
            SV.check_inputs(self.frame(**{col: np.nan}), cfg())                  # all-NA in the training years: `dead_columns` would drop it
        self.assertIn(col, str(cm.exception))
        self.assertIn("missing", str(cm.exception))

    def test_a_changed_frozen_list_is_refused_too(self):
        c = {**cfg(), "inputs": cfg()["inputs"][:-1]}
        with self.assertRaises(SV.ServingSpecMismatch) as cm:
            SV.check_inputs(self.frame(), c)
        self.assertIn("unexpected", str(cm.exception))

    def test_predict_week_refuses_before_fitting_anything_and_the_step_turns_it_into_a_warn(self):
        bad = self.frame(**{"team_spread": np.nan})
        with mock.patch.object(SV, "flat_predictions", side_effect=AssertionError("fitted a model the validation never saw")):
            with self.assertRaises(SV.ServingSpecMismatch):
                SV.predict_week(bad, 2026, 4, cfg(), log=lambda *_: None)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "logs").mkdir()
            (root / "logs" / "runs.csv").write_text("ran_at,season,week,status,fails,warns,detail,note\n2026-09-29T17:05:23Z,2026,4,OK,0,0,,\n")
            with mock.patch.object(SV, "ROOT", root), mock.patch.object(SV, "run", side_effect=SV.ServingSpecMismatch("the live feature set is not the frozen one")), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(SV.main([]), 0)
            with open(root / "logs" / "runs.csv", encoding="utf-8") as fh:
                last = list(csv.DictReader(fh))[-1]
            self.assertEqual((last["status"], last["warns"]), ("WARN", "1"))
            self.assertIn("ServingSpecMismatch: the live feature set is not the frozen one", last["detail"])


def shutil_copy(a: Path, b: Path) -> None:
    b.write_bytes(a.read_bytes())


# =========================================================================================== it never breaks the pipeline
class NeverBreaksThePipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "logs").mkdir()
        (self.root / "logs" / "runs.csv").write_text("ran_at,season,week,status,fails,warns,detail,note\n"
                                                    "2026-09-29T17:05:23Z,2026,4,WARN,0,1,status.practice: 0% have practice reports,\n")
        (self.root / "data").mkdir()
        self.files = {self.root / "data" / "model_pts.csv": b"season,week,gsis_id\n2026,3,A\n",
                      self.root / "data" / "projections.csv": b"x\n1\n"}
        for p, b in self.files.items():
            p.write_bytes(b)

    def tearDown(self):
        self.tmp.cleanup()

    def main_with(self, **patches) -> int:
        with mock.patch.object(SV, "ROOT", self.root), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with mock.patch.object(SV, "run", **patches):
                return SV.main([])

    def rows(self):
        return list(csv.DictReader(open(self.root / "logs" / "runs.csv", encoding="utf-8")))

    def check_untouched_and_warned(self, expect: str):
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual((rows[-1]["status"], rows[-1]["fails"], rows[-1]["warns"]), ("WARN", "0", "1"))
        self.assertTrue(rows[-1]["detail"].startswith("model.serve: serve failed, nothing written: " + expect), rows[-1]["detail"])
        self.assertEqual((rows[-1]["season"], rows[-1]["week"]), ("2026", "4"))        # next to the pipeline run it belongs to
        for p, b in self.files.items():
            self.assertEqual(p.read_bytes(), b)

    def test_any_exception_is_a_warn_row_nothing_written_and_exit_zero(self):
        self.assertEqual(self.main_with(side_effect=RuntimeError("could not fetch https://github.com/x: 503")), 0)
        self.check_untouched_and_warned("RuntimeError: could not fetch")

    def test_a_missing_library_is_the_same_thing(self):
        self.assertEqual(self.main_with(side_effect=ModuleNotFoundError("No module named 'lightgbm'")), 0)
        self.check_untouched_and_warned("ModuleNotFoundError: No module named 'lightgbm'")

    def test_bad_data_inside_the_real_run_path_is_caught_the_same_way(self):
        args_seen = []

        def boom(*a, **k):
            args_seen.append(a)
            raise KeyError("player_id")
        with mock.patch.object(SV, "ROOT", self.root), mock.patch.object(SV, "load_store", boom), \
                mock.patch.object(SV, "load_config", return_value={}), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            (self.root / "config.json").write_text('{"season": 2026}')
            rc = SV.main(["--no-refresh"])
        self.assertEqual(rc, 0)
        self.assertTrue(args_seen)
        self.check_untouched_and_warned("KeyError: 'player_id'")

    def test_the_warn_row_is_an_ordinary_integrity_row(self):
        self.main_with(side_effect=RuntimeError("x"))
        from ff import verify
        self.assertIsNotNone(verify.last_run_gap(self.root / "logs"))
        self.assertEqual(list(self.rows()[0].keys()), ["ran_at", "season", "week", "status", "fails", "warns", "detail", "note"])

    def test_even_if_the_log_cannot_be_written_the_step_exits_zero(self):
        with mock.patch("ff.verify.log_run", side_effect=OSError("disk full")):
            self.assertEqual(self.main_with(side_effect=RuntimeError("x")), 0)

    def test_a_completed_week_is_refused_without_a_dry_run(self):
        ns = argparse.Namespace(season=2026, week=3, now=None, dry_run=False, out=None, jobs=1, rebuild=False, no_refresh=True)
        with mock.patch.object(SV, "load_store", return_value=store()):
            with self.assertRaises(RuntimeError) as cm:
                SV.run(ns, root=REPO)
        self.assertIn("never rewritten", str(cm.exception))


# =========================================================================================== real data
class Unpublished(unittest.TestCase):
    def test_a_completed_game_without_stats_is_named_and_a_complete_week_is_quiet(self):
        self.assertEqual(SV.unpublished_games(store(), 2026, 4), [])
        pg = store()._tables["player_games"].df
        gid = sorted(store()._tables["game_results"].df.query("season == 2026 and week == 3")["game_id"])[0]
        cut = store().with_frame("player_games", pg[pg["game_id"] != gid])
        self.assertEqual(SV.unpublished_games(cut, 2026, 4), [gid])


class WeekSelection(unittest.TestCase):
    def test_the_week_follows_the_real_kickoffs(self):
        wk, tgs = SV.pick_week(store(), 2026, NOW)
        self.assertEqual(wk, 4)
        self.assertEqual(len(tgs), 2 * len({g.game_id for g in tgs}))
        self.assertTrue(all(g.kickoff > NOW for g in tgs))
        kicks = sorted({g.kickoff for g in tgs})
        mid = kicks[1] + pd.Timedelta(minutes=1)                        # Thursday night is under way
        wk2, tg2 = SV.pick_week(store(), 2026, mid)
        self.assertEqual(wk2, 4)
        self.assertLess(len([g for g in tg2 if g.kickoff > mid]), len(tg2))
        after = kicks[-1] + pd.Timedelta(minutes=1)
        self.assertEqual(SV.pick_week(store(), 2026, after)[0], 5)
        self.assertEqual(SV.pick_week(store(), 2026, pd.Timestamp("2027-03-01", tz="UTC"))[0], None)

    def test_a_sunday_morning_serve_is_still_week_5_and_only_games_still_to_kick_off_are_predicted(self):
        """The Phase 4 pre-lock run (Sunday 15:52 UTC): the real 2026 schedule says Thursday night, and any early international game, are
        under way, and the 1 pm slate is not. Asserted structurally (not on counts) so a later flex of a kickoff cannot break it."""
        sun = pd.Timestamp("2026-10-11T15:52:00Z")
        wk, tgs = SV.pick_week(store(), 2026, sun)
        self.assertEqual(wk, 5)
        games = {g.game_id: g.kickoff for g in tgs}
        upcoming = {i for i, k in games.items() if k > sun}
        under_way = set(games) - upcoming
        self.assertIn("2026_05_TB_DAL", under_way)                                         # Thursday night: frozen
        self.assertGreaterEqual(len(upcoming), 10)                                         # the 1 pm slate through Monday night: refreshed
        self.assertTrue(all(games[i] <= sun for i in under_way))
        self.assertEqual(len(tgs), 2 * len(games))
        wed = pd.Timestamp("2026-10-07T14:07:00Z")
        self.assertEqual(SV.pick_week(store(), 2026, wed)[0], 5)
        self.assertEqual(len({g.game_id for g in SV.pick_week(store(), 2026, wed)[1] if g.kickoff > wed}), len(games))   # Wednesday: every game
        tue = pd.Timestamp("2026-10-14T18:37:00Z")                                         # after Monday night: week 6
        self.assertEqual(SV.pick_week(store(), 2026, tue)[0], 6)

    def test_an_explicit_week_is_taken_as_asked(self):
        wk, tgs = SV.pick_week(store(), 2026, NOW, week=3)
        self.assertEqual((wk, len(tgs) > 0), (3, True))


class ServingRows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tgs = [g for g in tgs_of(2026, 4) if g.kickoff > NOW]
        cls.rows = SV.serve_frame(store(), cls.tgs)

    def test_rows_carry_the_training_schema_and_nothing_about_the_target_game(self):
        r = self.rows
        self.assertGreater(len(r), 500)
        self.assertEqual(set(r["week"]), {4})
        self.assertFalse(key(r).duplicated().any())
        X, _ = T.encode(r, T.feature_columns(r, SV.spec_from(cfg()), T.dead_columns(SV.join_frames(SV.training_frame(store(), 2026, 1), r))))
        self.assertEqual(list(X.columns), cfg()["inputs"])
        self.assertEqual(int(r["y_played"].sum()), 0)
        labels = [c for c in BF.LABELS if c not in ("y_played", "y_has_stats_row")]
        self.assertTrue(r[labels].isna().all().all())
        self.assertTrue((pd.to_datetime(r["kickoff_utc"], utc=True) > NOW).all())
        if Path(T.MATRIX).exists():
            self.assertEqual(sorted(c for c in matrix().columns if c != C.MISC), sorted(r.columns))    # the very same columns the models were trained on

    def test_nothing_a_serving_row_reads_is_stamped_at_or_after_its_kickoff(self):
        sample = []
        by_kick: dict = {}
        for tg in self.tgs:
            by_kick.setdefault(tg.kickoff, []).append(tg)
        self.assertGreaterEqual(len(by_kick), 3)                             # Thursday, Sunday early / late, Monday
        for kick, tgs in by_kick.items():
            sample += F.spine_for(store(), tgs[0])[:6]
        for t in sample:
            trace: list = []
            F.build_features(store(), t, trace=trace)
            self.assertGreater(len(trace), 10)
            for rec in trace:
                if rec["rows"]:
                    self.assertLess(rec["max_known_at"], rec["kickoff"], (t.player_id, rec["table"]))
                    self.assertLessEqual(rec["kickoff"], t.kickoff)

    def test_rebuilding_sampled_rows_from_a_store_cut_before_their_kickoff_changes_nothing(self):
        sample = []
        seen: set = set()
        for tg in self.tgs:
            if tg.kickoff not in seen and len(seen) < 4:
                seen.add(tg.kickoff)
                sample += F.spine_for(store(), tg)[:5]
        self.assertGreaterEqual(len(seen), 3)
        self.assertEqual(audit.audit_truncation(F.build_features, store(), sample), [])


class NoSkew(unittest.TestCase):
    def test_serving_rows_for_completed_weeks_are_the_matrixs_rows_bit_for_bit(self):
        m = matrix()
        dead = T.dead_columns(m)
        cols = [*T.feature_columns(m, SV.spec_from(cfg()), dead), *F.IDENTITY, "spine_src", "spine_depth_only", "lag1_week", "dvp_l4_games"]
        for season, week in ((2026, 3), (2025, 9), (2024, 14)):
            with self.subTest(season=season, week=week):
                served = SV.serve_frame(store(), tgs_of(season, week))
                want = m[(m["season"] == season) & (m["week"] == week)]
                self.assertGreater(len(want), 300)
                self.assertEqual(sorted(key(served)), sorted(key(want)))
                a, b = served.set_index(key(served)).sort_index(), want.set_index(key(want)).sort_index()
                bad = [c for c in cols if c in a.columns and c in b.columns and not same(a[c], b[c])]
                self.assertEqual(bad, [])

    def test_the_history_builder_gives_the_matrixs_2021_to_2025_rows_bit_for_bit(self):
        m = matrix()
        hist = SV.training_frame(store(), 2026, 1, jobs=4)
        want = m[m["season"] < 2026]
        self.assertEqual(len(hist), len(want))
        self.assertEqual(key(hist).tolist(), key(want).tolist())              # the same rows in the same order: the fit sees the same data
        bad = [c for c in [*T.feature_columns(m, SV.spec_from(cfg()), T.dead_columns(m)), *BF.LABELS, "base_trail3_ppr", "base_xfp_sameweek",
                           "spine_depth_only"] if not same(hist[c].reset_index(drop=True), want[c].reset_index(drop=True))]
        self.assertEqual(bad, [])

    def test_a_serve_of_a_completed_week_reproduces_the_backtest_predictions_for_that_week(self):
        season, week = 2026, 3
        m, c = matrix(), cfg()
        spec, params = SV.spec_from(c), T.load_params()
        leagues = L.load_leagues(REPO)
        hist = SV.training_frame(store(), season, week, jobs=4)
        served = SV.serve_frame(store(), tgs_of(season, week))
        df = SV.join_frames(hist, served, leagues)
        pred, comp, league_bands = SV.predict_week(df, season, week, c, leagues, log=lambda *_: None)
        sidx = df.index[len(hist):]
        skey = key(df.loc[sidx])
        mk = SV.add_league_targets(m, leagues)

        def ref(target=T.TARGET, lib="lightgbm", params_=None, quantile=None) -> pd.Series:
            r = B.walk_forward(mk, spec, lib, params_ if params_ is not None else params["point"][lib], season=season, weeks=(week,),
                               target=target, quantile=quantile)
            return pd.Series(r["pred"].to_numpy(), index=key(mk.loc[r.index]))

        def served_series(s: pd.Series) -> pd.Series:
            return pd.Series(s.loc[sidx].to_numpy(), index=skey)

        def equal(got: pd.Series, want: pd.Series, what: str) -> None:
            got, want = got.sort_index(), want.sort_index()
            self.assertEqual(list(got.index), list(want.index), what)
            self.assertTrue(np.array_equal(got.to_numpy(), want.to_numpy()), what)       # exactly equal, not close

        self.assertEqual(sorted(skey), sorted(key(m[(m["season"] == season) & (m["week"] == week)])))     # the same players in the same games
        # the flat models and their plain average
        flat = {lib: ref(lib=lib) for lib in T.LIBS}
        for lib in T.LIBS:
            equal(served_series(pred[f"lib_{lib}"]), flat[lib], lib)
        equal(served_series(pred["pts_model"]), pd.concat(list(flat.values()), axis=1).mean(axis=1), "average")
        # every component model, and PPR composed from them
        base = {k: v for k, v in params["point"]["lightgbm"].items() if k != "n_estimators"}
        ref_comp = {t: ref(target=t, params_={**base, "n_estimators": c["components"][t]["n_estimators"]}) for t in C.TARGETS}
        for t in C.TARGETS:
            equal(served_series(comp[t]), ref_comp[t], t)
        pos = mk.set_index(key(mk))["position"].reindex(ref_comp[C.TARGETS[0]].index)
        equal(served_series(pred["pts_model_components"]), C.compose(ref_comp, pos, C.PPR), "components under PPR")

        def ref_quantiles(target: str, season_: int, weeks) -> pd.DataFrame:
            res = None
            for a in B.QUANTILES:
                r = B.walk_forward(mk, spec, "lightgbm", params["quantile"][str(a)], season=season_, weeks=weeks, quantile=a,
                                   target=target).rename(columns={"pred": f"q{int(a * 100)}"})
                res = r if res is None else res.assign(**{f"q{int(a * 100)}": r[f"q{int(a * 100)}"]})
            return B.sort_quantiles(res)

        def check_band(got: pd.DataFrame, target: str, what: str, recal_target=None) -> None:
            q_hist = (B.sort_quantiles(B.walk_forward_quantiles(m, spec, params["quantile"], season=season - 1, weeks=B.WEEKS))
                      if target == T.TARGET else ref_quantiles(target, season - 1, B.WEEKS))
            q_cur = ref_quantiles(target, season, tuple(range(1, week + 1)))
            rec = E.recalibrate(q_cur.reset_index(drop=True), q_hist, mode=c["recalibration"]["mode"])
            rec.index = key(mk.loc[q_cur.index]).to_numpy()
            now_rows = q_cur["week"].to_numpy() == week
            for col, want_col in (("q10", "q10"), ("q50", "q50"), ("q90", "q90"), ("p10", "q10a"), ("p90", "q90a")):
                equal(served_series(got[col]), rec[want_col][now_rows], f"{what} {col}")
        # the PPR quantile bands and their recalibration, then each non-PPR league's own
        check_band(pred, T.TARGET, "ppr band")
        self.assertEqual(sorted(league_bands), ["we-can-think-of-something-funny", "where-you-at"])
        for slug, band in league_bands.items():
            check_band(band, SV.league_target(slug), f"{slug} band")
        # the published file's columns are all documented, and only non-PPR leagues carry bands of their own
        out, why = SV.assemble(store(), df.loc[sidx], pred, comp, leagues, root=REPO, season=season, week=week, served_at=NOW,
                               warnings=[], league_bands=league_bands)
        slugs = {s_.slug for s_ in leagues}
        undocumented = [x for x in out.columns if x not in SV.COLUMN_DOCS
                        and not any(x.startswith(p_) and x[len(p_):] in slugs for p_ in ("pts_", "sleeper_proj_", "e_pts_", "p10_", "p90_"))]
        self.assertEqual(undocumented, [])
        self.assertEqual(sorted(x for x in out.columns if x.startswith(("p10_", "p90_"))),
                         sorted(f"{p_}_{sl}" for sl in league_bands for p_ in ("p10", "p90")))
        self.assertTrue(set(out["gsis_id"]).isdisjoint(set(why.index)))


class BuilderJobs(unittest.TestCase):
    def test_forked_workers_build_the_same_rows_in_the_same_order(self):
        serial = BF.build([2025], store=store(), verbose=False, weeks=range(1, 3), jobs=1)
        forked = BF.build([2025], store=store(), verbose=False, weeks=range(1, 3), jobs=2)
        self.assertEqual(serial.shape, forked.shape)
        self.assertEqual(key(serial).tolist(), key(forked).tolist())
        for c in serial.columns:
            self.assertTrue(same(serial[c], forked[c]), c)


if __name__ == "__main__":
    unittest.main()
