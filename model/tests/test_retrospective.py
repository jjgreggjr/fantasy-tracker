"""The weekly retrospective: the classifier's rules, the inactive case, the partition replace and the failure contract.

    python -m unittest model.tests.test_retrospective

Synthetic frames in a temp directory (no network, nothing committed is touched). What is proven:

  * each classifier rule fires on a row built to hit it, in the documented priority order, with the secondary reason beside it
  * a player who did not play but was predicted above the bar is `inactive_missed`, carried apart from the scored misses
  * the per-week errors are the scoreboard's identical rows, deduped to one league per player in canonical order
  * re-running replaces a week's rows in data/model_misses.csv (no duplicates) and a rerun changes no byte of either output
  * a failure is one WARN row with check `model.retrospective`, exit 0, and nothing written
"""
from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from model import retrospective as R

SLUG = "lg-a"


def row(**kw) -> dict:
    """A big-miss row with volume as expected and nothing else going on; keyword overrides build each case."""
    base = {"error": -15.0, "position": "WR", "exp_vol": 10.0, "act_vol": 10.0, "exp_td": 0.3, "act_td": 0.0, "trail_snap_pct": 0.8,
            "snap_pct": 0.8, "margin": 3.0, "teammate_out_name": "", "played": True}
    base.update(kw)
    return base


class Classifier(unittest.TestCase):
    def test_td_variance_two_scores_on_expected_volume(self):
        self.assertEqual(R.classify(row(error=-14.0, act_td=2.0, exp_td=0.3)), ("td_variance", ""))

    def test_td_variance_zero_scores_on_heavy_volume(self):
        self.assertEqual(R.classify(row(error=13.0, position="RB", exp_vol=18.0, act_vol=19.0, exp_td=1.2, act_td=0.0)), ("td_variance", ""))

    def test_td_variance_is_only_secondary_when_volume_also_moved(self):
        self.assertEqual(R.classify(row(error=-20.0, exp_vol=8.0, act_vol=14.0, exp_td=0.0, act_td=2.0)), ("volume_up", "td_variance"))

    def test_volume_up_and_volume_down_follow_the_error_sign(self):
        self.assertEqual(R.classify(row(error=-13.0, exp_vol=8.0, act_vol=14.0)), ("volume_up", ""))
        self.assertEqual(R.classify(row(error=13.0, exp_vol=14.0, act_vol=6.0)), ("volume_down", ""))
        self.assertEqual(R.classify(row(error=13.0, exp_vol=8.0, act_vol=14.0)), ("unexplained", ""))     # more volume cannot explain an over-prediction

    def test_a_small_absolute_swing_is_not_a_volume_call(self):
        self.assertEqual(R.classify(row(error=-14.0, exp_vol=2.0, act_vol=3.0, act_td=2.0, exp_td=0.0)), ("td_variance", ""))   # +50% but one target

    def test_teammate_out_takes_precedence_over_volume_up(self):
        self.assertEqual(R.classify(row(error=-15.0, exp_vol=8.0, act_vol=16.0, teammate_out_name="Starter X")), ("teammate_out", "volume_up"))

    def test_left_early_on_a_collapsed_snap_share(self):
        self.assertEqual(R.classify(row(error=14.0, trail_snap_pct=0.85, snap_pct=0.3, exp_vol=12.0, act_vol=4.0)), ("left_early", "volume_down"))
        self.assertEqual(R.classify(row(error=-14.0, trail_snap_pct=0.85, snap_pct=0.3))[0], "unexplained")     # an under-prediction is not "left early"

    def test_blowout_script_needs_the_margin_and_a_volume_drop(self):
        self.assertEqual(R.classify(row(error=13.0, margin=-24.0, exp_vol=12.0, act_vol=8.0)), ("blowout_script", ""))
        self.assertEqual(R.classify(row(error=13.0, margin=-24.0, exp_vol=12.0, act_vol=6.0)), ("blowout_script", "volume_down"))
        self.assertEqual(R.classify(row(error=13.0, margin=-3.0, exp_vol=12.0, act_vol=8.0)), ("unexplained", ""))

    def test_inactive_beats_everything(self):
        self.assertEqual(R.classify(row(error=9.0, played=False, exp_vol=12.0, act_vol=np.nan)), ("inactive_missed", ""))

    def test_no_trailing_record_is_unexplained_not_an_error(self):
        self.assertEqual(R.classify(row(error=-20.0, exp_vol=np.nan, exp_td=np.nan, trail_snap_pct=np.nan, act_td=3.0)), ("unexplained", ""))


# --------------------------------------------------------------------------- a synthetic repo
def make_root() -> Path:
    """One completed week (2026 wk 4) in two leagues (111 = the canonical league, 222 another), six players, and a frozen week 5 with
    no lineups yet. Players: a (RB, TD variance), b (WR, volume up with his starter out), c (WR, inactive, predicted 9), d (QB, fine),
    e (TE, fine), f (RB, rostered only in league 222, left early)."""
    root = Path(tempfile.mkdtemp())
    (root / "data").mkdir()
    for slug, lid in ((R.CANONICAL, "111"), (SLUG, "222")):
        (root / "leagues" / slug).mkdir(parents=True)
        (root / "leagues" / slug / "league.json").write_text(json.dumps({"slug": slug, "league_id": lid, "name": slug}))
    players = {"a": ("RB", "DAL"), "b": ("WR", "DAL"), "c": ("WR", "PHI"), "d": ("QB", "DAL"), "e": ("TE", "PHI"), "f": ("RB", "PHI"), "x": ("WR", "DAL")}
    pred = {"a": 12.0, "b": 6.0, "c": 9.0, "d": 18.0, "e": 7.0, "f": 16.0}
    actual = {"a": 30.0, "b": 24.0, "c": 0.0, "d": 19.0, "e": 8.0, "f": 2.0}
    mp = []
    for wk in (4, 5):
        for g, p in pred.items():
            r = {"season": 2026, "week": wk, "gsis_id": g, "name": g.upper(), "position": players[g][0], "team": players[g][1],
                 "opponent": "PHI" if players[g][1] == "DAL" else "DAL", "pts_model": p, "pts_model_components": p}
            for slug in (R.CANONICAL, SLUG):
                r[f"pts_{slug}"] = p
                r[f"sleeper_proj_{slug}"] = p + 1.0
                r[f"e_pts_{slug}"] = p - 1.0
            mp.append(r)
    lp = []
    for g in ("a", "b", "c", "d", "e"):
        lp.append({"season": 2026, "week": 4, "league_id": "111", "league_name": "L1", "roster_id": 1, "gsis_id": g, "name": g.upper(),
                   "position": players[g][0], "points": actual[g], "started": 1})
    for g in ("a", "f", "x"):                                            # a is rostered in both leagues; f only here; x is the starter who sat
        lp.append({"season": 2026, "week": 4, "league_id": "222", "league_name": "L2", "roster_id": 1, "gsis_id": g, "name": g.upper(),
                   "position": players[g][0], "points": actual.get(g, 0.0), "started": 1})
    pw = []
    for wk in (1, 2, 3):                                                 # trailing record
        pw += [{"season": 2026, "week": wk, "gsis_id": "a", "name": "A", "position": "RB", "team": "DAL", "opponent": "NYG", "snaps": 50, "snap_pct": 0.8,
                "opps": 18, "carries": 15, "targets": 3, "rec_tds": 0, "rush_tds": 0, "pass_tds": 0, "pass_att": 0},
               {"season": 2026, "week": wk, "gsis_id": "b", "name": "B", "position": "WR", "team": "DAL", "opponent": "NYG", "snaps": 30, "snap_pct": 0.5,
                "opps": 4, "carries": 0, "targets": 4, "rec_tds": 0, "rush_tds": 0, "pass_tds": 0, "pass_att": 0},
               {"season": 2026, "week": wk, "gsis_id": "x", "name": "X", "position": "WR", "team": "DAL", "opponent": "NYG", "snaps": 60, "snap_pct": 0.9,
                "opps": 10, "carries": 0, "targets": 10, "rec_tds": 0, "rush_tds": 0, "pass_tds": 0, "pass_att": 0},
               {"season": 2026, "week": wk, "gsis_id": "d", "name": "D", "position": "QB", "team": "DAL", "opponent": "NYG", "snaps": 65, "snap_pct": 1.0,
                "opps": 3, "carries": 3, "targets": 0, "rec_tds": 0, "rush_tds": 0, "pass_tds": 2, "pass_att": 33},
               {"season": 2026, "week": wk, "gsis_id": "e", "name": "E", "position": "TE", "team": "PHI", "opponent": "NYG", "snaps": 50, "snap_pct": 0.8,
                "opps": 5, "carries": 0, "targets": 5, "rec_tds": 0, "rush_tds": 0, "pass_tds": 0, "pass_att": 0},
               {"season": 2026, "week": wk, "gsis_id": "f", "name": "F", "position": "RB", "team": "PHI", "opponent": "NYG", "snaps": 55, "snap_pct": 0.85,
                "opps": 16, "carries": 14, "targets": 2, "rec_tds": 0, "rush_tds": 0, "pass_tds": 0, "pass_att": 0}]
    pw += [{"season": 2026, "week": 4, "gsis_id": "a", "name": "A", "position": "RB", "team": "DAL", "opponent": "PHI", "snaps": 52, "snap_pct": 0.82,
            "opps": 19, "carries": 16, "targets": 3, "rec_tds": 1, "rush_tds": 2, "pass_tds": 0, "pass_att": 0},
           {"season": 2026, "week": 4, "gsis_id": "b", "name": "B", "position": "WR", "team": "DAL", "opponent": "PHI", "snaps": 62, "snap_pct": 0.95,
            "opps": 13, "carries": 0, "targets": 13, "rec_tds": 1, "rush_tds": 0, "pass_tds": 0, "pass_att": 0},
           {"season": 2026, "week": 4, "gsis_id": "d", "name": "D", "position": "QB", "team": "DAL", "opponent": "PHI", "snaps": 65, "snap_pct": 1.0,
            "opps": 3, "carries": 3, "targets": 0, "rec_tds": 0, "rush_tds": 0, "pass_tds": 2, "pass_att": 34},
           {"season": 2026, "week": 4, "gsis_id": "e", "name": "E", "position": "TE", "team": "PHI", "opponent": "DAL", "snaps": 50, "snap_pct": 0.8,
            "opps": 5, "carries": 0, "targets": 5, "rec_tds": 0, "rush_tds": 0, "pass_tds": 0, "pass_att": 0},
           {"season": 2026, "week": 4, "gsis_id": "f", "name": "F", "position": "RB", "team": "PHI", "opponent": "DAL", "snaps": 12, "snap_pct": 0.2,
            "opps": 3, "carries": 3, "targets": 0, "rec_tds": 0, "rush_tds": 0, "pass_tds": 0, "pass_att": 0}]
    # c and x have no week-4 stats row: c did not play (rostered, 0 points), x is the DAL WR1 who sat
    dc = [{"season": 2026, "week": 4, "team": "DAL", "position": "WR", "rank": 1, "gsis_id": "x", "name": "X"},
          {"season": 2026, "week": 4, "team": "DAL", "position": "WR", "rank": 2, "gsis_id": "b", "name": "B"},
          {"season": 2026, "week": 4, "team": "DAL", "position": "RB", "rank": 1, "gsis_id": "a", "name": "A"},
          {"season": 2026, "week": 4, "team": "DAL", "position": "QB", "rank": 1, "gsis_id": "d", "name": "D"},
          {"season": 2026, "week": 4, "team": "DAL", "position": "QB", "rank": 2, "gsis_id": "q2", "name": "Q2"},
          {"season": 2026, "week": 4, "team": "PHI", "position": "RB", "rank": 1, "gsis_id": "f", "name": "F"}]
    dvp = [{"season": 2026, "week": 4, "defense": "DAL", "position": "ALL", "points_allowed": 10.0},
           {"season": 2026, "week": 4, "defense": "PHI", "position": "ALL", "points_allowed": 31.0}]
    ev = []
    for wk, acc in ((3, 0.66), (4, 0.70)):
        for c in ("model_components", "model_flat", "model_blend", "sleeper", "e_pts"):
            ev.append({"season": 2026, "week": wk, "league": R.CANONICAL, "position": "ALL", "comparator": c, "n": 10, "spearman": 0.4,
                       "credit": acc * 100, "pairs": 100, "pick_acc": acc})
    for name, rows in (("model_pts", mp), ("lineups_played", lp), ("player_weeks", pw), ("depth_charts", dc), ("defense_vs_pos", dvp), ("model_eval", ev)):
        pd.DataFrame(rows).to_csv(root / "data" / f"{name}.csv", index=False)
    (root / "model").mkdir()
    (root / "model" / "serving_config.json").write_text(json.dumps({"inputs": ["opps_l1", "tm_out_above_n"]}))
    return root


class Evaluate(unittest.TestCase):
    def test_errors_are_the_scoreboards_rows_deduped_to_the_canonical_league(self):
        root = make_root()
        inp = R.load_inputs(root)
        e = R.week_errors(inp, 2026, 4)
        played = e[e["played"]]
        a = played[(played["gsis_id"] == "a") & (played["comparator"] == "model_components")]
        self.assertEqual(sorted(a["league"]), sorted([SLUG, R.CANONICAL]))         # a is scored in both leagues...
        d = R.dedupe(e)
        a1 = d[(d["gsis_id"] == "a") & (d["comparator"] == "model_components")]
        self.assertEqual(list(a1["league"]), [R.CANONICAL])                         # ...and kept once, from the canonical league
        f1 = d[(d["gsis_id"] == "f") & (d["comparator"] == "sleeper")]
        self.assertEqual(list(f1["league"]), [SLUG])                                # f is only rostered in the other league
        self.assertAlmostEqual(float(a1["error"].iloc[0]), 12.0 - 30.0)
        self.assertFalse((d["gsis_id"] == "x").any())                               # x did not play and was never predicted: nowhere
        self.assertEqual(R.completed_weeks(inp), [(2026, 4)])                        # week 5 is frozen but has no lineups

    def test_the_reasons_and_the_inactive_case(self):
        root = make_root()
        misses, q, inp = R.evaluate(root)
        m = misses[misses["comparator"] == "model_components"].set_index("gsis_id")
        self.assertEqual(m.loc["a", "reason"], "td_variance")                         # 19 opps on 18 expected, 3 TDs on 0
        self.assertEqual((m.loc["a", "exp_vol"], m.loc["a", "act_vol"], m.loc["a", "act_td"]), (18.0, 19.0, 3.0))
        self.assertEqual((m.loc["b", "reason"], m.loc["b", "secondary_reason"]), ("teammate_out", "volume_up"))
        self.assertEqual(m.loc["b", "teammate_out_name"], "X")                        # the WR1 above him sat
        self.assertEqual(m.loc["f", "reason"], "left_early")
        self.assertAlmostEqual(float(m.loc["f", "margin"]), -21.0)                    # PHI allowed 31, DAL allowed 10: PHI lost by 21
        self.assertEqual(m.loc["c", "reason"], "inactive_missed")
        self.assertFalse(bool(m.loc["c", "played"]))
        self.assertEqual(float(m.loc["c", "actual"]), 0.0)
        self.assertNotIn("d", m.index)                                                 # a 1-point miss is not a miss
        self.assertNotIn("e", m.index)
        # d's QB2 was never in the stats file: untracked, not "out"
        self.assertEqual(set(misses["comparator"]), set(R.COMPARATORS))
        shares = R.reason_shares(misses)["model_components"]
        self.assertEqual((shares["inactive_missed"], shares["td_variance"], shares["teammate_out"], shares["left_early"]), (1, 1, 1, 1))
        self.assertAlmostEqual(q[(2026, 4)]["model_components"], np.percentile([18.0, 18.0, 1.0, 1.0, 14.0], 95))

    def test_the_report_derives_its_tables_from_the_inputs(self):
        root = make_root()
        misses, q, inp = R.evaluate(root)
        text = R.render(misses, q, inp, R.frozen_inputs(root / "model" / "serving_config.json"))
        self.assertIn("| `model_components` | 0.660 | 0.700 | 0.680 | 0.028 |", text)      # drift: by week, mean, week sd
        self.assertIn("too few to call a trend", text)
        self.assertIn("Top 15 misses, 2026 week 4", text)
        self.assertIn("| B | WR | DAL | gooma-s-family-league | 6.0 | 24.0 | -18.0 | `teammate_out` | `volume_up` |", text)
        self.assertIn("| C | WR | PHI | gooma-s-family-league | 9.0 |", text)                  # the did-not-play list
        self.assertIn("served: `tm_out_above_n`", text)                                         # from the config's inputs...
        self.assertIn("pruned: `tm_out_group_opps`", text)                                      # ...and the registry
        self.assertIn("not built: `opps_trend`", text)

    def test_drift_trend_sentence_over_enough_weeks(self):
        ev = pd.DataFrame([{"season": 2026, "week": w, "league": "x", "position": "ALL", "comparator": c, "n": 1, "spearman": 0.1,
                            "credit": a * 100, "pairs": 100, "pick_acc": a}
                           for c, accs in (("model_components", [0.60, 0.65, 0.70, 0.75, 0.80]), ("sleeper", [0.70, 0.69, 0.71, 0.70, 0.70]))
                           for w, a in enumerate(accs, start=1)])
        s = R.trend_sentence(R.drift_table(ev))
        self.assertIn("`model_components` is rising", s)
        self.assertNotIn("sleeper", s)


class Files(unittest.TestCase):
    def test_rerun_replaces_the_partition_and_changes_no_byte(self):
        root = make_root()
        R.run(root)
        path, rep = root / "data" / "model_misses.csv", root / "model" / "reports" / "retrospective.md"
        first, text1 = path.read_bytes(), rep.read_text()
        R.run(root)
        self.assertEqual(path.read_bytes(), first)
        self.assertEqual(rep.read_text(), text1)
        m = pd.read_csv(path)
        self.assertFalse(m.duplicated(R.KEYS).any())
        self.assertEqual(list(m.columns), R.MISS_COLS)
        # a stale row of the week is replaced; a row of another week stays
        m.loc[m["gsis_id"] == "a", "reason"] = "stale"
        stale = m.iloc[[0]].assign(season=2025, week=17, reason="kept")
        pd.concat([m, stale]).to_csv(path, index=False)
        R.run(root)
        back = pd.read_csv(path)
        self.assertFalse((back["reason"] == "stale").any())
        self.assertEqual(int((back["reason"] == "kept").sum()), 1)
        self.assertFalse(back.duplicated(R.KEYS).any())

    def test_nothing_to_do_writes_the_report_but_no_misses_file(self):
        root = make_root()
        (root / "data" / "lineups_played.csv").unlink()
        text = R.run(root)
        self.assertIn("Nothing to post-mortem yet", text)
        self.assertFalse((root / "data" / "model_misses.csv").exists())
        self.assertTrue((root / "model" / "reports" / "retrospective.md").exists())

    def test_a_failure_is_a_warn_row_and_writes_nothing(self):
        root = make_root()
        (root / "data" / "lineups_played.csv").write_text("season,week\n2026,4\n")           # no league_id / position columns
        (root / "logs").mkdir()
        (root / "logs" / "runs.csv").write_text("ran_at,season,week,status,fails,warns,detail,note\n"
                                                "2026-09-29T00:00:00Z,2026,4,OK,0,0,,\n")
        with mock.patch.object(R, "ROOT", root), mock.patch("model.serve.ROOT", root), redirect_stdout(io.StringIO()), \
                redirect_stderr(io.StringIO()):
            rc = R.main([])
        self.assertEqual(rc, 0)
        with open(root / "logs" / "runs.csv") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 2)
        self.assertEqual((rows[-1]["status"], rows[-1]["fails"], rows[-1]["warns"]), ("WARN", "0", "1"))
        self.assertTrue(rows[-1]["detail"].startswith("model.retrospective: retrospective failed"))
        self.assertEqual((rows[-1]["season"], rows[-1]["week"]), ("2026", "4"))
        self.assertFalse((root / "data" / "model_misses.csv").exists())
        self.assertFalse((root / "model" / "reports" / "retrospective.md").exists())


if __name__ == "__main__":
    unittest.main()
