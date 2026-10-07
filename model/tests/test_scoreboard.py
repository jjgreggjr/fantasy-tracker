"""The live scoreboard: what it scores, on which rows, and what the adoption gate says.

    python3.12 -m unittest model.tests.test_scoreboard

Synthetic leagues, frozen model rows and played lineups in a temp directory (no network, nothing committed is touched), with
answers worked out by hand. What is proven:

  * the five comparators are built from the frozen columns exactly as documented, in the league's own scoring
  * pick accuracy and Spearman match hand computations, every comparator is scored on the IDENTICAL rows, and only players who
    played are scored
  * the gate needs 6 weeks AND a higher pooled pick accuracy than Sleeper: too few weeks is never MET, a model that loses is
    never MET, a model that wins over 6 weeks is MET; the gate sentence is in the report header verbatim
  * re-scoring replaces a week's rows in data/model_eval.csv (no duplicates), a rerun changes no byte, and the report has no clock
  * a failure is a WARN row with check `model.scoreboard` and exit 0, and writes nothing
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

from model import scoreboard as S

SLUG = "lg-a"


def make_root(weeks: int = 1, model_wins: bool = True, extra_rows: bool = True) -> Path:
    """A temp repo: one league (id 111), `weeks` completed weeks. Eight RBs and eight WRs per week whose actual points are
    8..1 in player order; the model predicts them in the right order (or reversed), Sleeper in a noisy order."""
    root = Path(tempfile.mkdtemp())
    (root / "data").mkdir()
    (root / "leagues" / SLUG).mkdir(parents=True)
    (root / "leagues" / SLUG / "league.json").write_text(json.dumps({"slug": SLUG, "league_id": "111", "name": "League A"}))
    mp, lp, pw = [], [], []
    for wk in range(1, weeks + 1):
        for pos, tag in (("RB", "r"), ("WR", "w")):
            for i in range(8):
                gid = f"{tag}{i}"
                actual = float(8 - i)
                good = actual if model_wins else -actual
                noisy = [5.0, 1.0, 7.0, 3.0, 8.0, 2.0, 6.0, 4.0][i]
                mp.append({"season": 2026, "week": wk, "gsis_id": gid, f"pts_{SLUG}": good, "pts_model": good + 0.5,
                           "pts_model_components": good - 0.5, f"sleeper_proj_{SLUG}": noisy, f"e_pts_{SLUG}": actual * 0.9})
                lp.append({"season": 2026, "week": wk, "league_id": "111", "league_name": "League A", "roster_id": 1,
                           "gsis_id": gid, "position": pos, "points": actual, "started": 1})
                pw.append({"season": 2026, "week": wk, "gsis_id": gid})
    pd.DataFrame(mp).to_csv(root / "data" / "model_pts.csv", index=False)
    pd.DataFrame(lp).to_csv(root / "data" / "lineups_played.csv", index=False)
    pd.DataFrame(pw).to_csv(root / "data" / "player_weeks.csv", index=False)
    return root


class Comparators(unittest.TestCase):
    def test_built_from_the_frozen_columns_in_the_leagues_scoring(self):
        mp = pd.DataFrame({"gsis_id": ["a", "b"], "pts_model": [10.0, 8.0], "pts_model_components": [9.0, 8.0], f"pts_{SLUG}": [9.5, 8.0],
                           f"sleeper_proj_{SLUG}": [11.0, 7.0], f"e_pts_{SLUG}": [12.0, np.nan]})
        c = S.comparators(mp, SLUG).set_index("gsis_id")
        self.assertEqual(c.loc["a", "model_components"], 9.5)
        self.assertEqual(c.loc["a", "model_flat"], 10.0 + (9.5 - 9.0))          # PPR average + the league's scoring difference
        self.assertEqual(c.loc["a", "model_blend"], 0.5 * 10.5 + 0.5 * 9.5)     # the 50/50 of flat and components, both in league scoring
        self.assertEqual(c.loc["b", "model_flat"], 8.0)                         # no difference: identical to pts_model
        self.assertEqual((c.loc["a", "sleeper"], c.loc["a", "e_pts"]), (11.0, 12.0))
        self.assertTrue(np.isnan(c.loc["b", "e_pts"]))

    def test_a_missing_column_is_nan_not_an_error(self):
        c = S.comparators(pd.DataFrame({"gsis_id": ["a"], "pts_model": [1.0]}), SLUG)
        self.assertTrue(c[["model_components", "sleeper", "e_pts"]].isna().all().all())


class Metrics(unittest.TestCase):
    def test_pick_accuracy_and_spearman_match_a_hand_computation(self):
        # four RBs, actual 10, 8, 6, 2. `p` ranks them 10 > 6 > 8 > 2: one of the six pairs is flipped (8 vs 6).
        e = pd.DataFrame({"position": "RB", "actual": [10.0, 8.0, 6.0, 2.0],
                          **{c: [10.0, 8.0, 6.0, 2.0] for c in S.COMPARATORS}})
        e["sleeper"] = [4.0, 2.0, 3.0, 1.0]
        rows = S.score_rows(e, 2026, 4, SLUG).set_index(["position", "comparator"])
        perfect, flipped = rows.loc[("RB", "e_pts")], rows.loc[("RB", "sleeper")]
        self.assertEqual((perfect["credit"], perfect["pairs"], perfect["spearman"]), (6.0, 6, 1.0))
        self.assertEqual((flipped["credit"], flipped["pairs"]), (5.0, 6))                  # 5 of 6 pairs right
        self.assertAlmostEqual(flipped["pick_acc"], 5 / 6)
        self.assertAlmostEqual(flipped["spearman"], 0.8)                                   # ranks 4,2,3,1 vs 4,3,2,1: 1 - 6*2/(4*15)
        self.assertEqual(rows.loc[("ALL", "sleeper"), "pairs"], 6)                         # other positions have no rows, contribute nothing
        self.assertEqual(rows.loc[("ALL", "sleeper"), "n"], 4)

    def test_a_tied_prediction_counts_half_and_equal_actuals_are_not_a_pair(self):
        e = pd.DataFrame({"position": "WR", "actual": [5.0, 5.0, 3.0], **{c: [1.0, 1.0, 1.0] for c in S.COMPARATORS}})
        r = S.score_rows(e, 2026, 4, SLUG).set_index(["position", "comparator"]).loc[("WR", "e_pts")]
        self.assertEqual((r["pairs"], r["credit"]), (2, 1.0))                              # (5,3) twice, each tied: 0.5 + 0.5


class Evaluate(unittest.TestCase):
    def test_only_players_who_played_are_scored_and_every_comparator_sees_the_same_rows(self):
        root = make_root(weeks=1)
        lp = pd.read_csv(root / "data" / "lineups_played.csv")
        pw = pd.read_csv(root / "data" / "player_weeks.csv")
        mp = pd.read_csv(root / "data" / "model_pts.csv")
        pw = pw[pw["gsis_id"] != "r7"]                                     # r7 has no stats row and 1 point: not scored as played...
        lp.loc[lp["gsis_id"] == "r7", "points"] = 0.0                      # ...because he scored nothing either
        lp.loc[lp["gsis_id"] == "w7", "points"] = 0.0                      # w7 has a stats row (played) with 0 points: scored
        mp.loc[mp["gsis_id"] == "r6", f"e_pts_{SLUG}"] = np.nan            # r6 lacks E_pts: dropped from ALL comparators
        for name, df in (("lineups_played", lp), ("player_weeks", pw), ("model_pts", mp)):
            df.to_csv(root / "data" / f"{name}.csv", index=False)
        ev, cov, avail = S.evaluate(root)
        rb = ev[(ev["position"] == "RB")]
        self.assertEqual(set(rb["n"]), {6})                                # 8 RBs - r7 (did not play) - r6 (no E_pts)
        self.assertEqual(set(ev[(ev["position"] == "WR")]["n"]), {8})
        self.assertEqual(cov.iloc[0]["played"], 15)
        self.assertEqual(cov.iloc[0]["scored"], 14)
        self.assertEqual(int(avail.iloc[0]["e_pts"]), 15)

    def test_a_week_with_no_recorded_lineups_is_not_scored(self):
        root = make_root(weeks=2)
        lp = pd.read_csv(root / "data" / "lineups_played.csv")
        lp[lp["week"] == 1].to_csv(root / "data" / "lineups_played.csv", index=False)
        ev, cov, avail = S.evaluate(root)
        self.assertEqual(sorted(ev["week"].unique()), [1])
        self.assertEqual(sorted(avail["week"]), [1, 2])                    # the availability table still says week 2 is frozen

    def test_no_model_rows_no_results(self):
        root = make_root()
        (root / "data" / "model_pts.csv").unlink()
        ev, cov, avail = S.evaluate(root)
        self.assertTrue(ev.empty)


class Gate(unittest.TestCase):
    def gate(self, **kw):
        root = make_root(**kw)
        ev, *_ = S.evaluate(root)
        return S.gate_status(ev), ev

    def test_six_weeks_and_a_better_pick_accuracy_meets_it(self):
        g, _ = self.gate(weeks=6, model_wins=True)
        self.assertEqual(g["weeks"], 6)
        self.assertGreater(g["model"], g["sleeper"])
        self.assertTrue(g["met"])

    def test_five_weeks_never_meets_it_however_good_the_model(self):
        g, _ = self.gate(weeks=5, model_wins=True)
        self.assertEqual(g["weeks"], 5)
        self.assertGreater(g["diff"], 0)
        self.assertFalse(g["met"])

    def test_a_model_that_loses_to_sleeper_does_not_meet_it_over_six_weeks(self):
        g, _ = self.gate(weeks=6, model_wins=False)
        self.assertLess(g["model"], g["sleeper"])
        self.assertFalse(g["met"])

    def test_the_report_header_carries_the_gate_sentence_verbatim(self):
        root = make_root(weeks=2)
        text = S.run(root)
        self.assertIn(S.GATE, text.splitlines()[2].strip("*"))
        self.assertIn("2 of 6 completed weeks scored", text)
        self.assertIn("NOT MET", text)
        self.assertIn("cannot be met before 6 weeks", text)


class TwoSeasons(unittest.TestCase):
    def test_the_same_week_number_in_two_seasons_is_two_rows_never_one(self):
        ev = pd.concat([S.score_rows(pd.DataFrame({"position": "RB", "actual": [9.0, 5.0, 2.0], **{c: [9.0, 5.0, 2.0] for c in S.COMPARATORS}}),
                                     season, 4, SLUG) for season in (2026, 2027)], ignore_index=True)
        ev.loc[ev["season"] == 2027, "credit"] = 0.0                              # 2027 wk4: not one pair right
        by_week = S._by(ev, "week").splitlines()
        self.assertEqual([l.split("|")[1].strip() for l in by_week[2:]], ["2026 wk04", "2027 wk04"])
        self.assertEqual(by_week[2].split("|")[2].strip(), "1.000")
        self.assertEqual(by_week[3].split("|")[2].strip(), "0.000")
        cov = pd.DataFrame({"season": [2026, 2027], "week": [4, 4], "league": SLUG, "played": [10, 20], "with_model_row": [9, 19], "scored": [8, 18]})
        text = S.render(ev, cov, pd.DataFrame(), S.gate_status(ev))
        self.assertIn("| 2026 | 4 | 10 | 9 | 8 |", text)
        self.assertIn("| 2027 | 4 | 20 | 19 | 18 |", text)
        self.assertEqual(S.gate_status(ev)["weeks"], 2)                          # (2026, 4) and (2027, 4) are two weeks


class Files(unittest.TestCase):
    def test_rescoring_replaces_a_weeks_rows_and_a_rerun_changes_no_byte(self):
        root = make_root(weeks=2)
        S.run(root)
        first = (root / "data" / "model_eval.csv").read_bytes()
        rep1 = (root / "model" / "reports" / "live_scoreboard.md").read_text()
        S.run(root)
        self.assertEqual((root / "data" / "model_eval.csv").read_bytes(), first)
        self.assertEqual((root / "model" / "reports" / "live_scoreboard.md").read_text(), rep1)     # no clock in the report
        ev = pd.read_csv(root / "data" / "model_eval.csv")
        self.assertFalse(ev.duplicated(S.KEYS).any())
        self.assertEqual(sorted(ev["week"].unique()), [1, 2])
        ev.loc[ev["week"] == 1, "pick_acc"] = -1.0
        ev.to_csv(root / "data" / "model_eval.csv", index=False)
        S.run(root)                                                   # re-scored: week 1 is replaced by the recomputed values
        back = pd.read_csv(root / "data" / "model_eval.csv")
        self.assertTrue((back["pick_acc"].dropna() > -1).all())
        # a week the inputs no longer hold stays in the record (it accumulates; only re-scored weeks are replaced)
        lp = pd.read_csv(root / "data" / "lineups_played.csv")
        lp[lp["week"] == 2].to_csv(root / "data" / "lineups_played.csv", index=False)
        back.loc[back["week"] == 1, "pick_acc"] = -2.0
        back.to_csv(root / "data" / "model_eval.csv", index=False)
        S.run(root)
        kept = pd.read_csv(root / "data" / "model_eval.csv")
        self.assertEqual(sorted(kept["week"].unique()), [1, 2])
        self.assertTrue((kept[kept["week"] == 1]["pick_acc"].dropna() == -2.0).all())

    def test_three_serves_of_the_week_in_progress_never_double_the_scoreboards_rows(self):
        """Phase 4 adds a Sunday run, so a week is now served three times (Wednesday, Friday, Sunday) and the scoreboard runs in the same step
        every time. The finished week is re-scored on each of them, and the week being served has model rows but no recorded lineups: it
        must add nothing to data/model_eval.csv, and the finished week's rows must come out byte for byte the same."""
        from model import serve as SV
        root = make_root(weeks=1)
        mp = pd.read_csv(root / "data" / "model_pts.csv")
        mp = mp.assign(game_id="2026_01_A_B", kickoff_utc="2026-09-13T17:00:00+00:00")
        mp.to_csv(root / "data" / "model_pts.csv", index=False)
        kick = {"2026_02_A_B": "2026-09-20T17:00:00+00:00", "2026_02_C_D": "2026-09-21T00:20:00+00:00"}
        seen = []
        for when in ("2026-09-16T14:07:00Z", "2026-09-18T22:11:00Z", "2026-09-20T15:52:00Z"):
            upcoming = {g: k for g, k in kick.items() if pd.Timestamp(k) > pd.Timestamp(when)}
            new = pd.DataFrame([{"season": 2026, "week": 2, "game_id": g, "kickoff_utc": k, "gsis_id": f"{g}{i}", f"pts_{SLUG}": 5.0 + i,
                                 "pts_model": 5.0, "pts_model_components": 5.0, f"sleeper_proj_{SLUG}": 4.0, f"e_pts_{SLUG}": 3.0}
                                for g, k in upcoming.items() for i in range(3)])
            SV.write_model_pts(root / "data" / "model_pts.csv", new, 2026, 2, pd.Timestamp(when))
            S.run(root)
            seen.append(((root / "data" / "model_eval.csv").read_bytes(), (root / "model" / "reports" / "live_scoreboard.md").read_text()))
        self.assertEqual(seen[0], seen[1])
        self.assertEqual(seen[1], seen[2])                                                   # nothing moved, nothing doubled
        ev = pd.read_csv(root / "data" / "model_eval.csv")
        self.assertFalse(ev.duplicated(S.KEYS).any())
        self.assertEqual(sorted(ev["week"].unique()), [1])                                   # week 2 has no recorded lineups: not scored, no row
        mpf = pd.read_csv(root / "data" / "model_pts.csv")
        self.assertFalse(mpf.duplicated(["season", "week", "gsis_id"]).any())
        self.assertEqual(len(mpf[mpf["week"] == 2]), 3 * len(kick))                          # the Sunday serve left the frozen game's rows in place
        # the week completes: it is scored once, by whichever run first sees its lineups, and a later run does not double it
        lp = pd.read_csv(root / "data" / "lineups_played.csv")
        pw = pd.read_csv(root / "data" / "player_weeks.csv")
        week2 = [{"season": 2026, "week": 2, "league_id": 111, "league_name": "League A", "roster_id": 1, "gsis_id": f"{g}{i}",
                  "position": "RB" if i < 2 else "WR", "points": float(10 - 3 * i), "started": 1} for g in kick for i in range(3)]
        pd.concat([lp, pd.DataFrame(week2)], ignore_index=True).to_csv(root / "data" / "lineups_played.csv", index=False)
        pd.concat([pw, pd.DataFrame([{"season": 2026, "week": 2, "gsis_id": r["gsis_id"]} for r in week2])], ignore_index=True).to_csv(root / "data" / "player_weeks.csv", index=False)
        S.run(root)
        first = (root / "data" / "model_eval.csv").read_bytes()
        S.run(root)
        self.assertEqual((root / "data" / "model_eval.csv").read_bytes(), first)
        ev = pd.read_csv(root / "data" / "model_eval.csv")
        self.assertEqual(sorted(ev["week"].unique()), [1, 2])
        self.assertFalse(ev.duplicated(S.KEYS).any())

    def test_nothing_to_score_writes_a_report_but_no_eval_file(self):
        root = make_root()
        (root / "data" / "model_pts.csv").unlink()
        text = S.run(root)
        self.assertIn("not started", text)
        self.assertFalse((root / "data" / "model_eval.csv").exists())
        self.assertIn(S.GATE, text)

    def test_a_failure_is_a_warn_row_and_writes_nothing(self):
        root = make_root()
        (root / "data" / "lineups_played.csv").write_text("season,week\n2026,1\n")          # no league_id / position columns
        (root / "logs").mkdir()
        (root / "logs" / "runs.csv").write_text("ran_at,season,week,status,fails,warns,detail,note\n"
                                                "2026-09-29T00:00:00Z,2026,4,OK,0,0,,\n")
        with mock.patch.object(S, "ROOT", root), mock.patch("model.serve.ROOT", root), redirect_stdout(io.StringIO()), \
                redirect_stderr(io.StringIO()):
            rc = S.main([])
        self.assertEqual(rc, 0)
        with open(root / "logs" / "runs.csv") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 2)
        self.assertEqual((rows[-1]["status"], rows[-1]["fails"], rows[-1]["warns"]), ("WARN", "0", "1"))
        self.assertTrue(rows[-1]["detail"].startswith("model.scoreboard: scoreboard failed"))
        self.assertEqual((rows[-1]["season"], rows[-1]["week"]), ("2026", "4"))
        self.assertFalse((root / "data" / "model_eval.csv").exists())
        self.assertFalse((root / "model" / "reports" / "live_scoreboard.md").exists())


if __name__ == "__main__":
    unittest.main()
