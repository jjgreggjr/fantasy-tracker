"""Repeat DNP and role changes in the status layer (ff.status.recent_dnp / role_changes / build) and their
effect on E_pts (ff.score.apply_flags).

    python -m unittest tests.test_repeat_dnp

The Caleb Williams case: 2026 week 5, no stat row in weeks 3-4, still 14.8 E_pts. Coker / Moore / Cooper were
catchable from an IR activation or a practice-squad elevation the same week.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ff import score, status


def weeks(rows) -> pd.DataFrame:
    """player_weeks rows: (week, gsis_id, team, opps, snaps)."""
    return pd.DataFrame([{"season": 2026, "week": w, "gsis_id": g, "team": t, "position": "QB",
                          "opps": o, "snaps": s} for w, g, t, o, s in rows])


def players(ids, teams=None, practice=None, injury=None) -> pd.DataFrame:
    teams = teams or {}
    practice = practice or {}
    injury = injury or {}
    return pd.DataFrame({"gsis_id": ids, "name": ids, "team": [teams.get(i, "CHI") for i in ids],
                         "injury_status": [injury.get(i) for i in ids], "injury_body_part": [None] * len(ids),
                         "practice_participation": [practice.get(i) for i in ids],
                         "news_updated": [None] * len(ids), "sleeper_depth_order": list(range(1, len(ids) + 1))})


def depth(ids, team="CHI") -> pd.DataFrame:
    return pd.DataFrame({"gsis_id": ids, "name": ids, "team": team, "position": "QB",
                         "rank": list(range(1, len(ids) + 1))})


def sched_with_bye(team="CHI", bye=4) -> pd.DataFrame:
    rows = [{"season": 2026, "week": w, "team": team, "opponent": "BYE" if w == bye else "GB"} for w in range(1, 10)]
    return pd.DataFrame(rows)


# weeks 1-4: caleb plays 1-2, misses 3-4 (no rows); tyson plays every week; bench never has a snap
PW = weeks([(1, "caleb", "CHI", 10, 70), (2, "caleb", "CHI", 5, 64),
            (1, "tyson", "CHI", 3, 20), (2, "tyson", "CHI", 3, 20), (3, "tyson", "CHI", 30, 70),
            (4, "tyson", "CHI", 30, 70), (3, "bench", "CHI", 0, 0), (4, "bench", "CHI", 0, 0)])


class RecentDnp(unittest.TestCase):
    def test_two_missed_weeks_are_flagged_missing_rows_count_as_zero(self):
        d = status.recent_dnp(PW, 2026, 5, players(["caleb", "tyson", "bench"]))
        got = dict(zip(d.gsis_id, d.dnp_weeks))
        self.assertEqual(got, {"caleb": "3,4", "bench": "3,4"})

    def test_one_missed_week_is_not_enough(self):
        self.assertEqual(list(status.recent_dnp(PW, 2026, 4, players(["caleb", "tyson"])).gsis_id), [])

    def test_a_bye_is_skipped_not_counted(self):
        # CHI's bye is week 4 and it has no stat rows that week: the last two games are 2 and 3,
        # and caleb played week 2
        pw = PW[PW.week != 4]
        d = status.recent_dnp(pw, 2026, 5, players(["caleb", "tyson"]), sched_with_bye(bye=4))
        self.assertEqual(dict(zip(d.gsis_id, d.dnp_weeks)), {})
        pw = weeks([(1, "caleb", "CHI", 10, 70), (2, "caleb", "CHI", 0, 0), (3, "caleb", "CHI", 0, 0),
                    (2, "tyson", "CHI", 3, 20), (3, "tyson", "CHI", 3, 20), (5, "tyson", "CHI", 3, 20)])
        d = status.recent_dnp(pw, 2026, 6, players(["caleb", "tyson"]), sched_with_bye(bye=4))
        # last two CHI games before 6 are 3 and 5 (4 is the bye): caleb had no row in 5 and 0 in 3
        self.assertEqual(dict(zip(d.gsis_id, d.dnp_weeks)), {"caleb": "3,5"})

    def test_team_on_bye_this_week_is_left_alone(self):
        d = status.recent_dnp(PW, 2026, 5, players(["caleb"]), sched_with_bye(bye=5))
        self.assertTrue(d.empty)

    def test_unpublished_stats_do_not_flag_a_whole_team(self):
        # a Tuesday run before week 4's stats land: the last two games with rows are 2 and 3
        pw = PW[PW.week <= 3]
        d = status.recent_dnp(pw, 2026, 5, players(["caleb", "tyson"]))
        self.assertEqual(dict(zip(d.gsis_id, d.dnp_weeks)), {})      # caleb played week 2
        self.assertEqual(list(status.recent_dnp(None, 2026, 5).columns), status.DNP_COLS)


class BuildAppliesTheRules(unittest.TestCase):
    def _build(self, practice=None, injury=None, roles=None, injuries=None):
        p = players(["caleb", "tyson"], practice=practice, injury=injury)
        dnp = status.recent_dnp(PW, 2026, 5, p)
        st = status.build(p, depth(["caleb", "tyson"]), None, 5, injuries, None, dnp, roles)
        return st.set_index("gsis_id")

    def test_capped_at_a_quarter_and_flagged(self):
        st = self._build()
        self.assertEqual(st.loc["caleb", "repeat_dnp"], 1)
        self.assertLessEqual(st.loc["caleb", "play_prob"], status.REPEAT_DNP_CAP)
        self.assertEqual(st.loc["caleb", "dnp_weeks"], "3,4")
        self.assertEqual(st.loc["tyson", "repeat_dnp"], 0)
        self.assertEqual(st.loc["tyson", "play_prob"], 0.97)

    def test_full_practice_exempts(self):
        st = self._build(practice={"caleb": "Full Participation in Practice"})
        self.assertEqual(st.loc["caleb", "repeat_dnp"], 0)
        self.assertGreater(st.loc["caleb", "play_prob"], 0.9)

    def test_dnp_practice_or_out_zeroes(self):
        st = self._build(practice={"caleb": "Did Not Participate In Practice"})
        self.assertEqual((st.loc["caleb", "repeat_dnp"], st.loc["caleb", "play_prob"]), (2, 0.0))
        st = self._build(injury={"caleb": "Doubtful"}, practice={"caleb": "Limited"})
        self.assertEqual((st.loc["caleb", "repeat_dnp"], st.loc["caleb", "play_prob"]), (2, 0.0))
        st = self._build(injury={"caleb": "Questionable"}, practice={"caleb": "Limited"})
        self.assertEqual(st.loc["caleb", "repeat_dnp"], 1)
        self.assertEqual(st.loc["caleb", "play_prob"], 0.25)

    def test_activation_clears_the_penalty(self):
        roles = pd.DataFrame([{"gsis_id": "caleb", "prev_status": "RES", "status": "ACT",
                               "role_change_kind": "IR activation", "role_change_known_at": None}])
        st = self._build(roles=roles)
        self.assertEqual(st.loc["caleb", "repeat_dnp"], 0)
        self.assertEqual(st.loc["caleb", "role_change"], 1)
        self.assertEqual(st.loc["caleb", "role_change_kind"], "IR activation")
        self.assertEqual(st.loc["caleb", "play_prob"], 0.97)
        self.assertTrue(any("IR activation" in w for w in status.warnings(st.reset_index())))

    def test_existing_columns_keep_their_meaning(self):
        st = self._build()
        for c in ("status_flag", "report_status", "play_prob", "inactive", "inactive_known_at", "practice",
                  "blocked_by", "opportunity_ahead", "repeat_dnp", "dnp_weeks", "role_change", "role_change_kind"):
            self.assertIn(c, st.columns)
        self.assertEqual(st.loc["caleb", "status_flag"], "healthy")      # the designation is untouched


class RoleChanges(unittest.TestCase):
    def rosters(self, rows):
        return pd.DataFrame([{"season": 2026, "week": w, "gsis_id": g, "status": s} for w, g, s in rows])

    def test_ir_activation_and_ps_elevation(self):
        r = self.rosters([(4, "moore", "RES"), (5, "moore", "ACT"),
                          (4, "cooper", "DEV"), (5, "cooper", "ACT"),
                          (4, "same", "ACT"), (5, "same", "ACT"),
                          (4, "cut", "ACT"), (5, "cut", "CUT"),
                          (3, "gap", "PUP"), (5, "gap", "ACT")])
        out = status.role_changes(r, 2026, 5, "2026-10-03T20:00:00Z")
        got = dict(zip(out.gsis_id, out.role_change_kind))
        self.assertEqual(got, {"moore": "IR activation", "cooper": "practice-squad elevation",
                               "gap": "PUP activation"})
        self.assertEqual(list(out.columns), status.ROLE_CHANGE_COLS)
        self.assertEqual(out.role_change_known_at.iloc[0], "2026-10-03T20:00:00Z")

    def test_no_rows_for_the_week_yet(self):
        r = self.rosters([(4, "moore", "RES")])
        self.assertTrue(status.role_changes(r, 2026, 5).empty)
        self.assertTrue(status.role_changes(None, 2026, 5).empty)


class ScoreFlags(unittest.TestCase):
    def frame(self, **rows):
        return pd.DataFrame({
            "name": list(rows), "E_opps": [10.0] * len(rows), "opponent": ["KC"] * len(rows),
            "injury_status": [r.get("injury_status") for r in rows.values()],
            "repeat_dnp": [r.get("repeat_dnp", 0) for r in rows.values()],
            "role_change": [r.get("role_change", 0) for r in rows.values()],
            "E_pts": [12.0] * len(rows)})

    def test_capped_is_flagged_zeroed_is_blanked(self):
        df = score.apply_flags(self.frame(capped={"repeat_dnp": 1}, zeroed={"repeat_dnp": 2},
                                          fine={}, back={"role_change": 1},
                                          hurt={"repeat_dnp": 2, "injury_status": "Out"},
                                          nan={"repeat_dnp": np.nan}))
        flags, pts = dict(zip(df.name, df.flag)), dict(zip(df.name, df.E_pts))
        self.assertEqual(flags["capped"], "REPEAT_DNP")
        self.assertEqual(pts["capped"], 12.0)
        self.assertEqual(flags["zeroed"], "REPEAT_DNP")
        self.assertTrue(np.isnan(pts["zeroed"]))
        self.assertEqual(flags["hurt"], "OUT")
        self.assertTrue(np.isnan(pts["hurt"]))
        self.assertEqual((flags["fine"], pts["fine"]), ("", 12.0))
        self.assertEqual((flags["back"], pts["back"]), ("ROLE_CHANGE", 12.0))
        self.assertEqual((flags["nan"], pts["nan"]), ("", 12.0))

    def test_frames_without_the_columns_are_unchanged(self):
        df = score.apply_flags(self.frame(fine={}).drop(columns=["repeat_dnp", "role_change"]))
        self.assertEqual((df.flag.iloc[0], df.E_pts.iloc[0]), ("", 12.0))


if __name__ == "__main__":
    unittest.main()
