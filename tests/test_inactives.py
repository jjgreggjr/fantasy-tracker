"""The gameday inactive list (nflverse weekly_rosters, status INA) reaches status.csv and blanks E_pts.

    python -m unittest tests.test_inactives

No network and none of the repo's data files. The Coker case: 2026 week 4, Questionable all week, on CAR's gameday
inactive list, and still in the E_pts-optimal lineup because nothing read the list.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from ff import analyze, score, sources, status, verify

WEEK = 6
THU = pd.Timestamp("2026-10-16 00:15", tz="UTC")        # Thursday night, 20:15 ET
SUN = pd.Timestamp("2026-10-18 17:00", tz="UTC")        # Sunday 1 pm ET
FRIDAY_RUN = pd.Timestamp("2026-10-16 22:11", tz="UTC")
SUNDAY_RUN = pd.Timestamp("2026-10-18 15:52", tz="UTC")


def rosters(rows) -> pd.DataFrame:
    """nflverse roster_weekly_{season}: (week, gsis_id, team, status). INA rows carry A01, as the real 2026 file does."""
    return pd.DataFrame([{"season": 2026, "week": wk, "team": team, "gsis_id": g, "full_name": g, "position": "WR",
                          "status": st, "status_description_abbr": "A01", "game_type": "REG"}
                         for wk, g, team, st in rows])


def kicks() -> pd.DataFrame:
    return pd.DataFrame({"season": 2026, "week": [5, 5, WEEK, WEEK, WEEK, WEEK],
                         "team": ["CAR", "KC", "CAR", "DAL", "KC", "PIT"],
                         "kickoff": [THU - pd.Timedelta(days=7), THU - pd.Timedelta(days=7), SUN, SUN, THU, THU]})


def players_frame(**sleeper) -> pd.DataFrame:
    ids = ["coker", "wr2", "wr3"]
    return pd.DataFrame({"gsis_id": ids, "name": ["Jalen Coker", "Second Guy", "Third Guy"],
                         "injury_status": [sleeper.get(i) for i in ids], "injury_body_part": [None] * 3,
                         "practice_participation": [None] * 3, "news_updated": [None] * 3,
                         "sleeper_depth_order": [1, 2, 3]})


def depth_frame() -> pd.DataFrame:
    return pd.DataFrame({"gsis_id": ["coker", "wr2", "wr3"], "name": ["Jalen Coker", "Second Guy", "Third Guy"],
                         "team": "CAR", "position": "WR", "rank": [1, 2, 3]})


def injuries(rows) -> pd.DataFrame:
    return pd.DataFrame([{"season": 2026, "week": wk, "gsis_id": g, "team": "CAR", "position": "WR",
                          "report_status": rep, "practice_status": prac,
                          "report_primary_injury": None, "practice_primary_injury": None}
                         for wk, g, rep, prac in rows])


class WeekInactives(unittest.TestCase):
    def test_this_weeks_list_for_a_game_not_started_is_trusted_with_its_known_at(self):
        got = status.week_inactives(rosters([(WEEK, "coker", "CAR", "INA"), (WEEK, "wr2", "CAR", "ACT")]),
                                    2026, WEEK, kicks(), SUNDAY_RUN)
        self.assertEqual(got.gsis_id.tolist(), ["coker"])
        self.assertEqual(got.inactive_known_at.iloc[0], "2026-10-18T15:30:00Z")      # kickoff - 90 min

    def test_a_previous_weeks_inactive_does_not_carry_over(self):
        """The season file keeps every week's list: week 5's INA is not week 6's."""
        r = rosters([(5, "coker", "CAR", "INA"), (WEEK, "coker", "CAR", "ACT")])
        self.assertTrue(status.week_inactives(r, 2026, WEEK, kicks(), SUNDAY_RUN).empty)

    def test_a_game_already_under_way_is_not_re_flagged(self):
        """Sunday's run sees Thursday night's list, but that game is locked: only the unplayed games count."""
        r = rosters([(WEEK, "kc_guy", "KC", "INA"), (WEEK, "coker", "CAR", "INA")])
        self.assertEqual(status.week_inactives(r, 2026, WEEK, kicks(), SUNDAY_RUN).gsis_id.tolist(), ["coker"])
        # Friday run, before Thursday kickoff would have kept it; after it, it is gone
        self.assertEqual(set(status.week_inactives(r, 2026, WEEK, kicks(), THU - pd.Timedelta(hours=1)).gsis_id),
                         {"kc_guy", "coker"})

    def test_unknown_kickoff_keeps_the_row_on_the_week_guard(self):
        got = status.week_inactives(rosters([(WEEK, "x", "ZZZ", "INA")]), 2026, WEEK, kicks(), SUNDAY_RUN)
        self.assertEqual(got.gsis_id.tolist(), ["x"])
        self.assertTrue(pd.isna(got.inactive_known_at.iloc[0]))

    def test_another_season_and_missing_inputs_are_empty(self):
        r = rosters([(WEEK, "coker", "CAR", "INA")]).assign(season=2025)
        self.assertTrue(status.week_inactives(r, 2026, WEEK, kicks(), SUNDAY_RUN).empty)
        self.assertTrue(status.week_inactives(None, 2026, WEEK).empty)
        self.assertTrue(status.week_inactives(pd.DataFrame({"x": [1]}), 2026, WEEK).empty)
        self.assertTrue(status.week_inactives(rosters([(WEEK, "c", "CAR", "INA")]), 2026, None).empty)

    def test_team_kickoffs_are_eastern_converted_to_utc(self):
        games = pd.DataFrame({"season": [2026], "week": [WEEK], "game_type": ["REG"], "gameday": ["2026-10-15"],
                              "gametime": ["20:15"], "home_team": ["PIT"], "away_team": ["KC"]})
        k = status.team_kickoffs(games, 2026).set_index("team").kickoff
        self.assertEqual(k["KC"], THU)
        self.assertEqual(k["PIT"], THU)
        self.assertTrue(status.team_kickoffs(None, 2026).empty)

    def test_the_archive_keeps_every_week_for_the_retrospective(self):
        a = status.inactive_archive(rosters([(5, "coker", "CAR", "INA"), (WEEK, "wr2", "CAR", "INA"),
                                             (WEEK, "wr3", "CAR", "ACT")]), 2026, kicks())
        self.assertEqual(sorted(zip(a.week, a.gsis_id)), [(5, "coker"), (WEEK, "wr2")])
        self.assertEqual(a.set_index("gsis_id").loc["wr2", "inactive_known_at"], "2026-10-18T15:30:00Z")


class CokerCase(unittest.TestCase):
    """Questionable all week, Limited in practice, then on the gameday inactive list."""

    def build(self, inactive_rows, now=SUNDAY_RUN):
        ina = status.week_inactives(rosters(inactive_rows), 2026, WEEK, kicks(), now)
        inj = injuries([(WEEK, "coker", "Questionable", "Limited Participation in Practice")])
        return status.build(players_frame(coker="Questionable"), depth_frame(), None, WEEK, inj, ina).set_index("gsis_id")

    def test_ina_overrides_the_designation_and_play_prob_is_zero(self):
        st = self.build([(WEEK, "coker", "CAR", "INA")])
        self.assertEqual(st.loc["coker", "status_flag"], "inactive")
        self.assertEqual(st.loc["coker", "play_prob"], 0.0)
        self.assertEqual(st.loc["coker", "inactive"], 1)
        self.assertEqual(st.loc["coker", "inactive_known_at"], "2026-10-18T15:30:00Z")
        self.assertEqual(st.loc["coker", "report_status"], "Questionable")              # the report is left as it was
        self.assertEqual(st.loc["wr2", "inactive"], 0)
        # the man behind him sees an inactive blocker, with his whole share in doubt
        self.assertEqual(st.loc["wr2", "blockers_out"], 1)
        self.assertIn("Jalen Coker (inactive", st.loc["wr2", "blocked_by"])

    def test_without_the_list_he_is_questionable_as_before(self):
        st = self.build([(WEEK, "coker", "CAR", "ACT")])
        self.assertEqual(st.loc["coker", "status_flag"], "questionable")
        self.assertGreater(st.loc["coker", "play_prob"], 0.5)
        self.assertEqual(st.loc["coker", "inactive"], 0)

    def test_a_stale_week_5_inactive_does_not_bench_him_in_week_6(self):
        st = self.build([(5, "coker", "CAR", "INA")])
        self.assertEqual(st.loc["coker", "status_flag"], "questionable")

    def test_describe_and_warnings_say_inactive(self):
        st = self.build([(WEEK, "coker", "CAR", "INA")]).reset_index()
        self.assertIn("INACTIVE", status.describe(st, "Coker"))
        self.assertIn("2026-10-18T15:30:00Z", status.describe(st, "Coker"))
        self.assertTrue(any("Jalen Coker: INACTIVE" in w for w in status.warnings(st)))

    def test_e_pts_is_blanked_and_the_injury_status_is_kept(self):
        df = pd.DataFrame({"name": ["Jalen Coker", "Second Guy", "Plain"], "E_opps": [6.0, 4.0, 5.0],
                           "opponent": ["DAL", "DAL", "DAL"], "injury_status": ["Questionable", None, None],
                           "inactive": [1, 0, np.nan], "E_pts": [11.0, 7.0, 9.0]})
        df = score.apply_flags(df)
        self.assertEqual(df.flag.tolist(), ["INACTIVE", "", ""])
        self.assertTrue(np.isnan(df.E_pts.iloc[0]))
        self.assertEqual(df.E_pts.iloc[1:].tolist(), [7.0, 9.0])
        self.assertEqual(df.injury_status.iloc[0], "Questionable")
        # a frame with no inactive column behaves exactly as before
        old = score.apply_flags(df.drop(columns=["inactive", "flag"]).assign(E_pts=[11.0, 7.0, 9.0]))
        self.assertEqual(old.flag.tolist(), ["QUESTIONABLE", "", ""])
        self.assertEqual(old.E_pts.iloc[0], 11.0)

    def test_score_compute_carries_the_column_from_players(self):
        """run_weekly sets players['inactive']; score.compute must pass it through to apply_flags."""
        players = pd.DataFrame({"gsis_id": ["coker"], "age": [24], "years_exp": [2], "injury_status": ["Questionable"],
                                "nfl_status": ["ACT"], "draft_number": [np.nan], "rookie_year": [2024], "inactive": [1]})
        roster = pd.DataFrame({"gsis_id": ["coker"], "name": ["Jalen Coker"], "position": ["WR"], "team": ["CAR"]})
        base = pd.DataFrame({"gsis_id": ["coker"], "avg_opps": [6.0], "avg_targets": [6.0], "avg_carries": [0.0],
                             "games": [4]})
        sched = pd.DataFrame({"week": [WEEK], "team": ["CAR"], "opponent": ["DAL"], "total_line": [44.0], "home": [1]})
        dvp = pd.DataFrame({"defense": ["DAL"], "position": ["WR"], "dvp_rank": [5]})
        depth = pd.DataFrame({"gsis_id": ["coker"], "rank": [1], "prev_rank": [1]})
        points = pd.DataFrame({"season": [2026], "week": [WEEK - 1], "gsis_id": ["coker"], "pts": [12.0]})
        out = score.compute(roster, players, base, pd.DataFrame(), dvp, sched, depth, points,
                            {"name": "t", "scoring": {}}, WEEK, 2026)
        self.assertEqual(out.flag.iloc[0], "INACTIVE")
        self.assertTrue(np.isnan(out.E_pts.iloc[0]))
        self.assertEqual(out.confidence.iloc[0], 0.0)

    def test_start_sit_blanks_the_score_too(self):
        roster = pd.DataFrame({"gsis_id": ["coker", "wr2"], "name": ["Jalen Coker", "Second Guy"],
                               "position": ["WR", "WR"], "team": ["CAR", "CAR"]})
        players = pd.DataFrame({"gsis_id": ["coker", "wr2"], "injury_status": ["Questionable", None], "inactive": [1, 0]})
        base = pd.DataFrame({"gsis_id": ["coker", "wr2"], "avg_opps": [6.0, 4.0], "avg_snap_pct": [0.8, 0.6]})
        sched = pd.DataFrame({"week": [WEEK], "team": ["CAR"], "opponent": ["DAL"], "home": [1], "spread_line": [1.0],
                              "total_line": [44.0], "gameday": ["2026-10-18"]})
        dvp = pd.DataFrame({"defense": ["DAL"], "position": ["WR"], "dvp_rank": [5], "fpts": [30.0]})
        depth = pd.DataFrame({"gsis_id": ["coker", "wr2"], "rank": [1, 2], "prev_rank": [1, 2]})
        out = analyze.start_sit(roster, players, base, dvp, sched, depth, WEEK,
                                {"opps": 0.5, "snap_pct": 0.3, "dvp_rank": 0.2}).set_index("gsis_id")
        self.assertEqual(out.loc["coker", "flag"], "INACTIVE")
        self.assertTrue(np.isnan(out.loc["coker", "start_score"]))
        self.assertFalse(np.isnan(out.loc["wr2", "start_score"]))


class MissingFileIsAWarn(unittest.TestCase):
    def test_load_nflverse_asks_for_the_weekly_rosters_csv(self):
        got = []
        with mock.patch.object(sources, "nflverse_csv",
                               lambda rel, fn, raw, required=True: got.append((rel, fn, required)) or None):
            out = sources.load_nflverse(2026, Path("x"), with_prior=False)
        self.assertIn(("weekly_rosters", "roster_weekly_2026.csv", False), got)
        self.assertIsNone(out["weekly_rosters"])

    def test_an_unreadable_file_is_none_not_a_crash(self):
        def fake(rel, fn, raw, required=True):
            if rel == "weekly_rosters":
                raise ValueError("garbled csv")
            return pd.DataFrame({"a": [1]})
        with mock.patch.object(sources, "nflverse_csv", fake):
            out = sources.load_nflverse(2026, Path("x"), with_prior=False)
        self.assertIsNone(out["weekly_rosters"])
        self.assertIsNotNone(out["injuries"])

    def check(self, **kw):
        st = pd.DataFrame({"gsis_id": ["a"], "practice": ["Full"]})
        with tempfile.TemporaryDirectory() as tmp:
            st.to_csv(Path(tmp) / "status.csv", index=False)
            res = verify.verify(Path(tmp), 2026, WEEK, **kw)
        return [r for r in res if r["check"] == "status.inactives"]

    def test_verify_warns_when_the_file_was_not_loaded(self):
        r = self.check(inactive_rows=None)
        self.assertEqual(r[0]["severity"], "WARN")
        self.assertIn("weekly_rosters", r[0]["detail"])

    def test_verify_is_ok_with_an_empty_list_and_silent_when_not_told(self):
        self.assertEqual(self.check(inactive_rows=0)[0]["severity"], "OK")
        self.assertEqual(self.check(inactive_rows=14)[0]["severity"], "OK")
        self.assertEqual(self.check(), [])


if __name__ == "__main__":
    unittest.main()
