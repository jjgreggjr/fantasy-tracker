"""Played-lineup capture: transforms, safe writes, integrity check and the recipe.

    python -m unittest discover -s tests -t .

Nothing here touches the network or the repo's data files.
"""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from ff import ask, build, espn, verify
from ff.run_weekly import _write_played


def players_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"gsis_id": "00-1", "sleeper_id": "101", "espn_id": "9001", "pfr_id": None,
         "name": "Pat Passer", "team": "KC", "position": "QB"},
        {"gsis_id": "00-2", "sleeper_id": "102", "espn_id": "9002", "pfr_id": None,
         "name": "Rex Runner", "team": "SF", "position": "RB"},
        {"gsis_id": "00-3", "sleeper_id": "103", "espn_id": "9003", "pfr_id": None,
         "name": "Wes Wideout", "team": "LA", "position": "WR"},
        {"gsis_id": "00-4", "sleeper_id": None, "espn_id": None, "pfr_id": None,
         "name": "Nobody Known", "team": "BUF", "position": "WR"},
    ])


# ---------------------------------------------------------------- Sleeper
def sleeper_league(median: bool = False) -> dict:
    return {
        "league": {"league_id": "L1", "name": "Sleeper League",
                   "roster_positions": ["QB", "RB", "WR", "K", "DEF", "BN", "BN"],
                   "settings": {"league_average_match": int(median)}},
        "rosters": [{"roster_id": 1, "owner_id": "u1"}, {"roster_id": 2, "owner_id": "u2"},
                    {"roster_id": 3, "owner_id": "u3"}, {"roster_id": 4, "owner_id": "u4"}],
        "users": [{"user_id": f"u{i}", "display_name": f"Owner{i}"} for i in range(1, 5)],
        "fetched_at": "2026-09-29T12:00:00Z",
    }


def sleeper_entry(rid: int, mid: int, points: float, started_extra: bool = True) -> dict:
    starters = ["101", "102", "103", "900", "SF"]
    if not started_extra:
        starters[3] = "0"                       # empty K slot
    pp = {"101": 20.0, "102": 10.0, "103": 5.0, "900": 8.0, "SF": 7.0, "555": 30.0}
    return {"roster_id": rid, "matchup_id": mid, "points": points,
            "starters": starters, "players": sorted(set(starters + ["555"]) - {"0"}),
            "players_points": pp}


SLEEPER_DICT = {
    "900": {"full_name": "Kirk Kicker", "position": "K", "team": "DET"},
    "SF": {"first_name": "San Francisco", "last_name": "49ers", "position": "DEF",
           "team": "SF"},
    "555": {"full_name": "Bench Guy", "position": "RB", "team": "NE"},
}


class SleeperPlayed(unittest.TestCase):
    def build(self, payload, median=False):
        return build.build_sleeper_played(sleeper_league(median), {3: payload},
                                          players_frame(), 2026, SLEEPER_DICT)

    def test_lineup_rows_slots_and_points(self):
        payload = [sleeper_entry(1, 7, 50.0), sleeper_entry(2, 7, 40.0, False),
                   sleeper_entry(3, 8, 60.0), sleeper_entry(4, 8, 60.0)]
        lu, rs = self.build(payload)
        t = lu[lu.roster_id == 1]
        self.assertEqual(list(lu.columns), build.PLAYED_COLS)
        self.assertEqual(int(t.started.sum()), 5)
        started = t[t.started == 1]
        self.assertEqual(list(started.slot), ["QB", "RB", "WR", "K", "DEF"])
        self.assertEqual(float(started.points.sum()), 50.0)
        bench = t[t.started == 0]
        self.assertEqual(list(bench.sleeper_id), ["555"])
        self.assertTrue(bench.slot.isna().all())          # Sleeper does not say
        self.assertEqual(float(bench.points.iloc[0]), 30.0)
        # crosswalk first, then the Sleeper dictionary for K/DEF
        by_id = t.set_index("sleeper_id")
        self.assertEqual(by_id.loc["101", "gsis_id"], "00-1")
        self.assertEqual(by_id.loc["900", "name"], "Kirk Kicker")
        self.assertEqual(by_id.loc["SF", "name"], "San Francisco 49ers")
        self.assertEqual(by_id.loc["SF", "position"], "DEF")
        self.assertEqual(t.owner_name.iloc[0], "Owner1")
        # an empty slot is not a player and keeps the other slots aligned
        e = lu[(lu.roster_id == 2) & (lu.started == 1)]
        self.assertEqual(len(e), 4)
        self.assertEqual(list(e.slot), ["QB", "RB", "WR", "DEF"])

    def test_results_pairs_and_ties(self):
        payload = [sleeper_entry(1, 7, 50.0), sleeper_entry(2, 7, 40.0),
                   sleeper_entry(3, 8, 60.0), sleeper_entry(4, 8, 60.0)]
        _, rs = self.build(payload)
        self.assertEqual(list(rs.columns), build.RESULT_COLS)
        r = rs.set_index("roster_id")
        self.assertEqual((r.loc[1, "won"], r.loc[2, "won"]), (1, 0))
        self.assertEqual(r.loc[1, "opponent_roster_id"], 2)
        self.assertEqual(r.loc[2, "opponent_owner_name"], "Owner1")
        self.assertEqual((r.loc[1, "points_for"], r.loc[1, "points_against"]), (50.0, 40.0))
        self.assertEqual((r.loc[3, "tie"], r.loc[3, "won"]), (1, 0))
        self.assertTrue(rs.median_won.isna().all())        # not a median league
        # every matchup is exactly one win and one loss, or two ties
        for _, g in rs.groupby("matchup_id"):
            self.assertIn(int(g.won.sum() + g.tie.sum()), (1, 2))

    def test_median_league(self):
        payload = [sleeper_entry(1, 7, 50.0), sleeper_entry(2, 7, 40.0),
                   sleeper_entry(3, 8, 60.0), sleeper_entry(4, 8, 30.0)]
        _, rs = self.build(payload, median=True)
        r = rs.set_index("roster_id")
        # median of 60, 50, 40, 30 is 45
        self.assertEqual(list(r.median_won.astype(int)), [1, 0, 1, 0])

    def test_unplayed_week_is_not_recorded(self):
        payload = [sleeper_entry(i, 1 + i // 2, 0.0) for i in range(1, 5)]
        lu, rs = self.build(payload)
        self.assertTrue(lu.empty and rs.empty)

    def test_bye_or_unmatched_roster_has_lineup_but_no_result(self):
        payload = [sleeper_entry(1, 7, 50.0), sleeper_entry(2, 7, 40.0),
                   sleeper_entry(3, None, 60.0), sleeper_entry(4, 9, 60.0)]
        lu, rs = self.build(payload)
        self.assertEqual(set(lu.roster_id), {1, 2, 3, 4})
        self.assertEqual(set(rs.roster_id), {1, 2})


# ------------------------------------------------------------------- ESPN
def espn_entry(pid, slot_id, points, pos_id=None, stats=True, name=None, team_id=None):
    player = {"id": pid, "fullName": name, "defaultPositionId": pos_id,
              "proTeamId": team_id}
    pe = {"player": player}
    if stats:
        player["stats"] = [
            {"scoringPeriodId": 0, "statSourceId": 0, "statSplitTypeId": 0, "appliedTotal": 999.0},
            {"scoringPeriodId": 3, "statSourceId": 1, "statSplitTypeId": 1, "appliedTotal": 99.0},
            {"scoringPeriodId": 3, "statSourceId": 0, "statSplitTypeId": 1, "appliedTotal": points}]
    else:
        pe["appliedStatTotal"] = points
    return {"lineupSlotId": slot_id, "playerPoolEntry": pe}


def espn_payload(**over) -> dict:
    p = {
        "settings": {"scheduleSettings": {"matchupPeriods": {"3": [3]}}},
        "teams": [
            {"id": 14, "roster": {"entries": [
                espn_entry(9001, 0, 20.0),                       # QB starts
                espn_entry(9002, 20, 12.5, stats=False),         # RB on the bench
                espn_entry(9003, 4, 5.0, stats=False),           # WR starts, applied total
                espn_entry(7777, 23, 9.0, pos_id=3, name="Rookie Unknown", team_id=1),
                espn_entry(-16014, 16, 7.0, pos_id=16, team_id=14),   # DEF starts
                espn_entry(4444, 17, 8.0, pos_id=5, name="Kick Er", team_id=12),
            ]}},
            {"id": 5, "roster": {"entries": [espn_entry(9001, 0, 10.0)]}},
            {"id": 8, "roster": {"entries": [espn_entry(9001, 0, 33.0)]}},
        ],
        "schedule": [
            {"id": 30, "matchupPeriodId": 3, "winner": "HOME",
             "home": {"teamId": 14, "totalPoints": 49.0}, "away": {"teamId": 5, "totalPoints": 10.0}},
            {"id": 31, "matchupPeriodId": 2, "winner": "AWAY",
             "home": {"teamId": 14, "totalPoints": 1.0}, "away": {"teamId": 5, "totalPoints": 2.0}},
            {"id": 32, "matchupPeriodId": 3, "winner": "UNDECIDED",
             "home": {"teamId": 8, "totalPoints": 33.0}},     # bye
        ],
    }
    p.update(over)
    return p


ESPN_META = {"league_id": "704757", "name": "Average Joes",
             "owners": {"14": "Nativity scene", "5": "Someone", "8": "Other"}}


class EspnPlayed(unittest.TestCase):
    def parse(self, data=None):
        return espn.parse_played(data or espn_payload(), players_frame(), 2026, 3, ESPN_META)

    def test_lineups(self):
        lu, _ = self.parse()
        self.assertEqual(list(lu.columns), build.PLAYED_COLS)
        t = lu[lu.roster_id == 14].set_index("sleeper_id")
        self.assertEqual(t.loc["101", "started"], 1)
        self.assertEqual(t.loc["101", "slot"], "QB")
        self.assertEqual(t.loc["101", "gsis_id"], "00-1")
        self.assertEqual(t.loc["102", "started"], 0)      # benched
        self.assertEqual(t.loc["102", "slot"], "BN")
        self.assertEqual(t.loc["102", "points"], 12.5)    # appliedStatTotal path
        self.assertEqual(t.loc["101", "points"], 20.0)    # per-period stat line, not 99 / 999
        self.assertEqual(t.loc["espn--16014", "position"], "DEF")
        self.assertEqual(t.loc["espn--16014", "name"], "LA DEF")
        self.assertEqual(t.loc["espn-4444", "name"], "Kick Er")
        # an unmatched skill player is kept, not dropped
        self.assertEqual(t.loc["espn-7777", "position"], "WR")
        self.assertEqual(t.loc["espn-7777", "started"], 1)
        self.assertEqual(t.owner_name.iloc[0], "Nativity scene")
        self.assertEqual(lu.sleeper_id.isna().sum(), 0)

    def test_snapshot_identity_is_unchanged(self):
        """The unmatched-skill fallback belongs to played lineups only."""
        rows = espn.parse_rosters(espn_payload(), players_frame(), 2026, 4,
                                  {**ESPN_META, "fetched_at": "t"})
        r = rows[rows.roster_id == 14].set_index("espn_id")
        self.assertTrue(pd.isna(r.loc["7777", "sleeper_id"]))
        self.assertEqual(r.loc["-16014", "sleeper_id"], "espn--16014")
        self.assertEqual(r.loc["4444", "position"], "K")
        self.assertEqual(r.loc["9002", "is_starter"], 0)

    def test_results(self):
        _, rs = self.parse()
        self.assertEqual(list(rs.columns), build.RESULT_COLS)
        self.assertEqual(len(rs), 2)                      # week 2 and the bye are out
        r = rs.set_index("roster_id")
        self.assertEqual((r.loc[14, "won"], r.loc[5, "won"]), (1, 0))
        self.assertEqual(r.loc[14, "opponent_owner_name"], "Someone")
        self.assertEqual(r.loc[14, "points_against"], 10.0)

    def test_undecided_falls_back_to_scores_and_ties(self):
        d = espn_payload()
        d["schedule"][0]["winner"] = "UNDECIDED"
        d["schedule"][0]["home"]["totalPoints"] = 10.0
        _, rs = self.parse(d)
        self.assertEqual(list(rs.tie), [1, 1])
        self.assertEqual(list(rs.won), [0, 0])
        d["schedule"][0]["home"]["totalPoints"] = 4.0
        _, rs = self.parse(d)
        self.assertEqual(rs.set_index("roster_id").loc[5, "won"], 1)

    def test_multi_week_matchup_writes_lineups_but_no_results(self):
        d = espn_payload()
        d["settings"]["scheduleSettings"]["matchupPeriods"] = {"3": [3, 4]}
        lu, rs = self.parse(d)
        self.assertFalse(lu.empty)
        self.assertTrue(rs.empty)

    def test_empty_payload_gives_empty_frames(self):
        lu, rs = self.parse({"teams": [], "schedule": []})
        self.assertTrue(lu.empty and rs.empty)


# ------------------------------------------------------------ safe writes
class SafeWrites(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        import ff.run_weekly as rw
        self.rw, self._data = rw, rw.DATA
        rw.DATA = Path(self.tmp.name)
        self.addCleanup(setattr, rw, "DATA", self._data)

    def frames(self, league, week, pts):
        lu = pd.DataFrame([{**{c: None for c in build.PLAYED_COLS},
                            "season": 2026, "week": week, "league_id": league,
                            "roster_id": 1, "sleeper_id": s, "started": 1, "points": pts}
                           for s in ("101", "SF")])
        rs = pd.DataFrame([{**{c: None for c in build.RESULT_COLS},
                            "season": 2026, "week": week, "league_id": league,
                            "roster_id": 1, "points_for": pts}])
        return lu, rs

    def test_replace_only_the_fetched_partition_and_never_on_empty(self):
        _write_played(*self.frames("A", 1, 1.0), [1, 2, 3])
        _write_played(*self.frames("A", 2, 2.0), [1, 2, 3])
        _write_played(*self.frames("704757", 1, 3.0), [1, 2, 3])
        _write_played(*self.frames("A", 1, 9.0), [1, 2, 3])           # refetch, new points
        lp = pd.read_csv(Path(self.tmp.name) / "lineups_played.csv")
        self.assertEqual(len(lp), 6)
        self.assertEqual(set(lp[(lp.league_id.astype(str) == "A") & (lp.week == 1)].points), {9.0})
        self.assertEqual(set(lp[(lp.league_id.astype(str) == "A") & (lp.week == 2)].points), {2.0})
        _write_played(lp.iloc[0:0], pd.DataFrame(columns=build.RESULT_COLS), [1, 2, 3])
        self.assertEqual(len(pd.read_csv(Path(self.tmp.name) / "lineups_played.csv")), 6)

    def test_in_progress_week_is_never_written(self):
        _write_played(*self.frames("A", 4, 0.0), [1, 2, 3])
        self.assertFalse((Path(self.tmp.name) / "lineups_played.csv").exists())
        self.assertFalse((Path(self.tmp.name) / "matchup_results.csv").exists())

    def test_unkeyable_rows_fall_back_to_upsert(self):
        lu, rs = self.frames("A", 1, 1.0)
        _write_played(lu, rs, [1])
        lu2, rs2 = self.frames("A", 1, 5.0)
        lu2.loc[1, "sleeper_id"] = None                 # one row can't be keyed
        _write_played(lu2, rs2, [1])
        lp = pd.read_csv(Path(self.tmp.name) / "lineups_played.csv")
        self.assertEqual(len(lp), 2)                    # the old SF row was not deleted
        self.assertEqual(sorted(lp.points), [1.0, 5.0])

    def test_completed_weeks(self):
        self.assertEqual(verify.completed_weeks(4), [1, 2, 3])
        self.assertEqual(verify.completed_weeks(1), [])
        self.assertEqual(verify.completed_weeks(25), list(range(1, 19)))


# -------------------------------------------------------- integrity check
class Coverage(unittest.TestCase):
    def check(self, tmp, week, ids):
        res = verify.verify(Path(tmp), 2026, week, ids)
        return {r["check"]: r for r in res}

    def test_warns_on_missing_week_and_league(self):
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame({"season": 2026, "week": [1, 2, 3, 1, 2], "league_id":
                          [1, 1, 1, 2, 2]}).to_csv(Path(tmp) / "lineups_played.csv", index=False)
            pd.DataFrame({"season": 2026, "week": [1, 2, 3], "league_id": [1, 1, 1]}
                         ).to_csv(Path(tmp) / "matchup_results.csv", index=False)
            c = self.check(tmp, 4, ["1", "2", "3"])
            lp = c["lineups_played.coverage"]
            self.assertEqual(lp["severity"], "WARN")
            self.assertIn("league 2 wk[3]", lp["detail"])
            self.assertIn("league 3 wk[1, 2, 3]", lp["detail"])
            ok = self.check(tmp, 4, ["1"])
            self.assertEqual(ok["lineups_played.coverage"]["severity"], "OK")
            # nothing to expect before a week has been completed
            self.assertNotIn("lineups_played.coverage", self.check(tmp, 1, ["1"]))

    def test_missing_file_is_a_warn_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.check(tmp, 4, ["1"])["lineups_played.coverage"]["severity"], "WARN")


# ----------------------------------------------------------------- recipe
class PlayedRecipe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root / "data").mkdir()
        (root / "leagues" / "x").mkdir(parents=True)
        (root / "leagues" / "x" / "league.json").write_text(json.dumps({
            "league_id": "704757", "platform": "espn", "name": "Average Joes",
            "roster_positions": ["QB", "WR", "DEF", "K", "BN", "IR", "FLEX"],
            "my_roster_id": 14}))
        lu, rs = espn.parse_played(espn_payload(), players_frame(), 2026, 3, ESPN_META)
        lu.to_csv(root / "data" / "lineups_played.csv", index=False)
        rs.to_csv(root / "data" / "matchup_results.csv", index=False)
        self._root, ask.ROOT = ask.ROOT, root
        self.addCleanup(setattr, ask, "ROOT", self._root)

    def test_played(self):
        start, note, bench = ask.played("x")            # default: latest recorded week
        self.assertEqual(list(start.name.iloc[:2]), ["Pat Passer", "Wes Wideout"])
        self.assertEqual(list(start.slot.iloc[:2]), ["QB", "WR"])
        self.assertEqual(list(bench.name), ["Rex Runner"])
        self.assertIn("scored 49.00", note)
        self.assertIn("against Someone (10.00) — WIN", note)
        self.assertIn("Scoring rank 1 of 2", note)
        self.assertIn("ACTUAL PLAYED LINEUP", start.attrs["header"])

    def test_unrecorded_week_and_unknown_state(self):
        start, note, _ = ask.played("x", 9)
        self.assertTrue(start.empty)
        self.assertIn("not recorded", note)
        (Path(self.tmp.name) / "data" / "lineups_played.csv").unlink()
        start, note, _ = ask.played("x")
        self.assertIn("No played lineups", note)

    def test_cli(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            ask.main(["played", "x", "3"])
        out = buf.getvalue()
        self.assertIn("ACTUAL PLAYED LINEUP", out)
        self.assertIn("Bench (did not start), by points:", out)
        buf = io.StringIO()
        with redirect_stdout(buf):
            ask.main(["played", "x", "third"])
        self.assertIn("takes a week number", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
