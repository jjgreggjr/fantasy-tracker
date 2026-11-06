"""The opt-in `nbr_health` family (model/features.py `_neighbor_health`) on a synthetic store. No network, no cache.

    python -m unittest model.tests.test_nbr_health

SEA 2025: RB1 (Charbonnet-like, the usage leader) plays weeks 1-3, is reported Out weeks 4-7 and plays week 8; RB2 (Wilson-like)
plays every week and carries the load while RB1 is out; RB3 is a deep backup. Proves: the backup's week-9 row sees the starter
return (returned_wks 1, healthy streak 1); a week-8 row cannot see week 8's result (point in time) and sees the starter still out;
this week's report is not used; no neighbor / never missed / week 1 / no history are NaN; an unreported absence after starting
counts as missed; a QB is ranked on snap share; the trace never touches the target game; the family is registered opt-in.
"""
from __future__ import annotations

import math
import unittest

import pandas as pd

from model import features as F
from model import serve as S
from model import train as T
from model.features import NAN, _neighbor_health, _run_stats
from model.point_in_time import KNOWN_AT, RawStore, RawTable, TargetRow

COLS = F.FAMILIES["nbr_health"]
SEASON = 2025
KICK0 = pd.Timestamp("2025-09-07 17:00", tz="UTC")
PLAYERS = (("RB1", "RB"), ("RB2", "RB"), ("RB3", "RB"), ("QB1", "QB"), ("QB2", "QB"))


def ko(week: int) -> pd.Timestamp:
    return KICK0 + pd.Timedelta(weeks=week - 1)


def gid(week: int) -> str:
    return f"{SEASON}_{week:02d}_DEN_SEA"


def same(a, b) -> bool:
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-9


def pg(pid: str, pos: str, week: int, carries: float, targets: float = 0.0, attempts: float = 0.0) -> dict:
    return {"player_id": pid, "position": pos, "season": SEASON, "week": week, "season_type": "REG", "game_id": gid(week),
            "team": "SEA", "opponent_team": "DEN", "fantasy_points_ppr": carries + targets, "carries": carries, "targets": targets,
            "attempts": attempts, "target_share": 0.1, "carry_share": 0.3, "receptions": 0.0, "receiving_yards": 0.0,
            "receiving_tds": 0.0, "rushing_yards": 0.0, "rushing_tds": 0.0, "passing_yards": 0.0, "passing_tds": 0.0,
            "passing_interceptions": 0.0, "completions": 0.0, KNOWN_AT: ko(week) + pd.Timedelta(days=2)}


def snap(pid: str, pos: str, week: int, pct: float) -> dict:
    return {"player_id": pid, "position": pos, "season": SEASON, "week": week, "game_type": "REG", "game_id": gid(week),
            "team": "SEA", "offense_pct": pct, "offense_snaps": 50.0 * pct, KNOWN_AT: ko(week) + pd.Timedelta(days=2)}


def inj(pid: str, pos: str, week: int, status: str) -> dict:
    return {"player_id": pid, "season": SEASON, "week": week, "game_type": "REG", "team": "SEA", "position": pos,
            "report_status": status, KNOWN_AT: ko(week) - pd.Timedelta(days=1)}


def gr(week: int) -> dict:
    return {"game_id": gid(week), "season": SEASON, "week": week, "game_type": "REG", "team": "SEA", "opponent": "DEN",
            "points_for": 20, "points_against": 17, "is_home": True, KNOWN_AT: ko(week) + pd.Timedelta(days=1)}


XFP_COLS = ["player_id", "game_id", "total_fantasy_points_exp", "rush_touchdown", "rush_touchdown_exp", "rec_touchdown",
            "rec_touchdown_exp", "pass_touchdown", "pass_touchdown_exp", KNOWN_AT]


def build(pgs, snaps, injs, weeks) -> RawStore:
    def tab(name, rows, cols=None):
        df = pd.DataFrame(rows, columns=cols)
        df[KNOWN_AT] = pd.to_datetime(df[KNOWN_AT], utc=True).astype("datetime64[ns, UTC]")
        return RawTable(name, df, "test")
    return RawStore([tab("player_games", pgs), tab("snap_counts", snaps), tab("injuries", injs, list(inj("x", "RB", 1, "Out"))), tab("xfp", [], XFP_COLS),
                     tab("game_results", [gr(w) for w in weeks])])


def rb_store(through: int = 8, extra_inj=()) -> RawStore:
    """Weeks 1..through completed. RB1: plays 1-3 and 8 (14 opps), reported Out 4-7. RB2: every week, 5 opps, then 13 while RB1
    is out. RB3: every week, 1 opp."""
    pgs, sn, ij = [], [], []
    for w in range(1, through + 1):
        healthy = w <= 3 or w == 8
        if healthy:
            pgs.append(pg("RB1", "RB", w, 12.0, 2.0))
            sn.append(snap("RB1", "RB", w, 0.6))
        else:
            ij.append(inj("RB1", "RB", w, "Out"))
        pgs.append(pg("RB2", "RB", w, 3.0 if healthy else 10.0, 2.0 if healthy else 3.0))
        sn.append(snap("RB2", "RB", w, 0.4))
        pgs.append(pg("RB3", "RB", w, 1.0))
        sn.append(snap("RB3", "RB", w, 0.05))
    return build(pgs, sn, ij + list(extra_inj), range(1, through + 1))


def target(pid: str, week: int, pos: str = "RB") -> TargetRow:
    return TargetRow(pid, SEASON, week, "SEA", "DEN", pos, gid(week), ko(week), True, ("usage",))


def feats(st: RawStore, pid: str, week: int, pos: str = "RB", trace=None) -> dict:
    t = target(pid, week, pos)
    have = set(st._tables["player_games"].df["player_id"])
    tab = {p: F._teammate_row(F._history(st, p, t, trace), t, ps, [], None) for p, ps in PLAYERS if p in have}
    return _neighbor_health(st, {"team_table": tab}, F._history(st, pid, t, trace), t, trace)


class Returned(unittest.TestCase):
    def test_the_backup_sees_the_starter_return_this_week(self):
        f = feats(rb_store(), "RB2", 9)                          # RB1 missed 4-7, played 8: week 9 is the first row to know
        self.assertEqual(set(f), set(COLS))
        self.assertEqual(f["nbr_ahead_returned_wks"], 1.0)
        self.assertEqual(f["nbr_ahead_healthy_streak"], 1.0)
        self.assertEqual(f["nbr_ahead_out_streak"], 0.0)
        rb1, rb2, rb3 = 14.0, (5.0 * 4 + 13.0 * 4) / 8, 1.0       # last-8 mean opps; RB1 has 4 appearances (all 14)
        self.assertTrue(same(f["nbr_ahead_share_l8"], rb1 / (rb1 + rb2 + rb3)))
        self.assertTrue(math.isnan(f["nbr_behind_returned_wks"]))    # RB3 never missed
        self.assertEqual(f["nbr_behind_healthy_streak"], 8.0)
        self.assertTrue(math.isnan(f["self_returned_wks"]))

    def test_the_starter_ranks_ahead_on_the_long_window(self):
        # RB2 is the backup: 5 opps healthy weeks, 16 while RB1 is out. A 3-game mean (16 vs RB1's 24 over his last 3) would
        # also hold; the point is the 8-appearance window keeps RB1 ahead after four missed games.
        pgs, sn = [], []
        for w in range(1, 9):
            if w <= 3 or w == 8:
                pgs.append(pg("RB1", "RB", w, 20.0, 4.0))
                sn.append(snap("RB1", "RB", w, 0.7))
            pgs.append(pg("RB2", "RB", w, 3.0 if w <= 3 or w == 8 else 14.0, 2.0))
            sn.append(snap("RB2", "RB", w, 0.3))
        ij = [inj("RB1", "RB", w, "Out") for w in (4, 5, 6, 7)]
        f = feats(build(pgs, sn, ij, range(1, 9)), "RB2", 9)
        self.assertEqual((f["nbr_ahead_returned_wks"], f["nbr_ahead_healthy_streak"], f["nbr_ahead_out_streak"]), (1.0, 1.0, 0.0))
        rb1, rb2 = 24.0, (5.0 * 4 + 16.0 * 4) / 8
        self.assertTrue(same(f["nbr_ahead_share_l8"], rb1 / (rb1 + rb2)))
        g = feats(build(pgs, sn, ij, range(1, 9)), "RB1", 9)       # the starter's own row: RB2 is behind, he returned himself
        self.assertEqual(g["self_returned_wks"], 1.0)
        self.assertTrue(math.isnan(g["nbr_ahead_returned_wks"]))
        self.assertEqual(g["nbr_behind_healthy_streak"], 8.0)
        self.assertTrue(math.isnan(g["nbr_behind_returned_wks"]))

    def test_returned_two_games_ago(self):
        pgs, sn = [], []
        for w in range(1, 10):
            if w in (1, 2, 3, 8, 9):
                pgs.append(pg("RB1", "RB", w, 20.0, 4.0))
                sn.append(snap("RB1", "RB", w, 0.7))
            pgs.append(pg("RB2", "RB", w, 3.0, 2.0))
            sn.append(snap("RB2", "RB", w, 0.3))
        f = feats(build(pgs, sn, [inj("RB1", "RB", w, "Out") for w in (4, 5, 6, 7)], range(1, 10)), "RB2", 10)
        self.assertEqual((f["nbr_ahead_returned_wks"], f["nbr_ahead_healthy_streak"]), (2.0, 2.0))

    def test_unreported_absence_after_starting_counts_as_missed(self):
        pgs, sn = [], []
        for w in range(1, 6):
            if w in (1, 2, 5):                                   # weeks 3-4 absent, never on the report (IR-like)
                pgs.append(pg("RB1", "RB", w, 20.0, 4.0))
                sn.append(snap("RB1", "RB", w, 0.7))
            pgs.append(pg("RB2", "RB", w, 3.0, 2.0))
            sn.append(snap("RB2", "RB", w, 0.3))
        f = feats(build(pgs, sn, [], range(1, 6)), "RB2", 6)
        self.assertEqual((f["nbr_ahead_returned_wks"], f["nbr_ahead_healthy_streak"]), (1.0, 1.0))

    def test_qb_is_ranked_on_snap_share(self):
        pgs = [pg("QB1", "QB", w, 1.0, attempts=30.0) for w in (1, 2, 4)] + [pg("QB2", "QB", w, 0.0, attempts=3.0) for w in (1, 2, 3, 4)]
        sn = [snap("QB1", "QB", w, 1.0) for w in (1, 2, 4)] + [snap("QB2", "QB", w, 0.1) for w in (1, 2, 3, 4)]
        f = feats(build(pgs, sn, [inj("QB1", "QB", 3, "Out")], range(1, 5)), "QB2", 5, pos="QB")
        self.assertEqual((f["nbr_ahead_returned_wks"], f["nbr_ahead_healthy_streak"]), (1.0, 1.0))


class PointInTime(unittest.TestCase):
    def test_week_n_cannot_see_week_n(self):
        # RB1's week-8 game is in the tables (stamped after its kickoff); the week-8 row must still see him out
        st = rb_store(through=8)
        f = feats(st, "RB2", 8)
        self.assertTrue(math.isnan(f["nbr_ahead_returned_wks"]))
        self.assertEqual((f["nbr_ahead_out_streak"], f["nbr_ahead_healthy_streak"]), (4.0, 0.0))

    def test_the_current_weeks_report_is_not_used(self):
        f = feats(rb_store(extra_inj=[inj("RB1", "RB", 9, "Out")]), "RB2", 9)   # known before the week-9 kickoff, still ignored
        self.assertEqual((f["nbr_ahead_healthy_streak"], f["nbr_ahead_out_streak"]), (1.0, 0.0))

    def test_trace_never_reads_the_target_game(self):
        trace: list = []
        feats(rb_store(), "RB2", 9, trace=trace)
        self.assertTrue(trace)
        for r in trace:
            self.assertTrue(r["max_known_at"] is None or r["max_known_at"] < ko(9), r)
            self.assertNotIn(gid(9), r["game_ids"])


class NaNCases(unittest.TestCase):
    def test_no_neighbor_is_nan(self):
        st = build([pg("RB1", "RB", w, 10.0) for w in (1, 2)], [snap("RB1", "RB", w, 0.5) for w in (1, 2)], [], (1, 2))
        f = feats(st, "RB1", 3)
        self.assertTrue(all(math.isnan(f[c]) for c in COLS))

    def test_never_missed_is_nan_but_the_streak_counts(self):
        f = feats(rb_store(), "RB1", 9)                            # RB3 is behind him: never missed
        self.assertTrue(math.isnan(f["nbr_behind_returned_wks"]))
        self.assertGreater(f["nbr_behind_healthy_streak"], 0.0)

    def test_week_one_is_all_nan(self):
        st = build([pg("RB1", "RB", 1, 10.0)], [snap("RB1", "RB", 1, 0.5)], [], (1,))
        self.assertTrue(all(math.isnan(v) for v in feats(st, "RB1", 1).values()))

    def test_a_player_with_no_history_has_no_rank_and_no_neighbors(self):
        f = feats(rb_store(), "RB9", 9)
        self.assertTrue(all(math.isnan(v) for v in f.values()))


class Unit(unittest.TestCase):
    def test_run_stats(self):
        t, f = True, False
        self.assertTrue(all(math.isnan(x) for x in _run_stats([])))
        self.assertTrue(math.isnan(_run_stats([t, t])[0]))
        self.assertEqual(_run_stats([t, f, f, t]), (1.0, 1.0, 0.0))
        self.assertEqual(_run_stats([t, f, t, t, t]), (3.0, 3.0, 0.0))
        self.assertEqual(_run_stats([f, t, f, f]), (3.0, 0.0, 2.0))      # returned, then missed again


class Registry(unittest.TestCase):
    def test_opt_in_servable_and_unshared(self):
        self.assertEqual(len(COLS), 7)
        self.assertIn("nbr_health", T.OPT_IN_FAMILIES)
        self.assertIsNone(S.family_source("nbr_health").table)
        others = {c for fam, cs in F.FAMILIES.items() if fam != "nbr_health" for c in cs}
        self.assertFalse(set(COLS) & others)


if __name__ == "__main__":
    unittest.main()
