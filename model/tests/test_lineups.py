"""Recorded optimal lineups: record, lock at kickoff, score, report.

    python3.12 -m unittest model.tests.test_lineups

A synthetic repo in a temp directory (no network, nothing committed is touched). `ff.calls.three_lineups` is another module's
job; here `stub_three` stands in for it with the same contract (greedy by E_pts / E_pts_model, NaN-safe), injected into `record`.
Proven: a first record writes three lineups; a re-record after some kickoffs keeps the locked rows byte for byte, swaps an injured
unlocked player out, and never lets a kicked-off bench player in; score totals each basis by sleeper_id (a missing player counts as
0 and in n_missing) with the `actual` basis from his started rows; the sleeper basis is recorded (a player with no Sleeper projection
is filled from E_pts); `perfect` is the hindsight-best lineup from his whole roster (a benched player who outscored a starter is in it)
and every row carries efficiency; backfill rebuilds completed weeks from frozen pre-kickoff rows, skips live-recorded weeks, marks
source and turns the status rule off for the call; the report, its Summary block and `ff.ask scoreboard` render; a failure is a WARN
row, exit 0, and writes nothing.
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

import pandas as pd

from ff import scoring
from model import lineups as L

SLUG, LID = "lg-a", "111"
META = {"slug": SLUG, "league_id": LID, "name": "League A", "platform": "sleeper", "my_roster_id": 1,
        "roster_positions": ["QB", "RB", "FLEX", "BN"]}
T0 = pd.Timestamp("2026-10-04T12:00:00Z")
EARLY, LATE = "2026-10-04T17:00:00+00:00", "2026-10-05T00:20:00+00:00"
# gsis, name, pos, team, E_pts, E_pts_model, kickoff
PLAYERS = [("q1", "Quinn", "QB", "AAA", 20.0, 19.0, EARLY), ("r1", "Rex", "RB", "AAA", 15.0, 14.0, EARLY),
           ("r2", "Ross", "RB", "BBB", 12.0, 13.0, LATE), ("w1", "Wade", "WR", "BBB", 11.0, 10.0, LATE),
           ("w2", "Wes", "WR", "AAA", 9.0, 8.0, EARLY), ("t1", "Tate", "TE", "BBB", 5.0, 6.0, LATE)]


STATUS_SEEN = []
SLEEPER = {"q1": 18.0, "r1": 14.0, "r2": 6.0, "w1": 13.0, "w2": 7.0}       # Sleeper likes Wade at FLEX; Tate has no row at all


def stub_three(roster, slots, idp_on, status=None):
    """Same contract as ff.calls.three_lineups: greedy fill by slot order; call = the model lineup."""
    STATUS_SEEN.append(status)
    out = {}
    for key, col in (("e_pts", "E_pts"), ("model", "E_pts_model"), ("call", "E_pts_model")):
        pool, rows = roster.sort_values(col, ascending=False, na_position="last"), []
        used = set()
        for s in slots:
            c = pool[[scoring.slot_allows(s, p) and g not in used for p, g in zip(pool["position"], pool["gsis_id"])]]
            if c.empty:
                rows.append({"slot": s})
                continue
            r = c.iloc[0].to_dict()
            used.add(r["gsis_id"])
            r["slot"] = s
            if key == "call":
                r.update(e_pick="e", m_pick="m", reason=f"call for {r['name']}")
            rows.append(r)
        out[key] = pd.DataFrame(rows)
    return out


def make_root(players=PLAYERS, injured=()) -> Path:
    root = Path(tempfile.mkdtemp())
    (root / "data").mkdir()
    ld = root / "leagues" / SLUG
    ld.mkdir(parents=True)
    (ld / "league.json").write_text(json.dumps(META))
    rows = [{"gsis_id": g, "sleeper_id": str(100 + i), "name": n, "position": pos, "team": t, "E_pts": e, "E_pts_model": m,
             "season": 2026, "week": 5, "injury_status": "Out" if g in injured else ""}
            for i, (g, n, pos, t, e, m, _) in enumerate(players)]
    pd.DataFrame(rows).to_csv(ld / "roster.csv", index=False)
    mp = [{"season": 2026, "week": 5, "gsis_id": g, "team": t, "kickoff_utc": k, f"sleeper_proj_{SLUG}": SLEEPER.get(g),
           "served_at": "2026-10-04T09:00:00Z"} for g, _, _, t, _, _, k in players if g != "t1"]
    pd.DataFrame(mp).to_csv(root / "data" / "model_pts.csv", index=False)       # t1 has no model row: schedule fallback below
    pd.DataFrame([{"season": 2026, "week": 5, "team": "BBB", "gameday": "2026-10-05"},
                  {"season": 2026, "week": 5, "team": "AAA", "gameday": "2026-10-04"}]).to_csv(root / "data" / "schedule.csv", index=False)
    return root


def three_with_injury(roster, slots, idp_on, status=None):
    """The stub, but 'Out' players never start (what the real optimizers do through E_pts)."""
    return stub_three(roster[roster["injury_status"].fillna("") != "Out"], slots, idp_on)


def rec(root, now, fn=stub_three):
    return L.record(root, now=pd.Timestamp(now), three_lineups=fn)


def opt(root):
    return L._read_str(root / "data" / L.OPT_FILE)


def names(df, basis):
    return df[df["basis"] == basis]["name"].tolist()


class Record(unittest.TestCase):
    def test_first_record_writes_four_lineups_with_kickoffs(self):
        root = make_root()
        rec(root, "2026-10-04T10:00:00Z")
        d = opt(root)
        self.assertEqual(sorted(d["basis"].unique()), ["call", "e_pts", "model", "sleeper"])
        self.assertEqual(set(d["source"]), {"live"})
        self.assertEqual(names(d, "e_pts"), ["Quinn", "Rex", "Ross"])           # QB, RB, FLEX (Ross 12 beats Wade 11)
        self.assertEqual(names(d, "model"), ["Quinn", "Rex", "Ross"])
        self.assertEqual(names(d, "sleeper"), ["Quinn", "Rex", "Wade"])         # Sleeper: Wade 13 beats Ross 6 at FLEX
        self.assertEqual(d[d["basis"] == "e_pts"]["slot"].tolist(), ["QB", "RB", "FLEX"])
        self.assertEqual(set(d["locked"]), {0})
        self.assertEqual(d[d["name"] == "Quinn"]["kickoff_utc"].iloc[0][:16], "2026-10-04T17:00")
        self.assertIn("call for", d[d["basis"] == "call"]["reason"].iloc[0])
        self.assertEqual(d["sleeper_id"].iloc[0], "100")                            # ids stay text, never '100.0'

    def test_sleeper_basis_fills_a_player_without_a_projection_from_e_pts(self):
        root = make_root(players=[p for p in PLAYERS if p[0] != "t1"] + [("t1", "Tate", "TE", "BBB", 50.0, 6.0, LATE)])
        rec(root, "2026-10-04T10:00:00Z")
        d = opt(root)
        self.assertEqual(names(d, "sleeper"), ["Quinn", "Rex", "Tate"])         # Tate: no Sleeper row, E_pts 50 stands in at FLEX
        self.assertEqual(d.query("basis == 'sleeper'")["slot"].tolist(), ["QB", "RB", "FLEX"])

    def test_schedule_fallback_for_a_player_with_no_model_row(self):
        root = make_root(players=[p for p in PLAYERS if p[0] != "t1"] + [("t1", "Tate", "TE", "CCC", 50.0, 50.0, LATE)])
        pd.DataFrame([{"season": 2026, "week": 5, "team": "CCC", "gameday": "2026-10-05"}]).to_csv(
            root / "data" / "schedule.csv", mode="a", header=False, index=False)
        rec(root, "2026-10-04T10:00:00Z")
        k = opt(root).query("name == 'Tate'")["kickoff_utc"].iloc[0]
        self.assertTrue(k.startswith("2026-10-05T17:00"), k)                     # schedule gameday at 17:00 UTC (team CCC has no model rows at all)

    def test_rerecord_keeps_locked_rows_swaps_injured_and_blocks_kicked_off_bench(self):
        root = make_root()
        rec(root, "2026-10-04T10:00:00Z")
        before = opt(root)
        # Sunday 12:30 ET-ish: the early games (AAA) have kicked off. Ross (BBB, late) is now Out; Wes (AAA, bench, early) has already
        # kicked off and must not be added; Wade (late) is the only legal FLEX.
        root2 = make_root(injured=("r2",))
        (root / "leagues" / SLUG / "roster.csv").write_text((root2 / "leagues" / SLUG / "roster.csv").read_text())
        rec(root, "2026-10-04T18:00:00Z", three_with_injury)
        d = opt(root)
        for basis in ("e_pts", "model", "call"):
            rows = d[d["basis"] == basis]
            self.assertEqual(rows["name"].tolist(), ["Quinn", "Rex", "Wade"], basis)         # Ross swapped out, Wes (kicked off) never enters
            self.assertEqual(rows["locked"].tolist(), [1, 1, 0])
            old = before[before["basis"] == basis].iloc[:2].reset_index(drop=True)
            new = rows.iloc[:2].reset_index(drop=True)
            pd.testing.assert_frame_equal(old.drop(columns=["locked"]), new.drop(columns=["locked"]))      # locked rows untouched, recorded_at too
        self.assertEqual(len(d), 12)                                                 # replaced the partition, no duplicates
        self.assertEqual(d[d["basis"] == "call"]["reason"].iloc[0], "call for Quinn")

    def test_first_record_after_kickoff_stamps_locked(self):
        root = make_root()
        rec(root, "2026-10-04T18:00:00Z")                                           # nothing recorded before: the whole roster is the pool
        d = opt(root).query("basis == 'e_pts'")
        self.assertEqual(d["name"].tolist(), ["Quinn", "Rex", "Ross"])
        self.assertEqual(d["locked"].tolist(), [1, 1, 0])

    def test_rerecord_is_idempotent(self):
        root = make_root()
        rec(root, "2026-10-04T18:00:00Z")
        first = (root / "data" / L.OPT_FILE).read_bytes()
        rec(root, "2026-10-04T18:00:00Z")
        self.assertEqual((root / "data" / L.OPT_FILE).read_bytes(), first)


def played_rows(rid_mine=1, with_ross=False):
    """lineups_played for league 111 week 5: his started lineup and a bench; Ross is not in the file (missing)."""
    base = {"season": 2026, "week": 5, "league_id": LID, "league_name": "League A", "espn_id": ""}
    rows = [dict(base, roster_id=rid_mine, sleeper_id="100", gsis_id="q1", position="QB", started=1, slot="QB", points=20.0),
            dict(base, roster_id=rid_mine, sleeper_id="101", gsis_id="r1", position="RB", started=1, slot="RB", points=10.0),
            dict(base, roster_id=rid_mine, sleeper_id="103", gsis_id="w1", position="WR", started=1, slot="FLEX", points=4.0),
            dict(base, roster_id=rid_mine, sleeper_id="104", gsis_id="w2", position="WR", started=0, slot="", points=30.0),
            dict(base, roster_id=2, sleeper_id="999", gsis_id="z", position="RB", started=1, slot="RB", points=99.0)]
    if with_ross:
        rows.append(dict(base, roster_id=rid_mine, sleeper_id="102", gsis_id="r2", position="RB", started=0, slot="", points=8.0))
    return pd.DataFrame(rows)


class Score(unittest.TestCase):
    def setUp(self):
        self.root = make_root()
        rec(self.root, "2026-10-04T10:00:00Z")
        played_rows().to_csv(self.root / "data" / "lineups_played.csv", index=False)

    def test_totals_missing_player_and_actual(self):
        res = L.score(self.root)
        t = res.set_index("basis")
        self.assertEqual(set(t.index), {"e_pts", "model", "sleeper", "call", "actual", "perfect"})
        # recorded Quinn 20 + Rex 10 + Ross (no played row) 0
        self.assertEqual((t.loc["e_pts", "points"], t.loc["e_pts", "n_missing"], t.loc["e_pts", "n_slots"]), (30.0, 1, 3))
        self.assertEqual(t.loc["call", "offense_points"], 30.0)
        # sleeper: Quinn 20 + Rex 10 + Wade 4
        self.assertEqual((t.loc["sleeper", "points"], t.loc["sleeper", "n_missing"]), (34.0, 0))
        # perfect: Quinn 20 + Rex 10 + FLEX Wes 30 (benched, outscored starter Wade 4)
        self.assertEqual((t.loc["perfect", "points"], t.loc["perfect", "offense_points"], t.loc["perfect", "n_slots"]), (60.0, 60.0, 3))
        self.assertEqual(t.loc["perfect", "efficiency"], 1.0)
        self.assertAlmostEqual(t.loc["e_pts", "efficiency"], 0.5)
        self.assertAlmostEqual(t.loc["sleeper", "efficiency"], 34 / 60, places=4)
        self.assertAlmostEqual(t.loc["actual", "offense_efficiency"], 34 / 60, places=4)
        self.assertEqual(set(t["source"]), {"live"})
        # his started: 20 + 10 + 4
        self.assertEqual((t.loc["actual", "points"], t.loc["actual", "n_slots"], t.loc["actual", "n_missing"]), (34.0, 3, 0))
        self.assertTrue((self.root / "data" / L.EVAL_FILE).exists())

    def test_idempotent_and_replaces_the_week(self):
        L.score(self.root)
        first = (self.root / "data" / L.EVAL_FILE).read_bytes()
        L.score(self.root)
        self.assertEqual((self.root / "data" / L.EVAL_FILE).read_bytes(), first)
        played_rows(with_ross=True).to_csv(self.root / "data" / "lineups_played.csv", index=False)      # Ross now has a row
        res = L.score(self.root)
        ev = pd.read_csv(self.root / "data" / L.EVAL_FILE)
        self.assertEqual(len(ev), 6)
        self.assertEqual(ev.query("basis == 'e_pts'")["points"].iloc[0], 38.0)
        self.assertEqual(ev.query("basis == 'e_pts'")["n_missing"].iloc[0], 0)

    def test_an_incomplete_week_is_not_scored(self):
        (self.root / "data" / "lineups_played.csv").write_text("season,week,league_id,roster_id,started,points,position,sleeper_id\n")
        self.assertTrue(L.score(self.root).empty)
        self.assertFalse((self.root / "data" / L.EVAL_FILE).exists())

    def test_report_section(self):
        self.assertIn("No weeks scored yet", L.section(self.root))
        L.score(self.root)
        text = L.section(self.root)
        self.assertIn(L.TITLE, text)
        self.assertIn("| actual | 34.0 | 34.0 | 56.7% | 56.7% | 1 | - | - |", text)
        self.assertIn("| e_pts | 30.0 | 30.0 | 50.0% | 50.0% | 1 | 0 | -4.0 |", text)
        self.assertIn("| sleeper | 34.0 | 34.0 | 56.7% | 56.7% | 1 | 1 | +0.0 |", text)
        self.assertIn("| perfect | 60.0 | 60.0 | 100.0% | 100.0% | 1 | - | +26.0 |", text)
        self.assertIn("| 2026 wk05 | 30.0 (50.0%) | 30.0 (50.0%) | 34.0 (56.7%) | 30.0 (50.0%) | 34.0 (56.7%) | 60.0 (100.0%) | sleeper |", text)
        self.assertIn("Selected team efficiency, season to date", text)
        self.assertIn("sleeper 56.7% (offense 56.7%)", text)
        self.assertEqual(text, L.section(self.root))                                # no clock

    def test_scoreboard_embeds_the_section(self):
        from model import scoreboard as S
        L.score(self.root)
        text = S.run(self.root, write=False)
        self.assertIn(L.TITLE, text)
        self.assertLess(text.index(L.TITLE), text.index("## Limits"))


class Perfect(unittest.TestCase):
    def test_a_benched_player_who_outscored_a_starter_is_in_the_perfect_lineup(self):
        team = played_rows()
        team = team[team["roster_id"] == 1]
        total, n = L.perfect_lineup(team, ["QB", "RB", "FLEX"], False)
        self.assertEqual((total, n), (60.0, 3))                                   # Wes (bench, 30) takes FLEX from Wade (4)
        off, _ = L.perfect_lineup(team.assign(position=team["position"].where(team["sleeper_id"] != "104", "DL")),
                                  ["QB", "RB", "FLEX"], False, offense_only=True)
        self.assertEqual(off, 34.0)                                              # a non-offense player is not in the offense lineup


def backfill_root():
    """Week 5 recorded live (kept); week 4 complete in lineups_played with no lineup: to be backfilled from frozen rows."""
    root = make_root()
    rec(root, "2026-10-04T10:00:00Z")
    mp = pd.read_csv(root / "data" / "model_pts.csv", dtype={"gsis_id": str})
    e = {g: ee for g, _, _, _, ee, _, _ in PLAYERS}
    m = {g: mm for g, _, _, _, _, mm, _ in PLAYERS}
    rows = []
    for g in ("q1", "r1", "w2", "r2", "w1"):
        early = g in ("q1", "r1", "w2")
        late_serve = g == "w2"                                                   # Wes's row was served after his kickoff: not frozen
        rows.append({"season": 2026, "week": 4, "gsis_id": g, "team": "AAA" if early else "BBB",
                     "kickoff_utc": "2026-09-27T17:00:00+00:00" if early else "2026-09-28T00:20:00+00:00",
                     "served_at": "2026-09-27T18:00:00Z" if late_serve else "2026-09-27T09:00:00Z",
                     f"e_pts_{SLUG}": e[g], f"pts_{SLUG}": m[g], "pts_model_components": m[g], "p10": 1.0, "p90": 30.0,
                     f"sleeper_proj_{SLUG}": SLEEPER[g]})
    pd.concat([mp, pd.DataFrame(rows)], ignore_index=True).to_csv(root / "data" / "model_pts.csv", index=False)
    snap = [{"season": 2026, "week": 4, "league_id": LID, "roster_id": 1, "sleeper_id": str(100 + i), "gsis_id": g, "name": n,
             "position": pos, "team": t, "is_ir": 0, "is_taxi": 0} for i, (g, n, pos, t, *_) in enumerate(PLAYERS)]
    snap += [dict(snap[0], name="Injured Guy", gsis_id="ir1", position="RB", is_ir=1),          # on IR: cannot start
             dict(snap[0], roster_id=2, name="Not Mine", gsis_id="z", position="RB")]            # another team
    pd.DataFrame(snap).to_csv(root / "data" / "league_rosters.csv", index=False)
    lp = pd.concat([played_rows().assign(week=4), played_rows()], ignore_index=True)
    lp.to_csv(root / "data" / "lineups_played.csv", index=False)
    return root


NOW = pd.Timestamp("2026-10-30T00:00:00Z")


class Backfill(unittest.TestCase):
    def test_backfills_missing_weeks_only_marks_source_and_turns_the_call_status_rule_off(self):
        root = backfill_root()
        live_before = opt(root).copy()
        STATUS_SEEN.clear()
        new = L.backfill(root, now=NOW, three_lineups=stub_three)
        self.assertEqual(set(new["week"]), {4})                                  # week 5 is live-recorded: never touched
        self.assertEqual(set(new["source"]), {"backfill"})
        self.assertEqual(sorted(new["basis"].unique()), ["call", "e_pts", "model", "sleeper"])
        d = opt(root)
        wk5 = d[d["week"] == 5].reset_index(drop=True)
        self.assertEqual(set(wk5["source"]), {"live"})
        pd.testing.assert_frame_equal(wk5, live_before[live_before["week"] == 5].reset_index(drop=True))
        w4 = d[d["week"] == 4]
        self.assertEqual(set(w4["locked"]), {1})
        # frozen rows only: Wes (served after kickoff) and the IR / other-team players are not candidates; Tate has no row
        self.assertEqual(names(w4, "e_pts"), ["Quinn", "Rex", "Ross"])
        self.assertEqual(names(w4, "sleeper"), ["Quinn", "Rex", "Wade"])
        for gone in ("Wes", "Injured Guy", "Not Mine", "Tate"):
            self.assertNotIn(gone, set(w4["name"]))
        # the call is built with an explicit empty status frame; the other bases are not given one
        self.assertEqual(sum(1 for x in STATUS_SEEN if isinstance(x, pd.DataFrame) and x.empty), 1)
        # idempotent: a second backfill changes nothing
        before = (root / "data" / L.OPT_FILE).read_bytes()
        L.backfill(root, now=NOW, three_lineups=stub_three)
        self.assertEqual((root / "data" / L.OPT_FILE).read_bytes(), before)

    def test_score_covers_backfilled_weeks_and_the_report_footnotes_them(self):
        root = backfill_root()
        L.backfill(root, now=NOW, three_lineups=stub_three)
        res = L.score(root)
        self.assertEqual(sorted(res["week"].unique()), [4, 5])
        self.assertEqual(set(res[res["week"] == 4]["source"]), {"backfill"})
        self.assertEqual(set(res[res["week"] == 5]["source"]), {"live"})
        text = L.section(root)
        self.assertIn("| 2026 wk04 (b) |", text)
        self.assertIn("| 2026 wk05 |", text)
        self.assertIn("Backfilled `call` lineups had no status rule", text)

    def test_a_week_whose_frozen_rows_were_overwritten_is_not_backfilled(self):
        root = backfill_root()
        mp = pd.read_csv(root / "data" / "model_pts.csv", dtype={"gsis_id": str})
        mp.loc[mp["week"] == 4, "served_at"] = "2026-09-27T18:00:00Z"            # every week-4 row served after its kickoff
        mp.to_csv(root / "data" / "model_pts.csv", index=False)
        skipped = []
        new = L.backfill(root, now=NOW, three_lineups=stub_three, skipped=skipped)
        self.assertTrue(new.empty)
        self.assertEqual(skipped[0][:3], (SLUG, 2026, 4))

    def test_main_runs_record_then_backfill_then_score(self):
        calls = []
        empty = pd.DataFrame(columns=L.OPT_COLS)
        with mock.patch.object(L, "record", side_effect=lambda *a, **k: calls.append("record") or empty), \
                mock.patch.object(L, "backfill", side_effect=lambda *a, **k: calls.append("backfill") or empty), \
                mock.patch.object(L, "score", side_effect=lambda *a, **k: calls.append("score") or pd.DataFrame()), \
                redirect_stdout(io.StringIO()):
            L.main([])
        self.assertEqual(calls, ["record", "backfill", "score"])


class Summary(unittest.TestCase):
    def setUp(self):
        self.root = backfill_root()
        L.backfill(self.root, now=NOW, three_lineups=stub_three)
        L.score(self.root)

    def test_summary_block_sits_above_the_gate_text(self):
        from model import scoreboard as S
        text = S.run(self.root, write=False)
        self.assertLess(text.index("## Summary"), text.index(S.GATE))
        head = text[:text.index(S.GATE)]
        self.assertIn("- Gate: NOT STARTED", head)
        self.assertIn("- Lineup efficiency (points / hindsight-best lineup, pooled over 2 league-weeks): e_pts ", head)
        self.assertIn("- Offense-only efficiency", head)
        self.assertIn("- Lineup wins", head)
        self.assertEqual(text, S.run(self.root, write=False))                       # no clock

    def test_ask_scoreboard_prints_summary_pooled_table_and_the_league_weeks(self):
        from ff import ask
        buf = io.StringIO()
        with mock.patch.object(ask, "ROOT", self.root), redirect_stdout(buf):
            ask.main(["scoreboard", SLUG])
        out = buf.getvalue()
        self.assertLess(out.index("## Summary"), out.index("| lineup | total points"))
        self.assertLess(out.index("| lineup | total points"), out.index("| season / week |"))
        self.assertIn("| perfect |", out)
        self.assertIn("| 2026 wk04 (b) |", out)
        buf = io.StringIO()
        with mock.patch.object(ask, "ROOT", self.root), redirect_stdout(buf):
            ask.main(["scoreboard"])                                                # no slug: no per-week table
        self.assertNotIn("| season / week |", buf.getvalue())


class Warn(unittest.TestCase):
    def test_failure_is_a_warn_row_exit_zero_and_writes_nothing(self):
        root = make_root()
        (root / "logs").mkdir()
        (root / "logs" / "runs.csv").write_text("ran_at,season,week,status,fails,warns,detail,note\n"
                                                "2026-09-29T00:00:00Z,2026,5,OK,0,0,,\n")
        with mock.patch.object(L, "ROOT", root), mock.patch("model.serve.ROOT", root), \
                mock.patch.object(L, "record", side_effect=RuntimeError("boom")), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc = L.main(["record"])
        self.assertEqual(rc, 0)
        with open(root / "logs" / "runs.csv") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 2)
        self.assertEqual((rows[-1]["status"], rows[-1]["warns"]), ("WARN", "1"))
        self.assertTrue(rows[-1]["detail"].startswith("model.lineups: lineups record failed"))
        self.assertFalse((root / "data" / L.OPT_FILE).exists())
        self.assertFalse((root / "data" / L.EVAL_FILE).exists())


if __name__ == "__main__":
    unittest.main()
