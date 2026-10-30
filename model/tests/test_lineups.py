"""Recorded optimal lineups: record, lock at kickoff, score, report.

    python3.12 -m unittest model.tests.test_lineups

A synthetic repo in a temp directory (no network, nothing committed is touched). `ff.calls.three_lineups` is another module's
job; here `stub_three` stands in for it with the same contract (greedy by E_pts / E_pts_model, NaN-safe), injected into `record`.
Proven: a first record writes three lineups; a re-record after some kickoffs keeps the locked rows byte for byte, swaps an injured
unlocked player out, and never lets a kicked-off bench player in; score totals each basis by sleeper_id (a missing player counts as
0 and in n_missing) with the `actual` basis from his started rows; the report section renders; a failure is a WARN row, exit 0,
and writes nothing.
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


def stub_three(roster, slots, idp_on):
    """Same contract as ff.calls.three_lineups: greedy fill by slot order; call = the model lineup."""
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
    mp = [{"season": 2026, "week": 5, "gsis_id": g, "team": t, "kickoff_utc": k} for g, _, _, t, _, _, k in players if g != "t1"]
    pd.DataFrame(mp).to_csv(root / "data" / "model_pts.csv", index=False)       # t1 has no model row: schedule fallback below
    pd.DataFrame([{"season": 2026, "week": 5, "team": "BBB", "gameday": "2026-10-05"},
                  {"season": 2026, "week": 5, "team": "AAA", "gameday": "2026-10-04"}]).to_csv(root / "data" / "schedule.csv", index=False)
    return root


def three_with_injury(roster, slots, idp_on):
    """The stub, but 'Out' players never start (what the real optimizers do through E_pts)."""
    return stub_three(roster[roster["injury_status"].fillna("") != "Out"], slots, idp_on)


def rec(root, now, fn=stub_three):
    return L.record(root, now=pd.Timestamp(now), three_lineups=fn)


def opt(root):
    return L._read_str(root / "data" / L.OPT_FILE)


def names(df, basis):
    return df[df["basis"] == basis]["name"].tolist()


class Record(unittest.TestCase):
    def test_first_record_writes_three_lineups_with_kickoffs(self):
        root = make_root()
        rec(root, "2026-10-04T10:00:00Z")
        d = opt(root)
        self.assertEqual(sorted(d["basis"].unique()), ["call", "e_pts", "model"])
        self.assertEqual(names(d, "e_pts"), ["Quinn", "Rex", "Ross"])           # QB, RB, FLEX (Ross 12 beats Wade 11)
        self.assertEqual(names(d, "model"), ["Quinn", "Rex", "Ross"])
        self.assertEqual(d[d["basis"] == "e_pts"]["slot"].tolist(), ["QB", "RB", "FLEX"])
        self.assertEqual(set(d["locked"]), {0})
        self.assertEqual(d[d["name"] == "Quinn"]["kickoff_utc"].iloc[0][:16], "2026-10-04T17:00")
        self.assertIn("call for", d[d["basis"] == "call"]["reason"].iloc[0])
        self.assertEqual(d["sleeper_id"].iloc[0], "100")                            # ids stay text, never '100.0'

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
        self.assertEqual(len(d), 9)                                                  # replaced the partition, no duplicates
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
        self.assertEqual(set(t.index), {"e_pts", "model", "call", "actual"})
        # recorded Quinn 20 + Rex 10 + Ross (no played row) 0
        self.assertEqual((t.loc["e_pts", "points"], t.loc["e_pts", "n_missing"], t.loc["e_pts", "n_slots"]), (30.0, 1, 3))
        self.assertEqual(t.loc["call", "offense_points"], 30.0)
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
        self.assertEqual(len(ev), 4)
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
        self.assertIn("| actual | 34.0 | 34.0 | 1 | - | - | - |", text)
        self.assertIn("| e_pts | 30.0 | 30.0 | 1 | 1 | -4.0 | -4.0 |", text)
        self.assertIn("| 2026 wk05 | 30.0 | 30.0 | 30.0 | 34.0 | e_pts / model / call |", text)
        self.assertEqual(text, L.section(self.root))                                # no clock

    def test_scoreboard_embeds_the_section(self):
        from model import scoreboard as S
        L.score(self.root)
        text = S.run(self.root, write=False)
        self.assertIn(L.TITLE, text)
        self.assertLess(text.index(L.TITLE), text.index("## Limits"))


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
