"""ff.checkin: health verdicts, the miss gate, the research queue and the `record` writer.

    python -m unittest discover -s tests -t .

Every test builds tiny files in a tempdir and passes that root in; nothing touches the repo's data.
"""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from ff import checkin

NOW = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
RUN_HEAD = "ran_at,season,week,status,fails,warns,detail,note\n"
MISS_HEAD = ("season,week,gsis_id,name,position,team,league,comparator,predicted,actual,error,reason,secondary_reason,"
             "exp_vol,act_vol,exp_td,act_td,trail_snap_pct,snap_pct,margin,teammate_out_name,played,on_inactive_list\n")
CAND = ("# Candidates\n\n| date raised | candidate | evidence that raised it | status |\n|---|---|---|---|\n"
        "| 2026-10-15 | `recent:4` | old | tested |\n\n## Batch\n\ntext\n")


def write(root: Path, rel: str, text: str):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def runs(root: Path, *rows: str):
    write(root, "logs/runs.csv", RUN_HEAD + "".join(r + "\n" for r in rows))


def health_text(root: Path) -> str:
    return "\n".join(checkin.health(root, NOW))


def run(root: Path, *argv: str):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = checkin.main(list(argv), root=root)
    return code, buf.getvalue()


def seed_misses(root: Path):
    write(root, "data/model_misses.csv", MISS_HEAD
          + "2026,9,00-1,Pat Passer,QB,KC,lg-a,e_pts,20,5,15,td_variance,,30,28,2,0,0.9,0.9,3,,True,0.0\n"
          + "2026,9,00-1,Pat Passer,QB,KC,lg-a,model_components,19,5,14,td_variance,,30,28,2,0,0.9,0.9,3,,True,0.0\n"
          + "2026,9,00-1,Pat Passer,QB,KC,lg-b,e_pts,18,5,13,td_variance,,30,28,2,0,0.9,0.9,3,,True,0.0\n"
          + "2026,9,00-2,Wes Wideout,WR,MIA,lg-a,model_components,4,26,-22,volume_up,,3,9,0,2,0.4,0.6,5,,True,0.0\n"
          + "2026,8,00-3,Old Guy,RB,NE,lg-a,e_pts,3,20,-17,unexplained,,3,9,0,2,0.4,0.6,5,,True,0.0\n")


class HealthTest(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.root = Path(self._t.name)

    def tearDown(self):
        self._t.cleanup()

    def test_healthy(self):
        runs(self.root, "2026-11-12T20:00:00Z,2026,10,OK,0,0,,")
        self.assertTrue(health_text(self.root).rstrip().endswith("HEALTH: healthy"))

    def test_fail_in_newest_row(self):
        runs(self.root, "2026-11-12T20:00:00Z,2026,10,FAIL,1,0,model.serve: boom,")
        last = health_text(self.root).rstrip().splitlines()[-1]
        self.assertIn("fix: read the detail, then press Run workflow on the Actions tab", last)
        self.assertTrue(last.startswith("HEALTH: ") and "healthy" not in last)

    def test_stale(self):
        runs(self.root, "2026-11-07T20:00:00Z,2026,9,OK,0,0,,")
        last = health_text(self.root).rstrip().splitlines()[-1]
        self.assertIn("stale", last)
        self.assertIn("press Run workflow on the repo's Actions tab", last)

    def test_early_week_warns_ignored(self):
        runs(self.root,
             "2026-11-12T19:00:00Z,2026,10,WARN,0,1,ecr: no rows for 2026 week 10 yet,",
             '2026-11-12T20:00:00Z,2026,10,WARN,0,2,player_weeks: no rows yet; status.practice: 0% have practice reports (normal),')
        txt = health_text(self.root)
        self.assertTrue(txt.rstrip().endswith("HEALTH: healthy"))
        self.assertNotIn("- warn:", txt)
        self.assertIn("WARN/FAIL rows, last 7 days: 2", txt)

    def test_other_warn_is_noted_but_healthy(self):
        runs(self.root, "2026-11-12T20:00:00Z,2026,10,WARN,0,1,player_weeks.volume: few rows,")
        txt = health_text(self.root)
        self.assertIn("- warn: player_weeks.volume: few rows", txt)
        self.assertTrue(txt.rstrip().endswith("HEALTH: healthy"))

    def test_data_recency_lines(self):
        runs(self.root, "2026-11-12T20:00:00Z,2026,10,OK,0,0,,")
        write(self.root, "data/model_pts.csv",
              "season,week,served_at,frozen_before_kickoff\n2026,10,2026-11-12T19:00:00Z,True\n2026,10,2026-11-12T19:00:00Z,False\n")
        write(self.root, "data/player_weeks.csv", "season,week,gsis_id\n2026,9,x\n")
        write(self.root, "data/status.csv", "gsis_id,status_flag,report_status,inactive,inactive_known_at\n"
              "a,out,,0,\nb,healthy,,0,\nc,questionable,,1,2026-11-12T10:00:00Z\n")
        txt = health_text(self.root)
        self.assertIn("frozen_before_kickoff 50% of 2 rows (wk10)", txt)
        self.assertIn("player_weeks: latest 2026 wk9", txt)
        self.assertIn("2 with an injury designation", txt)
        self.assertIn("1 inactive=1", txt)
        self.assertIn("latest inactive_known_at 2026-11-12T10:00:00Z", txt)


class GateAndMissesTest(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.root = Path(self._t.name)
        seed_misses(self.root)

    def tearDown(self):
        self._t.cleanup()

    def test_gate_open_then_researched(self):
        code, out = run(self.root, "gate")
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "GATE: open (2026 week 9): 2 big misses to research")
        write(self.root, "model/reports/miss_research.md",
              "# Miss research\n\n## 2026 week 8\n\nResearched: 2026-11-06. x\n\n## 2026 week 9\n\nResearched: 2026-11-13. Misses researched: 2.\n")
        code, out = run(self.root, "gate")
        self.assertEqual(code, 3)
        self.assertEqual(out.strip(), "GATE: researched (2026 week 9, 2026-11-13)")

    def test_gate_other_week_section_is_not_enough(self):
        write(self.root, "model/reports/miss_research.md", "## 2026 week 8\n\nResearched: 2026-11-06.\n")
        self.assertEqual(run(self.root, "gate")[0], 0)
        self.assertEqual(run(self.root, "gate", "--week", "8")[0], 3)

    def test_misses_json_groups_player(self):
        write(self.root, "data/schedule.csv",
              "season,week,team,opponent,home,gameday,spread_line,total_line,roof\n2026,9,KC,LV,1,2026-11-08,-6.5,47.5,dome\n")
        write(self.root, "data/player_weeks.csv",
              "season,week,gsis_id,snaps,fpts_ppr\n2026,6,00-1,50,10\n2026,7,00-1,60,20\n2026,9,00-1,70,5\n")
        code, out = run(self.root, "misses", "--json")
        self.assertEqual(code, 0)
        d = json.loads(out)
        self.assertEqual([x["gsis_id"] for x in d], ["00-2", "00-1"])  # worst error first
        p = d[1]
        self.assertEqual(p["leagues"], ["lg-a", "lg-b"])
        self.assertEqual(len(p["misses"]), 3)
        self.assertEqual({m["comparator"] for m in p["misses"]}, {"e_pts", "model"})
        self.assertEqual(p["game"]["opponent"], "LV")
        self.assertTrue(p["game"]["home"])
        self.assertEqual(p["box"]["snaps"], 70)
        self.assertEqual(p["trail"]["fpts_ppr"], 15)
        self.assertIsNone(d[0]["box"])
        self.assertIsNone(d[0]["game"])
        self.assertEqual(p["retro_note"], "")

    def test_report_block_labels_both(self):
        code, out = run(self.root, "misses")
        self.assertIn("| Pat Passer | QB | KC | both |", out)


class RecordTest(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.root = Path(self._t.name)
        write(self.root, "model/candidates.md", CAND)
        self.src = self.root / "in.json"
        self.payload = {"week": 9, "season": 2026, "researched_at": "2026-11-13", "misses": [
            {"gsis_id": "00-1", "name": "Pat Passer", "position": "QB", "team": "KC",
             "why": "Left | early with a rib injury (beat report: X).", "sources": ["beat report: X"],
             "foreseeable": True, "foreseeable_signal": "limited Friday", "candidate": "+cols:foo|bar",
             "candidate_evidence": "one | two"},
            {"gsis_id": "00-2", "name": "Wes Wideout", "position": "WR", "team": "MIA", "why": "TD luck.",
             "sources": [], "foreseeable": False, "foreseeable_signal": "", "candidate": "", "candidate_evidence": ""}]}
        self.src.write_text(json.dumps(self.payload), encoding="utf-8")

    def tearDown(self):
        self._t.cleanup()

    def rec(self, *extra):
        return run(self.root, "record", "--week", "9", "--from", str(self.src), *extra)

    def test_creates_file_and_appends_candidate(self):
        code, out = self.rec()
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "recorded 2026 week 9: 2 misses, 1 candidates (0 skipped as duplicates)")
        md = (self.root / "model/reports/miss_research.md").read_text()
        self.assertIn("## 2026 week 9\n", md)
        self.assertIn("Researched: 2026-11-13. Misses researched: 2. Candidates raised: 1.", md)
        self.assertIn("Left \\| early", md)  # pipes escaped
        self.assertIn("`+cols:foo\\|bar`", md)
        self.assertEqual(run(self.root, "gate")[0], 0)  # no misses file: open, not a crash
        cand = (self.root / "model/candidates.md").read_text().splitlines()
        i = cand.index("| 2026-10-15 | `recent:4` | old | tested |")
        self.assertEqual(cand[i + 1], "| 2026-11-13 | `+cols:foo\\|bar` | one \\| two (wk 9: Pat Passer) | open |")
        self.assertIn("## Batch", cand)  # inserted inside the table, not after later sections

    def test_second_run_refused_without_force(self):
        self.rec()
        before = (self.root / "model/reports/miss_research.md").read_text()
        code, out = self.rec()
        self.assertEqual(code, 2)
        self.assertIn("--force", out)
        self.assertEqual((self.root / "model/reports/miss_research.md").read_text(), before)

    def test_force_replaces_section_and_skips_duplicate_candidate(self):
        self.rec()
        write(self.root, "model/reports/miss_research.md",
              (self.root / "model/reports/miss_research.md").read_text() + "\n## 2026 week 10\n\nkeep me\n")
        code, out = self.rec("--force")
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "recorded 2026 week 9: 2 misses, 1 candidates (1 skipped as duplicates)")
        md = (self.root / "model/reports/miss_research.md").read_text()
        self.assertEqual(md.count("## 2026 week 9\n"), 1)
        self.assertIn("keep me", md)
        self.assertEqual((self.root / "model/candidates.md").read_text().count("+cols:foo"), 1)

    def test_week_mismatch_refused(self):
        code, _ = run(self.root, "record", "--week", "8", "--from", str(self.src))
        self.assertEqual(code, 2)
        self.assertFalse((self.root / "model/reports/miss_research.md").exists())


class EscapeTest(unittest.TestCase):
    def test_esc(self):
        self.assertEqual(checkin._esc("a|b\nc"), "a\\|b c")


if __name__ == "__main__":
    unittest.main()
