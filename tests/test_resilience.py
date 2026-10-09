"""Injury reports from nflverse, bye-week noise, nflverse renames and self-reported crashes.

    python -m unittest discover -s tests -t .

No network and none of the repo's data files.
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import requests

from ff import run_weekly, sources, status, verify


# ------------------------------------------------------------------ fixtures
def players_frame(**sleeper) -> pd.DataFrame:
    """Four players on one team. Sleeper's practice field is null for all of them (as it is
    all season); `sleeper` maps gsis_id -> Sleeper injury_status."""
    ids = ["00-A", "00-B", "00-C", "00-D"]
    return pd.DataFrame({
        "gsis_id": ids, "name": ["Aaron", "Bobby", "Carl", "Dane"],
        "injury_status": [sleeper.get(i) for i in ids],
        "injury_body_part": [None] * 4, "practice_participation": [None] * 4,
        "news_updated": [None] * 4, "sleeper_depth_order": [1, 2, 3, 4]})


def depth_frame() -> pd.DataFrame:
    ids = ["00-A", "00-B", "00-C", "00-D"]
    return pd.DataFrame({"gsis_id": ids, "name": ["Aaron", "Bobby", "Carl", "Dane"],
                         "team": "KC", "position": "RB", "rank": [1, 2, 3, 4]})


def injuries_frame(rows) -> pd.DataFrame:
    """nflverse injuries_{season}: one row per player-week, no date column (2025+)."""
    return pd.DataFrame(
        [{"season": 2026, "week": wk, "gsis_id": g, "team": "KC", "position": "RB",
          "report_status": rep, "practice_status": prac,
          "report_primary_injury": None, "practice_primary_injury": None}
         for wk, g, rep, prac in rows])


DNP = "Did Not Participate In Practice"
LIM = "Limited Participation in Practice"
FULL = "Full Participation in Practice"


# ------------------------------------------------------------------ 1. injury reports reach status.csv
class InjuryReportsReachStatus(unittest.TestCase):
    def build(self, rows, week=5, **sleeper):
        return status.build(players_frame(**sleeper), depth_frame(), None, week,
                            injuries_frame(rows)).set_index("gsis_id")

    def test_current_week_reports_give_nonzero_practice_coverage_and_a_designation(self):
        st = self.build([(5, "00-A", None, FULL), (5, "00-B", "Questionable", DNP),
                         (5, "00-C", None, LIM)])
        self.assertEqual(st.practice.notna().sum(), 3)
        self.assertEqual(st.loc["00-B", "practice"], DNP)
        self.assertEqual(st.loc["00-B", "report_status"], "Questionable")
        self.assertEqual(st.loc["00-B", "status_flag"], "questionable")
        self.assertTrue(pd.isna(st.loc["00-D", "practice"]))           # no report: blank, not guessed

    def test_designation_and_practice_move_play_prob(self):
        st = self.build([(5, "00-A", None, FULL), (5, "00-B", "Questionable", DNP),
                         (5, "00-C", "Doubtful", DNP), (5, "00-D", "Out", DNP)])
        self.assertGreater(st.loc["00-A", "play_prob"], st.loc["00-B", "play_prob"])
        self.assertGreater(st.loc["00-B", "play_prob"], st.loc["00-C", "play_prob"])
        self.assertEqual(st.loc["00-D", "play_prob"], 0.02)
        self.assertEqual(st.loc["00-D", "status_flag"], "out")

    def test_designation_blind_before_and_aware_after(self):
        """The bug being closed: a Questionable/DNP player used to be 0.97 because nothing carried the report."""
        blind = status.build(players_frame(), depth_frame(), None, 5, None).set_index("gsis_id")
        aware = self.build([(5, "00-B", "Questionable", DNP)])
        self.assertEqual(blind.loc["00-B", "play_prob"], 0.97)
        self.assertLess(aware.loc["00-B", "play_prob"], 0.4)

    def test_sleepers_live_status_stays_a_second_signal(self):
        # Sleeper knows more than a Friday report: it says Out on Sunday morning
        st = self.build([(5, "00-A", "Questionable", FULL)], **{"00-A": "Out"})
        self.assertEqual(st.loc["00-A", "status_flag"], "out")
        # ...and with no nflverse report at all Sleeper alone behaves exactly as before
        st = self.build([], **{"00-B": "Doubtful"})
        self.assertEqual(st.loc["00-B", "status_flag"], "doubtful")
        self.assertEqual(st.loc["00-B", "play_prob"], 0.28)
        # the worse of the two wins, in either direction
        st = self.build([(5, "00-C", "Out", DNP)], **{"00-C": "Questionable"})
        self.assertEqual(st.loc["00-C", "status_flag"], "out")

    def test_only_last_weeks_reports_are_not_carried_over(self):
        """Same rule as the model's 418862e: last week's final designation is not this week's."""
        st = self.build([(4, "00-A", "Out", DNP), (4, "00-B", "Questionable", LIM)])
        self.assertEqual(st.practice.notna().sum(), 0)
        self.assertTrue(st.report_status.isna().all())
        self.assertEqual(st.loc["00-A", "status_flag"], "healthy")
        self.assertEqual(st.loc["00-A", "play_prob"], 0.97)

    def test_a_player_with_two_rows_in_the_week_keeps_the_newest(self):
        inj = injuries_frame([(5, "00-A", None, DNP), (5, "00-A", "Questionable", LIM)])
        inj["date_modified"] = ["2026-10-07T20:00:00Z", "2026-10-09T20:00:00Z"]
        got = status.week_reports(inj, 5).set_index("gsis_id")
        self.assertEqual(got.loc["00-A", "practice_status"], LIM)
        self.assertEqual(got.loc["00-A", "report_status"], "Questionable")

    def test_blank_strings_are_not_reports(self):
        st = self.build([(5, "00-A", " ", "")])
        self.assertTrue(st.practice.isna().all() and st.report_status.isna().all())


class PracticeCheck(unittest.TestCase):
    def check(self, practice, week=5, **kw):
        st = pd.DataFrame({"gsis_id": ["a", "b", "c"], "practice": practice})
        with tempfile.TemporaryDirectory() as tmp:
            st.to_csv(Path(tmp) / "status.csv", index=False)
            res = verify.verify(Path(tmp), 2026, week, **kw)
        return [r for r in res if r["check"] == "status.practice"][0]

    def test_coverage_is_ok(self):
        r = self.check([DNP, None, FULL], injury_rows=200)
        self.assertEqual(r["severity"], "OK")
        self.assertIn("67%", r["detail"])

    def test_warns_only_when_nflverse_has_rows_and_none_matched(self):
        r = self.check([None, None, None], injury_rows=221)
        self.assertEqual(r["severity"], "WARN")
        self.assertIn("221", r["detail"])

    def test_early_week_run_is_quiet_and_keeps_its_wording(self):
        r = self.check([None, None, None], injury_rows=0)
        self.assertEqual(r["severity"], "OK")
        self.assertIn("normal before Wednesday", r["detail"])

    def test_a_missing_file_is_a_warn_after_week_one_only(self):
        self.assertEqual(self.check([None] * 3, injury_rows=None)["severity"], "WARN")
        self.assertEqual(self.check([None] * 3, week=1, injury_rows=None)["severity"], "OK")

    def test_a_caller_that_cannot_say_keeps_the_old_reading(self):
        r = self.check([None] * 3)
        self.assertEqual(r["severity"], "WARN")
        self.assertIn("normal before Wednesday", r["detail"])


# ------------------------------------------------------------------ 5. bye teams
class ByeTeams(unittest.TestCase):
    def run_check(self, include_bye_rows=True):
        teams = ["KC", "CAR", "BUF", "MIA"]
        # as build_schedule writes it: an explicit BYE row for the team with no game
        sched = pd.DataFrame([{"season": 2026, "week": w, "team": t,
                               "opponent": "BYE" if (w == 5 and t in ("KC", "CAR")) else "XXX"}
                              for w in (4, 5) for t in teams])
        proj = []
        for t in teams:
            for i in range(60):                       # 240 rows: clears the >=200 projections.week check
                on_bye = t in ("KC", "CAR")
                proj.append({"season": 2026, "week": 5, "team": t, "gsis_id": f"{t}{i}",
                             "proj_carry_share": float("nan") if on_bye else (0.5 if i < 2 else 0.0)})
        proj = pd.DataFrame(proj)
        if not include_bye_rows:
            proj = proj[~proj.team.isin(["KC", "CAR"])]
        with tempfile.TemporaryDirectory() as tmp:
            sched.to_csv(Path(tmp) / "schedule.csv", index=False)
            proj.to_csv(Path(tmp) / "projections.csv", index=False)
            res = verify.verify(Path(tmp), 2026, 5)
        return [r for r in res if r["check"] == "projections.shares"][0]

    def test_bye_teams_are_skipped(self):
        self.assertEqual(self.run_check()["severity"], "OK")

    def test_bye_detection_reads_the_schedule(self):
        rows = [{"season": 2026, "week": 4, "team": t, "opponent": "XXX"} for t in ("BUF", "MIA", "KC", "CAR")]
        sched = pd.DataFrame(rows + [{"season": 2026, "week": 5, "team": t, "opponent": "XXX"} for t in ("BUF", "MIA")])
        self.assertEqual(verify.bye_teams(sched, 2026, 5), {"KC", "CAR"})              # absent rows count
        sched = pd.DataFrame(rows + [{"season": 2026, "week": 5, "team": t, "opponent": o}
                                     for t, o in (("BUF", "MIA"), ("MIA", "BUF"), ("KC", "BYE"), ("CAR", "bye"))])
        self.assertEqual(verify.bye_teams(sched, 2026, 5), {"KC", "CAR"})              # explicit BYE rows
        self.assertEqual(verify.bye_teams(sched, 2026, 4), set())
        self.assertEqual(verify.bye_teams(None, 2026, 5), set())

    def test_a_real_off_team_still_warns(self):
        teams = ["KC", "BUF"]
        sched = pd.DataFrame([{"season": 2026, "week": 5, "team": t, "opponent": "XXX"} for t in teams])
        proj = pd.DataFrame([{"season": 2026, "week": 5, "team": t, "gsis_id": f"{t}{i}",
                              "proj_carry_share": 0.2 if t == "BUF" else 0.5}
                             for t in teams for i in range(120)])
        with tempfile.TemporaryDirectory() as tmp:
            sched.to_csv(Path(tmp) / "schedule.csv", index=False)
            proj.to_csv(Path(tmp) / "projections.csv", index=False)
            r = [x for x in verify.verify(Path(tmp), 2026, 5) if x["check"] == "projections.shares"][0]
        self.assertEqual(r["severity"], "WARN")


# ------------------------------------------------------------------ 4. nflverse asset renames
class FakeResp:
    def __init__(self, status_code=200, body=b"", headers=None):
        self.status_code, self._body, self.headers = status_code, body, headers or {}

    def iter_content(self, n):
        yield self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


CSV = b"a,b\n1,2\n3,4\n"


def gz(b: bytes) -> bytes:
    return gzip.compress(b)


class NflverseVariants(unittest.TestCase):
    def fetch(self, published: dict, filename="games.csv", required=True, tmp=None):
        """`published` maps asset name -> body; everything else 404s."""
        asked = []

        def fake_get(url, headers=None, timeout=None, stream=None):
            name = url.rsplit("/", 1)[1]
            asked.append(name)
            return FakeResp(200, published[name]) if name in published else FakeResp(404)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(sources.requests, "get", fake_get):
            df = sources.nflverse_csv("schedules", filename, Path(tmp or d), required=required)
        return df, asked

    def test_variant_order(self):
        self.assertEqual(sources.asset_variants("games.csv"), ["games.csv", "games.csv.gz", "games.parquet"])
        self.assertEqual(sources.asset_variants("games.csv.gz"), ["games.csv.gz", "games.csv", "games.parquet"])
        self.assertEqual(sources.asset_variants("odd.json"), ["odd.json"])

    def test_the_asset_asked_for_is_used_when_it_exists(self):
        df, asked = self.fetch({"games.csv": CSV})
        self.assertEqual(len(df), 2)
        self.assertEqual(asked, ["games.csv"])

    def test_a_renamed_csv_is_found_as_csv_gz_and_logged(self):
        with self.assertLogs("ff.sources", "WARNING") as cm:
            df, asked = self.fetch({"games.csv.gz": gz(CSV)})
        self.assertEqual(df["a"].tolist(), [1, 3])
        self.assertEqual(asked, ["games.csv", "games.csv.gz"])
        self.assertTrue(any("games.csv.gz served instead" in m for m in cm.output), cm.output)

    def test_the_oct_6_direction_works_too(self):
        df, asked = self.fetch({"games.csv": CSV}, filename="games.csv.gz")
        self.assertEqual(len(df), 2)
        self.assertEqual(asked, ["games.csv.gz", "games.csv"])

    def test_parquet_is_the_last_resort(self):
        buf = io.BytesIO()
        pd.DataFrame({"a": [7]}).to_parquet(buf)
        df, asked = self.fetch({"games.parquet": buf.getvalue()})
        self.assertEqual(df["a"].tolist(), [7])
        self.assertEqual(asked, ["games.csv", "games.csv.gz", "games.parquet"])

    def test_parquet_without_pyarrow_degrades_to_missing_not_a_crash(self):
        with mock.patch.object(sources.pd, "read_parquet", side_effect=ImportError("pyarrow")):
            df, _ = self.fetch({"games.parquet": b"x"}, required=False)
        self.assertIsNone(df)

    def test_all_variants_missing_raises_for_required_and_returns_none_otherwise(self):
        with self.assertRaises(FileNotFoundError) as cm:
            self.fetch({})
        self.assertIn("games.csv.gz", str(cm.exception))
        df, asked = self.fetch({}, required=False)
        self.assertIsNone(df)
        self.assertEqual(len(asked), 3)

    def test_a_dead_network_does_not_walk_the_variants(self):
        calls = []

        def boom(url, **kw):
            calls.append(url)
            raise requests.ConnectionError("down")
        with tempfile.TemporaryDirectory() as d, mock.patch.object(sources.requests, "get", boom):
            with self.assertRaises(requests.ConnectionError):
                sources.nflverse_csv("schedules", "games.csv", Path(d))
            self.assertIsNone(sources.nflverse_csv("schedules", "games.csv", Path(d), required=False))
        self.assertEqual(len(calls), 2)               # one per call, not three

    def test_a_cached_copy_still_serves_when_the_network_is_down(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "games.csv").write_bytes(CSV)

            def boom(url, **kw):
                raise requests.ConnectionError("down")
            with mock.patch.object(sources.requests, "get", boom):
                df = sources.nflverse_csv("schedules", "games.csv", Path(d))
        self.assertEqual(len(df), 2)

    def test_load_nflverse_asks_for_the_injury_report(self):
        got = []
        with mock.patch.object(sources, "nflverse_csv",
                               lambda rel, fn, raw, required=True: got.append((rel, fn)) or None):
            sources.load_nflverse(2026, Path("x"), with_prior=False)
        self.assertIn(("injuries", "injuries_2026.csv"), got)


# ------------------------------------------------------------------ 3. crashes write a FAIL row
class CrashRow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "logs").mkdir()
        cfg = root / "config.json"
        cfg.write_text(json.dumps({"season": 2026, "sleeper_username": "", "startsit": {"trailing_weeks": 4},
                                   "watchlist": {}}))
        self.root = root
        self.patches = [mock.patch.object(run_weekly, "LOGS", root / "logs"),
                        mock.patch.object(run_weekly, "CONFIG", cfg),
                        mock.patch.object(run_weekly, "setup_logging", lambda today: None)]
        logging.disable(logging.CRITICAL)             # the tracebacks are the point of the test, not the output
        for p in self.patches:
            p.start()

    def tearDown(self):
        logging.disable(logging.NOTSET)
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def runs(self) -> pd.DataFrame:
        return pd.read_csv(self.root / "logs" / "runs.csv")

    def crash(self, exc, argv=("--season", "2026", "--week", "5", "--skip-sleeper")):
        with mock.patch.object(run_weekly.sources, "load_nflverse", side_effect=exc):
            return run_weekly.main(list(argv))

    def test_an_unhandled_exception_leaves_a_fail_row_and_exits_nonzero(self):
        rc = self.crash(FileNotFoundError("games.csv not published yet (404)\nsecond line"))
        self.assertEqual(rc, 1)
        r = self.runs()
        self.assertEqual(len(r), 1)
        row = r.iloc[0]
        self.assertEqual(row.status, "FAIL")
        self.assertGreaterEqual(row.fails, 1)
        self.assertEqual(row.detail, "FileNotFoundError: games.csv not published yet (404)")
        self.assertEqual((row.season, row.week), (2026, 5))
        self.assertEqual(row.note, "crash")

    def test_the_detail_is_truncated_to_the_existing_cap(self):
        self.crash(RuntimeError("x" * 2000))
        d = self.runs().iloc[0].detail
        self.assertEqual(len(d), verify.DETAIL_CAP)
        self.assertTrue(d.startswith("RuntimeError: xxx"))

    def test_the_row_lands_after_the_rows_already_there(self):
        verify.log_run(self.root / "logs", 2026, 5, [])
        self.crash(KeyError("boom"))
        r = self.runs()
        self.assertEqual(r.status.tolist(), ["OK", "FAIL"])
        self.assertTrue(r.iloc[1].detail.startswith("KeyError: "))

    def test_a_crash_before_the_week_is_known_borrows_the_last_rows_week(self):
        verify.log_run(self.root / "logs", 2026, 7, [])
        with mock.patch.object(run_weekly.sources, "sleeper_state", side_effect=RuntimeError("x")), \
                mock.patch.object(run_weekly, "load_config", side_effect=ValueError("bad config")):
            rc = run_weekly.main([])
        self.assertEqual(rc, 1)
        self.assertEqual(int(self.runs().iloc[-1].week), 7)

    def test_verify_only_does_not_write_the_log(self):
        with mock.patch.object(run_weekly.verify_mod, "verify", side_effect=RuntimeError("x")):
            rc = run_weekly.main(["--verify-only"])
        self.assertEqual(rc, 1)
        self.assertFalse((self.root / "logs" / "runs.csv").exists())

    def test_an_unwritable_log_still_exits_nonzero(self):
        with mock.patch.object(run_weekly.verify_mod, "log_crash", side_effect=OSError("disk")):
            self.assertEqual(self.crash(RuntimeError("x")), 1)


if __name__ == "__main__":
    unittest.main()
