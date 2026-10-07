"""Phase 4: the player-props archive (The Odds API -> data/props.csv). Archive only; nothing reads it yet.

    python3.12 -m unittest model.tests.test_props

No network and no key: api.the-odds-api.com is blocked from the development sandbox and ODDS_API_KEY does not exist yet, so every
request goes through an injected stand-in (`FakeApi`) that serves a HAND-BUILT response in the shape the API documents
(`fixtures/odds_event_odds.sample.json`, not a recording; the live path is proven by the first keyed Actions run). What is proven:

  * the parser: Over/Under and anytime-TD Yes prices land side by side, one row per (book, market, player, line); unreadable outcomes are
    counted, unknown keys ignored
  * the crosswalk: names resolve to gsis ids through the repo's own matcher, within the game's two teams first; an unknown or ambiguous name
    keeps its raw name with a blank id and the row is NEVER dropped
  * events -> schedule games (week, nflverse game id), including a reversed home/away listing and an unknown team
  * which games are due, replayed over the real week-5 slate and the five weekly run times: exactly one snapshot per game, the
    credits that costs (the budget math written into PLAN_MODEL.md), and no double spend on a re-run
  * frozen per game: a game that has kicked off is never rewritten, a game not fetched this run keeps its last snapshot, a third
    run of the same week duplicates nothing, a rerun is byte-identical, and it goes through ff.build.replace_partition
  * the budget: credit headers logged, soonest games funded first, a WARN when the money runs out or runs low, a quota / refused /
    failing API stops the run after one or two requests
  * the key never appears in stdout, stderr, props.csv or logs/runs.csv, even inside a connection error that quotes the URL
  * no key means one WARN line, no request, no file, no run-log row, exit 0; any failure is one WARN row, nothing written, exit 0
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from model import fetch_props as P

REPO = Path(__file__).resolve().parents[2]
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "odds_event_odds.sample.json").read_text(encoding="utf-8"))
KEY = "sk_test_SECRET_7f3a9c1d"
NAME_OF = {v: k for k, v in P.TEAM_ABBR.items()}

# The 2026 week-5 slate as scheduled on 2026-10-07 (nflverse): Thursday night, an early international game, the 1 pm slate, the late
# games, Sunday night and Monday night. Hard-coded so the replay below does not move when a later flex changes the live schedule.
WEEK5 = {"2026_05_TB_DAL": "2026-10-09T00:15:00Z", "2026_05_PHI_JAX": "2026-10-11T13:30:00Z",
         "2026_05_IND_PIT": "2026-10-11T17:00:00Z", "2026_05_CIN_MIA": "2026-10-11T17:00:00Z", "2026_05_CHI_GB": "2026-10-11T17:00:00Z",
         "2026_05_LV_NE": "2026-10-11T17:00:00Z", "2026_05_CLE_NYJ": "2026-10-11T17:00:00Z", "2026_05_NYG_WAS": "2026-10-11T17:00:00Z",
         "2026_05_MIN_NO": "2026-10-11T17:00:00Z", "2026_05_HOU_TEN": "2026-10-11T17:00:00Z", "2026_05_DEN_LAC": "2026-10-11T20:05:00Z",
         "2026_05_SF_SEA": "2026-10-11T20:25:00Z", "2026_05_DET_ARI": "2026-10-11T20:25:00Z", "2026_05_BAL_ATL": "2026-10-12T00:20:00Z",
         "2026_05_BUF_LA": "2026-10-13T00:15:00Z"}
# The weekly workflow's run times for that week (UTC): the Tuesday main pull and backstop, Wednesday, Friday, and the Sunday pre-lock run.
RUNS = {"Tue 18:37": "2026-10-06T18:37:00Z", "Tue 19:41": "2026-10-06T19:41:00Z", "Wed 14:07": "2026-10-07T14:07:00Z",
        "Fri 22:11": "2026-10-09T22:11:00Z", "Sun 15:52": "2026-10-11T15:52:00Z"}


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)                                   # every instant in this file is written with a trailing Z: tz-aware UTC


def event_for(game_id: str, kick: str, reverse: bool = False) -> dict:
    _, _, away, home = game_id.split("_")
    h, a = (away, home) if reverse else (home, away)
    return {"id": "evt-" + game_id, "sport_key": "americanfootball_nfl", "sport_title": "NFL", "commence_time": kick,
            "home_team": NAME_OF[h], "away_team": NAME_OF[a]}


def payload_for(event: dict) -> dict:
    p = copy.deepcopy(FIXTURE)
    p.update({k: event[k] for k in ("id", "commence_time", "home_team", "away_team")})
    return p


def args(*extra: str, now: str | None = None) -> argparse.Namespace:
    return P.make_parser().parse_args(list(extra) + (["--now", now] if now else []))


class Resp:
    def __init__(self, status: int = 200, body=None, headers: dict | None = None, text: str | None = None):
        self.status_code, self.body, self.headers = status, body, headers or {}
        self.text = text if text is not None else json.dumps(body)

    def json(self):
        if self.body is None:
            raise ValueError("no body")
        return self.body


class FakeApi:
    """A stand-in for requests.get. Serves the events listing and, per event, the fixture payload; bills one credit per market; records
    every call. `script` maps an event id to a Resp (or an exception) to serve instead."""

    def __init__(self, events, remaining: int = 500, script: dict | None = None, events_resp: Resp | None = None, headers: bool = True):
        self.events, self.remaining, self.used = events, remaining, 500 - remaining
        self.script, self.events_resp, self.headers = script or {}, events_resp, headers
        self.calls: list[tuple[str, dict]] = []

    def credit_headers(self, last: int) -> dict:
        return {"x-requests-remaining": str(self.remaining), "x-requests-used": str(self.used), "x-requests-last": str(last)} if self.headers else {}

    def __call__(self, url, params=None, timeout=None, headers=None):
        path = url.replace(P.HOST, "")
        self.calls.append((path, dict(params or {})))
        if path.endswith("/events"):
            return self.events_resp or Resp(200, self.events, self.credit_headers(0))
        eid = path.split("/")[-2]
        if eid in self.script:
            got = self.script[eid]
            if isinstance(got, BaseException):
                raise got
            return got
        ev = next(e for e in self.events if e["id"] == eid)
        cost = len(params["markets"].split(","))
        self.remaining -= cost
        self.used += cost
        return Resp(200, payload_for(ev), self.credit_headers(cost))

    @property
    def paid(self) -> list[str]:
        return [p.split("/")[-2] for p, _ in self.calls if not p.endswith("/events")]


def make_root(runs_row: bool = True) -> Path:
    """A temp repo with the real schedule and player table, a config, and a run log whose newest row is the pipeline's week-5 run."""
    root = Path(tempfile.mkdtemp())
    (root / "data").mkdir()
    for f in ("schedule.csv", "players.csv"):
        shutil.copy(REPO / "data" / f, root / "data" / f)
    (root / "config.json").write_text('{"season": 2026}')
    if runs_row:
        (root / "logs").mkdir()
        (root / "logs" / "runs.csv").write_text("ran_at,season,week,status,fails,warns,detail,note\n2026-10-06T18:40:00Z,2026,5,OK,0,0,,\n")
    return root


def week5_events(**kw) -> list[dict]:
    return [event_for(g, k, **kw) for g, k in WEEK5.items()]


def quiet(fn, *a, **k):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        r = fn(*a, **k)
    return r, out.getvalue(), err.getvalue()


def runs_rows(root: Path) -> list[dict]:
    with open(root / "logs" / "runs.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_props(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "data" / "props.csv", dtype={"gsis_id": str})


class Scrubbing(unittest.TestCase):
    def test_the_key_and_anything_shaped_like_an_api_key_parameter_is_replaced(self):
        url = f"https://api.the-odds-api.com/v4/sports/x/events?apiKey={KEY}&regions=us"
        out = P.scrub(f"HTTPSConnectionPool: Max retries exceeded with url: {url} (Caused by ...)", KEY)
        self.assertNotIn(KEY, out)
        self.assertIn("apiKey=***&regions=us", out)
        self.assertNotIn("abc123", P.scrub("GET /x?apikey=abc123 failed"))              # a key we were never told about, same shape
        self.assertEqual(P.scrub("plain text with no secret", KEY), "plain text with no secret")
        self.assertNotIn(KEY, P.scrub(f"the key {KEY} is not valid", KEY))                # bare, with no apiKey= in front of it
        self.assertNotIn(KEY, P.scrub(f'{{"message": "bad key {KEY}"}}', KEY))


class Headers(unittest.TestCase):
    def test_the_credit_headers_are_read_case_insensitively_and_missing_ones_are_just_absent(self):
        self.assertEqual(P.credits_from({"X-Requests-Remaining": "479", "x-requests-used": "21", "X-REQUESTS-LAST": "6"}),
                         {"remaining": 479, "used": 21, "last": 6})
        self.assertEqual(P.credits_from({"x-requests-remaining": "12.0"}), {"remaining": 12})
        self.assertEqual(P.credits_from({"x-requests-remaining": "n/a"}), {})
        self.assertEqual(P.credits_from(None), {})


class ApiGet(unittest.TestCase):
    def test_a_connection_error_that_quotes_the_url_does_not_carry_the_key_out(self):
        def boom(url, params=None, **k):
            raise ConnectionError(f"HTTPSConnectionPool(host='api.the-odds-api.com'): Max retries exceeded with url: /v4/x?apiKey={params['apiKey']}")
        sleeps: list = []
        r = P.api_get("/v4/x", {}, KEY, get=boom, retries=3, sleep=sleeps.append)
        self.assertIsNone(r.status)
        self.assertNotIn(KEY, r.error)
        self.assertIn("ConnectionError", r.error)
        self.assertEqual(sleeps, [1, 2])                                                # 3 attempts, backoff between them, none after the last

    def test_it_retries_what_can_pass_and_never_a_refusal(self):
        seq = [Resp(503, None, {}, "busy"), Resp(429, {"error_code": "EXCEEDED_FREQ_LIMIT"}), Resp(200, [1], {"x-requests-remaining": "9"})]
        calls = []

        def get(url, **k):
            calls.append(url)
            return seq[len(calls) - 1]
        r = P.api_get("/v4/x", {}, KEY, get=get, sleep=lambda s: None)
        self.assertEqual((r.status, r.data, r.credits, len(calls)), (200, [1], {"remaining": 9}, 3))
        for status in (401, 403, 404, 422):
            calls.clear()
            r = P.api_get("/v4/x", {}, KEY, get=lambda url, **k: calls.append(url) or Resp(status, {"message": "no"}), sleep=lambda s: None)
            self.assertEqual((r.status, len(calls)), (status, 1), status)               # one request: a bad key or market costs one request
            self.assertTrue(r.error.startswith(f"HTTP {status}"))

    def test_an_exhausted_quota_is_recognised_and_not_retried(self):
        body = {"message": "Usage quota has been reached. See https://the-odds-api.com", "error_code": "OUT_OF_USAGE_CREDITS"}
        calls = []
        r = P.api_get("/v4/x", {}, KEY, get=lambda url, **k: calls.append(url) or Resp(401, body), sleep=lambda s: None)
        self.assertTrue(r.quota)
        self.assertEqual(len(calls), 1)

    def test_the_key_goes_only_in_the_query_and_the_response_text_is_scrubbed(self):
        seen = {}

        def get(url, params=None, headers=None, **k):
            seen.update(url=url, params=params, headers=headers)
            return Resp(400, None, {}, f"bad request for apiKey={KEY}")
        r = P.api_get("/v4/x", {"a": 1}, KEY, get=get)
        r2 = P.api_get("/v4/x", {"a": 1}, KEY, get=lambda url, **k: Resp(400, None, {}, f"invalid key {KEY}"))
        self.assertNotIn(KEY, r2.error)
        self.assertEqual(seen["params"]["apiKey"], KEY)
        self.assertNotIn(KEY, seen["url"])
        self.assertNotIn(KEY, json.dumps(seen["headers"]))
        self.assertNotIn(KEY, r.error)

    def test_a_200_that_is_not_json_is_an_error_not_a_crash(self):
        r = P.api_get("/v4/x", {}, KEY, get=lambda url, **k: Resp(200, None, {}, "<html>"), sleep=lambda s: None)
        self.assertEqual(r.status, 200)
        self.assertIn("not JSON", r.error)


class Parser(unittest.TestCase):
    GAME = P.Game("evt", "2026_05_TB_DAL", 2026, 5, pd.Timestamp("2026-10-09T00:15:00Z"), "DAL", "TB")

    def parse(self, payload="fixture", books=None):
        return P.parse_event_odds(FIXTURE if payload == "fixture" else payload, self.GAME, "2026-10-08T18:10:00Z", books)

    def test_over_and_under_land_on_one_row_and_the_anytime_td_yes_is_the_over_price_with_no_line(self):
        df, bad = self.parse()
        self.assertEqual(list(df.columns), P.COLS)
        row = lambda **kw: df[np.logical_and.reduce([df[k] == v for k, v in kw.items()])]
        r = row(bookmaker="draftkings", market="player_pass_yds", player="Dak Prescott").iloc[0]
        self.assertEqual((r["line"], r["over_price"], r["under_price"]), (268.5, -115.0, -105.0))
        self.assertEqual(r["book_updated"], "2026-10-08T18:02:11Z")
        t = row(market="player_anytime_td", player="CeeDee Lamb").iloc[0]
        self.assertTrue(np.isnan(t["line"]) and np.isnan(t["under_price"]) and t["over_price"] == 135.0)
        one_sided = row(bookmaker="fanduel", market="player_receptions", player="George Pickens").iloc[0]
        self.assertTrue(one_sided["over_price"] == 120.0 and np.isnan(one_sided["under_price"]))
        fd = row(bookmaker="fanduel", market="player_pass_yds", player="Dak Prescott")
        self.assertEqual((len(fd), fd.iloc[0]["line"]), (1, 270.5))                      # each book keeps its own line

    def test_every_row_carries_the_game_and_the_fetch_stamp_and_the_kickoff_format_matches_model_pts(self):
        df, _ = self.parse()
        self.assertEqual(set(df["game_id"]), {"2026_05_TB_DAL"})
        self.assertEqual(set(df["kickoff_utc"]), {"2026-10-09T00:15:00+00:00"})
        self.assertEqual(set(df["fetched_at"]), {"2026-10-08T18:10:00Z"})
        self.assertEqual((set(df["season"]), set(df["week"])), ({2026}, {5}))
        self.assertTrue(df["gsis_id"].isna().all())                                       # the crosswalk is a separate step

    def test_unreadable_outcomes_are_counted_and_skipped_and_unknown_keys_are_ignored(self):
        df, bad = self.parse()
        self.assertEqual(bad, 1)                                                          # "Broken Row With No Price"
        self.assertNotIn("Broken Row With No Price", set(df["player"]))
        self.assertEqual(len(df), 13)
        self.assertEqual(df.groupby(["bookmaker", "market"]).size().to_dict(),
                         {("draftkings", "player_anytime_td"): 4, ("draftkings", "player_pass_yds"): 2,
                          ("draftkings", "player_reception_yds"): 3, ("fanduel", "player_pass_yds"): 1,
                          ("fanduel", "player_receptions"): 2, ("fanduel", "player_rush_yds"): 1})

    def test_a_book_allow_list_trims_rows_and_an_empty_or_odd_payload_gives_an_empty_frame(self):
        df, _ = self.parse(books={"fanduel"})
        self.assertEqual(set(df["bookmaker"]), {"fanduel"})
        for payload in ({}, {"bookmakers": None}, {"bookmakers": [{"key": "x", "markets": None}]}, None):
            e, bad = self.parse(payload)
            self.assertTrue(e.empty and bad == 0)
            self.assertEqual(list(e.columns), P.COLS)


class Crosswalk(unittest.TestCase):
    GAME = P.Game("evt", "2026_05_TB_DAL", 2026, 5, pd.Timestamp("2026-10-09T00:15:00Z"), "DAL", "TB")

    def players(self, rows):
        return pd.DataFrame(rows, columns=["gsis_id", "name", "team"])

    def frame(self, names):
        return pd.DataFrame({"player": names, "x": range(len(names))})

    def test_a_name_resolves_within_the_games_teams_even_with_a_suffix_or_punctuation_difference(self):
        cw = P.build_crosswalk(self.players([("g1", "Chris Godwin Jr.", "TB"), ("g2", "CeeDee Lamb", "DAL"), ("g3", "Amon-Ra St. Brown", "DET")]))
        out, n, n_name = P.resolve_ids(self.frame(["Chris Godwin", "CeeDee Lamb", "Cee Dee Lamb"]), self.GAME, cw)
        self.assertEqual(out["gsis_id"].tolist(), ["g1", "g2", None])                      # "Cee Dee" is a different key: left blank
        self.assertEqual((n, n_name), (2, 0))

    def test_a_name_on_another_team_in_the_table_is_taken_only_when_it_is_unique_league_wide(self):
        cw = P.build_crosswalk(self.players([("g1", "Traded Guy", "MIA"), ("g2", "Common Name", "MIA"), ("g3", "Common Name", "NYJ")]))
        out, n, n_name = P.resolve_ids(self.frame(["Traded Guy", "Common Name"]), self.GAME, cw)
        self.assertEqual(out["gsis_id"].tolist(), ["g1", None])                            # a traded player is found; an ambiguous name is not guessed
        self.assertEqual((n, n_name), (1, 1))

    def test_two_players_with_one_name_on_the_games_teams_stay_blank(self):
        cw = P.build_crosswalk(self.players([("g1", "Josh Allen", "DAL"), ("g2", "Josh Allen", "TB")]))
        out, n, _ = P.resolve_ids(self.frame(["Josh Allen"]), self.GAME, cw)
        self.assertEqual((out["gsis_id"].tolist(), n), ([None], 0))

    def test_an_unmatched_name_keeps_its_raw_name_and_its_row(self):
        df, _ = P.parse_event_odds(FIXTURE, self.GAME, "2026-10-08T18:10:00Z")
        out, n, _ = P.resolve_ids(df, self.GAME, P.build_crosswalk(read_real_players()))
        self.assertEqual(len(out), len(df))                                                # nothing dropped
        miss = out[out["gsis_id"].isna()]
        self.assertEqual(set(miss["player"]), {"Mack Hollins Jr. (not a player we track)"})
        self.assertTrue(out.loc[out["player"] == "CeeDee Lamb", "gsis_id"].eq("00-0036358").all())
        self.assertTrue(out.loc[out["player"] == "Chris Godwin", "gsis_id"].eq("00-0033921").all())      # "Chris Godwin" -> "Chris Godwin Jr."

    def test_no_player_table_means_blank_ids_and_every_row_kept(self):
        df, _ = P.parse_event_odds(FIXTURE, self.GAME, "t")
        out, n, _ = P.resolve_ids(df, self.GAME, P.build_crosswalk(None))
        self.assertEqual((len(out), n), (len(df), 0))


def read_real_players() -> pd.DataFrame:
    return pd.read_csv(REPO / "data" / "players.csv", usecols=["gsis_id", "name", "team"], dtype={"gsis_id": str}, low_memory=False)


class ScheduleAndEvents(unittest.TestCase):
    def test_the_game_ids_built_from_the_schedule_are_the_ids_model_pts_already_uses(self):
        sched = P.schedule_games(REPO, 2026)
        ids = set(sched["game_id"])
        mp = pd.read_csv(REPO / "data" / "model_pts.csv", usecols=["game_id"])
        self.assertTrue(set(mp["game_id"]) <= ids, sorted(set(mp["game_id"]) - ids))
        self.assertEqual(len(sched), 272)
        self.assertEqual(sched["week"].value_counts().sort_index().tolist(), [16, 16, 16, 16, 15, 14, 14, 14, 15, 14, 13, 16, 14, 15, 16, 16, 16, 16])
        self.assertTrue(set(WEEK5) <= ids)

    def test_every_week5_event_matches_its_game_and_a_reversed_listing_still_matches_the_schedules_orientation(self):
        sched = P.schedule_games(REPO, 2026)
        games, notes = P.match_events(week5_events(), sched, 2026)
        self.assertEqual((len(games), notes), (15, []))
        self.assertEqual({g.game_id for g in games}, set(WEEK5))
        self.assertEqual({g.week for g in games}, {5})
        rev, _ = P.match_events([event_for("2026_05_PHI_JAX", WEEK5["2026_05_PHI_JAX"], reverse=True)], sched, 2026)
        self.assertEqual((rev[0].game_id, rev[0].home, rev[0].away), ("2026_05_PHI_JAX", "JAX", "PHI"))
        self.assertEqual(rev[0].kickoff, pd.Timestamp("2026-10-11T13:30:00Z"))

    def test_an_unknown_team_a_game_not_on_the_schedule_and_a_broken_event_are_skipped_and_named(self):
        sched = P.schedule_games(REPO, 2026)
        bad = [{"id": "x1", "commence_time": "2026-10-11T17:00:00Z", "home_team": "Toronto Argonauts", "away_team": "Dallas Cowboys"},
               {"id": "x2", "commence_time": "2027-01-17T18:00:00Z", "home_team": "Dallas Cowboys", "away_team": "Tampa Bay Buccaneers"},
               {"id": "x3", "home_team": "Dallas Cowboys"}]
        games, notes = P.match_events(bad, sched, 2026)
        self.assertEqual(games, [])
        self.assertEqual(len(notes), 3)
        self.assertIn("Toronto Argonauts", notes[0])

    def test_two_meetings_of_one_pair_are_told_apart_by_the_gameday(self):
        sched = pd.DataFrame({"game_id": ["2026_03_TB_DAL", "2026_12_DAL_TB"], "week": [3, 12], "home": ["DAL", "TB"], "away": ["TB", "DAL"],
                              "gameday": ["2026-09-27", "2026-11-29"]})
        ev = lambda t: {"id": "e", "commence_time": t, "home_team": "Tampa Bay Buccaneers", "away_team": "Dallas Cowboys"}
        self.assertEqual(P.match_events([ev("2026-11-29T18:00:00Z")], sched, 2026)[0][0].game_id, "2026_12_DAL_TB")
        # 2026-09-28T00:20Z is still Sunday 20:20 in New York: the gameday is the Eastern date
        self.assertEqual(P.match_events([ev("2026-09-28T00:20:00Z")], sched, 2026)[0][0].game_id, "2026_03_TB_DAL")


def games_of_week5() -> list:
    return P.match_events(week5_events(), P.schedule_games(REPO, 2026), 2026)[0]


class DueGames(unittest.TestCase):
    def test_the_five_weekly_runs_fetch_every_game_of_week_5_exactly_once_at_the_last_run_before_kickoff(self):
        games, snaps, fetched_at = games_of_week5(), {}, {}
        for label, t in RUNS.items():
            now = ts(t)
            for g in P.due_games(games, snaps, now):
                self.assertNotIn(g.game_id, fetched_at, f"{g.game_id} fetched twice (again at {label})")
                fetched_at[g.game_id] = label
                snaps[g.game_id] = now
        self.assertEqual(set(fetched_at), set(WEEK5))
        self.assertEqual(fetched_at["2026_05_TB_DAL"], "Wed 14:07")                       # Thursday night: the last run before it
        self.assertEqual(fetched_at["2026_05_PHI_JAX"], "Fri 22:11")                      # the 13:30 UTC game: Sunday's run is too late for it
        sunday = {g for g, run in fetched_at.items() if run == "Sun 15:52"}
        self.assertEqual(sunday, set(WEEK5) - {"2026_05_TB_DAL", "2026_05_PHI_JAX"})     # 1 pm slate through Monday night, one hour before the lock
        self.assertEqual(len(sunday), 13)

    def test_tuesdays_runs_buy_nothing_and_the_credits_for_the_week_are_games_times_markets(self):
        games = games_of_week5()
        self.assertEqual(P.due_games(games, {}, ts(RUNS["Tue 18:37"])), [])
        self.assertEqual(P.due_games(games, {}, ts(RUNS["Tue 19:41"])), [])
        self.assertEqual(len(P.MARKETS), 6)
        m = P.credit_math(len(games))
        self.assertEqual((m["per_game"], m["per_week"]), (6, 90))
        self.assertEqual((m["per_month"], m["share"]), (390.0, 0.78))                      # 90 x 52/12 weeks, of the 500 free credits

    def test_the_whole_regular_season_fits_the_free_tier_one_snapshot_at_a_time(self):
        per_week = P.schedule_games(REPO, 2026)["week"].value_counts().sort_index()
        worst = P.credit_math(per_week.max())
        self.assertEqual((per_week.max(), worst["per_week"], worst["per_month"]), (16, 96, 416.0))
        self.assertEqual(int(per_week.loc[5:].sum()) * 6, 1248)                           # weeks 5-18: 208 games x 6 credits
        self.assertLess(5 * worst["per_week"], 500)                                       # even a five-slate month of full 16-game weeks

    def test_a_late_friday_run_and_a_late_wednesday_run_change_nothing_but_a_dropped_sunday_run_loses_the_sunday_games(self):
        games = games_of_week5()
        wed_late, fri_late = ts("2026-10-07T17:00:00Z"), ts("2026-10-09T23:45:00Z")        # the first Friday run went 94 minutes late
        self.assertEqual([g.game_id for g in P.due_games(games, {}, wed_late)], ["2026_05_TB_DAL"])
        self.assertEqual([g.game_id for g in P.due_games(games, {}, fri_late)], ["2026_05_PHI_JAX"])
        after_early_slate = ts("2026-10-11T17:30:00Z")                                    # a Sunday run that missed the 1 pm lock: those games are gone
        due = {g.game_id for g in P.due_games(games, {}, after_early_slate)}
        self.assertNotIn("2026_05_IND_PIT", due)
        self.assertIn("2026_05_DEN_LAC", due)

    def test_a_snapshot_younger_than_the_refresh_age_is_not_bought_again_and_an_older_one_is(self):
        games = [g for g in games_of_week5() if g.game_id == "2026_05_DEN_LAC"]
        now = ts("2026-10-11T15:52:00Z")
        self.assertEqual(P.due_games(games, {"2026_05_DEN_LAC": now - pd.Timedelta(hours=2)}, now), [])
        self.assertEqual(len(P.due_games(games, {"2026_05_DEN_LAC": now - pd.Timedelta(hours=31)}, now)), 1)
        self.assertEqual(len(P.due_games(games, {"2026_05_DEN_LAC": now - pd.Timedelta(hours=2)}, now, refresh_hours=0)), 1)

    def test_a_game_that_has_kicked_off_or_is_beyond_the_horizon_is_never_due(self):
        g = [x for x in games_of_week5() if x.game_id == "2026_05_TB_DAL"]
        self.assertEqual(P.due_games(g, {}, ts("2026-10-09T00:15:00Z")), [])              # at kickoff: frozen
        self.assertEqual(P.due_games(g, {}, ts("2026-10-09T03:00:00Z")), [])
        self.assertEqual(P.due_games(g, {}, ts("2026-10-07T08:00:00Z")), [])              # 40.25 h out
        self.assertEqual(len(P.due_games(g, {}, ts("2026-10-07T08:30:00Z"))), 1)

    def test_due_games_come_soonest_first_so_a_short_budget_funds_the_nearest(self):
        due = P.due_games(games_of_week5(), {}, ts(RUNS["Sun 15:52"]))
        kicks = [g.kickoff for g in due]
        self.assertEqual(kicks, sorted(kicks))

    def test_snapshot_times_count_only_games_that_have_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "props.csv"
            self.assertEqual(P.snapshot_times(p), {})
            pd.DataFrame({"game_id": ["a", "a", "b"], "fetched_at": ["2026-10-07T14:07:00Z", "2026-10-07T14:09:00Z", "2026-10-09T22:11:00Z"]}).to_csv(p, index=False)
            s = P.snapshot_times(p)
            self.assertEqual(s["a"], pd.Timestamp("2026-10-07T14:09:00Z"))
            self.assertEqual(set(s), {"a", "b"})


def rows_for(game_id: str, kick: str, fetched: str, price: float = -110.0, players=("P1", "P2")) -> pd.DataFrame:
    _, wk, _, _ = game_id.split("_")
    return pd.DataFrame([{"season": 2026, "week": int(wk), "game_id": game_id, "kickoff_utc": pd.Timestamp(kick).isoformat(), "gsis_id": f"00-{i}",
                          "player": p, "market": "player_receptions", "bookmaker": "draftkings", "line": 5.5, "over_price": price, "under_price": price,
                          "book_updated": fetched, "fetched_at": fetched} for i, p in enumerate(players)], columns=P.COLS)


class FrozenArchive(unittest.TestCase):
    TNF, SUN, SUN2 = "2026-10-09T00:15:00Z", "2026-10-11T17:00:00Z", "2026-10-11T20:05:00Z"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "props.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def csv_rows(self, game_id: str) -> list[str]:
        lines = self.path.read_text().splitlines()
        return [ln for ln in lines if f",{game_id}," in ln]

    def test_rows_of_a_game_that_has_kicked_off_are_never_rewritten_and_the_rest_are_replaced(self):
        wed, sun = pd.Timestamp("2026-10-07T14:07:00Z"), pd.Timestamp("2026-10-11T15:52:00Z")
        P.write_props(self.path, pd.concat([rows_for("2026_05_TB_DAL", self.TNF, "2026-10-07T14:07:00Z", -110.0),
                                            rows_for("2026_05_IND_PIT", self.SUN, "2026-10-07T14:07:00Z", -110.0)]), wed)
        tnf_before = self.csv_rows("2026_05_TB_DAL")
        self.assertEqual(len(tnf_before), 2)
        # Sunday: Thursday night is over. A fresh snapshot of the 1 pm game replaces Wednesday's; a (late) new TNF row is dropped.
        n = P.write_props(self.path, pd.concat([rows_for("2026_05_IND_PIT", self.SUN, "2026-10-11T15:52:00Z", -125.0),
                                                rows_for("2026_05_TB_DAL", self.TNF, "2026-10-11T15:52:00Z", 999.0)]), sun)
        self.assertEqual(n, 2)                                                            # only the game still to kick off was written
        self.assertEqual(self.csv_rows("2026_05_TB_DAL"), tnf_before)                     # byte for byte
        d = pd.read_csv(self.path)
        self.assertEqual(set(d.loc[d["game_id"] == "2026_05_IND_PIT", "over_price"]), {-125.0})
        self.assertEqual(len(d), 4)

    def test_a_game_not_fetched_this_run_keeps_its_last_snapshot_and_other_weeks_are_untouched(self):
        t = pd.Timestamp("2026-10-07T14:07:00Z")
        P.write_props(self.path, pd.concat([rows_for("2026_05_IND_PIT", self.SUN, "a", -110.0), rows_for("2026_05_DEN_LAC", self.SUN2, "a", -110.0)]), t)
        P.write_props(self.path, rows_for("2026_04_X_Y", "2026-10-04T17:00:00Z", "b"), pd.Timestamp("2026-10-01T00:00:00Z"))
        P.write_props(self.path, rows_for("2026_05_DEN_LAC", self.SUN2, "c", -130.0), pd.Timestamp("2026-10-11T15:52:00Z"))
        d = pd.read_csv(self.path)
        self.assertEqual(set(d.loc[d["game_id"] == "2026_05_IND_PIT", "fetched_at"]), {"a"})      # kept: not in this run
        self.assertEqual(set(d.loc[d["game_id"] == "2026_05_DEN_LAC", "fetched_at"]), {"c"})
        self.assertEqual(len(d[d["week"] == 4]), 2)

    def test_a_third_run_of_the_same_week_duplicates_nothing_and_a_rerun_is_byte_identical(self):
        runs = [("2026-10-07T14:07:00Z", ["2026_05_TB_DAL"]), ("2026-10-09T22:11:00Z", ["2026_05_PHI_JAX"]), ("2026-10-11T15:52:00Z", ["2026_05_IND_PIT", "2026_05_DEN_LAC"])]
        kick = {"2026_05_TB_DAL": self.TNF, "2026_05_PHI_JAX": "2026-10-11T13:30:00Z", "2026_05_IND_PIT": self.SUN, "2026_05_DEN_LAC": self.SUN2}
        for when, gids in runs:
            P.write_props(self.path, pd.concat([rows_for(g, kick[g], when) for g in gids]), pd.Timestamp(when))
        first = self.path.read_bytes()
        P.write_props(self.path, pd.concat([rows_for(g, kick[g], runs[-1][0]) for g in runs[-1][1]]), pd.Timestamp(runs[-1][0]))   # the Sunday run again
        self.assertEqual(self.path.read_bytes(), first)
        d = pd.read_csv(self.path)
        self.assertFalse(d.duplicated(P.KEYS).any())
        self.assertEqual(d.groupby("game_id").size().to_dict(), {g: 2 for g in kick})
        self.assertEqual(sorted(p.name for p in self.path.parent.iterdir()), ["props.csv"])      # no temp file left behind

    def test_it_goes_through_the_pipelines_replace_partition_and_nothing_is_written_for_nothing(self):
        from ff import build as ffb
        with mock.patch.object(ffb, "replace_partition", wraps=ffb.replace_partition) as rp:
            P.write_props(self.path, rows_for("2026_05_IND_PIT", self.SUN, "a"), pd.Timestamp("2026-10-07T14:07:00Z"))
        self.assertEqual(rp.call_args.args[2:], (["season", "week"], P.KEYS))
        before = self.path.read_bytes()
        self.assertEqual(P.write_props(self.path, rows_for("2026_05_IND_PIT", self.SUN, "z"), pd.Timestamp("2026-10-12T00:00:00Z")), 0)   # all kicked off
        self.assertEqual(P.write_props(self.path, pd.DataFrame(columns=P.COLS), pd.Timestamp("2026-10-07T14:07:00Z")), 0)
        self.assertEqual(self.path.read_bytes(), before)

    def test_gsis_ids_and_blank_ids_survive_the_round_trip_through_replace_partition(self):
        df = rows_for("2026_05_IND_PIT", self.SUN, "a")
        df.loc[1, "gsis_id"] = None
        P.write_props(self.path, df, pd.Timestamp("2026-10-07T14:07:00Z"))
        P.write_props(self.path, rows_for("2026_05_DEN_LAC", self.SUN2, "b"), pd.Timestamp("2026-10-07T15:00:00Z"))
        d = pd.read_csv(self.path, dtype={"gsis_id": str})
        got = d.loc[d["game_id"] == "2026_05_IND_PIT", "gsis_id"].tolist()
        self.assertEqual(got[0], "00-0")
        self.assertTrue(pd.isna(got[1]))


class Runs(unittest.TestCase):
    """The whole run, against the stand-in API and a temp repo (the real schedule and player table)."""

    def setUp(self):
        self.root = make_root()
        self.env = {P.KEY_ENV: KEY}

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def go(self, api, *extra, now="2026-10-07T14:07:00Z", env=None):
        return P.run(args(*extra, now=now), root=self.root, env=self.env if env is None else env, get=api, sleep=lambda s: None)

    def test_no_key_is_one_warn_line_no_request_no_file_no_run_log_row_and_exit_zero(self):
        api = FakeApi(week5_events())
        runs_before = (self.root / "logs" / "runs.csv").read_bytes()
        for env in ({}, {P.KEY_ENV: ""}, {P.KEY_ENV: "   "}):
            lines = self.go(api, env=env)
            self.assertEqual(len(lines), 1)
            self.assertIn("ODDS_API_KEY is not set", lines[0])
            self.assertIn("WARN", lines[0])
            self.assertIn("skipped", lines[0])
        self.assertEqual(api.calls, [])
        self.assertFalse((self.root / "data" / "props.csv").exists())
        self.assertEqual((self.root / "logs" / "runs.csv").read_bytes(), runs_before)

    def test_the_skip_is_an_actions_annotation_on_the_runner_and_main_exits_zero_with_the_real_environment(self):
        with mock.patch.dict("os.environ", {"GITHUB_ACTIONS": "true"}, clear=False):
            self.assertTrue(P.skip_line().startswith("::warning title=model.props::"))
        with mock.patch.dict("os.environ", {"GITHUB_ACTIONS": "false"}, clear=False):
            self.assertTrue(P.skip_line().startswith("model.props: WARN"))
        with mock.patch.dict("os.environ", {}, clear=False), mock.patch.object(P, "ROOT", self.root):
            import os
            os.environ.pop(P.KEY_ENV, None)
            rc, out, err = quiet(P.main, [])
        self.assertEqual(rc, 0)
        self.assertIn("not set", out)
        self.assertEqual(err, "")
        self.assertFalse((self.root / "data" / "props.csv").exists())

    def test_wednesday_buys_thursday_night_only_with_the_documented_request_and_logs_the_credit_headers(self):
        api = FakeApi(week5_events())
        lines = self.go(api)
        self.assertEqual(api.paid, ["evt-2026_05_TB_DAL"])
        path, params = [c for c in api.calls if not c[0].endswith("/events")][0]
        self.assertEqual(path, "/v4/sports/americanfootball_nfl/events/evt-2026_05_TB_DAL/odds")
        self.assertEqual(params["regions"], "us")
        self.assertEqual(params["markets"], "player_receptions,player_rush_yds,player_reception_yds,player_pass_yds,player_pass_tds,player_anytime_td")
        self.assertEqual((params["oddsFormat"], params["dateFormat"], params["apiKey"]), ("american", "iso", KEY))
        text = "\n".join(lines)
        self.assertIn("credits last=6 used=6 remaining=494", text)
        self.assertIn("credits: used 6, remaining 494", text)
        self.assertIn("15 events listed, 15 matched", text)
        d = read_props(self.root)
        self.assertEqual(set(d["game_id"]), {"2026_05_TB_DAL"})
        self.assertEqual(list(d.columns), P.COLS)
        self.assertEqual(len(runs_rows(self.root)), 1)                                    # a healthy run adds no run-log row

    def test_the_first_keyed_run_needs_no_code_change_whatever_week_it_lands_in(self):
        """The key arrives mid-season: the same code, the same command, the next scheduled run archives whatever is inside the horizon."""
        api = FakeApi(week5_events())
        self.go(api, now="2026-10-11T15:52:00Z")
        self.assertEqual(len(api.paid), 13)
        self.assertEqual(api.remaining, 500 - 13 * 6)

    def test_a_whole_week_of_runs_spends_games_times_markets_and_leaves_every_game_once(self):
        api = FakeApi(week5_events())
        per_run = {}
        for label, t in RUNS.items():
            before = len(api.paid)
            self.go(api, now=t)
            per_run[label] = len(api.paid) - before
        self.assertEqual(per_run, {"Tue 18:37": 0, "Tue 19:41": 0, "Wed 14:07": 1, "Fri 22:11": 1, "Sun 15:52": 13})
        self.assertEqual(api.remaining, 500 - 15 * 6)                                     # 90 credits for the 15-game week
        d = read_props(self.root)
        self.assertEqual(set(d["game_id"]), set(WEEK5))
        self.assertFalse(d.duplicated(P.KEYS).any())
        self.assertEqual(len(d["game_id"].unique()), 15)
        self.assertEqual(len(runs_rows(self.root)), 1)

    def test_a_manual_rerun_inside_the_refresh_age_spends_nothing_and_a_forced_third_run_duplicates_nothing(self):
        api = FakeApi(week5_events())
        self.go(api, now="2026-10-11T15:52:00Z")
        first = (self.root / "data" / "props.csv").read_bytes()
        n = len(api.paid)
        self.go(api, now="2026-10-11T16:10:00Z")
        self.assertEqual(len(api.paid), n)
        self.assertEqual((self.root / "data" / "props.csv").read_bytes(), first)
        self.go(api, "--refresh-hours", "0", now="2026-10-11T15:52:00Z")                   # forced: the same instant, so identical rows
        self.assertEqual((self.root / "data" / "props.csv").read_bytes(), first)
        self.assertFalse(read_props(self.root).duplicated(P.KEYS).any())

    def test_a_run_cut_off_midway_keeps_the_games_it_already_paid_for(self):
        class Killed(BaseException):                                                       # the step's timeout is not an Exception either
            pass
        due = [g.event_id for g in P.due_games(games_of_week5(), {}, ts("2026-10-11T15:52:00Z"))]
        api = FakeApi(week5_events(), script={due[3]: Killed()})
        with self.assertRaises(Killed):
            self.go(api, now="2026-10-11T15:52:00Z")
        self.assertEqual(len(api.paid), 4)
        self.assertEqual(len(read_props(self.root)["game_id"].unique()), 3)                # games 1-3 are on disk, none lost with game 4
        self.assertFalse(read_props(self.root).duplicated(P.KEYS).any())

    def test_a_game_under_way_is_never_fetched_again_whatever_the_horizon_and_its_snapshot_stays_frozen(self):
        api = FakeApi(week5_events())
        self.go(api, now="2026-10-07T14:07:00Z")                                           # Thursday night's Wednesday snapshot
        tnf = [ln for ln in (self.root / "data" / "props.csv").read_text().splitlines() if "2026_05_TB_DAL" in ln]
        n = len(api.paid)
        self.go(api, "--refresh-hours", "0", "--horizon-hours", "200", now="2026-10-09T00:20:00Z")      # TNF is five minutes under way
        self.assertNotIn("evt-2026_05_TB_DAL", api.paid[n:])
        after = [ln for ln in (self.root / "data" / "props.csv").read_text().splitlines() if "2026_05_TB_DAL" in ln]
        self.assertEqual(after, tnf)

    def test_rows_are_matched_to_gsis_ids_through_the_repos_player_table_and_the_unmatched_name_is_kept(self):
        api = FakeApi([event_for("2026_05_TB_DAL", WEEK5["2026_05_TB_DAL"])])
        lines = self.go(api)
        d = read_props(self.root)
        self.assertTrue(d.loc[d["player"] == "Dak Prescott", "gsis_id"].eq("00-0033077").all())
        self.assertTrue(d.loc[d["player"] == "Chris Godwin", "gsis_id"].eq("00-0033921").all())
        self.assertEqual(d.loc[d["gsis_id"].isna(), "player"].unique().tolist(), ["Mack Hollins Jr. (not a player we track)"])
        self.assertIn("rows with a gsis_id", "\n".join(lines))

    def test_dry_run_reads_only_the_free_events_endpoint_and_writes_nothing(self):
        api = FakeApi(week5_events())
        lines = self.go(api, "--dry-run", now="2026-10-11T15:52:00Z")
        self.assertEqual(api.paid, [])
        self.assertEqual(len(api.calls), 1)
        self.assertFalse((self.root / "data" / "props.csv").exists())
        self.assertIn("would fetch 2026_05_IND_PIT", "\n".join(lines))
        self.assertIn("13 due", lines[0])

    # ---------------------------------------------------------------- the budget
    def test_the_budget_is_spent_soonest_kickoff_first_and_running_out_is_a_warn_row(self):
        api = FakeApi(week5_events(), remaining=14)                                        # two games' worth of credits
        lines = self.go(api, now="2026-10-11T15:52:00Z")
        self.assertEqual(len(api.paid), 2)
        self.assertEqual(api.remaining, 2)
        d = read_props(self.root)
        self.assertEqual(len(d["game_id"].unique()), 2)
        warn = runs_rows(self.root)[-1]
        self.assertEqual((warn["status"], warn["note"]), ("WARN", "model"))
        self.assertIn("model.props: credits ran out before 11 due game(s)", warn["detail"])
        self.assertIn("model.props: credits are low: 2 left", warn["detail"])
        self.assertIn("WARN credits ran out", "\n".join(lines))
        first_two = sorted(P.due_games(games_of_week5(), {}, ts("2026-10-11T15:52:00Z")), key=lambda g: (g.kickoff, g.game_id))[:2]
        self.assertEqual(set(d["game_id"]), {g.game_id for g in first_two})

    def test_credits_below_the_threshold_warn_and_above_it_do_not(self):
        api = FakeApi(week5_events(), remaining=150)
        self.go(api, now="2026-10-07T14:07:00Z")                                           # 150 -> 144 left: fine
        self.assertEqual(len(runs_rows(self.root)), 1)
        api = FakeApi(week5_events(), remaining=104)
        self.go(api, "--refresh-hours", "0", now="2026-10-07T14:07:00Z")                   # 104 -> 98 left: near the budget
        self.assertEqual(len(runs_rows(self.root)), 2)
        self.assertIn("credits are low: 98 left", runs_rows(self.root)[-1]["detail"])
        self.go(api, now="2026-10-06T18:37:00Z")                                           # a Tuesday that buys nothing does not repeat the nag
        self.assertEqual(len(runs_rows(self.root)), 2)

    def test_a_cost_above_the_plan_is_flagged_once(self):
        class Dear(FakeApi):
            def credit_headers(self, last):
                return super().credit_headers(last * 2)
        api = Dear(week5_events())
        self.go(api, now="2026-10-11T15:52:00Z")
        detail = runs_rows(self.root)[-1]["detail"]
        self.assertEqual(detail.count("was billed 12 credits against the planned 6"), 1)

    def test_a_missing_credit_header_does_not_stop_the_archive(self):
        api = FakeApi(week5_events(), headers=False)
        lines = self.go(api)
        self.assertEqual(len(api.paid), 1)
        self.assertIn("no x-requests-remaining header", "\n".join(lines))
        self.assertTrue((self.root / "data" / "props.csv").exists())

    # ---------------------------------------------------------------- the API says no
    def test_an_exhausted_quota_stops_after_one_request_and_writes_nothing(self):
        body = {"message": "Usage quota has been reached", "error_code": "OUT_OF_USAGE_CREDITS"}
        api = FakeApi(week5_events(), script={f"evt-{g}": Resp(401, body) for g in WEEK5})
        self.go(api, now="2026-10-11T15:52:00Z")
        self.assertEqual(len(api.paid), 1)
        self.assertFalse((self.root / "data" / "props.csv").exists())
        self.assertIn("monthly credits exhausted", runs_rows(self.root)[-1]["detail"])

    def test_a_rejected_key_or_market_stops_after_one_request(self):
        for status in (401, 403, 422):
            root = make_root()
            api = FakeApi(week5_events(), script={f"evt-{g}": Resp(status, {"message": "nope"}) for g in WEEK5})
            P.run(args(now="2026-10-11T15:52:00Z"), root=root, env=self.env, get=api, sleep=lambda s: None)
            self.assertEqual(len(api.paid), 1, status)
            self.assertFalse((root / "data" / "props.csv").exists())
            self.assertIn("the API refused the request", runs_rows(root)[-1]["detail"])
            shutil.rmtree(root, ignore_errors=True)

    def test_one_failed_game_is_skipped_and_two_in_a_row_stop_the_run(self):
        due = [g.event_id for g in P.due_games(games_of_week5(), {}, ts("2026-10-11T15:52:00Z"))]
        self.assertEqual(len(due), 13)
        down = Resp(404, {"message": "event not found"})
        api = FakeApi(week5_events(), script={due[1]: down})
        self.go(api, now="2026-10-11T15:52:00Z")
        self.assertEqual(len(api.paid), 13)                                                # one 404 among 13 due games: the rest still archived
        self.assertEqual(len(read_props(self.root)["game_id"].unique()), 12)
        root = make_root()
        api = FakeApi(week5_events(), script={due[i]: down for i in (1, 2, 3)})
        P.run(args(now="2026-10-11T15:52:00Z"), root=root, env=self.env, get=api, sleep=lambda s: None)
        self.assertEqual(len(api.paid), 3)                                                 # games 2 and 3 failed back to back: stop
        self.assertIn("two requests in a row failed", runs_rows(root)[-1]["detail"])
        shutil.rmtree(root, ignore_errors=True)

    def test_an_events_listing_failure_spends_nothing(self):
        for resp in (Resp(500, None, {}, "boom"), Resp(200, {"unexpected": "shape"})):
            root = make_root()
            api = FakeApi(week5_events(), events_resp=resp)
            P.run(args(now="2026-10-07T14:07:00Z"), root=root, env=self.env, get=api, sleep=lambda s: None)
            self.assertEqual(api.paid, [])
            self.assertFalse((root / "data" / "props.csv").exists())
            self.assertIn("events listing failed", runs_rows(root)[-1]["detail"])
            shutil.rmtree(root, ignore_errors=True)

    def test_props_nobody_has_posted_yet_leave_the_game_due_and_no_row(self):
        api = FakeApi(week5_events(), script={"evt-2026_05_TB_DAL": Resp(200, {"id": "x", "bookmakers": []}, {"x-requests-remaining": "490", "x-requests-last": "0"})})
        lines = self.go(api)
        self.assertFalse((self.root / "data" / "props.csv").exists())
        self.assertIn("no props posted yet", "\n".join(lines))
        self.assertIn("no props came back for any of the 1 due", runs_rows(self.root)[-1]["detail"])
        api2 = FakeApi(week5_events())
        self.go(api2, now="2026-10-07T15:00:00Z")                                          # the next run buys it: it never got a snapshot
        self.assertEqual(api2.paid, ["evt-2026_05_TB_DAL"])

    def test_a_low_match_rate_is_flagged_because_an_unjoinable_archive_is_worthless(self):
        (self.root / "data" / "players.csv").write_text("gsis_id,name,team\n00-1,Nobody Known,DAL\n")
        api = FakeApi(week5_events())
        self.go(api, now="2026-10-11T15:52:00Z")
        self.assertIn("matched a gsis_id", runs_rows(self.root)[-1]["detail"])
        self.assertTrue(read_props(self.root)["gsis_id"].isna().all())                      # every row still kept

    def test_a_missing_player_table_is_a_warn_not_a_crash(self):
        (self.root / "data" / "players.csv").unlink()
        self.go(FakeApi(week5_events()))
        self.assertIn("data/players.csv is missing", runs_rows(self.root)[-1]["detail"])
        self.assertTrue((self.root / "data" / "props.csv").exists())

    def test_the_book_filter_trims_rows_without_changing_what_is_requested(self):
        api = FakeApi(week5_events())
        self.go(api, "--books", "fanduel")
        self.assertEqual(set(read_props(self.root)["bookmaker"]), {"fanduel"})
        self.assertEqual(api.calls[-1][1]["regions"], "us")

    def test_a_scratch_output_leaves_no_run_log_row(self):
        scratch = self.root / "scratch.csv"
        api = FakeApi(week5_events(), remaining=14)
        self.go(api, "--out", str(scratch), now="2026-10-11T15:52:00Z")
        self.assertTrue(scratch.exists())
        self.assertFalse((self.root / "data" / "props.csv").exists())
        self.assertEqual(len(runs_rows(self.root)), 1)

    # ---------------------------------------------------------------- the key
    def test_the_key_never_reaches_stdout_stderr_props_csv_or_the_run_log_even_through_a_connection_error(self):
        def leaky(url, params=None, **k):
            raise ConnectionError(f"Max retries exceeded with url: {url}?apiKey={params['apiKey']}&regions=us")
        for api in (leaky, FakeApi(week5_events(), script={f"evt-{g}": Resp(500, None, {}, f"internal error apiKey={KEY}") for g in WEEK5}),
                    FakeApi(week5_events())):
            root = make_root()
            try:
                lines = P.run(args(now="2026-10-11T15:52:00Z"), root=root, env=self.env, get=api, sleep=lambda s: None)
                everything = "\n".join(lines) + "".join(p.read_text(errors="ignore") for p in root.rglob("*") if p.is_file())
                self.assertNotIn(KEY, everything)
                self.assertNotIn("SECRET_7f3a9c1d", everything)
            finally:
                shutil.rmtree(root, ignore_errors=True)

    def test_main_turns_any_failure_into_one_warn_row_with_the_key_scrubbed_nothing_written_and_exit_zero(self):
        boom = RuntimeError(f"could not reach https://api.the-odds-api.com/v4/x?apiKey={KEY}")
        with mock.patch.object(P, "ROOT", self.root), mock.patch("model.serve.ROOT", self.root), \
                mock.patch.dict("os.environ", {P.KEY_ENV: KEY}), mock.patch.object(P, "run", side_effect=boom):
            rc, out, err = quiet(P.main, [])
        self.assertEqual(rc, 0)
        self.assertNotIn(KEY, out + err)
        rows = runs_rows(self.root)
        self.assertEqual(len(rows), 2)
        self.assertEqual((rows[-1]["status"], rows[-1]["fails"], rows[-1]["warns"], rows[-1]["note"]), ("WARN", "0", "1", "model"))
        self.assertTrue(rows[-1]["detail"].startswith("model.props: props fetch failed, nothing written: RuntimeError"))
        self.assertNotIn(KEY, rows[-1]["detail"])
        self.assertEqual((rows[-1]["season"], rows[-1]["week"]), ("2026", "5"))             # next to the pipeline run it belongs to
        self.assertFalse((self.root / "data" / "props.csv").exists())

    def test_a_missing_requests_library_is_the_same_thing(self):
        with mock.patch.object(P, "ROOT", self.root), mock.patch("model.serve.ROOT", self.root), \
                mock.patch.dict("os.environ", {P.KEY_ENV: KEY}), mock.patch.object(P, "run", side_effect=ModuleNotFoundError("No module named 'requests'")):
            rc, out, err = quiet(P.main, [])
        self.assertEqual(rc, 0)
        self.assertIn("ModuleNotFoundError", runs_rows(self.root)[-1]["detail"])

    def test_even_if_the_run_log_cannot_be_written_main_exits_zero(self):
        with mock.patch.object(P, "ROOT", self.root), mock.patch.dict("os.environ", {P.KEY_ENV: KEY}), \
                mock.patch.object(P, "run", side_effect=RuntimeError("x")), mock.patch("ff.verify.log_run", side_effect=OSError("disk full")):
            rc, _, _ = quiet(P.main, [])
        self.assertEqual(rc, 0)


class Docs(unittest.TestCase):
    def test_every_column_is_documented_and_the_cli_prints_the_reference(self):
        self.assertEqual(set(P.COLUMN_DOCS), set(P.COLS))
        rc, out, _ = quiet(P.main, ["--columns"])
        self.assertEqual(rc, 0)
        for c in P.COLS:
            self.assertIn(c, out)

    def test_the_default_markets_are_the_six_the_work_order_names(self):
        self.assertEqual(set(P.MARKETS), {"player_receptions", "player_rush_yds", "player_reception_yds", "player_pass_yds", "player_pass_tds", "player_anytime_td"})

    def test_every_nfl_team_has_a_full_name_and_the_names_are_the_schedules_abbreviations(self):
        sched = pd.read_csv(REPO / "data" / "schedule.csv", usecols=["team"])
        self.assertEqual(set(P.TEAM_ABBR.values()), set(sched["team"]))
        self.assertEqual(len(P.TEAM_ABBR), 32)


if __name__ == "__main__":
    unittest.main()
