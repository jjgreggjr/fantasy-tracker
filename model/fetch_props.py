"""Archive NFL player props from The Odds API into data/props.csv. ARCHIVE ONLY: nothing reads it yet, no model input.

    python -m model.fetch_props                  # what the weekly workflow runs
    python -m model.fetch_props --dry-run        # list the events and the plan (the free call only), write nothing
    python -m model.fetch_props --columns        # the column reference for data/props.csv

Why it lives in model/ and not ff/: ff/ is the strict pipeline (its failure turns the run red), and this is a new network source
that must never be able to do that; model/ already holds the failure-isolated fetchers (fetch_adp, fetch_cfbd) and the run-log
helper, and the archive exists to answer a model question later. It needs only pandas and requests, so it runs on the pipeline's
own install even when the model's dependencies failed to install.

What a run does:

  1. No ODDS_API_KEY in the environment (the Actions secret James has not created yet): print one WARN line, exit 0, touch nothing.
  2. List the events (`/v4/sports/americanfootball_nfl/events`, free) and match each to a game in data/schedule.csv (which gives the
     week and the nflverse game id).
  3. Pick the games that are DUE (see `due_games`): not kicked off, kicking off within HORIZON_HOURS, and with no snapshot younger
     than REFRESH_HOURS. On the weekly schedule that is exactly one snapshot per game, taken at the last run before its kickoff
     (Wednesday for Thursday night, Friday for an early international game, the Sunday pre-lock run for everything from the 1 pm
     slate to Monday night). One snapshot per game is the credit budget (below).
  4. One request per due game to the event-level odds endpoint (`/events/{id}/odds`), one region, the six markets in MARKETS.
     Player props are only served by that endpoint, and it bills markets x regions per request: 6 credits a game.
  5. Parse, match player names to gsis ids through data/players.csv (ff.build.name_key, the repo's matcher; a name that does not match
     keeps its raw name with a blank id, it is never dropped), and write data/props.csv through ff.build.replace_partition, FROZEN per
     game: the rows of a game that has kicked off are never rewritten.

Credit budget (free tier: 500 credits a month). Every paid call logs the `x-requests-last/used/remaining` headers; the free events
call is read for them too, so the budget is known before anything is spent. Games are fetched soonest-kickoff first and the run stops
funding when the remaining credits cannot pay for another game. WARNs (logs/runs.csv, check `model.props`): fewer than WARN_BELOW
credits left, a due game left unfunded, a key the API rejects, an exhausted quota, a billed cost above the plan, repeated failures, a
low name-match rate, a large file. An unset key is NOT a run-log row (it is not a failure, and a row per run forever is noise).

It can never break the pipeline: every failure is a WARN row, nothing is written, exit 0. The key is read from the environment only,
sent only as the API's `apiKey` query parameter, and scrubbed from every message this module prints, logs or raises (the key sits in
the URL, so a connection error would otherwise carry it into a committed log row).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROPS_FILE = ROOT / "data" / "props.csv"
CHECK = "model.props"
KEY_ENV = "ODDS_API_KEY"
HOST = "https://api.the-odds-api.com"
SPORT = "americanfootball_nfl"
REGION = "us"
UA = "fantasy-tracker-model/1.0 (+https://github.com/jjgreggjr/fantasy-tracker)"

MARKETS = ("player_receptions", "player_rush_yds", "player_reception_yds", "player_pass_yds", "player_pass_tds", "player_anytime_td")
HORIZON_HOURS = 40          # fetch a game only when it kicks off within this many hours
REFRESH_HOURS = 30          # ... and its newest snapshot is at least this old (a manual dispatch cannot double-spend)
WARN_BELOW = 100            # credits left that count as "near the month's budget": about one slate (15 games x 6 = 90)
MONTHLY_CREDITS = 500       # the free tier; only used for the math printed with the plan
WEEKS_PER_MONTH = 52 / 12
FILE_WARN_BYTES = 40 * 1024 * 1024    # GitHub warns on a 50 MB file
MATCH_WARN = 0.70           # gsis-id match rate below which the archive is flagged as not joinable
RETRY_STATUS = {429, 500, 502, 503, 504}

COLS = ["season", "week", "game_id", "kickoff_utc", "gsis_id", "player", "market", "bookmaker", "line", "over_price", "under_price",
        "book_updated", "fetched_at"]
KEYS = ["season", "week", "game_id", "market", "bookmaker", "player", "line"]      # sort order; (a book lists one line per player-market)
PARTITION = ["season", "week"]

COLUMN_DOCS = {
    "season": "NFL season",
    "week": "NFL week of the game (data/schedule.csv)",
    "game_id": "nflverse game id (season_week_away_home), built from data/schedule.csv: the join key to data/model_pts.csv",
    "kickoff_utc": "the game's kickoff as The Odds API reports it (commence_time, UTC). Rows of a game are frozen once this has passed",
    "gsis_id": "nflverse player id, through data/players.csv (ff.build.name_key within the game's two teams, else a name unique in "
               "the league). BLANK when the name did not match: the raw name is kept in `player`, the row is never dropped",
    "player": "the player's name exactly as the bookmaker spelled it (the API's outcome `description`)",
    "market": "The Odds API market key: " + ", ".join(MARKETS),
    "bookmaker": "bookmaker key (draftkings, fanduel, ...). Every book the one region returns is archived",
    "line": "the prop line (yards, receptions, TDs); blank for player_anytime_td, which has no line",
    "over_price": "American odds of the Over (of Yes for player_anytime_td). Stored as a float: replace_partition re-reads the file "
                  "with default dtypes, which cannot keep a gappy integer column",
    "under_price": "American odds of the Under; blank when the book posts only one side (player_anytime_td always)",
    "book_updated": "when the bookmaker last updated this market (the API's `last_update`), UTC: how stale the quote already was",
    "fetched_at": "UTC time of the request that returned this row",
}

TEAM_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF", "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
    "Detroit Lions": "DET", "Green Bay Packers": "GB", "Houston Texans": "HOU", "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN", "New England Patriots": "NE",
    "New Orleans Saints": "NO", "New York Giants": "NYG", "New York Jets": "NYJ", "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def column_docs() -> str:
    return "\n".join(f"{k:14s} {v}" for k, v in COLUMN_DOCS.items())


# --------------------------------------------------------------------------- the key never leaves the process
def scrub(text, key: str = "") -> str:
    """`text` with the key, and anything shaped like `apiKey=...`, replaced by ***. The key is a URL parameter, so the text of a
    connection error (which quotes the URL) would carry it into the Actions log and, through the run-log WARN, into a committed file."""
    s = str(text)
    if key:
        s = s.replace(key, "***")
    return re.sub(r"(?i)(api_?key=)[^&\s'\"<>)]+", r"\1***", s)


# --------------------------------------------------------------------------- the HTTP layer
@dataclass
class Reply:
    status: int | None                       # None: no HTTP answer (connection failure)
    data: object = None                      # the parsed JSON body, or None
    credits: dict = field(default_factory=dict)       # remaining / used / last, ints, for the headers the response carried
    error: str = ""                          # scrubbed; "" on a 200
    quota: bool = False                      # the account is out of credits


def credits_from(headers) -> dict:
    """{remaining, used, last}: the `x-requests-*` headers (the credits left in the month, used, and billed for THIS call)."""
    low = {str(k).lower(): v for k, v in dict(headers or {}).items()}
    out = {}
    for name in ("remaining", "used", "last"):
        try:
            out[name] = int(float(low[f"x-requests-{name}"]))
        except (KeyError, TypeError, ValueError):
            pass
    return out


def api_get(path: str, params: dict, key: str, *, get=None, retries: int = 3, sleep=time.sleep, timeout: int = 30) -> Reply:
    """GET one endpoint. Never raises for a network or HTTP problem: the Reply says what happened, with the key scrubbed out. Retries
    only what can pass by itself (connection errors, 429, 5xx) and never a refusal, so a bad key or market costs one request."""
    import requests
    get = get or requests.get
    last = Reply(None, error="no attempt made")
    for attempt in range(retries):
        try:
            r = get(HOST + path, params={**params, "apiKey": key}, timeout=timeout, headers={"User-Agent": UA, "Accept": "application/json"})
        except Exception as e:                                           # requests.RequestException, and anything a stand-in raises
            last = Reply(None, error=scrub(f"{type(e).__name__}: {e}", key)[:300])
            if attempt < retries - 1:
                sleep(2 ** attempt)
            continue
        credits = credits_from(r.headers)
        try:
            body = r.json()
        except Exception:
            body = None
        text = scrub(getattr(r, "text", "") or "", key)[:200]
        if r.status_code == 200:
            if body is None:
                return Reply(200, None, credits, f"HTTP 200 but not JSON: {text!r}")
            return Reply(200, body, credits)
        code = str(body.get("error_code", "")) if isinstance(body, dict) else ""
        quota = code == "OUT_OF_USAGE_CREDITS" or "usage quota" in text.lower()
        last = Reply(r.status_code, body, credits, f"HTTP {r.status_code}" + (f" {code}" if code else "") + f": {text!r}", quota)
        if quota or r.status_code not in RETRY_STATUS:
            return last
        if attempt < retries - 1:
            sleep(2 ** attempt)
    return last


def list_events(key: str, **kw) -> Reply:
    """Every upcoming NFL event: id, commence_time, home_team, away_team. The endpoint is free."""
    return api_get(f"/v4/sports/{SPORT}/events", {"dateFormat": "iso"}, key, **kw)


def event_odds(key: str, event_id: str, markets, region: str = REGION, **kw) -> Reply:
    """One event's odds for `markets` in `region`. Billed markets x regions per request (the credit headers say what it really cost)."""
    return api_get(f"/v4/sports/{SPORT}/events/{event_id}/odds",
                   {"regions": region, "markets": ",".join(markets), "oddsFormat": "american", "dateFormat": "iso"}, key, **kw)


# --------------------------------------------------------------------------- events -> games
@dataclass
class Game:
    event_id: str
    game_id: str
    season: int
    week: int
    kickoff: object          # tz-aware pd.Timestamp (UTC)
    home: str
    away: str


def schedule_games(root: Path, season: int):
    """One row per regular-season game of `season` from data/schedule.csv: game_id (nflverse: season_week_away_home), week, home, away,
    gameday. Byes and the away half of each game are dropped."""
    import pandas as pd
    p = root / "data" / "schedule.csv"
    cols = ["game_id", "week", "home", "away", "gameday"]
    if not p.exists():
        return pd.DataFrame(columns=cols)
    s = pd.read_csv(p, usecols=["season", "week", "team", "opponent", "home", "gameday"])
    s = s[(s["season"] == season) & (s["home"] == 1) & (s["opponent"] != "BYE")]
    s = s.assign(week=s["week"].astype(int))
    return pd.DataFrame({"game_id": [f"{season}_{w:02d}_{o}_{t}" for w, o, t in zip(s["week"], s["opponent"], s["team"])],
                         "week": s["week"].to_numpy(), "home": s["team"].to_numpy(), "away": s["opponent"].to_numpy(),
                         "gameday": s["gameday"].to_numpy()}, columns=cols).reset_index(drop=True)


def match_events(events, sched, season: int):
    """(games, notes): each listed event matched to its schedule game by the unordered team pair (a neutral-site game may list the
    teams the other way round) and, when the pair meets twice, by the closest gameday. An event that matches nothing (a preseason or
    playoff game, an unknown team name) is skipped and named in `notes`."""
    import pandas as pd
    games, notes = [], []
    for ev in events or []:
        try:
            h, a = TEAM_ABBR.get(ev["home_team"]), TEAM_ABBR.get(ev["away_team"])
            kick = pd.Timestamp(ev["commence_time"])
            kick = kick.tz_localize("UTC") if kick.tzinfo is None else kick.tz_convert("UTC")
            eid = str(ev["id"])
        except Exception as e:
            notes.append(f"unreadable event ({type(e).__name__}: {e})")
            continue
        if not h or not a:
            notes.append(f"unknown team name in {ev.get('away_team')!r} at {ev.get('home_team')!r}")
            continue
        cand = sched[((sched["home"] == h) & (sched["away"] == a)) | ((sched["home"] == a) & (sched["away"] == h))]
        if cand.empty:
            notes.append(f"{a} at {h} {kick:%Y-%m-%d} is not a {season} regular-season game")
            continue
        day = kick.tz_convert("America/New_York").normalize().tz_localize(None)
        gap = (pd.to_datetime(cand["gameday"]) - day).abs()
        row = cand.loc[gap.idxmin()]
        if gap.min() > pd.Timedelta(days=1):
            notes.append(f"{a} at {h} {kick:%Y-%m-%d} is more than a day from its scheduled {row['gameday']}")
            continue
        games.append(Game(eid, str(row["game_id"]), season, int(row["week"]), kick, str(row["home"]), str(row["away"])))
    return games, notes


# --------------------------------------------------------------------------- which games to spend credits on
def due_games(games, snapshots: dict, now, horizon_hours: float = HORIZON_HOURS, refresh_hours: float = REFRESH_HOURS) -> list:
    """The games worth a paid request now, soonest kickoff first: not kicked off, kicking off within `horizon_hours`, and with no snapshot
    younger than `refresh_hours` (`snapshots`: game_id -> the fetched_at of its newest archived row).

    The horizon is what makes it one snapshot per game, as late as the weekly runs allow (Tue 18:37, Tue 19:41, Wed 14:07, Fri 22:11 and
    Sun 15:52 UTC): Thursday night is 34 h away on Wednesday and 54 h away on Tuesday; an international 13:30 UTC Sunday game is 39 h away
    on Friday; the 17:00 UTC slate is 43 h away on Friday (not due) and 1 h away on Sunday (due); Sunday night and Monday night are 32-34 h
    away on Sunday. The refresh age stops a manual re-run from paying twice for a snapshot taken hours ago."""
    import pandas as pd
    out = []
    for g in games:
        ahead = (g.kickoff - now) / pd.Timedelta(hours=1)
        if not 0 < ahead <= horizon_hours:
            continue
        seen = snapshots.get(g.game_id)
        if seen is not None and (now - seen) < pd.Timedelta(hours=refresh_hours):
            continue
        out.append(g)
    return sorted(out, key=lambda g: (g.kickoff, g.game_id))


def snapshot_times(path: Path) -> dict:
    """game_id -> newest fetched_at in the archive. A game whose request returned no props has no row, so it counts as unfetched."""
    import pandas as pd
    if not Path(path).exists():
        return {}
    d = pd.read_csv(path, usecols=["game_id", "fetched_at"], low_memory=False)
    d["fetched_at"] = pd.to_datetime(d["fetched_at"], utc=True, errors="coerce")
    return d.dropna().groupby("game_id")["fetched_at"].max().to_dict()


def credit_math(games_per_week: float, n_markets: int = len(MARKETS), regions: int = 1, monthly: int = MONTHLY_CREDITS) -> dict:
    """The weekly and monthly cost of one snapshot per game, and how much of the monthly budget it takes."""
    week = games_per_week * n_markets * regions
    month = week * WEEKS_PER_MONTH
    return {"per_game": n_markets * regions, "per_week": week, "per_month": round(month, 1), "share": round(month / monthly, 3)}


# --------------------------------------------------------------------------- parsing
def parse_event_odds(payload, game: Game, fetched_at: str, books=None):
    """One event-odds response as a frame of COLS (gsis_id still blank): one row per (bookmaker, market, player, line) with the Over and
    Under (or the anytime-TD Yes) prices side by side. Tolerant of what it does not know: a market or outcome it cannot read is
    skipped, an unknown key is ignored. Returns (frame, n_unreadable_outcomes)."""
    import pandas as pd
    rows: dict = {}
    bad = 0
    for bk in (payload or {}).get("bookmakers") or []:
        bkey = bk.get("key")
        if not bkey or (books and bkey not in books):
            continue
        for mk in bk.get("markets") or []:
            market = mk.get("key")
            for o in mk.get("outcomes") or []:
                player, side = o.get("description"), str(o.get("name") or "").lower()
                try:
                    price = float(o["price"])
                    point = float(o["point"]) if o.get("point") is not None else float("nan")
                except (KeyError, TypeError, ValueError):
                    bad += 1
                    continue
                if not market or not player or side not in ("over", "yes", "under", "no"):
                    bad += 1
                    continue
                r = rows.setdefault((bkey, market, str(player), point if point == point else None),
                                    {"over": float("nan"), "under": float("nan"), "updated": mk.get("last_update") or bk.get("last_update")})
                r["over" if side in ("over", "yes") else "under"] = price
    df = pd.DataFrame([{"season": game.season, "week": game.week, "game_id": game.game_id, "kickoff_utc": game.kickoff.isoformat(),
                        "gsis_id": None, "player": p, "market": m, "bookmaker": b, "line": float("nan") if pt is None else pt,
                        "over_price": r["over"], "under_price": r["under"], "book_updated": r["updated"], "fetched_at": fetched_at}
                       for (b, m, p, pt), r in rows.items()], columns=COLS)
    return df, bad


def build_crosswalk(players):
    """({(team, name_key): [gsis_id]}, {name_key: [gsis_id]}) from data/players.csv, the pipeline's own id table."""
    from ff.build import name_key
    by_team: dict = {}
    by_name: dict = {}
    if players is None or not len(players):
        return by_team, by_name
    for gsis, name, team in zip(players["gsis_id"], players["name"], players["team"]):
        k = name_key(name)
        if not k or gsis != gsis or gsis is None:
            continue
        by_team.setdefault((team, k), []).append(gsis)
        by_name.setdefault(k, []).append(gsis)
    return by_team, by_name


def resolve_ids(frame, game: Game, crosswalk):
    """`frame` with gsis_id filled where the name resolves to exactly one player: within the game's two teams first, then (a traded player
    can sit on his old team in data/players.csv) a name that is unique in the whole table. An ambiguous or unknown name stays blank.
    Returns (frame, n_matched_in_game, n_matched_by_name_only)."""
    from ff.build import name_key
    by_team, by_name = crosswalk
    cache: dict = {}

    def one(player):
        if player in cache:
            return cache[player]
        k = name_key(player)
        hit = {g for t in (game.home, game.away) for g in by_team.get((t, k), [])}
        how = "team"
        if not hit:
            hit, how = set(by_name.get(k, [])), "name"
        cache[player] = (next(iter(hit)), how) if len(hit) == 1 and k else (None, "")
        return cache[player]
    ids = [one(p) for p in frame["player"]]
    out = frame.assign(gsis_id=[i for i, _ in ids])
    n_name = sum(1 for i, h in ids if i and h == "name")
    return out, int(out["gsis_id"].notna().sum()), n_name


# --------------------------------------------------------------------------- writing, frozen per game
def merge_frozen(old, new, now):
    """Everything the archive holds for the (season, week) partitions `new` touches, after this run: the old rows of every game NOT in
    `new` (a game under way or finished is the frozen record; a game not fetched this run keeps its last snapshot) plus `new`'s rows,
    minus any new row of a game that has already kicked off (it is never rewritten)."""
    import pandas as pd
    if new is None or new.empty:
        return new
    new = new[pd.to_datetime(new["kickoff_utc"], utc=True) > now]
    if new.empty:
        return new
    if old is None or old.empty:
        return new
    touched = set(zip(new["season"], new["week"]))
    in_week = pd.Series([(s, w) in touched for s, w in zip(old["season"], old["week"])], index=old.index)
    keep = old[in_week & ~old["game_id"].isin(set(new["game_id"]))]
    return pd.concat([keep, new], ignore_index=True)


def write_props(path: Path, new, now) -> int:
    """Replace the touched (season, week) partitions of the archive through ff.build.replace_partition, keeping every frozen game. Works on
    a copy and renames it over the file, so a crash cannot leave a half-written archive. Returns the number of new rows written."""
    import pandas as pd
    from ff import build
    path = Path(path)
    old = pd.read_csv(path, low_memory=False, dtype={"gsis_id": str}) if path.exists() else None
    merged = merge_frozen(old, new, now)
    if merged is None or merged.empty:
        return 0
    written = int((pd.to_datetime(new["kickoff_utc"], utc=True) > now).sum())
    tmp = Path(str(path) + ".tmp")
    if path.exists():
        tmp.write_bytes(path.read_bytes())
    elif tmp.exists():
        tmp.unlink()
    build.replace_partition(tmp, merged, PARTITION, KEYS)
    os.replace(tmp, path)
    return written


# --------------------------------------------------------------------------- the run
def skip_line() -> str:
    line = (f"model.props: WARN {KEY_ENV} is not set: player-props archive skipped, nothing fetched or written "
            "(sign up at the-odds-api.com and add the Actions secret ODDS_API_KEY; the next run archives with no code change)")
    return f"::warning title=model.props::{line}" if os.environ.get("GITHUB_ACTIONS") == "true" else line


def read_players(root: Path):
    import pandas as pd
    p = root / "data" / "players.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, usecols=["gsis_id", "name", "team"], dtype={"gsis_id": str}, low_memory=False)


def run(args, *, root: Path = ROOT, env=None, get=None, sleep=time.sleep) -> list[str]:
    """Plan, fetch, write. Returns the printable summary; every WARN is also a row of logs/runs.csv (check `model.props`) when this is the
    live archive. Raises only for a programming or data error (main turns that into a WARN row too)."""
    import pandas as pd
    env = os.environ if env is None else env
    key = (env.get(KEY_ENV) or "").strip()
    if not key:
        return [skip_line()]
    now = pd.Timestamp(args.now, tz="UTC") if args.now else pd.Timestamp.now(tz="UTC")
    out_path = Path(args.out) if args.out else root / "data" / "props.csv"
    live = out_path == root / "data" / "props.csv"
    season = args.season or int(json.loads((root / "config.json").read_text(encoding="utf-8"))["season"])
    markets = tuple(m.strip() for m in args.markets.split(",") if m.strip())
    books = {b.strip() for b in (args.books or "").split(",") if b.strip()}
    cost_each = len(markets)                                           # one region
    lines: list[str] = []
    warns: list[str] = []
    week_seen: set[int] = set()

    def finish():
        for w in warns:
            lines.append(f"  WARN {w}")
        if warns and live:
            from model.serve import log_warns
            log_warns(root, warns, season, max(week_seen) if week_seen else None, check=CHECK)
        return lines

    sched = schedule_games(root, season)
    if sched.empty:
        raise RuntimeError(f"data/schedule.csv has no {season} games to match the events to")
    ev = list_events(key, get=get, sleep=sleep)
    credits = dict(ev.credits)
    if ev.error or not isinstance(ev.data, list):
        warns.append(f"events listing failed, nothing fetched: {scrub(ev.error or 'unexpected response shape', key)}")
        return finish()
    games, notes = match_events(ev.data, sched, season)
    remaining = credits.get("remaining")
    snaps = snapshot_times(out_path)
    due = due_games(games, snaps, now, args.horizon_hours, args.refresh_hours)
    lines.append(f"model.props: {season}: {len(ev.data)} events listed, {len(games)} matched to the schedule, {len(due)} due "
                 f"(kickoff within {args.horizon_hours:g} h, no snapshot under {args.refresh_hours:g} h old); {len(markets)} markets x 1 region = "
                 f"{cost_each} credits a game, {cost_each * len(due)} planned"
                 + (f"; credits remaining {remaining}" if remaining is not None else "; credits remaining not reported by the events call"))
    for n in notes[:5]:
        lines.append(f"  skipped event: {n}")
    if args.dry_run:
        for g in due:
            lines.append(f"  would fetch {g.game_id} (kickoff {g.kickoff.isoformat()})")
        return lines + ["  (dry run: nothing fetched, nothing written)"]
    players = read_players(root)
    if players is None:
        warns.append("data/players.csv is missing: every gsis_id is blank in this run's rows")
    crosswalk = build_crosswalk(players)

    frames, skipped, failures, consecutive, over_plan, written, paid = [], [], 0, 0, False, 0, 0
    stop = ""
    for g in due:
        if remaining is not None and remaining < cost_each:
            skipped.append(g.game_id)
            continue
        rep = event_odds(key, g.event_id, markets, REGION, get=get, sleep=sleep)
        paid += 1
        credits.update(rep.credits)
        remaining = credits.get("remaining", remaining)
        tag = f"  {g.game_id} (kickoff {g.kickoff:%a %Y-%m-%d %H:%MZ})"
        used = (f"credits last={credits['last']} used={credits.get('used', '?')} remaining={remaining if remaining is not None else '?'}"
                if "last" in credits else "credits not reported")
        if rep.quota:
            warns.append(f"monthly credits exhausted ({rep.error}): archive stopped at {g.game_id}")
            stop = "quota"
            break
        if rep.status in (401, 403, 422):
            warns.append(f"the API refused the request ({rep.error}): archive stopped at {g.game_id}; check the key, the plan and the markets")
            stop = "refused"
            break
        if rep.error:
            failures += 1
            consecutive += 1
            lines.append(f"{tag}: request failed, game skipped ({rep.error})")
            if consecutive >= 2:
                warns.append(f"two requests in a row failed ({rep.error}): archive stopped at {g.game_id}")
                stop = "failures"
                break
            continue
        consecutive = 0
        if "last" in credits and credits["last"] > cost_each and not over_plan:
            over_plan = True
            warns.append(f"a request was billed {credits['last']} credits against the planned {cost_each}: the budget math is off")
        fetched_at = (pd.Timestamp.now(tz="UTC") if not args.now else now).strftime("%Y-%m-%dT%H:%M:%SZ")
        df, bad = parse_event_odds(rep.data, g, fetched_at, books or None)
        week_seen.add(g.week)
        if df.empty:
            lines.append(f"{tag}: no props posted yet (the game stays due); {used}")
            continue
        df, n_id, n_name = resolve_ids(df, g, crosswalk)
        # Written game by game: the credits for this game are already spent, so a run cut off by the step's timeout (or a crash in a
        # later game) must not lose the snapshots it paid for.
        written += write_props(out_path, df, now if args.now else pd.Timestamp.now(tz="UTC"))
        frames.append((g, df, n_id, n_name))
        lines.append(f"{tag}: {len(df)} rows from {df['bookmaker'].nunique()} books, {df['player'].nunique()} players "
                     f"({n_id} rows with a gsis_id, {n_name} by name only){f', {bad} unreadable outcomes' if bad else ''}; {used}")
    if skipped:
        warns.append(f"credits ran out before {len(skipped)} due game(s) could be funded ({', '.join(skipped[:4])}"
                     f"{'...' if len(skipped) > 4 else ''}); remaining {remaining}, {cost_each} needed per game")
    if due and not frames and not stop and not skipped and failures == 0:
        warns.append(f"no props came back for any of the {len(due)} due game(s): not posted yet, or this plan does not include these markets")
    if frames:
        new = pd.concat([f for _, f, _, _ in frames], ignore_index=True)
        total = len(pd.read_csv(out_path, usecols=["game_id"])) if out_path.exists() else 0
        match = float(new["gsis_id"].notna().mean())
        lines.append(f"  wrote {out_path.name}: {written} rows for {len(frames)} game(s), {total} rows in the file; "
                     f"gsis_id on {match:.1%} of the new rows")
        if len(new) >= 50 and match < MATCH_WARN:
            names = sorted(set(new.loc[new["gsis_id"].isna(), "player"]))[:6]
            warns.append(f"only {match:.0%} of the new rows matched a gsis_id (unmatched e.g. {', '.join(names)}): the archive is not "
                         "joinable to the repo's player ids until the crosswalk is looked at")
        if out_path.exists() and out_path.stat().st_size > FILE_WARN_BYTES:
            warns.append(f"{out_path.name} is {out_path.stat().st_size / 2 ** 20:.0f} MB (GitHub warns at 50 MB): restrict --books in the "
                         "workflow step, which trims rows without changing what the credits buy")
    if remaining is not None:
        lines.append(f"  credits: used {credits.get('used', '?')}, remaining {remaining} (x-requests-remaining; the free tier is "
                     f"{MONTHLY_CREDITS} a month)")
        if remaining < args.warn_below and paid:          # only a run that spent credits nags: a quiet Tuesday does not repeat it
            warns.append(f"credits are low: {remaining} left (warning below {args.warn_below}; one 15-game slate is {15 * cost_each}); "
                         "the free tier resets monthly")
    else:
        lines.append("  credits: the API reported no x-requests-remaining header")
    return finish()


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--season", type=int)
    ap.add_argument("--markets", default=",".join(MARKETS), help="comma-separated Odds API market keys (each costs 1 credit per game)")
    ap.add_argument("--books", default="", help="keep only these bookmaker keys (default: every book the region returns). Trims rows, not credits")
    ap.add_argument("--horizon-hours", type=float, default=HORIZON_HOURS)
    ap.add_argument("--refresh-hours", type=float, default=REFRESH_HOURS)
    ap.add_argument("--warn-below", type=int, default=WARN_BELOW, help="WARN when fewer credits than this remain")
    ap.add_argument("--now", help="treat this UTC instant as now (tests, replays)")
    ap.add_argument("--out", help="write here instead of data/props.csv (no run-log row)")
    ap.add_argument("--dry-run", action="store_true", help="list the events and the plan (free call), fetch and write nothing")
    ap.add_argument("--columns", action="store_true", help="print the column reference and exit")
    return ap


def main(argv=None) -> int:
    ap = make_parser()
    args = ap.parse_args(argv)
    if args.columns:
        print(column_docs())
        return 0
    try:
        for line in run(args, root=ROOT):
            print(line)
    except Exception as e:                      # ANY failure: a WARN row, no writes, exit 0. The pipeline never sees it.
        key = (os.environ.get(KEY_ENV) or "").strip()
        tb = traceback.extract_tb(e.__traceback__)
        where = f" at {Path(tb[-1].filename).name}:{tb[-1].lineno}" if tb else ""
        msg = scrub(f"{type(e).__name__}: {e}{where}", key)
        print(f"model.props FAILED, outputs untouched: {msg}", file=sys.stderr)
        print(scrub(traceback.format_exc(), key), file=sys.stderr)
        from model.serve import log_warns
        log_warns(ROOT, [f"props fetch failed, nothing written: {msg}"[:400]], check=CHECK)
    return 0


if __name__ == "__main__":
    sys.exit(main())
