# Fantasy volume tracker

Tracks weekly opportunity — targets, carries, snap share, depth-chart role — for
NFL skill players, derives defense-vs-position from the same data, and writes a
weekly start/sit + dynasty watchlist report for every Sleeper league you're in.

All data is free and public. No logins, no scraping, and no API keys except the one optional
free key for the player-props archive (below).

## Where it runs

**GitHub Actions (current).** `.github/workflows/pipeline.yml` runs the pipeline
on GitHub's machines Tuesday 12:37 (with a 13:41 backstop), Wednesday 08:07,
Friday 16:11 and Sunday 09:52 Mountain,
plus a manual "Run workflow" button on the Actions tab. Every run commits
`data/`, `leagues/`, `reports/` and `logs/runs.csv` back to the repo, so the
repo *is* the database and every week is a diffable commit. The job is marked
failed — and GitHub emails you — if an integrity check fails. Your PC is not
involved. To refresh mid-week, press the button.

ESPN cookies, if you track that league, go in repo Settings → Secrets and
variables → Actions as `ESPN_S2` and `ESPN_SWID` (SWID with its braces); the
workflow writes them to `secrets/` at run time and they are never committed.
`espn_s2` expires after roughly a year — when the run starts warning that the
ESPN league could not be loaded, refresh that one secret. The league itself is
configured in `config.json` → `espn_leagues` by league id and **team id**, so
renaming the team changes nothing.

The player-props archive (`data/props.csv`) is the one optional thing that needs a key:
sign up for a free key at the-odds-api.com and add it as the Actions secret `ODDS_API_KEY`.
Until then the step prints one WARN line and skips; once the secret exists the next run
archives with no code change. The free tier is 500 credits a month and one snapshot of a
16-game week costs 96 (six markets, one region, six credits a game; the step logs the
`x-requests-remaining` header every run and warns below 100).

**On your PC (retired).** `setup_schedule.ps1` registered Windows tasks that did
the same thing locally. Once the workflow is live, remove them so there is only
one writer: `.\setup_schedule.ps1 -Remove`. The local folder can stay as a
reference copy or become a `git clone` of the repo.

## Setup (once)

PowerShell:

```
cd $env:USERPROFILE\Projects\fantasy
pip install -r requirements.txt
```

Command Prompt uses `cd %USERPROFILE%\Projects\fantasy` instead. Either way you
must be **inside the `fantasy` folder** before running anything — `python -m ff...`
only finds the `ff` package from there.

Requires Python 3.9+. Check with `python --version`.

**Python 3.12 is canonical.** That is what the GitHub Actions workflow
pins and what every committed result is produced on. A local copy may run
a different version — anything 3.9+ works for reading and for `ff.ask` —
but if a local run and a committed run ever disagree, the workflow's
result is the one that counts. `requirements.txt` also pins `pandas<3`:
pandas 3 stopped upcasting `pd.NA` into float columns and breaks
`build_players()`.

`config.json` is already filled in with your Sleeper username. The first run
discovers every league you're in for the season and writes them into
`config.json` under `leagues`. Delete any you don't want tracked and re-run.

## Weekly run

```
cd $env:USERPROFILE\Projects\fantasy
python -m ff.run_weekly
```

That picks the week automatically from Sleeper. To force one:

```
python -m ff.run_weekly --season 2026 --week 3
python -m ff.run_weekly --skip-sleeper     # nflverse only, no league sections
```

Four scheduled runs: **Tuesday noon** (main pull), **Wednesday 8am** (snap
backfill), **Friday 4pm** (status refresh — final injury designations and
the last practice report land Friday afternoon, so a Tuesday answer about
availability is always provisional) and **Sunday 9:52am** (pre-lock refresh,
an hour before the 1 pm Eastern slate locks: Sleeper statuses, projections and
the model columns for every game that has not kicked off; games already under
way keep the numbers they were frozen with).

Run it **Tuesday around noon Mountain**. Stats land Monday night; snap counts
come from Pro Football Reference and often don't post until Tuesday afternoon.
If the run warns that snaps are missing, run it again Wednesday morning — it
backfills them and rewrites the report.

Re-running is always safe. It replaces rows for the weeks it processes rather
than duplicating them, so the same command twice produces identical files.

## Weekly automation

Run once to register two Windows scheduled tasks — Tuesday noon for the main
pull, Wednesday 8am to backfill snap counts that post late:

```
cd $env:USERPROFILE\Projects\fantasy
.\setup_schedule.ps1
```

If PowerShell blocks the script, allow local scripts for your user once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Remove them with `.\setup_schedule.ps1 -Remove`. Test immediately with
`Start-ScheduledTask -TaskName 'Fantasy weekly pull (Tue)'`.

This runs on your machine and does not depend on Claude being open.

## Asking it questions

Each league has its own folder under `leagues/`, so nothing gets mixed between
teams. From this folder:

```
python -m ff.ask lineup  where-you-at
python -m ff.ask played  where-you-at 3
python -m ff.ask cut     where-you-at 3
python -m ff.ask options where-you-at "Kyle Pitts"
python -m ff.ask explain where-you-at "Kyle Pitts"
python -m ff.ask trade   where-you-at
python -m ff.ask trade   where-you-at "Alvin Kamara"
```

`played` is the one that looks backward: what you actually started in a
completed week (default: the latest), with each player's points, the team
total, the opponent and result, and your scoring rank. It reads
`data/lineups_played.csv` and `data/matchup_results.csv`, never the
pre-game roster snapshots.

Or just ask in a Cowork session with your Projects folder connected — same
answers, in plain English.

`E_pts` is expected points **in that league's scoring**: the week's projected
stat line blended with recent production (projections carry more weight early
in the season, actual production more later), then adjusted up to ±12% for the
matchup. It ranks opportunity; it is not a point projection. `explain` shows
every component that produced it.

## Roster policy

`cut` uses **keep-value**, not this week's score: value over the waiver wire,
remaining age runway from positional aging research, and usage trend — mixed
according to league type and bench depth. See `POLICY.md` for the age curves,
the weights, and the sources behind them.

## Data integrity and missed runs

**If your PC is off at a scheduled time**, Windows runs the task as soon as the
machine is available again (`-StartWhenAvailable` is set on all three tasks).
You do not need to do anything.

**If a whole week is missed**, most of it repairs itself. nflverse ships the
entire season in one file, so stats, snap counts and the schedule simply
reappear on the next run. Depth charts are snapshots, so the pipeline rebuilds
missing weeks from the snapshot that was current on that week's first game day.
Two things cannot be recovered because the source only ever reports *now*:
league rosters and injury status for that week. Those gaps are logged, not
silently skipped. What a team actually *played* is the exception: Sleeper and
ESPN both keep each completed week's lineup and points, so every run re-fetches
all completed weeks into `data/lineups_played.csv` and
`data/matchup_results.csv`, which also backfills a league the tracker has only
just started following.

**Every run is checked and recorded.** `logs/runs.csv` gets a row per run with
pass/fail counts and what went wrong. Each report carries a "Data integrity"
section. To check on demand without changing anything:

```
python -m ff.run_weekly --verify-only
```

That prints every check — team counts, duplicate rows, week continuity, ID
match rate, projection coverage, played-lineup coverage, file freshness — plus how long since the last
recorded run.

## What lands where

| Path | What it is |
|---|---|
| `data/players.csv` | One row per skill player: ids, team, position, age, experience, injury |
| `data/player_weeks.csv` | One row per player per week: targets, carries, snaps, yards, TDs, opponent |
| `data/depth_charts.csv` | Weekly snapshot of each team's QB/RB/WR/TE order, with last week's rank |
| `data/defense_vs_pos.csv` | Per defense, per position, per week: yards/TDs/targets/carries allowed |
| `data/schedule.csv` | Every team's opponent, spread and total, by week (BYE rows included) |
| `data/projections.csv` | Weekly projected stat lines per player, with derived committee shares |
| `data/trending.csv` | Most added/dropped across Sleeper in the last 24h |
| `data/status.csv` | Availability, practice reports, and the blocker chain for every player |
| `logs/runs.csv` | One row per run: when, week, pass/fail, and what failed |
| `leagues/<slug>/roster.csv` | Your team in that league, scored; plus `E_pts_model`, `p10`, `p90` from the model (blank when the model has no row) |
| `leagues/<slug>/available.csv` | That league's waiver wire, scored |
| `leagues/<slug>/all_rosters.csv` | Every owner's team, scored (trade prep) |
| `leagues/<slug>/player_points.csv` | Every player's points in that league's scoring, all weeks |
| `leagues/<slug>/transactions.csv` | Adds, drops, claims and trades by week |
| `leagues/<slug>/report.md` | That league's weekly report |
| `data/league_rosters.csv` | Every rostered player in every league you're in, with owner. Weekly snapshots are read before games: pre-game, not what played |
| `data/my_roster.csv` | Just your teams |
| `data/lineups_played.csv` | What each team actually started in each completed week: one row per player per team, with slot, started 0/1 and the platform's points. The only valid source for "who played" |
| `data/matchup_results.csv` | One row per team per completed week: opponent, points for/against, won/tie (and `median_won` in leagues that score against the league median) |
| `reports/latest.md` | The newest report; also saved as `reports/2026_wkNN.md` |
| `data/model_pts.csv` | The model's prediction for every player in the upcoming week, frozen at the last run before each game's kickoff: PPR point estimate, components, points in each league's scoring, p10/p50/p90, and Sleeper's projection and `E_pts` as they stood. Written by `python -m model.serve`; completed weeks are never rewritten |
| `data/model_eval.csv` | The live scoreboard's rows: per completed week, league, position and comparator, pick accuracy and Spearman against the platform's actual points |
| `model/reports/live_scoreboard.md` | The scoreboard summary and the adoption gate (see `POLICY.md`, "Model column") |
| `data/props.csv` | Player-props archive from The Odds API: one row per game, player, market, bookmaker and line, with Over/Under (or anytime-TD Yes) American odds, the bookmaker's own update time and `fetched_at`; `gsis_id` blank when a name did not match. One snapshot per game at the last run before its kickoff, frozen once the game has kicked off. Written by `python -m model.fetch_props` (reads the `ODDS_API_KEY` Actions secret; without it the step skips with one WARN line). **Archive only: no model or report reads it.** `python -m model.fetch_props --columns` documents every column |
| `raw/` | Cached downloads. Safe to delete; they re-download. |
| `logs/` | One log per run, including any player that failed to match an ID |

### Model columns

`E_pts_model`, `p10` and `p90` are display-only: `E_pts` stays authoritative while the model is on probation. Points are
composed from fourteen component models under each league's own scoring, **linear terms only**. Not scored, in any league:
yardage bonuses (100/200-yard rushing and receiving, 300/400-yard passing), 40+ and 50+ yard touchdown bonuses, first downs,
per-distance reception bins, and anything a kicker, team defence or IDP player earns. In this repo's leagues that is the
dynasty league's yardage and long-TD bonuses; the IDP league is scored half-PPR on skill positions only. A player who is Out,
IR, PUP, suspended or cut has no row (availability is the status layer's). `python -m model.serve --columns` documents every
column of `data/model_pts.csv`.

## Tests

```
python -m unittest discover -s tests -t .
```

They need no network and touch none of the data files.

The model has its own suite (Python 3.12, `pip install -r model/requirements.txt`, real nflverse data cached under
`model/cache/`): `python3.12 -m unittest discover -s model/tests -t .`

## Reading the report

**Opportunity score** is `0.5 x volume + 0.2 x snap share + 0.3 x matchup`,
normalized within position. It is deliberately not a point projection — it ranks
who has the best *chance* to produce, which is what volume actually tells you.

**DvP rank 1 = the softest matchup**, i.e. the defense that has allowed the most
PPR points to that position. This is the opposite of how "defensive rank" usually
reads, so every table says so.

**Baselines** blend last season with this one: 75/25 in Week 1, decaying to
current-season-only from Week 4. Each player's row shows which mix produced it.
Players with no history at all are flagged `NO BASELINE` rather than scored
against zero.

## Known limits

- nflverse lost its injury feed after 2024, so injury status comes from Sleeper.
  Running with `--skip-sleeper` leaves it blank.
- Snap counts depend on Pro Football Reference and lag the stats by a day.
- Depth charts are ESPN's, refreshed daily at 7am UTC. They are a decent proxy
  for role, not gospel — a team can list someone RB2 who plays 8 snaps.
- Long-touchdown bonuses (40+/50+ yard TDs) cannot be scored: our stat line has
  per-game totals, not the length of each score. Only "We can think of something
  funny" uses them, worth roughly 0.1-0.3 pts/game for a player who scores. Every
  run logs this as a warning so it stays visible.
- Kickers, team defenses and IDP players are never scored. In an IDP league
  they still count toward the roster limit, so `cut` accounts for them.
- ESPN is fetched only by the workflow. Claude's question sandbox cannot reach
  ESPN's API (the egress policy refuses the host), so `ff.ask live` for an ESPN
  league returns the committed lineup labelled with its time rather than a
  live read; the Sleeper leagues do get a live read. A private ESPN league
  needs the two secrets above or it is skipped with a warning.
- ESPN member ids are never written to league files: owners are keyed by team
  id and labelled by display name, because the repo is public.
- Practice-squad players sometimes lack a Sleeper or PFR id; they're logged in
  `logs/` and simply carry blank snap data.
