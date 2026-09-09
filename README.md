# Fantasy volume tracker

Tracks weekly opportunity — targets, carries, snap share, depth-chart role — for
NFL skill players, derives defense-vs-position from the same data, and writes a
weekly start/sit + dynasty watchlist report for every Sleeper league you're in.

All data is free and public. No API keys, no logins, no scraping.

## Where it runs

**GitHub Actions (current).** `.github/workflows/pipeline.yml` runs the pipeline
on GitHub's machines Tuesday 12:30, Wednesday 08:00 and Friday 16:00 Mountain,
plus a manual "Run workflow" button on the Actions tab. Every run commits
`data/`, `leagues/`, `reports/` and `logs/runs.csv` back to the repo, so the
repo *is* the database and every week is a diffable commit. The job is marked
failed — and GitHub emails you — if an integrity check fails. Your PC is not
involved. To refresh mid-week, press the button.

ESPN cookies, if you track that league, go in repo Settings → Secrets as
`ESPN_S2` and `ESPN_SWID`; the workflow writes them to `secrets/` at run time
and they are never committed.

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

Three scheduled runs: **Tuesday noon** (main pull), **Wednesday 8am** (snap
backfill), and **Friday 4pm** (status refresh — final injury designations and
the last practice report land Friday afternoon, so a Tuesday answer about
availability is always provisional).

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
python -m ff.ask cut     where-you-at 3
python -m ff.ask options where-you-at "Kyle Pitts"
python -m ff.ask explain where-you-at "Kyle Pitts"
python -m ff.ask trade   where-you-at
python -m ff.ask trade   where-you-at "Alvin Kamara"
```

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
silently skipped.

**Every run is checked and recorded.** `logs/runs.csv` gets a row per run with
pass/fail counts and what went wrong. Each report carries a "Data integrity"
section. To check on demand without changing anything:

```
python -m ff.run_weekly --verify-only
```

That prints every check — team counts, duplicate rows, week continuity, ID
match rate, projection coverage, file freshness — plus how long since the last
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
| `leagues/<slug>/roster.csv` | Your team in that league, scored |
| `leagues/<slug>/available.csv` | That league's waiver wire, scored |
| `leagues/<slug>/all_rosters.csv` | Every owner's team, scored (trade prep) |
| `leagues/<slug>/player_points.csv` | Every player's points in that league's scoring, all weeks |
| `leagues/<slug>/transactions.csv` | Adds, drops, claims and trades by week |
| `leagues/<slug>/report.md` | That league's weekly report |
| `data/league_rosters.csv` | Every rostered player in every league you're in, with owner |
| `data/my_roster.csv` | Just your teams |
| `reports/latest.md` | The newest report; also saved as `reports/2026_wkNN.md` |
| `raw/` | Cached downloads. Safe to delete; they re-download. |
| `logs/` | One log per run, including any player that failed to match an ID |

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
- ESPN leagues need `secrets/espn_cookies.json` if the league is private. See
  `config.example.json` for the shape; those values never leave your machine.
- Practice-squad players sometimes lack a Sleeper or PFR id; they're logged in
  `logs/` and simply carry blank snap data.
