# Plan: a boosted-tree projection model, tested honestly on 2025

Goal: predict weekly fantasy points per player with gradient-boosted trees
(LightGBM / XGBoost / CatBoost), report which inputs actually predict (SHAP),
and beat our current opportunity-based `E_pts` and the platforms on
rank-ordering — measured, not assumed. This is a work order in the style of
PLAN_ESPN_AND_PLAYER_PROCESS.md: phases execute in separate sessions, each
phase appends its findings here, and the durable pieces land in POLICY.md,
README.md and the skill when the model ships. Delete this file at the end.

## Ground rules

- **The leakage rule is the whole project.** Every feature for a
  (player, game) row must be knowable strictly before that game's kickoff.
  This is enforced structurally, not by care:
  1. Every raw table carries an explicit availability timestamp
     (`known_at`): game rows use the game's kickoff, injury/practice
     reports their report date, depth charts their snapshot date, Vegas
     lines their close, season-static data (ADP, draft capital, college
     stats, combine) the season start.
  2. One join gate — a single `as_of_join(features_for, kickoff_ts)`
     helper in `model/point_in_time.py` — is the only way feature code may
     touch raw tables. It filters `known_at < kickoff_ts`. No ad-hoc merges.
  3. An audit test proves it: rebuild a sample of feature rows from a copy
     of the raw data truncated strictly before each row's kickoff and
     assert byte-identical features; plus a canary test that injects a
     future-dated row and asserts the gate excludes it.
  4. Walk-forward evaluation only: to score week N, train on strictly
     earlier data. Never shuffle weeks into folds.
- Known leakage traps, named so nobody steps in them:
  - ffopportunity xFP for week N is computed **from** week N — only lagged
    xFP (weeks < N) is a feature; same-week xFP is a *baseline to beat*,
    never an input.
  - nflverse schedules' temp/wind are observed (post-game) values. Using
    them approximates a perfect forecast. Acceptable for v1 as a proxy,
    documented; v2 can switch to Open-Meteo's historical *forecast* archive.
  - Platform projections: historical ESPN/Sleeper projections are not
    publicly archived. Our own repo commits point-in-time Sleeper
    projections each run (since 2026 week 1), so that feature accrues
    going forward but does not exist for the 2025 backtest.
  - DvP windows for week N end at week N-1.
- Model code lives in `model/`, separate from `ff/` (the pipeline). The
  pipeline is not touched until Phase 3. Question sessions read; nothing in
  `model/` runs in the weekly workflow until Phase 3 lands it there.
- Python 3.12. Model deps (lightgbm, xgboost, shap, etc.) go in
  `model/requirements.txt`, not the pipeline's `requirements.txt`.
- Seeds fixed everywhere; a rerun reproduces every number.
- Free, API-accessible data only. No credentials in the repo; any API key
  (CollegeFootballData) enters as a GitHub Actions secret only, added by
  James.

## Data sources (Phase 0 validates each from the sandbox)

| Source | What | known_at |
|---|---|---|
| nflverse via `nfl_data_py` | weekly stats, PBP (air yards, red zone, shares), snap counts, injuries + practice reports, depth charts, rosters, combine, draft picks | game kickoff / report date / snapshot date |
| nflverse schedules | kickoff datetime, roof/surface, temp, wind, spread_line, total_line (implied team totals) | lines close pre-kickoff; weather = observed (v1 proxy) |
| ffopportunity releases | expected fantasy points (xFP) | game kickoff (lag-only feature; same-week = baseline) |
| FantasyFootballCalculator API | historical ADP by season/format | season start |
| CollegeFootballData API | college production for rookie priors (market share, dominator, breakout age) | season start (career-to-date) |
| Open-Meteo archive | v2 weather-forecast replacement | forecast issue time |

## Feature set (Phase 1 builds; SHAP in Phase 2 ranks them)

Per (player, week): lags of points/volume over prior 1, 2, 3 weeks and
season-to-date (E_opps, carries, targets, target share, carry share, snap%,
route data where available); role (depth rank, teammate injuries — who above
or beside him is Out, and the redistributed volume); opponent DvP vs his
position over last 2 and last 4 regular-season weeks; Vegas spread, total
and implied team total; home/away, rest days, divisional flag; weather +
roof; ADP as season prior; TD-luck regression (actual minus expected TDs to
date); age, experience, career games. Rookies (no separate model — trees
handle missing history): draft capital, combine, college market share,
breakout age, with NFL usage taking over as games accumulate.

## Targets, metrics, baselines

- Target: PPR points (and league-scoring variants later). Quantile models
  (p10/p50/p90) alongside the point estimate — floor/ceiling is the start/sit
  product.
- Metrics per position: MAE/RMSE; Spearman rank correlation within
  position-week (the start/sit metric); head-to-head "who scored more" pick
  accuracy on same-position pairs; quantile calibration.
- Baselines to beat, same walk-forward weeks: trailing-3-game average;
  ffopportunity same-week xFP; our pipeline's `E_pts` where reconstructable.
- Train 2021–2024, walk-forward every week of 2025. 2025 is never trained on
  out of order.

## Phases

- **Phase 0 — source validation + leakage harness (this dispatch).**
  Probe every source above from the sandbox with small real pulls; document
  reachability, schema, and the `known_at` column for each; build `model/`
  skeleton with `point_in_time.py` and `model/tests/test_no_leakage.py`
  (truncation audit + future-row canary) proven on a real slice: a handful
  of players, one 2024 week, features built end-to-end through the gate.
  Record findings below. Anything unreachable gets a fallback noted
  (e.g. fetch via GitHub Actions instead of the sandbox).
- **Phase 1 — feature store.** Build 2021–2025 raw tables + feature builder
  through the gate; leakage tests extended to every feature family.
- **Phase 2 — models + the answer.** LightGBM vs XGBoost vs CatBoost,
  quantile variants, walk-forward 2025, SHAP importance report: which data
  actually predicts, per position. Kill features that don't earn their
  complexity.
- **Phase 3 — ship.** Winning model writes a column next to `E_pts` in the
  weekly pipeline; run side-by-side for several weeks before any recipe
  prefers it. Rookie college priors join here. POLICY.md/skill updated,
  this file deleted.

## Phase 0 findings

Done 2026-09-29 (branch `claude/eager-goldberg-zmevs9`). Shipped: `model/point_in_time.py` (loaders +
gate), `model/features.py`, `model/audit.py`, `model/probe_sources.py`, `model/tests/test_no_leakage.py`,
`model/requirements.txt`. Run: `python3.12 -m venv model/.venv && model/.venv/bin/pip install -r
model/requirements.txt`, then `model/.venv/bin/python -m unittest discover -s model/tests -t .` (24 tests,
~30 s, real 2024 data, first run downloads ~11 MB into git-ignored `model/cache/`) and `python -m
model.audit` (proof slice + leaky-join catch). `python -m model.probe_sources` re-runs the source probe
(use it from Actions for the blocked hosts).

### Source verdicts

| Source | From sandbox | Verdict and what we read | known_at |
|---|---|---|---|
| nflverse release assets (`github.com/nflverse/nflverse-data/releases/download/...`) | Yes | Everything we need is there. Read directly with pandas (see finding 1). Weekly stats: `stats_player/stats_player_week_{y}.parquet` (2021 to 2026 wk3, 150 cols, has `game_id`, includes K/DEF rows: filter to QB/RB/WR/TE). Snap counts (16 cols, PFR ids, 99.76% of skill rows map to gsis via `players.parquet`). Injuries (16 cols, one row per player-week, the week's final designation). Depth charts (finding 3). Combine (18 cols, PFR ids only, 79% map to gsis). Draft picks (36 cols, incl. career outcomes: trap). Players (39 cols, incl. current-state fields: trap). PBP reachable (HEAD 200, 20.6 MB/season), not pulled. | game rows: kickoff; injuries: `date_modified` (<=2024) else derived; depth: derived or snapshot `dt`; static: season start - 7d |
| nflverse schedules (`schedules/games.parquet`) | Yes | 46 cols. 2024: `gameday`+`gametime` (US Eastern) 100%, roof 100%, spread/total 100%, moneylines, `home_rest`/`away_rest` (0 mismatches vs recomputed from gamedays, 538 team-games), `location` (Neutral flag), temp/wind 97.3% of outdoor games (NaN indoors by design). | fixtures: season start (playoffs kickoff-24h); lines: kickoff-60min; weather: kickoff-24h, declared proxy; results: kickoff |
| ffopportunity (`ffverse/ffopportunity/releases/download/latest-data/ep_weekly_{y}.parquet`) | Yes (assets). GitHub REST API/release pages for the repo: 403 "not enabled for this session" (irrelevant to asset URLs) | 2021 to 2026 wk3, current. 159 cols, player-week xFP: `total_fantasy_points_exp` plus `pass_/rec_/rush_fantasy_points_exp` and `_diff`/`_team` variants. Scoring is PPR (mean abs diff to nflverse `fantasy_points_ppr` 0.049; half-PPR 2.24). 7% of rows have null `player_id` (419 of 6,005 in 2024; dropped). Only 89% of skill stat rows have an xFP row (players with no pass/rush/target opportunity). `game_id` joins 100%. | game kickoff (lag-only; same-week = baseline) |
| FantasyFootballCalculator ADP | **No**: `ProxyError ... Tunnel connection failed: 403 Forbidden` (egress policy; also `www.` host) | Schema NOT observed. Nothing faked. | season start (check whether the response carries a drafts date range) |
| CollegeFootballData | **No**: same 403 on `api.collegefootballdata.com` and `apinext.collegefootballdata.com`. The unauthenticated 401 shape was therefore not observed | Needs a free key. **Follow-up for James: sign up for a free CFBD key and add it as Actions secret `CFBD_KEY`** (never in the repo). `model.probe_sources` reads `CFBD_KEY` from the env. | season start, career-to-date through the prior season only |
| Open-Meteo | **No**: same 403 on `archive-api`, `historical-forecast-api` and `api.open-meteo.com` | Not observed. Per Open-Meteo's docs (unverified here) `archive-api` is reanalysis (observed-like), so the v2 forecast-issue-time replacement needs `historical-forecast-api`, not the host the plan names. | forecast issue time |
| nfl_data_py 0.3.3 | pip works (PyPI is reachable) | Do not depend on it (finding 1). Probed with `--no-deps`: weekly 2024 ok, weekly 2025 404, snap counts/injuries/depth charts/combine/draft picks/ids ok, `import_schedules` and `import_weekly_rosters` 403 (plain-HTTP habitatring.com). | n/a |

Fallback for the three blocked hosts: fetch in GitHub Actions and cache (Actions artifact or a small data
file). Not built. Alternative: an environment admin adds `fantasyfootballcalculator.com`,
`api.collegefootballdata.com`, `archive-api.open-meteo.com` and `historical-forecast-api.open-meteo.com` to the
allowed domains. Only ADP (Phase 1) and CFBD (Phase 3) are on the critical path.

### What changes Phase 1's design

1. **nfl_data_py is out; loaders read release assets directly.** 0.3.3 pins pandas<2 and numpy<2 (unsolvable
   against the repo's pandas>=2 on 3.12; it only installs with `--no-deps`), its weekly stats read the legacy
   `player_stats` release, frozen since May 2025 (2025 = 404, so the 2025 backtest is impossible through it),
   and schedules come over plain HTTP. `stats_player_week` replaced it; renames: `recent_team`->`team`,
   `interceptions`->`passing_interceptions`, `sacks`->`sacks_suffered`, `sack_yards`->`sack_yards_lost`.
2. **The row universe must be defined from pre-kickoff information.** Training only on (player, week) rows that
   have a stats row conditions on "he played" (survivorship): a player ruled Out has no row, and at serving time
   you do not know he will not play. Phase 1 needs an explicit spine (rostered/depth-chart/injury-listed) with
   DNP scored as 0 or modeled as a separate P(plays). `make_target` (identity only) is a Phase 0 stand-in and
   is the one function allowed to read target-week rows.
3. **Depth charts change format between train and test.** <=2024: weekly, no timestamp (we stamp them with the
   team's kickoff that week, so week N's chart cannot inform game N), `depth_team` 1/2/3. 2025+: ESPN-style
   snapshots with a real `dt` (221 in the 2025 file, 206 so far in 2026, roughly daily, 554k rows/season, all
   position groups; loader keeps offense), `pos_slot`/`pos_rank`. Same feature, different generating process:
   either exclude depth rank from v1 or reconstruct a weekly view from the snapshots (e.g. last snapshot
   before a fixed weekday) for 2025 and validate it against the old semantics. 234 `SBBYE` rows in 2024 have no
   week and are dropped.
4. **Injury timestamps changed too.** <=2024 rows carry `date_modified` (lead before kickoff: median 47.2 h, p1
   23.9 h; 1 of 6,215 rows in 2024 was modified after kickoff, BAL Van Noy for the KC opener, +2.9 h: the gate
   excludes it). 2025+ rows have no timestamp: we derive kickoff-24h (conservative). Consequence:
   `inj_days_since_report` is a real duration in training and exactly ~1 day in the 2025 test, a train/test
   skew. Prefer `inj_weeks_since_report` (built) or derive the same rule for every season. Also: the table is
   one final snapshot per player-week (Friday designation), so Wednesday/Thursday practice trend does not
   exist here; 2 duplicate (player, week) rows in 2024; 36 whitespace-only `practice_status` (cleaned).
5. **Observed weather is backtest-only.** The v1 proxy cannot exist at prediction time (2026 schedule rows have
   temp/wind on only 12% of games, filling in weeks late), so a model trained on it has train/serve skew in
   Phase 3. Treat weather as a backtest curiosity, or plan the forecast source (blocked host) before Phase 3.
6. **Static tables are whitelisted column-by-column in the loaders.** `draft_picks` carries `w_av`, `car_av`,
   `games`, yards/TDs, `probowls`, `allpro`; `players` carries `status`, `latest_team`, `last_season`,
   `years_of_experience` (all "as of today"). A test denies those columns. Experience is `season - rookie_season`.
7. **known_at = kickoff for game rows is a lower bound on availability.** Harmless today because features only
   cross a team's own games and its opponent's (>= 3 days apart), but a cross-team same-week feature (e.g.
   league-wide points allowed through Sunday's early slate) needs kickoff + ~4 h. Snap counts (PFR) and xFP
   also publish after the game; for the Phase 3 live run check `Last-Modified` (2026 wk3: stats 15:45Z,
   injuries 13:59Z, schedules 17:06Z on Tue, ahead of the 18:37Z run). nflverse stat files are regenerated
   after corrections, so backtest lags are final values, slightly cleaner than real time.
8. **Lines are closing lines** (no close timestamp published); sign convention verified (`spread_line` > 0 =
   home favored; corr 0.456 with home margin, 1,359 regular-season games 2021-25). A mid-week (Wed) line would be
   more honest for start/sit but nflverse does not archive opens.
9. **xFP coverage:** decide NaN vs 0 for players with no xFP row (11% of skill stat rows); the lag features
   currently give NaN.

### Volumes and cost (2021-2025 through the loaders)

`player_games` 30,750 skill rows; `snap_counts` 132k; `xfp` 28k; `injuries` 29k; `depth_charts` 617k offense
rows (554k of them 2025 snapshots: dedupe to one per team-week before modeling); `fixtures` 1,424 games. Store
builds in ~12 s and ~275 MB of pandas frames (185 MB is depth charts); on disk 22 MB of parquet. PBP would add
~20 MB/season. 2026 data through week 3 is already in every release, so the live pipeline can use the same loaders.

### The harness

**Design.** Every loader stamps an explicit `known_at` (rule written beside the table; `RawStore.describe()` lists
them, weather carries a `proxy` flag), and `as_of_join(store, table, kickoff, **keys)` is the only door: it
returns rows with `known_at < kickoff` strictly, the store hides its frames, and an AST test fails the build if
`features.py` imports a loader or touches raw frames. The audit rebuilds each row from a store physically cut to
`known_at < kickoff` and demands byte-identical features (`features.fingerprint`), and each build can return a
trace of exactly what it consumed.

**Tests** (`model/tests/test_no_leakage.py`, 24, all pass on real data, `-W error::DeprecationWarning` clean):
gate semantics (strict `<`: a row at kickoff excluded, 1 ns before included; naive timestamps, unknown columns
and `None` keys refused; a table with no/NaT `known_at` cannot exist); loader contracts (every derived
`known_at` pinned to its rule; DST: Nov 3 1pm ET = 18:00Z vs Oct 27 = 17:00Z; 2025 injury/depth formats;
denylisted columns); (a) truncation audit over 306 rows (weeks 1-22, 105 distinct kickoffs, all four positions,
55 rookie rows, 13 listed players with no stats row): 0 mismatches; (b) canaries: fake monster game next week,
fake 'Out' report +1 h, wild line/weather/draft pick, rows stamped exactly at kickoff, and the target game's own
rows replaced with 987654: features unchanged; a control stamps the same rows 1 s BEFORE kickoff and every
touched family moves (so (b) cannot pass vacuously); (c) schedule sanity: every consumed row has `known_at <
kickoff`, the target game_id never appears in any result table's inputs, all consumed games are from earlier
weeks, `lagK_week` strictly decreasing and < week, DvP windows end at week-1, same-week xFP is not a feature.

**The audit bites.** `audit.leaky_features` does the ad-hoc merge (history `week <= target`, same-week xFP as a lag,
injury by (player, week) with no `known_at`): the truncation audit flags it on 281 of 306 rows (281 of 293 rows
whose own game is in the raw table; its lag-1 equals the target game's actual score on every such row), the gated
builder gets 0. Mutating the gate itself also fails the suite: `<=` fails 10 tests, ignoring `known_at` fails 11.
Independent checks on the proof slice: DvP recomputed from raw parquet matches exactly (DAL vs RB 25.6; DET vs WR
37.075); DET's bye week is skipped because windows count games, not calendar weeks.

**Residual risk the audit cannot see:** a wrong `known_at` (the audit trusts it). That is why each derivation is
pinned by a test and why 2025 injury/depth stamps are conservative.

### Phase 1 watch-outs

Define the spine first (finding 2). Decide within-season vs cross-season lags (v1 is within-season, so week 1 is
all NaN except static features). Keep every new family in `features.py` behind the gate and extend
`test_no_leakage.py` per family (truncation sample, canary keys, `expect` set in the control). Never add a column
to a static loader without checking it is knowable at season start. Train/test skew list: depth-chart semantics,
`inj_days_since_report`, weather. Fix the numpy pin (`<2.5`) only when pandas is bumped.
