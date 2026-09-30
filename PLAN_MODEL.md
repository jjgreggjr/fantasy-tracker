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

## Phase 1 findings

Done 2026-09-29 (branch `claude/eager-goldberg-zmevs9`). Shipped: `model/features.py` (spine + every feature
family, still only `as_of_join`), `model/labels.py` (y_* and baselines: the only reader of target-week rows besides
the spine plumbing), `model/build_features.py`, `model/adp.py`, `model/fetch_adp.py`, `model/fetch_cfbd.py`,
`.github/workflows/model_fetch.yml`, an indexed gate in `model/point_in_time.py`, `model/tests/test_feature_families.py`.
Run: `model/.venv/bin/pip install -r model/requirements.txt`; `model/.venv/bin/python -m model.build_features` writes the
git-ignored `model/cache/features.parquet` plus `features_schema.json` (column -> family/role) and a report;
`model/.venv/bin/python -m unittest discover -s model/tests -t .` (70 tests, ~3.5 min). Nothing under `ff/`,
`pipeline.yml` or committed pipeline data was touched; the pipeline's own 20 tests still pass.

### Result

48,658 rows x 133 columns (107 feature columns in 12 families; the rest are identity, spine flags, `y_*`, `base_*`),
4.9 MB parquet, 72 MB in pandas. Build: 310 s single process (15 s store load, ~6 ms per row), 1.25 GB peak RSS.
33,423 rows are `y_played == 1` (29,391 with a stats row, 4,032 snap-only appearances scored 0.0); 15,235 spine rows
are DNP/unused (NaN target). Regular season, completed games only (2026 = weeks 1-3; week 4 has lines but no result).

| season | QB | RB | TE | WR | rows | played | depth-only rows |
|---|---|---|---|---|---|---|---|
| 2021 | 1,366 | 2,232 | 2,034 | 3,574 | 9,206 | 6,344 | 625 |
| 2022 | 1,321 | 2,190 | 2,058 | 3,486 | 9,055 | 6,302 | 686 |
| 2023 | 1,342 | 2,114 | 2,010 | 3,493 | 8,959 | 6,407 | 598 |
| 2024 | 1,323 | 2,051 | 2,024 | 3,499 | 8,897 | 6,407 | 549 |
| 2025 | 1,542 | 2,584 | 2,366 | 4,060 | 10,552 | 6,755 | 2,201 |
| 2026 wk1-3 | 297 | 438 | 487 | 767 | 1,989 | 1,208 | 445 |

Mean null rate by family on played rows (2021 / 2022 / 2023 / 2024 / 2025 / 2026): lags .13/.13/.12/.12/.15/.62 (2026
has three weeks), prev_season .16/.16/.18/.16/.18/.13 (rookies), td_luck .09/.09/.09/.08/.11/.34, role .00 all, injury
.47/.45/.46/.46/.48/.66 (NA = not on a report, the normal state), dvp .05/.04/.04/.04/.06/.33, vegas 0, context 0,
weather .16/.32/.22/.18/.17/.16 (see below), static .21/.21/.21/.23/.24/.24 (combine coverage, undrafted), adp 1.0,
college 1.0 (both NA by design, below). Baselines on played rows with >=3 games: trailing-3 average MAE 4.35-4.66,
corr .62-.65; same-week xFP MAE 2.56-2.77, corr .85-.87; both stable across 2021-25, so the bar Phase 2 has to beat is
about 4.4 MAE (trailing) and 2.6 (xFP, an oracle: it is post-game).

### The row universe (spine)

A row is a (player, game) admitted by at least one signal known before kickoff, recorded in `spine_src`: `usage`
(appeared for the team in one of its last 3 games, crossing seasons), `injury` (this week's report, any designation,
so Out players with no stats row are rows), `depth` (the newest chart known before kickoff, if under 21 days old),
`draft` (the draft class, week 1 only). Usage-only candidates are dropped if a fresher signal puts them on another team.
`y_played` = stats row OR offense_snaps > 0; `y_points_ppr` = the stats value, 0.0 for a snap-only appearance, NaN for a
DNP. Deleting or absurdifying the target game's rows changes no row and no feature (tested).

Recall of players who actually played (share reached by the pre-kickoff spine):

| season | all played rows | week 1 | weeks 2+ | rows with >=5 opportunities | PPR-weighted |
|---|---|---|---|---|---|
| 2021 | .937 | .737 | .950 | .965 | .970 |
| 2022 | .943 | .698 | .959 | .970 | .969 |
| 2023 | .954 | .714 | .969 | .978 | .981 |
| 2024 | .954 | .731 | .967 | .979 | .980 |
| 2025 | .996 | .992 | .997 | .999 | 1.000 |
| 2026 | .993 | .990 | .995 | 1.000 | .999 |

**This is the biggest Phase 2 trap.** The 2025 depth chart is a timestamped daily snapshot, so it sees offseason
signings and trades; 2021-24 have no pre-kickoff roster source, so a veteran who changed teams in the offseason and is
not on the week-1 injury report is absent from week 1 (recall ~.70-.74, weeks 2+ ~.95-.97). Training rows therefore miss
~5% of played rows (mostly low-volume, new-to-team players) that the 2025 test set contains. `spine_depth_only == 1`
(admitted only by the chart: 549-686 rows a year to 2024, 2,201 in 2025) marks that population; score 2025 both with and
without it. Nothing better exists: nflverse rosters carry no snapshot time, so they cannot be stamped.

### Feature families and the calls behind them

- **Lags**: last 1/2/3 appearances this season (points, opportunities = carries + targets, carries, targets, pass
  attempts, target share, carry share, snap share, lagged xFP) with weeks-ago, plus season-to-date means. Carry share is
  computed at load from the same game's team totals (nflverse has no such column). Lags stay within-season, so week 1 is
  NaN; **`prev_season_*` (games, ppg, opps/carries/targets per game, target share, snap share, xFP per game) carries the
  prior season** and `td_luck_total_prev` its TD luck. 2020 is loaded as history so 2021 is not a different kind of row.
- **xFP coverage (finding 9 resolved)**: the 7% null-`player_id` rows in ffopportunity are team-level aggregate rows
  (no name, no position), not lost players, and of 32,363 regular-season stats rows with any pass attempt, carry or
  target (2020-26) none lacks an xFP row. Rows without one are exactly the zero-opportunity games, so no xFP row +
  no opportunity = 0.0, never NaN. Snap-only appearances are 12% of played rows (median 8 snaps) and count as 0-volume
  games in lags, not skipped games.
- **TD luck**: cumulative (actual - expected) rush/rec/pass/total TDs this season from lagged ffopportunity.
- **Role** (usage-derived, not depth-chart derived): `role_score` (trailing-3 snap share, last season's if none),
  `role_rank_pos`, and teammate-injury context over teammates who appeared in the team's last 3 games: same-position
  Out/Doubtful count, count ranked above him, Questionable count, group size, trailing volume vacated (same-position
  opportunities; team-wide targets and carries).
- **DvP**: opponent PPR allowed to his position over its last 2 / 4 / all completed regular-season games.
- **Vegas** (closing lines; also opponent implied total), **context** (home, rest days both sides, divisional, neutral,
  week, season), **static** (age, experience, career games, draft round/pick with the players table as fallback,
  undrafted flag, height/weight, six combine metrics), **weather** (`wx_temp_obs`, `wx_wind_obs`, `wx_roof_obs`,
  `wx_indoor_obs`), **adp** and **college** (present, all NA).
- Career games = a static table of skill-position games through 2019 (stamped at the 2020 opener) + gated history from
  2020, so the matrix builder refuses a store missing any season since 2020. Draft picks use PFR team codes; mapped to
  nflverse's (GNB->GB, KAN->KC, LAR->LA, LVR->LV, NOR->NO, NWE->NE, SFO->SF, TAM->TB).

### Cross-season comparability calls

| Field | Problem | Call |
|---|---|---|
| Depth-chart rank | Two formats are two signals. The chart in force for the previous game names the week's snap leader at RB 56-66% of the time through 2024 vs 75% (2025) and 82% (2026); WR top-3 overlap .64-.68 vs .79-.80; QB .79-.81 vs .85-.89; the WR "first string" holds 2.4-2.6 players to 2024, exactly 3 from 2025 (agreement measured chart-in-force-for-the-previous-game vs the target week's snaps, weeks 2+) | **Excluded as a feature** (no `depth*` column exists; a test pins that). The chart is used only for spine admission, and role comes from usage. |
| `inj_days_since_report` | real report timestamps to 2024, derived kickoff-24h from 2025 (always ~1 day): pure train/test skew | Produced (Phase 0 tests read it), **excluded from the matrix** (`EXCLUDED_FROM_MATRIX`); `inj_weeks_since_report` is the feature. |
| Injury designation mix | Skill-position rows per team-week: Out .49/.57/.50/.52 (2021-24) vs **.74** (2025); Doubtful .11/.09/.09/.13 vs .05; Questionable .75/.80/.85/.72 vs .61; total listed rows stable at 2.8-3.0. Out players who played anyway: ~0% every season, so "Out" means out throughout | Teammate features count only players who appeared in the team's last 3 games (a long-term IR listing cannot move them) and treat Out+Doubtful together. Mean same-position Out/Doubtful teammates on played rows: .155/.178/.141/.151/.189 (2021-25). The own-status column `inj_report_status` still drifts (Out 4.2% of rows in 2024, 6.2% in 2025): consider collapsing it. |
| Weather | nflverse outdoor-game temp/wind is null for 0% (2021), **46% (2022)**, 17% (2023), 3% (2024), 2% (2025); roof coding split retractable roofs into closed/open through 2023 and calls every one 'closed' from 2024 | `wx_roof_obs` = dome / retractable / outdoors (comparable, and the day-of open/closed decision is not smuggled in); `wx_indoor_obs` = fixed dome only. Observed values stay backtest-only; 2022-23 temp/wind are a missingness artifact, not a signal. `roof` was removed from the season-start fixtures table (it was stamped as known at season start; the retractable state is decided on game day). |
| Game-result availability | game rows are stamped at kickoff, a lower bound on availability (finding 7). The extended audit found a real case: a teammate on SEA's chart had played the 4:25pm TB@LAC game 3h55m before SEA's Sunday-night kickoff | The four result tables are read at **kickoff - 4h** (`RESULT_LAG`). Thursday -> Sunday and same-team games are untouched. |
| Weight/height | the players table is an as-of-today bio, so veterans' weight can reflect later seasons | Kept (small, slow-moving); ablate if it ranks high. |

`depth_charts` loading drops 10,560 skill rows with no gsis id and 1,159 no-week SBBYE snapshot rows.

### ADP and CFBD via Actions: not run, exact reason

`.github/workflows/model_fetch.yml` is on the branch (dispatch-only, no schedule, never references pipeline.yml, commits
`model/data/adp/` and `model/data/cfbd/` CSVs plus `fetch_log.csv` with the exact status/exception of every request, and
re-runs `model.probe_sources` into `model/data/probe_report_actions.txt`; the `CFBD_KEY` secret reaches one step through
the environment and is never logged). **Triggering it failed**: `POST /repos/jjgreggjr/fantasy-tracker/actions/workflows/
model_fetch.yml/dispatches` returned `404 Not Found` (twice, after each push), and `list_workflows` shows only
`pipeline.yml`. GitHub registers a `workflow_dispatch` workflow from the default branch only, and `main` does not have this
file; `pipeline.yml`, which does, was dispatched against this branch by James at 17:04Z, so a branch ref works once the file
is on `main`. Not worked around (pushing another branch or adding a push trigger were outside the brief).
**To finish, in this order**: put only `.github/workflows/model_fetch.yml` on `main` (the run uses the branch's code
through `ref`), dispatch it with ref `claude/eager-goldberg-zmevs9`, `git pull`. Then ADP lights up with no code change
(`load_store` adds an `adp` table when `model/data/adp/adp_*_*.csv` exist; `known_at` = season start, or the FFC drafts
window end if the response carries one, whichever is later, so it can only hide ADP, never leak it) and the FFC schema is
finally observed. The sandbox-side FFC error is unchanged (`ProxyError ... Tunnel connection failed: 403 Forbidden`);
whether Actions can reach FFC/CFBD is still unknown. ADP and the four college columns are 100% NA in the matrix.
**First Phase 2 to-do**: read `model/data/cfbd/*.csv` and write `model/college.py`. Wiring was deliberately not written
blind (the CFBD schema was never observed; Phase 0's rule is that nothing is faked), and the college fetch keeps only names
matching a drafted QB/RB/WR/TE, so market share/dominator can be derived once stat-type names are seen. Breakout age needs
more than the final college season the fetch pulls.

### Tests

70 tests, all passing with `-W error::DeprecationWarning`: the 24 from Phase 0 plus 46 new (indexed gate == the Phase 0
mask gate on >1,000 random queries; spine and every feature identical from physically truncated tables over ~700 sampled
rows from 2022/2024/2025/2026; canaries stamped after/at kickoff move nothing in any family (a post-kickoff teammate
report, next week's monster game, an opponent's future game, a phantom teammate, phantoms for each spine signal, a
prior-season game, static rows); controls just inside the cutoff move each family; deleting or perturbing the target game
changes no row or feature; matrix contract; label recomputation from raw; ADP and fetcher unit tests on synthetic rows,
including that the CFBD key is never logged). The audit still catches a deliberately leaky teammate builder on a planted
post-kickoff report. Mutations, each run against the whole suite in an isolated copy (every one is caught): `<=` at kickoff in the gate
fails 14 tests; the indexed gate ignoring `known_at` 28; `RESULT_LAG = 0` 1 (the 4h invariant; a pin test now also fails
it, because the controls read the constant); DvP reading a week ahead 10; lags/TD luck reading a day past the target game
16; the teammate report read a day late 3; **the spine reading injuries a day late 1, the canary test only**. That last
one is the honest limit of the truncation audit: real data has no injury row inside that 24-hour window (2025+ stamps are
derived at kickoff - 24h; <=2024 has one late row in ~6,000), so the audit passes and the planted-row canaries are what
bite. Keep both. Four Phase 0 assertions changed because the extension made them wrong, not to make them pass: the table set gained
`career_pre_cutoff`; the xFP name list gained `xfp_std_mean` and `prev_season_xfp_pg` (still an exact list); "consumed games
are from earlier weeks" became "had finished before kickoff (+4h)" (same-week Thursday games are legitimate for a player
who changed teams); the control canaries for result tables sit inside the 4-hour cutoff.
Three spot-checks against raw parquet with plain pandas (independent path, 114 fields): Eno Benjamin 2022 wk6 (ARI vs SEA),
Davante Adams 2025 wk5 (LA vs SF), Dohnte Meyers 2026 wk3 (CIN at PIT, rookie): every lag, mean, TD luck, DvP, Vegas, rest,
teammate-Out count, age, career games, draft capital, target and same-week xFP matched. The first pass showed 2 diffs on
the 2022 row, which were the hand check ignoring snap-only appearances in last season's games; with the documented
definition it matches exactly.

### What Phase 2 must watch for

1. **Spine asymmetry** (above): train on `y_played == 1`, evaluate 2025 with and without `spine_depth_only`. Any rank
   metric that includes new-to-team veterans is easier or harder for reasons unrelated to the model.
2. **Weather is backtest-only** and its NaN pattern is a season effect (2022-23). Ablate `wx_*` first; do not ship it
   without the forecast source (Open-Meteo host still blocked).
3. **Drop all-NA columns** (`adp_*`, `college_*`) from any fit until they exist; do not let a constant column reach SHAP.
4. **`season` and `week` are features** (the plan listed them); `season` cannot extrapolate to 2025 in a tree, ablate it.
5. **Snap-only appearances** (12% of played rows, y = 0): decide whether to train on `y_played` or `y_has_stats_row`.
6. **2026 has three weeks**: usable for live checks, not for training or a season-level metric.
7. `inj_report_status` drifts in mix (Out share up in 2025) and `inj_practice_status` was cleaned but not audited for drift.
8. Lines are closing lines; `base_xfp_sameweek` is post-game and an oracle, so "beating" it is not the goal, matching the
   trailing-3 baseline's 4.4 MAE by a wide margin is.
9. Playoffs are not built (fantasy-irrelevant, thin); `build(..., )` takes weeks/seasons and the spine already handles them.
10. Residual risk the audit cannot see is unchanged: a wrong `known_at`. New stamps this phase: depth snapshots (`dt`),
    ADP (season start or window end), career table (2020 opener); all pinned by tests except the unobserved FFC window.

## Phase 2 findings

Done 2026-09-29 (branch `claude/eager-goldberg-zmevs9`). Shipped: `model/college.py` (CFBD wiring), `model/train.py` (design
matrix, three libraries, seeded tuning), `model/backtest.py` (walk-forward, scoring, ablation stages), `model/explain.py` (SHAP,
prune list), `model/report_phase2.py`, `model/tuned_params.json`, `model/reports/phase2_backtest.md` (every table below and more;
committed), `model/tests/test_backtest.py`; `model/requirements.txt` gained lightgbm/xgboost/catboost/shap. Run, in order:
`python -m model.train --tune`, `python -m model.backtest bakeoff|quantiles|ablate|replicate|prune|cadence`, `python -m
model.explain --lib lightgbm [--season 2024]`, `python -m model.report_phase2`. Outputs go to git-ignored `model/cache/phase2/`;
`ff/` and `pipeline.yml` untouched. 108 model tests (70 + 38 new) and the pipeline's 20 pass. About 35 minutes of compute in all
(4 cores; bake-off 3 min, one weekly walk-forward of LightGBM 17 s, XGBoost 60 s, CatBoost 120 s).

### Result: the answer to the question

**LightGBM wins by tie-break; it beats the trailing-3 average clearly and closes a fifth of the gap to the post-game oracle.**
Head-to-head rows (2025 played rows with at least one earlier game, 6,139), ALL positions: MAE 4.02 vs 4.31 trailing vs 2.50 oracle,
RMSE 5.71 / 6.43 / 4.07, Spearman within position-week .655 / .602 / .866, pick accuracy .773 / .748 / .876 (start/sit-relevant pairs
.641 / .597 / .784). Bootstrap (18 weeks resampled) intervals for model-minus-trailing exclude zero for every metric in every
position. Per position (LightGBM / trailing / oracle): MAE QB 6.18 / 6.89 / 4.22, RB 4.24 / 4.54 / 2.71, WR 3.99 / 4.26 / 2.49, TE 3.00 /
3.17 / 1.65; Spearman QB .48 / .41 / .77, RB .75 / .71 / .91, WR .70 / .63 / .88, TE .69 / .65 / .91. Gap to oracle closed: 16% of MAE,
31% of RMSE, 20% of Spearman, 20% of pick accuracy. LightGBM, XGBoost and CatBoost are indistinguishable (pooled RMSE 5.711 / 5.706 /
5.710; both intervals against LightGBM span zero); the pre-declared rule (lowest RMSE, a statistical tie goes to LightGBM) chose the
library that also does the quantiles and SHAP natively and refits fastest. One model with position as a feature: per-position
models were worse in both years (dRMSE +0.024 in 2025, +0.017 in 2024, intervals span zero), so no position got its own model.
Quarterbacks are the weak spot. Tuned parameters are in `tuned_params.json` (12-config random search per library, train 2021-23,
validate 2024 only; every configuration landed within 0.5% of the others, so tuning barely matters here).

**Quantiles (LightGBM p10/p50/p90), all 2025 played rows:** actuals at or below p10 / p50 / p90 = 11.9% / 53.2% / 89.9% (targets 10 / 50
/ 90); the p10-p90 band covers 78.1% (target 80%; QB 74.4%, RB 78.9%, WR 78.1%, TE 78.6%). Floor slightly too high, ceiling right,
QB band too narrow. 3.2% of rows had crossed raw quantiles (TE 6.6%); scored values are sorted per row.

**Which data predicts (SHAP, out of sample, 2025):** lags 51% of total attribution (`xfp_std_mean` 14%, `pts_ppr_std_mean` 11%,
`snap_pct_l1` 5%), ADP 19% (`adp_ppr_pos_rank` 9%), role 11% (`role_score` 6%, teammate-out volume 2%), prior season 6.5%, static
4.8%, Vegas 3.0% (`implied_team_total`; the family whose removal hurts quarterbacks most, dRMSE +0.09), then context 1.7%, dvp 1.2%, td_luck 0.9%,
injury 0.6%, college 0.2%. The same 8-10 features top every position; QB adds `pos_QB`, `is_home`, `pass_att_l1`, RB `carry_share_l1`, TE `weight_lb`
(heavier = blocker = fewer points). Per-position top-15 tables with one-line readings and three worked predictions (Gibbs wk14 pred 23.4
actual 37.0; Tucker wk5, a volatile WR, pred 11.2 actual 11.1; Egbuka wk6, a rookie, pred 18.2 actual 4.4) are in the report.
**SHAP attribution is not marginal value**: ADP holds 19% of attribution but dropping it costs only +0.007 RMSE in 2025 (+0.025 in 2024),
because season-to-date form substitutes for it after a few weeks. 43 of 114 columns are under 0.15% of attribution (listed in the report).

### Ablations and how far to trust them

Every ablation is scored on the same rows with a week-blocked bootstrap and **replicated on the 2024 walk-forward** (`replicate`
stage). That replication is what the conclusions rest on: most one-at-a-time 2025 "gains" flipped sign in 2024, and the decisions
below only use effects that agree.

| Question (watch-list item) | dRMSE 2025 [95% CI] | dRMSE 2024 | Call |
|---|---|---|---|
| Train on stats-row players only instead of `y_played` (5) | +0.018 [+0.005, +0.029] | +0.027 | worse in both years: **train on `y_played`** (snap-only 0.0 rows stay) |
| Add `season` (4) | -0.015 [-0.024, -0.006] | +0.012 | mixed, left out (it is a calendar index; if recency matters, add a league-environment feature instead) |
| Add all observed weather (2) | -0.022 [-0.036, -0.007] | +0.003 | does not replicate; **weather stays out**; roof structure alone -0.006 / +0.018, temp+wind alone -0.015 / +0.009 |
| Drop the depth-chart-only rows from training (1) | +0.000 [-0.008, +0.008] | +0.007 | no effect; scoring 2025 without those rows moves model RMSE 5.711 -> 5.727 and trailing 6.432 -> 6.451 (same margin) |
| One model per position | +0.024 [-0.001, +0.048] | +0.017 | keep one model |
| Drop `week` | -0.011 [-0.022, -0.000] | +0.007 | mixed |
| Drop lags / vegas / role / adp | +0.058 / +0.016 / +0.014 / +0.007 | +0.084 / +0.040 / +0.033 / +0.025 | load-bearing in both years (lags and vegas significant in 2025; role and adp only in 2024) |
| Drop prev_season / injury / static / college / dvp / td_luck / context | -0.001 to -0.017 | +0.001 to +0.022 | sign flips: not separable from noise one at a time; joint drops `lean_A`/`lean_B` picked from the 2025 table also failed in 2024 |
| **Prune the 39 columns under 0.15% of 2024 SHAP**, test on 2025 | -0.015 [-0.027, -0.002] | n/a | the list came from 2024, so 2025 is clean: 75 columns, no worse |
| Pruned set **without ADP** (serving candidate) | +0.008 [-0.005, +0.021] vs primary | | 71 columns, RMSE 5.719, Spearman .655: still ahead of trailing-3 |
| Refit every 4 weeks / never inside the season | -0.007 / +0.003 (both span zero) | +0.010 / +0.028 (both exclude zero) | staleness is free in 2025 and costs 0.2-0.5% in 2024: refit weekly, it is cheap |

The numbering in the dispatch differed from this file's watch list (it called `season` item 7 and the target choice item 8; here they
are items 4 and 5). Each item was handled by its content: 1 spine asymmetry (above), 2 weather (out of the primary, ablated), 3 all-NA
columns (`train.dead_columns`: only `college_breakout_age`), 4 season, 5 snap-only rows, 6 2026 (three weeks, not used), 7 injury drift
(own report status collapsed to none / Questionable / Doubtful-or-Out, practice status to none / full / limited / DNP; on played rows
Out fell from 1.3% of rows in 2021 to 0.2% in 2025 and DNP practice from 5.0% to 2.4%, so the drift is real but the family carries 0.6%
of attribution), 8 oracle framing (xFP is reported as an oracle, never as the bar), 9 playoffs (not built), 10 residual risk (a wrong
`known_at`; the new stamp is the college table's, pinned by a test).

### College wiring (task 1)

`model/college.py` reads the fetched CSVs (long format: `season, playerId, player, position, team, conference, category, statType,
stat`; team totals `season, team, statName, statValue`, FBS teams only) and produces 13 columns: receiving and rushing shares of team
yards and TDs, per-game volume, per-touch efficiency behind a minimum-touch floor, a dominator rating (NA for QBs), passing rates for
real passers, a power-conference flag, and `college_breakout_age`, which is always NA (it needs several college seasons; the fetch pulls
one). Matching is name + college (+ position, or a position switch on college agreement), because the fetch's name filter is
over-inclusive: three same-name pairs in the pulls (Kevin Harris, Zach Evans, Justin Shorter) would otherwise take the wrong player's
line; the college crosswalk is a normaliser plus eight aliases, each read off a real disagreement. Ambiguous, shared-id, FCS (no team
totals, and one FCS line shows a 28-attempt QB season) and unverifiable cases are NA. **Match rate: 434 of 475 drafted QB/RB/WR/TE
(91.4%)**: 2021 90.7%, 2022 87.3%, 2023 95.0%, 2024 92.2%, 2025 89.4%, 2026 93.7%. The 41 misses are 14 FCS schools, and 27 with no
CFBD row under the drafted name (nicknames such as "Cam Ward" = "Cameron Ward", whom the fetch's exact-name filter dropped, and 2020
opt-outs); a Phase 3 fetch that keeps every FBS QB/RB/WR/TE row would recover the nicknames. `known_at` = the draft year's season start,
through the gate; the columns show only in the rookie season (a veteran row is NA, so coverage never depends on how many classes the
CSVs hold). **Coverage on rookie rows that played** (`is_rookie`, `y_played`): 2021 80.4%, 2022 68.1%, 2023 75.3%, 2024 78.1%, 2025 75.8%,
2026 81.9% (drafted rookies alone 86-94%; undrafted rookies cannot match). Leakage tests: identity rules on synthetic and real
namesakes, hand-computed features, post-kickoff and at-kickoff canaries plus a 1-second-before control that bites, a veteran-row canary,
and a truncation audit over 40 sampled rookie rows (added to the family audit). **It earns nothing**: college is 0.2% of attribution, its
removal moves RMSE by -0.011 (2025) / +0.007 (2024), and rookies are scored 5.25 RMSE (model) against 5.76 (trailing): the draft-capital
and ADP columns already carry the rookie prior.

### Housekeeping the work exposed

* The Phase 1 suite was 69/70, not 70/70: `test_every_table_has_a_rule...` still expected the pre-ADP table set. Fixed (`adp`,
  `college` are now expected tables).
* The matrix was rebuilt for the college columns. Four columns differ from the Phase 1 matrix at the 1e-15 level (`dvp_ppr_l4`,
  `dvp_ppr_std`, `tm_out_targets_all`, `tm_out_carries_all`: float summation order in the earlier build); the current code builds
  bit-identically across processes (two `PYTHONHASHSEED`s, checked) and equals this rebuild. The design matrix is float32, which hides
  the difference from the models.
* Mutation checks: `<=` in the train mask, or `season <=`, fail 6 of 9 targeted tests each; with the runtime assertion disabled the
  perturbation tests still fail (predictions move by orders of magnitude when a future label leaks).
* Not done: a comparison with our pipeline's `E_pts` (it blends platform projections that are not archived for 2025 with trailing form
  and a DvP multiplier, so it cannot be rebuilt for 2025) or with platform projections; league-scoring variants (the target is PPR).

### Phase 3 recommendation

(Updated by Phase 2.5: see the recommendation at the end of "Phase 2.5 findings" for the model design that ships; the feature set below stands unchanged.)

* **Model:** LightGBM, squared-error point estimate (the conditional mean) plus p10/p50/p90 quantile models, one model with position as
  a feature, trained on `y_played` rows, parameters as in `tuned_params.json`.
* **Features (71 columns):** the registered families minus `wx_*`, `season`, `college_*`, the 39 low-attribution columns
  (`model/cache/phase2/prune_list.json`, regenerated by `backtest prune`) and, **until a preseason snapshot exists, ADP**. Blocker to
  resolve before any recipe uses the model: the ADP fetch returned only a 29-player in-season window for 2026, so 98.6% of 2026 rows are
  NA where 67.5% of 2025 rows were; a model trained with ADP would see a train/serve skew. Either schedule an August Actions snapshot of
  FFC ADP (and stamp it at capture) for 2027, or ship without ADP (costs +0.008 RMSE, within noise). Weather, `season` and college are
  not worth wiring; the Open-Meteo forecast source is unnecessary. Load-bearing inputs to protect: season-to-date xFP and points,
  snap share (last game and trailing 3), the prior-season anchor, the Vegas implied total, and teammate-out volume.
* **Cadence:** features refresh every Tuesday after nflverse stats land (2026 week 3 stats were up 15:45Z, ahead of the 18:37Z run);
  refit the model weekly (about 1 s for the point model, a few seconds for the quantiles; a stale model cost nothing in 2025 and up to
  +0.028 RMSE in 2024, so weekly is the safe choice at no cost); retune once each preseason on the prior season (under a minute for LightGBM) and regenerate the prune list
  then. Recalibrate the quantile band with trailing residuals (the QB band covers 74% against 80% nominal) before showing floors and ceilings.
* **How to judge it in 2026 before any recipe prefers it:** the pipeline commits point-in-time Sleeper projections since 2026 week 1, so
  score the model, Sleeper and `E_pts` on the same 2026 weeks with `backtest.score` (Spearman within position-week and pick accuracy;
  the head-to-head metric is the one that matters; three weeks of 2026 make MAE differences of a few tenths noise). The 2025
  result to beat: Spearman .655 and pick accuracy .773 pooled, .48 Spearman at QB.
* **Open design question for James:** league-scoring variants. The model predicts PPR; half-PPR/standard/TE-premium need either component
  models (receptions, yards, TDs) or a per-position linear map. The component route is the honest one and is a Phase 3 sub-task.
* Drop `week` from the inputs only if a joint test on both years agrees (it flipped sign); everything else in the prune list was tested.

## Phase 2.5 — volume-first experiments (added 2026-09-30, runs before Phase 3)

The oracle gap says the remaining edge is predicting volume, so every experiment
here points at volume. Baseline to beat is the Phase 2 primary on the same 6,139
head-to-head rows: RMSE 5.711, Spearman .655, pick accuracy .773. Same
walk-forward, same seeds, and the Phase 2 discipline holds: a gain that does not
replicate on the 2024 walk-forward is noise, and every new feature goes through
the gate with its own truncation and canary coverage.

Experiments, in order of expected payoff:

1. **Usage micro-signals from nflverse play-by-play** (lagged only, `known_at` =
   the source game's kickoff): red-zone and inside-10 target/carry share, air
   yards share, aDOT, WOPR, and route participation where a source covers the
   backtest years (probe FTN charting and the participation feed; document the
   year coverage and skip what 2021–2025 cannot support consistently — a
   feature that exists only in test years is a train/serve skew, not a signal).
2. **Two-stage model:** stage one predicts week-N volume (targets, carries —
   effectively predicting xFP's inputs), stage two predicts points from
   predicted volume plus efficiency priors. Judged both as a point model and by
   whether its stage-one output beats the lagged-xFP features when added to the
   flat model.
3. **Component models:** predict receptions, receiving yards, receiving TDs,
   carries, rushing yards, rushing TDs, passing lines separately; compose
   points in any league's scoring. Judged on composed PPR against the primary —
   and adopted for Phase 3 even on an accuracy tie, because it unlocks
   league-scoring variants (the open design question above).
4. **Cheap wins:** ensemble of the three tied libraries (simple average, then a
   weight fit on 2024 only); quantile recalibration from trailing residuals
   (target: QB p10–p90 coverage from 74% to ~80% without widening RB/WR/TE).

Adoption rule: an experiment ships into the Phase 3 recommendation only if it
beats the primary on 2025 AND replicates directionally on 2024, or (component
models only) ties within noise. The findings section below records per-
experiment deltas with the same bootstrap intervals as Phase 2.

## Phase 2.5 findings

Done 2026-09-30 (branch `claude/eager-goldberg-zmevs9`). Shipped: three play-by-play raw tables and four opt-in feature families through the gate
(`model/point_in_time.py`, `model/features.py`), component labels (`model/labels.py`), `model/p25run.py` (cached walk-forwards, tree counts,
head-to-head tables), `model/volume.py` (two-stage), `model/components.py` (component models, scoring composition), `model/ensemble.py` (ensemble,
quantile recalibration), `model/phase25.py` (runner), `model/report_phase25.py`, `model/reports/phase25_experiments.md` (every table below and more;
committed), `model/tests/test_pbp_families.py` and `test_phase25_models.py`. Run: `python -m model.build_features`, then `python -m model.phase25
reproduce|exp1|exp2|exp3|exp4|all`, then `python -m model.report_phase25` (about 90 minutes of compute in all, 4 cores, matrix rebuild 12 minutes). `ff/`,
`pipeline.yml` and the pipeline's 20 tests are untouched and pass; the model suite is 172 tests (108 + 64 new).

**Baseline held.** The matrix was rebuilt with 85 new columns; the 142 Phase 2 columns are bit-identical (same dtypes, same values) and the Phase 2 primary recomputed
from it equals the Phase 2 predictions exactly (max |difference| 0.0 for LightGBM, XGBoost and CatBoost 2025 and LightGBM 2024); the tree-count protocol reproduces the
primary's 148. New families are opt-in for the models (`train.OPT_IN_FAMILIES`), so the primary keeps its Phase 2 inputs. Every variant is scored on the same 6,139 (2025) and
6,025 (2024) head-to-head rows with the Phase 2 week-blocked bootstrap; a variant with new inputs gets its own tree count by early stopping on 2024 after training on 2021-2023
(the protocol that produced the primary's), so 2024 carries a mild in-sample edge for every variant equally.

### Verdicts against the adoption rule

dRMSE is variant minus primary (negative = better), 95% week-blocked interval. The rule: beat the primary on 2025 AND the same sign on 2024 (component models: a tie within
noise also ships).

| # | Experiment | 2025 dRMSE | 2024 dRMSE | Verdict |
|---|---|---|---|---|
| 1 | play-by-play, servable set (`pbp`: usage + team pace) | -0.008 [-0.024, +0.007] | +0.004 [-0.009, +0.016] | **does not ship** (sign flips) |
| 1 | usage alone / team alone / participation alone | -0.008 / +0.000 / -0.001 | +0.009 / +0.009 / +0.001 | does not ship |
| 1 | + participation (`pbp_part`, backtest-only) | -0.010 [-0.028, +0.006] | -0.005 [-0.017, +0.006] | passes on direction only, inside the noise; **not adopted**: feed not published in-season |
| 2 | two-stage (stage two alone) | +0.067 [+0.031, +0.101] | +0.088 [+0.047, +0.127] | **does not ship** (worse both years) |
| 2 | flat + stage-one predictions / + efficiency priors / both | -0.003 / -0.004 / -0.010 | +0.020 / +0.006 / +0.009 | does not ship |
| 3 | component models composed to PPR | -0.008 [-0.029, +0.013] | +0.006 [-0.020, +0.034] | **ships (tie within noise)** |
| 3 | components + efficiency family | +0.006 [-0.022, +0.034] | +0.003 [-0.030, +0.036] | ties too; not preferred (18 extra features, no gain) |
| 4a | average of LightGBM / XGBoost / CatBoost | -0.011 [-0.020, -0.003] | -0.004 [-0.012, +0.005] | **ships** (2025 significant, 2024 same direction) |
| 4a | weights fit on the previous year | -0.009 [-0.015, -0.004] | -0.002 [-0.013, +0.008] | passes, but no better than equal weights: use the plain average |
| 4b | quantile recalibration from trailing residuals (`underage_only`) | QB coverage .744 to .789 | QB coverage .751 to .794 | **ships, one flagged miss**: RB untouched, WR/TE widened +0.5% / +1.2% (2025), +2.2% / +1.4% (2024) where their own trailing coverage was below 77%; my +2% cap is missed by 0.2 point (2024 WR); `qb_only` meets it exactly |

### 1. Play-by-play micro-signals: nothing to add

Sources (probed 2026-09-30): **play-by-play** covers 2020 through 2026 week 3, every regular-season game, refreshed in-season (2026: Tue 15:44Z). **Participation** covers
100% of regular-season scrimmage plays 2020-2025 with GSIS ids on the field, but has no per-player route (`route` is one value per play, blank on non-targets, and its vocabulary
changed in 2023), so route participation proper cannot be built for 2021-2025; pass-play snap share is the proxy. **It is published after the season** (2025 file last modified
2026-02-10; no 2026 file exists yet), so its columns are 100% NA for 2026 and cannot be served in-season: a declared backtest proxy, like weather. **FTN charting** covers 2022-2025
(99.7-100% of plays) but 2021 is a 404 and it has no player column: excluded per the plan, not built. NGS receiving is a 404 every year.

The new tables reproduce nflverse's own player stats (targets 100%, carries 99.98% or better, team target totals give `target_share` exactly), and every red-zone / inside-10 /
air-yards value is recomputed from the raw parquet with plain pandas in the tests. Coverage on played rows, share non-null: `pbp_usage` .87 (2021-24) / .85 (2025) / .60 (2026, three
weeks), `pbp_team` .96 / .95 / .74, `pbp_part` .92 / .90 / .09 (2026 has no feed), `eff` .45-.47 (null by design for opportunities a player never had). The flat model does not use any of it
better: the candidate `pbp` moves RMSE by -0.008 then +0.004, and play-by-play usage turns out to be a **substitute for the lagged xFP columns, not an addition** (drop the xFP lags:
+0.010 in 2024; drop them and add `pbp`: -0.000): ffopportunity's expected points are built from the same air yards and field position. Nothing from this experiment enters the feature set.

### 2. Two-stage: volume is the ceiling, and it is only modestly predictable

Stage one (LightGBM on the primary features, honest walk-forward predictions so stage two never trains on a fit that saw its own week) beats the trailing-3 mean of the same quantity by
5.9% (2025) / 5.8% (2024) on target MAE (RB/WR/TE), 5.4% / 6.1% on carries (RB and QB) and 9.2% / 9.6% on quarterback attempts, every interval excluding zero. In absolute terms targets
are off by 1.44 a game on a mean of 2.77 (R-squared .57 against .47 for the trailing mean), RB carries by 3.10 on 7.5, quarterback attempts by 7.5 on 26.6. That is worth a few percent of MAE,
not the 1.6-point RMSE gap to the same-week xFP oracle, which sees the volume the player actually got: nothing knowable before kickoff that was tried here recovers it (game script and in-game
injuries are the likely sources; that is inference, not measured). The end-to-end two-stage model is worse in
both years and worse than its matched control (trained on the same 2022+ window: +0.048 / +0.077); stage-one predictions as extra flat inputs do not replicate (2024: +0.020, significantly worse);
they do not replace lagged xFP; the efficiency priors alone do nothing.

### 3. Component models: a tie, and the league-scoring path works

Fourteen LightGBM models (targets, receptions, receiving yards and TDs, carries, rushing yards and TDs, pass attempts, completions, yards, TDs, interceptions, fumbles lost, a two-point /
special-teams bucket) compose to PPR exactly as nflverse defines it from actual components (max difference 7.1e-15 over all 33,423 played rows), and the composed prediction ties the flat model
(2025 -0.008 [-0.029, +0.013], 2024 +0.006 [-0.020, +0.034]; Spearman +0.008 then -0.000), by position too (QB -0.099 significant in 2025, +0.017 in 2024). The alternate scoring composes: the dynasty
league's TE premium (`bonus_rec_te` 0.5 on full PPR) scored against actual components is as good as or better than the naive route (PPR + 0.5 x recent catches) on tight ends in both years
(-0.017 / -0.013, spanning zero) and ignoring the premium costs +0.155 / +0.214 on tight ends (significant). Predictable per component (skill against a constant per position, 2025): targets .54,
receptions .47, carries .60, yards .40-.42, but receiving and rushing TDs .10 / .11, passing TDs .21, interceptions .04, fumbles ~0: touchdowns carry six points and almost no signal, the other half
of the ceiling. What linear composition cannot do: threshold bonuses (this league's 100/200-yard and 40/50-yard TD bonuses; its interception is -1, which composes trivially).

### 4. Cheap wins

**Ensemble.** The plain average of the three libraries is -0.011 [-0.020, -0.003] in 2025 (Spearman +0.004, pick accuracy +0.001, both significant) and -0.004 [-0.012, +0.005] in 2024; fitted weights
(2024 weights on 2025: .55 / .17 / .28; 2023 weights on 2024: .21 / .42 / .37) move between years and add nothing. A 0.2% RMSE gain, at 11 seconds of weekly refit instead of 1.
**Recalibration.** QB p10-p90 coverage .744 to .789 (2025) and .751 to .794 (2024) with 95% intervals that now contain .80, at +12% QB band width and an interval score that is flat (2025) or better
(2024); RB is untouched. The QB miss flipped sides between years (2025: below p10 14.2%, above p90 11.3%; 2024: 10.6% / 14.3%), so the width is the robust part and the tails are partly noise.
**Post-hoc lead (not in the adoption table, thought of after the results):** averaging the flat three-library average with the component composition gives -0.020 [-0.033, -0.007] in 2025 and -0.009
[-0.025, +0.006] in 2024, Spearman +0.008 / +0.002: the two designs make different errors. Worth carrying into the 2026 side-by-side; not adopted on this evidence.

### Phase 3 recommendation (updated; supersedes the one above where they differ)

* **Model design.** Two layers on the same inputs. (1) **PPR point estimate: the plain average of flat LightGBM, XGBoost and CatBoost** (parameters as in `tuned_params.json`, squared error, one model with
  position as a feature, trained on `y_played` rows, refit every week, about 11 seconds). (2) **A scoring layer of the 14 LightGBM component models** (primary features, own early-stopped tree counts) **composed under
  each league's linear scoring**: it ties the flat model on PPR, so it costs nothing in accuracy and it is what answers TE premium, half-PPR and any other linear scoring, which the flat model cannot. Run both in
  the 2026 side-by-side, and test the 50/50 average of the two for PPR (the post-hoc lead). Floors and ceilings: the flat LightGBM p10 / p50 / p90 models plus **`underage_only` recalibration** from trailing
  residuals (QB coverage to about 79%; use `qb_only` if RB/WR/TE bands must not move at all); means do not compose into quantiles, so a non-PPR league needs its own quantile fit on that scoring's actual points (a
  Phase 3 sub-task, one extra target per scoring). Threshold bonuses (100-yard, long-TD) need a distributional model and are out of scope; this league's interception at -1 is a one-line weight.
* **Features: exactly the Phase 2 set (71 columns), no additions.** Play-by-play usage, team pace, participation, the efficiency priors and stage-one volume all failed to replicate; the ADP blocker (no preseason
  snapshot) and the weather / `season` / college exclusions stand. The play-by-play and participation tables and the opt-in families stay in the repo, tested and off: the weekly pipeline does not need to fetch
  play-by-play at all. Load-bearing inputs to protect are unchanged: season-to-date xFP and points, snap share, the prior-season anchor, the Vegas implied total, teammate-out volume.
* **Expectation.** Accuracy at the Phase 2 level: pooled RMSE about 5.70 (from 5.71), Spearman about .66, pick accuracy about .775; per-position QB is still the weak spot (Spearman .48). Volume and touchdowns are at
  their ceiling for pre-game information, so the next gains are not in richer usage features; look at availability and news (injury timing, inactive lists), which the model treats as a separate layer.
* **Cadence and judging** are as in Phase 2 (weekly refit, preseason retune, Tuesday feature refresh); score the average, the components and Sleeper on the same 2026 weeks with `backtest.score` before any recipe prefers them.

## Phase 3 — ship (work order, 2026-09-30)

Ship the Phase 2.5 recommendation into the weekly pipeline, side by side with
`E_pts`, judged live before any recipe prefers it. `ff/` and `pipeline.yml` may
now change; everything else in the ground rules still holds (gate, seeds, no
committed binaries or matrices, no secrets).

1. **Serving path** (`python -m model.serve`): build prediction rows for the
   UPCOMING week — the spine from current pre-kickoff information and the
   schedule's kickoffs, features through the same `as_of_join` gate (serving
   must reuse the training feature code, not re-implement it), then refit and
   predict: the 3-library average (PPR point estimate), the 14 component
   models composed under each configured league's linear scoring
   (`leagues/*/league.json`; threshold bonuses ignored and documented), and
   p10/p50/p90 with `underage_only` recalibration. Tests: serving rows carry
   the training schema, contain nothing stamped at or after their kickoff, and
   a serve for a COMPLETED week reproduces backtest predictions for that week.
2. **Committed output**: `data/model_pts.csv`, appended per week via
   `build.replace_partition` on (season, week) — one row per predicted player:
   ids, name, position, team, `pts_model` (PPR average), `pts_model_components`,
   per-league composed points (one column per league slug), p10/p50/p90, and
   `sleeper_proj` captured at prediction time (the side-by-side needs it frozen).
   History accumulates; nothing is overwritten after its week completes.
3. **Pipeline step**: a step in `pipeline.yml` after the existing run — install
   `model/requirements.txt` (pip cache), `python -m model.serve`, commit. It
   must NEVER break the pipeline: any model failure is a logged WARN
   (`model.serve` check in the run log) and the pipeline's own outputs are
   untouched. Budget: a few minutes is fine; cache pip.
4. **Surfacing**: `roster.csv` (all leagues) gains `E_pts_model`, `p10`, `p90`
   joined from `data/model_pts.csv` — that league's composed scoring, not raw
   PPR, for the non-PPR leagues; `ff.ask live` and `lineup` print them beside
   `E_pts`. `E_pts` stays authoritative for every recipe and report during the
   probation period.
5. **Live scoreboard** (`python -m model.scoreboard`, run in the same step):
   after each completed week, score `pts_model`, the components, the 50/50
   blend of the two, frozen `sleeper_proj`, and `E_pts` (from the committed
   roster history) against actual points from `data/lineups_played.csv` —
   Spearman within position-week and pick accuracy, appended to
   `data/model_eval.csv` and summarized in `model/reports/live_scoreboard.md`.
   Adoption gate, written down now: the model earns recipe preference only
   when it beats Sleeper's projection on pick accuracy over at least 6
   completed 2026 weeks.
6. **Docs**: POLICY.md gets a short "Model column" section (what
   `E_pts_model` is, probation rule, the adoption gate); `skills/fantasy/SKILL.md`
   mentions the columns and the rule in a few lines, everything else verbatim;
   README's data-file list updated. This file stays until the adoption gate
   resolves; "delete at the end" now means then.

## Phase 3 findings

Done 2026-09-30 (branch `claude/eager-goldberg-zmevs9`, not merged). Shipped: `model/serve.py` (`python -m model.serve`), `model/serving_config.json`
(the frozen design: the 39 pruned columns, the 14 component tree counts, the recalibration mode), `model/leaguescore.py`, `model/scoreboard.py`
(`python -m model.scoreboard`), `model/phase3.py` (validation of the shipped design), `model/retro.py` (a replay of weeks 1-3), `ff/modelcols.py`
(the display join), changes to `model/build_features.py` (rows and frames for any list of games, forked workers), `ff/ask.py` and `ff/live.py`
(print only), `.github/workflows/pipeline.yml`, `POLICY.md`, `skills/fantasy/SKILL.md`, `README.md`, and tests: 70 new model tests (242 in all, all
green under `-W error::DeprecationWarning`) and 19 new pipeline tests (39 in all). The pipeline's own code path (`ff/run_weekly.py`, `build`, `leagues`, `verify`, ...)
is untouched, so with the model step removed its outputs are what they were; `git diff 91eb9e8 -- ff` is `ask.py`, `live.py` and the new `modelcols.py` only.

### What ships, per work-order item

1. **Serving path.** `serve` builds the upcoming week's rows with `build_features.build_games` (the matrix builder's own `spine_for` and `build_features`
   through `as_of_join`, each game at its real kickoff), then calls `backtest.walk_forward(weeks=(W,))`, the function the backtest uses, for every model,
   and `ensemble.recalibrate` for the bands. Nothing is re-implemented for serving. The tests that prove it, on real data: serving rows for completed weeks
   (2026 wk3, 2025 wk9, 2024 wk14) equal the Phase 2.5 matrix's rows **bit for bit** on every model input; the serving history builder reproduces the matrix's
   46,669 rows of 2021-2025 bit for bit and in the same order (the order matters: a bagged fit sees rows in that order); a serve of 2026 week 3 reproduces
   the backtest's predictions **exactly** (array equality, not closeness) for the three flat libraries and their average, all 14 component models and PPR
   composed from them, the p10/p50/p90 models and their recalibration, and both non-PPR leagues' own bands; every gate call of sampled upcoming-week rows
   reads only rows stamped before the row's kickoff, and a truncation audit of upcoming-week rows from a store cut before each kickoff changes nothing.
2. **`data/model_pts.csv`**, via `ff.build.replace_partition` on (season, week) (a temp copy renamed over the file). One row per predicted player: `gsis_id`, name,
   position, team, opponent, game and kickoff, `report_status`, `pts_model` (PPR average), `pts_model_components`, `pts_<slug>` per league, `p10/p50/p90`,
   `p10_<slug>/p90_<slug>` for the two non-PPR leagues, `sleeper_proj` (from `data/projections.csv` already on disk: no new network call), and the comparators
   the scoreboard needs frozen (`sleeper_proj_<slug>`, `e_pts_<slug>`), plus `served_at`. **Frozen per game**: a run rewrites only games that have not kicked off;
   rows of a game under way are kept exactly as written, and a week whose games have all kicked off is never touched. So the Wednesday row of a Thursday-night
   player is the one the scoreboard judges, while Sunday rows are refreshed by the Friday run (final injury reports, lines). `python -m model.serve --columns`
   documents every column, including the ignored threshold bonuses. ID columns are `gsis_id` only: `replace_partition` re-reads with default dtypes and would turn a
   gappy `sleeper_id` column into `8183.0`.
3. **Pipeline step.** After `Run pipeline`: a second `setup-python` (3.12, pip cache keyed on `model/requirements.txt`), install (5 min cap, continue-on-error), `actions/cache`
   on `model/cache` (keyed on content and ISO week after review, below), then `python -m model.serve && python -m model.scoreboard` (10 min cap, continue-on-error), committed by the usual commit step (which now also
   adds `model/reports/live_scoreboard.md`). Both commands catch every exception, append a WARN row through the pipeline's own `verify.log_run` (check `model.serve` or
   `model.scoreboard`, the `week` of the run they belong to) and exit 0 having written nothing: all roster texts are rendered before `model_pts.csv` is replaced, and every file is
   replaced atomically. Tested: a raised error, a missing library, bad data inside `run`, and a log that cannot be written all end in exit 0, one WARN row, byte-identical
   files. The job timeout went from 20 to 30 minutes so the model's caps can never cost the commit. `model/requirements.txt` pins lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10
   exactly (the versions every number here came from).
4. **Surfacing.** `roster.csv` of all four leagues gains `E_pts_model`, `p10`, `p90` appended by the model step with every existing byte of every existing column unchanged (tested on the real
   files; the csv module keeps each field's text, pandas would rewrite `1` as `1.0`). `ff.ask live` and `lineup` print them beside `E_pts`, joined from `data/model_pts.csv` for the frame's own
   week, with one header line that E_pts stays authoritative; with no model file they print what they always printed. The lineup picks the same players in the same slots when the model disagrees about
   every one of them (tested). `E_pts_model` is the **league-composed components** (the work order's "that league's composed scoring"); the flat 3-library average is `pts_model`, scored beside it.
5. **Scoreboard.** Per completed week and league, on the identical rows of played rostered QB/RB/WR/TE, against the platform's own points: `model_components` (E_pts_model), `model_flat` (the
   average moved to the league's scoring by the components' scoring difference: identical to `pts_model` in a PPR league), `model_blend` (their 50/50), `sleeper` and `e_pts`, in that league's
   scoring; Spearman within position-week-league and pick accuracy, into `data/model_eval.csv` (`replace_partition`, additive `credit`/`pairs`) and `model/reports/live_scoreboard.md`, which carries the gate
   sentence verbatim. The gate is evaluated on `model_components`, the number the recipes show; flat and blend are reported beside it. **E_pts history: frozen going forward, not mined from git.** The
   depth-1 checkout the workflow uses has no history, `roster.csv` alone holds 15-31 players, and Sleeper's league-scoring projection is overwritten every run, so `serve` freezes `e_pts_<slug>` and
   `sleeper_proj_<slug>` next to the prediction (from the league files the pipeline step just wrote). Which weeks have which comparators: **weeks 4 onward** have the model (all variants), Sleeper (PPR and
   per league) and E_pts, all frozen before each game's kickoff: week 4 is scored by the first run after Monday night (Tue Oct 6), and the gate (6 weeks) cannot be met before week 9 completes.
   **Weeks 1-3 have no frozen model rows** (it did not exist); `python3.12 -m model.retro` replays them (see below) and is not the gate. Today `live_scoreboard.md` says "not started" and `data/model_eval.csv` does not exist yet.
6. **Docs.** `POLICY.md` "Model column" section; `skills/fantasy/SKILL.md` one definition bullet, everything else verbatim; `README.md` data-file rows, a "Model columns" note, the model test command.

### Decisions and the evidence behind them

* **The shipped design (71 inputs) measured end to end** (`python3.12 -m model.phase3 shipped`, the Phase 2 head-to-head rows, 6,139 in 2025 / 6,025 in 2024, week-blocked bootstrap). Phase 2.5 had measured the average,
  the components and the recalibration on the 114-input primary, not on these 71. RMSE / Spearman / pick accuracy, 2025 (2024 in brackets): **3-library average 5.710 / .657 / .774** (5.779 / .663 / .774);
  its LightGBM alone 5.719 / .655 / .773 (5.792 / .660 / .773); Phase 2 primary LightGBM 5.711 / .655 / .773 (5.771 / .663 / .775); primary average 5.699 / .659 / .774 (5.767 / .663 / .775); components composed to PPR
  5.733 / .656 / .774 (5.784 / .665 / .774); 50/50 blend 5.711 / .659 / .775 (5.771 / .665 / .774); trailing-3 6.432 / .602 / .748 (6.407 / .612 / .750); same-week xFP oracle 4.073 / .866 / .876. What that says:
  (a) the **average beats its own LightGBM in both years, both significant** (dRMSE -0.009 [-0.016, -0.001], -0.013 [-0.018, -0.008]) and beats trailing-3 by 0.7 RMSE; (b) **pruning to 71 inputs costs about 0.011 RMSE in the average** against the 114-input
  average (2025 [-0.024, +0.002], 2024 [-0.024, -0.002]): the price of shipping without ADP, as Phase 2 found for one model; (c) the **components tie the average on ranking** (dSpearman -0.002 [-0.007, +0.004] / +0.002, dPick -0.000 / -0.001,
  intervals span zero) but are **worse on RMSE in 2025** (+0.023 [+0.006, +0.040]; 2024 +0.005, spans zero), a little more than Phase 2.5's tie against LightGBM alone suggested; this is the cost of `E_pts_model` being the
  components, paid for the league-scoring path, and is what the scoreboard's `model_flat` column is there to watch; (d) the post-hoc blend is the best ranker in both years by a hair (dSpearman +0.002 [-0.000, +0.005] / +0.003, dPick +0.001 [+0.000, +0.001] / +0.000), still not adopted.
  QB stays the weak spot (Spearman .48 in 2025, .52 in 2024). 2024 carries the usual mild in-sample edge (tree counts were early-stopped on it).
* **Bands.** PPR p10-p90 coverage, raw to recalibrated: QB .735 to .780 (2025) and .738 to .796 (2024); all positions .779 to .790 and .766 to .795. `underage_only` also widened TE (2025 .790 to .816, 2024 .741 to .798) and WR in 2024 (.775 to .797),
  the flagged Phase 2.5 behaviour; RB untouched; band width +0.2 to +0.35 points; interval score flat; p10 <= p50 <= p90 on 100% after recalibration.
* **A non-PPR league needs its own quantile models, and the shortcut measurably fails.** Moving the PPR band by the change in the mean covered only 68% of the dynasty league's points in 2025 (tight ends 47%; 2024: 69%, TE 44%) and 88% of the
  half-PPR league's (2024: 88%): the premium is earned on the same catches that make the good games, so the low tail moves less than the mean. `serve` therefore fits p10/p50/p90 LightGBMs on each non-PPR league's composed
  actual points (`y_pts_<slug>` from the component labels; same tuned shape and tree counts as the PPR quantiles) and recalibrates them the same way: recalibrated coverage of that league's own points .790 / .797 (dynasty 2025 / 2024) and .785 / .804 (IDP/half-PPR),
  QB .794 / .797 and .789 / .796, TE .793 / .805 and .788 / .809. A PPR league uses the PPR band; a non-PPR league without its own columns would get blanks, never a shifted guess. Composed points rank that league's points as well as the flat average moved to its scoring
  (dynasty 2025 Spearman .656 / .660, pick .774 / .775; IDP .650 / .651, .769 / .769).
* **Who gets no row (how Out players are handled).** The model predicts points if he plays; availability belongs to the status layer. A row is withheld when Sleeper's `injury_status` is one of `ff.status.OUT_STATES` (Out, IR, PUP, Sus: Sleeper also
  uses Out for a coach's-decision inactive), when the nflverse report **for the served week** says Out, or when his nflverse roster status is cut, retired, exempt or reserve. Doubtful and Questionable players keep a row, and `report_status`
  carries this week's designation (the shipped model does not read its own designation: the two injury status columns are in the pruned 39). The first Actions run taught one rule: the injury feature is "the newest report this season", so on a Wednesday it can
  be last week's final Out; that withheld Jayden Daniels and showed Puka Nacua as Doubtful while Sleeper, refreshed that morning, said Questionable for both. Only a report issued for the served week withholds now (tested), and `report_status` is blank until the week has one.
  A withheld player is not scored by the scoreboard either (no row, no comparison), and the scoreboard only scores players who played.
* **The composition.** `leaguescore` maps each `league.json` scoring dict onto the 14 labels. Four leagues: two are exactly PPR (Gooma's, the ESPN league), the dynasty league is PPR with a -1 interception and +0.5 per TE catch, the IDP league is half PPR. Ignored and named:
  the dynasty league's yardage bonuses (100/200 rush and rec, 300/400 pass) and 40+/50+ yard TD bonuses (`bonus_*_yd_*`, `*_td_40p`, `*_td_50p`, 12 keys); kicking, defence and IDP never score on skill positions. Cross-checked against `ff.scoring.score_frame` on random stat lines below every threshold (equal to the cent, all four leagues).
  One approximation: the misc bucket (2-point conversions and special-teams TDs) is one model composed with one weight (the two-point value over 2), so the ESPN league, which has no special-teams TD key, credits those at 6 anyway (about 0.01 points a game).
* **Serve time is not backtest time in two small ways, both documented and neither fixable here.** Lines are closing lines in training but whatever the schedule shows on Tuesday-Friday at serve time (Phase 0 finding 8: nflverse keeps no opens); and Wednesday's injury table holds only
  the reports issued so far (the model's injury inputs are few, `inj_weeks_since_report` only, so this matters little). The Friday run, before Sunday, is the best-informed one and is the one frozen for every Sunday and Monday game.
* **A look-back at weeks 1-3** (`python3.12 -m model.retro`, needs the full git history; a replay, NOT the gate: replayed model rows with the final injury report and closing lines, E_pts and Sleeper's league-scoring projection read per game from the newest commit before its kickoff; the ESPN league has no snapshot before week 2, so it is weeks 2-3 only).
  Pooled pick accuracy on identical rows (2,005 player-league-weeks, 54,492 pairs): **`model_components` .663, `model_flat` .664, `model_blend` .664, Sleeper .685, E_pts .671**; Spearman .363 / .358 / .366 / .398 / .327. Model minus Sleeper -0.022, week-blocked interval [-0.041, -0.003]; by week .639 / .670 / .679 against Sleeper's
  .680 / .693 / .683 (the gap closes as in-season form accrues; week 1 is the model without ADP and with no current-season lags, a week the Phase 2 backtest never scored). By position the model trails Sleeper at RB (.730 vs .753), WR (.648 vs .672) and TE (.563 vs .586), and ties at QB (.565 vs .569); it beats E_pts at QB (.565 vs .521) and TE (.563 vs .555) and trails it at RB (.730 vs .737) and WR (.648 vs .666), .008 behind overall. The absolute levels are far below Phase 2's .774 because the
  scoreboard's population is rostered players who played, whose pairs are harder; on the model's own population (every player with a stats row) the same three weeks give pick accuracy .760 (trailing-3 .714), in line with 2025. So: no sign the replay or the scoreboard is broken, an early sign the model does not yet beat Sleeper on the players James actually owns, and three weeks prove nothing either way.
* **Runtime.** Cold runner (first run, 4 vCPUs): install 31 s plus 5 min 47 s of serve (raw data 23 s, 2021-2025 history 106 s with 4 forked workers, three quantile histories 50 s each, fits 207 s in all); warm (run 31): install 26 s plus **71 s**. Steady state per run is the 2026 weeks so far plus one set of fits, growing
  by about 3 s per completed week per quantile target. The two pip caches are ~700 MB each (xgboost pulls `nvidia-nccl-cu13`); harmless under the 10 GB quota.

### Validation, in the order asked

(a) All model tests (242, 10 min) and the pipeline's (39) pass under python3.12. (b) `python3.12 -m model.serve --dry-run` and the real write path on a copy of the repo data: 59 s warm / 181 s cold locally, idempotent (a second run changes nothing but `served_at`).
(c) Pushed; `pipeline.yml` dispatched on the branch three times: run **36759198077** (cold caches, success), **36760150982** (warm, success) and **36760474080** (final code, success; the `Fail the job` step skipped). `logs/runs.csv` gained only the expected `status.practice: 0% have practice reports` WARN per run, no FAIL, no `model.*` row.
(d) One iteration: run 1 exposed the stale-report rule above; fixed in `418862e`, verified in run 3. The coordinator's review then changed seven things (next section); re-validated by run **36765150760** (success).

**Week-4 serve, final run:** 512 rows (QB 86, RB 113, TE 124, WR 189) for 16 games over 7 kickoffs, 95 withheld (status layer 95; nflverse has no week-4 report yet on a Wednesday). `sleeper_proj` (PPR) on 393 rows, `e_pts_<slug>` and `sleeper_proj_<slug>` on 509.
Rostered skill players with a row: Gooma's 190 of 209 (18 withheld as unavailable, 1 not in the week's spine), ESPN league 132 of 143 (11, 0), dynasty 214 of 242 (27, 1), IDP 222 of 253 (30, 1); James's own `roster.csv`: 16 of 18, 14 of 15, 28 of 31, 21 of 22 have `E_pts_model`. Spearman of `pts_model` against Sleeper's projection .947, against `pts_model_components` .986; p10 <= p50 <= p90 on every row and `pts_model` inside [p10, p90] on 99%.
Three spot-checks against raw lines: **A.J. Brown** (NE WR, Sleeper IR, roster status RES), **De'Von Achane** (MIA RB, IR) and **Baker Mayfield** (TB QB, Sleeper Out): no row, status layer; the model would only have said "points if he plays". **Josh Allen** (BUF at NE): last three games 17.5 / 40.8 / 35.7 PPR (raw nflverse lines: 29 / 31 / 26 attempts, 334 / 248 / 204 passing yards), season xFP 25.5 a game, implied team total 27.5,
model 22.3 (components 23.0, p10-p90 8.8-33.6) against Sleeper 23.1 and E_pts 25.7: the model pulls a 31-point average toward the position mean, as it should. **Marcus Mariota** (WAS QB, his starter out): 8.7 then 20.4 points (31 attempts, all snaps, in week 3), implied total 22.0 as a 3.5-point underdog, model 13.2 (components 13.7, p10-p90 2.8-23.2) against Sleeper 16.7 and E_pts 14.8.

### Review fixes (coordinator review, same day; all tests green, re-validated by run 36765150760)

Must fix:
1. **roster.csv columns came from the newly predicted rows only.** On a Friday run `out` has no Thursday-night player (his game has kicked off), so every Thursday player lost `E_pts_model`/`p10`/`p90` in
   `roster.csv` although `model_pts.csv` kept his frozen row. `write_outputs` and the serve summary now build from `merged_week` (frozen rows plus new), the week as `model_pts.csv` will hold it. Regression test
   `FridayServe` (a Wednesday serve, then a Friday serve with Thursday's game under way: Thursday's columns persist, Sunday's refresh, a player with no row anywhere is blank); it fails against the old code.
   Verified on the real run: every player in each league's `roster.csv` who has a `model_pts.csv` row has the columns filled (16/18, 14/15, 28/31, 21/22; the rest have no row).
2. **`week_of` took the season mode and the week mode separately**, which at a season boundary can name a pair no row holds ((2025, 18) x3 beside (2026, 1) and (2026, 2) gave (2026, 18)). It takes the most common
   (season, week) pair now, the later on a tie. Tests: that exact frame, a tie, and `attach` across the boundary; both fail against `mode()`.
3. **The frozen `inputs` in `serving_config.json` were never read.** `backtest.walk_forward` silently drops a column that turns dead in the data, so a data shift would have served a model nobody validated.
   `predict_week` now checks the live encoded inputs against `cfg["inputs"]` before fitting anything and raises `ServingSpecMismatch` naming what is missing or unexpected; the step turns it into the usual WARN row with nothing written.
   Tests: live set equals frozen, a column gone all-NA is refused, a changed frozen list is refused, and no model is fitted when it is refused.

Cheap fixes:
4. `model_fetch.yml`: the ADP, CFBD and step-summary steps are `continue-on-error`, so a raise in one cannot discard the CSVs another wrote before the commit step (tested as text).
5. `scoreboard`: the by-week and coverage tables group on (season, week); a 2026 week 4 and a 2027 week 4 are two rows (tested, including the gate's week count).
6. `pipeline.yml`: the model cache is keyed on `hashFiles` of the model requirements and feature code plus the ISO week (a one-line step computes it), with the prefix restore-key kept. Observed: restored by prefix from
   the previous entry, saved once under the new key (34 MB), and the ~700 MB pip caches now report "cache hit on the primary key, not saving". One consequence to know: in a new season the raw files of the season just ended are as of the
   last run that refreshed them (only the current season's files are re-downloaded).
7. `tests/test_modelcols.py`'s docstring no longer describes the abandoned shifted-band shortcut.

Optional, done: the scoreboard builds each league-week's comparators once; `serve.quantile_walk` calls `backtest.walk_forward_quantiles`, which takes a `target` now, instead of duplicating it.
Optional, skipped: `labels.attach_labels` still materialises its dictionaries per call. It is matrix-building code (changing it changes the code hash that keys the history cache and needs a bit-identity proof against the 46,669-row matrix),
and the cost is seconds in a cold build that happens once per code change.

### What is not done, and what to watch

* **No merge.** The first scoreboard row arrives with the Tuesday Oct 6 run (week 4); the gate needs week 9. Nothing in `E_pts`, the recipes, the sorts or the reports reads the model columns.
* **Retune and refreeze each preseason** (`python -m model.serve --freeze` rewrites the component tree counts; the prune list in `serving_config.json` is from 2024 SHAP and was frozen, not regenerated). An August ADP snapshot for 2027 would let ADP back in; week 1 of this season (the model's weakest replayed week) is where it would have mattered.
* **Week-4 frozen rows are Wednesday's until the Friday run refreshes Sunday/Monday games**; Thursday's game keeps its Wednesday row (the Friday run starts after its kickoff). `model.serve` warns (WARN row) when last week's stats are not fully published, when no Sleeper projections exist for the week, or when a league's files belong to another week.
* The recommendation to test the 50/50 blend stays open; the scoreboard reports it beside the two designs. If `model_flat` keeps beating `model_components` on RMSE-like grounds while tying on ranking, switching `E_pts_model` to the flat average (plus the components' scoring difference) is a one-line change in `serve.assemble`/`ff.modelcols` and a documented decision, not a silent one.
* The scoreboard scores skill positions only. The platform's points for those players include what the composition cannot express (the dynasty league's yardage and long-TD bonuses), which the model cannot predict and which adds noise to that league's comparison equally for every comparator.
