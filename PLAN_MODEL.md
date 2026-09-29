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
