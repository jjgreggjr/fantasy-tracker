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

(appended by the Phase 0 session)
