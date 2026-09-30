# Phase 2.5: volume-first experiments

Four experiments on the Phase 2 primary (LightGBM, squared error, one model with position as a feature, tuned parameters,
weekly walk-forward). Every result is scored on the same head-to-head rows as Phase 2 (played rows with an earlier game this season: 2025 has
6,139) and **replicated on the 2024 walk-forward** with the same fixed seeds and the same week-blocked
bootstrap. Adoption rule (PLAN_MODEL.md): an experiment ships into the Phase 3 recommendation only if it beats the primary on 2025 AND replicates directionally
on 2024 (pooled dRMSE below zero in both years), or, for component models only, ties within noise (neither year's 95% interval lies entirely above zero).
"Significant" below means the 2025 interval excludes zero; the rule itself is about direction in both years, so a direction-only pass is called out.

All numbers are a pure function of `model/cache/features.parquet` (rebuilt for this phase: the 142 Phase 2 columns are bit-identical, 85 columns
added), `model/tuned_params.json` and fixed seeds; a rerun reproduces every number. The primary recomputed from the rebuilt matrix equals Phase 2's
predictions exactly (max |difference| 0.0e+00 over LightGBM, XGBoost and
CatBoost 2025 and LightGBM 2024). Hyper-parameters were tuned on 2024 (validation), and each variant's tree count is picked by early stopping on 2024 after
training on 2021-2023, so 2024 carries a mild, symmetric in-sample edge for every variant; 2025 is never seen before its walk-forward scores it.

## Summary

| experiment | variant | 2025 result | 2024 result | rule | verdict |
|---|---|---|---|---|---|
| 1 play-by-play | pbp_usage | dRMSE -0.008 [-0.021, +0.003] | dRMSE +0.009 [-0.004, +0.022] | beat + replicate | does not ship |
| 1 play-by-play | pbp | dRMSE -0.008 [-0.024, +0.007] | dRMSE +0.004 [-0.009, +0.016] | beat + replicate | does not ship |
| 1 play-by-play | pbp_part | dRMSE -0.010 [-0.028, +0.006] | dRMSE -0.005 [-0.017, +0.006] | beat + replicate | passes the accuracy rule on direction only (both intervals span zero); NOT ADOPTED: the participation feed is not published in-season, so it is all-NA for 2026 (train/serve skew, like weather) |
| 1 play-by-play | pbp_team | dRMSE +0.000 [-0.011, +0.012] | dRMSE +0.009 [-0.008, +0.024] | beat + replicate | does not ship |
| 1 play-by-play | part_only | dRMSE -0.001 [-0.012, +0.009] | dRMSE +0.001 [-0.010, +0.013] | beat + replicate | does not ship |
| 2 two-stage | two_stage | dRMSE +0.067 [+0.031, +0.101] | dRMSE +0.088 [+0.047, +0.127] | beat + replicate | does not ship |
| 2 two-stage | flat_plus_s1 | dRMSE -0.003 [-0.013, +0.008] | dRMSE +0.020 [+0.007, +0.033] | beat + replicate | does not ship |
| 2 two-stage | flat_plus_s1_eff | dRMSE -0.010 [-0.021, +0.001] | dRMSE +0.009 [-0.004, +0.020] | beat + replicate | does not ship |
| 2 two-stage | flat_plus_eff | dRMSE -0.004 [-0.015, +0.006] | dRMSE +0.006 [-0.009, +0.021] | beat + replicate | does not ship |
| 2 two-stage | flat_no_xfp_plus_s1 | dRMSE -0.007 [-0.021, +0.007] | dRMSE +0.003 [-0.013, +0.020] | beat + replicate | does not ship |
| 3 components | components | dRMSE -0.008 [-0.029, +0.013] | dRMSE +0.006 [-0.020, +0.034] | tie ships | SHIPS (tie within noise) |
| 3 components | components_eff | dRMSE +0.006 [-0.022, +0.034] | dRMSE +0.003 [-0.030, +0.036] | tie ships | SHIPS (tie within noise); not preferred: 18 extra features for no gain |
| 4a ensemble | average | dRMSE -0.011 [-0.020, -0.003] | dRMSE -0.004 [-0.012, +0.005] | beat + replicate | SHIPS (significant in 2025) |
| 4a ensemble | weighted | dRMSE -0.009 [-0.015, -0.004] | dRMSE -0.002 [-0.013, +0.008] | beat + replicate | SHIPS (significant in 2025) |
| 4b quantile recalibration | underage_only | QB coverage 0.744 to 0.789 | QB coverage 0.751 to 0.794 | QB within 2 pts of 80%; RB/WR/TE width change <= +2% | QB target met in both years; caps missed: widest RB/WR/TE change +1.2% (TE) in 2025, +2.2% (WR) in 2024 against the +2% cap. SHIPS as `underage_only` with that miss flagged (`qb_only` meets the cap by construction) |

## Experiment 1: play-by-play micro-signals

Sources probed from the sandbox on 2026-09-30 (`model/cache/nflverse/`, nflverse-data release assets). `Last modified` is the last re-upload of the file, not when it first appeared.

| season | PBP REG games | PBP scrimmage plays | PBP last modified | participation: scrimmage plays with players on field | participation: scrimmage plays with a route value (one per play) | participation last modified | FTN: scrimmage plays matched | FTN last modified |
|---|---|---|---|---|---|---|---|---|
| 2020 | 256 | 32575 | not probed | 1.000 | 0.531 | not probed |  | not probed |
| 2021 | 272 | 34139 | 2026-01-08 | 1.000 | 0.528 | 2023-12-19 |  | 404: not published |
| 2022 | 271 | 33770 | 2026-02-12 | 1.000 | 0.512 | 2023-12-19 | 0.997 | 2024-10-10 |
| 2023 | 272 | 33957 | 2026-02-12 | 1.000 | 0.523 | 2025-09-04 | 1.000 | 2024-09-06 |
| 2024 | 272 | 33470 | 2026-08-13 | 1.000 | 0.516 | 2025-09-04 | 1.000 | 2025-09-01 |
| 2025 | 272 | 32941 | 2026-08-13 | 1.000 | 0.516 | 2026-02-10 | 1.000 | 2026-09-23 |
| 2026 | 48 | 5829 | 2026-09-29 (weeks 1-3) |  |  | 404: not published | 1.000 | 2026-09-29 |

* **Play-by-play** covers 2020 through 2026 week 3 for every regular-season game and is refreshed in-season (2026: Tue 2026-09-29 15:44Z,
  ahead of the pipeline run). The new tables reproduce nflverse's own player stats: targets match `stats_player_week` on 100% of rows in
  every season, carries on 99.98% or better (kneels and scrambles are carries in both), and the team target totals give nflverse's
  `target_share` exactly. `pbp_usage`, `pbp_team` and every red-zone / inside-10 / air-yards value are also recomputed from the raw parquet
  with plain pandas in the tests.
* **Participation** covers 100% of regular-season scrimmage plays 2020-2025 (91% of all plays in 2021-22, 100% 2023+; the gap is
  non-scrimmage plays) with GSIS ids on the field, so a share of the team's dropbacks / runs / red-zone plays a player was on the field for is
  consistent across the whole backtest. It does **not** contain a per-player route: `route` is one value per play (the targeted route, blank on
  non-targets) and its vocabulary changed in 2023 ("HITCH" to "HITCH/CURL"), so route participation proper cannot be built for 2021-2025. Pass-play
  snap share (`part_pass_snap_share_*`) is the closest free proxy (blockers count). **The feed is published after the season**: the 2025 file was
  last modified 2026-02-10 and no 2026 file exists as of 2026-09-30, so these columns are 100% NA for 2026 and cannot be served in-season. It is
  stamped at kickoff for the backtest like the other game tables, with the store's declared-proxy flag (`RawTable.proxy`), and reported separately.
* **FTN charting** covers 2022-2025 (99.7-100% of scrimmage plays) but 2021 is a 404. Per the plan a source that does not cover the whole backtest
  is excluded: it would be all-NA for a full training year, and its fields are play-level flags (play action, motion, screen, RPO, drop, contested
  catch, blitzers) with no player column, so they are not volume signals either. Not built.
* **NGS** receiving files (`nextgen_stats/ngs_{y}_receiving`) are 404 for every year.

### What was built

Three raw tables, each row stamped with its source game's kickoff and read at kickoff - 4h like every result table:
`pbp_usage` (player-game targets, red-zone and inside-10 targets, air yards, carries, red-zone and inside-10 carries), `pbp_team` (team-game
totals, the denominators of every share, plus plays, dropbacks, neutral-situation plays) and `pbp_part` (plays on the field, participation
feed). Four opt-in feature families through the gate: `pbp_usage` (32 columns: red-zone and inside-10 target/carry shares, air-yards share, aDOT,
WOPR, red-zone and inside-10 opportunities per game, over the last game / last three / season to date, plus last season's five main values),
`pbp_team` (10: the team's pace, dropback rate and neutral dropback rate, and the plays and dropback rate its opponent's defense has faced),
`pbp_part` (10: share of the team's dropbacks, non-dropback runs and red-zone plays on the field), `eff` (18, used by experiments 2 and 3).
Shares are ratios of sums over the window; nothing reads the target game. Coverage on played rows (share non-null):

| share of played rows that are non-null | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|
| pbp_usage: mean non-null over 32 columns | 0.873 | 0.872 | 0.873 | 0.875 | 0.845 | 0.597 |
| pbp_team: mean non-null over 10 columns | 0.963 | 0.966 | 0.965 | 0.964 | 0.953 | 0.738 |
| pbp_part: mean non-null over 10 columns | 0.923 | 0.926 | 0.922 | 0.926 | 0.896 | 0.085 |
| eff: mean non-null over 18 columns | 0.448 | 0.470 | 0.473 | 0.470 | 0.460 | 0.363 |
| pbp_usage: pbp_wopr_l1 | 0.937 | 0.940 | 0.939 | 0.940 | 0.909 | 0.609 |
| pbp_usage: pbp_wopr_std | 0.937 | 0.940 | 0.939 | 0.940 | 0.909 | 0.609 |
| pbp_usage: pbp_prev_wopr | 0.814 | 0.816 | 0.800 | 0.817 | 0.799 | 0.847 |
| pbp_team: team_pass_rate_t3 | 0.954 | 0.957 | 0.956 | 0.956 | 0.942 | 0.672 |
| pbp_part: part_pass_snap_share_l1 | 0.937 | 0.940 | 0.939 | 0.940 | 0.909 | 0.000 |
| pbp_part: part_prev_pass_snap_share | 0.814 | 0.816 | 0.800 | 0.817 | 0.799 | 0.847 |
| eff: eff_rec_ypt_std | 0.760 | 0.778 | 0.776 | 0.754 | 0.723 | 0.420 |

Null rates track the lag family (week 1 has no current-season window, and a player with no history has none): 2021-2025 about 13-15%
for `pbp_usage`, 4-5% for the team columns, 2026 has three weeks. `pbp_part` is 91.5% null in 2026 (no feed). `eff` is null by design for
opportunities a player never had (a receiver has no passing rate).

### Results: the flat model with the new inputs

Pre-declared: the candidate for Phase 3 is **`pbp`** (usage + team: everything servable in-season). `pbp_part` adds the participation columns (backtest-only) and is reported as the upper bound; the rest are diagnostics. Same tuned LightGBM shape as the primary, own tree count by the early-stopping protocol. `no_xfp` drops the five lagged-xFP columns and `no_xfp_pbp` drops them and adds `pbp` (does play-by-play usage substitute for the expected-points lags it overlaps?).

**2025 walk-forward** (6,139 head-to-head rows, ALL positions)

| run | RMSE | MAE | Spearman | Pick acc |
|---|---|---|---|---|
| primary | 5.711 | 4.019 | 0.655 | 0.773 |
| pbp_usage | 5.702 | 4.012 | 0.657 | 0.774 |
| pbp | 5.703 | 4.016 | 0.656 | 0.773 |
| pbp_team | 5.711 | 4.023 | 0.655 | 0.773 |
| pbp_part | 5.700 | 4.013 | 0.657 | 0.774 |
| part_only | 5.709 | 4.016 | 0.655 | 0.773 |
| no_xfp | 5.707 | 4.020 | 0.657 | 0.774 |
| no_xfp_pbp | 5.704 | 4.011 | 0.657 | 0.774 |
| trailing3 | 6.432 | 4.314 | 0.602 | 0.748 |
| xfp_oracle | 4.073 | 2.503 | 0.866 | 0.876 |

| run minus base | dRMSE | dMAE | dSpearman | dPick |
|---|---|---|---|---|
| pbp_usage | -0.008 [-0.021, +0.003] | -0.007 [-0.019, +0.002] | +0.0024 [-0.0002, +0.0051] | +0.0007 [-0.0001, +0.0017] |
| pbp | -0.008 [-0.024, +0.007] | -0.003 [-0.015, +0.007] | +0.0006 [-0.0023, +0.0036] | +0.0005 [-0.0003, +0.0013] |
| pbp_part | -0.010 [-0.028, +0.006] | -0.005 [-0.020, +0.010] | +0.0019 [-0.0010, +0.0049] | +0.0010 [-0.0004, +0.0023] |
| pbp_team | +0.000 [-0.011, +0.012] | +0.004 [-0.004, +0.013] | -0.0001 [-0.0018, +0.0015] | +0.0001 [-0.0005, +0.0008] |
| part_only | -0.001 [-0.012, +0.009] | -0.003 [-0.009, +0.003] | -0.0003 [-0.0025, +0.0019] | +0.0003 [-0.0004, +0.0011] |
| no_xfp | -0.003 [-0.022, +0.014] | +0.001 [-0.015, +0.016] | +0.0017 [-0.0014, +0.0049] | +0.0007 [-0.0004, +0.0019] |
| no_xfp_pbp | -0.006 [-0.026, +0.013] | -0.008 [-0.024, +0.007] | +0.0015 [-0.0026, +0.0054] | +0.0008 [-0.0002, +0.0020] |

**2024 walk-forward** (6,025 head-to-head rows, ALL positions)

| run | RMSE | MAE | Spearman | Pick acc |
|---|---|---|---|---|
| primary | 5.771 | 4.065 | 0.663 | 0.775 |
| pbp_usage | 5.779 | 4.076 | 0.660 | 0.774 |
| pbp | 5.774 | 4.067 | 0.661 | 0.774 |
| pbp_team | 5.779 | 4.074 | 0.662 | 0.774 |
| pbp_part | 5.766 | 4.069 | 0.662 | 0.774 |
| part_only | 5.772 | 4.065 | 0.663 | 0.775 |
| no_xfp | 5.780 | 4.081 | 0.661 | 0.774 |
| no_xfp_pbp | 5.770 | 4.071 | 0.663 | 0.775 |
| trailing3 | 6.407 | 4.338 | 0.612 | 0.750 |
| xfp_oracle | 4.006 | 2.493 | 0.857 | 0.879 |

| run minus base | dRMSE | dMAE | dSpearman | dPick |
|---|---|---|---|---|
| pbp_usage | +0.009 [-0.004, +0.022] | +0.012 [+0.000, +0.024] | -0.0028 [-0.0066, +0.0003] | -0.0002 [-0.0013, +0.0007] |
| pbp | +0.004 [-0.009, +0.016] | +0.003 [-0.007, +0.012] | -0.0015 [-0.0059, +0.0024] | -0.0005 [-0.0016, +0.0006] |
| pbp_part | -0.005 [-0.017, +0.006] | +0.004 [-0.005, +0.012] | -0.0012 [-0.0037, +0.0016] | -0.0001 [-0.0014, +0.0011] |
| pbp_team | +0.009 [-0.008, +0.024] | +0.010 [-0.002, +0.023] | -0.0006 [-0.0040, +0.0024] | -0.0005 [-0.0014, +0.0005] |
| part_only | +0.001 [-0.010, +0.013] | +0.001 [-0.007, +0.008] | +0.0003 [-0.0018, +0.0025] | +0.0003 [-0.0009, +0.0013] |
| no_xfp | +0.010 [-0.003, +0.021] | +0.016 [+0.008, +0.025] | -0.0019 [-0.0051, +0.0013] | -0.0004 [-0.0011, +0.0003] |
| no_xfp_pbp | -0.000 [-0.013, +0.012] | +0.007 [-0.000, +0.014] | -0.0001 [-0.0042, +0.0036] | +0.0002 [-0.0006, +0.0010] |

Per position, the pre-declared candidate `pbp` (dRMSE and dSpearman against the primary):

2025:

| position | dRMSE | dSpearman |
|---|---|---|
| QB | -0.000 [-0.043, +0.044] | -0.0001 [-0.0126, +0.0119] |
| RB | -0.026 [-0.061, +0.006] | +0.0036 [+0.0008, +0.0064] |
| WR | -0.006 [-0.028, +0.014] | +0.0007 [-0.0013, +0.0027] |
| TE | +0.008 [-0.011, +0.026] | -0.0016 [-0.0052, +0.0020] |
| ALL | -0.008 [-0.024, +0.007] | +0.0006 [-0.0023, +0.0036] |

2024:

| position | dRMSE | dSpearman |
|---|---|---|
| QB | +0.021 [-0.028, +0.077] | -0.0046 [-0.0199, +0.0093] |
| RB | +0.005 [-0.023, +0.031] | +0.0001 [-0.0034, +0.0034] |
| WR | -0.005 [-0.028, +0.019] | -0.0017 [-0.0047, +0.0009] |
| TE | +0.009 [-0.015, +0.030] | +0.0001 [-0.0038, +0.0038] |
| ALL | +0.004 [-0.009, +0.016] | -0.0015 [-0.0059, +0.0024] |

### What the model does with them

Gain share by family in the last 2025 fit of the all-families model (trained on everything before week 18):

| family | gain share |
|---|---|
| lags | 0.646 |
| adp | 0.181 |
| pbp_part | 0.054 |
| prev_season | 0.037 |
| role | 0.021 |
| pbp_usage | 0.016 |
| vegas | 0.015 |
| pbp_team | 0.009 |
| static | 0.008 |
| position | 0.004 |
| td_luck | 0.003 |
| dvp | 0.002 |
| context | 0.002 |
| injury | 0.001 |
| college | 0.001 |

Top new columns by gain:

| feature | family | share |
|---|---|---|
| part_pass_snap_share_l1 | pbp_part | 0.0318 |
| part_pass_snap_share_std | pbp_part | 0.0090 |
| part_pass_snap_share_t3 | pbp_part | 0.0037 |
| part_run_snap_share_l1 | pbp_part | 0.0036 |
| part_run_snap_share_t3 | pbp_part | 0.0026 |
| pbp_adot_t3 | pbp_usage | 0.0021 |
| team_plays_pg_std | pbp_team | 0.0021 |
| pbp_adot_std | pbp_usage | 0.0019 |
| opp_plays_faced_pg_std | pbp_team | 0.0016 |
| pbp_wopr_std | pbp_usage | 0.0015 |
| opp_pass_rate_faced_std | pbp_team | 0.0015 |
| part_run_snap_share_std | pbp_part | 0.0013 |

### Reading

* **Play-by-play micro-signals do not improve the flat model.** The pre-declared candidate `pbp` moves pooled RMSE by -0.008 [-0.024, +0.007] in 2025
  and +0.004 [-0.009, +0.016] in 2024: the sign flips, both intervals span zero, and Spearman / pick accuracy move by less than 0.003. The pieces do no
  better alone (`pbp_usage` -0.008 [-0.021, +0.003] / +0.009 [-0.004, +0.022]; `pbp_team` +0.000 [-0.011, +0.012] / +0.009 [-0.008, +0.024]).
* **Play-by-play usage is a substitute for the lagged xFP columns, not an addition to them.** Dropping the five xFP lags moves RMSE by +0.010 [-0.003, +0.021] in 2024 (MAE +0.016 [+0.008, +0.025], the only significant one) and -0.003 [-0.022, +0.014] in 2025;
  dropping them and adding `pbp` gives -0.006 [-0.026, +0.013] / -0.000 [-0.013, +0.012], i.e. back to the primary. ffopportunity's expected points are built from the
  same air yards and field position, so red-zone and air-yards shares carry the information the model already had.
* **The participation columns are the only variant that keeps its sign** (`pbp_part`: -0.010 [-0.028, +0.006] / -0.005 [-0.017, +0.006]), by a margin inside the noise
  in both years, and they cannot be served in-season. They take 5.4% of the gain in the all-families fit (pass-play snap share partly replaces snap percentage) without improving the fit.
  Not adopted.
* Nothing from Experiment 1 goes into the Phase 3 feature set. The tables, the tests and the loader stay in the repo (opt-in families) so the question can be
  reopened if a better feed appears.


## Experiment 2: two-stage (volume first)

Stage one is three LightGBM models (targets, carries, pass attempts) on the primary features, walk-forward like the point model, with
their own tree counts. Stage two predicts points from the stage-one predictions, position and 18 shrunken efficiency priors (career and
season-to-date catch rate, yards and TDs per target and per carry, per-attempt passing rates). Stage two trains on **honest** stage-one predictions
(each earlier row's value came from a model trained strictly before its week), so it never learns from a volume fit that saw its own answer; the price
is that 2021 has none, so stage two trains on 2022+ and `flat_2022` is its matched control.

### Stage one: how well can volume itself be predicted?

MAE of the walk-forward model against the trailing-3 mean of the same quantity (the average of the player's last three appearances this season, the number a fantasy player would carry in his head). Head-to-head rows: played, with an earlier game this season, at the positions where the quantity is real volume (targets: RB/WR/TE, carries: RB/QB, attempts: QB; `relevant` pools them). dMAE is model minus trailing-3 with a 95% week-blocked interval; R2 is against the group's own mean.

**targets, 2025**

| position | n | y_mean | MAE_model | MAE_trail3 | MAE_season_mean | dMAE | dMAE_lo | dMAE_hi | RMSE_model | RMSE_trail3 | corr_model | corr_trail3 | R2_model | R2_trail3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RB | 1481 | 1.842 | 1.151 | 1.212 | 1.192 | -0.061 | -0.098 | -0.029 | 1.604 | 1.784 | 0.663 | 0.607 | 0.439 | 0.306 |
| WR | 2532 | 3.508 | 1.702 | 1.822 | 1.809 | -0.120 | -0.180 | -0.066 | 2.304 | 2.566 | 0.748 | 0.700 | 0.556 | 0.450 |
| TE | 1525 | 2.432 | 1.294 | 1.367 | 1.329 | -0.073 | -0.106 | -0.039 | 1.811 | 1.990 | 0.750 | 0.707 | 0.560 | 0.469 |
| relevant | 5538 | 2.766 | 1.442 | 1.534 | 1.512 | -0.091 | -0.129 | -0.060 | 2.005 | 2.225 | 0.754 | 0.708 | 0.566 | 0.466 |

**targets, 2024**

| position | n | y_mean | MAE_model | MAE_trail3 | MAE_season_mean | dMAE | dMAE_lo | dMAE_hi | RMSE_model | RMSE_trail3 | corr_model | corr_trail3 | R2_model | R2_trail3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RB | 1448 | 1.838 | 1.223 | 1.298 | 1.230 | -0.075 | -0.115 | -0.035 | 1.629 | 1.823 | 0.610 | 0.539 | 0.367 | 0.209 |
| WR | 2449 | 3.878 | 1.812 | 1.933 | 1.877 | -0.122 | -0.169 | -0.075 | 2.467 | 2.721 | 0.739 | 0.693 | 0.547 | 0.448 |
| TE | 1526 | 2.356 | 1.335 | 1.402 | 1.370 | -0.067 | -0.106 | -0.032 | 1.965 | 2.150 | 0.741 | 0.691 | 0.546 | 0.456 |
| relevant | 5423 | 2.905 | 1.520 | 1.614 | 1.562 | -0.094 | -0.121 | -0.070 | 2.132 | 2.352 | 0.750 | 0.703 | 0.563 | 0.468 |

**carries, 2025**

| position | n | y_mean | MAE_model | MAE_trail3 | MAE_season_mean | dMAE | dMAE_lo | dMAE_hi | RMSE_model | RMSE_trail3 | corr_model | corr_trail3 | R2_model | R2_trail3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RB | 1481 | 7.485 | 3.105 | 3.263 | 3.227 | -0.159 | -0.262 | -0.058 | 4.276 | 4.695 | 0.794 | 0.758 | 0.630 | 0.553 |
| QB | 601 | 3.354 | 1.810 | 1.963 | 1.902 | -0.153 | -0.230 | -0.076 | 2.318 | 2.571 | 0.525 | 0.473 | 0.246 | 0.071 |
| relevant | 2082 | 6.293 | 2.731 | 2.888 | 2.845 | -0.157 | -0.242 | -0.077 | 3.815 | 4.194 | 0.801 | 0.766 | 0.642 | 0.567 |

**carries, 2024**

| position | n | y_mean | MAE_model | MAE_trail3 | MAE_season_mean | dMAE | dMAE_lo | dMAE_hi | RMSE_model | RMSE_trail3 | corr_model | corr_trail3 | R2_model | R2_trail3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RB | 1448 | 7.559 | 3.303 | 3.486 | 3.510 | -0.183 | -0.298 | -0.068 | 4.468 | 4.842 | 0.784 | 0.754 | 0.614 | 0.547 |
| QB | 602 | 3.523 | 1.738 | 1.923 | 1.866 | -0.185 | -0.273 | -0.097 | 2.265 | 2.547 | 0.646 | 0.595 | 0.407 | 0.249 |
| relevant | 2050 | 6.374 | 2.844 | 3.027 | 3.027 | -0.184 | -0.280 | -0.092 | 3.951 | 4.297 | 0.795 | 0.765 | 0.632 | 0.565 |

**pass attempts, 2025**

| position | n | y_mean | MAE_model | MAE_trail3 | dMAE | dMAE_lo | dMAE_hi | RMSE_model | RMSE_trail3 | corr_model | corr_trail3 | R2_model | R2_trail3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| QB | 601 | 26.624 | 7.529 | 8.296 | -0.766 | -1.048 | -0.464 | 9.881 | 11.238 | 0.654 | 0.593 | 0.422 | 0.253 |

**pass attempts, 2024**

| position | n | y_mean | MAE_model | MAE_trail3 | dMAE | dMAE_lo | dMAE_hi | RMSE_model | RMSE_trail3 | corr_model | corr_trail3 | R2_model | R2_trail3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| QB | 602 | 27.171 | 7.642 | 8.457 | -0.815 | -1.171 | -0.450 | 10.040 | 11.298 | 0.645 | 0.593 | 0.415 | 0.259 |

### Stage two and the flat variants

Base for the deltas is the Phase 2 **primary**. `two_stage` = stage two alone; `flat_plus_s1` = primary features + stage-one predictions; `flat_plus_eff` and `flat_plus_s1_eff` separate the efficiency priors from the volume predictions; `flat_no_xfp` drops the five lagged-xFP columns and `flat_no_xfp_plus_s1` replaces them with the stage-one predictions (does stage one beat lagged xFP as the volume signal?).

**2025 walk-forward** (6,139 head-to-head rows, ALL positions)

| run | RMSE | MAE | Spearman | Pick acc |
|---|---|---|---|---|
| primary | 5.711 | 4.019 | 0.655 | 0.773 |
| two_stage | 5.777 | 4.060 | 0.641 | 0.769 |
| flat_2022 | 5.729 | 4.032 | 0.655 | 0.773 |
| flat_plus_s1 | 5.708 | 4.012 | 0.655 | 0.773 |
| flat_plus_eff | 5.706 | 4.017 | 0.656 | 0.774 |
| flat_plus_s1_eff | 5.700 | 4.006 | 0.657 | 0.774 |
| flat_no_xfp | 5.707 | 4.020 | 0.657 | 0.774 |
| flat_no_xfp_plus_s1 | 5.704 | 4.003 | 0.658 | 0.774 |
| trailing3 | 6.432 | 4.314 | 0.602 | 0.748 |
| xfp_oracle | 4.073 | 2.503 | 0.866 | 0.876 |

| run minus base | dRMSE | dMAE | dSpearman | dPick |
|---|---|---|---|---|
| flat_2022 | +0.019 [+0.003, +0.034] | +0.013 [+0.005, +0.022] | -0.0005 [-0.0037, +0.0028] | -0.0004 [-0.0012, +0.0004] |
| two_stage | +0.067 [+0.031, +0.101] | +0.041 [+0.013, +0.069] | -0.0144 [-0.0223, -0.0070] | -0.0039 [-0.0063, -0.0014] |
| flat_plus_s1 | -0.003 [-0.013, +0.008] | -0.007 [-0.015, +0.003] | +0.0003 [-0.0025, +0.0034] | +0.0000 [-0.0008, +0.0009] |
| flat_plus_eff | -0.004 [-0.015, +0.006] | -0.002 [-0.014, +0.008] | +0.0008 [-0.0014, +0.0031] | +0.0006 [-0.0004, +0.0016] |
| flat_plus_s1_eff | -0.010 [-0.021, +0.001] | -0.012 [-0.022, -0.002] | +0.0015 [-0.0021, +0.0050] | +0.0012 [+0.0005, +0.0020] |
| flat_no_xfp | -0.003 [-0.022, +0.014] | +0.001 [-0.015, +0.016] | +0.0017 [-0.0014, +0.0049] | +0.0007 [-0.0004, +0.0019] |
| flat_no_xfp_plus_s1 | -0.007 [-0.021, +0.007] | -0.016 [-0.028, -0.005] | +0.0026 [+0.0004, +0.0050] | +0.0013 [+0.0002, +0.0024] |

**2024 walk-forward** (6,025 head-to-head rows, ALL positions)

| run | RMSE | MAE | Spearman | Pick acc |
|---|---|---|---|---|
| primary | 5.771 | 4.065 | 0.663 | 0.775 |
| two_stage | 5.858 | 4.116 | 0.653 | 0.769 |
| flat_2022 | 5.781 | 4.077 | 0.661 | 0.774 |
| flat_plus_s1 | 5.790 | 4.078 | 0.658 | 0.773 |
| flat_plus_eff | 5.777 | 4.073 | 0.662 | 0.774 |
| flat_plus_s1_eff | 5.779 | 4.068 | 0.662 | 0.774 |
| flat_no_xfp | 5.780 | 4.081 | 0.661 | 0.774 |
| flat_no_xfp_plus_s1 | 5.774 | 4.070 | 0.661 | 0.774 |
| trailing3 | 6.407 | 4.338 | 0.612 | 0.750 |
| xfp_oracle | 4.006 | 2.493 | 0.857 | 0.879 |

| run minus base | dRMSE | dMAE | dSpearman | dPick |
|---|---|---|---|---|
| flat_2022 | +0.011 [-0.007, +0.028] | +0.012 [-0.001, +0.027] | -0.0015 [-0.0071, +0.0032] | -0.0010 [-0.0021, +0.0000] |
| two_stage | +0.088 [+0.047, +0.127] | +0.051 [+0.016, +0.084] | -0.0103 [-0.0173, -0.0031] | -0.0057 [-0.0084, -0.0030] |
| flat_plus_s1 | +0.020 [+0.007, +0.033] | +0.014 [+0.005, +0.023] | -0.0046 [-0.0083, -0.0008] | -0.0011 [-0.0022, -0.0001] |
| flat_plus_eff | +0.006 [-0.009, +0.021] | +0.008 [-0.004, +0.021] | -0.0012 [-0.0039, +0.0017] | -0.0003 [-0.0012, +0.0005] |
| flat_plus_s1_eff | +0.009 [-0.004, +0.020] | +0.003 [-0.006, +0.011] | -0.0014 [-0.0045, +0.0017] | -0.0002 [-0.0009, +0.0005] |
| flat_no_xfp | +0.010 [-0.003, +0.021] | +0.016 [+0.008, +0.025] | -0.0019 [-0.0051, +0.0013] | -0.0004 [-0.0011, +0.0003] |
| flat_no_xfp_plus_s1 | +0.003 [-0.013, +0.020] | +0.005 [-0.007, +0.018] | -0.0018 [-0.0047, +0.0010] | -0.0002 [-0.0010, +0.0006] |

`two_stage` against its matched control `flat_2022` (both train on 2022+):

2025:

| run minus base | dRMSE | dMAE | dSpearman | dPick |
|---|---|---|---|---|
| two_stage | +0.048 [+0.007, +0.086] | +0.028 [-0.002, +0.057] | -0.0139 [-0.0206, -0.0073] | -0.0035 [-0.0064, -0.0007] |
| primary | -0.019 [-0.034, -0.003] | -0.013 [-0.022, -0.005] | +0.0005 [-0.0028, +0.0037] | +0.0004 [-0.0004, +0.0012] |

2024:

| run minus base | dRMSE | dMAE | dSpearman | dPick |
|---|---|---|---|---|
| two_stage | +0.077 [+0.041, +0.112] | +0.039 [+0.008, +0.069] | -0.0088 [-0.0170, +0.0001] | -0.0047 [-0.0069, -0.0023] |
| primary | -0.011 [-0.028, +0.007] | -0.012 [-0.027, +0.001] | +0.0015 [-0.0032, +0.0071] | +0.0010 [-0.0000, +0.0021] |


### Reading

* **Volume is only modestly predictable, and that is where the ceiling problem lives.** The stage-one models beat the trailing-3 mean by
  5.9% (2025) and 5.8% (2024) on target MAE, 5.4% / 6.1% on carries (RB and QB) and
  9.2% / 9.6% on quarterback attempts; every interval excludes zero. In absolute terms a receiver's targets are still off by
  1.44 a game on a mean of 2.77 (R-squared 0.57 against 0.47 for the trailing mean), a
  back's carries by 3.10 on 7.48, a quarterback's attempts by 7.5 on 26.6. Better volume estimates are worth a few percent of MAE,
  not the 1.6-point RMSE gap to the same-week xFP oracle, which sees the volume the player actually got: nothing knowable before kickoff that these experiments tried recovers it. Game script,
  in-game injuries and coaching choices are the likely sources; that part is inference, not something measured here.
* **The end-to-end two-stage model loses.** Stage two alone moves RMSE by +0.067 [+0.031, +0.101] in 2025 and +0.088 [+0.047, +0.127] in 2024, worse in both years, and
  worse than its matched control that trains on the same window (+0.048 [+0.007, +0.086] / +0.077 [+0.041, +0.112]). Stage-one error compounds, and the structure also drops the
  role, Vegas, injury and prior-season columns the flat model uses (`flat_plus_s1_eff`, which keeps them, does not beat the primary either).
* **Stage-one predictions as extra flat inputs do not replicate**: `flat_plus_s1` -0.003 [-0.013, +0.008] / +0.020 [+0.007, +0.033] (2024 significantly worse);
  they do not replace lagged xFP either (`flat_no_xfp_plus_s1` -0.007 [-0.021, +0.007] / +0.003 [-0.013, +0.020]). The efficiency priors alone are indistinguishable from
  nothing (`flat_plus_eff` -0.004 [-0.015, +0.006] / +0.006 [-0.009, +0.021]).
* Nothing from Experiment 2 goes into the Phase 3 design. The stage-one volume models are kept: they are components in Experiment 3.


## Experiment 3: component models

One LightGBM per scoring component (14 of them: tgt, rec, rec_yds, rec_td, car, rush_yds, rush_td, att, cmp, pass_yds, pass_td, int, fum_lost, misc_pts), each walk-forward on the same
rows, each with its own early-stopped tree count, then composed with the scoring weights. `components` uses the primary features; `components_eff`
adds the `eff` family (shrunken efficiency), because a yards model without any yards-per-target history is handicapped. Points are the
conditional-mean sum, so any linear scoring composes exactly.

**Identity check.** Composed from the *actual* components, PPR equals nflverse's `fantasy_points_ppr` on every played row of the matrix
(max abs difference 7.1e-15 over 33,423 rows; pinned by a test), so the composition path cannot drift. Tie-break between the two variants if both tie:
the one that adds no new features (`components`).

### Composed PPR against the primary

**2025 walk-forward** (6,139 head-to-head rows, ALL positions)

| run | RMSE | MAE | Spearman | Pick acc |
|---|---|---|---|---|
| primary | 5.711 | 4.019 | 0.655 | 0.773 |
| components | 5.703 | 4.002 | 0.663 | 0.774 |
| components_eff | 5.717 | 4.015 | 0.660 | 0.774 |
| trailing3 | 6.432 | 4.314 | 0.602 | 0.748 |
| xfp_oracle | 4.073 | 2.503 | 0.866 | 0.876 |

| run minus base | dRMSE | dMAE | dSpearman | dPick |
|---|---|---|---|---|
| components | -0.008 [-0.029, +0.013] | -0.017 [-0.036, +0.003] | +0.0084 [+0.0035, +0.0139] | +0.0013 [-0.0001, +0.0027] |
| components_eff | +0.006 [-0.022, +0.034] | -0.004 [-0.027, +0.019] | +0.0052 [-0.0017, +0.0137] | +0.0007 [-0.0009, +0.0022] |

**2024 walk-forward** (6,025 head-to-head rows, ALL positions)

| run | RMSE | MAE | Spearman | Pick acc |
|---|---|---|---|---|
| primary | 5.771 | 4.065 | 0.663 | 0.775 |
| components | 5.777 | 4.073 | 0.663 | 0.774 |
| components_eff | 5.773 | 4.067 | 0.666 | 0.774 |
| trailing3 | 6.407 | 4.338 | 0.612 | 0.750 |
| xfp_oracle | 4.006 | 2.493 | 0.857 | 0.879 |

| run minus base | dRMSE | dMAE | dSpearman | dPick |
|---|---|---|---|---|
| components | +0.006 [-0.020, +0.034] | +0.008 [-0.009, +0.026] | -0.0002 [-0.0059, +0.0047] | -0.0007 [-0.0022, +0.0008] |
| components_eff | +0.003 [-0.030, +0.036] | +0.002 [-0.019, +0.022] | +0.0028 [-0.0049, +0.0110] | -0.0003 [-0.0016, +0.0010] |

Per position (composed `components` minus primary):

2025:

| position | dRMSE | dSpearman |
|---|---|---|
| QB | -0.099 [-0.175, -0.016] | +0.0247 [+0.0062, +0.0445] |
| RB | +0.023 [-0.020, +0.073] | +0.0043 [-0.0009, +0.0092] |
| WR | +0.005 [-0.018, +0.028] | -0.0015 [-0.0052, +0.0022] |
| TE | -0.014 [-0.040, +0.011] | +0.0059 [-0.0001, +0.0118] |
| ALL | -0.008 [-0.029, +0.013] | +0.0084 [+0.0035, +0.0139] |

2024:

| position | dRMSE | dSpearman |
|---|---|---|
| QB | +0.017 [-0.091, +0.125] | -0.0011 [-0.0203, +0.0153] |
| RB | +0.014 [-0.031, +0.066] | -0.0012 [-0.0067, +0.0044] |
| WR | +0.008 [-0.026, +0.046] | -0.0034 [-0.0071, +0.0004] |
| TE | -0.016 [-0.051, +0.016] | +0.0048 [-0.0002, +0.0098] |
| ALL | +0.006 [-0.020, +0.034] | -0.0002 [-0.0059, +0.0047] |

### How much of each component is predictable

R-squared-style skill (1 - MSE / MSE of a constant per position) on the head-to-head rows, `components` (primary features):

2025:

| component | n | mean | RMSE | RMSE_pos_const | skill_vs_pos_const |
|---|---|---|---|---|---|
| y_tgt | 6139 | 2.499 | 1.906 | 2.812 | 0.540 |
| y_rec | 6139 | 1.687 | 1.488 | 2.047 | 0.471 |
| y_rec_yds | 6139 | 18.523 | 20.396 | 26.345 | 0.401 |
| y_rec_td | 6139 | 0.123 | 0.348 | 0.367 | 0.101 |
| y_car | 6139 | 2.201 | 2.242 | 3.563 | 0.604 |
| y_rush_yds | 6139 | 9.658 | 14.582 | 19.193 | 0.423 |
| y_rush_td | 6139 | 0.076 | 0.286 | 0.304 | 0.114 |
| y_att | 6139 | 2.609 | 3.097 | 4.068 | 0.421 |
| y_cmp | 6139 | 1.679 | 2.081 | 2.712 | 0.412 |
| y_pass_yds | 6139 | 18.429 | 24.729 | 31.856 | 0.397 |
| y_pass_td | 6139 | 0.124 | 0.333 | 0.373 | 0.205 |
| y_int | 6139 | 0.058 | 0.242 | 0.247 | 0.041 |
| y_fum_lost | 6139 | 0.032 | 0.180 | 0.181 | 0.007 |
| y_misc_pts | 6139 | 0.050 | 0.419 | 0.419 | -0.002 |

2024:

| component | n | mean | RMSE | RMSE_pos_const | skill_vs_pos_const |
|---|---|---|---|---|---|
| y_tgt | 6025 | 2.615 | 2.023 | 2.936 | 0.525 |
| y_rec | 6025 | 1.791 | 1.584 | 2.166 | 0.465 |
| y_rec_yds | 6025 | 19.638 | 21.065 | 27.321 | 0.406 |
| y_rec_td | 6025 | 0.125 | 0.348 | 0.366 | 0.095 |
| y_car | 6025 | 2.242 | 2.325 | 3.664 | 0.597 |
| y_rush_yds | 6025 | 9.984 | 14.589 | 19.705 | 0.452 |
| y_rush_td | 6025 | 0.077 | 0.275 | 0.301 | 0.169 |
| y_att | 6025 | 2.719 | 3.181 | 4.150 | 0.412 |
| y_cmp | 6025 | 1.781 | 2.157 | 2.821 | 0.415 |
| y_pass_yds | 6025 | 19.535 | 25.456 | 32.724 | 0.395 |
| y_pass_td | 6025 | 0.126 | 0.344 | 0.380 | 0.177 |
| y_int | 6025 | 0.059 | 0.254 | 0.261 | 0.059 |
| y_fum_lost | 6025 | 0.035 | 0.182 | 0.183 | 0.007 |
| y_misc_pts | 6025 | 0.040 | 0.343 | 0.343 | 0.000 |

### The alternate scoring: the dynasty league's TE premium

`leagues/we-can-think-of-something-funny/league.json`: full PPR (`rec` = 1.0) plus `bonus_rec_te` = 0.5 per tight-end catch. The composition adds the bonus to predicted tight-end receptions, the flat model has no reception estimate, so its baseline (`flat_plus_trailing_catches`) is the naive route: the PPR prediction plus the bonus times the tight end's average catches over his last three appearances (from the matrix's own earlier rows, a baseline never an input). Actual points are rebuilt from actual components, and the league's other divergences from the composed scoring are listed below: they are not composed.

Divergences between this league's real scoring and the composed one: pass_int: league -1.0 vs PPR -2.0; threshold bonuses (need a distribution, not a mean): bonus_pass_yd_300, bonus_pass_yd_400, bonus_rec_yd_100, bonus_rec_yd_200, bonus_rush_yd_100, bonus_rush_yd_200, fgmiss_50p, pass_td_40p, pass_td_50p, rec_td_40p, rec_td_50p, rush_td_40p, rush_td_50p.

**2025, TE-premium points, ALL positions** (6,139 rows; only tight ends' actual points differ from PPR)

| run | RMSE | MAE | Spearman | Pick acc |
|---|---|---|---|---|
| flat_plus_trailing_catches | 5.842 | 4.132 | 0.657 | 0.774 |
| flat_ppr_unshifted | 5.876 | 4.118 | 0.656 | 0.773 |
| components | 5.833 | 4.119 | 0.665 | 0.775 |
| components_eff | 5.847 | 4.133 | 0.662 | 0.774 |

| run minus base | dRMSE | dMAE | dSpearman | dPick |
|---|---|---|---|---|
| components | -0.009 [-0.030, +0.012] | -0.013 [-0.031, +0.007] | +0.0074 [+0.0024, +0.0128] | +0.0010 [-0.0003, +0.0022] |
| components_eff | +0.005 [-0.025, +0.034] | +0.001 [-0.023, +0.024] | +0.0044 [-0.0026, +0.0127] | +0.0003 [-0.0010, +0.0017] |
| flat_ppr_unshifted | +0.034 [+0.016, +0.054] | -0.014 [-0.026, -0.003] | -0.0010 [-0.0022, +0.0001] | -0.0004 [-0.0008, -0.0001] |

2025, tight ends only (1,525 rows):

| run | RMSE | MAE | Spearman | Pick acc |
|---|---|---|---|---|
| flat_plus_trailing_catches | 5.140 | 3.458 | 0.704 | 0.785 |
| flat_ppr_unshifted | 5.295 | 3.401 | 0.700 | 0.783 |
| components | 5.123 | 3.430 | 0.706 | 0.787 |
| components_eff | 5.141 | 3.447 | 0.706 | 0.787 |

| run minus base | dRMSE | dMAE | dSpearman | dPick |
|---|---|---|---|---|
| components | -0.017 [-0.055, +0.018] | -0.028 [-0.062, +0.007] | +0.0021 [-0.0032, +0.0075] | +0.0012 [-0.0014, +0.0039] |
| components_eff | +0.001 [-0.041, +0.043] | -0.010 [-0.044, +0.023] | +0.0019 [-0.0042, +0.0079] | +0.0012 [-0.0016, +0.0039] |
| flat_ppr_unshifted | +0.155 [+0.077, +0.226] | -0.057 [-0.104, -0.014] | -0.0038 [-0.0088, +0.0005] | -0.0021 [-0.0042, -0.0003] |

**2024, TE-premium points, ALL positions** (6,025 rows; only tight ends' actual points differ from PPR)

| run | RMSE | MAE | Spearman | Pick acc |
|---|---|---|---|---|
| flat_plus_trailing_catches | 5.911 | 4.186 | 0.664 | 0.775 |
| flat_ppr_unshifted | 5.958 | 4.189 | 0.664 | 0.775 |
| components | 5.917 | 4.200 | 0.664 | 0.775 |
| components_eff | 5.912 | 4.193 | 0.667 | 0.775 |

| run minus base | dRMSE | dMAE | dSpearman | dPick |
|---|---|---|---|---|
| components | +0.006 [-0.021, +0.035] | +0.014 [-0.006, +0.035] | +0.0000 [-0.0058, +0.0049] | -0.0007 [-0.0023, +0.0008] |
| components_eff | +0.001 [-0.032, +0.034] | +0.007 [-0.015, +0.029] | +0.0030 [-0.0048, +0.0108] | -0.0002 [-0.0016, +0.0011] |
| flat_ppr_unshifted | +0.047 [+0.022, +0.073] | +0.003 [-0.015, +0.022] | +0.0001 [-0.0006, +0.0010] | -0.0000 [-0.0003, +0.0003] |

2024, tight ends only (1,526 rows):

| run | RMSE | MAE | Spearman | Pick acc |
|---|---|---|---|---|
| flat_plus_trailing_catches | 5.019 | 3.389 | 0.688 | 0.791 |
| flat_ppr_unshifted | 5.233 | 3.400 | 0.688 | 0.791 |
| components | 5.006 | 3.402 | 0.693 | 0.793 |
| components_eff | 4.978 | 3.393 | 0.695 | 0.794 |

| run minus base | dRMSE | dMAE | dSpearman | dPick |
|---|---|---|---|---|
| components | -0.013 [-0.056, +0.029] | +0.013 [-0.015, +0.039] | +0.0057 [+0.0010, +0.0103] | +0.0020 [-0.0005, +0.0047] |
| components_eff | -0.041 [-0.092, +0.006] | +0.003 [-0.029, +0.032] | +0.0069 [+0.0014, +0.0124] | +0.0028 [-0.0003, +0.0059] |
| flat_ppr_unshifted | +0.214 [+0.107, +0.324] | +0.011 [-0.058, +0.084] | +0.0006 [-0.0024, +0.0038] | -0.0002 [-0.0018, +0.0014] |


### Reading

* **Composing from component models ties the flat model on PPR**: `components` moves RMSE by -0.008 [-0.029, +0.013] in 2025 and +0.006 [-0.020, +0.034] in 2024
  (both intervals span zero: a tie, which ships under the rule), and Spearman by +0.0084 [+0.0035, +0.0139] in 2025 and
  -0.0002 [-0.0059, +0.0047] in 2024. Adding the efficiency family does not help (`components_eff` +0.006 [-0.022, +0.034] / +0.003 [-0.030, +0.036]),
  so the simpler variant is the design. By position the composed model is better for quarterbacks in 2025
  (dRMSE -0.099) and not in 2024 (+0.017): no position is reliably different.
* **The composition path works for another scoring.** Under the league's TE premium the composed prediction is at least as good as the naive route on tight ends
  (-0.017 [-0.055, +0.018] in 2025, -0.013 [-0.056, +0.029] in 2024, against PPR-plus-bonus-times-recent-catches), and ignoring the premium altogether costs
  +0.155 [+0.077, +0.226] / +0.214 [+0.107, +0.324] on tight ends, so the shift matters and the components get it right without a
  per-position patch. Any linear weights compose the same way; threshold bonuses (this league has 100/200-yard and 40/50-yard TD bonuses, and scores an interception at -1) need
  the yardage distribution, which a mean model does not give (interceptions compose trivially and are a one-line change).
* **What is predictable, component by component** (skill against a constant per position, 2025): targets 0.54, receptions 0.47, carries 0.60, rushing and
  receiving yards 0.42 / 0.40, pass attempts and yards 0.42 / 0.40, but receiving and rushing TDs only 0.10 / 0.11,
  passing TDs 0.21, interceptions 0.04, fumbles lost 0.01, and the two-point / special-teams bucket none. Touchdowns carry six points and almost no
  predictable signal, which is the other half of the ceiling.


## Experiment 4: cheap wins

### Three-library ensemble

The simple average of LightGBM, XGBoost and CatBoost, and weights (non-negative, sum to one, 0.01 grid, minimising RMSE on the head-to-head rows) fit on ONE earlier walk-forward year and scored on the next: 2024 weights scored on 2025 (the plan) and, as the out-of-time replication, 2023 weights scored on 2024. The weights are fit on predictions that were themselves made walk-forward.

**2025 walk-forward** (6,139 head-to-head rows, ALL positions); weights fit on 2024: lightgbm 0.55, xgboost 0.17, catboost 0.28

| run | RMSE | MAE | Spearman | Pick acc |
|---|---|---|---|---|
| lightgbm | 5.711 | 4.019 | 0.655 | 0.773 |
| xgboost | 5.706 | 4.017 | 0.657 | 0.774 |
| catboost | 5.710 | 4.015 | 0.661 | 0.774 |
| average | 5.699 | 4.010 | 0.659 | 0.774 |
| weighted | 5.701 | 4.012 | 0.657 | 0.774 |
| trailing3 | 6.432 | 4.314 | 0.602 | 0.748 |
| xfp_oracle | 4.073 | 2.503 | 0.866 | 0.876 |

| run minus base | dRMSE | dMAE | dSpearman | dPick |
|---|---|---|---|---|
| xgboost | -0.004 [-0.021, +0.014] | -0.002 [-0.015, +0.013] | +0.0023 [-0.0021, +0.0067] | +0.0009 [-0.0000, +0.0019] |
| catboost | -0.001 [-0.012, +0.012] | -0.004 [-0.013, +0.005] | +0.0057 [+0.0023, +0.0090] | +0.0008 [-0.0002, +0.0019] |
| average | -0.011 [-0.020, -0.003] | -0.009 [-0.015, -0.002] | +0.0038 [+0.0015, +0.0060] | +0.0013 [+0.0008, +0.0018] |
| weighted | -0.009 [-0.015, -0.004] | -0.007 [-0.011, -0.003] | +0.0018 [+0.0000, +0.0035] | +0.0009 [+0.0004, +0.0013] |

**2024 walk-forward** (6,025 head-to-head rows, ALL positions); weights fit on 2023: lightgbm 0.21, xgboost 0.42, catboost 0.37

| run | RMSE | MAE | Spearman | Pick acc |
|---|---|---|---|---|
| lightgbm | 5.771 | 4.065 | 0.663 | 0.775 |
| xgboost | 5.778 | 4.066 | 0.663 | 0.775 |
| catboost | 5.782 | 4.072 | 0.661 | 0.774 |
| average | 5.767 | 4.061 | 0.663 | 0.775 |
| weighted | 5.768 | 4.061 | 0.664 | 0.775 |
| trailing3 | 6.407 | 4.338 | 0.612 | 0.750 |
| xfp_oracle | 4.006 | 2.493 | 0.857 | 0.879 |

| run minus base | dRMSE | dMAE | dSpearman | dPick |
|---|---|---|---|---|
| xgboost | +0.008 [-0.008, +0.024] | +0.002 [-0.011, +0.016] | -0.0001 [-0.0029, +0.0029] | +0.0000 [-0.0008, +0.0007] |
| catboost | +0.012 [-0.007, +0.029] | +0.008 [-0.007, +0.021] | -0.0022 [-0.0063, +0.0015] | -0.0004 [-0.0016, +0.0008] |
| average | -0.004 [-0.012, +0.005] | -0.004 [-0.011, +0.004] | +0.0005 [-0.0016, +0.0030] | +0.0005 [-0.0000, +0.0011] |
| weighted | -0.002 [-0.013, +0.008] | -0.003 [-0.012, +0.006] | +0.0007 [-0.0015, +0.0032] | +0.0006 [+0.0000, +0.0013] |

### Post-hoc: averaging the flat ensemble with the component composition

Not part of the pre-declared plan and not in the adoption table: since both the ensemble and the component design pass their rules, does averaging their PPR predictions help? (LightGBM components, composed to PPR, averaged 50/50 with the flat three-library average or with flat LightGBM.) Read as a lead, not a result: it was thought of after seeing the tables above.

**2025**

| run | RMSE | MAE | Spearman | Pick acc |
|---|---|---|---|---|
| lightgbm | 5.711 | 4.019 | 0.655 | 0.773 |
| average | 5.699 | 4.010 | 0.659 | 0.774 |
| components | 5.703 | 4.002 | 0.663 | 0.774 |
| average_and_components | 5.690 | 3.999 | 0.663 | 0.775 |
| lightgbm_and_components | 5.694 | 4.002 | 0.661 | 0.775 |

| run minus base | dRMSE | dMAE | dSpearman | dPick |
|---|---|---|---|---|
| average | -0.011 [-0.020, -0.003] | -0.009 [-0.015, -0.002] | +0.0038 [+0.0015, +0.0060] | +0.0013 [+0.0008, +0.0018] |
| components | -0.008 [-0.029, +0.013] | -0.017 [-0.036, +0.003] | +0.0084 [+0.0035, +0.0139] | +0.0013 [-0.0001, +0.0027] |
| average_and_components | -0.020 [-0.033, -0.007] | -0.020 [-0.030, -0.009] | +0.0081 [+0.0046, +0.0118] | +0.0021 [+0.0011, +0.0031] |
| lightgbm_and_components | -0.016 [-0.026, -0.006] | -0.017 [-0.026, -0.007] | +0.0064 [+0.0038, +0.0094] | +0.0016 [+0.0008, +0.0024] |

**2024**

| run | RMSE | MAE | Spearman | Pick acc |
|---|---|---|---|---|
| lightgbm | 5.771 | 4.065 | 0.663 | 0.775 |
| average | 5.767 | 4.061 | 0.663 | 0.775 |
| components | 5.777 | 4.073 | 0.663 | 0.774 |
| average_and_components | 5.761 | 4.061 | 0.665 | 0.775 |
| lightgbm_and_components | 5.761 | 4.061 | 0.666 | 0.775 |

| run minus base | dRMSE | dMAE | dSpearman | dPick |
|---|---|---|---|---|
| average | -0.004 [-0.012, +0.005] | -0.004 [-0.011, +0.004] | +0.0005 [-0.0016, +0.0030] | +0.0005 [-0.0000, +0.0011] |
| components | +0.006 [-0.020, +0.034] | +0.008 [-0.009, +0.026] | -0.0002 [-0.0059, +0.0047] | -0.0007 [-0.0022, +0.0008] |
| average_and_components | -0.009 [-0.025, +0.006] | -0.004 [-0.015, +0.007] | +0.0018 [-0.0017, +0.0048] | +0.0008 [-0.0001, +0.0016] |
| lightgbm_and_components | -0.010 [-0.022, +0.003] | -0.003 [-0.012, +0.006] | +0.0027 [-0.0008, +0.0060] | +0.0009 [+0.0000, +0.0017] |

### Quantile recalibration

The band is LightGBM p10-p90. For each target week, per position, the band is shifted by the empirical residual quantiles of the rows
*before* it: q10 by the 10th percentile of (actual - q10), q90 by the 90th percentile of (actual - q90), over the previous season's out-of-sample
walk-forward plus the current season's earlier weeks (played rows). Default rule (`underage_only`): a position is touched only if its trailing
p10-p90 coverage is below 80% by more than 3 points, so a position that is already calibrated is left alone; a position needs 100 trailing rows.
Variants: `qb_only` (the same rule, quarterbacks only, so every other band is exactly as it was) and `all` (every position shifted, a diagnostic).
Coverage is on all played rows of the season (the Phase 2 convention); intervals are week-blocked bootstraps. The pass criteria written before the
runs: quarterbacks within 2 points of 80% in both years, and no RB/WR/TE position widened by more than 2% (mean band width).

**2025** (all played rows, n = 6,755; history = 2024 walk-forward)

| position | n | coverage before | after: underage_only | after: qb_only | after: all shifted | mean width before | mean width after (underage_only) | width change |
|---|---|---|---|---|---|---|---|---|
| QB | 681 | 0.744 [0.716, 0.769] | 0.789 [0.762, 0.812] | 0.789 | 0.789 | 17.705 | 19.807 | +11.9% |
| RB | 1628 | 0.789 [0.770, 0.809] | 0.789 [0.770, 0.809] | 0.789 | 0.808 | 12.420 | 12.420 | +0.0% |
| WR | 2777 | 0.781 [0.769, 0.795] | 0.787 [0.772, 0.803] | 0.781 | 0.812 | 12.004 | 12.059 | +0.5% |
| TE | 1669 | 0.786 [0.765, 0.808] | 0.806 [0.782, 0.829] | 0.786 | 0.818 | 8.598 | 8.704 | +1.2% |
| ALL | 6755 | 0.781 [0.772, 0.790] | 0.792 [0.783, 0.804] | 0.785 | 0.810 | 11.837 | 12.098 | +2.2% |

Interval score (Winkler, alpha .2, lower is better): 17.137 before, 17.142 after (underage_only), 17.146 with every position shifted; quarterbacks alone 26.366 to 26.350. Quarterbacks below p10 / above p90: 0.142 / 0.113 before, 0.129 / 0.082 after (targets .100 / .100). Which positions the rule touched: QB, WR, TE.

**2024** (all played rows, n = 6,407; history = 2023 walk-forward)

| position | n | coverage before | after: underage_only | after: qb_only | after: all shifted | mean width before | mean width after (underage_only) | width change |
|---|---|---|---|---|---|---|---|---|
| QB | 659 | 0.751 [0.708, 0.791] | 0.794 [0.752, 0.831] | 0.794 | 0.794 | 17.064 | 19.164 | +12.3% |
| RB | 1529 | 0.780 [0.757, 0.802] | 0.780 [0.757, 0.802] | 0.780 | 0.789 | 12.553 | 12.553 | +0.0% |
| WR | 2606 | 0.765 [0.748, 0.782] | 0.809 [0.793, 0.826] | 0.765 | 0.809 | 12.489 | 12.769 | +2.2% |
| TE | 1613 | 0.753 [0.735, 0.772] | 0.801 [0.784, 0.817] | 0.753 | 0.801 | 8.248 | 8.366 | +1.4% |
| ALL | 6407 | 0.764 [0.752, 0.775] | 0.798 [0.788, 0.808] | 0.768 | 0.801 | 11.907 | 12.267 | +3.0% |

Interval score (Winkler, alpha .2, lower is better): 17.759 before, 17.713 after (underage_only), 17.717 with every position shifted; quarterbacks alone 26.392 to 25.967. Quarterbacks below p10 / above p90: 0.106 / 0.143 before, 0.100 / 0.106 after (targets .100 / .100). Which positions the rule touched: QB, WR, TE.


### Reading

* **The simple average of the three libraries ships; fitted weights add nothing.** `average` moves RMSE by -0.011 [-0.020, -0.003] in 2025 (interval excludes zero) and
  -0.004 [-0.012, +0.005] in 2024 (same direction, interval spans zero), and Spearman by +0.0038 [+0.0015, +0.0060] / +0.0005 [-0.0016, +0.0030]. The weights fit on one year
  (0.55 / 0.17 / 0.28 on 2024, 0.21 / 0.42 / 0.37 on 2023, LightGBM / XGBoost / CatBoost) move
  between years and score no better than equal weights (-0.009 [-0.015, -0.004] / -0.002 [-0.013, +0.008]). The gain is small (about 0.2% of RMSE) because the three libraries
  are nearly the same model on the same features; it costs about 11 seconds of refit a week instead of 1.
* **Quantile recalibration fixes the quarterback band and, by design, leaves the calibrated positions alone.** QB p10-p90 coverage goes from 0.744 to 0.789 in 2025 and
  0.751 to 0.794 in 2024 (target .800; the 95% intervals now contain it), at a cost of about +12% in QB band width; RB is untouched in both years. The rule also widened WR and TE slightly (2025: +0.5% / +1.2%; 2024: +2.2% /
  +1.4%), because their own trailing coverage was below 77%: in 2024 their uncalibrated coverage really was 76.5% / 75.3%, and it ends at 80.9% / 80.1%. That 2024 WR widening is 0.2 point over
  the +2% cap written before the runs, so the pass criteria are missed by that margin; `qb_only` meets them exactly (every other band unchanged, QB numbers identical). The side that
  misses for quarterbacks flipped between years (below p10 0.142 vs above p90 0.113 in 2025; 0.106 vs 0.143 in 2024), so the tail shifts are partly chasing noise and the robust part is the width.
  The Winkler interval score does not get worse (flat in 2025, better in 2024).

