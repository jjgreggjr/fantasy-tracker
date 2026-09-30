# Phase 2 backtest: which model, and which data predicts

Walk-forward over every regular-season week of 2025: to score week N the model is refit on 2021-2024 plus 2025 weeks below N, so nothing at or after the target week is ever trained on (asserted before every fit and proven by perturbation in `model/tests/test_backtest.py`). Targets are PPR points. All numbers below are a pure function of `model/cache/features.parquet`, `model/tuned_params.json` and fixed seeds; a rerun reproduces them (`python -m model.explain` refits the winner and asserts bit-identical predictions).

## Summary

* **Winner: LightGBM.** The three libraries tie (pooled head-to-head RMSE LightGBM 5.711, XGBoost 5.706, CatBoost 5.710; neither XGBoost nor CatBoost differs from LightGBM beyond noise, both 95% intervals span zero), so the pre-declared tie-break picks LightGBM: one library for the point estimate, the quantiles and SHAP, and the fastest weekly refit (18 refits in about 20 seconds against 60 for XGBoost and 120 for CatBoost). One model with position as a feature; per-position models were not better (dRMSE 2025 +0.024; 2024 +0.017).
* **It beats the trailing-3 average clearly, but not by a lot.** MAE 4.02 vs 4.31 (-0.30, 95% CI [-0.39, -0.21]), RMSE 5.71 vs 6.43 (-0.72), Spearman 0.655 vs 0.602 (+0.053), head-to-head pick accuracy 77.3% vs 74.8% (+2.5%). Every interval excludes zero in every position for every metric. On start/sit-relevant pairs the pick accuracy is 64.1% vs 59.7% for the trailing average.
* **Against the same-week xFP oracle it closes 16% of the MAE gap, 31% of the RMSE gap, 20% of the Spearman gap and 20% of the pick-accuracy gap.** The oracle is post-game information (it sees the carries and targets the player actually got), so this is a ceiling for pre-game information, not a target; weekly fantasy scoring is mostly opportunity plus luck the model cannot see. Quarterbacks are the hardest position (Spearman 0.48 vs RB 0.75, WR 0.70, TE 0.69).
* **Quantiles are usable, slightly narrow.** Share of actuals at or below p10 / p50 / p90: 11.9% / 53.2% / 89.9% (targets 10 / 50 / 90); the p10-p90 band covers 78.1% (target 80%), QB 74.4%, TE 78.6%. The floor (p10) is too high by about two points and the median a little high; the ceiling is right.
* **What predicts (SHAP, out of sample):** by family lags 51%, adp 19%, role 11%, prev_season 6%, static 5%, vegas 3%. Top three features: `xfp_std_mean` (14%), `pts_ppr_std_mean` (11%), `adp_ppr_pos_rank` (9%). Season-to-date expected fantasy points (opportunity quality) and actual points are the core; the draft-market prior (ADP) and snap share come next; Vegas implied team total is the only game-environment input that matters (largest for quarterbacks); college production, combine numbers, injury designations, TD luck, defense-vs-position and rest/venue carry little (college 0.2% of total attribution, injury 0.6%, td_luck 0.9%, dvp 1.2%, context 1.7%).
* **Depth-chart-only rows (watch-list item 1):** they are 5.8% of all played 2025 rows and 1.9% of the head-to-head rows. Scored without them the model's head-to-head RMSE is 5.727 (vs 5.711 with) and the trailing average's 6.451 (vs 6.432): the margin over the trailing average is unchanged. Training without the depth-only rows changes nothing (row below).
* **Ablations (dRMSE variant minus primary, 2025 then 2024 replication; positive = variant worse):** training on stats-row players only +0.018 [+0.005, +0.029] (2024: +0.027 [+0.013, +0.041]) (worse both years: keep `y_played`); `season` -0.015 [-0.024, -0.006] (2024: +0.012 [-0.003, +0.027]); all observed weather -0.022 [-0.036, -0.007] (2024: +0.003 [-0.010, +0.016]), roof structure only -0.006 [-0.014, +0.003] (2024: +0.018 [+0.003, +0.032]), temperature and wind only -0.015 [-0.029, -0.003] (2024: +0.009 [-0.004, +0.023]) (the 2025 gains do not replicate: weather, even as a perfect forecast, does not reliably help); training without depth-chart-only rows +0.000 [-0.008, +0.008] (2024: +0.007 [-0.005, +0.019]); per-position models +0.024 [-0.001, +0.048] (2024: +0.017 [-0.009, +0.041]); without `week` -0.011 [-0.022, -0.000] (2024: +0.007 [-0.008, +0.022]).
* **Which families are load-bearing (dropping them hurts in both years; only lags and vegas have a 2025 interval that excludes zero, role and adp only in 2024):** lags +0.058 [+0.031, +0.085] (2024: +0.084 [+0.060, +0.107]); vegas +0.016 [+0.001, +0.030] (2024: +0.040 [+0.021, +0.057]); role +0.014 [-0.006, +0.034] (2024: +0.033 [+0.017, +0.051]); adp +0.007 [-0.008, +0.023] (2024: +0.025 [+0.007, +0.044]). The other families (prev_season, injury, static, college, dvp, td_luck, context) move RMSE by less than 0.02 in either direction and change sign between 2025 and 2024: one-at-a-time drops cannot separate them from noise, and the joint drops chosen from the 2025 table (`lean_A`, `lean_B`) also fail to replicate in 2024.
* **Pruning test (the honest one):** dropping the 39 registry columns that earned under 0.15% of out-of-sample SHAP on the 2024 walk-forward gives 2025 RMSE 5.696 vs 5.711 (dRMSE -0.015, CI [-0.027, -0.002]): a smaller model that is no worse.
* **The serving candidate for 2026 is that pruned set without ADP** (no preseason 2026 ADP snapshot exists: the fetch returned a 29-player in-season window, so 98.6% of 2026 rows are NA on ADP against 67.5% of 2025 rows, a train/serve skew): 2025 RMSE 5.719 (dRMSE +0.008 vs primary, CI [-0.005, +0.021]), Spearman 0.655, MAE 4.023, still ahead of the trailing average (6.432 RMSE, 0.602 Spearman).
* **Refit cadence (dRMSE of a staler model vs the weekly refit; positive = staleness costs):** every 4 weeks 2025 -0.007 [-0.015, +0.001]; 2024 +0.010 [+0.002, +0.021]; never inside the season 2025 +0.003 [-0.011, +0.015]; 2024 +0.028 [+0.008, +0.046]. Small and not consistent across years (nothing in 2025, a real but small cost in 2024), and a weekly refit is free.
* **Not tested and worth knowing:** the target is PPR only (league-scoring variants are a linear recombination that needs the component model in Phase 3); there is no platform-projection baseline (historical ESPN/Sleeper projections are not archived); our pipeline's `E_pts` blends platform projections (not archived for 2025) with trailing form and a DvP multiplier, so it cannot be reconstructed for 2025 and was not compared. `college_breakout_age` is always NA (needs multi-season CFBD data); 27 of 475 drafted skill players fail to match only because the fetch kept exact drafted names (nicknames) or the school is FCS.


## 1. What was scored

* Scored rows: 2025 regular season, players who appeared (`y_played == 1`: a stats row or offensive snaps): **6,755 rows**. The head-to-head set additionally requires at least one earlier game this season (the trailing average exists): **6,139 rows** (QB 601, RB 1,481, WR 2,532, TE 1,525). Every model and baseline is scored on identical rows in each table.
* Training rows: `y_played == 1` with the same `y_points_ppr` (a snap-only appearance is 0.0). Refit weekly; 25,460 rows at week 1 growing to 31,823 at week 18.
* Primary inputs: 114 columns (12 registered families minus `wx_*` and `season`, position as one-hots). Dropped as empty in 2021-2024: college_breakout_age.
* Predictors compared: the three libraries, the **trailing-3 average** (mean PPR of the last <=3 appearances this season) and **same-week xFP** (ffopportunity expected points for the game itself: post-game information, an oracle, not a fair bar).

## 2. The verdict table (head-to-head rows, all 2025 weeks)

Winner by the pre-declared rule (lowest pooled RMSE; a statistical tie goes to LightGBM): **LightGBM**.

**MAE (points, lower is better)**

|  | QB | RB | WR | TE | ALL |
|---|---|---|---|---|---|
| LightGBM | 6.18 | 4.24 | 3.99 | 3.00 | 4.02 |
| XGBoost | 6.18 | 4.23 | 3.99 | 3.00 | 4.02 |
| CatBoost | 6.14 | 4.23 | 4.02 | 2.96 | 4.01 |
| trailing-3 average | 6.89 | 4.54 | 4.26 | 3.17 | 4.31 |
| same-week xFP (oracle) | 4.22 | 2.71 | 2.49 | 1.65 | 2.50 |

**RMSE (points)**

|  | QB | RB | WR | TE | ALL |
|---|---|---|---|---|---|
| LightGBM | 7.79 | 6.07 | 5.55 | 4.51 | 5.71 |
| XGBoost | 7.77 | 6.05 | 5.56 | 4.51 | 5.71 |
| CatBoost | 7.73 | 6.06 | 5.58 | 4.50 | 5.71 |
| trailing-3 average | 9.01 | 6.79 | 6.23 | 5.03 | 6.43 |
| same-week xFP (oracle) | 5.67 | 4.51 | 3.95 | 2.92 | 4.07 |

**Spearman rank correlation within position-week (mean over weeks; the start/sit metric)**

|  | QB | RB | WR | TE | ALL |
|---|---|---|---|---|---|
| LightGBM | 0.478 | 0.753 | 0.695 | 0.694 | 0.655 |
| XGBoost | 0.478 | 0.757 | 0.694 | 0.700 | 0.657 |
| CatBoost | 0.493 | 0.755 | 0.694 | 0.701 | 0.661 |
| trailing-3 average | 0.410 | 0.714 | 0.631 | 0.653 | 0.602 |
| same-week xFP (oracle) | 0.768 | 0.914 | 0.878 | 0.905 | 0.866 |

**Head-to-head pick accuracy, all same-position pairs in a week (share of pairs where the higher prediction scored more; ties in the prediction earn 0.5)**

|  | QB | RB | WR | TE | ALL |
|---|---|---|---|---|---|
| LightGBM | 0.675 | 0.786 | 0.771 | 0.781 | 0.773 |
| XGBoost | 0.676 | 0.789 | 0.771 | 0.783 | 0.774 |
| CatBoost | 0.681 | 0.788 | 0.771 | 0.784 | 0.774 |
| trailing-3 average | 0.647 | 0.769 | 0.742 | 0.760 | 0.748 |
| same-week xFP (oracle) | 0.808 | 0.886 | 0.869 | 0.897 | 0.876 |

**Head-to-head pick accuracy on start/sit-relevant pairs only (both players in the top 14/30/40/14 QB/RB/WR/TE by trailing average that week)**

|  | QB | RB | WR | TE | ALL |
|---|---|---|---|---|---|
| LightGBM | 0.569 | 0.667 | 0.638 | 0.614 | 0.641 |
| XGBoost | 0.579 | 0.670 | 0.637 | 0.606 | 0.642 |
| CatBoost | 0.590 | 0.664 | 0.633 | 0.618 | 0.639 |
| trailing-3 average | 0.536 | 0.623 | 0.594 | 0.555 | 0.597 |
| same-week xFP (oracle) | 0.740 | 0.790 | 0.787 | 0.773 | 0.784 |

### Does the winner beat the trailing average, and how close is it to the oracle?

Paired week-blocked bootstrap (2,000 resamples of the 18 weeks), LightGBM minus trailing-3. Negative dMAE/dRMSE and positive dSpearman/dPick favour the model.

|  | dMAE | dMAE_lo | dMAE_hi | dRMSE | dRMSE_lo | dRMSE_hi | dSpearman | dSpearman_lo | dSpearman_hi | dPick | dPick_lo | dPick_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| QB | -0.713 | -1.164 | -0.354 | -1.223 | -1.740 | -0.823 | +0.067 | +0.029 | +0.107 | +0.028 | +0.015 | +0.042 |
| RB | -0.295 | -0.425 | -0.178 | -0.715 | -0.916 | -0.550 | +0.039 | +0.027 | +0.053 | +0.018 | +0.012 | +0.024 |
| WR | -0.272 | -0.422 | -0.138 | -0.676 | -0.895 | -0.484 | +0.064 | +0.053 | +0.077 | +0.029 | +0.023 | +0.036 |
| TE | -0.171 | -0.245 | -0.097 | -0.526 | -0.662 | -0.410 | +0.041 | +0.024 | +0.057 | +0.021 | +0.014 | +0.027 |
| ALL | -0.295 | -0.389 | -0.214 | -0.722 | -0.885 | -0.600 | +0.053 | +0.041 | +0.065 | +0.025 | +0.021 | +0.030 |

Share of the gap between trailing-3 and the xFP oracle that the model closes (0 = trailing average, 1 = oracle):

|  | QB | RB | WR | TE | ALL |
|---|---|---|---|---|---|
| MAE closed | 27% | 16% | 15% | 11% | 16% |
| RMSE closed | 37% | 31% | 30% | 25% | 31% |
| Spearman closed | 19% | 20% | 26% | 16% | 20% |
| pick closed | 18% | 15% | 23% | 15% | 20% |

Library head-to-head (pooled, LightGBM is the reference; a negative dRMSE means the row's library is better):

|  | dRMSE | lo | hi | dMAE | dSpearman | dPick |
|---|---|---|---|---|---|---|
| XGBoost | -0.004 | -0.021 | +0.014 | -0.002 | +0.002 | +0.001 |
| CatBoost | -0.001 | -0.012 | +0.012 | -0.004 | +0.006 | +0.001 |

## 3. All played rows, including week 1 and new players (no trailing average exists)

Trailing-3 is undefined for a player's first appearance of a season, so on this set it falls back to last season's points per game, then to the 2021-2024 position mean (a stronger baseline than the strict one above).

**MAE** (n = QB 681, RB 1,628, WR 2,777, TE 1,669)

|  | QB | RB | WR | TE | ALL |
|---|---|---|---|---|---|
| LightGBM | 6.11 | 4.14 | 3.95 | 2.95 | 3.97 |
| trailing-3, prior-season fallback | 7.04 | 4.50 | 4.30 | 3.14 | 4.34 |
| same-week xFP (oracle) | 4.09 | 2.61 | 2.44 | 1.59 | 2.44 |

**RMSE** (n = QB 681, RB 1,628, WR 2,777, TE 1,669)

|  | QB | RB | WR | TE | ALL |
|---|---|---|---|---|---|
| LightGBM | 7.72 | 5.92 | 5.50 | 4.42 | 5.63 |
| trailing-3, prior-season fallback | 9.12 | 6.65 | 6.21 | 4.95 | 6.40 |
| same-week xFP (oracle) | 5.58 | 4.38 | 3.89 | 2.84 | 4.00 |

**Spearman** (n = QB 681, RB 1,628, WR 2,777, TE 1,669)

|  | QB | RB | WR | TE | ALL |
|---|---|---|---|---|---|
| LightGBM | 0.533 | 0.759 | 0.695 | 0.693 | 0.670 |
| trailing-3, prior-season fallback | 0.423 | 0.705 | 0.621 | 0.640 | 0.597 |
| same-week xFP (oracle) | 0.799 | 0.918 | 0.881 | 0.907 | 0.876 |

**Pick accuracy** (n = QB 681, RB 1,628, WR 2,777, TE 1,669)

|  | QB | RB | WR | TE | ALL |
|---|---|---|---|---|---|
| LightGBM | 0.693 | 0.791 | 0.772 | 0.782 | 0.775 |
| trailing-3, prior-season fallback | 0.648 | 0.765 | 0.738 | 0.756 | 0.744 |
| same-week xFP (oracle) | 0.821 | 0.889 | 0.873 | 0.900 | 0.880 |

## 4. Scoring with and without the depth-chart-only rows (watch-list item 1)

`spine_depth_only == 1` rows were admitted to the row universe only by the 2025 daily depth-chart snapshot: they are 2,201 of 2025's rows against 549-686 a year before, and that population (new-to-team veterans, low volume) is under-represented in training. Head-to-head rows, with and without them:

|  | n | MAE | RMSE | spearman | pick_acc |
|---|---|---|---|---|---|
| LightGBM (with) | 6139 | 4.019 | 5.711 | 0.655 | 0.773 |
| LightGBM (without) | 6020 | 4.036 | 5.727 | 0.643 | 0.772 |
| trailing-3 average (with) | 6139 | 4.314 | 6.432 | 0.602 | 0.748 |
| trailing-3 average (without) | 6020 | 4.340 | 6.451 | 0.590 | 0.747 |
| same-week xFP (oracle) (with) | 6139 | 2.503 | 4.073 | 0.866 | 0.876 |
| same-week xFP (oracle) (without) | 6020 | 2.525 | 4.095 | 0.859 | 0.875 |

Depth-only rows in the head-to-head set: 119 of 6,139 (1.9%).

Every played row (week 1 and new players included; trailing-3 with the prior-season fallback): 394 of 6,755 (5.8%) are depth-only.

|  | n | MAE | RMSE | spearman | pick_acc |
|---|---|---|---|---|---|
| LightGBM (with) | 6755 | 3.967 | 5.631 | 0.670 | 0.775 |
| LightGBM (without) | 6361 | 4.027 | 5.703 | 0.644 | 0.772 |
| trailing-3, prior-season fallback (with) | 6755 | 4.339 | 6.399 | 0.597 | 0.744 |
| trailing-3, prior-season fallback (without) | 6361 | 4.350 | 6.424 | 0.589 | 0.746 |
| same-week xFP (oracle) (with) | 6755 | 2.436 | 3.997 | 0.876 | 0.880 |
| same-week xFP (oracle) (without) | 6361 | 2.509 | 4.071 | 0.858 | 0.876 |

## 5. Quantile models (LightGBM p10 / p50 / p90)

Independent models can cross on a row; crossed rows are reported and the scored quantiles are sorted per row. Targets: below_q10 0.10, below_q50 0.50, below_q90 0.90, coverage_10_90 0.80. All 2025 played rows.

|  | n | below_q10 | below_q50 | below_q90 | coverage_10_90 | mean_width | crossed_share |
|---|---|---|---|---|---|---|---|
| QB | 681 | 0.142 | 0.488 | 0.887 | 0.744 | 17.7 | 0.000 |
| RB | 1628 | 0.106 | 0.522 | 0.895 | 0.789 | 12.4 | 0.015 |
| WR | 2777 | 0.126 | 0.552 | 0.907 | 0.781 | 12.0 | 0.030 |
| TE | 1669 | 0.110 | 0.528 | 0.896 | 0.786 | 8.6 | 0.066 |
| ALL | 6755 | 0.119 | 0.532 | 0.899 | 0.781 | 11.8 | 0.032 |

The p50 model as a point estimate (median, so it under-predicts the mean of a right-skewed target; MAE-optimal, not RMSE-optimal), head-to-head rows:

|  | MAE | RMSE | spearman | pick_acc |
|---|---|---|---|---|
| QB | 6.166 | 7.947 | 0.458 | 0.668 |
| RB | 4.051 | 6.238 | 0.757 | 0.789 |
| WR | 3.731 | 5.654 | 0.694 | 0.771 |
| TE | 2.820 | 4.670 | 0.692 | 0.779 |
| ALL | 3.820 | 5.846 | 0.650 | 0.773 |

Interval coverage by week ranges 0.76 to 0.83 (mean 0.780); first four weeks 0.807, weeks 5-18 0.773.

## 6. Ablations (winner library, same rows, same weekly refit)

Each row changes one thing against the primary config and is scored on the same head-to-head rows. dRMSE is variant minus primary (positive = the variant is worse) with a 95% week-blocked bootstrap interval. The primary config excludes `wx_*` and `season`, trains on `y_played` rows, and uses one model with position as a feature. Every row is replicated on the 2024 walk-forward (train 2021-2023 plus 2024 weeks below N; the same code, one year earlier): a decision read off 2025 only counts if 2024 agrees. (Hyper-parameters were tuned on 2024, a mild advantage shared by every variant.)

Primary: RMSE 5.711, MAE 4.019, Spearman 0.655, pick 0.773 (2025, n = 6,139); 2024 replication set n = 6,025: RMSE 5.771, MAE 4.065, Spearman 0.663.

**Design choices**

| variant | dRMSE 2025 | 95% CI 2025 | dMAE 2025 | dSpearman 2025 | dRMSE 2024 | 95% CI 2024 | dSpearman 2024 | both years | dRMSE QB 2025 | dRMSE RB 2025 | dRMSE WR 2025 | dRMSE TE 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stats_row_target | +0.018 | [+0.005, +0.029] | +0.124 | +0.001 | +0.027 | [+0.013, +0.041] | -0.002 | variant worse in both years | +0.007 | -0.005 | +0.027 | +0.036 |
| with_season | -0.015 | [-0.024, -0.006] | -0.015 | +0.004 | +0.012 | [-0.003, +0.027] | -0.003 | mixed (sign flips) | -0.015 | -0.038 | -0.010 | +0.002 |
| with_weather | -0.022 | [-0.036, -0.007] | -0.013 | +0.002 | +0.003 | [-0.010, +0.016] | -0.001 | mixed (sign flips) | -0.028 | -0.025 | -0.021 | -0.015 |
| with_roof_structure | -0.006 | [-0.014, +0.003] | -0.004 | +0.001 | +0.018 | [+0.003, +0.032] | -0.002 | mixed (sign flips) | +0.012 | -0.026 | -0.006 | +0.011 |
| with_temp_wind | -0.015 | [-0.029, -0.003] | -0.006 | +0.001 | +0.009 | [-0.004, +0.023] | -0.001 | mixed (sign flips) | +0.002 | -0.034 | -0.016 | +0.000 |
| train_no_depth_only | +0.000 | [-0.008, +0.008] | +0.001 | -0.001 | +0.007 | [-0.005, +0.019] | -0.003 | no detectable effect | +0.023 | -0.007 | -0.004 | +0.002 |
| per_position_models | +0.024 | [-0.001, +0.048] | +0.003 | -0.002 | +0.017 | [-0.009, +0.041] | -0.007 | no detectable effect | -0.009 | +0.032 | +0.028 | +0.028 |
| drop_week | -0.011 | [-0.022, -0.000] | -0.008 | +0.000 | +0.007 | [-0.008, +0.022] | -0.001 | mixed (sign flips) | +0.001 | -0.039 | -0.006 | +0.008 |

`with_weather` adds all four `wx_*` columns; `with_roof_structure` only `wx_roof_obs` / `wx_indoor_obs` (fixed dome, retractable roof or open air: a stadium attribute that IS known before kickoff); `with_temp_wind` only the observed temperature and wind (post-game values, impossible at prediction time).

**Dropping one feature family at a time** (sorted by 2025 dRMSE)

| variant | dRMSE 2025 | 95% CI 2025 | dMAE 2025 | dSpearman 2025 | dRMSE 2024 | 95% CI 2024 | dSpearman 2024 | both years | dRMSE QB 2025 | dRMSE RB 2025 | dRMSE WR 2025 | dRMSE TE 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| drop_lags | +0.058 | [+0.031, +0.085] | +0.062 | -0.004 | +0.084 | [+0.060, +0.107] | -0.015 | family earns its place | +0.027 | +0.069 | +0.064 | +0.052 |
| drop_vegas | +0.016 | [+0.001, +0.030] | +0.011 | -0.007 | +0.040 | [+0.021, +0.057] | -0.008 | family earns its place | +0.092 | -0.004 | +0.011 | +0.002 |
| drop_role | +0.014 | [-0.006, +0.034] | +0.015 | -0.003 | +0.033 | [+0.017, +0.051] | -0.007 | family earns its place | +0.027 | +0.011 | +0.018 | +0.000 |
| drop_adp | +0.007 | [-0.008, +0.023] | -0.001 | +0.000 | +0.025 | [+0.007, +0.044] | -0.004 | family earns its place | +0.020 | -0.000 | +0.005 | +0.013 |
| drop_prev_season | -0.001 | [-0.014, +0.011] | +0.009 | +0.000 | +0.001 | [-0.009, +0.011] | -0.001 | no detectable effect | -0.014 | -0.032 | +0.017 | +0.009 |
| drop_injury | -0.008 | [-0.020, +0.003] | -0.004 | -0.001 | +0.009 | [-0.001, +0.020] | -0.002 | no detectable effect | +0.012 | -0.029 | -0.007 | +0.002 |
| drop_static | -0.009 | [-0.019, +0.000] | -0.009 | +0.001 | +0.018 | [+0.005, +0.030] | -0.003 | mixed (sign flips) | -0.002 | -0.029 | -0.007 | +0.008 |
| drop_college | -0.011 | [-0.019, -0.002] | -0.006 | +0.001 | +0.007 | [-0.009, +0.020] | -0.002 | mixed (sign flips) | -0.004 | -0.034 | -0.009 | +0.009 |
| drop_dvp | -0.013 | [-0.022, -0.004] | -0.007 | +0.002 | +0.005 | [-0.007, +0.019] | -0.003 | mixed (sign flips) | -0.003 | -0.034 | -0.014 | +0.011 |
| drop_td_luck | -0.013 | [-0.025, -0.003] | -0.016 | +0.002 | +0.011 | [-0.002, +0.024] | -0.002 | mixed (sign flips) | -0.019 | -0.020 | -0.010 | -0.008 |
| drop_context | -0.017 | [-0.030, -0.003] | -0.010 | +0.004 | +0.022 | [+0.007, +0.038] | -0.004 | mixed (sign flips) | -0.027 | -0.029 | -0.016 | +0.002 |

**Joint drops** (chosen after reading the one-at-a-time 2025 table, so only the 2024 column is out-of-time evidence): `lean_A` drops context, td_luck, dvp, college; `lean_B` drops context, td_luck, dvp, college, static, injury

| variant | dRMSE 2025 | 95% CI 2025 | dMAE 2025 | dSpearman 2025 | dRMSE 2024 | 95% CI 2024 | dSpearman 2024 | both years | dRMSE QB 2025 | dRMSE RB 2025 | dRMSE WR 2025 | dRMSE TE 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lean_A | -0.019 | [-0.033, -0.007] | -0.013 | +0.004 | +0.011 | [-0.005, +0.027] | -0.002 | mixed (sign flips) | -0.032 | -0.027 | -0.018 | -0.004 |
| lean_B | -0.016 | [-0.030, -0.002] | -0.012 | +0.002 | +0.018 | [+0.001, +0.037] | -0.004 | mixed (sign flips) | -0.004 | -0.037 | -0.017 | +0.002 |

### Where the model helps

|  | n | model RMSE | trailing RMSE | oracle RMSE | model Spearman | trailing Spearman |
|---|---|---|---|---|---|---|
| weeks 2-4 (weeks 1 excluded: no trailing average) | 1108 | 5.564 | 6.638 | 4.075 | 0.634 | 0.556 |
| weeks 5-12 | 2741 | 5.737 | 6.402 | 4.164 | 0.664 | 0.624 |
| weeks 13-18 | 2290 | 5.749 | 6.368 | 3.960 | 0.654 | 0.596 |
| rookies | 992 | 5.248 | 5.755 | 3.881 | 0.651 | 0.578 |
| veterans | 5147 | 5.795 | 6.555 | 4.109 | 0.651 | 0.594 |

### How often to refit

Weekly refit (the primary) against a model refit every 4 weeks and against one that is never refit inside the season (trained on the earlier seasons only, week 1's fit predicts all 18 weeks). Same rows; dRMSE is stale minus weekly (positive = staleness costs accuracy).

| cadence | dRMSE 2025 | 95% CI 2025 | dSpearman 2025 | dRMSE 2024 | 95% CI 2024 | dSpearman 2024 |
|---|---|---|---|---|---|---|
| refit_every_4 | -0.007 | [-0.015, +0.001] | +0.001 | +0.010 | [+0.002, +0.021] | +0.000 |
| refit_never | +0.003 | [-0.011, +0.015] | +0.001 | +0.028 | [+0.008, +0.046] | -0.005 |

## 7. Which data actually predicts (SHAP, LightGBM, out-of-sample 2025)

Mean |SHAP| in points over the 2025 played rows, each week's contributions from that week's walk-forward model. `direction` is the rank correlation between the feature's value and its SHAP contribution across that position's rows (+ = higher value raises the prediction).

**By feature family**

|  | mean|SHAP| | share |
|---|---|---|
| lags | 3.755 | 50.6% |
| adp | 1.390 | 18.7% |
| role | 0.782 | 10.5% |
| prev_season | 0.480 | 6.5% |
| static | 0.354 | 4.8% |
| vegas | 0.221 | 3.0% |
| context | 0.129 | 1.7% |
| dvp | 0.092 | 1.2% |
| position | 0.092 | 1.2% |
| td_luck | 0.068 | 0.9% |
| injury | 0.041 | 0.6% |
| college | 0.017 | 0.2% |

**Top 15 overall**

| feature | family | mean|SHAP| | share |
|---|---|---|---|
| xfp_std_mean | lags | 1.026 | 13.8% |
| pts_ppr_std_mean | lags | 0.815 | 11.0% |
| adp_ppr_pos_rank | adp | 0.664 | 8.9% |
| role_score | role | 0.436 | 5.9% |
| snap_pct_l1 | lags | 0.402 | 5.4% |
| adp_std | adp | 0.365 | 4.9% |
| xfp_l1 | lags | 0.240 | 3.2% |
| adp_std_pos_rank | adp | 0.218 | 2.9% |
| prev_season_ppg | prev_season | 0.206 | 2.8% |
| pts_ppr_l1 | lags | 0.203 | 2.7% |
| adp_ppr | adp | 0.143 | 1.9% |
| implied_team_total | vegas | 0.142 | 1.9% |
| tm_out_group_opps | role | 0.124 | 1.7% |
| opps_std_mean | lags | 0.108 | 1.5% |
| prev_season_xfp_pg | prev_season | 0.104 | 1.4% |

**QB: top 15 features**

| feature | family | mean|SHAP| | share | direction | reads as |
|---|---|---|---|---|---|
| xfp_std_mean | lags | 1.598 | 14.9% | +0.92 | season-to-date expected fantasy points per game (opportunity quality): the best single predictor |
| pts_ppr_std_mean | lags | 1.241 | 11.6% | +0.97 | season-to-date actual PPR per game; tracks xFP, the pair together is the 'true talent' estimate |
| adp_ppr_pos_rank | adp | 0.854 | 8.0% | -0.78 | preseason market rank within position (PPR); a lower rank number is a better player, NA = not in the drafted-player list |
| role_score | role | 0.659 | 6.1% | +0.57 | trailing-3 offensive snap share (last season's if none): how much he is on the field |
| snap_pct_l1 | lags | 0.629 | 5.9% | +0.69 | snap share in his last game: usage momentum |
| pos_QB | position | 0.384 | 3.6% |  | position offset: quarterbacks score more than the other three groups |
| adp_std | adp | 0.343 | 3.2% | -0.89 | preseason overall pick number (standard scoring): draft-market prior, decays as games accumulate |
| xfp_l1 | lags | 0.336 | 3.1% | +0.56 | last game's expected points: recent role change, weighted well below the season mean |
| prev_season_ppg | prev_season | 0.312 | 2.9% | +0.86 | last season's PPR per game: early-season anchor that fades as this season's games arrive |
| pts_ppr_l1 | lags | 0.301 | 2.8% | +0.90 | last game's actual points: recent form (noisy, small weight) |
| adp_std_pos_rank | adp | 0.295 | 2.8% | -0.68 | preseason market rank within position (standard scoring): same prior, second vote |
| implied_team_total | vegas | 0.280 | 2.6% | +0.86 | Vegas implied team points: game script and scoring chances |
| prev_season_xfp_pg | prev_season | 0.184 | 1.7% | +0.77 | last season's expected points per game: same anchor, opportunity-based |
| pass_att_l1 | lags | 0.166 | 1.5% | +0.06 | QB attempts in his last game: weak, mostly game-script noise |
| is_home | context | 0.135 | 1.3% | +0.87 | home game (a small, consistent QB edge) |

**RB: top 15 features**

| feature | family | mean|SHAP| | share | direction | reads as |
|---|---|---|---|---|---|
| xfp_std_mean | lags | 1.012 | 13.2% | +0.98 | season-to-date expected fantasy points per game (opportunity quality): the best single predictor |
| pts_ppr_std_mean | lags | 0.800 | 10.4% | +0.97 | season-to-date actual PPR per game; tracks xFP, the pair together is the 'true talent' estimate |
| adp_ppr_pos_rank | adp | 0.705 | 9.2% | -0.92 | preseason market rank within position (PPR); a lower rank number is a better player, NA = not in the drafted-player list |
| adp_std | adp | 0.415 | 5.4% | -0.88 | preseason overall pick number (standard scoring): draft-market prior, decays as games accumulate |
| snap_pct_l1 | lags | 0.402 | 5.2% | +0.85 | snap share in his last game: usage momentum |
| role_score | role | 0.392 | 5.1% | +0.91 | trailing-3 offensive snap share (last season's if none): how much he is on the field |
| adp_std_pos_rank | adp | 0.237 | 3.1% | -0.83 | preseason market rank within position (standard scoring): same prior, second vote |
| xfp_l1 | lags | 0.233 | 3.0% | +0.85 | last game's expected points: recent role change, weighted well below the season mean |
| pts_ppr_l1 | lags | 0.230 | 3.0% | +0.87 | last game's actual points: recent form (noisy, small weight) |
| prev_season_ppg | prev_season | 0.220 | 2.9% | +0.80 | last season's PPR per game: early-season anchor that fades as this season's games arrive |
| adp_ppr | adp | 0.176 | 2.3% | -0.61 | preseason overall pick number (PPR): draft-market prior |
| carry_share_l1 | lags | 0.165 | 2.1% | +0.91 | share of the team's carries in his last game (RB workload) |
| implied_team_total | vegas | 0.161 | 2.1% | +0.92 | Vegas implied team points: game script and scoring chances |
| tm_out_group_opps | role | 0.150 | 2.0% | +0.49 | trailing carries + targets vacated by same-position teammates ruled Out/Doubtful: an injury-driven role expansion |
| opps_std_mean | lags | 0.119 | 1.6% | +0.19 | season-to-date carries + targets per game: workload |

**WR: top 15 features**

| feature | family | mean|SHAP| | share | direction | reads as |
|---|---|---|---|---|---|
| xfp_std_mean | lags | 0.954 | 13.6% | +0.97 | season-to-date expected fantasy points per game (opportunity quality): the best single predictor |
| pts_ppr_std_mean | lags | 0.764 | 10.9% | +0.97 | season-to-date actual PPR per game; tracks xFP, the pair together is the 'true talent' estimate |
| adp_ppr_pos_rank | adp | 0.610 | 8.7% | -0.95 | preseason market rank within position (PPR); a lower rank number is a better player, NA = not in the drafted-player list |
| role_score | role | 0.438 | 6.3% | +0.96 | trailing-3 offensive snap share (last season's if none): how much he is on the field |
| snap_pct_l1 | lags | 0.390 | 5.6% | +0.96 | snap share in his last game: usage momentum |
| adp_std | adp | 0.380 | 5.4% | -0.84 | preseason overall pick number (standard scoring): draft-market prior, decays as games accumulate |
| xfp_l1 | lags | 0.236 | 3.4% | +0.89 | last game's expected points: recent role change, weighted well below the season mean |
| adp_std_pos_rank | adp | 0.195 | 2.8% | -0.87 | preseason market rank within position (standard scoring): same prior, second vote |
| prev_season_ppg | prev_season | 0.188 | 2.7% | +0.86 | last season's PPR per game: early-season anchor that fades as this season's games arrive |
| pts_ppr_l1 | lags | 0.186 | 2.7% | +0.90 | last game's actual points: recent form (noisy, small weight) |
| adp_ppr | adp | 0.148 | 2.1% | -0.66 | preseason overall pick number (PPR): draft-market prior |
| tm_out_group_opps | role | 0.137 | 2.0% | +0.66 | trailing carries + targets vacated by same-position teammates ruled Out/Doubtful: an injury-driven role expansion |
| implied_team_total | vegas | 0.124 | 1.8% | +0.93 | Vegas implied team points: game script and scoring chances |
| opps_std_mean | lags | 0.110 | 1.6% | +0.71 | season-to-date carries + targets per game: workload |
| pts_ppr_l2 | lags | 0.105 | 1.5% | +0.85 | two games ago actual points: recent form |

**TE: top 15 features**

| feature | family | mean|SHAP| | share | direction | reads as |
|---|---|---|---|---|---|
| xfp_std_mean | lags | 0.927 | 14.2% | +0.94 | season-to-date expected fantasy points per game (opportunity quality): the best single predictor |
| pts_ppr_std_mean | lags | 0.740 | 11.3% | +0.93 | season-to-date actual PPR per game; tracks xFP, the pair together is the 'true talent' estimate |
| adp_ppr_pos_rank | adp | 0.635 | 9.7% | -0.49 | preseason market rank within position (PPR); a lower rank number is a better player, NA = not in the drafted-player list |
| role_score | role | 0.384 | 5.9% | +0.96 | trailing-3 offensive snap share (last season's if none): how much he is on the field |
| snap_pct_l1 | lags | 0.331 | 5.1% | +0.97 | snap share in his last game: usage momentum |
| adp_std | adp | 0.298 | 4.6% | -0.78 | preseason overall pick number (standard scoring): draft-market prior, decays as games accumulate |
| xfp_l1 | lags | 0.217 | 3.3% | +0.90 | last game's expected points: recent role change, weighted well below the season mean |
| adp_std_pos_rank | adp | 0.208 | 3.2% | -0.62 | preseason market rank within position (standard scoring): same prior, second vote |
| prev_season_ppg | prev_season | 0.178 | 2.7% | +0.49 | last season's PPR per game: early-season anchor that fades as this season's games arrive |
| pts_ppr_l1 | lags | 0.166 | 2.5% | +0.87 | last game's actual points: recent form (noisy, small weight) |
| opps_std_mean | lags | 0.110 | 1.7% | +0.88 | season-to-date carries + targets per game: workload |
| adp_ppr | adp | 0.109 | 1.7% | -0.36 | preseason overall pick number (PPR): draft-market prior |
| weight_lb | static | 0.107 | 1.6% | -0.57 | heavier tight ends are blockers and score less |
| prev_season_xfp_pg | prev_season | 0.107 | 1.6% | +0.74 | last season's expected points per game: same anchor, opportunity-based |
| pts_ppr_l2 | lags | 0.104 | 1.6% | +0.88 | two games ago actual points: recent form |

**Features that earn nothing** (43 of 114: under 0.15% of total attribution overall and under 0.3% in every position), by family:

* college (12): `college_power_conf`, `college_pass_att_pg`, `college_pass_ypa`, `college_pass_td_rate`, `college_car_pg`, `college_dominator`, `college_rush_share`, `college_rec_td_share`, `college_rec_market_share`, `college_ypc`, `college_ypr`, `college_rec_pg`
* context (4): `neutral_site`, `div_game`, `rest_days`, `week`
* injury (3): `inj_report_ord`, `inj_listed_l4wk`, `inj_practice_ord`
* lags (10): `targets_l3`, `pass_att_l2`, `pass_att_l3`, `lag3_weeks_ago`, `targets_l2`, `targets_l1`, `carries_l3`, `lag2_weeks_ago`, `snap_pct_l3`, `opps_l3`
* prev_season (1): `prev_season_games`
* role (2): `tm_q_group_n`, `tm_out_group_n`
* static (8): `is_undrafted`, `is_rookie`, `height_in`, `exp_years`, `combine_shuttle`, `career_games_prior`, `combine_forty`, `combine_broad_jump`
* td_luck (2): `td_luck_rec_std`, `td_luck_rush_std`
* vegas (1): `implied_opp_total`

**Pruning test.** The list above comes from 2025 SHAP, so testing it on 2025 would be circular; what was tested is a list chosen on 2024: every registry column under 0.15% of out-of-sample SHAP on the **2024** walk-forward (39 columns: `career_games_prior`, `carries_l3`, `college_car_pg`, `college_dominator`, `college_pass_att_pg`, `college_pass_td_rate`, `college_pass_ypa`, `college_power_conf`, `college_rec_market_share`, `college_rec_pg`, `college_rec_td_share`, `college_rush_share`, `college_ypc`, `college_ypr`, `combine_broad_jump`, `combine_forty`, `combine_shuttle`, `div_game`, `draft_round`, `exp_years`, `games_std`, `height_in`, `implied_opp_total`, `inj_listed_l4wk`, `inj_practice_status`, `inj_report_status`, `is_rookie`, `is_undrafted`, `lag2_weeks_ago`, `lag3_weeks_ago`, `neutral_site`, `pass_att_l2`, `pass_att_l3`, `targets_l1`, `targets_l2`, `targets_l3`, `tm_out_group_n`, `tm_q_group_n`, `week`); the model is then walked forward through 2025 without them. Result: RMSE 5.696 vs 5.711 primary (dRMSE -0.015, 95% CI [-0.027, -0.002]), Spearman 0.659 vs 0.655 (dSpearman +0.004), MAE 4.010 vs 4.019, with 39 fewer columns.

## 8. Three worked predictions

Each is a held-out 2025 prediction. `base` is the model's average prediction; contributions add up to the prediction exactly. A blank `value` means the feature is NA for that player (no ADP entry, for example).

### stud: Jahmyr Gibbs (RB, DET vs DAL, week 14)

Prediction **23.4** PPR (band p10 8.4 / p50 19.5 / p90 34.3); actual **37.0**; his last three games 11.6, 55.4, 19.6. Base 6.96.

| feature | value | contribution |
|---|---|---|
| pts_ppr_std_mean | 22.55 | +2.49 |
| xfp_std_mean | 16.97 | +2.41 |
| adp_ppr_pos_rank | 3.00 | +2.09 |
| adp_std | 4.10 | +1.28 |
| implied_team_total | 29.50 | +0.81 |
| adp_std_pos_rank | 4.00 | +0.78 |
| prev_season_ppg | 21.35 | +0.74 |
| role_score | 0.72 | +0.52 |
| all other features |  | +5.32 |

By family: lags +7.84, adp +4.55, prev_season +1.15, vegas +1.13, role +0.97, static +0.72, dvp -0.19, injury +0.17, context +0.12, td_luck -0.03

### volatile WR: Tre Tucker (WR, LV vs IND, week 5)

Prediction **11.2** PPR (band p10 2.6 / p50 9.5 / p90 17.3); actual **11.1**; his last three games 4.2, 40.9, 4.9. Base 6.99.

| feature | value | contribution |
|---|---|---|
| pts_ppr_std_mean | 15.85 | +1.39 |
| xfp_std_mean | 10.55 | +1.20 |
| role_score | 0.94 | +1.00 |
| snap_pct_l1 | 0.98 | +0.87 |
| adp_ppr_pos_rank |  | -0.50 |
| pts_ppr_l2 | 40.90 | +0.26 |
| adp_std_pos_rank |  | -0.21 |
| adp_std |  | -0.19 |
| all other features |  | +0.37 |

By family: lags +4.51, adp -1.07, role +0.84, prev_season -0.15, vegas -0.07, static +0.05, dvp +0.05, injury +0.03, td_luck +0.02, position -0.01, context -0.01

### rookie: Emeka Egbuka (WR, TB vs SF, week 6)

Prediction **18.2** PPR (band p10 4.4 / p50 15.7 / p90 30.3); actual **4.4**; his last three games 31.3, 20.1, 14.5. Base 6.99.

| feature | value | contribution |
|---|---|---|
| pts_ppr_std_mean | 20.48 | +2.10 |
| xfp_std_mean | 13.95 | +1.39 |
| tm_out_group_opps | 16.33 | +1.34 |
| tm_out_targets_all | 21.33 | +0.76 |
| adp_ppr_pos_rank | 41.00 | +0.71 |
| pts_ppr_l1 | 31.30 | +0.69 |
| snap_pct_l1 | 0.85 | +0.48 |
| role_score | 0.79 | +0.40 |
| all other features |  | +3.34 |

By family: lags +6.00, role +2.71, adp +1.46, static +0.49, td_luck +0.27, vegas +0.20, college +0.11, prev_season -0.07, dvp +0.04, injury +0.02, context -0.02

## 9. College priors: match rate and coverage

Drafted QB/RB/WR/TE (with a gsis id) per class and what happened to each; the matcher can only shrink coverage (name + college + position, or a position switch on college agreement; FCS schools have no team totals and get NA; see `model/college.py`).

| draft class | drafted_skill | matched | no_draft_college | no_name_match | college_disagree | ambiguous | shared_cfbd_id | non_fbs | match_rate | tier2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2021 | 75 | 68 | 0 | 7 | 0 | 0 | 0 | 0 | 0.907 | 0 |
| 2022 | 79 | 69 | 0 | 7 | 0 | 0 | 0 | 3 | 0.873 | 1 |
| 2023 | 80 | 76 | 0 | 1 | 0 | 0 | 0 | 3 | 0.950 | 0 |
| 2024 | 77 | 71 | 0 | 3 | 0 | 0 | 0 | 3 | 0.922 | 0 |
| 2025 | 85 | 76 | 0 | 6 | 0 | 0 | 0 | 3 | 0.894 | 0 |
| 2026 | 79 | 74 | 0 | 3 | 0 | 0 | 0 | 2 | 0.937 | 0 |

All classes: 434 of 475 drafted skill players matched (91.4%). The unmatched are FCS schools (`non_fbs`), players whose CFBD name differs from the draft name ("Cam Ward" is "Cameron Ward": the fetch kept only exact drafted names) and players with no final college season in the file (2020 opt-outs).

Coverage on rookie rows (`is_rookie == 1`) of the matrix. Undrafted rookies cannot match (no draft record, and the fetch kept drafted names only), so the ceiling is the drafted share of rookie rows. No non-rookie row carries a college value by construction.

| season | rookie rows | with college | share | played rows | played with college | played share | drafted share of played | drafted-only coverage |
|---|---|---|---|---|---|---|---|---|
| 2021 | 1219 | 933 | 0.765 | 828 | 666 | 0.804 | 0.900 | 0.894 |
| 2022 | 1408 | 937 | 0.665 | 977 | 665 | 0.681 | 0.793 | 0.858 |
| 2023 | 1487 | 1067 | 0.718 | 1058 | 797 | 0.753 | 0.803 | 0.938 |
| 2024 | 1314 | 973 | 0.740 | 930 | 726 | 0.781 | 0.867 | 0.901 |
| 2025 | 1788 | 1204 | 0.673 | 1098 | 832 | 0.758 | 0.833 | 0.909 |
| 2026 | 296 | 215 | 0.726 | 144 | 118 | 0.819 | 0.896 | 0.915 |

`college_breakout_age` is always NA (it needs several college seasons; the fetch pulls the last one) and is dropped as an empty column. Non-rookie rows with a college value: 0.

## 10. Tuning and chosen parameters

Seeded random search, 12 configurations per library (config 0 is the untuned default), train 2021-2023, validate on 2024 with early stopping (19,053 train / 6,407 validation rows). 2025 is never seen. Final tree count = best early-stopped count x 1.1, fixed for all weekly refits. Seed 20260929, 4 threads, squared-error objective.

* LightGBM: `{"num_leaves": 15, "min_child_samples": 200, "colsample_bytree": 0.4, "subsample": 1.0, "reg_lambda": 20.0, "learning_rate": 0.05, "n_estimators": 148}` (best validation RMSE 5.7695)
* XGBoost: `{"max_depth": 6, "min_child_weight": 20, "colsample_bytree": 1.0, "subsample": 0.8, "reg_lambda": 50.0, "learning_rate": 0.02, "n_estimators": 282}` (best validation RMSE 5.7784)
* CatBoost: `{"depth": 8, "l2_leaf_reg": 10.0, "learning_rate": 0.05, "rsm": 0.8, "iterations": 354}` (best validation RMSE 5.7687)
* LightGBM quantiles: same shape, tree count from the pinball loss per alpha: p10 n_estimators=631, p50 n_estimators=160, p90 n_estimators=138

<details><summary>LightGBM search log</summary>

| config | valid_rmse | trees | learning_rate | num_leaves | min_child_samples | colsample_bytree | subsample | reg_lambda |
|---|---|---|---|---|---|---|---|---|
| 1 | 5.769 | 135 | 0.05 | 15 | 200 | 0.4 | 1 | 20 |
| 8 | 5.77 | 303 | 0.02 | 31 | 50 | 0.4 | 0.6 | 50 |
| 0 | 5.785 | 211 | 0.03 | 15 | 50 | 0.7 | 0.8 | 5 |
| 5 | 5.787 | 412 | 0.02 | 7 | 100 | 0.4 | 0.6 | 5 |
| 10 | 5.79 | 246 | 0.02 | 15 | 100 | 0.8 | 0.6 | 0 |
| 3 | 5.791 | 223 | 0.05 | 7 | 200 | 0.6 | 1 | 20 |
| 9 | 5.795 | 254 | 0.02 | 31 | 200 | 1 | 1 | 20 |
| 4 | 5.796 | 207 | 0.03 | 15 | 100 | 0.8 | 1 | 20 |
| 6 | 5.797 | 153 | 0.03 | 15 | 50 | 0.8 | 0.8 | 0 |
| 7 | 5.803 | 160 | 0.05 | 7 | 50 | 0.6 | 1 | 20 |
| 2 | 5.806 | 123 | 0.03 | 63 | 100 | 0.6 | 1 | 20 |
| 11 | 5.809 | 99 | 0.05 | 63 | 20 | 0.8 | 0.8 | 50 |

</details>

<details><summary>XGBoost search log</summary>

| config | valid_rmse | trees | learning_rate | max_depth | min_child_weight | colsample_bytree | subsample | reg_lambda |
|---|---|---|---|---|---|---|---|---|
| 5 | 5.778 | 256 | 0.02 | 6 | 20 | 1 | 0.8 | 50 |
| 9 | 5.782 | 523 | 0.02 | 3 | 20 | 0.6 | 0.6 | 50 |
| 1 | 5.784 | 323 | 0.02 | 4 | 5 | 0.6 | 0.8 | 5 |
| 11 | 5.785 | 158 | 0.03 | 6 | 20 | 0.6 | 0.6 | 1 |
| 2 | 5.786 | 246 | 0.02 | 6 | 50 | 0.8 | 1 | 20 |
| 4 | 5.786 | 247 | 0.02 | 8 | 50 | 0.4 | 1 | 20 |
| 6 | 5.788 | 138 | 0.05 | 3 | 50 | 0.8 | 0.6 | 5 |
| 10 | 5.79 | 108 | 0.05 | 4 | 5 | 0.8 | 0.8 | 50 |
| 0 | 5.792 | 259 | 0.03 | 4 | 10 | 0.7 | 0.8 | 5 |
| 7 | 5.793 | 366 | 0.03 | 3 | 5 | 1 | 1 | 20 |
| 3 | 5.799 | 178 | 0.05 | 3 | 5 | 1 | 1 | 5 |
| 8 | 5.802 | 192 | 0.05 | 3 | 1 | 0.8 | 1 | 5 |

</details>

<details><summary>CatBoost search log</summary>

| config | valid_rmse | trees | learning_rate | depth | l2_leaf_reg | rsm |
|---|---|---|---|---|---|---|
| 2 | 5.769 | 322 | 0.05 | 8 | 10 | 0.8 |
| 3 | 5.772 | 286 | 0.05 | 6 | 3 | 0.5 |
| 5 | 5.774 | 217 | 0.05 | 6 | 1 | 0.5 |
| 9 | 5.775 | 280 | 0.03 | 8 | 1 | 0.8 |
| 8 | 5.776 | 392 | 0.03 | 6 | 3 | 0.5 |
| 10 | 5.777 | 94 | 0.08 | 8 | 3 | 1 |
| 6 | 5.782 | 205 | 0.05 | 6 | 3 | 1 |
| 0 | 5.786 | 180 | 0.05 | 6 | 5 |  |
| 7 | 5.791 | 497 | 0.03 | 4 | 1 | 1 |
| 11 | 5.793 | 70 | 0.08 | 8 | 1 | 1 |
| 4 | 5.801 | 260 | 0.05 | 4 | 3 | 1 |
| 1 | 5.809 | 436 | 0.05 | 3 | 3 | 0.8 |

</details>
