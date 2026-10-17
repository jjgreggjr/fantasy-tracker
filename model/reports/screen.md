# Predictability screen

Run 2026-10-17 (`python -m model.screen`), 9.2 min. Matrix key `d8c17dab9e330a3f` (49,859 rows x 235 columns). Screened rows: 24,102 head-to-head rows of 2022-2025 (played, with an earlier game that season); 2026 null rates on 1,469 2026 head-to-head rows (weeks 2-5).

`rho_resid` / `rho_y`: Spearman with the frozen LightGBM primary's out-of-sample residual / with actual PPR, within (season, week, position), mean over groups; `t` = that mean over its standard error. `gain` / `dRMSE` are **screen-grade**: one fit per year on 2021..y-1 with the column added alone (tuned parameters, no retree, no weekly refit), gain share of that column and RMSE minus the primary's same single fit on year y's head-to-head rows. Verdicts: PROMISING (top-decile |rho_resid|, dRMSE < 0 both years, 2026 null < 50%), UNCONFIRMED (top decile, single fits not both better), REDUNDANT (|rho_y| >= 0.1), NOISE, SKEW (2026 null >= 50%), DEAD (no information 2021-2024). The top 40 by |rho_resid| (plus every PROMISING) are shown, grouped by family; every number is in `model/cache/screen/d8c17dab9e330a3f.parquet`.

| family | column | rho_resid | t | rho_y | gain 2024 | gain 2025 | dRMSE 2024 | dRMSE 2025 | null | null 2026 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pbp_part | `part_run_snap_share_std` | -0.060 | -7.1 | +0.497 | 0.3% | 0.2% | -0.0029 | +0.0040 | 0.0% | 100.0% | SKEW |
| pbp_part | `part_run_snap_share_t3` | -0.055 | -6.2 | +0.497 | 0.4% | 0.3% | -0.0132 | +0.0076 | 0.0% | 100.0% | SKEW |
| pbp_part | `part_prev_pass_snap_share` | -0.050 | -5.5 | +0.467 | 0.1% | 0.1% | -0.0025 | +0.0098 | 19.0% | 14.4% | UNCONFIRMED |
| pbp_part | `part_pass_snap_share_std` | -0.048 | -5.6 | +0.575 | 0.8% | 0.4% | -0.0029 | +0.0055 | 0.0% | 100.0% | SKEW |
| pbp_part | `part_rz_snap_share_std` | -0.048 | -5.4 | +0.551 | 0.3% | 0.2% | -0.0040 | +0.0071 | 0.2% | 100.0% | SKEW |
| pbp_part | `part_run_snap_share_l1` | -0.048 | -5.7 | +0.482 | 0.6% | 0.4% | +0.0003 | +0.0080 | 0.0% | 100.0% | SKEW |
| pbp_part | `part_pass_snap_share_t3` | -0.046 | -5.3 | +0.578 | 1.3% | 1.7% | -0.0014 | +0.0033 | 0.0% | 100.0% | SKEW |
| pbp_part | `part_rz_snap_share_t3` | -0.042 | -5.0 | +0.547 | 0.2% | 0.2% | -0.0035 | +0.0056 | 0.2% | 100.0% | SKEW |
| pbp_part | `part_rz_snap_share_l1` | -0.034 | -4.1 | +0.498 | 0.3% | 0.1% | -0.0074 | +0.0054 | 1.8% | 100.0% | SKEW |
| pbp_part | `part_pass_snap_share_l1` | -0.033 | -3.8 | +0.572 | 2.9% | 2.7% | -0.0113 | -0.0004 | 0.0% | 100.0% | SKEW |
| pbp_usage | `pbp_wopr_t3` | -0.055 | -7.3 | +0.498 | 0.1% | 0.1% | -0.0030 | +0.0054 | 0.0% | 0.0% | UNCONFIRMED |
| pbp_usage | `pbp_wopr_l1` | -0.052 | -6.5 | +0.489 | 0.1% | 0.1% | -0.0057 | +0.0024 | 0.0% | 0.0% | UNCONFIRMED |
| pbp_usage | `pbp_wopr_std` | -0.048 | -6.1 | +0.494 | 0.1% | 0.1% | -0.0048 | +0.0069 | 0.0% | 0.0% | UNCONFIRMED |
| pbp_usage | `pbp_rz_opps_pg_t3` | -0.047 | -5.7 | +0.454 | 0.0% | 0.1% | -0.0061 | +0.0043 | 0.0% | 0.0% | UNCONFIRMED |
| pbp_usage | `pbp_rz_opps_pg_std` | -0.045 | -5.2 | +0.509 | 0.3% | 0.1% | -0.0102 | +0.0029 | 0.0% | 0.0% | REDUNDANT |
| pbp_usage | `pbp_rz_tgt_share_std` | -0.042 | -5.1 | +0.419 | 0.1% | 0.2% | -0.0141 | +0.0049 | 0.6% | 2.5% | REDUNDANT |
| pbp_usage | `pbp_i10_opps_pg_t3` | -0.041 | -4.9 | +0.375 | 0.0% | 0.0% | -0.0055 | +0.0083 | 0.0% | 0.0% | REDUNDANT |
| pbp_usage | `pbp_i10_opps_pg_std` | -0.041 | -4.8 | +0.449 | 0.2% | 0.1% | -0.0107 | +0.0079 | 0.0% | 0.0% | REDUNDANT |
| pbp_usage | `pbp_ay_share_std` | -0.039 | -4.7 | +0.260 | 0.1% | 0.2% | -0.0064 | +0.0029 | 0.0% | 0.0% | REDUNDANT |
| pbp_usage | `pbp_ay_share_t3` | -0.038 | -4.9 | +0.287 | 0.1% | 0.1% | -0.0047 | +0.0062 | 0.0% | 0.0% | REDUNDANT |
| pbp_usage | `pbp_rz_opps_pg_l1` | -0.037 | -4.6 | +0.344 | 0.0% | 0.0% | -0.0023 | +0.0118 | 0.0% | 0.0% | REDUNDANT |
| pbp_usage | `pbp_prev_wopr` | -0.035 | -4.3 | +0.422 | 0.2% | 0.2% | +0.0004 | +0.0051 | 19.0% | 14.4% | REDUNDANT |
| pbp_usage | `pbp_rz_tgt_share_t3` | -0.035 | -4.3 | +0.388 | 0.1% | 0.1% | -0.0092 | +0.0079 | 0.6% | 2.5% | REDUNDANT |
| pbp_usage | `pbp_i10_tgt_share_std` | -0.034 | -4.2 | +0.357 | 0.1% | 0.1% | -0.0089 | +0.0090 | 3.1% | 7.4% | REDUNDANT |
| pbp_usage | `pbp_prev_rz_car_share` | -0.034 | -3.6 | +0.274 | 0.3% | 0.1% | -0.0082 | +0.0053 | 19.1% | 14.4% | REDUNDANT |
| pbp_usage | `pbp_i10_tgt_share_t3` | -0.033 | -4.2 | +0.302 | 0.1% | 0.0% | -0.0031 | +0.0111 | 4.3% | 7.4% | REDUNDANT |
| pbp_usage | `pbp_prev_rz_tgt_share` | -0.033 | -3.9 | +0.393 | 0.1% | 0.0% | -0.0069 | +0.0059 | 19.1% | 14.4% | REDUNDANT |
| pbp_usage | `pbp_ay_share_l1` | -0.033 | -4.0 | +0.287 | 0.1% | 0.2% | -0.0056 | +0.0033 | 0.0% | 0.0% | REDUNDANT |
| pbp_usage | `pbp_adot_t3` | -0.031 | -4.5 | +0.059 | 0.1% | 0.2% | -0.0129 | +0.0092 | 22.5% | 26.8% | NOISE |
| pbp_usage | `pbp_i10_opps_pg_l1` | -0.030 | -3.6 | +0.266 | 0.0% | 0.0% | -0.0044 | +0.0104 | 0.0% | 0.0% | REDUNDANT |
| pbp_usage | `pbp_rz_tgt_share_l1` | -0.027 | -3.3 | +0.308 | 0.1% | 0.0% | -0.0052 | +0.0105 | 5.9% | 3.1% | REDUNDANT |
| pbp_usage | `pbp_adot_l1` | -0.025 | -3.0 | +0.049 | 0.1% | 0.2% | -0.0092 | +0.0013 | 35.1% | 34.7% | NOISE |
| lags | `targets_l1` | -0.053 | -6.4 | +0.495 | 0.0% | 0.0% | -0.0040 | +0.0042 | 0.0% | 0.0% | UNCONFIRMED |
| lags | `targets_l2` | -0.042 | -5.0 | +0.481 | 0.0% | 0.0% | -0.0011 | +0.0091 | 9.5% | 31.4% | REDUNDANT |
| lags | `targets_l3` | -0.039 | -4.4 | +0.460 | 0.0% | 0.0% | -0.0111 | +0.0132 | 18.5% | 59.2% | SKEW |
| eff | `eff_pass_cmp_rate_career` | +0.043 | +2.3 | +0.137 | 0.1% | 0.1% | -0.0066 | +0.0098 | 80.2% | 80.3% | SKEW |
| eff | `eff_rec_ypt_career` | +0.026 | +2.7 | +0.141 | 0.2% | 0.2% | -0.0038 | +0.0084 | 9.5% | 9.4% | REDUNDANT |
| eff | `eff_pass_int_rate_std` | -0.025 | -1.1 | -0.170 | 0.1% | 0.1% | -0.0052 | +0.0067 | 88.2% | 90.5% | SKEW |
| pbp_team | `team_plays_pg_std` | -0.031 | -4.3 | +0.021 | 0.6% | 0.6% | -0.0165 | +0.0096 | 0.0% | 0.0% | NOISE |
| college | `college_ypc` | +0.029 | +0.6 | +0.300 | 0.0% | 0.0% | -0.0089 | +0.0098 | 96.1% | 98.4% | SKEW |

Noise floor (placebo columns, same fits): placebo_2 rho_resid -0.012, dRMSE -0.0046 / +0.0062; placebo_1 rho_resid -0.010, dRMSE -0.0028 / +0.0050; placebo_5 rho_resid +0.007, dRMSE -0.0070 / +0.0113; placebo_3 rho_resid -0.001, dRMSE -0.0031 / +0.0076; placebo_4 rho_resid +0.000, dRMSE -0.0068 / +0.0078.

## Reading

121 columns screened: 6 UNCONFIRMED, 35 REDUNDANT, 40 NOISE, 40 SKEW. The top-decile cut is |rho_resid| >= 0.045; the strongest residual signals are `part_run_snap_share_std` -0.060 (rho_y +0.497), `pbp_wopr_t3` -0.055 (rho_y +0.498), `part_run_snap_share_t3` -0.055 (rho_y +0.497). By family, median |rho_resid|: pbp_part 0.048, pbp_usage 0.033, adp 0.021, role 0.018. The 5 placebo columns (pure noise) move the single-fit RMSE by -0.0070 to +0.0113 and their |rho_resid| peaks at 0.012: a dRMSE or rho inside those bands is not evidence. Nothing is PROMISING: no column both tracks the primary's misses (top decile) and lowers the single-fit RMSE in both years with a usable 2026 feed. Spearman is rank-only and within position-week; a column the trees could use through an interaction can show a small rho, which is why the gain step exists.

## PROMISING set

None.
