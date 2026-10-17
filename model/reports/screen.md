# Predictability screen

Run 2026-10-17 (`python -m model.screen --threads 2`), 7.1 min. Matrix key `5fdf934f02612d1a` (49,859 rows x 319 columns). Screened rows: 24,102 head-to-head rows of 2022-2025 (played, with an earlier game that season); 2026 null rates on 1,469 2026 head-to-head rows (weeks 2-5).

`rho_resid` / `rho_y`: Spearman with the frozen LightGBM primary's out-of-sample residual / with actual PPR, within (season, week, position), mean over groups; `t` = that mean over its standard error. `gain` / `dRMSE` are **screen-grade**: one fit per year on 2021..y-1 with the column added alone (tuned parameters, no retree, no weekly refit), gain share of that column and RMSE minus the primary's same single fit on year y's head-to-head rows. Verdicts: PROMISING (top-decile |rho_resid|, dRMSE < 0 both years, 2026 null < 50%), UNCONFIRMED (top decile, single fits not both better), REDUNDANT (|rho_y| >= 0.1), NOISE, SKEW (2026 null >= 50%), DEAD (no information 2021-2024). The top 40 by |rho_resid| (plus every PROMISING) are shown, grouped by family; every number is in `model/cache/screen/5fdf934f02612d1a.parquet`.

| family | column | rho_resid | t | rho_y | gain 2024 | gain 2025 | dRMSE 2024 | dRMSE 2025 | null | null 2026 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| market | `mkt_vs_xfp_l3` | +0.121 | +14.9 | +0.009 | 5.0% | 5.1% | -0.0843 | -0.0917 | 9.7% | 6.7% | PROMISING |
| pbp_part | `part_run_snap_share_std` | -0.060 | -7.1 | +0.497 | 0.3% | 0.2% | -0.0029 | +0.0040 | 0.0% | 100.0% | SKEW |
| pbp_part | `part_run_snap_share_t3` | -0.055 | -6.2 | +0.497 | 0.4% | 0.3% | -0.0132 | +0.0076 | 0.0% | 100.0% | SKEW |
| pbp_part | `part_prev_pass_snap_share` | -0.050 | -5.5 | +0.467 | 0.1% | 0.1% | -0.0025 | +0.0098 | 19.0% | 14.4% | UNCONFIRMED |
| pbp_part | `part_pass_snap_share_std` | -0.048 | -5.6 | +0.575 | 0.8% | 0.4% | -0.0029 | +0.0055 | 0.0% | 100.0% | SKEW |
| pbp_part | `part_rz_snap_share_std` | -0.048 | -5.4 | +0.551 | 0.3% | 0.2% | -0.0040 | +0.0071 | 0.2% | 100.0% | SKEW |
| pbp_part | `part_run_snap_share_l1` | -0.048 | -5.7 | +0.482 | 0.6% | 0.4% | +0.0003 | +0.0080 | 0.0% | 100.0% | SKEW |
| pbp_part | `part_pass_snap_share_t3` | -0.046 | -5.3 | +0.578 | 1.3% | 1.7% | -0.0014 | +0.0033 | 0.0% | 100.0% | SKEW |
| pbp_part | `part_rz_snap_share_t3` | -0.042 | -5.0 | +0.547 | 0.2% | 0.2% | -0.0035 | +0.0056 | 0.2% | 100.0% | SKEW |
| pbp_part | `part_rz_snap_share_l1` | -0.034 | -4.1 | +0.498 | 0.3% | 0.1% | -0.0074 | +0.0054 | 1.8% | 100.0% | SKEW |
| pbp_usage | `pbp_wopr_t3` | -0.055 | -7.3 | +0.498 | 0.1% | 0.1% | -0.0030 | +0.0054 | 0.0% | 0.0% | UNCONFIRMED |
| pbp_usage | `pbp_wopr_l1` | -0.052 | -6.5 | +0.489 | 0.1% | 0.1% | -0.0057 | +0.0024 | 0.0% | 0.0% | UNCONFIRMED |
| pbp_usage | `pbp_wopr_std` | -0.048 | -6.1 | +0.494 | 0.1% | 0.1% | -0.0048 | +0.0069 | 0.0% | 0.0% | UNCONFIRMED |
| pbp_usage | `pbp_rz_opps_pg_t3` | -0.047 | -5.7 | +0.454 | 0.0% | 0.1% | -0.0061 | +0.0043 | 0.0% | 0.0% | UNCONFIRMED |
| pbp_usage | `pbp_rz_opps_pg_std` | -0.045 | -5.2 | +0.509 | 0.3% | 0.1% | -0.0102 | +0.0029 | 0.0% | 0.0% | UNCONFIRMED |
| pbp_usage | `pbp_rz_tgt_share_std` | -0.042 | -5.1 | +0.419 | 0.1% | 0.2% | -0.0141 | +0.0049 | 0.6% | 2.5% | UNCONFIRMED |
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
| lags | `targets_l1` | -0.053 | -6.4 | +0.495 | 0.0% | 0.0% | -0.0040 | +0.0042 | 0.0% | 0.0% | UNCONFIRMED |
| lags | `targets_l2` | -0.042 | -5.0 | +0.481 | 0.0% | 0.0% | -0.0011 | +0.0091 | 9.5% | 31.4% | UNCONFIRMED |
| lags | `targets_l3` | -0.039 | -4.4 | +0.460 | 0.0% | 0.0% | -0.0111 | +0.0132 | 18.5% | 59.2% | SKEW |
| ecr | `ecr_delta_l1` | -0.048 | -5.0 | -0.069 | 0.3% | 0.4% | -0.0144 | -0.0008 | 27.5% | 3.3% | PROMISING |
| pepa | `pepa_rec_fd_l3` | -0.048 | -5.9 | +0.518 | 0.0% | 0.0% | -0.0030 | +0.0078 | 0.0% | 0.0% | UNCONFIRMED |
| pepa | `pepa_rec_fd_std` | -0.046 | -5.6 | +0.511 | 0.3% | 0.2% | -0.0068 | -0.0009 | 0.0% | 0.0% | PROMISING |
| pepa | `pepa_rec_yac_l3` | -0.041 | -5.4 | +0.474 | 0.2% | 0.2% | -0.0121 | +0.0091 | 0.0% | 0.0% | REDUNDANT |
| pepa | `pepa_rec_air_yards_std` | -0.039 | -4.7 | +0.262 | 0.1% | 0.2% | -0.0051 | +0.0069 | 0.0% | 0.0% | REDUNDANT |
| pepa | `pepa_rec_air_yards_l3` | -0.039 | -4.9 | +0.287 | 0.2% | 0.2% | -0.0090 | +0.0066 | 0.0% | 0.0% | REDUNDANT |
| pepa | `pepa_rec_yac_std` | -0.036 | -4.4 | +0.472 | 0.2% | 0.1% | -0.0143 | +0.0012 | 0.0% | 0.0% | REDUNDANT |
| pepa | `pepa_rec_air_yards_l1` | -0.036 | -4.3 | +0.284 | 0.1% | 0.2% | -0.0108 | +0.0071 | 0.0% | 0.0% | REDUNDANT |
| pepa | `pepa_rec_yac_l1` | -0.035 | -4.5 | +0.435 | 0.1% | 0.2% | -0.0161 | +0.0093 | 0.0% | 0.0% | REDUNDANT |
| pepa | `pepa_rec_fd_l1` | -0.034 | -4.2 | +0.441 | 0.1% | 0.0% | -0.0045 | +0.0092 | 0.0% | 0.0% | REDUNDANT |
| eff | `eff_pass_cmp_rate_career` | +0.043 | +2.3 | +0.137 | 0.1% | 0.1% | -0.0066 | +0.0098 | 80.2% | 80.3% | SKEW |

Noise floor (placebo columns, same fits): placebo_2 rho_resid -0.012, dRMSE -0.0046 / +0.0062; placebo_1 rho_resid -0.010, dRMSE -0.0028 / +0.0050; placebo_5 rho_resid +0.007, dRMSE -0.0070 / +0.0113; placebo_3 rho_resid -0.001, dRMSE -0.0031 / +0.0076; placebo_4 rho_resid +0.000, dRMSE -0.0068 / +0.0078.

## Reading

205 columns screened: 3 PROMISING, 10 UNCONFIRMED, 73 REDUNDANT, 70 NOISE, 49 SKEW. The top-decile cut is |rho_resid| >= 0.042; the strongest residual signals are `mkt_vs_xfp_l3` +0.121 (rho_y +0.009), `part_run_snap_share_std` -0.060 (rho_y +0.497), `pbp_wopr_t3` -0.055 (rho_y +0.498). By family, median |rho_resid|: pbp_part 0.048, pbp_usage 0.033, adp 0.021, role 0.018. The 5 placebo columns (pure noise) move the single-fit RMSE by -0.0070 to +0.0113 and their |rho_resid| peaks at 0.012: a dRMSE or rho inside those bands is not evidence. PROMISING: `ecr_delta_l1`, `mkt_vs_xfp_l3`, `pepa_rec_fd_std`; that set still has to pass `model.trial` (full walk-forward, 2025 interval below zero, 2024 same sign) before it means anything. Spearman is rank-only and within position-week; a column the trees could use through an interaction can show a small rho, which is why the gain step exists.

## PROMISING set

Columns: `ecr_delta_l1`, `mkt_vs_xfp_l3`, `pepa_rec_fd_std`.

```
python -m model.trial --candidate +family:ecr --candidate +family:market --candidate +family:pepa
```

* +family:ecr adds all 8 ecr columns (1 screened PROMISING): trial.py has no per-column add for it
* +family:market adds all 11 market columns (1 screened PROMISING): trial.py has no per-column add for it
* +family:pepa adds all 44 pepa columns (1 screened PROMISING): trial.py has no per-column add for it
