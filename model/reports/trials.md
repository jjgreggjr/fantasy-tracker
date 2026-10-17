# Trials

One row per `python -m model.trial` run (PLAN_MODEL.md "Phase 5"). dRMSE / dPick are variant minus primary on the Phase 2.5
head-to-head rows with 95% week-blocked bootstrap intervals; negative dRMSE and positive dPick mean the variant is better. The verdict
is the Phase 2.5 adoption rule on pooled dRMSE: SHIPS (beats on 2025, interval below zero, same sign on 2024), DIRECTION ONLY (better
both years, 2025 interval spans zero), DOES NOT SHIP, or TIE (components) under `--rule tie`. 2026 is the live year, reported only: it
never decides. Adopting a SHIPS is a human edit of `model/serving_config.json` or `model/tuned_params.json`, then
`python -m model.serve --freeze`.

| date | candidate | lib | 2025 dRMSE | 2025 dPick | 2024 dRMSE | 2024 dPick | 2026 dRMSE | 2026 dPick | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-16 | `+family:streak` | lightgbm | +0.004 [-0.005, +0.014] | +0.001 [+0.000, +0.001] | -0.009 [-0.017, -0.001] | +0.001 [-0.000, +0.002] | -0.004 [-0.017, +0.018] (5 wk, n=1,469) | +0.001 [-0.001, +0.003] | DOES NOT SHIP |
| 2026-10-16 | `recent:4` | lightgbm | +0.005 [-0.008, +0.018] | +0.000 [-0.001, +0.001] | -0.002 [-0.012, +0.008] | +0.000 [-0.001, +0.001] | +0.028 [+0.016, +0.040] (5 wk, n=1,469) | -0.001 [-0.002, +0.000] | DOES NOT SHIP |
| 2026-10-16 | `recent:4x2` | lightgbm | +0.003 [-0.006, +0.011] | -0.001 [-0.001, +0.000] | -0.001 [-0.012, +0.009] | -0.000 [-0.001, +0.000] | +0.015 [+0.008, +0.026] (5 wk, n=1,469) | -0.000 [-0.001, +0.001] | DOES NOT SHIP |
| 2026-10-16 | `recent:8` | lightgbm | +0.005 [-0.009, +0.019] | +0.000 [-0.001, +0.001] | +0.001 [-0.014, +0.015] | -0.000 [-0.001, +0.001] | +0.018 [+0.006, +0.030] (5 wk, n=1,469) | +0.000 [-0.001, +0.001] | DOES NOT SHIP |
| 2026-10-16 | `+cols:tm_out_group_n,tm_q_group_n` | lightgbm | +0.002 [-0.007, +0.010] | +0.000 [-0.000, +0.001] | +0.003 [-0.012, +0.017] | -0.000 [-0.001, +0.000] | +0.006 [-0.006, +0.020] (5 wk, n=1,469) | +0.001 [-0.000, +0.003] | DOES NOT SHIP |
| 2026-10-16 | `param:lightgbm.reg_lambda=50 param:lightgbm.min_child_samples=400` | lightgbm | +0.008 [-0.001, +0.015] | -0.000 [-0.001, +0.000] | -0.007 [-0.015, +0.002] | -0.000 [-0.001, -0.000] | +0.022 [+0.005, +0.041] (5 wk, n=1,469) | +0.000 [-0.001, +0.001] | DOES NOT SHIP |
| 2026-10-16 | `param:lightgbm.reg_lambda=5 param:lightgbm.min_child_samples=100` | lightgbm | -0.002 [-0.012, +0.008] | +0.000 [-0.001, +0.001] | -0.001 [-0.010, +0.009] | -0.000 [-0.001, +0.000] | +0.001 [-0.003, +0.006] (5 wk, n=1,469) | -0.000 [-0.001, +0.000] | DIRECTION ONLY |
| 2026-10-16 | `+family:pbp_usage` | lightgbm | +0.011 [-0.001, +0.022] | -0.000 [-0.001, +0.001] | -0.013 [-0.026, +0.000] | +0.001 [+0.000, +0.002] | -0.002 [-0.018, +0.012] (5 wk, n=1,469) | +0.001 [+0.000, +0.002] | DOES NOT SHIP |
| 2026-10-16 | `+family:pbp_team` | lightgbm | +0.010 [-0.002, +0.021] | +0.000 [-0.001, +0.001] | -0.006 [-0.016, +0.005] | -0.000 [-0.001, +0.001] | -0.008 [-0.027, +0.012] (5 wk, n=1,469) | +0.001 [-0.001, +0.003] | DOES NOT SHIP |
| 2026-10-16 | `+family:eff` | lightgbm | -0.004 [-0.014, +0.006] | +0.001 [-0.000, +0.001] | -0.008 [-0.022, +0.005] | +0.000 [-0.001, +0.001] | -0.004 [-0.025, +0.016] (5 wk, n=1,469) | +0.000 [-0.002, +0.002] | DIRECTION ONLY |
| 2026-10-16 | `+cols:inj_report_status,inj_practice_status,inj_listed_l4wk` | lightgbm | +0.001 [-0.007, +0.008] | +0.000 [-0.001, +0.001] | +0.004 [-0.003, +0.011] | -0.000 [-0.001, +0.000] | +0.002 [-0.012, +0.015] (5 wk, n=1,469) | +0.002 [+0.000, +0.003] | DOES NOT SHIP |
| 2026-10-16 | `+cols:targets_l1,targets_l2,targets_l3,carries_l3,pass_att_l2,pass_att_l3` | lightgbm | -0.001 [-0.012, +0.011] | -0.000 [-0.001, +0.001] | -0.011 [-0.020, -0.003] | +0.000 [-0.000, +0.001] | +0.003 [-0.007, +0.014] (5 wk, n=1,469) | +0.001 [-0.001, +0.002] | DIRECTION ONLY |
| 2026-10-16 | `+cols:implied_opp_total,div_game,neutral_site,week` | lightgbm | -0.005 [-0.018, +0.007] | +0.000 [-0.001, +0.001] | -0.009 [-0.018, +0.000] | +0.000 [-0.000, +0.001] | +0.001 [-0.014, +0.016] (5 wk, n=1,469) | +0.000 [-0.002, +0.002] | DIRECTION ONLY |
| 2026-10-16 | `+cols:draft_round,exp_years,is_rookie,is_undrafted,height_in,career_games_prior` | lightgbm | -0.001 [-0.010, +0.008] | +0.001 [+0.000, +0.001] | +0.001 [-0.008, +0.012] | +0.000 [-0.000, +0.001] | +0.007 [-0.014, +0.035] (5 wk, n=1,469) | +0.000 [-0.001, +0.001] | DOES NOT SHIP |
| 2026-10-16 | `+family:streak recent:4` | lightgbm | +0.007 [-0.009, +0.023] | +0.001 [+0.000, +0.002] | -0.013 [-0.025, -0.003] | +0.001 [-0.000, +0.002] | +0.019 [-0.009, +0.037] (5 wk, n=1,469) | +0.000 [-0.002, +0.002] | DOES NOT SHIP |
| 2026-10-16 | `+family:eff +family:pbp_usage +family:streak +cols:tm_out_group_n,tm_q_group_n` | lightgbm | +0.005 [-0.005, +0.015] | +0.001 [+0.000, +0.002] | -0.002 [-0.014, +0.010] | -0.000 [-0.001, +0.001] | +0.003 [-0.023, +0.022] (5 wk, n=1,469) | +0.001 [-0.002, +0.003] | DOES NOT SHIP |
| 2026-10-16 | `+family:eff +family:pbp_usage +family:pbp_team +family:streak +cols:tm_out_group_n,tm_q_group_n,targets_l1,targets_l2,targets_l3,inj_report_status,inj_practice_status recent:4` | lightgbm | +0.014 [-0.005, +0.030] | +0.001 [-0.000, +0.002] | -0.020 [-0.039, +0.002] | +0.001 [-0.000, +0.002] | -0.020 [-0.049, +0.008] (5 wk, n=1,469) | +0.003 [+0.001, +0.004] | DOES NOT SHIP |
| 2026-10-16 | `+family:ecr` | lightgbm | -0.099 [-0.129, -0.066] | +0.004 [+0.002, +0.006] | -0.069 [-0.111, -0.030] | +0.004 [+0.002, +0.006] | -0.156 [-0.198, -0.103] (5 wk, n=1,469) | +0.008 [+0.004, +0.014] | SHIPS |
| 2026-10-17 | `+family:nfelo` | lightgbm | +0.004 [-0.010, +0.015] | -0.000 [-0.001, +0.001] | +0.005 [-0.007, +0.018] | -0.001 [-0.001, +0.000] | +0.012 [-0.011, +0.041] (5 wk, n=1,469) | +0.001 [-0.000, +0.002] | DOES NOT SHIP |
| 2026-10-17 | `+family:pepa` | lightgbm | +0.006 [-0.008, +0.020] | -0.000 [-0.001, +0.001] | -0.019 [-0.032, -0.006] | +0.001 [+0.000, +0.002] | +0.020 [+0.004, +0.034] (5 wk, n=1,469) | +0.001 [-0.001, +0.002] | DOES NOT SHIP |
| 2026-10-17 | `+family:nfelo +family:pepa` | lightgbm | +0.005 [-0.016, +0.027] | +0.000 [-0.001, +0.002] | -0.011 [-0.029, +0.006] | +0.001 [-0.000, +0.001] | +0.019 [+0.003, +0.034] (5 wk, n=1,469) | +0.001 [+0.000, +0.001] | DOES NOT SHIP |
