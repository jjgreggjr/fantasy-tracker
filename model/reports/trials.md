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
