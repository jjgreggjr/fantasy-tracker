# Retrospective

What each comparator got most wrong in every completed week, and why, judged on the scoreboard's identical rows (`python -m model.retrospective`, run after `model.scoreboard`). It explains misses; it is not the adoption gate.

Weeks post-mortemed: 2026 wk4, 2026 wk5, 2026 wk6, 2026 wk7. A big miss is |predicted - actual| >= 12 points in the league's scoring, one row per player-week per comparator (the `gooma-s-family-league` row where he is rostered there, else the first other league alphabetically). Comparators: `model_components` (E_pts_model), `sleeper`, `e_pts`.

## Drift: pick accuracy by week

From `data/model_eval.csv`, pooled over leagues (the scoreboard's number). Cumulative mean and the week-to-week standard deviation of the weekly values.

| comparator | 2026 wk04 | 2026 wk05 | 2026 wk06 | 2026 wk07 | mean | week sd |
|---|---|---|---|---|---|---|
| `model_components` | 0.679 | 0.675 | 0.645 | 0.672 | 0.668 | 0.016 |
| `model_flat` | 0.670 | 0.670 | 0.638 | 0.673 | 0.663 | 0.016 |
| `model_blend` | 0.678 | 0.674 | 0.643 | 0.676 | 0.668 | 0.017 |
| `sleeper` | 0.644 | 0.700 | 0.635 | 0.673 | 0.663 | 0.030 |
| `e_pts` | 0.647 | 0.668 | 0.636 | 0.673 | 0.656 | 0.017 |

No comparator is trending: every span move is inside twice its week-to-week standard deviation.

## Why the big misses happened

Share of each comparator's big misses by reason, all weeks pooled. `inactive_missed` (predicted > 5, did not play) is outside the scored rows and is counted apart: it is the status layer's miss, not the model's. Learnable = the pre-kickoff record could in principle have carried the signal (`volume_up`, `volume_down`, `teammate_out`, `blowout_script`); noise = `td_variance`, `left_early`, `unexplained`.

| comparator | big misses | `left_early` | `teammate_out` | `blowout_script` | `volume_up` | `volume_down` | `td_variance` | `unexplained` | learnable | noise | `inactive_missed` (apart) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `model_components` | 63 | 1 (2%) | 4 (6%) | 1 (2%) | 23 (37%) | 3 (5%) | 14 (22%) | 17 (27%) | 49% | 51% | 30 |
| `sleeper` | 61 | 3 (5%) | 4 (7%) | 2 (3%) | 22 (36%) | 6 (10%) | 12 (20%) | 12 (20%) | 56% | 44% | 17 |
| `e_pts` | 69 | 1 (1%) | 4 (6%) | 2 (3%) | 23 (33%) | 9 (13%) | 13 (19%) | 17 (25%) | 55% | 45% | 16 |

Per week, the 95th percentile of |error| on the scored rows (the `top 5%` rule would have cut here instead of at 12):

| season / week | `model_components` | `sleeper` | `e_pts` |
|---|---|---|---|
| 2026 wk04 | 14.7 | 14.9 | 15.8 |
| 2026 wk05 | 15.4 | 15.2 | 15.7 |
| 2026 wk06 | 12.7 | 14.2 | 13.6 |
| 2026 wk07 | 13.3 | 12.6 | 13.1 |

## Top 15 misses, 2026 week 7

Each comparator's largest played misses that week with the evidence behind the reason (expected = trailing mean over his earlier played weeks this season; volume = pass attempts + carries for a QB, carries + targets otherwise; margin = his team's final margin).

**`model_components`** (15 shown)

| player | pos | team | league | predicted | actual | error | reason | secondary | vol exp/act | TD exp/act | snap% trail/act | margin | teammate out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tony Pollard | RB | TEN | gooma-s-family-league | 12.1 | 36.5 | -24.4 | `volume_up` | `td_variance` | 18.3/28 | 0.3/3 | 0.59/0.75 | -3 |  |
| Mason Taylor | TE | NYJ | we-can-think-of-something-funny | 6.0 | 27.7 | -21.7 | `td_variance` |  | 4.0/6 | 0.0/2 | 0.60/0.63 | -3 |  |
| KC Concepcion | WR | CLE | gooma-s-family-league | 9.0 | 23.2 | -14.2 | `volume_up` |  | 5.7/13 | 0.0/0 | 0.63/0.79 | 3 |  |
| Jameson Williams | WR | DET | gooma-s-family-league | 12.1 | 26.1 | -14.0 | `volume_up` | `td_variance` | 5.3/9 | 0.7/2 | 0.90/0.84 | 3 |  |
| Nico Collins | WR | HOU | gooma-s-family-league | 16.6 | 2.6 | 14.0 | `unexplained` |  | 7.0/5 | 0.7/0 | 0.71/0.67 | -6 |  |
| Bryce Young | QB | CAR | gooma-s-family-league | 19.2 | 5.4 | 13.8 | `blowout_script` | `volume_down` | 44.7/28 | 1.7/1 | 1.00/0.77 | -30 |  |
| Omar Cooper Jr. | WR | NYJ | gooma-s-family-league | 8.9 | 22.5 | -13.6 | `unexplained` |  | 4.0/5 | 0.5/1 | 0.44/0.64 | -3 |  |
| Tee Higgins | WR | CIN | gooma-s-family-league | 13.3 | 26.8 | -13.5 | `unexplained` |  | 9.0/7 | 0.3/1 | 0.80/0.87 | 22 |  |
| Cam Skattebo | RB | NYG | gooma-s-family-league | 5.7 | 19.1 | -13.4 | `unexplained` |  | 19.7/18 | 0.0/1 | 0.68/0.56 | 6 |  |
| Malik Willis | QB | MIA | gooma-s-family-league | 12.7 | 26.0 | -13.3 | `td_variance` |  | 33.0/29 | 0.7/3 | 1.00/1.00 | 3 |  |
| Caleb Douglas | WR | MIA | gooma-s-family-league | 7.7 | 21.0 | -13.3 | `td_variance` |  | 5.0/3 | 0.0/2 | 0.89/0.68 | 3 |  |
| Luther Burden III | WR | CHI | gooma-s-family-league | 11.1 | 23.7 | -12.6 | `unexplained` |  | 8.3/5 | 0.0/1 | 0.57/0.61 | 6 |  |
| Quinshon Judkins | RB | CLE | gooma-s-family-league | 10.9 | 23.4 | -12.5 | `td_variance` |  | 20.7/18 | 0.3/2 | 0.57/0.53 | 3 |  |
| Trey McBride | TE | ARI | gooma-s-family-league | 14.7 | 27.1 | -12.4 | `unexplained` |  | 10.3/13 | 0.3/1 | 0.95/0.93 | -14 |  |
| Derrick Henry | RB | BAL | gooma-s-family-league | 18.0 | 5.9 | 12.1 | `td_variance` |  | 23.0/19 | 0.7/0 | 0.58/0.51 | -22 |  |

**`sleeper`** (13 shown)

| player | pos | team | league | predicted | actual | error | reason | secondary | vol exp/act | TD exp/act | snap% trail/act | margin | teammate out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tony Pollard | RB | TEN | gooma-s-family-league | 12.0 | 36.5 | -24.5 | `volume_up` | `td_variance` | 18.3/28 | 0.3/3 | 0.59/0.75 | -3 |  |
| Mason Taylor | TE | NYJ | we-can-think-of-something-funny | 5.3 | 27.7 | -22.4 | `td_variance` |  | 4.0/6 | 0.0/2 | 0.60/0.63 | -3 |  |
| Nico Collins | WR | HOU | gooma-s-family-league | 17.1 | 2.6 | 14.5 | `unexplained` |  | 7.0/5 | 0.7/0 | 0.71/0.67 | -6 |  |
| Cam Skattebo | RB | NYG | gooma-s-family-league | 4.9 | 19.1 | -14.2 | `unexplained` |  | 19.7/18 | 0.0/1 | 0.68/0.56 | 6 |  |
| Jalen Coker | WR | CAR | gooma-s-family-league | 15.6 | 1.9 | 13.7 | `blowout_script` | `volume_down` | 8.7/4 | 0.0/0 | 0.62/0.66 | -30 |  |
| Jameson Williams | WR | DET | gooma-s-family-league | 12.4 | 26.1 | -13.7 | `volume_up` | `td_variance` | 5.3/9 | 0.7/2 | 0.90/0.84 | 3 |  |
| KC Concepcion | WR | CLE | gooma-s-family-league | 9.7 | 23.2 | -13.5 | `volume_up` |  | 5.7/13 | 0.0/0 | 0.63/0.79 | 3 |  |
| Caleb Douglas | WR | MIA | gooma-s-family-league | 7.6 | 21.0 | -13.4 | `td_variance` |  | 5.0/3 | 0.0/2 | 0.89/0.68 | 3 |  |
| Omar Cooper Jr. | WR | NYJ | gooma-s-family-league | 9.5 | 22.5 | -13.0 | `unexplained` |  | 4.0/5 | 0.5/1 | 0.44/0.64 | -3 |  |
| Bryce Young | QB | CAR | gooma-s-family-league | 18.0 | 5.4 | 12.6 | `blowout_script` | `volume_down` | 44.7/28 | 1.7/1 | 1.00/0.77 | -30 |  |
| Garrett Wilson | WR | NYJ | gooma-s-family-league | 16.9 | 4.6 | 12.3 | `volume_down` |  | 7.0/3 | 0.3/0 | 0.97/0.85 | -3 |  |
| Tre Tucker | WR | LV | gooma-s-family-league | 9.1 | 21.3 | -12.2 | `volume_up` |  | 7.3/12 | 0.7/1 | 0.71/0.82 | -14 |  |
| Quinshon Judkins | RB | CLE | gooma-s-family-league | 11.3 | 23.4 | -12.0 | `td_variance` |  | 20.7/18 | 0.3/2 | 0.57/0.53 | 3 |  |

**`e_pts`** (13 shown)

| player | pos | team | league | predicted | actual | error | reason | secondary | vol exp/act | TD exp/act | snap% trail/act | margin | teammate out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tony Pollard | RB | TEN | gooma-s-family-league | 9.1 | 36.5 | -27.4 | `volume_up` | `td_variance` | 18.3/28 | 0.3/3 | 0.59/0.75 | -3 |  |
| Mason Taylor | TE | NYJ | we-can-think-of-something-funny | 6.5 | 27.7 | -21.2 | `td_variance` |  | 4.0/6 | 0.0/2 | 0.60/0.63 | -3 |  |
| KC Concepcion | WR | CLE | gooma-s-family-league | 8.8 | 23.2 | -14.4 | `volume_up` |  | 5.7/13 | 0.0/0 | 0.63/0.79 | 3 |  |
| Nico Collins | WR | HOU | gooma-s-family-league | 16.3 | 2.6 | 13.7 | `unexplained` |  | 7.0/5 | 0.7/0 | 0.71/0.67 | -6 |  |
| Jameson Williams | WR | DET | gooma-s-family-league | 12.5 | 26.1 | -13.6 | `volume_up` | `td_variance` | 5.3/9 | 0.7/2 | 0.90/0.84 | 3 |  |
| Jalen Coker | WR | CAR | gooma-s-family-league | 15.5 | 1.9 | 13.6 | `blowout_script` | `volume_down` | 8.7/4 | 0.0/0 | 0.62/0.66 | -30 |  |
| Kenneth Walker III | RB | KC | gooma-s-family-league | 23.1 | 9.7 | 13.4 | `td_variance` |  | 21.7/19 | 2.0/0 | 0.74/0.74 | -17 |  |
| Malik Willis | QB | MIA | gooma-s-family-league | 12.9 | 26.0 | -13.1 | `td_variance` |  | 33.0/29 | 0.7/3 | 1.00/1.00 | 3 |  |
| Isaiah Likely | TE | NYG | gooma-s-family-league | 16.3 | 3.2 | 13.1 | `volume_down` |  | 12.0/4 | 1.0/0 | 0.79/0.86 | 6 |  |
| Caleb Douglas | WR | MIA | gooma-s-family-league | 7.9 | 21.0 | -13.1 | `td_variance` |  | 5.0/3 | 0.0/2 | 0.89/0.68 | 3 |  |
| Omar Cooper Jr. | WR | NYJ | gooma-s-family-league | 9.4 | 22.5 | -13.1 | `unexplained` |  | 4.0/5 | 0.5/1 | 0.44/0.64 | -3 |  |
| Trey McBride | TE | ARI | gooma-s-family-league | 14.4 | 27.1 | -12.7 | `unexplained` |  | 10.3/13 | 0.3/1 | 0.95/0.93 | -14 |  |
| Cam Skattebo | RB | NYG | gooma-s-family-league | 6.4 | 19.1 | -12.7 | `unexplained` |  | 19.7/18 | 0.0/1 | 0.68/0.56 | 6 |  |

**Did not play, predicted > 5 (`model_components`, 4 players; status layer, not scored)**

Gameday inactive list (nflverse weekly_rosters, posted ~90 min before kickoff; the pre-lock run and serve act on it): 4 of 4 on the gameday inactive list. The rest dressed and did not play, or were never listed.

| player | pos | team | league | predicted | on inactive list |
|---|---|---|---|---|---|
| Lamar Jackson | QB | BAL | gooma-s-family-league | 18.2 | yes |
| Breece Hall | RB | NYJ | gooma-s-family-league | 13.6 | yes |
| Kyle Monangai | RB | CHI | gooma-s-family-league | 10.4 | yes |
| Tyjae Spears | RB | TEN | gooma-s-family-league | 6.5 | yes |

## Candidate signals (derived)

For each reason class, the share of the model's played big misses it explains, the features that would carry the signal, and whether each is served (among the 90 frozen `inputs` of `model/serving_config.json`), pruned (built by `model.features`, dropped by the Phase 2 SHAP prune) or not built. A served feature that still misses means the signal is in the inputs but the week's value was not (`tm_out_*` are read from the report as of serve time: a teammate ruled out after the last run is invisible). A reason whose features are all served is not a new-feature candidate; one whose share is high and whose features are pruned or not built is.

| reason | share of model big misses | features that carry the signal | status in `serving_config.json` inputs |
|---|---|---|---|
| `volume_up` | 37% (23) | `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos`, `opps_trend` | served: `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos` ; not built: `opps_trend` |
| `volume_down` | 5% (3) | `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos`, `opps_trend` | served: `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos` ; not built: `opps_trend` |
| `teammate_out` | 6% (4) | `tm_out_above_n`, `tm_out_group_opps`, `tm_out_targets_all`, `tm_out_carries_all`, `tm_group_size` | served: `tm_out_above_n`, `tm_out_group_opps`, `tm_out_targets_all`, `tm_out_carries_all`, `tm_group_size` |
| `blowout_script` | 2% (1) | `team_spread`, `implied_team_total`, `total_line` | served: `team_spread`, `implied_team_total`, `total_line` |
| `left_early` | 2% (1) | `snap_pct_l1`, `inj_weeks_since_report`, `inj_report_status`, `inj_practice_status` | served: `snap_pct_l1`, `inj_weeks_since_report` ; pruned: `inj_report_status`, `inj_practice_status` |
| `td_variance` | 22% (14) | `td_luck_rush_std`, `td_luck_rec_std`, `td_luck_pass_std`, `td_luck_total_std`, `td_luck_total_prev` | served: `td_luck_rush_std`, `td_luck_rec_std`, `td_luck_pass_std`, `td_luck_total_std`, `td_luck_total_prev` |
| `inactive_missed` | 30 rows, counted apart (18 of 30 on the gameday inactive list) | status layer (`ff.status`, Sleeper `injury_status`, the nflverse report, the gameday inactive list), not a feature | n/a |

## Limits

- Rules, not judgment: thresholds are fixed (35% and 3+ opportunities (QB 6+) for a volume call, 25% or 3 opportunities for 'as expected', snap share < 60% of trailing, margin >= 17, TDs 1.5 from the trailing rate). Priority order: left_early, teammate_out, blowout_script, volume, td_variance. A row can satisfy two; the second is `secondary_reason`.
- Expected volume is the trailing mean of his earlier played weeks this season, not E_opps (not frozen per week) and not the model's own component predictions (not frozen either). A player with no earlier week this season can only be `td_variance`-free `unexplained`, `left_early` (never: no trailing snaps) or `inactive_missed`.
- Teammate out is read from that week's depth chart against `data/player_weeks.csv` and `data/lineups_played.csv`; `data/status.csv` is a snapshot of the upcoming week with no week column and is not used. A teammate who is simply untracked (never a stats row this season) is not counted as out.
- Margins come from `data/defense_vs_pos.csv` (points allowed, `ALL` rows); a week without them has no `blowout_script`.
- Two weeks of misses are a list, not a distribution; read the reason shares after six.
