# Retrospective

What each comparator got most wrong in every completed week, and why, judged on the scoreboard's identical rows (`python -m model.retrospective`, run after `model.scoreboard`). It explains misses; it is not the adoption gate.

Weeks post-mortemed: 2026 wk4, 2026 wk5. A big miss is |predicted - actual| >= 12 points in the league's scoring, one row per player-week per comparator (the `gooma-s-family-league` row where he is rostered there, else the first other league alphabetically). Comparators: `model_components` (E_pts_model), `sleeper`, `e_pts`.

## Drift: pick accuracy by week

From `data/model_eval.csv`, pooled over leagues (the scoreboard's number). Cumulative mean and the week-to-week standard deviation of the weekly values.

| comparator | 2026 wk04 | 2026 wk05 | mean | week sd |
|---|---|---|---|---|
| `model_components` | 0.679 | 0.675 | 0.677 | 0.003 |
| `model_flat` | 0.670 | 0.670 | 0.670 | 0.000 |
| `model_blend` | 0.678 | 0.674 | 0.676 | 0.003 |
| `sleeper` | 0.644 | 0.700 | 0.672 | 0.040 |
| `e_pts` | 0.647 | 0.668 | 0.658 | 0.015 |

2 week(s) scored: too few to call a trend for any comparator (a slope needs 4+ weeks, and the week-to-week standard deviation above is the size of ordinary noise).

## Why the big misses happened

Share of each comparator's big misses by reason, all weeks pooled. `inactive_missed` (predicted > 5, did not play) is outside the scored rows and is counted apart: it is the status layer's miss, not the model's. Learnable = the pre-kickoff record could in principle have carried the signal (`volume_up`, `volume_down`, `teammate_out`, `blowout_script`); noise = `td_variance`, `left_early`, `unexplained`.

| comparator | big misses | `left_early` | `teammate_out` | `blowout_script` | `volume_up` | `volume_down` | `td_variance` | `unexplained` | learnable | noise | `inactive_missed` (apart) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `model_components` | 34 | 1 (3%) | 2 (6%) | 0 (0%) | 16 (47%) | 2 (6%) | 7 (21%) | 6 (18%) | 59% | 41% | 23 |
| `sleeper` | 34 | 3 (9%) | 2 (6%) | 0 (0%) | 13 (38%) | 4 (12%) | 7 (21%) | 5 (15%) | 56% | 44% | 10 |
| `e_pts` | 40 | 1 (2%) | 2 (5%) | 1 (2%) | 15 (38%) | 5 (12%) | 7 (18%) | 9 (22%) | 58% | 42% | 10 |

Per week, the 95th percentile of |error| on the scored rows (the `top 5%` rule would have cut here instead of at 12):

| season / week | `model_components` | `sleeper` | `e_pts` |
|---|---|---|---|
| 2026 wk04 | 14.7 | 14.9 | 15.8 |
| 2026 wk05 | 15.4 | 15.2 | 15.7 |

## Top 15 misses, 2026 week 5

Each comparator's largest played misses that week with the evidence behind the reason (expected = trailing mean over his earlier played weeks this season; volume = pass attempts + carries for a QB, carries + targets otherwise; margin = his team's final margin).

**`model_components`** (15 shown)

| player | pos | team | league | predicted | actual | error | reason | secondary | vol exp/act | TD exp/act | snap% trail/act | margin | teammate out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Isaiah Likely | TE | NYG | gooma-s-family-league | 9.3 | 32.4 | -23.1 | `volume_up` | `td_variance` | 9.0/13 | 0.0/2 | 0.73/0.80 | 0 |  |
| Jaylen Warren | RB | PIT | gooma-s-family-league | 14.5 | 36.8 | -22.3 | `td_variance` |  | 19.7/21 | 0.0/2 | 0.86/0.87 | 12 |  |
| Drake Maye | QB | NE | gooma-s-family-league | 17.8 | 39.5 | -21.6 | `td_variance` |  | 33.3/41 | 1.0/5 | 0.94/1.00 | 28 |  |
| Christian Watson | WR | GB | gooma-s-family-league | 12.5 | 32.3 | -19.8 | `td_variance` |  | 8.0/8 | 0.7/2 | 0.79/0.80 | 24 |  |
| Bucky Irving | RB | TB | gooma-s-family-league | 12.5 | 31.5 | -19.0 | `unexplained` |  | 18.7/24 | 0.0/2 | 0.63/0.54 | 8 |  |
| Parker Washington | WR | JAX | gooma-s-family-league | 11.5 | 30.4 | -18.9 | `td_variance` |  | 7.0/9 | 0.3/2 | 0.81/0.82 | 2 |  |
| Jayden Daniels | QB | WAS | gooma-s-family-league | 13.4 | 32.3 | -18.9 | `unexplained` |  | 31.5/34 | 1.5/2 | 0.78/1.00 | 0 |  |
| Puka Nacua | WR | LA | gooma-s-family-league | 19.8 | 37.7 | -17.9 | `volume_up` |  | 11.0/26 | 0.5/1 | 0.80/0.85 | -3 |  |
| Malik Nabers | WR | NYG | gooma-s-family-league | 12.3 | 29.1 | -16.8 | `volume_up` |  | 5.7/20 | 0.3/1 | 0.76/0.90 | 0 |  |
| DeMario Douglas | WR | NE | gooma-s-family-league | 5.7 | 22.5 | -16.8 | `td_variance` |  | 3.3/5 | 0.0/2 | 0.51/0.49 | 28 |  |
| Mike Evans | WR | SF | gooma-s-family-league | 11.1 | 26.3 | -15.2 | `volume_up` |  | 5.7/10 | 0.3/1 | 0.49/0.63 | 3 |  |
| Kyle Williams | WR | NE | we-can-think-of-something-funny | 3.0 | 18.1 | -15.1 | `unexplained` |  | 2.3/3 | 0.0/1 | 0.27/0.36 | 28 |  |
| CeeDee Lamb | WR | DAL | gooma-s-family-league | 18.0 | 2.9 | 15.1 | `volume_down` |  | 13.0/5 | 1.0/0 | 0.81/0.57 | -8 |  |
| Tyler Shough | QB | NO | gooma-s-family-league | 14.5 | -0.2 | 14.7 | `volume_down` | `td_variance` | 45.3/25 | 2.3/0 | 1.00/1.00 | -10 |  |
| George Pickens | WR | DAL | gooma-s-family-league | 14.9 | 28.0 | -13.1 | `volume_up` |  | 7.3/13 | 0.0/1 | 0.82/0.88 | -8 |  |

**`sleeper`** (15 shown)

| player | pos | team | league | predicted | actual | error | reason | secondary | vol exp/act | TD exp/act | snap% trail/act | margin | teammate out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jaylen Warren | RB | PIT | gooma-s-family-league | 15.3 | 36.8 | -21.5 | `td_variance` |  | 19.7/21 | 0.0/2 | 0.86/0.87 | 12 |  |
| Isaiah Likely | TE | NYG | gooma-s-family-league | 11.4 | 32.4 | -21.0 | `volume_up` | `td_variance` | 9.0/13 | 0.0/2 | 0.73/0.80 | 0 |  |
| Drake Maye | QB | NE | gooma-s-family-league | 18.6 | 39.5 | -20.9 | `td_variance` |  | 33.3/41 | 1.0/5 | 0.94/1.00 | 28 |  |
| CeeDee Lamb | WR | DAL | gooma-s-family-league | 20.7 | 2.9 | 17.8 | `volume_down` |  | 13.0/5 | 1.0/0 | 0.81/0.57 | -8 |  |
| Christian Watson | WR | GB | gooma-s-family-league | 15.3 | 32.3 | -17.0 | `td_variance` |  | 8.0/8 | 0.7/2 | 0.79/0.80 | 24 |  |
| Puka Nacua | WR | LA | gooma-s-family-league | 20.7 | 37.7 | -17.0 | `volume_up` |  | 11.0/26 | 0.5/1 | 0.80/0.85 | -3 |  |
| Parker Washington | WR | JAX | gooma-s-family-league | 14.0 | 30.4 | -16.4 | `td_variance` |  | 7.0/9 | 0.3/2 | 0.81/0.82 | 2 |  |
| Bucky Irving | RB | TB | gooma-s-family-league | 15.1 | 31.5 | -16.4 | `unexplained` |  | 18.7/24 | 0.0/2 | 0.63/0.54 | 8 |  |
| DeMario Douglas | WR | NE | gooma-s-family-league | 6.6 | 22.5 | -15.9 | `td_variance` |  | 3.3/5 | 0.0/2 | 0.51/0.49 | 28 |  |
| Jayden Daniels | QB | WAS | gooma-s-family-league | 16.6 | 32.3 | -15.6 | `unexplained` |  | 31.5/34 | 1.5/2 | 0.78/1.00 | 0 |  |
| Malik Nabers | WR | NYG | gooma-s-family-league | 14.0 | 29.1 | -15.1 | `volume_up` |  | 5.7/20 | 0.3/1 | 0.76/0.90 | 0 |  |
| Kyle Williams | WR | NE | we-can-think-of-something-funny | 3.0 | 18.1 | -15.1 | `unexplained` |  | 2.3/3 | 0.0/1 | 0.27/0.36 | 28 |  |
| Tyler Shough | QB | NO | gooma-s-family-league | 14.8 | -0.2 | 14.9 | `volume_down` | `td_variance` | 45.3/25 | 2.3/0 | 1.00/1.00 | -10 |  |
| George Pickens | WR | DAL | gooma-s-family-league | 13.8 | 28.0 | -14.2 | `volume_up` |  | 7.3/13 | 0.0/1 | 0.82/0.88 | -8 |  |
| Pat Freiermuth | TE | PIT | gooma-s-family-league | 8.2 | 21.5 | -13.3 | `td_variance` |  | 5.3/8 | 0.3/2 | 0.67/0.68 | 12 |  |

**`e_pts`** (15 shown)

| player | pos | team | league | predicted | actual | error | reason | secondary | vol exp/act | TD exp/act | snap% trail/act | margin | teammate out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Drake Maye | QB | NE | gooma-s-family-league | 15.8 | 39.5 | -23.7 | `td_variance` |  | 33.3/41 | 1.0/5 | 0.94/1.00 | 28 |  |
| CeeDee Lamb | WR | DAL | gooma-s-family-league | 25.4 | 2.9 | 22.5 | `volume_down` |  | 13.0/5 | 1.0/0 | 0.81/0.57 | -8 |  |
| Isaiah Likely | TE | NYG | gooma-s-family-league | 10.3 | 32.4 | -22.1 | `volume_up` | `td_variance` | 9.0/13 | 0.0/2 | 0.73/0.80 | 0 |  |
| Jaylen Warren | RB | PIT | gooma-s-family-league | 14.8 | 36.8 | -22.0 | `td_variance` |  | 19.7/21 | 0.0/2 | 0.86/0.87 | 12 |  |
| Bucky Irving | RB | TB | gooma-s-family-league | 11.8 | 31.5 | -19.7 | `unexplained` |  | 18.7/24 | 0.0/2 | 0.63/0.54 | 8 |  |
| Christian Watson | WR | GB | gooma-s-family-league | 13.4 | 32.3 | -18.9 | `td_variance` |  | 8.0/8 | 0.7/2 | 0.79/0.80 | 24 |  |
| DeMario Douglas | WR | NE | gooma-s-family-league | 4.7 | 22.5 | -17.8 | `td_variance` |  | 3.3/5 | 0.0/2 | 0.51/0.49 | 28 |  |
| Parker Washington | WR | JAX | gooma-s-family-league | 12.9 | 30.4 | -17.5 | `td_variance` |  | 7.0/9 | 0.3/2 | 0.81/0.82 | 2 |  |
| George Pickens | WR | DAL | gooma-s-family-league | 11.5 | 28.0 | -16.5 | `volume_up` |  | 7.3/13 | 0.0/1 | 0.82/0.88 | -8 |  |
| Tyler Shough | QB | NO | gooma-s-family-league | 15.8 | -0.2 | 16.0 | `volume_down` | `td_variance` | 45.3/25 | 2.3/0 | 1.00/1.00 | -10 |  |
| Kyle Williams | WR | NE | we-can-think-of-something-funny | 2.4 | 18.1 | -15.7 | `unexplained` |  | 2.3/3 | 0.0/1 | 0.27/0.36 | 28 |  |
| Jayden Daniels | QB | WAS | gooma-s-family-league | 16.6 | 32.3 | -15.7 | `unexplained` |  | 31.5/34 | 1.5/2 | 0.78/1.00 | 0 |  |
| Malik Nabers | WR | NYG | gooma-s-family-league | 13.7 | 29.1 | -15.4 | `volume_up` |  | 5.7/20 | 0.3/1 | 0.76/0.90 | 0 |  |
| Mike Evans | WR | SF | gooma-s-family-league | 11.1 | 26.3 | -15.2 | `volume_up` |  | 5.7/10 | 0.3/1 | 0.49/0.63 | 3 |  |
| Puka Nacua | WR | LA | gooma-s-family-league | 22.8 | 37.7 | -14.9 | `volume_up` |  | 11.0/26 | 0.5/1 | 0.80/0.85 | -3 |  |

**Did not play, predicted > 5 (`model_components`, 9 players; status layer, not scored)**

Gameday inactive list (nflverse weekly_rosters, posted ~90 min before kickoff; the pre-lock run and serve act on it): 7 of 9 on the gameday inactive list. The rest dressed and did not play, or were never listed.

| player | pos | team | league | predicted | on inactive list |
|---|---|---|---|---|---|
| Caleb Williams | QB | CHI | gooma-s-family-league | 14.8 | yes |
| Jordan Addison | WR | MIN | gooma-s-family-league | 10.9 | yes |
| Rico Dowdle | RB | PIT | gooma-s-family-league | 8.4 | yes |
| DJ Moore | WR | BUF | gooma-s-family-league | 8.0 | yes |
| Ladd McConkey | WR | LAC | gooma-s-family-league | 7.8 | no |
| Ted Hurst III | WR | TB | we-can-think-of-something-funny | 7.0 | no |
| Carson Beck | QB | ARI | we-can-think-of-something-funny | 6.4 | yes |
| Darius Slayton | WR | IND | gooma-s-family-league | 5.7 | yes |
| Devin Singletary | RB | NYG | gooma-s-family-league | 5.4 | yes |

## Candidate signals (derived)

For each reason class, the share of the model's played big misses it explains, the features that would carry the signal, and whether each is served (among the 90 frozen `inputs` of `model/serving_config.json`), pruned (built by `model.features`, dropped by the Phase 2 SHAP prune) or not built. A served feature that still misses means the signal is in the inputs but the week's value was not (`tm_out_*` are read from the report as of serve time: a teammate ruled out after the last run is invisible). A reason whose features are all served is not a new-feature candidate; one whose share is high and whose features are pruned or not built is.

| reason | share of model big misses | features that carry the signal | status in `serving_config.json` inputs |
|---|---|---|---|
| `volume_up` | 47% (16) | `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos`, `opps_trend` | served: `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos` ; not built: `opps_trend` |
| `volume_down` | 6% (2) | `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos`, `opps_trend` | served: `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos` ; not built: `opps_trend` |
| `teammate_out` | 6% (2) | `tm_out_above_n`, `tm_out_group_opps`, `tm_out_targets_all`, `tm_out_carries_all`, `tm_group_size` | served: `tm_out_above_n`, `tm_out_group_opps`, `tm_out_targets_all`, `tm_out_carries_all`, `tm_group_size` |
| `blowout_script` | 0% (0) | `team_spread`, `implied_team_total`, `total_line` | served: `team_spread`, `implied_team_total`, `total_line` |
| `left_early` | 3% (1) | `snap_pct_l1`, `inj_weeks_since_report`, `inj_report_status`, `inj_practice_status` | served: `snap_pct_l1`, `inj_weeks_since_report` ; pruned: `inj_report_status`, `inj_practice_status` |
| `td_variance` | 21% (7) | `td_luck_rush_std`, `td_luck_rec_std`, `td_luck_pass_std`, `td_luck_total_std`, `td_luck_total_prev` | served: `td_luck_rush_std`, `td_luck_rec_std`, `td_luck_pass_std`, `td_luck_total_std`, `td_luck_total_prev` |
| `inactive_missed` | 23 rows, counted apart (13 of 23 on the gameday inactive list) | status layer (`ff.status`, Sleeper `injury_status`, the nflverse report, the gameday inactive list), not a feature | n/a |

## Limits

- Rules, not judgment: thresholds are fixed (35% and 3+ opportunities (QB 6+) for a volume call, 25% or 3 opportunities for 'as expected', snap share < 60% of trailing, margin >= 17, TDs 1.5 from the trailing rate). Priority order: left_early, teammate_out, blowout_script, volume, td_variance. A row can satisfy two; the second is `secondary_reason`.
- Expected volume is the trailing mean of his earlier played weeks this season, not E_opps (not frozen per week) and not the model's own component predictions (not frozen either). A player with no earlier week this season can only be `td_variance`-free `unexplained`, `left_early` (never: no trailing snaps) or `inactive_missed`.
- Teammate out is read from that week's depth chart against `data/player_weeks.csv` and `data/lineups_played.csv`; `data/status.csv` is a snapshot of the upcoming week with no week column and is not used. A teammate who is simply untracked (never a stats row this season) is not counted as out.
- Margins come from `data/defense_vs_pos.csv` (points allowed, `ALL` rows); a week without them has no `blowout_script`.
- Two weeks of misses are a list, not a distribution; read the reason shares after six.
