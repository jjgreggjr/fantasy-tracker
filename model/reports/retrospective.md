# Retrospective

What each comparator got most wrong in every completed week, and why, judged on the scoreboard's identical rows (`python -m model.retrospective`, run after `model.scoreboard`). It explains misses; it is not the adoption gate.

Weeks post-mortemed: 2026 wk4, 2026 wk5, 2026 wk6, 2026 wk7, 2026 wk8, 2026 wk9, 2026 wk10. A big miss is |predicted - actual| >= 12 points in the league's scoring, one row per player-week per comparator (the `gooma-s-family-league` row where he is rostered there, else the first other league alphabetically). Comparators: `model_components` (E_pts_model), `sleeper`, `e_pts`.

## Drift: pick accuracy by week

From `data/model_eval.csv`, pooled over leagues (the scoreboard's number). Cumulative mean and the week-to-week standard deviation of the weekly values.

| comparator | 2026 wk04 | 2026 wk05 | 2026 wk06 | 2026 wk07 | 2026 wk08 | 2026 wk09 | 2026 wk10 | mean | week sd |
|---|---|---|---|---|---|---|---|---|---|
| `model_components` | 0.679 | 0.675 | 0.645 | 0.672 | 0.666 | 0.683 | 0.664 | 0.669 | 0.013 |
| `model_flat` | 0.670 | 0.670 | 0.638 | 0.673 | 0.667 | 0.689 | 0.658 | 0.666 | 0.015 |
| `model_blend` | 0.678 | 0.674 | 0.643 | 0.676 | 0.668 | 0.687 | 0.659 | 0.669 | 0.015 |
| `sleeper` | 0.644 | 0.700 | 0.635 | 0.673 | 0.666 | 0.682 | 0.657 | 0.665 | 0.022 |
| `e_pts` | 0.647 | 0.668 | 0.636 | 0.673 | 0.666 | 0.674 | 0.657 | 0.660 | 0.014 |

No comparator is trending: every span move is inside twice its week-to-week standard deviation.

## Why the big misses happened

Share of each comparator's big misses by reason, all weeks pooled. `inactive_missed` (predicted > 5, did not play) is outside the scored rows and is counted apart: it is the status layer's miss, not the model's. Learnable = the pre-kickoff record could in principle have carried the signal (`volume_up`, `volume_down`, `teammate_out`, `blowout_script`); noise = `td_variance`, `left_early`, `unexplained`.

| comparator | big misses | `left_early` | `teammate_out` | `blowout_script` | `volume_up` | `volume_down` | `td_variance` | `unexplained` | learnable | noise | `inactive_missed` (apart) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `model_components` | 116 | 5 (4%) | 6 (5%) | 2 (2%) | 36 (31%) | 7 (6%) | 28 (24%) | 32 (28%) | 44% | 56% | 46 |
| `sleeper` | 109 | 7 (6%) | 7 (6%) | 3 (3%) | 32 (29%) | 9 (8%) | 26 (24%) | 25 (23%) | 47% | 53% | 30 |
| `e_pts` | 124 | 4 (3%) | 7 (6%) | 3 (2%) | 36 (29%) | 14 (11%) | 27 (22%) | 33 (27%) | 48% | 52% | 32 |

Per week, the 95th percentile of |error| on the scored rows (the `top 5%` rule would have cut here instead of at 12):

| season / week | `model_components` | `sleeper` | `e_pts` |
|---|---|---|---|
| 2026 wk04 | 14.7 | 14.9 | 15.8 |
| 2026 wk05 | 15.4 | 15.2 | 15.7 |
| 2026 wk06 | 12.7 | 14.2 | 13.6 |
| 2026 wk07 | 13.3 | 12.6 | 13.1 |
| 2026 wk08 | 15.5 | 14.7 | 15.2 |
| 2026 wk09 | 14.3 | 14.0 | 14.7 |
| 2026 wk10 | 15.4 | 15.0 | 15.4 |

## Band coverage: actuals against the frozen p10 / p90

Share of played rows (PPR, `data/player_weeks.csv` against `data/model_pts.csv`) inside the frozen band, above p90 and below p10, by position and pooled, all post-mortemed weeks. The band is recalibrated with `serving_config.json` `recalibration.mode`; a tail over 10% is where `underage_by_position` would stretch it (model/ensemble.py).

| position | n | inside [p10, p90] | above p90 | below p10 |
|---|---|---|---|---|
| QB | 237 | 82.7% | 9.7% | 7.6% |
| RB | 509 | 79.2% | 12.6% | 8.3% |
| WR | 870 | 78.6% | 12.6% | 8.7% |
| TE | 514 | 80.7% | 11.3% | 8.0% |
| ALL | 2130 | 79.7% | 12.0% | 8.3% |

Targets: 80% inside, 10% above, 10% below. A tail over 10% is short (stretch it), one under 10% is slack (shrink only on a say-so).

## Top 15 misses, 2026 week 10

Each comparator's largest played misses that week with the evidence behind the reason (expected = trailing mean over his earlier played weeks this season; volume = pass attempts + carries for a QB, carries + targets otherwise; margin = his team's final margin).

**`model_components`** (15 shown)

| player | pos | team | league | predicted | actual | error | reason | secondary | vol exp/act | TD exp/act | snap% trail/act | margin | teammate out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| George Pickens | WR | DAL | gooma-s-family-league | 14.1 | 43.5 | -29.4 | `volume_up` | `td_variance` | 7.0/18 | 0.3/3 | 0.68/0.89 | -6 |  |
| Emmett Johnson | RB | KC | gooma-s-family-league | 8.2 | 35.1 | -26.9 | `teammate_out` | `volume_up` | 5.0/29 | 0.7/2 | 0.25/0.88 | 22 | Kenneth Walker III |
| Christian McCaffrey | RB | SF | gooma-s-family-league | 24.3 | 46.3 | -22.0 | `unexplained` |  | 20.3/17 | 1.7/3 | 0.68/0.60 | 6 |  |
| Jahmyr Gibbs | RB | DET | gooma-s-family-league | 23.6 | 44.2 | -20.6 | `td_variance` |  | 27.7/30 | 0.7/2 | 0.78/0.69 | 25 |  |
| Chuba Hubbard | RB | CAR | gooma-s-family-league | 16.7 | 34.2 | -17.5 | `volume_up` | `td_variance` | 21.0/32 | 0.3/2 | 0.81/0.84 | 3 |  |
| Isaiah Likely | TE | NYG | gooma-s-family-league | 10.4 | 27.7 | -17.3 | `volume_up` |  | 7.0/11 | 0.7/1 | 0.84/0.88 | -3 |  |
| Drake Maye | QB | NE | gooma-s-family-league | 19.3 | 3.0 | 16.3 | `left_early` | `blowout_script` | 34.0/5 | 3.0/0 | 0.94/0.17 | -25 |  |
| Patrick Mahomes | QB | KC | gooma-s-family-league | 19.2 | 35.4 | -16.2 | `td_variance` |  | 37.0/37 | 1.7/5 | 0.98/0.95 | 22 |  |
| Dak Prescott | QB | DAL | gooma-s-family-league | 18.7 | 34.4 | -15.7 | `volume_up` | `td_variance` | 33.7/55 | 2.3/5 | 1.00/1.00 | -6 |  |
| Ja'Marr Chase | WR | CIN | gooma-s-family-league | 18.7 | 34.1 | -15.4 | `unexplained` |  | 11.7/14 | 1.0/0 | 0.89/0.89 | 20 |  |
| Deebo Samuel Sr. | WR | SF | gooma-s-family-league | 10.8 | 25.9 | -15.0 | `td_variance` |  | 5.3/6 | 0.7/2 | 0.62/0.68 | 6 | Mike Evans |
| Chase Brown | RB | CIN | gooma-s-family-league | 14.1 | 28.3 | -14.2 | `td_variance` |  | 16.3/19 | 0.0/2 | 0.69/0.73 | 20 |  |
| Parker Washington | WR | JAX | gooma-s-family-league | 14.0 | 28.2 | -14.2 | `unexplained` |  | 7.0/9 | 0.3/1 | 0.69/0.77 | 17 |  |
| Joe Burrow | QB | CIN | gooma-s-family-league | 19.5 | 33.2 | -13.6 | `unexplained` |  | 38.3/45 | 4.7/4 | 0.98/0.95 | 20 |  |
| Jalen Coker | WR | CAR | gooma-s-family-league | 12.7 | 0.3 | 12.4 | `unexplained` |  | 5.0/5 | 0.3/0 | 0.78/0.80 | 3 |  |

**`sleeper`** (14 shown)

| player | pos | team | league | predicted | actual | error | reason | secondary | vol exp/act | TD exp/act | snap% trail/act | margin | teammate out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| George Pickens | WR | DAL | gooma-s-family-league | 13.4 | 43.5 | -30.1 | `volume_up` | `td_variance` | 7.0/18 | 0.3/3 | 0.68/0.89 | -6 |  |
| Emmett Johnson | RB | KC | gooma-s-family-league | 9.8 | 35.1 | -25.3 | `teammate_out` | `volume_up` | 5.0/29 | 0.7/2 | 0.25/0.88 | 22 | Kenneth Walker III |
| Christian McCaffrey | RB | SF | gooma-s-family-league | 22.6 | 46.3 | -23.7 | `unexplained` |  | 20.3/17 | 1.7/3 | 0.68/0.60 | 6 |  |
| Jahmyr Gibbs | RB | DET | gooma-s-family-league | 24.5 | 44.2 | -19.7 | `td_variance` |  | 27.7/30 | 0.7/2 | 0.78/0.69 | 25 |  |
| Drake Maye | QB | NE | gooma-s-family-league | 22.0 | 3.0 | 18.9 | `left_early` | `blowout_script` | 34.0/5 | 3.0/0 | 0.94/0.17 | -25 |  |
| Patrick Mahomes | QB | KC | gooma-s-family-league | 18.5 | 35.4 | -16.9 | `td_variance` |  | 37.0/37 | 1.7/5 | 0.98/0.95 | 22 |  |
| Dak Prescott | QB | DAL | gooma-s-family-league | 17.9 | 34.4 | -16.5 | `volume_up` | `td_variance` | 33.7/55 | 2.3/5 | 1.00/1.00 | -6 |  |
| Chuba Hubbard | RB | CAR | gooma-s-family-league | 18.0 | 34.2 | -16.2 | `volume_up` | `td_variance` | 21.0/32 | 0.3/2 | 0.81/0.84 | 3 |  |
| Isaiah Likely | TE | NYG | gooma-s-family-league | 12.5 | 27.7 | -15.2 | `volume_up` |  | 7.0/11 | 0.7/1 | 0.84/0.88 | -3 |  |
| Deebo Samuel Sr. | WR | SF | gooma-s-family-league | 11.0 | 25.9 | -14.9 | `td_variance` |  | 5.3/6 | 0.7/2 | 0.62/0.68 | 6 | Mike Evans |
| Chase Brown | RB | CIN | gooma-s-family-league | 13.8 | 28.3 | -14.5 | `td_variance` |  | 16.3/19 | 0.0/2 | 0.69/0.73 | 20 |  |
| Ja'Marr Chase | WR | CIN | gooma-s-family-league | 19.9 | 34.1 | -14.2 | `unexplained` |  | 11.7/14 | 1.0/0 | 0.89/0.89 | 20 |  |
| Parker Washington | WR | JAX | gooma-s-family-league | 14.7 | 28.2 | -13.5 | `unexplained` |  | 7.0/9 | 0.3/1 | 0.69/0.77 | 17 |  |
| Jalen Coker | WR | CAR | gooma-s-family-league | 12.9 | 0.3 | 12.6 | `unexplained` |  | 5.0/5 | 0.3/0 | 0.78/0.80 | 3 |  |

**`e_pts`** (15 shown)

| player | pos | team | league | predicted | actual | error | reason | secondary | vol exp/act | TD exp/act | snap% trail/act | margin | teammate out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| George Pickens | WR | DAL | gooma-s-family-league | 12.4 | 43.5 | -31.1 | `volume_up` | `td_variance` | 7.0/18 | 0.3/3 | 0.68/0.89 | -6 |  |
| Emmett Johnson | RB | KC | gooma-s-family-league | 10.0 | 35.1 | -25.1 | `teammate_out` | `volume_up` | 5.0/29 | 0.7/2 | 0.25/0.88 | 22 | Kenneth Walker III |
| Jahmyr Gibbs | RB | DET | gooma-s-family-league | 21.7 | 44.2 | -22.5 | `td_variance` |  | 27.7/30 | 0.7/2 | 0.78/0.69 | 25 |  |
| Christian McCaffrey | RB | SF | gooma-s-family-league | 23.9 | 46.3 | -22.4 | `unexplained` |  | 20.3/17 | 1.7/3 | 0.68/0.60 | 6 |  |
| Patrick Mahomes | QB | KC | gooma-s-family-league | 17.6 | 35.4 | -17.8 | `td_variance` |  | 37.0/37 | 1.7/5 | 0.98/0.95 | 22 |  |
| Drake Maye | QB | NE | gooma-s-family-league | 20.0 | 3.0 | 17.0 | `left_early` | `blowout_script` | 34.0/5 | 3.0/0 | 0.94/0.17 | -25 |  |
| Chuba Hubbard | RB | CAR | gooma-s-family-league | 17.6 | 34.2 | -16.6 | `volume_up` | `td_variance` | 21.0/32 | 0.3/2 | 0.81/0.84 | 3 |  |
| Dak Prescott | QB | DAL | gooma-s-family-league | 18.0 | 34.4 | -16.4 | `volume_up` | `td_variance` | 33.7/55 | 2.3/5 | 1.00/1.00 | -6 |  |
| Deebo Samuel Sr. | WR | SF | gooma-s-family-league | 10.1 | 25.9 | -15.8 | `td_variance` |  | 5.3/6 | 0.7/2 | 0.62/0.68 | 6 | Mike Evans |
| Ja'Marr Chase | WR | CIN | gooma-s-family-league | 18.8 | 34.1 | -15.3 | `unexplained` |  | 11.7/14 | 1.0/0 | 0.89/0.89 | 20 |  |
| Isaiah Likely | TE | NYG | gooma-s-family-league | 12.4 | 27.7 | -15.3 | `volume_up` |  | 7.0/11 | 0.7/1 | 0.84/0.88 | -3 |  |
| Chase Brown | RB | CIN | gooma-s-family-league | 14.2 | 28.3 | -14.1 | `td_variance` |  | 16.3/19 | 0.0/2 | 0.69/0.73 | 20 |  |
| Joe Burrow | QB | CIN | gooma-s-family-league | 19.1 | 33.2 | -14.1 | `unexplained` |  | 38.3/45 | 4.7/4 | 0.98/0.95 | 20 |  |
| Parker Washington | WR | JAX | gooma-s-family-league | 15.4 | 28.2 | -12.8 | `unexplained` |  | 7.0/9 | 0.3/1 | 0.69/0.77 | 17 |  |
| Jalen Coker | WR | CAR | gooma-s-family-league | 13.0 | 0.3 | 12.7 | `unexplained` |  | 5.0/5 | 0.3/0 | 0.78/0.80 | 3 |  |

**Did not play, predicted > 5 (`model_components`, 7 players; status layer, not scored)**

Gameday inactive list (nflverse weekly_rosters, posted ~90 min before kickoff; the pre-lock run and serve act on it): 4 of 7 on the gameday inactive list. The rest dressed and did not play, or were never listed.

| player | pos | team | league | predicted | on inactive list |
|---|---|---|---|---|---|
| Kenneth Walker III | RB | KC | gooma-s-family-league | 17.9 | yes |
| Jordan Addison | WR | MIN | gooma-s-family-league | 11.0 | yes |
| Ladd McConkey | WR | LAC | gooma-s-family-league | 9.8 | yes |
| Omar Cooper Jr. | WR | NYJ | gooma-s-family-league | 9.2 | no |
| David Montgomery | RB | HOU | gooma-s-family-league | 7.8 | yes |
| Demarcus Robinson | WR | SF | gooma-s-family-league | 7.3 | no |
| Reggie Virgil | WR | ARI | we-can-think-of-something-funny | 5.9 | no |

## Candidate signals (derived)

For each reason class, the share of the model's played big misses it explains, the features that would carry the signal, and whether each is served (among the 90 frozen `inputs` of `model/serving_config.json`), pruned (built by `model.features`, dropped by the Phase 2 SHAP prune) or not built. A served feature that still misses means the signal is in the inputs but the week's value was not (`tm_out_*` are read from the report as of serve time: a teammate ruled out after the last run is invisible). A reason whose features are all served is not a new-feature candidate; one whose share is high and whose features are pruned or not built is.

| reason | share of model big misses | features that carry the signal | status in `serving_config.json` inputs |
|---|---|---|---|
| `volume_up` | 31% (36) | `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos`, `opps_trend` | served: `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos` ; not built: `opps_trend` |
| `volume_down` | 6% (7) | `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos`, `opps_trend` | served: `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos` ; not built: `opps_trend` |
| `teammate_out` | 5% (6) | `tm_out_above_n`, `tm_out_group_opps`, `tm_out_targets_all`, `tm_out_carries_all`, `tm_group_size` | served: `tm_out_above_n`, `tm_out_group_opps`, `tm_out_targets_all`, `tm_out_carries_all`, `tm_group_size` |
| `blowout_script` | 2% (2) | `team_spread`, `implied_team_total`, `total_line` | served: `team_spread`, `implied_team_total`, `total_line` |
| `left_early` | 4% (5) | `snap_pct_l1`, `inj_weeks_since_report`, `inj_report_status`, `inj_practice_status` | served: `snap_pct_l1`, `inj_weeks_since_report` ; pruned: `inj_report_status`, `inj_practice_status` |
| `td_variance` | 24% (28) | `td_luck_rush_std`, `td_luck_rec_std`, `td_luck_pass_std`, `td_luck_total_std`, `td_luck_total_prev` | served: `td_luck_rush_std`, `td_luck_rec_std`, `td_luck_pass_std`, `td_luck_total_std`, `td_luck_total_prev` |
| `inactive_missed` | 46 rows, counted apart (24 of 46 on the gameday inactive list) | status layer (`ff.status`, Sleeper `injury_status`, the nflverse report, the gameday inactive list), not a feature | n/a |

## Limits

- Rules, not judgment: thresholds are fixed (35% and 3+ opportunities (QB 6+) for a volume call, 25% or 3 opportunities for 'as expected', snap share < 60% of trailing, margin >= 17, TDs 1.5 from the trailing rate). Priority order: left_early, teammate_out, blowout_script, volume, td_variance. A row can satisfy two; the second is `secondary_reason`.
- Expected volume is the trailing mean of his earlier played weeks this season, not E_opps (not frozen per week) and not the model's own component predictions (not frozen either). A player with no earlier week this season can only be `td_variance`-free `unexplained`, `left_early` (never: no trailing snaps) or `inactive_missed`.
- Teammate out is read from that week's depth chart against `data/player_weeks.csv` and `data/lineups_played.csv`; `data/status.csv` is a snapshot of the upcoming week with no week column and is not used. A teammate who is simply untracked (never a stats row this season) is not counted as out.
- Margins come from `data/defense_vs_pos.csv` (points allowed, `ALL` rows); a week without them has no `blowout_script`.
- Two weeks of misses are a list, not a distribution; read the reason shares after six.
