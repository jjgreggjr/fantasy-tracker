# Retrospective

What each comparator got most wrong in every completed week, and why, judged on the scoreboard's identical rows (`python -m model.retrospective`, run after `model.scoreboard`). It explains misses; it is not the adoption gate.

Weeks post-mortemed: 2026 wk4, 2026 wk5, 2026 wk6. A big miss is |predicted - actual| >= 12 points in the league's scoring, one row per player-week per comparator (the `gooma-s-family-league` row where he is rostered there, else the first other league alphabetically). Comparators: `model_components` (E_pts_model), `sleeper`, `e_pts`.

## Drift: pick accuracy by week

From `data/model_eval.csv`, pooled over leagues (the scoreboard's number). Cumulative mean and the week-to-week standard deviation of the weekly values.

| comparator | 2026 wk04 | 2026 wk05 | 2026 wk06 | mean | week sd |
|---|---|---|---|---|---|
| `model_components` | 0.679 | 0.675 | 0.645 | 0.666 | 0.019 |
| `model_flat` | 0.670 | 0.670 | 0.638 | 0.659 | 0.018 |
| `model_blend` | 0.678 | 0.674 | 0.643 | 0.665 | 0.019 |
| `sleeper` | 0.644 | 0.700 | 0.635 | 0.660 | 0.035 |
| `e_pts` | 0.647 | 0.668 | 0.636 | 0.650 | 0.016 |

3 week(s) scored: too few to call a trend for any comparator (a slope needs 4+ weeks, and the week-to-week standard deviation above is the size of ordinary noise).

## Why the big misses happened

Share of each comparator's big misses by reason, all weeks pooled. `inactive_missed` (predicted > 5, did not play) is outside the scored rows and is counted apart: it is the status layer's miss, not the model's. Learnable = the pre-kickoff record could in principle have carried the signal (`volume_up`, `volume_down`, `teammate_out`, `blowout_script`); noise = `td_variance`, `left_early`, `unexplained`.

| comparator | big misses | `left_early` | `teammate_out` | `blowout_script` | `volume_up` | `volume_down` | `td_variance` | `unexplained` | learnable | noise | `inactive_missed` (apart) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `model_components` | 48 | 1 (2%) | 4 (8%) | 0 (0%) | 20 (42%) | 3 (6%) | 9 (19%) | 11 (23%) | 56% | 44% | 26 |
| `sleeper` | 48 | 3 (6%) | 4 (8%) | 0 (0%) | 18 (38%) | 5 (10%) | 9 (19%) | 9 (19%) | 56% | 44% | 13 |
| `e_pts` | 56 | 1 (2%) | 4 (7%) | 1 (2%) | 20 (36%) | 8 (14%) | 9 (16%) | 13 (23%) | 59% | 41% | 12 |

Per week, the 95th percentile of |error| on the scored rows (the `top 5%` rule would have cut here instead of at 12):

| season / week | `model_components` | `sleeper` | `e_pts` |
|---|---|---|---|
| 2026 wk04 | 14.7 | 14.9 | 15.8 |
| 2026 wk05 | 15.4 | 15.2 | 15.7 |
| 2026 wk06 | 12.7 | 14.2 | 13.6 |

## Top 15 misses, 2026 week 6

Each comparator's largest played misses that week with the evidence behind the reason (expected = trailing mean over his earlier played weeks this season; volume = pass attempts + carries for a QB, carries + targets otherwise; margin = his team's final margin).

**`model_components`** (14 shown)

| player | pos | team | league | predicted | actual | error | reason | secondary | vol exp/act | TD exp/act | snap% trail/act | margin | teammate out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Omar Cooper Jr. | WR | NYJ | gooma-s-family-league | 2.4 | 22.3 | -19.9 | `teammate_out` | `volume_up` | 1.0/7 | 0.0/1 | 0.07/0.82 | -7 | Adonai Mitchell |
| Germie Bernard | WR | PIT | gooma-s-family-league | 7.9 | 25.7 | -17.8 | `volume_up` |  | 4.3/12 | 0.0/1 | 0.53/0.94 | 7 |  |
| Jaylin Noel | WR | HOU | we-can-think-of-something-funny | 6.3 | 22.7 | -16.4 | `unexplained` |  | 4.3/4 | 0.3/1 | 0.44/0.49 | 3 |  |
| Bijan Robinson | RB | ATL | gooma-s-family-league | 23.6 | 7.3 | 16.3 | `volume_down` |  | 25.3/11 | 1.3/0 | 0.71/0.78 | -8 |  |
| Tre Tucker | WR | LV | gooma-s-family-league | 11.0 | 26.7 | -15.7 | `td_variance` |  | 6.3/7 | 0.0/2 | 0.66/0.70 | -3 |  |
| Travis Hunter | WR | JAX | james-gregg-espn | 1.6 | 15.9 | -14.3 | `unexplained` |  | n/a/n/a | n/a/n/a | n/a/n/a | -3 |  |
| Kyle Pitts | TE | ATL | gooma-s-family-league | 9.3 | 23.0 | -13.7 | `volume_up` | `td_variance` | 3.7/7 | 0.3/2 | 0.59/0.78 | -8 |  |
| Tyrone Tracy Jr. | RB | NYG | gooma-s-family-league | 5.1 | 18.5 | -13.4 | `teammate_out` | `volume_up` | 2.7/9 | 0.0/2 | 0.13/0.38 | -7 | Cam Skattebo |
| DeMario Douglas | WR | NE | gooma-s-family-league | 6.6 | 19.6 | -13.0 | `volume_up` |  | 4.3/8 | 0.7/1 | 0.51/0.59 | 7 |  |
| Bhayshul Tuten | RB | JAX | gooma-s-family-league | 12.0 | 24.7 | -12.7 | `td_variance` |  | 20.3/16 | 0.3/2 | 0.58/0.58 | -3 |  |
| Malik Nabers | WR | NYG | gooma-s-family-league | 15.8 | 28.5 | -12.7 | `volume_up` |  | 11.0/15 | 0.7/0 | 0.81/0.86 | -7 |  |
| Derrick Henry | RB | BAL | gooma-s-family-league | 17.6 | 5.0 | 12.6 | `unexplained` |  | 26.0/18 | 1.3/0 | 0.61/0.58 | -3 |  |
| MarShawn Lloyd | RB | GB | gooma-s-family-league | 13.6 | 1.0 | 12.6 | `unexplained` |  | 10.3/9 | 0.7/0 | 0.41/0.40 | 10 |  |
| Isaiah Likely | TE | NYG | gooma-s-family-league | 11.4 | 23.9 | -12.5 | `unexplained` |  | 10.0/11 | 0.7/1 | 0.77/0.91 | -7 |  |

**`sleeper`** (14 shown)

| player | pos | team | league | predicted | actual | error | reason | secondary | vol exp/act | TD exp/act | snap% trail/act | margin | teammate out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Bijan Robinson | RB | ATL | gooma-s-family-league | 25.2 | 7.3 | 17.9 | `volume_down` |  | 25.3/11 | 1.3/0 | 0.71/0.78 | -8 |  |
| Tre Tucker | WR | LV | gooma-s-family-league | 9.1 | 26.7 | -17.6 | `td_variance` |  | 6.3/7 | 0.0/2 | 0.66/0.70 | -3 |  |
| Jaylin Noel | WR | HOU | we-can-think-of-something-funny | 5.5 | 22.7 | -17.2 | `unexplained` |  | 4.3/4 | 0.3/1 | 0.44/0.49 | 3 |  |
| Omar Cooper Jr. | WR | NYJ | gooma-s-family-league | 5.9 | 22.3 | -16.4 | `teammate_out` | `volume_up` | 1.0/7 | 0.0/1 | 0.07/0.82 | -7 | Adonai Mitchell |
| Germie Bernard | WR | PIT | gooma-s-family-league | 10.1 | 25.7 | -15.6 | `volume_up` |  | 4.3/12 | 0.0/1 | 0.53/0.94 | 7 |  |
| J.J. McCarthy | QB | NYG | where-you-at | 0.9 | 16.3 | -15.5 | `volume_up` |  | 2.0/23 | 0.0/1 | 0.06/0.40 | -7 |  |
| Kyle Pitts | TE | ATL | gooma-s-family-league | 7.6 | 23.0 | -15.4 | `volume_up` | `td_variance` | 3.7/7 | 0.3/2 | 0.59/0.78 | -8 |  |
| DeMario Douglas | WR | NE | gooma-s-family-league | 4.2 | 19.6 | -15.4 | `volume_up` |  | 4.3/8 | 0.7/1 | 0.51/0.59 | 7 |  |
| Tyrone Tracy Jr. | RB | NYG | gooma-s-family-league | 3.9 | 18.5 | -14.6 | `teammate_out` | `volume_up` | 2.7/9 | 0.0/2 | 0.13/0.38 | -7 | Cam Skattebo |
| Travis Hunter | WR | JAX | james-gregg-espn | 1.6 | 15.9 | -14.3 | `unexplained` |  | n/a/n/a | n/a/n/a | n/a/n/a | -3 |  |
| Malik Nabers | WR | NYG | gooma-s-family-league | 14.9 | 28.5 | -13.6 | `volume_up` |  | 11.0/15 | 0.7/0 | 0.81/0.86 | -7 |  |
| Jahan Dotson | WR | ATL | where-you-at | 3.0 | 16.2 | -13.2 | `unexplained` |  | 3.3/5 | 0.3/1 | 0.65/0.85 | -8 |  |
| Derrick Henry | RB | BAL | gooma-s-family-league | 17.9 | 5.0 | 12.9 | `unexplained` |  | 26.0/18 | 1.3/0 | 0.61/0.58 | -3 |  |
| Bhayshul Tuten | RB | JAX | gooma-s-family-league | 11.9 | 24.7 | -12.8 | `td_variance` |  | 20.3/16 | 0.3/2 | 0.58/0.58 | -3 |  |

**`e_pts`** (15 shown)

| player | pos | team | league | predicted | actual | error | reason | secondary | vol exp/act | TD exp/act | snap% trail/act | margin | teammate out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Germie Bernard | WR | PIT | gooma-s-family-league | 8.6 | 25.7 | -17.1 | `volume_up` |  | 4.3/12 | 0.0/1 | 0.53/0.94 | 7 |  |
| Tre Tucker | WR | LV | gooma-s-family-league | 9.6 | 26.7 | -17.1 | `td_variance` |  | 6.3/7 | 0.0/2 | 0.66/0.70 | -3 |  |
| Omar Cooper Jr. | WR | NYJ | gooma-s-family-league | 5.9 | 22.3 | -16.4 | `teammate_out` | `volume_up` | 1.0/7 | 0.0/1 | 0.07/0.82 | -7 | Adonai Mitchell |
| Kyle Pitts | TE | ATL | gooma-s-family-league | 7.1 | 23.0 | -15.9 | `volume_up` | `td_variance` | 3.7/7 | 0.3/2 | 0.59/0.78 | -8 |  |
| Bijan Robinson | RB | ATL | gooma-s-family-league | 22.9 | 7.3 | 15.6 | `volume_down` |  | 25.3/11 | 1.3/0 | 0.71/0.78 | -8 |  |
| Tyrone Tracy Jr. | RB | NYG | gooma-s-family-league | 3.2 | 18.5 | -15.3 | `teammate_out` | `volume_up` | 2.7/9 | 0.0/2 | 0.13/0.38 | -7 | Cam Skattebo |
| Kyren Williams | RB | LA | gooma-s-family-league | 24.6 | 9.7 | 14.9 | `volume_down` | `td_variance` | 25.0/16 | 1.0/0 | 0.76/0.72 | -2 |  |
| J.J. McCarthy | QB | NYG | where-you-at | 2.1 | 16.3 | -14.2 | `volume_up` |  | 2.0/23 | 0.0/1 | 0.06/0.40 | -7 |  |
| Travis Hunter | WR | JAX | james-gregg-espn | 1.7 | 15.9 | -14.2 | `unexplained` |  | n/a/n/a | n/a/n/a | n/a/n/a | -3 |  |
| Jaylin Noel | WR | HOU | we-can-think-of-something-funny | 8.8 | 22.7 | -13.9 | `unexplained` |  | 4.3/4 | 0.3/1 | 0.44/0.49 | 3 |  |
| Derrick Henry | RB | BAL | gooma-s-family-league | 17.8 | 5.0 | 12.8 | `unexplained` |  | 26.0/18 | 1.3/0 | 0.61/0.58 | -3 |  |
| DeMario Douglas | WR | NE | gooma-s-family-league | 7.0 | 19.6 | -12.6 | `volume_up` |  | 4.3/8 | 0.7/1 | 0.51/0.59 | 7 |  |
| Juwan Johnson | TE | NO | gooma-s-family-league | 12.6 | 0.0 | 12.6 | `volume_down` |  | 7.0/2 | 0.7/0 | 0.68/0.58 | 7 |  |
| Rhamondre Stevenson | RB | NE | gooma-s-family-league | 16.1 | 3.6 | 12.5 | `unexplained` |  | 14.7/14 | 0.7/0 | 0.63/0.60 | 7 |  |
| Malik Nabers | WR | NYG | gooma-s-family-league | 16.3 | 28.5 | -12.2 | `volume_up` |  | 11.0/15 | 0.7/0 | 0.81/0.86 | -7 |  |

**Did not play, predicted > 5 (`model_components`, 3 players; status layer, not scored)**

Gameday inactive list (nflverse weekly_rosters, posted ~90 min before kickoff; the pre-lock run and serve act on it): 1 of 3 on the gameday inactive list. The rest dressed and did not play, or were never listed.

| player | pos | team | league | predicted | on inactive list |
|---|---|---|---|---|---|
| Cam Skattebo | RB | NYG | gooma-s-family-league | 12.9 | no |
| DJ Moore | WR | BUF | gooma-s-family-league | 11.3 | yes |
| Tua Tagovailoa | QB | ATL | gooma-s-family-league | 5.4 | no |

## Candidate signals (derived)

For each reason class, the share of the model's played big misses it explains, the features that would carry the signal, and whether each is served (among the 90 frozen `inputs` of `model/serving_config.json`), pruned (built by `model.features`, dropped by the Phase 2 SHAP prune) or not built. A served feature that still misses means the signal is in the inputs but the week's value was not (`tm_out_*` are read from the report as of serve time: a teammate ruled out after the last run is invisible). A reason whose features are all served is not a new-feature candidate; one whose share is high and whose features are pruned or not built is.

| reason | share of model big misses | features that carry the signal | status in `serving_config.json` inputs |
|---|---|---|---|
| `volume_up` | 42% (20) | `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos`, `opps_trend` | served: `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos` ; not built: `opps_trend` |
| `volume_down` | 6% (3) | `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos`, `opps_trend` | served: `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos` ; not built: `opps_trend` |
| `teammate_out` | 8% (4) | `tm_out_above_n`, `tm_out_group_opps`, `tm_out_targets_all`, `tm_out_carries_all`, `tm_group_size` | served: `tm_out_above_n`, `tm_out_group_opps`, `tm_out_targets_all`, `tm_out_carries_all`, `tm_group_size` |
| `blowout_script` | 0% (0) | `team_spread`, `implied_team_total`, `total_line` | served: `team_spread`, `implied_team_total`, `total_line` |
| `left_early` | 2% (1) | `snap_pct_l1`, `inj_weeks_since_report`, `inj_report_status`, `inj_practice_status` | served: `snap_pct_l1`, `inj_weeks_since_report` ; pruned: `inj_report_status`, `inj_practice_status` |
| `td_variance` | 19% (9) | `td_luck_rush_std`, `td_luck_rec_std`, `td_luck_pass_std`, `td_luck_total_std`, `td_luck_total_prev` | served: `td_luck_rush_std`, `td_luck_rec_std`, `td_luck_pass_std`, `td_luck_total_std`, `td_luck_total_prev` |
| `inactive_missed` | 26 rows, counted apart (14 of 26 on the gameday inactive list) | status layer (`ff.status`, Sleeper `injury_status`, the nflverse report, the gameday inactive list), not a feature | n/a |

## Limits

- Rules, not judgment: thresholds are fixed (35% and 3+ opportunities (QB 6+) for a volume call, 25% or 3 opportunities for 'as expected', snap share < 60% of trailing, margin >= 17, TDs 1.5 from the trailing rate). Priority order: left_early, teammate_out, blowout_script, volume, td_variance. A row can satisfy two; the second is `secondary_reason`.
- Expected volume is the trailing mean of his earlier played weeks this season, not E_opps (not frozen per week) and not the model's own component predictions (not frozen either). A player with no earlier week this season can only be `td_variance`-free `unexplained`, `left_early` (never: no trailing snaps) or `inactive_missed`.
- Teammate out is read from that week's depth chart against `data/player_weeks.csv` and `data/lineups_played.csv`; `data/status.csv` is a snapshot of the upcoming week with no week column and is not used. A teammate who is simply untracked (never a stats row this season) is not counted as out.
- Margins come from `data/defense_vs_pos.csv` (points allowed, `ALL` rows); a week without them has no `blowout_script`.
- Two weeks of misses are a list, not a distribution; read the reason shares after six.
