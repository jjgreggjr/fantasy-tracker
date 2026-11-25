# Retrospective

What each comparator got most wrong in every completed week, and why, judged on the scoreboard's identical rows (`python -m model.retrospective`, run after `model.scoreboard`). It explains misses; it is not the adoption gate.

Weeks post-mortemed: 2026 wk4, 2026 wk5, 2026 wk6, 2026 wk7, 2026 wk8, 2026 wk9, 2026 wk10, 2026 wk11. A big miss is |predicted - actual| >= 12 points in the league's scoring, one row per player-week per comparator (the `gooma-s-family-league` row where he is rostered there, else the first other league alphabetically). Comparators: `model_components` (E_pts_model), `sleeper`, `e_pts`.

## Drift: pick accuracy by week

From `data/model_eval.csv`, pooled over leagues (the scoreboard's number). Cumulative mean and the week-to-week standard deviation of the weekly values.

| comparator | 2026 wk04 | 2026 wk05 | 2026 wk06 | 2026 wk07 | 2026 wk08 | 2026 wk09 | 2026 wk10 | 2026 wk11 | mean | week sd |
|---|---|---|---|---|---|---|---|---|---|---|
| `model_components` | 0.679 | 0.675 | 0.645 | 0.672 | 0.666 | 0.683 | 0.664 | 0.658 | 0.668 | 0.012 |
| `model_flat` | 0.670 | 0.670 | 0.638 | 0.673 | 0.667 | 0.689 | 0.658 | 0.661 | 0.666 | 0.014 |
| `model_blend` | 0.678 | 0.674 | 0.643 | 0.676 | 0.668 | 0.687 | 0.659 | 0.658 | 0.668 | 0.014 |
| `sleeper` | 0.644 | 0.700 | 0.635 | 0.673 | 0.666 | 0.682 | 0.657 | 0.664 | 0.665 | 0.021 |
| `e_pts` | 0.647 | 0.668 | 0.636 | 0.673 | 0.666 | 0.674 | 0.657 | 0.654 | 0.659 | 0.013 |

No comparator is trending: every span move is inside twice its week-to-week standard deviation.

## Why the big misses happened

Share of each comparator's big misses by reason, all weeks pooled. `inactive_missed` (predicted > 5, did not play) is outside the scored rows and is counted apart: it is the status layer's miss, not the model's. Learnable = the pre-kickoff record could in principle have carried the signal (`volume_up`, `volume_down`, `teammate_out`, `blowout_script`); noise = `td_variance`, `left_early`, `unexplained`.

| comparator | big misses | `left_early` | `teammate_out` | `blowout_script` | `volume_up` | `volume_down` | `td_variance` | `unexplained` | learnable | noise | `inactive_missed` (apart) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `model_components` | 122 | 5 (4%) | 6 (5%) | 2 (2%) | 40 (33%) | 8 (7%) | 29 (24%) | 32 (26%) | 46% | 54% | 53 |
| `sleeper` | 117 | 7 (6%) | 7 (6%) | 3 (3%) | 37 (32%) | 11 (9%) | 27 (23%) | 25 (21%) | 50% | 50% | 35 |
| `e_pts` | 133 | 4 (3%) | 8 (6%) | 3 (2%) | 41 (31%) | 15 (11%) | 28 (21%) | 34 (26%) | 50% | 50% | 41 |

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
| 2026 wk11 | 11.6 | 11.6 | 11.9 |

## Band coverage: actuals against the frozen p10 / p90

Share of played rows (PPR, `data/player_weeks.csv` against `data/model_pts.csv`) inside the frozen band, above p90 and below p10, by position and pooled, all post-mortemed weeks. The band is recalibrated with `serving_config.json` `recalibration.mode`; a tail over 10% is where `underage_by_position` would stretch it (model/ensemble.py).

| position | n | inside [p10, p90] | above p90 | below p10 |
|---|---|---|---|---|
| QB | 269 | 83.6% | 9.7% | 6.7% |
| RB | 572 | 80.1% | 11.7% | 8.2% |
| WR | 985 | 78.9% | 12.5% | 8.6% |
| TE | 576 | 80.9% | 11.5% | 7.6% |
| ALL | 2402 | 80.2% | 11.7% | 8.1% |

Targets: 80% inside, 10% above, 10% below. A tail over 10% is short (stretch it), one under 10% is slack (shrink only on a say-so).

## Top 15 misses, 2026 week 11

Each comparator's largest played misses that week with the evidence behind the reason (expected = trailing mean over his earlier played weeks this season; volume = pass attempts + carries for a QB, carries + targets otherwise; margin = his team's final margin).

**`model_components`** (6 shown)

| player | pos | team | league | predicted | actual | error | reason | secondary | vol exp/act | TD exp/act | snap% trail/act | margin | teammate out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jonah Coleman | RB | DEN | gooma-s-family-league | 14.9 | 34.9 | -20.0 | `volume_up` | `td_variance` | 16.0/29 | 0.7/3 | 0.64/0.85 | 14 |  |
| Ted Hurst III | WR | TB | gooma-s-family-league | 2.9 | 22.6 | -19.7 | `volume_up` | `td_variance` | 1.0/7 | 0.0/2 | 0.33/0.64 | -10 |  |
| Dak Prescott | QB | DAL | gooma-s-family-league | 20.3 | 35.9 | -15.6 | `td_variance` |  | 42.3/37 | 3.0/5 | 1.00/1.00 | 21 |  |
| Ashton Jeanty | RB | LV | gooma-s-family-league | 13.2 | 27.9 | -14.7 | `volume_up` |  | 10.7/19 | 0.0/1 | 0.58/0.84 | -14 |  |
| Jack Bech | WR | LV | where-you-at | 4.2 | 18.8 | -14.6 | `volume_up` |  | 3.7/9 | 0.0/1 | 0.48/0.79 | -14 |  |
| Ja'Marr Chase | WR | CIN | gooma-s-family-league | 20.6 | 8.1 | 12.5 | `volume_down` |  | 14.0/9 | 0.7/0 | 0.90/0.97 | -2 |  |

**`sleeper`** (8 shown)

| player | pos | team | league | predicted | actual | error | reason | secondary | vol exp/act | TD exp/act | snap% trail/act | margin | teammate out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ted Hurst III | WR | TB | gooma-s-family-league | 1.8 | 22.6 | -20.9 | `volume_up` | `td_variance` | 1.0/7 | 0.0/2 | 0.33/0.64 | -10 |  |
| Jonah Coleman | RB | DEN | gooma-s-family-league | 17.4 | 34.9 | -17.5 | `volume_up` | `td_variance` | 16.0/29 | 0.7/3 | 0.64/0.85 | 14 |  |
| Ashton Jeanty | RB | LV | gooma-s-family-league | 12.9 | 27.9 | -15.0 | `volume_up` |  | 10.7/19 | 0.0/1 | 0.58/0.84 | -14 |  |
| Dak Prescott | QB | DAL | gooma-s-family-league | 21.1 | 35.9 | -14.8 | `td_variance` |  | 42.3/37 | 3.0/5 | 1.00/1.00 | 21 |  |
| Jack Bech | WR | LV | where-you-at | 4.4 | 18.8 | -14.4 | `volume_up` |  | 3.7/9 | 0.0/1 | 0.48/0.79 | -14 |  |
| Ja'Marr Chase | WR | CIN | gooma-s-family-league | 21.9 | 8.1 | 13.8 | `volume_down` |  | 14.0/9 | 0.7/0 | 0.90/0.97 | -2 |  |
| Antonio Williams | WR | WAS | gooma-s-family-league | 8.2 | 21.2 | -13.0 | `volume_up` |  | 5.0/11 | 0.3/0 | 0.84/0.84 | 2 |  |
| Luther Burden III | WR | CHI | gooma-s-family-league | 15.3 | 3.3 | 12.0 | `volume_down` |  | 8.0/4 | 1.0/0 | 0.58/0.59 | 8 |  |

**`e_pts`** (9 shown)

| player | pos | team | league | predicted | actual | error | reason | secondary | vol exp/act | TD exp/act | snap% trail/act | margin | teammate out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ted Hurst III | WR | TB | gooma-s-family-league | 1.6 | 22.6 | -21.0 | `volume_up` | `td_variance` | 1.0/7 | 0.0/2 | 0.33/0.64 | -10 |  |
| Jonah Coleman | RB | DEN | gooma-s-family-league | 17.4 | 34.9 | -17.5 | `volume_up` | `td_variance` | 16.0/29 | 0.7/3 | 0.64/0.85 | 14 |  |
| Jack Bech | WR | LV | where-you-at | 4.0 | 18.8 | -14.8 | `volume_up` |  | 3.7/9 | 0.0/1 | 0.48/0.79 | -14 |  |
| Ashton Jeanty | RB | LV | gooma-s-family-league | 14.1 | 27.9 | -13.8 | `volume_up` |  | 10.7/19 | 0.0/1 | 0.58/0.84 | -14 |  |
| Antonio Williams | WR | WAS | gooma-s-family-league | 7.8 | 21.2 | -13.4 | `volume_up` |  | 5.0/11 | 0.3/0 | 0.84/0.84 | 2 |  |
| Dak Prescott | QB | DAL | gooma-s-family-league | 23.3 | 35.9 | -12.6 | `td_variance` |  | 42.3/37 | 3.0/5 | 1.00/1.00 | 21 |  |
| Emmett Johnson | RB | KC | gooma-s-family-league | 11.8 | 24.3 | -12.5 | `teammate_out` | `volume_up` | 13.7/27 | 1.3/1 | 0.46/0.82 | 27 | Kenneth Walker III |
| Ja'Marr Chase | WR | CIN | gooma-s-family-league | 20.5 | 8.1 | 12.4 | `volume_down` |  | 14.0/9 | 0.7/0 | 0.90/0.97 | -2 |  |
| Michael Mayer | TE | LV | gooma-s-family-league | 13.5 | 1.4 | 12.1 | `unexplained` |  | 4.3/4 | 0.0/0 | 0.79/0.67 | -14 | Brock Bowers |

**Did not play, predicted > 5 (`model_components`, 7 players; status layer, not scored)**

Gameday inactive list (nflverse weekly_rosters, posted ~90 min before kickoff; the pre-lock run and serve act on it): 4 of 7 on the gameday inactive list. The rest dressed and did not play, or were never listed.

| player | pos | team | league | predicted | on inactive list |
|---|---|---|---|---|---|
| Carson Beck | QB | ARI | we-can-think-of-something-funny | 10.4 | yes |
| Dalton Kincaid | TE | BUF | gooma-s-family-league | 8.8 | yes |
| Ja'Kobi Lane | WR | BAL | gooma-s-family-league | 7.5 | no |
| Tyjae Spears | RB | TEN | gooma-s-family-league | 6.8 | no |
| Oronde Gadsden II | TE | LAC | we-can-think-of-something-funny | 5.9 | no |
| Brock Bowers | TE | LV | gooma-s-family-league | 5.3 | yes |
| Audric Estime | RB | NO | james-gregg-espn | 5.1 | yes |

## Candidate signals (derived)

For each reason class, the share of the model's played big misses it explains, the features that would carry the signal, and whether each is served (among the 90 frozen `inputs` of `model/serving_config.json`), pruned (built by `model.features`, dropped by the Phase 2 SHAP prune) or not built. A served feature that still misses means the signal is in the inputs but the week's value was not (`tm_out_*` are read from the report as of serve time: a teammate ruled out after the last run is invisible). A reason whose features are all served is not a new-feature candidate; one whose share is high and whose features are pruned or not built is.

| reason | share of model big misses | features that carry the signal | status in `serving_config.json` inputs |
|---|---|---|---|
| `volume_up` | 33% (40) | `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos`, `opps_trend` | served: `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos` ; not built: `opps_trend` |
| `volume_down` | 7% (8) | `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos`, `opps_trend` | served: `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos` ; not built: `opps_trend` |
| `teammate_out` | 5% (6) | `tm_out_above_n`, `tm_out_group_opps`, `tm_out_targets_all`, `tm_out_carries_all`, `tm_group_size` | served: `tm_out_above_n`, `tm_out_group_opps`, `tm_out_targets_all`, `tm_out_carries_all`, `tm_group_size` |
| `blowout_script` | 2% (2) | `team_spread`, `implied_team_total`, `total_line` | served: `team_spread`, `implied_team_total`, `total_line` |
| `left_early` | 4% (5) | `snap_pct_l1`, `inj_weeks_since_report`, `inj_report_status`, `inj_practice_status` | served: `snap_pct_l1`, `inj_weeks_since_report` ; pruned: `inj_report_status`, `inj_practice_status` |
| `td_variance` | 24% (29) | `td_luck_rush_std`, `td_luck_rec_std`, `td_luck_pass_std`, `td_luck_total_std`, `td_luck_total_prev` | served: `td_luck_rush_std`, `td_luck_rec_std`, `td_luck_pass_std`, `td_luck_total_std`, `td_luck_total_prev` |
| `inactive_missed` | 53 rows, counted apart (28 of 53 on the gameday inactive list) | status layer (`ff.status`, Sleeper `injury_status`, the nflverse report, the gameday inactive list), not a feature | n/a |

## Limits

- Rules, not judgment: thresholds are fixed (35% and 3+ opportunities (QB 6+) for a volume call, 25% or 3 opportunities for 'as expected', snap share < 60% of trailing, margin >= 17, TDs 1.5 from the trailing rate). Priority order: left_early, teammate_out, blowout_script, volume, td_variance. A row can satisfy two; the second is `secondary_reason`.
- Expected volume is the trailing mean of his earlier played weeks this season, not E_opps (not frozen per week) and not the model's own component predictions (not frozen either). A player with no earlier week this season can only be `td_variance`-free `unexplained`, `left_early` (never: no trailing snaps) or `inactive_missed`.
- Teammate out is read from that week's depth chart against `data/player_weeks.csv` and `data/lineups_played.csv`; `data/status.csv` is a snapshot of the upcoming week with no week column and is not used. A teammate who is simply untracked (never a stats row this season) is not counted as out.
- Margins come from `data/defense_vs_pos.csv` (points allowed, `ALL` rows); a week without them has no `blowout_script`.
- Two weeks of misses are a list, not a distribution; read the reason shares after six.
