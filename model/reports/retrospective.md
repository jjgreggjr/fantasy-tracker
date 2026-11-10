# Retrospective

What each comparator got most wrong in every completed week, and why, judged on the scoreboard's identical rows (`python -m model.retrospective`, run after `model.scoreboard`). It explains misses; it is not the adoption gate.

Weeks post-mortemed: 2026 wk4, 2026 wk5, 2026 wk6, 2026 wk7, 2026 wk8, 2026 wk9. A big miss is |predicted - actual| >= 12 points in the league's scoring, one row per player-week per comparator (the `gooma-s-family-league` row where he is rostered there, else the first other league alphabetically). Comparators: `model_components` (E_pts_model), `sleeper`, `e_pts`.

## Drift: pick accuracy by week

From `data/model_eval.csv`, pooled over leagues (the scoreboard's number). Cumulative mean and the week-to-week standard deviation of the weekly values.

| comparator | 2026 wk04 | 2026 wk05 | 2026 wk06 | 2026 wk07 | 2026 wk08 | 2026 wk09 | mean | week sd |
|---|---|---|---|---|---|---|---|---|
| `model_components` | 0.679 | 0.675 | 0.645 | 0.672 | 0.666 | 0.683 | 0.670 | 0.014 |
| `model_flat` | 0.670 | 0.670 | 0.638 | 0.673 | 0.667 | 0.689 | 0.668 | 0.016 |
| `model_blend` | 0.678 | 0.674 | 0.643 | 0.676 | 0.668 | 0.687 | 0.671 | 0.015 |
| `sleeper` | 0.644 | 0.700 | 0.635 | 0.673 | 0.666 | 0.682 | 0.667 | 0.024 |
| `e_pts` | 0.647 | 0.668 | 0.636 | 0.673 | 0.666 | 0.674 | 0.661 | 0.016 |

No comparator is trending: every span move is inside twice its week-to-week standard deviation.

## Why the big misses happened

Share of each comparator's big misses by reason, all weeks pooled. `inactive_missed` (predicted > 5, did not play) is outside the scored rows and is counted apart: it is the status layer's miss, not the model's. Learnable = the pre-kickoff record could in principle have carried the signal (`volume_up`, `volume_down`, `teammate_out`, `blowout_script`); noise = `td_variance`, `left_early`, `unexplained`.

| comparator | big misses | `left_early` | `teammate_out` | `blowout_script` | `volume_up` | `volume_down` | `td_variance` | `unexplained` | learnable | noise | `inactive_missed` (apart) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `model_components` | 101 | 4 (4%) | 5 (5%) | 2 (2%) | 32 (32%) | 7 (7%) | 24 (24%) | 27 (27%) | 46% | 54% | 39 |
| `sleeper` | 95 | 6 (6%) | 6 (6%) | 3 (3%) | 28 (29%) | 9 (9%) | 22 (23%) | 21 (22%) | 48% | 52% | 23 |
| `e_pts` | 106 | 3 (3%) | 6 (6%) | 3 (3%) | 31 (29%) | 13 (12%) | 23 (22%) | 27 (25%) | 50% | 50% | 22 |

Per week, the 95th percentile of |error| on the scored rows (the `top 5%` rule would have cut here instead of at 12):

| season / week | `model_components` | `sleeper` | `e_pts` |
|---|---|---|---|
| 2026 wk04 | 14.7 | 14.9 | 15.8 |
| 2026 wk05 | 15.4 | 15.2 | 15.7 |
| 2026 wk06 | 12.7 | 14.2 | 13.6 |
| 2026 wk07 | 13.3 | 12.6 | 13.1 |
| 2026 wk08 | 15.5 | 14.7 | 15.2 |
| 2026 wk09 | 14.3 | 14.0 | 14.7 |

## Band coverage: actuals against the frozen p10 / p90

Share of played rows (PPR, `data/player_weeks.csv` against `data/model_pts.csv`) inside the frozen band, above p90 and below p10, by position and pooled, all post-mortemed weeks. The band is recalibrated with `serving_config.json` `recalibration.mode`; a tail over 10% is where `underage_by_position` would stretch it (model/ensemble.py).

| position | n | inside [p10, p90] | above p90 | below p10 |
|---|---|---|---|---|
| QB | 201 | 83.6% | 9.0% | 7.5% |
| RB | 440 | 80.0% | 12.0% | 8.0% |
| WR | 750 | 78.7% | 12.0% | 9.3% |
| TE | 445 | 80.4% | 11.7% | 7.9% |
| ALL | 1836 | 80.0% | 11.6% | 8.4% |

Targets: 80% inside, 10% above, 10% below. A tail over 10% is short (stretch it), one under 10% is slack (shrink only on a say-so).

## Top 15 misses, 2026 week 9

Each comparator's largest played misses that week with the evidence behind the reason (expected = trailing mean over his earlier played weeks this season; volume = pass attempts + carries for a QB, carries + targets otherwise; margin = his team's final margin).

**`model_components`** (15 shown)

| player | pos | team | league | predicted | actual | error | reason | secondary | vol exp/act | TD exp/act | snap% trail/act | margin | teammate out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kayshon Boutte | WR | HOU | gooma-s-family-league | 4.1 | 28.5 | -24.4 | `td_variance` |  | 3.3/5 | 0.0/2 | 0.44/0.33 | 22 |  |
| Jordan Love | QB | GB | gooma-s-family-league | 16.5 | 36.2 | -19.7 | `unexplained` |  | 43.7/30 | 2.3/4 | 1.00/0.88 | 18 |  |
| C.J. Stroud | QB | HOU | gooma-s-family-league | 16.1 | 35.4 | -19.3 | `td_variance` |  | 35.0/37 | 2.3/5 | 1.00/0.89 | 22 |  |
| Matthew Golden | WR | GB | gooma-s-family-league | 14.1 | 32.4 | -18.3 | `unexplained` |  | 12.7/10 | 0.7/1 | 0.85/0.74 | 18 |  |
| Tank Dell | WR | HOU | gooma-s-family-league | 3.2 | 20.5 | -17.3 | `volume_up` |  | 2.0/9 | 0.0/1 | 0.23/0.47 | 22 |  |
| Christian Watson | WR | GB | gooma-s-family-league | 13.3 | 30.1 | -16.8 | `unexplained` |  | 8.7/8 | 0.0/1 | 0.73/0.77 | 18 |  |
| Darren Waller | TE | CAR | gooma-s-family-league | 6.2 | 23.0 | -16.8 | `volume_up` |  | 2.0/10 | 0.3/1 | 0.51/0.53 | -5 |  |
| Kenneth Walker III | RB | KC | gooma-s-family-league | 20.7 | 36.3 | -15.6 | `td_variance` |  | 22.3/22 | 0.7/3 | 0.76/0.65 | 23 |  |
| D'Andre Swift | RB | CHI | gooma-s-family-league | 18.0 | 3.1 | 14.9 | `volume_down` | `td_variance` | 25.7/13 | 2.0/0 | 0.74/0.72 | 3 |  |
| Puka Nacua | WR | LA | gooma-s-family-league | 20.5 | 35.0 | -14.5 | `volume_up` |  | 10.3/21 | 0.7/1 | 0.90/0.93 | 7 |  |
| Jared Goff | QB | DET | gooma-s-family-league | 19.5 | 5.3 | 14.2 | `td_variance` |  | 31.0/29 | 2.3/0 | 0.98/1.00 | 10 |  |
| CeeDee Lamb | WR | DAL | gooma-s-family-league | 19.5 | 33.5 | -14.0 | `unexplained` |  | 13.7/16 | 0.7/1 | 0.82/0.94 | 4 |  |
| Ja'Marr Chase | WR | CIN | gooma-s-family-league | 17.8 | 31.5 | -13.7 | `unexplained` |  | 11.0/12 | 1.0/1 | 0.91/0.84 | 17 |  |
| Josh Jacobs | RB | GB | gooma-s-family-league | 14.2 | 0.5 | 13.7 | `left_early` | `blowout_script` | 22.5/4 | 1.5/0 | 0.61/0.15 | 18 |  |
| Malik Willis | QB | MIA | gooma-s-family-league | 16.5 | 3.8 | 12.7 | `td_variance` |  | 32.3/27 | 2.3/0 | 1.00/1.00 | -10 |  |

**`sleeper`** (15 shown)

| player | pos | team | league | predicted | actual | error | reason | secondary | vol exp/act | TD exp/act | snap% trail/act | margin | teammate out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kayshon Boutte | WR | HOU | gooma-s-family-league | 3.8 | 28.5 | -24.7 | `td_variance` |  | 3.3/5 | 0.0/2 | 0.44/0.33 | 22 |  |
| Jordan Love | QB | GB | gooma-s-family-league | 16.6 | 36.2 | -19.6 | `unexplained` |  | 43.7/30 | 2.3/4 | 1.00/0.88 | 18 |  |
| C.J. Stroud | QB | HOU | gooma-s-family-league | 16.0 | 35.4 | -19.4 | `td_variance` |  | 35.0/37 | 2.3/5 | 1.00/0.89 | 22 |  |
| Tank Dell | WR | HOU | gooma-s-family-league | 2.8 | 20.5 | -17.7 | `volume_up` |  | 2.0/9 | 0.0/1 | 0.23/0.47 | 22 |  |
| Matthew Golden | WR | GB | gooma-s-family-league | 14.9 | 32.4 | -17.5 | `unexplained` |  | 12.7/10 | 0.7/1 | 0.85/0.74 | 18 |  |
| Darren Waller | TE | CAR | gooma-s-family-league | 6.6 | 23.0 | -16.4 | `volume_up` |  | 2.0/10 | 0.3/1 | 0.51/0.53 | -5 |  |
| D'Andre Swift | RB | CHI | gooma-s-family-league | 19.0 | 3.1 | 15.9 | `volume_down` | `td_variance` | 25.7/13 | 2.0/0 | 0.74/0.72 | 3 |  |
| Kenneth Walker III | RB | KC | gooma-s-family-league | 21.8 | 36.3 | -14.5 | `td_variance` |  | 22.3/22 | 0.7/3 | 0.76/0.65 | 23 |  |
| Jared Goff | QB | DET | gooma-s-family-league | 19.6 | 5.3 | 14.3 | `td_variance` |  | 31.0/29 | 2.3/0 | 0.98/1.00 | 10 |  |
| Christian Watson | WR | GB | gooma-s-family-league | 15.8 | 30.1 | -14.3 | `unexplained` |  | 8.7/8 | 0.0/1 | 0.73/0.77 | 18 |  |
| Malik Willis | QB | MIA | gooma-s-family-league | 17.5 | 3.8 | 13.7 | `td_variance` |  | 32.3/27 | 2.3/0 | 1.00/1.00 | -10 |  |
| Josh Jacobs | RB | GB | gooma-s-family-league | 14.0 | 0.5 | 13.5 | `left_early` | `blowout_script` | 22.5/4 | 1.5/0 | 0.61/0.15 | 18 |  |
| Kaleb Johnson | RB | GB | we-can-think-of-something-funny | 0.0 | 12.8 | -12.8 | `teammate_out` | `volume_up` | 5.3/14 | 0.0/1 | 0.16/0.39 | 18 | MarShawn Lloyd |
| Justice Hill | RB | BAL | where-you-at | 5.1 | 17.7 | -12.6 | `volume_up` |  | 7.0/12 | 0.0/1 | 0.46/0.52 | -23 |  |
| Jahmyr Gibbs | RB | DET | gooma-s-family-league | 25.7 | 13.4 | 12.3 | `unexplained` |  | 31.0/21 | 1.0/1 | 0.79/0.68 | 10 |  |

**`e_pts`** (15 shown)

| player | pos | team | league | predicted | actual | error | reason | secondary | vol exp/act | TD exp/act | snap% trail/act | margin | teammate out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kayshon Boutte | WR | HOU | gooma-s-family-league | 4.2 | 28.5 | -24.3 | `td_variance` |  | 3.3/5 | 0.0/2 | 0.44/0.33 | 22 |  |
| C.J. Stroud | QB | HOU | gooma-s-family-league | 14.9 | 35.4 | -20.5 | `td_variance` |  | 35.0/37 | 2.3/5 | 1.00/0.89 | 22 |  |
| Jordan Love | QB | GB | gooma-s-family-league | 16.4 | 36.2 | -19.8 | `unexplained` |  | 43.7/30 | 2.3/4 | 1.00/0.88 | 18 |  |
| Tank Dell | WR | HOU | gooma-s-family-league | 3.2 | 20.5 | -17.3 | `volume_up` |  | 2.0/9 | 0.0/1 | 0.23/0.47 | 22 |  |
| Darren Waller | TE | CAR | gooma-s-family-league | 6.4 | 23.0 | -16.6 | `volume_up` |  | 2.0/10 | 0.3/1 | 0.51/0.53 | -5 |  |
| Kenneth Walker III | RB | KC | gooma-s-family-league | 19.8 | 36.3 | -16.5 | `td_variance` |  | 22.3/22 | 0.7/3 | 0.76/0.65 | 23 |  |
| Matthew Golden | WR | GB | gooma-s-family-league | 16.2 | 32.4 | -16.2 | `unexplained` |  | 12.7/10 | 0.7/1 | 0.85/0.74 | 18 |  |
| Jared Goff | QB | DET | gooma-s-family-league | 20.9 | 5.3 | 15.6 | `td_variance` |  | 31.0/29 | 2.3/0 | 0.98/1.00 | 10 |  |
| Ja'Marr Chase | WR | CIN | gooma-s-family-league | 16.1 | 31.5 | -15.4 | `unexplained` |  | 11.0/12 | 1.0/1 | 0.91/0.84 | 17 |  |
| Malik Willis | QB | MIA | gooma-s-family-league | 18.9 | 3.8 | 15.1 | `td_variance` |  | 32.3/27 | 2.3/0 | 1.00/1.00 | -10 |  |
| CeeDee Lamb | WR | DAL | gooma-s-family-league | 19.2 | 33.5 | -14.3 | `unexplained` |  | 13.7/16 | 0.7/1 | 0.82/0.94 | 4 |  |
| Christian Watson | WR | GB | gooma-s-family-league | 16.3 | 30.1 | -13.8 | `unexplained` |  | 8.7/8 | 0.0/1 | 0.73/0.77 | 18 |  |
| Puka Nacua | WR | LA | gooma-s-family-league | 21.3 | 35.0 | -13.7 | `volume_up` |  | 10.3/21 | 0.7/1 | 0.90/0.93 | 7 |  |
| D'Andre Swift | RB | CHI | gooma-s-family-league | 16.3 | 3.1 | 13.2 | `volume_down` | `td_variance` | 25.7/13 | 2.0/0 | 0.74/0.72 | 3 |  |
| Kaleb Johnson | RB | GB | we-can-think-of-something-funny | 0.0 | 12.8 | -12.8 | `teammate_out` | `volume_up` | 5.3/14 | 0.0/1 | 0.16/0.39 | 18 | MarShawn Lloyd |

**Did not play, predicted > 5 (`model_components`, 4 players; status layer, not scored)**

Gameday inactive list (nflverse weekly_rosters, posted ~90 min before kickoff; the pre-lock run and serve act on it): 1 of 4 on the gameday inactive list. The rest dressed and did not play, or were never listed.

| player | pos | team | league | predicted | on inactive list |
|---|---|---|---|---|---|
| Ladd McConkey | WR | LAC | gooma-s-family-league | 12.0 | no |
| Brian Thomas Jr. | WR | JAX | gooma-s-family-league | 6.0 | no |
| Jacoby Brissett | QB | ARI | gooma-s-family-league | 5.2 | no |
| MarShawn Lloyd | RB | GB | gooma-s-family-league | 5.1 | yes |

## Candidate signals (derived)

For each reason class, the share of the model's played big misses it explains, the features that would carry the signal, and whether each is served (among the 90 frozen `inputs` of `model/serving_config.json`), pruned (built by `model.features`, dropped by the Phase 2 SHAP prune) or not built. A served feature that still misses means the signal is in the inputs but the week's value was not (`tm_out_*` are read from the report as of serve time: a teammate ruled out after the last run is invisible). A reason whose features are all served is not a new-feature candidate; one whose share is high and whose features are pruned or not built is.

| reason | share of model big misses | features that carry the signal | status in `serving_config.json` inputs |
|---|---|---|---|
| `volume_up` | 32% (32) | `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos`, `opps_trend` | served: `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos` ; not built: `opps_trend` |
| `volume_down` | 7% (7) | `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos`, `opps_trend` | served: `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos` ; not built: `opps_trend` |
| `teammate_out` | 5% (5) | `tm_out_above_n`, `tm_out_group_opps`, `tm_out_targets_all`, `tm_out_carries_all`, `tm_group_size` | served: `tm_out_above_n`, `tm_out_group_opps`, `tm_out_targets_all`, `tm_out_carries_all`, `tm_group_size` |
| `blowout_script` | 2% (2) | `team_spread`, `implied_team_total`, `total_line` | served: `team_spread`, `implied_team_total`, `total_line` |
| `left_early` | 4% (4) | `snap_pct_l1`, `inj_weeks_since_report`, `inj_report_status`, `inj_practice_status` | served: `snap_pct_l1`, `inj_weeks_since_report` ; pruned: `inj_report_status`, `inj_practice_status` |
| `td_variance` | 24% (24) | `td_luck_rush_std`, `td_luck_rec_std`, `td_luck_pass_std`, `td_luck_total_std`, `td_luck_total_prev` | served: `td_luck_rush_std`, `td_luck_rec_std`, `td_luck_pass_std`, `td_luck_total_std`, `td_luck_total_prev` |
| `inactive_missed` | 39 rows, counted apart (20 of 39 on the gameday inactive list) | status layer (`ff.status`, Sleeper `injury_status`, the nflverse report, the gameday inactive list), not a feature | n/a |

## Limits

- Rules, not judgment: thresholds are fixed (35% and 3+ opportunities (QB 6+) for a volume call, 25% or 3 opportunities for 'as expected', snap share < 60% of trailing, margin >= 17, TDs 1.5 from the trailing rate). Priority order: left_early, teammate_out, blowout_script, volume, td_variance. A row can satisfy two; the second is `secondary_reason`.
- Expected volume is the trailing mean of his earlier played weeks this season, not E_opps (not frozen per week) and not the model's own component predictions (not frozen either). A player with no earlier week this season can only be `td_variance`-free `unexplained`, `left_early` (never: no trailing snaps) or `inactive_missed`.
- Teammate out is read from that week's depth chart against `data/player_weeks.csv` and `data/lineups_played.csv`; `data/status.csv` is a snapshot of the upcoming week with no week column and is not used. A teammate who is simply untracked (never a stats row this season) is not counted as out.
- Margins come from `data/defense_vs_pos.csv` (points allowed, `ALL` rows); a week without them has no `blowout_script`.
- Two weeks of misses are a list, not a distribution; read the reason shares after six.
