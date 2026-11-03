# Retrospective

What each comparator got most wrong in every completed week, and why, judged on the scoreboard's identical rows (`python -m model.retrospective`, run after `model.scoreboard`). It explains misses; it is not the adoption gate.

Weeks post-mortemed: 2026 wk4, 2026 wk5, 2026 wk6, 2026 wk7, 2026 wk8. A big miss is |predicted - actual| >= 12 points in the league's scoring, one row per player-week per comparator (the `gooma-s-family-league` row where he is rostered there, else the first other league alphabetically). Comparators: `model_components` (E_pts_model), `sleeper`, `e_pts`.

## Drift: pick accuracy by week

From `data/model_eval.csv`, pooled over leagues (the scoreboard's number). Cumulative mean and the week-to-week standard deviation of the weekly values.

| comparator | 2026 wk04 | 2026 wk05 | 2026 wk06 | 2026 wk07 | 2026 wk08 | mean | week sd |
|---|---|---|---|---|---|---|---|
| `model_components` | 0.679 | 0.675 | 0.645 | 0.672 | 0.666 | 0.667 | 0.014 |
| `model_flat` | 0.670 | 0.670 | 0.638 | 0.673 | 0.667 | 0.664 | 0.014 |
| `model_blend` | 0.678 | 0.674 | 0.643 | 0.676 | 0.668 | 0.668 | 0.014 |
| `sleeper` | 0.644 | 0.700 | 0.635 | 0.673 | 0.666 | 0.664 | 0.026 |
| `e_pts` | 0.647 | 0.668 | 0.636 | 0.673 | 0.666 | 0.658 | 0.016 |

No comparator is trending: every span move is inside twice its week-to-week standard deviation.

## Why the big misses happened

Share of each comparator's big misses by reason, all weeks pooled. `inactive_missed` (predicted > 5, did not play) is outside the scored rows and is counted apart: it is the status layer's miss, not the model's. Learnable = the pre-kickoff record could in principle have carried the signal (`volume_up`, `volume_down`, `teammate_out`, `blowout_script`); noise = `td_variance`, `left_early`, `unexplained`.

| comparator | big misses | `left_early` | `teammate_out` | `blowout_script` | `volume_up` | `volume_down` | `td_variance` | `unexplained` | learnable | noise | `inactive_missed` (apart) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `model_components` | 83 | 3 (4%) | 5 (6%) | 2 (2%) | 28 (34%) | 6 (7%) | 19 (23%) | 20 (24%) | 49% | 51% | 35 |
| `sleeper` | 79 | 5 (6%) | 5 (6%) | 3 (4%) | 25 (32%) | 8 (10%) | 17 (22%) | 16 (20%) | 52% | 48% | 22 |
| `e_pts` | 89 | 3 (3%) | 5 (6%) | 3 (3%) | 27 (30%) | 12 (13%) | 18 (20%) | 21 (24%) | 53% | 47% | 21 |

Per week, the 95th percentile of |error| on the scored rows (the `top 5%` rule would have cut here instead of at 12):

| season / week | `model_components` | `sleeper` | `e_pts` |
|---|---|---|---|
| 2026 wk04 | 14.7 | 14.9 | 15.8 |
| 2026 wk05 | 15.4 | 15.2 | 15.7 |
| 2026 wk06 | 12.7 | 14.2 | 13.6 |
| 2026 wk07 | 13.3 | 12.6 | 13.1 |
| 2026 wk08 | 15.5 | 14.7 | 15.2 |

## Band coverage: actuals against the frozen p10 / p90

Share of played rows (PPR, `data/player_weeks.csv` against `data/model_pts.csv`) inside the frozen band, above p90 and below p10, by position and pooled, all post-mortemed weeks. The band is recalibrated with `serving_config.json` `recalibration.mode`; a tail over 10% is where `underage_by_position` would stretch it (model/ensemble.py).

| position | n | inside [p10, p90] | above p90 | below p10 |
|---|---|---|---|---|
| QB | 164 | 84.8% | 8.5% | 6.7% |
| RB | 365 | 78.4% | 13.4% | 8.2% |
| WR | 621 | 77.9% | 11.8% | 10.3% |
| TE | 370 | 78.6% | 12.7% | 8.6% |
| ALL | 1520 | 78.9% | 12.0% | 9.0% |

Targets: 80% inside, 10% above, 10% below. A tail over 10% is short (stretch it), one under 10% is slack (shrink only on a say-so).

## Top 15 misses, 2026 week 8

Each comparator's largest played misses that week with the evidence behind the reason (expected = trailing mean over his earlier played weeks this season; volume = pass attempts + carries for a QB, carries + targets otherwise; margin = his team's final margin).

**`model_components`** (15 shown)

| player | pos | team | league | predicted | actual | error | reason | secondary | vol exp/act | TD exp/act | snap% trail/act | margin | teammate out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tyler Allgeier | RB | ARI | gooma-s-family-league | 8.5 | 36.1 | -27.6 | `unexplained` |  | 12.0/16 | 0.0/2 | 0.49/0.59 | 3 |  |
| Emmett Johnson | RB | KC | we-can-think-of-something-funny | 5.2 | 30.3 | -25.1 | `unexplained` |  | 3.3/6 | 0.0/1 | 0.28/0.19 | -4 |  |
| Tee Higgins | WR | CIN | gooma-s-family-league | 14.4 | 37.5 | -23.1 | `td_variance` |  | 9.0/11 | 0.3/2 | 0.78/0.88 | 7 |  |
| Adonai Mitchell | WR | NYJ | we-can-think-of-something-funny | 6.8 | 29.2 | -22.4 | `teammate_out` | `volume_up` | 6.0/9 | 0.0/1 | 0.65/0.61 | -5 | Omar Cooper Jr. |
| Zay Flowers | WR | BAL | gooma-s-family-league | 19.4 | 0.0 | 19.4 | `blowout_script` | `volume_down` | 11.0/5 | 0.3/0 | 0.85/0.82 | -26 |  |
| Joe Burrow | QB | CIN | gooma-s-family-league | 20.6 | 39.5 | -18.8 | `td_variance` |  | 41.7/51 | 3.3/6 | 0.98/1.00 | 7 |  |
| Lamar Jackson | QB | BAL | gooma-s-family-league | 18.4 | 0.0 | 18.4 | `left_early` | `blowout_script` | 27.7/1 | 1.7/0 | 0.84/0.02 | -26 |  |
| Michael Penix Jr. | QB | ATL | gooma-s-family-league | 14.6 | -1.9 | 16.5 | `td_variance` |  | 35.7/29 | 2.3/0 | 0.98/1.00 | -16 |  |
| Drake London | WR | ATL | gooma-s-family-league | 18.2 | 1.8 | 16.4 | `volume_down` |  | 10.3/6 | 0.3/0 | 0.93/0.96 | -16 |  |
| Bhayshul Tuten | RB | JAX | gooma-s-family-league | 14.2 | 29.9 | -15.7 | `td_variance` |  | 20.0/18 | 0.7/2 | 0.62/0.50 | 35 |  |
| D'Andre Swift | RB | CHI | gooma-s-family-league | 13.9 | 27.9 | -14.0 | `volume_up` | `td_variance` | 20.3/30 | 1.3/3 | 0.67/0.77 | 17 |  |
| Jacoby Brissett | QB | ARI | gooma-s-family-league | 17.4 | 3.5 | 14.0 | `left_early` | `volume_down` | 47.0/26 | 1.3/0 | 1.00/0.58 | 3 |  |
| Cam Ward | QB | TEN | gooma-s-family-league | 12.3 | 26.3 | -13.9 | `volume_up` | `td_variance` | 31.7/47 | 0.3/2 | 1.00/1.00 | -7 |  |
| Tony Pollard | RB | TEN | gooma-s-family-league | 15.1 | 28.7 | -13.6 | `unexplained` |  | 21.0/26 | 1.0/2 | 0.66/0.82 | -7 |  |
| Baker Mayfield | QB | TB | gooma-s-family-league | 14.6 | 1.4 | 13.2 | `volume_down` |  | 28.7/7 | 1.0/0 | 0.95/1.00 | 16 |  |

**`sleeper`** (15 shown)

| player | pos | team | league | predicted | actual | error | reason | secondary | vol exp/act | TD exp/act | snap% trail/act | margin | teammate out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tyler Allgeier | RB | ARI | gooma-s-family-league | 8.4 | 36.1 | -27.7 | `unexplained` |  | 12.0/16 | 0.0/2 | 0.49/0.59 | 3 |  |
| Emmett Johnson | RB | KC | we-can-think-of-something-funny | 4.3 | 30.3 | -26.0 | `unexplained` |  | 3.3/6 | 0.0/1 | 0.28/0.19 | -4 |  |
| Adonai Mitchell | WR | NYJ | we-can-think-of-something-funny | 5.4 | 29.2 | -23.8 | `teammate_out` | `volume_up` | 6.0/9 | 0.0/1 | 0.65/0.61 | -5 | Omar Cooper Jr. |
| Tee Higgins | WR | CIN | gooma-s-family-league | 16.0 | 37.5 | -21.5 | `td_variance` |  | 9.0/11 | 0.3/2 | 0.78/0.88 | 7 |  |
| Lamar Jackson | QB | BAL | gooma-s-family-league | 20.1 | 0.0 | 20.1 | `left_early` | `blowout_script` | 27.7/1 | 1.7/0 | 0.84/0.02 | -26 |  |
| Zay Flowers | WR | BAL | gooma-s-family-league | 18.5 | 0.0 | 18.5 | `blowout_script` | `volume_down` | 11.0/5 | 0.3/0 | 0.85/0.82 | -26 |  |
| Joe Burrow | QB | CIN | gooma-s-family-league | 21.9 | 39.5 | -17.6 | `td_variance` |  | 41.7/51 | 3.3/6 | 0.98/1.00 | 7 |  |
| Michael Penix Jr. | QB | ATL | gooma-s-family-league | 14.0 | -1.9 | 15.9 | `td_variance` |  | 35.7/29 | 2.3/0 | 0.98/1.00 | -16 |  |
| Jacoby Brissett | QB | ARI | gooma-s-family-league | 18.3 | 3.5 | 14.8 | `left_early` | `volume_down` | 47.0/26 | 1.3/0 | 1.00/0.58 | 3 |  |
| Drake London | WR | ATL | gooma-s-family-league | 16.5 | 1.8 | 14.7 | `volume_down` |  | 10.3/6 | 0.3/0 | 0.93/0.96 | -16 |  |
| Bhayshul Tuten | RB | JAX | gooma-s-family-league | 15.4 | 29.9 | -14.5 | `td_variance` |  | 20.0/18 | 0.7/2 | 0.62/0.50 | 35 |  |
| Bijan Robinson | RB | ATL | gooma-s-family-league | 22.0 | 7.6 | 14.4 | `td_variance` |  | 20.0/20 | 0.0/0 | 0.80/0.81 | -16 |  |
| Tony Pollard | RB | TEN | gooma-s-family-league | 14.7 | 28.7 | -14.0 | `unexplained` |  | 21.0/26 | 1.0/2 | 0.66/0.82 | -7 |  |
| Cam Ward | QB | TEN | gooma-s-family-league | 12.5 | 26.3 | -13.7 | `volume_up` | `td_variance` | 31.7/47 | 0.3/2 | 1.00/1.00 | -7 |  |
| D'Andre Swift | RB | CHI | gooma-s-family-league | 14.7 | 27.9 | -13.2 | `volume_up` | `td_variance` | 20.3/30 | 1.3/3 | 0.67/0.77 | 17 |  |

**`e_pts`** (15 shown)

| player | pos | team | league | predicted | actual | error | reason | secondary | vol exp/act | TD exp/act | snap% trail/act | margin | teammate out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tyler Allgeier | RB | ARI | gooma-s-family-league | 9.5 | 36.1 | -26.6 | `unexplained` |  | 12.0/16 | 0.0/2 | 0.49/0.59 | 3 |  |
| Emmett Johnson | RB | KC | we-can-think-of-something-funny | 4.7 | 30.3 | -25.6 | `unexplained` |  | 3.3/6 | 0.0/1 | 0.28/0.19 | -4 |  |
| Adonai Mitchell | WR | NYJ | we-can-think-of-something-funny | 6.6 | 29.2 | -22.6 | `teammate_out` | `volume_up` | 6.0/9 | 0.0/1 | 0.65/0.61 | -5 | Omar Cooper Jr. |
| Lamar Jackson | QB | BAL | gooma-s-family-league | 20.5 | 0.0 | 20.5 | `left_early` | `blowout_script` | 27.7/1 | 1.7/0 | 0.84/0.02 | -26 |  |
| Zay Flowers | WR | BAL | gooma-s-family-league | 20.5 | 0.0 | 20.5 | `blowout_script` | `volume_down` | 11.0/5 | 0.3/0 | 0.85/0.82 | -26 |  |
| Tee Higgins | WR | CIN | gooma-s-family-league | 17.7 | 37.5 | -19.8 | `td_variance` |  | 9.0/11 | 0.3/2 | 0.78/0.88 | 7 |  |
| Joe Burrow | QB | CIN | gooma-s-family-league | 21.2 | 39.5 | -18.3 | `td_variance` |  | 41.7/51 | 3.3/6 | 0.98/1.00 | 7 |  |
| Drake London | WR | ATL | gooma-s-family-league | 18.4 | 1.8 | 16.6 | `volume_down` |  | 10.3/6 | 0.3/0 | 0.93/0.96 | -16 |  |
| Tony Pollard | RB | TEN | gooma-s-family-league | 13.3 | 28.7 | -15.4 | `unexplained` |  | 21.0/26 | 1.0/2 | 0.66/0.82 | -7 |  |
| Bhayshul Tuten | RB | JAX | gooma-s-family-league | 14.6 | 29.9 | -15.3 | `td_variance` |  | 20.0/18 | 0.7/2 | 0.62/0.50 | 35 |  |
| Jacoby Brissett | QB | ARI | gooma-s-family-league | 18.2 | 3.5 | 14.7 | `left_early` | `volume_down` | 47.0/26 | 1.3/0 | 1.00/0.58 | 3 |  |
| Baker Mayfield | QB | TB | gooma-s-family-league | 15.9 | 1.4 | 14.5 | `volume_down` |  | 28.7/7 | 1.0/0 | 0.95/1.00 | 16 |  |
| Michael Penix Jr. | QB | ATL | gooma-s-family-league | 12.3 | -1.9 | 14.2 | `td_variance` |  | 35.7/29 | 2.3/0 | 0.98/1.00 | -16 |  |
| Javonte Williams | RB | DAL | gooma-s-family-league | 20.5 | 6.7 | 13.8 | `unexplained` |  | 17.7/26 | 1.0/0 | 0.74/0.74 | -3 |  |
| D'Andre Swift | RB | CHI | gooma-s-family-league | 14.2 | 27.9 | -13.7 | `volume_up` | `td_variance` | 20.3/30 | 1.3/3 | 0.67/0.77 | 17 |  |

**Did not play, predicted > 5 (`model_components`, 5 players; status layer, not scored)**

Gameday inactive list (nflverse weekly_rosters, posted ~90 min before kickoff; the pre-lock run and serve act on it): 1 of 5 on the gameday inactive list. The rest dressed and did not play, or were never listed.

| player | pos | team | league | predicted | on inactive list |
|---|---|---|---|---|---|
| Omar Cooper Jr. | WR | NYJ | gooma-s-family-league | 10.8 | no |
| Jake Ferguson | TE | DAL | gooma-s-family-league | 8.8 | no |
| Cade Otton | TE | TB | gooma-s-family-league | 8.8 | no |
| Emanuel Wilson | RB | SEA | gooma-s-family-league | 7.8 | no |
| Colby Parkinson | TE | LA | where-you-at | 5.5 | yes |

## Candidate signals (derived)

For each reason class, the share of the model's played big misses it explains, the features that would carry the signal, and whether each is served (among the 90 frozen `inputs` of `model/serving_config.json`), pruned (built by `model.features`, dropped by the Phase 2 SHAP prune) or not built. A served feature that still misses means the signal is in the inputs but the week's value was not (`tm_out_*` are read from the report as of serve time: a teammate ruled out after the last run is invisible). A reason whose features are all served is not a new-feature candidate; one whose share is high and whose features are pruned or not built is.

| reason | share of model big misses | features that carry the signal | status in `serving_config.json` inputs |
|---|---|---|---|
| `volume_up` | 34% (28) | `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos`, `opps_trend` | served: `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos` ; not built: `opps_trend` |
| `volume_down` | 7% (6) | `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos`, `opps_trend` | served: `opps_l1`, `opps_l2`, `opps_l3`, `target_share_l1`, `carry_share_l1`, `role_score`, `role_rank_pos` ; not built: `opps_trend` |
| `teammate_out` | 6% (5) | `tm_out_above_n`, `tm_out_group_opps`, `tm_out_targets_all`, `tm_out_carries_all`, `tm_group_size` | served: `tm_out_above_n`, `tm_out_group_opps`, `tm_out_targets_all`, `tm_out_carries_all`, `tm_group_size` |
| `blowout_script` | 2% (2) | `team_spread`, `implied_team_total`, `total_line` | served: `team_spread`, `implied_team_total`, `total_line` |
| `left_early` | 4% (3) | `snap_pct_l1`, `inj_weeks_since_report`, `inj_report_status`, `inj_practice_status` | served: `snap_pct_l1`, `inj_weeks_since_report` ; pruned: `inj_report_status`, `inj_practice_status` |
| `td_variance` | 23% (19) | `td_luck_rush_std`, `td_luck_rec_std`, `td_luck_pass_std`, `td_luck_total_std`, `td_luck_total_prev` | served: `td_luck_rush_std`, `td_luck_rec_std`, `td_luck_pass_std`, `td_luck_total_std`, `td_luck_total_prev` |
| `inactive_missed` | 35 rows, counted apart (19 of 35 on the gameday inactive list) | status layer (`ff.status`, Sleeper `injury_status`, the nflverse report, the gameday inactive list), not a feature | n/a |

## Limits

- Rules, not judgment: thresholds are fixed (35% and 3+ opportunities (QB 6+) for a volume call, 25% or 3 opportunities for 'as expected', snap share < 60% of trailing, margin >= 17, TDs 1.5 from the trailing rate). Priority order: left_early, teammate_out, blowout_script, volume, td_variance. A row can satisfy two; the second is `secondary_reason`.
- Expected volume is the trailing mean of his earlier played weeks this season, not E_opps (not frozen per week) and not the model's own component predictions (not frozen either). A player with no earlier week this season can only be `td_variance`-free `unexplained`, `left_early` (never: no trailing snaps) or `inactive_missed`.
- Teammate out is read from that week's depth chart against `data/player_weeks.csv` and `data/lineups_played.csv`; `data/status.csv` is a snapshot of the upcoming week with no week column and is not used. A teammate who is simply untracked (never a stats row this season) is not counted as out.
- Margins come from `data/defense_vs_pos.csv` (points allowed, `ALL` rows); a week without them has no `blowout_script`.
- Two weeks of misses are a list, not a distribution; read the reason shares after six.
