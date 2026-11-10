# Live scoreboard

## Summary

- Gate: MET, 6 of 6 completed weeks scored (model minus Sleeper +0.004 pick accuracy).
- Pick accuracy, pooled on identical rows: model 0.670, sleeper 0.666, e_pts 0.660.
- Lineup efficiency (points / hindsight-best lineup, pooled over 20 league-weeks): e_pts 80.1%, model 82.8%, sleeper 81.5%, call 82.0%, actual 94.9%.
- Offense-only efficiency (QB/RB/WR/TE, the like-for-like number): e_pts 83.0%, model 86.0%, sleeper 84.6%, call 85.1%, actual 86.8%.
- Lineup wins (highest total of e_pts / model / sleeper / call): e_pts 10, model 12, sleeper 12, call 12.

**Adoption gate: the model earns recipe preference only when it beats Sleeper's projection on pick accuracy over at least 6 completed 2026 weeks.**

**Gate status: MET.** 6 of 6 completed weeks scored (2026 wk4, 2026 wk5, 2026 wk6, 2026 wk7, 2026 wk8, 2026 wk9). Pooled pick accuracy on identical rows: model (E_pts_model) 0.670, Sleeper 0.666 (difference 0.004). 

Week-blocked 95% interval for that difference: [-0.012, +0.020] (informational; the gate is the point estimate over 6+ weeks, as written).

## What is scored

Each completed week, each league, every rostered QB/RB/WR/TE who played, against the platform's own points in that league's scoring (`data/lineups_played.csv`). Five predictions, all frozen before kickoff in `data/model_pts.csv`, all in that league's scoring, scored on the **identical rows**:

| comparator | what it is |
|---|---|
| `model_components` | E_pts_model: the 14 component models composed under the league's scoring (the gate's model) |
| `model_flat` | the 3-library PPR average, moved to the league's scoring by the components' scoring difference |
| `model_blend` | 50/50 average of the two |
| `sleeper` | Sleeper's projection in the league's scoring, captured at serve time |
| `e_pts` | the pipeline's E_pts for the league, captured at serve time (a ranking of opportunity, not a point projection) |

Metrics: **pick accuracy** = over all same-position pairs in a league-week with different actual points, the share where the higher prediction scored more (ties count half); **Spearman** = rank correlation within position-week-league, averaged. Played = an nflverse stats row that week or nonzero points: the model predicts points if he plays, availability is a separate layer, and a player ruled Out has no model row.

## Results, all scored weeks pooled

| comparator | players | pairs | pick accuracy | Spearman |
|---|---|---|---|---|
| `model_components` | 3676 | 82296 | 0.670 | 0.435 |
| `model_flat` | 3676 | 82296 | 0.668 | 0.434 |
| `model_blend` | 3676 | 82296 | 0.671 | 0.439 |
| `sleeper` | 3676 | 82296 | 0.666 | 0.431 |
| `e_pts` | 3676 | 82296 | 0.660 | 0.394 |

### By position

**QB**

| comparator | players | pairs | pick accuracy | Spearman |
|---|---|---|---|---|
| `model_components` | 576 | 6854 | 0.645 | 0.407 |
| `model_flat` | 576 | 6854 | 0.658 | 0.437 |
| `model_blend` | 576 | 6854 | 0.657 | 0.436 |
| `sleeper` | 576 | 6854 | 0.665 | 0.445 |
| `e_pts` | 576 | 6854 | 0.622 | 0.336 |

**RB**

| comparator | players | pairs | pick accuracy | Spearman |
|---|---|---|---|---|
| `model_components` | 1074 | 24143 | 0.687 | 0.501 |
| `model_flat` | 1074 | 24143 | 0.684 | 0.497 |
| `model_blend` | 1074 | 24143 | 0.690 | 0.507 |
| `sleeper` | 1074 | 24143 | 0.683 | 0.494 |
| `e_pts` | 1074 | 24143 | 0.686 | 0.502 |

**WR**

| comparator | players | pairs | pick accuracy | Spearman |
|---|---|---|---|---|
| `model_components` | 1446 | 44297 | 0.673 | 0.466 |
| `model_flat` | 1446 | 44297 | 0.669 | 0.461 |
| `model_blend` | 1446 | 44297 | 0.672 | 0.470 |
| `sleeper` | 1446 | 44297 | 0.665 | 0.454 |
| `e_pts` | 1446 | 44297 | 0.661 | 0.448 |

**TE**

| comparator | players | pairs | pick accuracy | Spearman |
|---|---|---|---|---|
| `model_components` | 580 | 7002 | 0.625 | 0.365 |
| `model_flat` | 580 | 7002 | 0.615 | 0.343 |
| `model_blend` | 580 | 7002 | 0.615 | 0.342 |
| `sleeper` | 580 | 7002 | 0.610 | 0.329 |
| `e_pts` | 580 | 7002 | 0.600 | 0.289 |

### By week (pick accuracy, pooled over leagues)

| season / week | model_components | model_flat | model_blend | sleeper | e_pts |
|---|---|---|---|---|---|
| 2026 wk04 | 0.679 | 0.670 | 0.678 | 0.644 | 0.647 |
| 2026 wk05 | 0.675 | 0.670 | 0.674 | 0.700 | 0.668 |
| 2026 wk06 | 0.645 | 0.638 | 0.643 | 0.635 | 0.636 |
| 2026 wk07 | 0.672 | 0.673 | 0.676 | 0.673 | 0.673 |
| 2026 wk08 | 0.666 | 0.667 | 0.668 | 0.666 | 0.666 |
| 2026 wk09 | 0.683 | 0.689 | 0.687 | 0.682 | 0.674 |

### By league (pick accuracy, pooled over weeks)

| league | model_components | model_flat | model_blend | sleeper | e_pts |
|---|---|---|---|---|---|
| gooma-s-family-league | 0.670 | 0.666 | 0.670 | 0.665 | 0.657 |
| james-gregg-espn | 0.636 | 0.631 | 0.637 | 0.630 | 0.627 |
| we-can-think-of-something-funny | 0.687 | 0.686 | 0.689 | 0.683 | 0.677 |
| where-you-at | 0.668 | 0.667 | 0.669 | 0.664 | 0.659 |

## Coverage

Played rostered skill players per week (summed over leagues), how many had a model row, and how many made the identical-rows set (a player missing the model, Sleeper or E_pts is dropped from every comparator):

| season | week | played | with a model row | scored |
|---|---|---|---|---|
| 2026 | 4 | 675 | 675 | 675 |
| 2026 | 5 | 631 | 624 | 624 |
| 2026 | 6 | 609 | 596 | 596 |
| 2026 | 7 | 609 | 583 | 583 |
| 2026 | 8 | 610 | 590 | 590 |
| 2026 | 9 | 622 | 608 | 608 |

## Which comparators exist

Non-blank frozen values per league-week in `data/model_pts.csv` (the actual-points column is the recorded league rows):

| season | week | league | model rows | actual rows | model_components | model_flat | model_blend | sleeper | e_pts |
|---|---|---|---|---|---|---|---|---|---|
| 2026 | 4 | gooma-s-family-league | 495 | 211 | 495 | 495 | 495 | 492 | 492 |
| 2026 | 4 | james-gregg-espn | 495 | 143 | 495 | 495 | 495 | 492 | 492 |
| 2026 | 4 | we-can-think-of-something-funny | 495 | 242 | 495 | 495 | 495 | 492 | 492 |
| 2026 | 4 | where-you-at | 495 | 253 | 495 | 495 | 495 | 492 | 492 |
| 2026 | 5 | gooma-s-family-league | 457 | 210 | 457 | 457 | 457 | 454 | 454 |
| 2026 | 5 | james-gregg-espn | 457 | 142 | 457 | 457 | 457 | 454 | 454 |
| 2026 | 5 | we-can-think-of-something-funny | 457 | 242 | 457 | 457 | 457 | 454 | 454 |
| 2026 | 5 | where-you-at | 457 | 253 | 457 | 457 | 457 | 454 | 454 |
| 2026 | 6 | gooma-s-family-league | 424 | 219 | 424 | 424 | 424 | 422 | 422 |
| 2026 | 6 | james-gregg-espn | 424 | 142 | 424 | 424 | 424 | 422 | 422 |
| 2026 | 6 | we-can-think-of-something-funny | 424 | 241 | 424 | 424 | 424 | 422 | 422 |
| 2026 | 6 | where-you-at | 424 | 255 | 424 | 424 | 424 | 422 | 422 |
| 2026 | 7 | gooma-s-family-league | 435 | 218 | 435 | 435 | 435 | 432 | 432 |
| 2026 | 7 | james-gregg-espn | 435 | 143 | 435 | 435 | 435 | 432 | 432 |
| 2026 | 7 | we-can-think-of-something-funny | 435 | 240 | 435 | 435 | 435 | 432 | 432 |
| 2026 | 7 | where-you-at | 435 | 253 | 435 | 435 | 435 | 432 | 432 |
| 2026 | 8 | gooma-s-family-league | 416 | 218 | 416 | 416 | 416 | 414 | 414 |
| 2026 | 8 | james-gregg-espn | 416 | 141 | 416 | 416 | 416 | 414 | 414 |
| 2026 | 8 | we-can-think-of-something-funny | 416 | 242 | 416 | 416 | 416 | 414 | 414 |
| 2026 | 8 | where-you-at | 416 | 251 | 416 | 416 | 416 | 414 | 414 |
| 2026 | 9 | gooma-s-family-league | 433 | 218 | 433 | 433 | 433 | 429 | 427 |
| 2026 | 9 | james-gregg-espn | 433 | 139 | 433 | 433 | 433 | 429 | 427 |
| 2026 | 9 | we-can-think-of-something-funny | 433 | 241 | 433 | 433 | 433 | 429 | 427 |
| 2026 | 9 | where-you-at | 433 | 252 | 433 | 433 | 433 | 429 | 427 |
| 2026 | 10 | gooma-s-family-league | 381 | 0 | 381 | 381 | 381 | 378 | 378 |
| 2026 | 10 | james-gregg-espn | 381 | 0 | 381 | 381 | 381 | 378 | 378 |
| 2026 | 10 | we-can-think-of-something-funny | 381 | 0 | 381 | 381 | 381 | 378 | 378 |
| 2026 | 10 | where-you-at | 381 | 0 | 381 | 381 | 381 | 378 | 378 |

## Lineups: which number should pick my lineup

Each week the pipeline records four lineups per league: optimal by E_pts, by the model, by Sleeper's projection, and the call (frozen slot by slot at kickoff), then scores them with the platform's real points next to the lineup actually started and the `perfect` lineup (the best his roster could have started, in hindsight). Efficiency = points / perfect points. A win is the highest total of e_pts / model / sleeper / call in a league-week (a tie counts for each). `actual` and `perfect` also count K / DEF / IDP slots the recorded lineups may leave empty, so compare lineups on the offense columns (QB/RB/WR/TE). A recorded player with no played row scores 0.

### All leagues pooled

| lineup | total points | offense points | efficiency | offense efficiency | league-weeks | wins | points vs actual |
|---|---|---|---|---|---|---|---|
| e_pts | 2705.3 | 2626.3 | 80.1% | 83.0% | 20 | 10 | -498.1 |
| model | 2797.6 | 2718.6 | 82.8% | 86.0% | 20 | 12 | -405.7 |
| sleeper | 2753.6 | 2674.6 | 81.5% | 84.6% | 20 | 12 | -449.7 |
| call | 2769.7 | 2690.7 | 82.0% | 85.1% | 20 | 12 | -433.7 |
| actual | 3203.4 | 2744.2 | 94.9% | 86.8% | 20 | - | - |
| perfect | 3377.0 | 3162.5 | 100.0% | 100.0% | 20 | - | +173.6 |

### gooma-s-family-league

| season / week | e_pts | model | sleeper | call | actual | perfect | winner |
|---|---|---|---|---|---|---|---|
| 2026 wk04 (b) | 121.8 (67.0%) | 159.9 (88.0%) | 151.1 (83.2%) | 121.8 (67.0%) | 120.8 (66.5%) | 181.7 (100.0%) | model |
| 2026 wk05 (b) | 159.4 (92.6%) | 159.4 (92.6%) | 164.6 (95.6%) | 159.4 (92.6%) | 172.4 (100.1%) | 172.2 (100.0%) | sleeper |
| 2026 wk07 (b) | 134.3 (76.3%) | 157.3 (89.4%) | 157.3 (89.4%) | 157.3 (89.4%) | 174.1 (98.9%) | 176.0 (100.0%) | model / sleeper / call |
| 2026 wk08 | 155.5 (95.1%) | 155.5 (95.1%) | 155.5 (95.1%) | 155.5 (95.1%) | 165.5 (101.2%) | 163.5 (100.0%) | e_pts / model / sleeper / call |
| 2026 wk09 | 136.1 (94.8%) | 136.1 (94.8%) | 136.1 (94.8%) | 136.1 (94.8%) | 157.0 (109.3%) | 143.6 (100.0%) | e_pts / model / sleeper / call |

Selected team efficiency, season to date (points / hindsight-best lineup): e_pts 84.5% (offense 84.5%), model 91.8% (offense 91.8%), sleeper 91.3% (offense 91.3%), call 87.2% (offense 87.2%), actual 94.4% (offense 87.4%).

### james-gregg-espn

| season / week | e_pts | model | sleeper | call | actual | perfect | winner |
|---|---|---|---|---|---|---|---|
| 2026 wk04 (b) | 137.6 (80.5%) | 152.2 (89.1%) | 152.2 (89.1%) | 152.2 (89.1%) | 164.2 (96.1%) | 170.9 (100.0%) | model / sleeper / call |
| 2026 wk05 (b) | 162.6 (99.1%) | 162.6 (99.1%) | 162.6 (99.1%) | 162.6 (99.1%) | 168.6 (102.7%) | 164.1 (100.0%) | e_pts / model / sleeper / call |
| 2026 wk07 (b) | 77.8 (66.4%) | 55.8 (47.6%) | 77.8 (66.4%) | 77.8 (66.4%) | 109.5 (93.4%) | 117.2 (100.0%) | e_pts / sleeper / call |
| 2026 wk08 | 83.4 (80.9%) | 83.4 (80.9%) | 83.4 (80.9%) | 83.4 (80.9%) | 91.4 (88.6%) | 103.1 (100.0%) | e_pts / model / sleeper / call |
| 2026 wk09 | 112.2 (92.9%) | 105.7 (87.5%) | 114.3 (94.6%) | 112.2 (92.9%) | 122.7 (101.6%) | 120.7 (100.0%) | sleeper |

Selected team efficiency, season to date (points / hindsight-best lineup): e_pts 84.8% (offense 84.8%), model 82.8% (offense 82.8%), sleeper 87.3% (offense 87.3%), call 87.0% (offense 87.0%), actual 97.1% (offense 88.7%).

### we-can-think-of-something-funny

| season / week | e_pts | model | sleeper | call | actual | perfect | winner |
|---|---|---|---|---|---|---|---|
| 2026 wk04 (b) | 172.9 (76.0%) | 184.2 (80.9%) | 164.4 (72.2%) | 181.7 (79.8%) | 179.9 (79.0%) | 227.6 (100.0%) | model |
| 2026 wk05 (b) | 147.0 (81.0%) | 145.3 (80.1%) | 133.1 (73.3%) | 145.3 (80.1%) | 156.1 (86.0%) | 181.5 (100.0%) | e_pts |
| 2026 wk07 (b) | 168.7 (69.0%) | 219.1 (89.6%) | 202.3 (82.8%) | 186.3 (76.2%) | 237.4 (97.1%) | 244.4 (100.0%) | model |
| 2026 wk08 | 220.3 (94.2%) | 211.2 (90.3%) | 220.3 (94.2%) | 220.3 (94.2%) | 231.6 (99.0%) | 233.9 (100.0%) | e_pts / sleeper / call |
| 2026 wk09 | 142.7 (89.8%) | 141.0 (88.7%) | 141.0 (88.7%) | 143.3 (90.1%) | 157.3 (98.9%) | 159.0 (100.0%) | call |

Selected team efficiency, season to date (points / hindsight-best lineup): e_pts 81.4% (offense 81.4%), model 86.1% (offense 86.1%), sleeper 82.3% (offense 82.3%), call 83.8% (offense 83.8%), actual 92.0% (offense 85.8%).

### where-you-at

| season / week | e_pts | model | sleeper | call | actual | perfect | winner |
|---|---|---|---|---|---|---|---|
| 2026 wk04 (b) | 89.0 (56.2%) | 84.7 (53.5%) | 67.0 (42.3%) | 84.7 (53.5%) | 177.5 (112.1%) | 158.4 (100.0%) | e_pts |
| 2026 wk05 (b) | 120.7 (69.1%) | 126.5 (72.4%) | 108.3 (62.0%) | 126.5 (72.4%) | 150.2 (86.0%) | 174.7 (100.0%) | model / call |
| 2026 wk07 (b) | 86.2 (57.9%) | 86.2 (57.9%) | 86.2 (57.9%) | 86.2 (57.9%) | 151.6 (101.8%) | 148.9 (100.0%) | e_pts / model / sleeper / call |
| 2026 wk08 | 147.6 (89.7%) | 141.5 (86.0%) | 146.1 (88.8%) | 147.6 (89.7%) | 153.1 (93.0%) | 164.6 (100.0%) | e_pts / call |
| 2026 wk09 | 129.5 (75.7%) | 130.1 (76.1%) | 130.1 (76.1%) | 129.5 (75.7%) | 162.5 (95.0%) | 171.0 (100.0%) | model / sleeper |

Selected team efficiency, season to date (points / hindsight-best lineup): e_pts 70.1% (offense 81.9%), model 69.6% (offense 81.2%), sleeper 65.8% (offense 76.1%), call 70.3% (offense 82.2%), actual 97.2% (offense 85.4%).

(b) = backfilled week (12 league-weeks): built after the fact from the pre-game roster snapshot and the predictions frozen before kickoff in `data/model_pts.csv`. Backfilled `call` lineups had no status rule (no historical practice / play_prob), and K / DEF / IDP slots have no frozen projection, so they are empty there. A league-week where a serve at kickoff overwrote the frozen rows (under 80% of his projected offensive players still frozen) is not backfilled.

Frozen coverage: gooma-s-family-league 2026 week 6 skipped, only 25% of its projected offensive players have a pre-kickoff prediction (need 80%).

Frozen coverage: james-gregg-espn 2026 week 6 skipped, only 30% of its projected offensive players have a pre-kickoff prediction (need 80%).

Frozen coverage: we-can-think-of-something-funny 2026 week 6 skipped, only 48% of its projected offensive players have a pre-kickoff prediction (need 80%).

Frozen coverage: where-you-at 2026 week 6 skipped, only 68% of its projected offensive players have a pre-kickoff prediction (need 80%).

## Limits

- Frozen means frozen at the last run before each game's kickoff; weeks served before this file existed have no rows and are not scored.
- Few weeks are noise. Pick accuracy moves by several points from week to week; read the interval, not the ranking.
- Snap-only appearances (a player who played and scored zero without a stats row) and non-appearances are not scored.
- The IDP league is scored on skill positions only, like everything else in the repo; bonuses the composition cannot express (`python -m model.serve --columns`) are in the platform's actual points but not in the model's.
