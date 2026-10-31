# Live scoreboard

## Summary

- Gate: NOT MET, 4 of 6 completed weeks scored (model minus Sleeper +0.006 pick accuracy).
- Pick accuracy, pooled on identical rows: model 0.669, sleeper 0.662, e_pts 0.655.
- Lineup efficiency (points / hindsight-best lineup, pooled over 12 league-weeks): e_pts 74.5%, model 80.0%, sleeper 76.8%, call 77.5%, actual 92.7%.
- Offense-only efficiency (QB/RB/WR/TE, the like-for-like number): e_pts 78.7%, model 84.4%, sleeper 81.1%, call 81.8%, actual 85.7%.
- Lineup wins (highest total of e_pts / model / sleeper / call): e_pts 5, model 8, sleeper 6, call 6.

**Adoption gate: the model earns recipe preference only when it beats Sleeper's projection on pick accuracy over at least 6 completed 2026 weeks.**

**Gate status: NOT MET.** 4 of 6 completed weeks scored (2026 wk4, 2026 wk5, 2026 wk6, 2026 wk7). Pooled pick accuracy on identical rows: model (E_pts_model) 0.669, Sleeper 0.662 (difference 0.006). The gate cannot be met before 6 weeks are scored, whatever the difference. 

Week-blocked 95% interval for that difference: [-0.017, +0.028] (informational; the gate is the point estimate over 6+ weeks, as written).

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
| `model_components` | 2478 | 56529 | 0.669 | 0.429 |
| `model_flat` | 2478 | 56529 | 0.663 | 0.423 |
| `model_blend` | 2478 | 56529 | 0.669 | 0.431 |
| `sleeper` | 2478 | 56529 | 0.662 | 0.430 |
| `e_pts` | 2478 | 56529 | 0.655 | 0.388 |

### By position

**QB**

| comparator | players | pairs | pick accuracy | Spearman |
|---|---|---|---|---|
| `model_components` | 385 | 4599 | 0.661 | 0.446 |
| `model_flat` | 385 | 4599 | 0.676 | 0.483 |
| `model_blend` | 385 | 4599 | 0.674 | 0.480 |
| `sleeper` | 385 | 4599 | 0.690 | 0.512 |
| `e_pts` | 385 | 4599 | 0.643 | 0.404 |

**RB**

| comparator | players | pairs | pick accuracy | Spearman |
|---|---|---|---|---|
| `model_components` | 726 | 16633 | 0.696 | 0.513 |
| `model_flat` | 726 | 16633 | 0.689 | 0.500 |
| `model_blend` | 726 | 16633 | 0.698 | 0.514 |
| `sleeper` | 726 | 16633 | 0.694 | 0.513 |
| `e_pts` | 726 | 16633 | 0.692 | 0.505 |

**WR**

| comparator | players | pairs | pick accuracy | Spearman |
|---|---|---|---|---|
| `model_components` | 983 | 30704 | 0.665 | 0.452 |
| `model_flat` | 983 | 30704 | 0.659 | 0.440 |
| `model_blend` | 983 | 30704 | 0.664 | 0.453 |
| `sleeper` | 983 | 30704 | 0.652 | 0.423 |
| `e_pts` | 983 | 30704 | 0.649 | 0.420 |

**TE**

| comparator | players | pairs | pick accuracy | Spearman |
|---|---|---|---|---|
| `model_components` | 384 | 4593 | 0.602 | 0.306 |
| `model_flat` | 384 | 4593 | 0.585 | 0.270 |
| `model_blend` | 384 | 4593 | 0.589 | 0.276 |
| `sleeper` | 384 | 4593 | 0.587 | 0.271 |
| `e_pts` | 384 | 4593 | 0.576 | 0.225 |

### By week (pick accuracy, pooled over leagues)

| season / week | model_components | model_flat | model_blend | sleeper | e_pts |
|---|---|---|---|---|---|
| 2026 wk04 | 0.679 | 0.670 | 0.678 | 0.644 | 0.647 |
| 2026 wk05 | 0.675 | 0.670 | 0.674 | 0.700 | 0.668 |
| 2026 wk06 | 0.645 | 0.638 | 0.643 | 0.635 | 0.636 |
| 2026 wk07 | 0.672 | 0.673 | 0.676 | 0.673 | 0.673 |

### By league (pick accuracy, pooled over weeks)

| league | model_components | model_flat | model_blend | sleeper | e_pts |
|---|---|---|---|---|---|
| gooma-s-family-league | 0.667 | 0.661 | 0.666 | 0.660 | 0.652 |
| james-gregg-espn | 0.626 | 0.617 | 0.625 | 0.617 | 0.610 |
| we-can-think-of-something-funny | 0.689 | 0.685 | 0.690 | 0.684 | 0.676 |
| where-you-at | 0.666 | 0.662 | 0.666 | 0.660 | 0.655 |

## Coverage

Played rostered skill players per week (summed over leagues), how many had a model row, and how many made the identical-rows set (a player missing the model, Sleeper or E_pts is dropped from every comparator):

| season | week | played | with a model row | scored |
|---|---|---|---|---|
| 2026 | 4 | 675 | 675 | 675 |
| 2026 | 5 | 631 | 624 | 624 |
| 2026 | 6 | 609 | 596 | 596 |
| 2026 | 7 | 609 | 583 | 583 |

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
| 2026 | 8 | gooma-s-family-league | 442 | 0 | 442 | 442 | 442 | 440 | 438 |
| 2026 | 8 | james-gregg-espn | 442 | 0 | 442 | 442 | 442 | 440 | 438 |
| 2026 | 8 | we-can-think-of-something-funny | 442 | 0 | 442 | 442 | 442 | 440 | 438 |
| 2026 | 8 | where-you-at | 442 | 0 | 442 | 442 | 442 | 440 | 438 |

## Lineups: which number should pick my lineup

Each week the pipeline records four lineups per league: optimal by E_pts, by the model, by Sleeper's projection, and the call (frozen slot by slot at kickoff), then scores them with the platform's real points next to the lineup actually started and the `perfect` lineup (the best his roster could have started, in hindsight). Efficiency = points / perfect points. A win is the highest total of e_pts / model / sleeper / call in a league-week (a tie counts for each). `actual` and `perfect` also count K / DEF / IDP slots the recorded lineups may leave empty, so compare lineups on the offense columns (QB/RB/WR/TE). A recorded player with no played row scores 0.

### All leagues pooled

| lineup | total points | offense points | efficiency | offense efficiency | league-weeks | wins | points vs actual |
|---|---|---|---|---|---|---|---|
| e_pts | 1578.1 | 1578.1 | 74.5% | 78.7% | 12 | 5 | -384.3 |
| model | 1693.2 | 1693.2 | 80.0% | 84.4% | 12 | 8 | -269.1 |
| sleeper | 1626.9 | 1626.9 | 76.8% | 81.1% | 12 | 6 | -335.4 |
| call | 1641.9 | 1641.9 | 77.5% | 81.8% | 12 | 6 | -320.5 |
| actual | 1962.4 | 1719.7 | 92.7% | 85.7% | 12 | - | - |
| perfect | 2117.6 | 2006.1 | 100.0% | 100.0% | 12 | - | +155.3 |

### gooma-s-family-league

| season / week | e_pts | model | sleeper | call | actual | perfect | winner |
|---|---|---|---|---|---|---|---|
| 2026 wk04 (b) | 121.8 (67.0%) | 159.9 (88.0%) | 151.1 (83.2%) | 121.8 (67.0%) | 120.8 (66.5%) | 181.7 (100.0%) | model |
| 2026 wk05 (b) | 159.4 (92.6%) | 159.4 (92.6%) | 164.6 (95.6%) | 159.4 (92.6%) | 172.4 (100.1%) | 172.2 (100.0%) | sleeper |
| 2026 wk07 (b) | 134.3 (76.3%) | 157.3 (89.4%) | 157.3 (89.4%) | 157.3 (89.4%) | 174.1 (98.9%) | 176.0 (100.0%) | model / sleeper / call |

Selected team efficiency, season to date (points / hindsight-best lineup): e_pts 78.4% (offense 78.4%), model 89.9% (offense 89.9%), sleeper 89.3% (offense 89.3%), call 82.8% (offense 82.8%), actual 88.2% (offense 84.4%).

### james-gregg-espn

| season / week | e_pts | model | sleeper | call | actual | perfect | winner |
|---|---|---|---|---|---|---|---|
| 2026 wk04 (b) | 137.6 (80.5%) | 152.2 (89.1%) | 152.2 (89.1%) | 152.2 (89.1%) | 164.2 (96.1%) | 170.9 (100.0%) | model / sleeper / call |
| 2026 wk05 (b) | 162.6 (99.1%) | 162.6 (99.1%) | 162.6 (99.1%) | 162.6 (99.1%) | 168.6 (102.7%) | 164.1 (100.0%) | e_pts / model / sleeper / call |
| 2026 wk07 (b) | 77.8 (66.4%) | 55.8 (47.6%) | 77.8 (66.4%) | 77.8 (66.4%) | 109.5 (93.4%) | 117.2 (100.0%) | e_pts / sleeper / call |

Selected team efficiency, season to date (points / hindsight-best lineup): e_pts 83.6% (offense 83.6%), model 82.0% (offense 82.0%), sleeper 86.8% (offense 86.8%), call 86.8% (offense 86.8%), actual 97.8% (offense 92.1%).

### we-can-think-of-something-funny

| season / week | e_pts | model | sleeper | call | actual | perfect | winner |
|---|---|---|---|---|---|---|---|
| 2026 wk04 (b) | 172.9 (76.0%) | 184.2 (80.9%) | 164.4 (72.2%) | 181.7 (79.8%) | 179.9 (79.0%) | 227.6 (100.0%) | model |
| 2026 wk05 (b) | 147.0 (81.0%) | 145.3 (80.1%) | 133.1 (73.3%) | 145.3 (80.1%) | 156.1 (86.0%) | 181.5 (100.0%) | e_pts |
| 2026 wk07 (b) | 168.7 (69.0%) | 219.1 (89.6%) | 202.3 (82.8%) | 186.3 (76.2%) | 237.4 (97.1%) | 244.4 (100.0%) | model |

Selected team efficiency, season to date (points / hindsight-best lineup): e_pts 74.8% (offense 74.8%), model 84.0% (offense 84.0%), sleeper 76.5% (offense 76.5%), call 78.5% (offense 78.5%), actual 87.7% (offense 83.4%).

### where-you-at

| season / week | e_pts | model | sleeper | call | actual | perfect | winner |
|---|---|---|---|---|---|---|---|
| 2026 wk04 (b) | 89.0 (56.2%) | 84.7 (53.5%) | 67.0 (42.3%) | 84.7 (53.5%) | 177.5 (112.1%) | 158.4 (100.0%) | e_pts |
| 2026 wk05 (b) | 120.7 (69.1%) | 126.5 (72.4%) | 108.3 (62.0%) | 126.5 (72.4%) | 150.2 (86.0%) | 174.7 (100.0%) | model / call |
| 2026 wk07 (b) | 86.2 (57.9%) | 86.2 (57.9%) | 86.2 (57.9%) | 86.2 (57.9%) | 151.6 (101.8%) | 148.9 (100.0%) | e_pts / model / sleeper / call |

Selected team efficiency, season to date (points / hindsight-best lineup): e_pts 61.4% (offense 79.9%), model 61.7% (offense 80.3%), sleeper 54.3% (offense 70.6%), call 61.7% (offense 80.3%), actual 99.5% (offense 84.0%).

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
