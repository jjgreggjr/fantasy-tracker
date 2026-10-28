# Live scoreboard

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
| 2026 | 8 | gooma-s-family-league | 453 | 0 | 453 | 453 | 453 | 451 | 451 |
| 2026 | 8 | james-gregg-espn | 453 | 0 | 453 | 453 | 453 | 451 | 451 |
| 2026 | 8 | we-can-think-of-something-funny | 453 | 0 | 453 | 453 | 453 | 451 | 451 |
| 2026 | 8 | where-you-at | 453 | 0 | 453 | 453 | 453 | 451 | 451 |

## Limits

- Frozen means frozen at the last run before each game's kickoff; weeks served before this file existed have no rows and are not scored.
- Few weeks are noise. Pick accuracy moves by several points from week to week; read the interval, not the ranking.
- Snap-only appearances (a player who played and scored zero without a stats row) and non-appearances are not scored.
- The IDP league is scored on skill positions only, like everything else in the repo; bonuses the composition cannot express (`python -m model.serve --columns`) are in the platform's actual points but not in the model's.
