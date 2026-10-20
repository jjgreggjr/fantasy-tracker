# Live scoreboard

**Adoption gate: the model earns recipe preference only when it beats Sleeper's projection on pick accuracy over at least 6 completed 2026 weeks.**

**Gate status: NOT MET.** 3 of 6 completed weeks scored (2026 wk4, 2026 wk5, 2026 wk6). Pooled pick accuracy on identical rows: model (E_pts_model) 0.667, Sleeper 0.659 (difference 0.008). The gate cannot be met before 6 weeks are scored, whatever the difference. 

Week-blocked 95% interval for that difference: [-0.025, +0.035] (informational; the gate is the point estimate over 6+ weeks, as written).

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
| `model_components` | 1895 | 44181 | 0.667 | 0.448 |
| `model_flat` | 1895 | 44181 | 0.660 | 0.429 |
| `model_blend` | 1895 | 44181 | 0.666 | 0.440 |
| `sleeper` | 1895 | 44181 | 0.659 | 0.438 |
| `e_pts` | 1895 | 44181 | 0.650 | 0.377 |

### By position

**QB**

| comparator | players | pairs | pick accuracy | Spearman |
|---|---|---|---|---|
| `model_components` | 291 | 3512 | 0.688 | 0.527 |
| `model_flat` | 291 | 3512 | 0.690 | 0.520 |
| `model_blend` | 291 | 3512 | 0.688 | 0.524 |
| `sleeper` | 291 | 3512 | 0.702 | 0.545 |
| `e_pts` | 291 | 3512 | 0.653 | 0.425 |

**RB**

| comparator | players | pairs | pick accuracy | Spearman |
|---|---|---|---|---|
| `model_components` | 562 | 13327 | 0.708 | 0.553 |
| `model_flat` | 562 | 13327 | 0.701 | 0.539 |
| `model_blend` | 562 | 13327 | 0.710 | 0.554 |
| `sleeper` | 562 | 13327 | 0.703 | 0.544 |
| `e_pts` | 562 | 13327 | 0.699 | 0.525 |

**WR**

| comparator | players | pairs | pick accuracy | Spearman |
|---|---|---|---|---|
| `model_components` | 751 | 23825 | 0.652 | 0.415 |
| `model_flat` | 751 | 23825 | 0.645 | 0.402 |
| `model_blend` | 751 | 23825 | 0.651 | 0.418 |
| `sleeper` | 751 | 23825 | 0.638 | 0.384 |
| `e_pts` | 751 | 23825 | 0.636 | 0.381 |

**TE**

| comparator | players | pairs | pick accuracy | Spearman |
|---|---|---|---|---|
| `model_components` | 291 | 3517 | 0.601 | 0.296 |
| `model_flat` | 291 | 3517 | 0.581 | 0.255 |
| `model_blend` | 291 | 3517 | 0.585 | 0.263 |
| `sleeper` | 291 | 3517 | 0.591 | 0.278 |
| `e_pts` | 291 | 3517 | 0.560 | 0.177 |

### By week (pick accuracy, pooled over leagues)

| season / week | model_components | model_flat | model_blend | sleeper | e_pts |
|---|---|---|---|---|---|
| 2026 wk04 | 0.679 | 0.670 | 0.678 | 0.644 | 0.647 |
| 2026 wk05 | 0.675 | 0.670 | 0.674 | 0.700 | 0.668 |
| 2026 wk06 | 0.645 | 0.638 | 0.643 | 0.635 | 0.636 |

### By league (pick accuracy, pooled over weeks)

| league | model_components | model_flat | model_blend | sleeper | e_pts |
|---|---|---|---|---|---|
| gooma-s-family-league | 0.664 | 0.656 | 0.663 | 0.656 | 0.645 |
| james-gregg-espn | 0.629 | 0.617 | 0.627 | 0.616 | 0.605 |
| we-can-think-of-something-funny | 0.688 | 0.681 | 0.687 | 0.681 | 0.672 |
| where-you-at | 0.665 | 0.660 | 0.664 | 0.657 | 0.651 |

## Coverage

Played rostered skill players per week (summed over leagues), how many had a model row, and how many made the identical-rows set (a player missing the model, Sleeper or E_pts is dropped from every comparator):

| season | week | played | with a model row | scored |
|---|---|---|---|---|
| 2026 | 4 | 675 | 675 | 675 |
| 2026 | 5 | 631 | 624 | 624 |
| 2026 | 6 | 609 | 596 | 596 |

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
| 2026 | 6 | gooma-s-family-league | 424 | 218 | 424 | 424 | 424 | 422 | 422 |
| 2026 | 6 | james-gregg-espn | 424 | 142 | 424 | 424 | 424 | 422 | 422 |
| 2026 | 6 | we-can-think-of-something-funny | 424 | 241 | 424 | 424 | 424 | 422 | 422 |
| 2026 | 6 | where-you-at | 424 | 255 | 424 | 424 | 424 | 422 | 422 |
| 2026 | 7 | gooma-s-family-league | 389 | 0 | 389 | 389 | 389 | 386 | 386 |
| 2026 | 7 | james-gregg-espn | 389 | 0 | 389 | 389 | 389 | 386 | 386 |
| 2026 | 7 | we-can-think-of-something-funny | 389 | 0 | 389 | 389 | 389 | 386 | 386 |
| 2026 | 7 | where-you-at | 389 | 0 | 389 | 389 | 389 | 386 | 386 |

## Limits

- Frozen means frozen at the last run before each game's kickoff; weeks served before this file existed have no rows and are not scored.
- Few weeks are noise. Pick accuracy moves by several points from week to week; read the interval, not the ranking.
- Snap-only appearances (a player who played and scored zero without a stats row) and non-appearances are not scored.
- The IDP league is scored on skill positions only, like everything else in the repo; bonuses the composition cannot express (`python -m model.serve --columns`) are in the platform's actual points but not in the model's.
