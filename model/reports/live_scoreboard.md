# Live scoreboard

**Adoption gate: the model earns recipe preference only when it beats Sleeper's projection on pick accuracy over at least 6 completed 2026 weeks.**

**Gate status: NOT MET.** 2 of 6 completed weeks scored (2026 wk4, 2026 wk5). Pooled pick accuracy on identical rows: model (E_pts_model) 0.677, Sleeper 0.669 (difference 0.008). The gate cannot be met before 6 weeks are scored, whatever the difference. 

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
| `model_components` | 1299 | 30940 | 0.677 | 0.457 |
| `model_flat` | 1299 | 30940 | 0.670 | 0.436 |
| `model_blend` | 1299 | 30940 | 0.676 | 0.449 |
| `sleeper` | 1299 | 30940 | 0.669 | 0.467 |
| `e_pts` | 1299 | 30940 | 0.656 | 0.391 |

### By position

**QB**

| comparator | players | pairs | pick accuracy | Spearman |
|---|---|---|---|---|
| `model_components` | 199 | 2476 | 0.696 | 0.534 |
| `model_flat` | 199 | 2476 | 0.700 | 0.526 |
| `model_blend` | 199 | 2476 | 0.697 | 0.528 |
| `sleeper` | 199 | 2476 | 0.723 | 0.588 |
| `e_pts` | 199 | 2476 | 0.673 | 0.471 |

**RB**

| comparator | players | pairs | pick accuracy | Spearman |
|---|---|---|---|---|
| `model_components` | 397 | 9914 | 0.714 | 0.575 |
| `model_flat` | 397 | 9914 | 0.707 | 0.559 |
| `model_blend` | 397 | 9914 | 0.715 | 0.575 |
| `sleeper` | 397 | 9914 | 0.710 | 0.580 |
| `e_pts` | 397 | 9914 | 0.706 | 0.556 |

**WR**

| comparator | players | pairs | pick accuracy | Spearman |
|---|---|---|---|---|
| `model_components` | 503 | 16061 | 0.664 | 0.445 |
| `model_flat` | 503 | 16061 | 0.657 | 0.431 |
| `model_blend` | 503 | 16061 | 0.664 | 0.449 |
| `sleeper` | 503 | 16061 | 0.647 | 0.406 |
| `e_pts` | 503 | 16061 | 0.638 | 0.374 |

**TE**

| comparator | players | pairs | pick accuracy | Spearman |
|---|---|---|---|---|
| `model_components` | 200 | 2489 | 0.596 | 0.275 |
| `model_flat` | 200 | 2489 | 0.574 | 0.226 |
| `model_blend` | 200 | 2489 | 0.581 | 0.244 |
| `sleeper` | 200 | 2489 | 0.596 | 0.295 |
| `e_pts` | 200 | 2489 | 0.561 | 0.163 |

### By week (pick accuracy, pooled over leagues)

| season / week | model_components | model_flat | model_blend | sleeper | e_pts |
|---|---|---|---|---|---|
| 2026 wk04 | 0.679 | 0.670 | 0.678 | 0.644 | 0.647 |
| 2026 wk05 | 0.675 | 0.670 | 0.674 | 0.700 | 0.668 |

### By league (pick accuracy, pooled over weeks)

| league | model_components | model_flat | model_blend | sleeper | e_pts |
|---|---|---|---|---|---|
| gooma-s-family-league | 0.676 | 0.667 | 0.674 | 0.667 | 0.653 |
| james-gregg-espn | 0.636 | 0.624 | 0.633 | 0.625 | 0.608 |
| we-can-think-of-something-funny | 0.699 | 0.692 | 0.699 | 0.691 | 0.679 |
| where-you-at | 0.672 | 0.667 | 0.672 | 0.667 | 0.655 |

## Coverage

Played rostered skill players per week (summed over leagues), how many had a model row, and how many made the identical-rows set (a player missing the model, Sleeper or E_pts is dropped from every comparator):

| season | week | played | with a model row | scored |
|---|---|---|---|---|
| 2026 | 4 | 675 | 675 | 675 |
| 2026 | 5 | 631 | 624 | 624 |

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
| 2026 | 6 | gooma-s-family-league | 448 | 0 | 448 | 448 | 448 | 445 | 445 |
| 2026 | 6 | james-gregg-espn | 448 | 0 | 448 | 448 | 448 | 445 | 445 |
| 2026 | 6 | we-can-think-of-something-funny | 448 | 0 | 448 | 448 | 448 | 445 | 445 |
| 2026 | 6 | where-you-at | 448 | 0 | 448 | 448 | 448 | 445 | 445 |

## Limits

- Frozen means frozen at the last run before each game's kickoff; weeks served before this file existed have no rows and are not scored.
- Few weeks are noise. Pick accuracy moves by several points from week to week; read the interval, not the ranking.
- Snap-only appearances (a player who played and scored zero without a stats row) and non-appearances are not scored.
- The IDP league is scored on skill positions only, like everything else in the repo; bonuses the composition cannot express (`python -m model.serve --columns`) are in the platform's actual points but not in the model's.
