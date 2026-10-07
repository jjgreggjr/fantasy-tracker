# Live scoreboard

**Adoption gate: the model earns recipe preference only when it beats Sleeper's projection on pick accuracy over at least 6 completed 2026 weeks.**

**Gate status: NOT MET.** 1 of 6 completed weeks scored (2026 wk4). Pooled pick accuracy on identical rows: model (E_pts_model) 0.679, Sleeper 0.644 (difference 0.035). The gate cannot be met before 6 weeks are scored, whatever the difference. 

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
| `model_components` | 675 | 17004 | 0.679 | 0.450 |
| `model_flat` | 675 | 17004 | 0.670 | 0.412 |
| `model_blend` | 675 | 17004 | 0.678 | 0.434 |
| `sleeper` | 675 | 17004 | 0.644 | 0.365 |
| `e_pts` | 675 | 17004 | 0.647 | 0.333 |

### By position

**QB**

| comparator | players | pairs | pick accuracy | Spearman |
|---|---|---|---|---|
| `model_components` | 99 | 1220 | 0.682 | 0.538 |
| `model_flat` | 99 | 1220 | 0.685 | 0.503 |
| `model_blend` | 99 | 1220 | 0.686 | 0.514 |
| `sleeper` | 99 | 1220 | 0.670 | 0.477 |
| `e_pts` | 99 | 1220 | 0.647 | 0.406 |

**RB**

| comparator | players | pairs | pick accuracy | Spearman |
|---|---|---|---|---|
| `model_components` | 209 | 5513 | 0.726 | 0.599 |
| `model_flat` | 209 | 5513 | 0.723 | 0.588 |
| `model_blend` | 209 | 5513 | 0.731 | 0.603 |
| `sleeper` | 209 | 5513 | 0.701 | 0.530 |
| `e_pts` | 209 | 5513 | 0.710 | 0.548 |

**WR**

| comparator | players | pairs | pick accuracy | Spearman |
|---|---|---|---|---|
| `model_components` | 267 | 9038 | 0.662 | 0.437 |
| `model_flat` | 267 | 9038 | 0.650 | 0.397 |
| `model_blend` | 267 | 9038 | 0.660 | 0.429 |
| `sleeper` | 267 | 9038 | 0.618 | 0.309 |
| `e_pts` | 267 | 9038 | 0.624 | 0.331 |

**TE**

| comparator | players | pairs | pick accuracy | Spearman |
|---|---|---|---|---|
| `model_components` | 100 | 1233 | 0.586 | 0.228 |
| `model_flat` | 100 | 1233 | 0.563 | 0.159 |
| `model_blend` | 100 | 1233 | 0.575 | 0.191 |
| `sleeper` | 100 | 1233 | 0.556 | 0.145 |
| `e_pts` | 100 | 1233 | 0.533 | 0.046 |

### By week (pick accuracy, pooled over leagues)

| season / week | model_components | model_flat | model_blend | sleeper | e_pts |
|---|---|---|---|---|---|
| 2026 wk04 | 0.679 | 0.670 | 0.678 | 0.644 | 0.647 |

### By league (pick accuracy, pooled over weeks)

| league | model_components | model_flat | model_blend | sleeper | e_pts |
|---|---|---|---|---|---|
| gooma-s-family-league | 0.684 | 0.675 | 0.684 | 0.647 | 0.649 |
| james-gregg-espn | 0.621 | 0.604 | 0.618 | 0.577 | 0.585 |
| we-can-think-of-something-funny | 0.700 | 0.691 | 0.700 | 0.666 | 0.669 |
| where-you-at | 0.676 | 0.670 | 0.676 | 0.645 | 0.647 |

## Coverage

Played rostered skill players per week (summed over leagues), how many had a model row, and how many made the identical-rows set (a player missing the model, Sleeper or E_pts is dropped from every comparator):

| season | week | played | with a model row | scored |
|---|---|---|---|---|
| 2026 | 4 | 675 | 675 | 675 |

## Which comparators exist

Non-blank frozen values per league-week in `data/model_pts.csv` (the actual-points column is the recorded league rows):

| season | week | league | model rows | actual rows | model_components | model_flat | model_blend | sleeper | e_pts |
|---|---|---|---|---|---|---|---|---|---|
| 2026 | 4 | gooma-s-family-league | 495 | 211 | 495 | 495 | 495 | 492 | 492 |
| 2026 | 4 | james-gregg-espn | 495 | 143 | 495 | 495 | 495 | 492 | 492 |
| 2026 | 4 | we-can-think-of-something-funny | 495 | 242 | 495 | 495 | 495 | 492 | 492 |
| 2026 | 4 | where-you-at | 495 | 253 | 495 | 495 | 495 | 492 | 492 |
| 2026 | 5 | gooma-s-family-league | 405 | 0 | 405 | 405 | 405 | 402 | 402 |
| 2026 | 5 | james-gregg-espn | 405 | 0 | 405 | 405 | 405 | 402 | 402 |
| 2026 | 5 | we-can-think-of-something-funny | 405 | 0 | 405 | 405 | 405 | 402 | 402 |
| 2026 | 5 | where-you-at | 405 | 0 | 405 | 405 | 405 | 402 | 402 |

## Limits

- Frozen means frozen at the last run before each game's kickoff; weeks served before this file existed have no rows and are not scored.
- Few weeks are noise. Pick accuracy moves by several points from week to week; read the interval, not the ranking.
- Snap-only appearances (a player who played and scored zero without a stats row) and non-appearances are not scored.
- The IDP league is scored on skill positions only, like everything else in the repo; bonuses the composition cannot express (`python -m model.serve --columns`) are in the platform's actual points but not in the model's.
