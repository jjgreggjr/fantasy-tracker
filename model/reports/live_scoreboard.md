# Live scoreboard

**Adoption gate: the model earns recipe preference only when it beats Sleeper's projection on pick accuracy over at least 6 completed 2026 weeks.**

**Gate status: not started.** No completed week has frozen model predictions yet (0 of 6 weeks).

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

## Results

Nothing to score yet. The first frozen week is the one `data/model_pts.csv` holds; it is scored by the first run after its last game, when the pipeline records that week in `data/lineups_played.csv`.

## Which comparators exist

Non-blank frozen values per league-week in `data/model_pts.csv` (the actual-points column is the recorded league rows):

| season | week | league | model rows | actual rows | model_components | model_flat | model_blend | sleeper | e_pts |
|---|---|---|---|---|---|---|---|---|---|
| 2026 | 4 | gooma-s-family-league | 495 | 0 | 495 | 495 | 495 | 492 | 492 |
| 2026 | 4 | james-gregg-espn | 495 | 0 | 495 | 495 | 495 | 492 | 492 |
| 2026 | 4 | we-can-think-of-something-funny | 495 | 0 | 495 | 495 | 495 | 492 | 492 |
| 2026 | 4 | where-you-at | 495 | 0 | 495 | 495 | 495 | 492 | 492 |

## Limits

- Frozen means frozen at the last run before each game's kickoff; weeks served before this file existed have no rows and are not scored.
- Few weeks are noise. Pick accuracy moves by several points from week to week; read the interval, not the ranking.
- Snap-only appearances (a player who played and scored zero without a stats row) and non-appearances are not scored.
- The IDP league is scored on skill positions only, like everything else in the repo; bonuses the composition cannot express (`python -m model.serve --columns`) are in the platform's actual points but not in the model's.
