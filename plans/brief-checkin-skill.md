# Brief: the `/checkin` skill (new chat)

Written 2026-11-13 at the end of the status-cap chat (PR #27). Start a new chat with "read plans/brief-checkin-skill.md and build it".

## What James wants

One skill he runs every time he checks his lineups: Thursday night, again Saturday or Sunday morning, any time. Same report shape every
time, fresh injuries every time, and it must be cheap to re-run. It is the Thursday routine, written down:

1. **Health check.** Did the pipeline run? When? Any WARN rows (`logs/runs.csv`)? Is the data recent (latest `served_at` in
   `data/model_pts.csv`, latest week in `data/player_weeks.csv`, `data/status.csv` age)? Injuries and inactives pulled? Say "healthy" or
   name the broken piece and the fix.
2. **Reflect on last week.** Points per league (`data/matchup_results.csv`, `data/lineup_eval.csv`), and the model vs Sleeper vs E_pts
   from the committed scoreboard (`model/reports/live_scoreboard.md`, `data/model_eval.csv`). Gate status in one line.
3. **Season lineup efficiency.** The model's suggested lineups vs the hindsight-best lineups, and E_pts's, and the call's. Already in the
   scoreboard summary (lineup efficiency, offense-only efficiency, lineup wins) from `data/lineups_optimal.csv` and
   `data/lineups_played.csv`; just read and present it.
4. **Was the model refit?** The boosted trees retrain on every serve (walk-forward through last week), so this is "did the serve land for
   this week, and when" (`served_at`, `frozen_before_kickoff`). Hyperparameters and tree counts are frozen by design; say so once.
5. **Per league (all four):** the E_pts optimal lineup, the model optimal lineup (`data/lineups_optimal.csv` has e_pts / model / sleeper),
   and the call James should make, with the injury and starter checks the fantasy skill already does. Call `skills/fantasy/SKILL.md`
   per league; do not re-implement it. Lineup freshness rules in that skill apply (Sleeper live lineups first).
6. **Miss research (the new piece).** For each big miss of the previous week in `data/model_misses.csv` (and the retrospective
   `model/reports/retrospective.md`), dig up WHY: news, beat reports, snap counts, game script, injury-in-game, coaching decisions.
   Then ask: is there a factor here the model could have seen before kickoff? If yes, write it to `model/candidates.md` as an `open`
   row (date, candidate, evidence) in the existing table format, worth testing with `python -m model.trial`. Do not test it in the
   check-in; just raise it. Once per week: skip when `candidates.md` (or a small marker file) already shows that week was researched.

## How James wants the research done

- **Haiku agents, a full swarm, goal-directed.** James said "ultracode, full swarm": he has opted in to multi-agent orchestration
  (the Workflow tool) for the miss-research step. One Haiku agent per big miss (or per miss x source), a verifier pass, then one merge.
  Give each agent a goal and a report shape, not a script.
- Sonnet still does the code changes (CLAUDE.md), one agent per piece, worktrees.
- Fable plans, merges, and answers in the chat voice CLAUDE.md describes: decision first, short bullets, offer depth.

## Shape

- New `skills/checkin/SKILL.md` (and whatever helper script makes the deterministic parts one command, e.g. `python -m ff.checkin`
  that prints the health + scoreboard + efficiency block from the committed files; read-only, never runs the pipeline).
- Steps 1-5 are reads of committed files plus the fantasy skill: fast, deterministic, re-runnable.
- Step 6 is the only slow step and the only one that writes (candidates.md rows). Gate it on "not yet done for that week".
- Report: health line, last week in 3 bullets, season efficiency in one small table, then one block per league (E_pts lineup, model
  lineup, the call, injury flags), then "misses researched: N, candidates raised: M" with the one-line reasons.

## Already true (do not rebuild)

- Pipeline: GitHub Actions, Tue / Wed / Fri / Sun x2; scoreboard and retrospective run every pipeline run. Never run it locally.
- Model: gradient-boosted trees (LightGBM + XGBoost + CatBoost average, plus 14 LightGBM component models composed per league).
  Refit each serve. Status-layer cap (PR #27) applies at serve time.
- The gate (model beats Sleeper on pick accuracy over 6+ weeks) is MET as of week 9 but by a hair; E_pts is still authoritative.

## Open questions for the new chat to settle with James

- Where should the weekly "misses researched" marker live (a column in `data/model_misses.csv`, or `model/reports/miss_research.md`)?
- Should the scheduled routines he already has call `/checkin` on Thursday automatically, or stay manual?
