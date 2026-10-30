# Brief: two model-feature candidates (new chat)

Written 2026-10-30. Paste or point the new chat at this file.

## Context

- The model is 14 LightGBM component models (`model/features.py`, `model/serve.py`), composed under each league's scoring.
  New inputs ship only through a trial that beats the frozen spec out of sample (`model/trials.md`, `SHIPS` gate,
  `python -m model.serve --freeze` after adoption). The model decides the weight; nothing is forced.
- E_pts is a hand-built formula (`ff/score.py`): projection x matchup multiplier (`data/defense_vs_pos.csv`, one bucket per
  position) + TD regression. Changing it IS forcing. Do not touch E_pts until the model backtest says a signal is real.
- The live scoreboard (`python -m ff.ask scoreboard`) now tracks pick accuracy and lineup efficiency per basis, so any
  adopted feature can be judged on real weeks afterwards.

## Candidate 1: DvP by receiver archetype

Question it answers: "how does Seattle do against big-bodied downfield receivers?" Today the answer is one number for
all WRs (SEA = DvP 31, -10%).

One-off evidence (2026 weeks 1-7, WRs with 3+ targets vs SEA, PPR allowed): big body (6'2"+ or 210+) 8.9/game on 9
games vs 5.0 for smaller WRs; deep targets (NGS aDOT 12+) 14.8 on 3 games vs 8.0 short. League averages 10.6 / 11.2.
Odunze is 6'3" 215, aDOT 12.9: the group SEA is weakest against, yet the blanket haircut treats him like Waddle.

Build:
1. Classify each receiver-week with data the pipeline already pulls: height/weight (nflverse rosters), NGS receiving
   `avg_intended_air_yards`, `avg_separation`, `percent_share_of_intended_air_yards` (`point_in_time.NGS_STATS`),
   `air_yds_share` in `data/player_weeks.csv`. Buckets to start: size (big / not), depth (aDOT >= 12 / not), separation
   (above / below league median). Slot vs wide alignment is not public per game; aDOT + size is the proxy.
2. Per defense, allowed PPR per target and per game by bucket, with shrinkage toward the plain position number
   (samples are 3-9 games). Point-in-time: only games before the target week (`features.as_of_join`).
3. Add as a feature family (`def_vs_archetype_*`) and run the trial exactly like `market` / `ecr` were run. Report
   SHIPS / NO SHIP in `model/trials.md`.
4. Only if it ships: discuss (separately) whether the E_pts matchup multiplier should use it.

## Candidate 2: teammate returned / share shift

Question: Aaron Jones (MIN) week 7, Jordan Mason back from absence: Jones 28 opps -> 9, snaps 87% -> 25%. Nothing flags it.

What exists: `data/status.csv` `blocked_by` = "Jordan Mason (healthy)" (a label only), `depth_disagreement` 1,
`role_change` 0 (only fires for IR / practice-squad activations). E_opps is 100% Sleeper projection (`trail_cap` = 0 after
the backtest), so the drop is priced only as far as Sleeper prices it. Model teammate features (`tm_out_*`,
`role_rank_pos`) count teammates listed Out/Doubtful; a healthy teammate returning raises nothing. `role_opps` /
`peak_opps_3g` in roster.csv describe the old workload.

Build:
1. `ff/status.py`: a `return_from` flag: a same-team, same-position player who had 0 snaps in the prior 2+ weeks played
   >= 40% of snaps last game, or jumped the depth chart. Surface in status.csv and in the `why` text.
2. `ff/score.py` `_why`: a `share_shift` flag when last game's team carry/target share is 15+ points under his trailing
   average, or `trend_snap` < -0.3. Flag only, no E_pts change.
3. Model features: `tm_returned_l1` (binary), `share_delta_l1` (last game share minus trailing 3), `snap_delta_l1`. Trial
   as above. These should react within one game; check the backtest does not just re-learn Sleeper's projection.

## Ground rules

- Sonnet agents in worktrees for the code; Fable plans, merges, tests, opens and merges the PR.
- Tests: `python -m unittest discover -s tests -t .` and the touched `model/tests` modules. Model libs install with
  `pip install -r model/requirements.txt` (slow; `test_serve` takes ~20 min with them).
- The workflow writes data; never commit `data/`, `logs/`, `model/reports/` from a dev session.
- Three serve tests error in this sandbox on main too (pandas datetime units); not a regression signal.
