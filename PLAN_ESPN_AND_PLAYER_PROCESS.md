# Plan: track the ESPN league, and make every player answer follow one process

Two jobs, one plan, meant to be executed start to finish in a single session.
Part A (ESPN) is independent of Part B (per-player process). Do A first: it
needs a pipeline run to validate, and that run takes minutes you can spend on B.

Delete this file as the final step. It is a work order, not documentation;
the durable pieces land in `POLICY.md`, `README.md` and the skill.

Ground rules that still apply (from the migration):
- `secrets/*.json` stays gitignored. ESPN cookies enter the repo only as the
  GitHub Actions secrets `ESPN_S2` and `ESPN_SWID`, and only James adds them.
  Never ask for, print, log or commit their values.
- Question sessions read; the workflow writes. Code changes go through a
  commit on `main` and a pipeline run; nothing is "fixed" until a run proves it.
- No model identifiers in commits, comments or docs.
- Python 3.12, `pandas>=2.0,<3`.

---

## Part A — the ESPN league: DONE (2026-09-15)

Executed and proven by pipeline run 13 on `main`: `leagues/james-gregg-espn/`
exists (league "Average Joes", 10 teams, full PPR, QB/2RB/2WR/TE/FLEX/K/DEF,
7 bench, 1 IR), his team is identified by team id 14, `roster_fetched_at` is
set, `ff.ask live|lineup|cut|options` run on it, the routine and skill name
it, and a GUID scan of the league files is clean.

Facts the next session should know:
- ESPN's API is blocked from the question sandbox (egress policy 403). The
  workflow fetches it; `ff.ask live james-gregg-espn` returns the committed
  snapshot with its time and says "COMMITTED SNAPSHOT" in line one. Nothing
  to fix; it is the design.
- `config.json` → `espn_leagues[0]` = league 704757, `my_team_id` 14. The
  cookies live only in the `ESPN_S2` / `ESPN_SWID` secrets. `espn_s2`
  expires; the symptom is the `espn.league` WARN check returning.
- Owners in the league files are keyed by team id and labelled by ESPN
  display name; `team_names` carries the team names. No member ids anywhere.

Residue, small, for this session or the next:
- K and DEF entries on the ESPN roster have no row in `data/players.csv`, so
  they carry no ids and drop out of `data/my_roster.csv`; `ff.ask cut` then
  counts 14 rostered against 17 slots instead of 16. Keep them with a
  position from ESPN's `defaultPositionId` (`POS_ID`) even when unmatched,
  as the Sleeper path does, so the overage count is right.
- Format: ESPN's settings say not a keeper league, so `type` is redraft.
  Confirm with James; set `"type"` in the config entry only if he says
  otherwise.
- A6 (ESPN's own projections as a second source) remains optional.

---

## Part B — one per-player process, used every time

### B0. Why

The skill is a set of rules, not a sequence, and it never tells the reader
to use the two recipes that already answer most of the question:
`ff.ask explain` (the component card) and `ff.ask options` (the alternatives,
with `opportunity_ahead`). The order below is the consensus of the start/sit
methodology James asked for: volume first, points from that volume, recency,
matchup as a modifier, availability and who is hurt ahead of him, then floor
versus ceiling. `POLICY.md` already encodes the same premise; the reading was
the gap, not the model. No new weights, no new model.

### B1. The process (goes into `POLICY.md` verbatim, and the skill)

Fixed order. Stop early where it says stop.

```
0  FRAME      this league's scoring + slots · is he starter / bench / free agent
              right now (live read, or snapshot with its time) · which question:
              this week, or the roster spot
1  AVAILABLE  status_flag · play_prob · practice · injury_body_part · news age
              → Out / IR on a this-week question: stop here
2  ROLE       depth rank vs last week · depth_disagreement (ESPN vs Sleeper) ·
              blocked_by with THEIR statuses · opportunity_ahead
3  VOLUME     E_opps and its projection/recent split · snap% · share of team
              targets (WR/TE) or carries (RB) or pass attempts (QB) · trend ·
              the role he actually held (role_opps, role_snap)
4  POINTS     projected pts in league scoring · trailing 3 · E_pts ·
              median / ceiling · conf
5  MATCHUP    opponent · DvP rank (1 = softest) · multiplier · game total · home
              ← a nudge of at most ±12%, never the reason
6  VERSUS     the alternative(s): same card side by side, with the gap
              (free agents and other owners' benches, from `options`)
7  VERDICT    the decision, then WHICH SECTION decided it: volume, matchup, or
              a touchdown assumption. Roster-spot questions add vor · keep_pts ·
              age runway per POLICY.md
```

The answer is the verdict plus the one section that did the work. The card
is evidence; it is not recited.

### B2. Code: `ff/ask.py` — a `player` recipe that prints the card

`python -m ff.ask player <slug> "<name>"` prints sections 0 to 6 in that
layout, stitched from what exists:

- 0: `league.json` (name, type, starting slots) + where he is right now via
  `live_mod.live_league` (fall back to the snapshot per A4 and say so).
- 1 and 2: the `data/status.csv` row (`_status()` / `_status_block()` already
  format it) plus `depth_rank`, `prev_rank`, `depth_disagreement`,
  `blocked_by`, `opportunity_ahead`.
- 3 and 4: the `explain` rows, reordered to this sequence, with `snap%` and
  `share` from `live._with_volume`, `trend_opps`, `trend_snap`, `role_opps`,
  `role_snap`, `median_pts`, `ceiling_pts`.
- 5: `opponent`, `dvp_rank`, `mult`, `total_line`, `home`.
- 6: the top 3 rows from `options`, each with E_pts, E_opps, snap%, share
  and `vs_him`.
- Roster-spot block: `vor`, `keep_pts`, `keep_value`, `age`, `age_score`.

Search order for the name: his roster, then the league's other rosters, then
free agents; say which pool he came from. Print "STOP: Out/IR this week"
after section 1 when that is the case, then continue (the card is still
useful for the roster question). Keep every value that is already printed
by `explain`; this recipe supersedes reading three CSVs by hand, it does not
change any number. Add `player` to the `choices` list in `main()`.

Write one test that runs the recipe against each committed league for one
starter, one benched Out/IR player, and one free agent, and asserts the
eight headers appear in order.

### B3. Skill: `skills/fantasy/SKILL.md`

Add a section **"Per-player process"** immediately after "Lineup freshness":
the eight steps from B1, the rule "run `ff.ask player <slug> "<name>"` and
read it top to bottom; stop at 1 if he is out", and the verdict rule ("name
the section that decided it"). Reference `explain` and `options` as the
building blocks so a reader without the new recipe still follows the
order. Keep every existing rule verbatim, including the Charbonnet rule,
"Look at the bench too", and "Points and volume, together, every time".
Package and hand over the `.skill` as in A7 (one bundle for A7 and B3).

### B4. Routine

In the Tuesday routine prompt, make the Health, Sit and Waiver lines cite
the process by section: Health = steps 1 and 2 (starters and the injured
bench), Sit = steps 3 to 6 for the swap, Waiver = step 6 with vor. Same
delete-and-recreate as A7; do it once for both parts.

### B5. `POLICY.md`

New section "Per-player process" with the B1 block and two sentences: why
the order (volume is the stable part; matchup is a modifier), and the
verdict rule. Cross-reference the `player` recipe.

### B6. Phase 2 (separate job, not this session)

Three inputs the methodology calls for that the pipeline does not compute:
red-zone share (targets and carries inside the 20), route participation,
and air yards / aDOT. nflverse play-by-play has red zone and air yards;
its participation feed has routes. They would enter step 3. Scope it after
A and B have run for a week.

---

## Order of work and definition of done

Part A is done. What remains is Part B plus the residue above:

1. B2 → test → B5 → B3 → commit.
2. B3 skill edit → one `.skill` package → hand over (the ESPN wording from
   Part A is already in the skill; keep it).
3. B4 routine → one delete-and-recreate (the current routine already names
   four leagues; keep that text).
4. Part A residue (K/DEF roster count). Delete this file. Commit and push
   `main`.

Done means: `ff.ask player` working on all four leagues; the process in
`POLICY.md` and the skill; the routine's Health/Sit/Waiver lines citing the
process sections; the ESPN overage count reading 16 of 17.
