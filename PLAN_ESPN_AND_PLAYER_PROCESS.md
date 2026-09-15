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

## Part A — the ESPN league

### A0. What is true today (verified 2026-09-15)

- Support is coded but off. `ff/espn.py` fetches
  `lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leagues/{id}`
  with views `mSettings`, `mRoster`, `mTeam`, parses scoring, slots, owners and
  rosters. `ff/run_weekly.py` (the "optional ESPN league" loop, ~line 294)
  runs it for every entry in `config.json` → `espn_leagues`. That list is
  `[]`, so the loop never runs. That is the whole reason ESPN never appears.
- The workflow already handles the credentials: the "ESPN cookies" step in
  `.github/workflows/pipeline.yml` writes `secrets/espn_cookies.json` from the
  two secrets when both are set, and prints a skip line otherwise.
- **ESPN's API is blocked from Claude's sandbox.** The egress proxy answers
  403 to `CONNECT lm-api-reads.fantasy.espn.com:443` (policy denial). Sleeper
  and GitHub are allowed. Consequences:
  - The GitHub Actions runner is the only place ESPN can be fetched. All
    ESPN validation happens by dispatching `pipeline.yml` and reading the job
    log and the committed files, not by running `ff.espn` locally.
  - There is no live ESPN lineup read at question time. `ff.ask live` for the
    ESPN league must return the committed snapshot with its timestamp and say
    so plainly. If James later allows that host in the environment's network
    policy, the code below tries the live read first and this limitation
    lifts on its own.
- Player crosswalk coverage: 693 active QB/RB/WR/TE in `data/players.csv`,
  51 without an `espn_id` (29 of them 2026 rookies, mostly undrafted). Across
  his three Sleeper rosters only one player lacks one. Expect a handful of
  unmatched ESPN roster entries; they show as unscored, not as errors.
- `config.json` → `sources.espn_projections` is referenced nowhere in `ff/`.
  Dead key; see A6.

### A1. What only James can supply (ask for these in chat, first message)

1. **League ID.** The number after `leagueId=` in any fantasy.espn.com URL
   for that league. Safe to say in chat; it is not a credential.
2. **Whether the league is private.** ESPN leagues are private by default.
   If in doubt, treat it as private. Private means the two secrets are
   required or the fetch returns 401 and the league is skipped with a warning.
3. **His team name** as shown in ESPN (the team, not his display name), and
   the league's format: redraft, keeper or dynasty. ESPN has no dynasty flag
   in its settings, so the format comes from config (A3).
4. **The two secrets, added by him in GitHub, never in chat:** repo
   Settings → Secrets and variables → Actions → New repository secret:
   `ESPN_S2` = the `espn_s2` cookie value, `ESPN_SWID` = the `SWID` cookie
   value **including the curly braces**. Both are visible in the browser
   while logged into fantasy.espn.com: DevTools → Application (Chrome) or
   Storage (Firefox) → Cookies → `https://fantasy.espn.com`. `espn_s2` is
   long and expires after roughly a year; `SWID` is stable.

Do not block on item 4 to start coding. Everything in A2 to A5 is written
against the documented response shape and validated by the first run.

### A2. Code: `ff/espn.py`

1. **Identify his team without relying on a display name.** ESPN's `members[].id`
   is the SWID (with braces), and `teams[].owners` lists member ids. When
   cookies are loaded, `my_roster_id` = the team whose `owners` contains the
   SWID from `secrets/espn_cookies.json`, and `my_user_id` = that SWID.
   Fallbacks, in order: `my_team_name` matched case-insensitively against
   `teams[].name` (and `location + " " + nickname` for older payloads), then
   against the owner display name. Today `run_weekly` matches
   `my_team_name` against `owner_name` only, which is the member display
   name, not the team; that is a naming trap. Move the matching into
   `espn.py` (`find_my_team(data, meta, cookies, my_team_name) -> team id`)
   and have `run_weekly` select `mine` by `roster_id`.
2. **`parse_meta` additions:** `fetched_at` (UTC ISO, same format as
   `sources.sleeper_league_detail`), `my_roster_id`, `my_user_id`, `season`,
   and a `type` override from the config entry when present. Remove the dead
   `keeper = ... and False` line.
3. **Slot names must match `scoring.SLOT_ELIGIBILITY`.** Extend `SLOT_ID`:
   `3 → "WRRB_FLEX"`, `5 → "REC_FLEX"`, `7 → "SUPER_FLEX"` (ESPN calls it OP),
   `23 → "FLEX"`, `8..15 → IDP names` (DT, DE, LB, DL, CB, S, DB, DP),
   `18 → "P"`, `19 → "HC"`, `24 → "ER"`, `25 → "ROOKIE"`. Compute `idp` from
   the IDP set, not from `startswith("SLOT")`. Unknown ids still become
   `SLOT{n}` and are logged.
4. **Scoring position overrides.** ESPN expresses TE-premium and similar as
   `pointsOverrides` keyed by position id on a scoring item. Today
   `parse_scoring` takes the first override as the base value, which is
   wrong for every other position. Instead: base = `points`; for statId 53
   (receptions) with an override for position id 4 (TE), 2 (RB) or 3 (WR),
   emit `bonus_rec_te` / `bonus_rec_rb` / `bonus_rec_wr` = override − base,
   which `scoring.POS_REC_BONUS` already understands. Log any other override
   as unmapped.
5. **Crosswalk to Sleeper ids.** In `parse_rosters`, after matching on
   `espn_id`, fill `sleeper_id` from `players` by `gsis_id`. Everything
   downstream (`data/my_roster.csv`, `ff/live.py`'s pool, the `ask` recipes)
   keys on `sleeper_id`; ESPN rows currently carry `pd.NA` there and would
   be dropped or mis-keyed.
6. **User agent.** Keep the current one for the first run. If the run log
   shows 403 from ESPN on a league that is public or has valid cookies, switch
   `UA` to a plain browser string; ESPN's CDN sometimes rejects bot-like agents.
7. **Verify the statId map against the league's real `scoringItems`** in the
   first run's log (the code logs unmapped ids at INFO). 41 is mapped to
   `rec_tgt`; if this league scores 41 and 53 both, one of them is receptions
   and PPR would double count. Fix the map from the evidence, not from memory.

### A3. Config: `config.json`

Add one entry (values from A1; slug is your choice, kebab-case):

```json
"espn_leagues": [
  {"league_id": "<id>", "slug": "<slug>", "my_team_name": "<team name>",
   "type": "redraft"}
]
```

Update `config.example.json` to show the same shape. Delete the unused
`sources.espn_projections` key from both files (see A6 before deleting).

### A4. Code: the rest of the pipeline

- `ff/run_weekly.py`, ESPN loop: set `meta["roster_fetched_at"] = meta["fetched_at"]`;
  select `mine` by the team id from A2.1; upsert the ESPN rosters into
  `data/league_rosters.csv` and `mine` into `data/my_roster.csv` with the
  existing keys (which now work because of A2.5); add the ESPN league to the
  `freshness["league rosters"]` line. Keep the "could not be loaded" warning
  path: a private league without secrets must degrade to a warning, never a
  FAIL.
- `ff/leagues.py` `write_league`: no change needed; the transactions block
  is already Sleeper-only. Confirm `report.md` renders with
  `roster_fetched_at` for the ESPN meta.
- `ff/live.py`: branch on `meta.get("platform")`. For `"espn"`: try
  `espn._get(season, league_id, ["mRoster"], cookies={})` once (works only if
  the league is public and the host is allowed); on any exception build the
  same result dict from the committed `roster.csv` / `available.csv`, with
  `fetched_at = meta["roster_fetched_at"]`, `diff` empty, and a new key
  `live: False`. `render()` prints
  `Lineup as of <time> (committed snapshot — ESPN cannot be read live from
  here)` as its first line when `live` is false. `ff/ask.py`'s `live`
  recipe must not print "LIVE READ FAILED" for this case; it is expected.
- `ff/verify.py`: add a WARN when an `espn_leagues` entry produced no league
  folder this run, so a silently skipped league is visible in `runs.csv`.

### A5. Validate with a real run

1. Commit A2 to A4 on `main` with the config entry. Dispatch `pipeline.yml`.
2. Read the job log (`actions_get` / `get_job_logs`). Expect: the "wrote
   secrets/espn_cookies.json" line (or the skip line if James has not added
   the secrets yet), no ESPN warning, an INFO line with unmapped statIds,
   and the unmatched-player count.
3. On `main` after the run: `leagues/<slug>/league.json` has
   `platform: espn`, `my_roster_id` set, `roster_fetched_at` set, the slot
   list matching what ESPN shows, and `scoring` with the right PPR value;
   `roster.csv` has his players with `E_pts`; `available.csv` exists;
   `reports/latest.md` has a fourth league section.
4. Locally: `python -m ff.ask live <espn-slug>`, `lineup`, `cut`, `options`,
   `explain`, `trade` all run without error and the first line of `live`
   names the snapshot time.
5. Run `python -m ff.run_weekly --verify-only` and the existing checks; E_pts
   for the three Sleeper leagues must be unchanged by this work.

If the run logs 401: the secrets are missing or wrong; tell James which (401
with cookies written means the values are wrong or expired). If 403: A2.6.

### A6. ESPN projections (optional, decide during A5)

The `mRoster` view carries ESPN's own weekly projection per player
(`playerPoolEntry.player.stats` entries with `statSourceId == 1` for the
current `scoringPeriodId`). Today all projections come from Sleeper. Wiring
ESPN's as a second source would let `E_pts` blend two projections and is
what the dead `espn_projections` key was reserved for. Not required for the
league to be tracked. If skipped, delete the key; if done, it becomes a
pipeline addition alongside Part B's phase 2.

### A7. Skill, routine, docs

- `skills/fantasy/SKILL.md`: add the ESPN slug to "His leagues" with
  "(ESPN, <format>)"; under "Lineup freshness" add: for the ESPN league there
  is no live read; `ff.ask live` returns the committed snapshot and states its
  time; say "your ESPN lineup as of <time>" and never present it as current.
  Fix the sentence "You do not need Sleeper access" to say the Sleeper live
  read is the exception. Package the skill (`python -m scripts.package_skill`
  from the skill-creator) and hand James the `.skill` file; he installs it
  with "Save skill". Nobody else can.
- Routine "Fantasy weekly check-in (Tue)": add the ESPN slug to the slug
  list and one line: "the ESPN league has no live read; use its committed
  snapshot and state the time". Prompt edits are blocked on update; delete
  and recreate with `notifications: {push: true, email: true}`.
- `README.md`: the ESPN paragraph says cookies "never leave your machine";
  rewrite for the Actions secrets flow and note the sandbox block.

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

1. Ask James for A1 items 1 to 3 (and remind him of item 4, secrets only in
   GitHub). Start A2 while waiting.
2. A2, A3, A4 → commit → dispatch → A5. Fix and re-run until step 3 of A5
   passes. If secrets are still missing, the league folder will not exist;
   everything else must still be green, and say exactly that.
3. B2 → test → B5 → B3 → commit.
4. A7 + B3 skill edit → one `.skill` package → hand over.
5. A7 + B4 routine → one delete-and-recreate.
6. `README.md` ESPN paragraph. Delete this file. Commit and push `main`.

Done means: a fourth league folder on `main` with his ESPN team scored and
timestamped (or, if secrets are absent, a WARN naming that and nothing else
red); `ff.ask player` working on all leagues; the process in `POLICY.md`
and the skill; the routine naming four leagues.
