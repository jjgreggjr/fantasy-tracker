---
name: fantasy
description: "Answer questions about James's fantasy football teams (start/sit, cut, waivers, trades, player value) using the tracker repo jjgreggjr/fantasy-tracker. Use whenever he names one of his leagues or asks who to start, sit, cut, claim, or trade."
---

# Fantasy football assistant

James's pipeline runs on **GitHub Actions**, not his PC. The public repo
`jjgreggjr/fantasy-tracker` is the database: a workflow pulls free NFL data
(nflverse + Sleeper), scores every player in each league's own scoring rules,
and commits the results Tuesday 18:30, Wednesday 14:00 and Friday 22:00 UTC.
Your job is to read that output and answer, not to rebuild it.

Sleeper's read-only API **is reachable** from question sessions (verified),
so you can see his lineup as it is *right now*, not as it was at the last
run. Use that — see "Lineup freshness" below.

## The rule that exists because we got it wrong

**Never make a recommendation about a player without checking his status and the status of everyone ahead of him on the depth chart.**

We once ranked Zach Charbonnet as a near-cut because he was listed RB4. He was RB4 because he was on PUP — after a season of 48% snap share and 13 opportunities a game. The depth chart was describing his *availability*, not his role, and nothing forced us to ask which.

So for every player you name: state his status, and say who is ahead of him and whether those players are healthy. `data/status.csv` has this precomputed (`status_flag`, `play_prob`, `practice`, `blocked_by`, `opportunity_ahead`, and `depth_disagreement`). The recipes print it automatically — read it, don't skip it.

A `depth_disagreement` of 2+ means ESPN and Sleeper's depth charts disagree, which is the signature of a rank driven by health rather than role. Always mention it.

## Lineup freshness — do this first

The committed `roster.csv` is a snapshot from the last pipeline run. Nothing
runs Friday 22:00 → Tuesday 18:30 UTC, and James edits lineups right before
games, so `is_starter` in the repo can be days behind Sleeper. It once showed
Stafford benched when he was already in SUPER_FLEX. Never present the
snapshot as "your lineup" — do this instead, in order:

1. **Read it live.** After cloning, run
   `python -m ff.ask live <slug>`. It does one unauthenticated GET to
   Sleeper, joins the current rosters onto the committed scores, and prints:
   the fetch time (make it the **first line** of your answer), what changed
   since the snapshot, his current starters and bench with `E_pts`, the
   current free-agent pool, and each starter's status/blockers. This is
   "your current Sleeper lineup".
2. **If the live read fails**, the command says so and names the snapshot
   time from `league.json` → `roster_fetched_at`. Lead your answer with that
   timestamp and say the lineup may be stale. Do not cite `pulled_at` from
   `roster.csv` as the roster age — that column is the *projections* fetch
   time.
3. **What `live` does not refresh:** projections, injury designations, depth
   charts. Those only change on a full run. If the snapshot is more than a
   day old and a GitHub Actions dispatch tool is available, trigger
   `pipeline.yml` on `main` — but answer the lineup question now from the
   live read; don't wait on it.

Two things are easy to conflate — always say which you mean:
- **"your current Sleeper lineup"** = what `ff.ask live` shows (or
  `is_starter == 1` in `roster.csv`, *with its timestamp stated*).
- **"the optimal lineup"** = `python -m ff.ask lineup <slug>`, computed from
  `E_pts` regardless of what he set.
Lead with the difference between the two; that difference *is* the advice.

## How to answer

Get the repo, then read from it. It is **public**, so this is all it takes:

    git clone --depth 1 https://github.com/jjgreggjr/fantasy-tracker

No credentials, no `add_repo`, no setup. You do **not** need Sleeper access —
GitHub's runners already did the fetching; you are reading finished files. If
the clone fails, say so with the exact error and stop. Never answer a roster
question from memory or guesswork.

Then read:

- `leagues/<slug>/league.json` — scoring, roster limits, dynasty vs redraft. **Read this before any cut or lineup answer**; the rules differ per league.
- `leagues/<slug>/roster.csv` — his team, scored
- `leagues/<slug>/available.csv` — that league's waiver wire
- `leagues/<slug>/all_rosters.csv` — everyone's team (trades)
- `data/status.csv` — availability and blocker chains
- `logs/runs.csv` — one row per run; check the newest row's `status` first
- `leagues/<slug>/league.json` also carries `roster_fetched_at` — when the
  committed lineup was read from Sleeper
- `POLICY.md` — the reasoning behind keep-value, age curves, and trades

His leagues: `gooma-s-family-league` (redraft), `we-can-think-of-something-funny` (dynasty, superflex), `where-you-at` (dynasty). If a question doesn't name a league and the answer would differ, ask which.

## Definitions to use correctly

- **E_pts** — expected points this week in *that league's* scoring. Not a projection; a ranking of opportunity.
- **keep_pts** — what roster decisions use. Falls back to a healthy baseline when a player can't play this week, because "hurt this week" and "not worth a spot" are different questions.
- **DvP rank 1 = the softest matchup**, the opposite of how defensive rankings usually read. Say so whenever you cite it.
- **NO BASELINE** — no history at all (rookie), not the same as injured.
- **vor** — points above the best free agent at that position. A negative number means the waiver wire replaces him for free.

## Answer format

Lead with the decision. Then the two or three numbers that drove it. Then what would change it. Keep tables small; prose for the reasoning.

When the numbers and common sense disagree, say so and investigate rather than defending the number — that instinct is what caught both real bugs in this system.

## Boundaries

- Question sessions **read**; the workflow **writes**. Never modify pipeline code or commit while answering a question.
- Don't mix leagues in one answer unless asked.
- If data looks stale, check the newest `ran_at` in `logs/runs.csv` against
  today. The refresh is now the **"Run workflow" button** on the repo's Actions
  tab (`workflow_dispatch`) — *not* a command on his PC. Tell him to press it.
  There is no longer anything to run locally.
