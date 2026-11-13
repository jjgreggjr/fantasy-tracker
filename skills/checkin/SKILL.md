---
name: checkin
description: "James's lineup check-in for the fantasy tracker (jjgreggjr/fantasy-tracker): pipeline health, last week's results, season lineup efficiency, whether the model served, the E_pts / model / call lineup for all four leagues, and a once-a-week research pass on the model's big misses that raises candidates. Use whenever he says /checkin, 'check-in', 'Thursday check', or asks to check his lineups across leagues."
---

# /checkin

The Thursday routine, written down. Same shape every run, fresh injuries every run,
cheap to re-run. Steps 1-5 only read committed files (plus the live Sleeper read).
Step 6 is the one slow step and the only one that writes; it runs once per week and
skips itself after that.

Clone or `cd` into the repo first (public, no credentials):
`git clone --depth 1 https://github.com/jjgreggjr/fantasy-tracker`. Never run the
pipeline or the model from here; the workflow writes the data, this skill reads it.

## Steps 1-4: one command

```bash
python -m ff.checkin
```

Prints `## Health`, `## Last week`, `## Season efficiency`, `## Serve`, `## Miss research`.
Read-only. Quote from it; do not recompute anything it already says.

- **Health** ends in `HEALTH: healthy` or `HEALTH: <piece> — fix: <...>`. That line is the
  first line of the report. If it names a fix, say it in James's words ("press Run
  workflow on the Actions tab"). WARNs about "no rows yet" or "0% have practice
  reports" are normal early in the week and the command already ignores them.
- **Last week**: three bullets at most. Points and W/L per league, actual vs perfect
  lineup points, model vs Sleeper vs E_pts pick accuracy. Then the `Gate:` line
  verbatim. The gate is met by a hair; E_pts is still authoritative.
- **Season efficiency**: paste the small table as printed. Nothing else.
- **Serve**: one line, when the serve for this week landed and whether it was frozen
  before kickoff. The refit sentence it prints is the whole answer to "was the model
  refit": trees retrain every serve, hyperparameters are frozen by design. Say it once.

## Step 5: per league, the fantasy skill

Follow `skills/fantasy/SKILL.md` exactly. For each of the four leagues
(`gooma-s-family-league`, `we-can-think-of-something-funny`, `where-you-at`,
`james-gregg-espn`), in that order:

```bash
python -m ff.ask live <slug>
python -m ff.ask lineup <slug>
```

Lineup freshness rules from that skill apply: the live read's fetch time is the first
line of the league block; the ESPN league is a committed snapshot and is labelled as
such. Print, per league, in full: the slot-by-slot table (E_pts lineup, model lineup,
the call), the `Net vs current:` / `Totals` / `Differ:` lines, and the "Status of every
starter" block. Then one line of injury flags: any starter who is Out / Doubtful /
IR / inactive, and any benched Out/IR player whose healthy role beats a starter ("is
he back yet?"). The legend lines from the fantasy skill go above the first table.
Points and volume together, every player you name. The tables are the deliverable;
the prose around them stays short.

## Step 6: miss research (once a week, writes)

James has opted in to a Haiku swarm for this step ("ultracode, full swarm"), run with
the Workflow tool. It researches each big miss of the last completed week, asks
whether a pre-kickoff feature could have seen it coming, and raises `open` rows in
`model/candidates.md` for anything testable with `python -m model.trial`. It never
tests a candidate; it only raises it.

1. **Gate.**
   ```bash
   python -m ff.checkin gate
   ```
   Exit 3 = `GATE: researched (...)`: the week's section already exists in
   `model/reports/miss_research.md`. Skip the rest of step 6 and report
   "misses researched: already done for week W on <date>" with the counts from that
   section. Exit 0 = `GATE: open`: continue. The gate week is the latest week in
   `data/model_misses.csv`, which is the last completed week once the pipeline's
   retrospective has run; it is usually one behind the week being served.

2. **Queue.** Write the research queue to a scratch file (the scratchpad directory, not
   the repo):
   ```bash
   python -m ff.checkin misses --json > <scratch>/misses.json
   ```

3. **Swarm.** Call the Workflow tool with the script in this folder; pass the queue and
   today's date in `args` (scripts cannot read the clock):
   ```
   Workflow({
     scriptPath: "skills/checkin/miss_research.workflow.js",
     args: { season: 2026, week: <W>, researched_at: "<YYYY-MM-DD>", misses: <contents of misses.json> }
   })
   ```
   Per miss: a Haiku news agent (web, anchored on the game date and opponent) and a
   Haiku data agent (the repo's own usage files), a Haiku synthesis, then two Haiku
   skeptics that try to refute any "foreseeable before kickoff" claim. One Sonnet
   merge turns the surviving signals into runnable `model.trial` strings, dedupes them
   against `model/candidates.md`, and returns the record JSON. Watch for the
   completion notification; do not poll. If the Workflow tool is not available in the
   session, run the same prompts as parallel `Agent` calls with `model: "haiku"` and
   assemble the same JSON by hand.

4. **Record.** Save the workflow's return value to `<scratch>/research.json`, then:
   ```bash
   python -m ff.checkin record --week <W> --from <scratch>/research.json
   ```
   This is the only write: it appends a `## 2026 week W` section to
   `model/reports/miss_research.md` (the skip marker for next time) and one `open`
   row per new candidate to `model/candidates.md`, skipping strings already there.
   It refuses to run twice for the same week without `--force`.

5. **Commit.** Branch from `origin/main`, commit the two files, push, open the PR,
   merge it (CLAUDE.md). Commit message: `checkin: miss research for 2026 week W
   (N misses, M candidates)`. If the push or PR is not possible from the session,
   say so and leave the diff in place.

## Report shape

In this order, in the CLAUDE.md voice (decision first, short bullets, no preamble):

1. `HEALTH:` line.
2. Last week: three bullets, then the `Gate:` line.
3. Season efficiency: the small table.
4. One block per league: fetch time, the lineup table, the call, injury flags.
5. `Misses researched: N, candidates raised: M`, then one line per candidate
   (player, the signal, the candidate string) or "already done for week W on <date>".
6. One line: "want the full breakdown?"

## Re-running

Run it again Saturday morning, Sunday morning, any time: steps 1-5 re-read the
files and the live Sleeper lineups, step 6 sees its marker and skips. The next week
opens when the pipeline's retrospective writes the new week's rows into
`data/model_misses.csv`.
