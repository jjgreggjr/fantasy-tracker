// Miss research swarm for the /checkin skill (skills/checkin/SKILL.md, step 6).
//
// Run with the Workflow tool:
//   Workflow({ scriptPath: "skills/checkin/miss_research.workflow.js",
//              args: { season, week, researched_at, misses } })
// where `misses` is the JSON array printed by `python -m ff.checkin misses --json`
// and `researched_at` is today's date (scripts cannot read the clock).
//
// Shape: one Haiku news agent + one Haiku data agent per big miss, a Haiku
// synthesis per miss, two Haiku refuters per "foreseeable" claim, then one
// Sonnet merge that turns the surviving signals into runnable
// `python -m model.trial` candidate strings. Returns the JSON that
// `python -m ff.checkin record --week W --from <file>` consumes. Nothing here
// writes to the repo.

export const meta = {
  name: 'checkin-miss-research',
  description: 'Research why each big model miss of the week happened; raise testable candidates',
  phases: [
    { title: 'Research', detail: 'per miss: news agent + data agent (Haiku)', model: 'haiku' },
    { title: 'Synthesize', detail: 'per miss: one why, one foreseeable claim (Haiku)', model: 'haiku' },
    { title: 'Refute', detail: 'two skeptics per foreseeable claim (Haiku)', model: 'haiku' },
    { title: 'Merge', detail: 'one Sonnet agent: dedupe, map to model.trial candidates' },
  ],
}

const { season, week, researched_at } = args
const misses = (args.misses || []).filter(Boolean)
if (!misses.length) return { week, season, researched_at, misses: [], note: 'no misses passed in' }
log(`${misses.length} big misses to research for ${season} week ${week}`)

const FINDINGS = {
  type: 'object',
  properties: {
    why: { type: 'string', description: 'Two or three sentences: what happened in that game and why the line was what it was.' },
    sources: { type: 'array', items: { type: 'string' }, description: 'URLs or "local: data/<file>" strings you actually used.' },
    pre_kickoff_signals: {
      type: 'array', items: { type: 'string' },
      description: 'Things knowable BEFORE kickoff that pointed at this outcome (injury report, beat report, depth chart move, Vegas line, usage trend). Empty if none.',
    },
    in_game_only: { type: 'boolean', description: 'true if the outcome was decided by events during the game (injury, blowout, TD luck) that nothing pre-kickoff hinted at.' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
  required: ['why', 'sources', 'pre_kickoff_signals', 'in_game_only', 'confidence'],
}

const SYNTH = {
  type: 'object',
  properties: {
    why: { type: 'string', description: 'One or two sentences, source named inline (e.g. "per ESPN: ..."). Plain words.' },
    sources: { type: 'array', items: { type: 'string' } },
    foreseeable: { type: 'boolean', description: 'Could a feature computed before kickoff have carried this signal?' },
    foreseeable_signal: { type: 'string', description: 'The exact pre-kickoff signal, or "" if none.' },
    candidate_idea: { type: 'string', description: 'What to test, in words (feature family, column, weighting, status rule). "" if none.' },
  },
  required: ['why', 'sources', 'foreseeable', 'foreseeable_signal', 'candidate_idea'],
}

const VERDICT = {
  type: 'object',
  properties: {
    refuted: { type: 'boolean' },
    reason: { type: 'string' },
  },
  required: ['refuted', 'reason'],
}

const MERGE = {
  type: 'object',
  properties: {
    week: { type: 'integer' },
    season: { type: 'integer' },
    researched_at: { type: 'string' },
    misses: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          gsis_id: { type: 'string' },
          name: { type: 'string' },
          position: { type: 'string' },
          team: { type: 'string' },
          why: { type: 'string' },
          sources: { type: 'array', items: { type: 'string' } },
          foreseeable: { type: 'boolean' },
          foreseeable_signal: { type: 'string' },
          candidate: { type: 'string', description: 'Exact runnable `python -m model.trial --candidate` string, or "".' },
          candidate_evidence: { type: 'string' },
        },
        required: ['gsis_id', 'name', 'position', 'team', 'why', 'sources', 'foreseeable', 'foreseeable_signal', 'candidate', 'candidate_evidence'],
      },
    },
  },
  required: ['week', 'season', 'researched_at', 'misses'],
}

const who = (m) => `${m.name} (${m.position}, ${m.team})`
const gameLine = (m) => {
  const g = m.game || {}
  return `${season} week ${week}, ${m.team} ${g.home ? 'vs' : 'at'} ${g.opponent || '?'} on ${g.gameday || '?'} (spread ${g.spread_line ?? '?'}, total ${g.total_line ?? '?'})`
}
const missLines = (m) => (m.misses || []).map(x =>
  `  - ${x.league}: ${x.comparator} predicted ${x.predicted} -> actual ${x.actual} (error ${x.error}, reason ${x.reason}${x.secondary_reason ? ' / ' + x.secondary_reason : ''})`).join('\n')

const newsPrompt = (m) => `Goal: explain in plain words why ${who(m)} scored what he did in ${gameLine(m)}.

What the model expected vs what happened:
${missLines(m)}
Volume expected/actual: ${m.exp_vol}/${m.act_vol}; TDs expected/actual: ${m.exp_td}/${m.act_td}; snap% trailing/actual: ${m.trail_snap_pct}/${m.snap_pct}; played: ${m.played}; on inactive list: ${m.on_inactive_list}; teammate out: ${m.teammate_out_name || 'none'}.
Box score that week: ${JSON.stringify(m.box)}
Trailing 3-week means: ${JSON.stringify(m.trail)}
Retrospective note: ${m.retro_note || '(none)'}

How: first call ToolSearch with query "select:WebSearch,WebFetch" to load the web tools, then search. Search for the game recap, beat-reporter notes, snap counts, injury news and the Friday/Sunday injury report for that date. ANCHOR EVERY SEARCH ON THE DATE AND OPPONENT ABOVE: search results mix seasons, so discard anything not about the ${season} game on ${m.game && m.game.gameday}. Two to four searches is plenty; fetch at most two pages.

Answer the questions in the schema. "pre_kickoff_signals" means only things a careful reader could have known before kickoff on ${m.game && m.game.gameday}: a limited practice, a beat report about a role change, a teammate ruled out, a line move. In-game injuries, game flow and touchdowns are NOT pre-kickoff signals. If you could not verify the cause, say so in "why" and set confidence low. Return only the structured output.`

const dataPrompt = (m) => `Goal: from the repo's own committed files only (no web), work out what the usage data says about ${who(m)} in ${gameLine(m)}, and whether anything visible before kickoff pointed at it.

What the model expected vs what happened:
${missLines(m)}
Box score that week: ${JSON.stringify(m.box)}
Trailing 3-week means: ${JSON.stringify(m.trail)}

Read (small python/pandas one-offs are fine, read-only):
- data/player_weeks.csv: his rows this season, and every ${m.team} player at his position group that week and the two weeks before (who took the snaps/targets/carries he did not, or that he gained).
- data/inactives.csv and data/depth_charts.csv: which ${m.team} teammates were inactive or moved on the depth chart for week ${week}.
- data/schedule.csv and data/defense_vs_pos.csv: the spread, total and final margin (game script).
- model/reports/retrospective.md, section "Candidate signals (derived)": which served features would carry the signal for his reason class.
Do not run the pipeline or the model; do not modify anything.

Report: what changed in his usage vs trailing (snaps, share, touches), where the volume went or came from, the game script, and whether the trailing weeks already showed the trend (a trend across the prior two or three weeks IS a pre-kickoff signal; a one-game spike is not). Return only the structured output.`

const synthPrompt = (m, news, data) => `Combine two reports on one big miss into one verdict. Player: ${who(m)}, ${gameLine(m)}.
Model vs actual:
${missLines(m)}

News report: ${JSON.stringify(news)}
Data report: ${JSON.stringify(data)}

Rules: "why" is one or two plain sentences naming the source. "foreseeable" is true only if a specific signal existed before kickoff that a pre-kickoff feature could compute (injury report fields, practice status, depth chart, teammate out, usage trend over prior weeks, Vegas line); TD luck, in-game injuries and game flow are not foreseeable. If the two reports disagree, prefer the one with the better-dated source and say so in "why". "candidate_idea" is the lever to test in words: a feature family or column, a weighting, or a status-layer rule; "" when nothing pre-kickoff would have helped. Return only the structured output.`

const refutePrompt = (m, s, lens) => `Try to refute this claim about a fantasy-football model miss. Default to refuted=true if you are not sure.

Player: ${who(m)}, ${gameLine(m)}.
Claim: the miss was FORESEEABLE before kickoff. Signal: "${s.foreseeable_signal}". Why: "${s.why}". Sources: ${JSON.stringify(s.sources)}.

Your lens: ${lens}.
${lens === 'timing' ? 'Was the signal really public before kickoff on ' + (m.game && m.game.gameday) + ', or only clear in hindsight (box score, game recap, post-game quotes)? Check the dates on the sources if you can (ToolSearch "select:WebFetch" to open one).' : ''}
${lens === 'feature' ? 'Could the pipeline compute the signal before kickoff from its inputs (nflverse stats and injury reports, Sleeper, ESPN depth charts, Vegas lines, the gameday inactive list)? Read model/reports/retrospective.md "Candidate signals (derived)" and `python -m model.trial --list` (read-only) to see what exists. A signal that needs a beat reporter\'s judgement is not computable.' : ''}
Return only the structured output: refuted (boolean) and a one-sentence reason.`

// Research -> Synthesize -> Refute, per miss, no barrier between misses.
const perMiss = await pipeline(
  misses,
  (m, _item, i) => parallel([
    () => agent(newsPrompt(m), { label: `news:${m.name}`, phase: 'Research', model: 'haiku', schema: FINDINGS }),
    () => agent(dataPrompt(m), { label: `data:${m.name}`, phase: 'Research', model: 'haiku', schema: FINDINGS }),
  ]),
  ([news, data], m) => agent(synthPrompt(m, news, data), { label: `synth:${m.name}`, phase: 'Synthesize', model: 'haiku', schema: SYNTH }),
  async (s, m) => {
    if (!s) return null
    if (!s.foreseeable) return { ...s, refuted: false, refute_reasons: [] }
    const votes = (await parallel(['timing', 'feature'].map(lens => () =>
      agent(refutePrompt(m, s, lens), { label: `refute:${lens}:${m.name}`, phase: 'Refute', model: 'haiku', schema: VERDICT })
    ))).filter(Boolean)
    const refuted = votes.filter(v => v.refuted).length >= 1   // either skeptic kills the claim
    return { ...s, foreseeable: s.foreseeable && !refuted, refuted, refute_reasons: votes.map(v => v.reason) }
  },
)

const researched = misses.map((m, i) => ({ miss: m, verdict: perMiss[i] })).filter(r => r.verdict)
const dropped = misses.length - researched.length
if (dropped) log(`${dropped} miss(es) produced no verdict (agent skipped or failed); they are left out`)
const foreseeable = researched.filter(r => r.verdict.foreseeable).length
log(`${researched.length} researched, ${foreseeable} foreseeable claims survived the refuters; merging`)

// Barrier is correct here: the merge has to dedupe candidate ideas ACROSS misses.
const mergePrompt = `You are merging the week's miss research into the JSON that \`python -m ff.checkin record\` writes to model/reports/miss_research.md and model/candidates.md. Season ${season}, week ${week}, researched_at "${researched_at}".

Per-miss verdicts (foreseeable=true means two skeptics failed to refute it):
${JSON.stringify(researched.map(r => ({
  gsis_id: r.miss.gsis_id, name: r.miss.name, position: r.miss.position, team: r.miss.team,
  misses: r.miss.misses, verdict: r.verdict,
})), null, 1)}

Read first (read-only; never run the pipeline, never modify files):
- model/candidates.md: the format, and every candidate string already raised or tested. Never re-raise a string that is already there.
- \`python -m model.trial --list\`: the families, flags and which columns are in the serving inputs. A candidate must be a string that command accepts verbatim (e.g. \`+cols:<col>\`, \`recent:4\`, \`param:lightgbm.reg_lambda=50\`, a family flag). Read \`model/trial.py\`'s argument parsing if the list leaves the syntax unclear.
- model/reports/retrospective.md, section "Candidate signals (derived)".

Rules:
- One output row per input miss, same gsis_id/name/position/team, with the verdict's why/sources/foreseeable/foreseeable_signal carried over (tighten the prose; one or two sentences; pipes "|" are not allowed in text).
- "candidate" is non-empty ONLY when (a) foreseeable is true and (b) there is a runnable model.trial string that would carry that signal and (c) it is not already in candidates.md. Several misses pointing at the same lever share ONE candidate string: put it on the first such miss and leave "" on the others (mention "same lever as <name>" in their candidate_evidence).
- A signal that belongs to the status layer (Out/Doubtful/inactive timing, practice status) is NOT a model.trial candidate: candidate "" and say "status layer, not a trial.py candidate" in candidate_evidence.
- A signal with no built feature: candidate "" and candidate_evidence "needs a feature first: <what>".
- candidate_evidence (when candidate is set): one or two sentences in the voice of the existing candidates.md evidence column: which miss(es), the signal, what the trial tests.
Return only the structured output.`

const merged = await agent(mergePrompt, { label: 'merge', phase: 'Merge', model: 'sonnet', schema: MERGE })
if (!merged) return { week, season, researched_at, misses: [], note: 'merge agent failed; per-miss verdicts follow', researched }
merged.week = week; merged.season = season; merged.researched_at = researched_at
const raised = merged.misses.filter(x => x.candidate).length
log(`done: ${merged.misses.length} misses recorded, ${raised} candidate(s) raised`)
return merged
