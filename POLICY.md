# Roster policy — how keep-value is decided

This is the reasoning the tracker applies when it answers "who should I cut."
It is deliberately written down so the answers are consistent week to week and
so you can argue with the assumptions rather than with a black box.

## Two different questions

`E_pts` (PlayScore) answers **"will he produce this week."** It is projection
blended with recent production, adjusted for the matchup.

`keep_value` answers **"should he occupy a roster spot."** In a dynasty league
that is a different question with a longer horizon, so it uses different
inputs. A 31-year-old back can have a higher E_pts and a lower keep-value than
a 23-year-old backup, and that is correct.

## The three inputs to keep-value

**1. Value over replacement (not raw points).** A roster spot is only worth
what it buys you over the waiver wire. If the best free-agent WR scores 7.3 and
your WR6 scores 6.0, that player is worth *negative* 1.3 — cutting him costs
nothing because the wire replaces him. This is why raw points are the wrong
measure for cut decisions.

**2. Age runway, from positional aging research.** Encoded in
`ff/policy.py` as `AGE_CURVES`, expressed as remaining dynasty value from 1.0
(full runway) to near zero.

| Position | Peak | Decline begins | Notes |
|---|---|---|---|
| RB | 24–26 (recent studies put the average peak season near 24.8) | 27, steep after 28 | ~93% of peak seasons happen before age 29. The commonly cited sell window is 25–26 — by 27–28 you are already late. |
| WR | 26–28 | 29, steep at 32–33 | Breakouts overwhelmingly happen in years 2–4. Deep threats age worse than possession receivers. |
| TE | 26–29 | 31–32, gradually | Slowest to develop and slowest to fall off; holds within ~10–15% of peak into the early 30s. |
| QB | 27–35, nearly flat | Late, abrupt, variable | Depends on processing more than athleticism, so age matters least here. |

Sources: [4for4 production curves](https://www.4for4.com/2025/preseason/production-curves-positional-breakouts-prime-years-and-falloffs-age),
[Apex Fantasy Leagues RB peak age](https://apexfantasyleagues.com/peak-age-nfl-running-back/),
[fantasyhistorydata age curves](https://fantasyhistorydata.com/age-curves-and-historical-fantasy-production/),
[Fantasy Footballers dynasty RB lifecycle](https://www.thefantasyfootballers.com/articles/the-lifecycle-of-a-dynasty-running-back-fantasy-football/).
The curves are our encoding of the consensus across these, not any single
article's table, and are deliberately coarse — they are population averages
and individual players vary.

**3. Usage trend.** Direction of opportunity over recent weeks. A rising
snap/target share is evidence the role is real; a falling one is evidence it is
being taken away.

## How the three are weighted

Two things move the mix, both of them structural rather than opinion:

- **League type.** Redraft ignores age entirely (85% value-over-replacement,
  15% trend). Dynasty and keeper leagues weight the future.
- **Bench depth.** A deep bench is what makes stashing young players *possible*.
  With 16 bench spots you can afford to hold upside; with 3 you cannot. So the
  bench-to-starter ratio raises the weight on age.

| Bench / starters | Value over replacement | Age runway | Trend |
|---|---|---|---|
| under 1.0 (shallow) | 60% | 25% | 15% |
| 1.0 – 1.5 (normal) | 50% | 30% | 20% |
| over 1.5 (deep) | 40% | 40% | 20% |
| any redraft league | 85% | 0% | 15% |

## Protections

- **Current starters are never suggested as cuts.** The lineup is solved first.
- **Young stashes are protected.** A skill player 24 or under with two years or
  less of experience is sorted behind everyone else, because a developmental
  asset that is not yet producing is exactly what a deep dynasty bench is for.
  This is the rule that stops the system from dropping a rookie for a
  replaceable veteran who scores more today.
- **Injured players are surfaced, not auto-cut.** IR and taxi slots exist to
  hold them; using a slot is usually better than dropping the player.

## Status policy — why a depth rank says what it says

A depth chart lists an injured player last. Treating that rank as evidence of
his role punishes him for being hurt, and we did exactly that: Zach Charbonnet
was ranked a near-cut because he sits RB4 on PUP, despite a 2025 season of 48%
snap share and 13 opportunities per game. Three separate penalties, one cause.

Every run now builds `data/status.csv`, one row per player on a depth chart:

| Field | What it is for |
|---|---|
| `status_flag` | healthy / questionable / doubtful / out / ir / pup / unknown |
| `play_prob` | chance he suits up — designation blended with practice participation, which is weighted more heavily because it is the better predictor |
| `practice` | DNP / Limited / Full, from Sleeper |
| `espn_rank` vs `sleeper_rank` | **two independent depth charts.** A disagreement of 2+ places is the signature of a rank driven by availability rather than role |
| `blocked_by` | every player listed ahead of him, **with their statuses** |
| `opportunity_ahead` | share of team volume held by players ahead of him who probably will not play |

`opportunity_ahead` is what turns news into a number: when the starters ahead
of a backup are hurt, his expected volume should rise *this week*, before any
projection updates.

**Enforcement.** Every recipe prints a status line for every player it names,
including who is ahead of him. Every league report opens with a status-warnings
section. No recommendation is made without that check, because the failure was
never a lack of data — it was a missing question.

Roster decisions also use `keep_pts` rather than `E_pts`: a player who cannot
play this week still occupies a spot for a reason, so his value falls back to
what he produced in the role he actually held, discounted 15% for the
uncertainty of a return.

## Trade policy

A trade closes when it solves a structural problem for the other owner, not
when a value chart says it is fair. `ff/trade.py` looks for four concrete
reasons someone would want a specific player, all checkable from data we hold:

1. **Handcuff.** You hold the direct backup to a starter they own. This is the
   strongest single reason, because the player is worth more to that one owner
   than to the market.
2. **Positional need.** They have two or fewer startable players at the
   position.
3. **Aging room.** They are in win-now posture and two or more of their
   startable players at that position are 28 or older.
4. **Plain production.** He would immediately be their #1 or #2 at the
   position. Without this, your best player looks unwanted, which is wrong.

**Posture** is the average age of an owner's startable players. Above 26.5 is
win-now — they will pay for production. Below is rebuilding — they will pay for
age. This decides both who to approach and what to ask for.

**The ask always comes from their bench.** An owner will part with a bench
player far more readily than a starter, and a trade that costs them a starting
spot usually dies. In dynasty leagues the ask is filtered to players 25 or
under who have a path to snaps.

Every suggested ask is labeled with whether you are gaining value, roughly
even, or giving value up, using the same keep-value scale as cuts. Asking for
more than you give is fine — it just means expect a counter, and the tool says
so rather than pretending the deal is balanced.

**Sell-high candidates** are your players who still produce but have passed
their position's age cliff: RB 26+, WR 28+, TE 29+, QB 33+. That is the window
the aging research points to — selling at 27 for a running back means selling
after the market already knows.

## What this policy deliberately does not do

It does not price draft picks, it does not value future rookie drafts, and it
does not know an owner's personal preferences or rivalries. It also does not
override you — it ranks and explains, and every number is in the CSV so you can
check it.
