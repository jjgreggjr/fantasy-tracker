# Miss review — 2026 weeks 4–6 (written 2026-10-27)

Three passes on why E_pts / model / Sleeper were wrong: news on each big miss,
residual attribution on repo data, and an audit of what the tree model ingests.

## Scoreboard (pooled over 4 leagues, position ALL)

| | e_pts | model_blend | Sleeper | market (wk5–6, n=336) |
|---|---|---|---|---|
| pick accuracy | 0.650 | 0.666 | 0.659 | — |
| MAE | 5.64 | 5.45 | 5.42 | 5.41 |
| corr | — | .549 | — | .578 |

Residual SD is ~7 at every position for every predictor. MAE ~5.4 is the floor.

## The 18 biggest misses, by cause

- **In-game variance (7, the largest point misses):** McMillan wk4, Lamb wk4 and wk5,
  Maye wk5, Kyren wk4, Likely wk5, Warren wk5. Multi-TD games or 45–59 attempt
  shootouts. Vegas pointed the wrong way (B-Rob) or gave nothing (Maye). No weather
  or CB1-out factor in any of them. Not fixable.
- **Status misses knowable by Friday (6):** Coker wk4, DJ Moore wk5/6, Caleb Williams
  wk5, E. Wilson/Price wk4, Skowronek wk6. Practice trend (DNP/LP), coach pressers,
  IR/PS transactions. Caleb wk5 is a bug: he had missed wk3–4 and was still 14.8.
- **Teammate-out redistribution reported 1–3 days early (4):** Bernard wk6, Cooper wk6,
  E. Wilson wk4, McMillan wk4 (partial). `tm_out_*` exists but trailing volume lags.
- **Coaching decision with only soft signals (1):** Tracy over Skattebo wk6.

## Residual attribution (eta² on model residual, 597 player-weeks)

| factor | eta² | direction | in tree? |
|---|---|---|---|
| model vs props market | .059 | model < market: +2.7 (WR/TE +3.3) | opt-in family, not primary |
| TD rate per opp (e_pts only) | .037 | e_pts high-TD −1.6, low-TD +1.8 | tree yes (td_luck), E_pts no |
| dvp_rank tercile | .011 | easy D +1.6, WR vs easy +2.3 | yes, under-weighted |
| total_line / small favorite | .008 | high total +1.6, fav ≤6.5 +1.5 | yes, shrunk |
| home/away, roof, div, slot, depth, confidence, trending | ≤.005 | no stable sign | mixed |

Calibration: 78.6% inside [p10,p90]; 12.9% above p90 vs 8.5% below. RB/WR upper
tails too short, TE lower tail too fat, QB band too wide.

## What we have vs what we'd need

- **Have, unused by the model:** `defense_vs_pos.csv` rec/rush/pass yds + TD splits;
  pbp 2026 (pass_location, air_yards, yardline) via existing `pbp_usage` loader;
  injuries table for all positions (DBs dropped by `OFFENSE_DEPTH_POS`); props.csv wk5+.
- **Free and in-season:** nflverse pbp (Tue), FTN charting (Wed: box count, pass
  rushers, blitzers by defteam), Open-Meteo forecast (reachable from Actions only).
- **Not free / not in-season:** slot alignment and man/zone coverage (participation
  data is post-season only; PFF/NGS paid). Drop.
- **Observed weather already ablated flat** (2025 −0.022, 2024 +0.003).

## Plan, in order (each is an opt-in family through `model.trial`, SHIPS rule unchanged)

1. `dvp_split`: pass/rush/TE/RB-receiving yds + TD allowed, opp EPA/play pass vs rush,
   pressure and light-box rate by defteam, l4. Cheapest new information.
2. `opp_db_out`: count of Out/Doubtful starting DBs on the opponent from the injuries
   table (needs DB depth retained in depth_charts).
3. Components mode: target-share stability (std of share l6) and TD/opp as shrinkage
   inputs. Also port `td_luck` regression into E_pts itself.
4. Status layer: zero any player with 2+ consecutive DNP weeks and no Full practice;
   ingest IR activations / PS elevations (fixes Caleb, Coker, Moore, Cooper).
5. Widen RB/WR p90, narrow QB band.
6. Promote the market family once props history reaches 6+ weeks (wk 11).
7. Forecast weather only if 1–3 move nothing.

Judge on pick accuracy at WR/TE, not RMSE alone. Changing the serving spec restarts the
6-week live gate (POLICY.md).
