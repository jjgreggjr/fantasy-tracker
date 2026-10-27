"""Feature builder: every read of raw data goes through `as_of_join` (model/point_in_time.py).
This module must not import a loader, read a file, or touch a store's private frames; the AST test in
tests/test_no_leakage.py fails the build if it does. It may import exactly `TargetRow` and `as_of_join`,
which is why a few constants below are repeated from point_in_time (tests pin them equal).

Two entry points, both pure functions of (store, target):

  spine_for(store, team_game)   who is eligible to be a row for this team-game, from information known
                                before kickoff ONLY (never from 'has a stats row that week')
  build_features(store, t)      one feature row for one (player, game)

Feature families (FAMILIES below is the registry; `feature_columns()` is what Phase 2 trains on):

  lags        last 1/2/3 games the player APPEARED in this season (stats row or offensive snaps), with the
              weeks-ago of each; season-to-date means. Appearance = stats row OR offense_snaps > 0: a TE who
              only blocked appears with 0 points and 0 volume, because that is what he scored.
  prev_season last season's per-game averages (REG). Week-1 and early-season signal; NaN for rookies.
  td_luck     cumulative (actual - expected) TDs to date from ffopportunity, lagged like every other stat
  role        usage-derived role: trailing snap share, rank within team+position, size of the group, and
              teammate-injury context (how many Out/Doubtful teammates at his position, above him, and how
              much trailing volume they vacate). Built from usage, NOT depth-chart rank: the depth-chart
              format changed in 2025 and is not comparable (see load_depth_charts).
  injury      his own most recent report before kickoff. `inj_days_since_report` is produced (Phase 0 tests
              use it) but EXCLUDED from the matrix: its meaning shifts between <=2024 (real timestamps) and
              2025+ (derived kickoff-24h).
  dvp         mean PPR the opponent allowed to his position over its last 2 / 4 / all completed REG games
              (windows count games, end at week-1 by construction)
  vegas       closing spread / total from his team's view, implied team and opponent totals
  context     home/away, rest days, divisional, neutral site, week, season
  weather     OBSERVED temp/wind/roof (v1 proxy for a forecast; `weather_is_backtest_only`). All columns are
              prefixed `wx_` so Phase 2 can ablate the family in one line.
  static      age, experience, career games, draft capital, height/weight, combine (season-start known_at)
  adp         FFC ADP as a season prior (NA until model_fetch has run and the CSVs are committed)
  college     final-college-season production and market share (CFBD), ROOKIE SEASON ONLY: the columns are NA once
              a player is past the season he was drafted into (model/college.py has the identity rules; breakout age
              is always NA, it needs more than the final college season)

Phase 2.5 families (opt-in for the models: `train.OPT_IN_FAMILIES`, so the Phase 2 primary keeps exactly its Phase 2 inputs):

  pbp_usage   play-by-play usage micro-signals over the player's last game (l1), last three games (t3) and season to date
              (std): red-zone and inside-10 target/carry shares of his team, air-yards share, aDOT, WOPR, red-zone and
              inside-10 opportunities per game; plus last season's values for five of them. Shares are ratios of sums.
  pbp_team    the team's own pace, dropback rate and neutral-situation dropback rate (t3, std, last season) and the
              plays and dropback rate his opponent's defense has faced
  pbp_part    share of the team's dropbacks / non-dropback runs / red-zone plays he was on the field for (participation
              feed; the closest free proxy for route participation). BACKTEST-ONLY: the feed is published after the season
  eff         shrunken career-to-date and season-to-date efficiency: catch rate, yards and TDs per target, yards and TDs
              per carry, yards / completion / TD / INT per attempt (fixed priors from 2015-2019, see EFF_PRIORS)
  streak      (Phase 5 candidate, 2026-10-15) has he beaten or missed expectation several games running? Expectation is his
              own lagged xFP (no historical Sleeper projection exists) and his trailing-3 volume: signed residuals of the last
              1/2/3 games, their 4-game sum, run lengths of consecutive over/under games (capped +-6), the volume analogue, and
              the last two games against last season's per-game mean. Last season's REG games precede this season's so the
              first weeks have a window (`_cur_prev`); a game without xFP has a NaN residual and ends a run.
  shape       (Phase 5 candidate, 2026-10-27) player archetype from `Hist` alone (no new table, no new gate call): how steady
              his share is (std of target / carry share over his last 6 games), how spiky his scoring is (coefficient of variation
              of PPR over his last 8, fraction of those 8 at or above 1.5x their mean), how TD-dependent he is (season-to-date
              TDs per opportunity shrunk toward a position prior, share of PPR from TDs over the last 8), and the slope of his
              opportunities over the last 4. Last season's REG games precede this season's, as for `streak`. Raised by the week
              4-7 miss review: high-CV players land above p90 15% of the time against 9-12%, and E_pts's TD-rate bias.
  ecr         (Phase 5 candidate, 2026-10-16) FantasyPros weekly expert consensus (model/fetch_ecr.py), the newest scrape
              assigned to THIS week that is known before kickoff: ecr_rank = PPR superflex OVERALL ECR (the archive's only
              weekly cross-position offense list), ecr_pos_rank = ordinal rank on his weekly PPR position page, ecr_sd / best /
              worst / spread = expert dispersion on that position page, ecr_owned_pct = FP's average roster %, ecr_delta_l1 =
              ecr_pos_rank minus last week's (week - 1, same season). NaN when he has no row for this week known before his
              kickoff: unranked, week 18 and most week 1s (not in the archive), or a Thursday game before that week's scrape.

  nfelo       (Phase 5 candidate, 2026-10-16) greerreNFL public CSVs: nfelo unit values (pass/rush off and def) for his team and
              the opponent and their matchup edges, nfelo elo and QB adjustment, his team's starting QB's 538-style value,
              season-to-date WEPA net, open-to-last spread / total movement and spread tickets share, film margin over the
              last 4 games. Team-game level; model/nfelo.py has the known_at rules (coaches.csv is skipped: not point-in-time).
  pepa        (Phase 5 candidate, 2026-10-16) his own EPA and efficiency from the weekly stats: pass/rush/rec EPA, air yards,
              YAC, first downs, sacks, CPOE, PACR/RACR over l1 / l3 / std, and per-opportunity EPA / YAC rates over l3.

  market      (Phase 5 candidate, 2026-10-16) Sleeper's (Rotowire's) archived weekly projection for THIS game: PPR / standard
              points, targets, carries, pass attempts, yards, total TDs, rank within position that week, and the projection
              minus his own trailing-3 xFP. Table `sleeper_proj` (model/fetch_sleeper_proj.py), stamped kickoff - 1h, read
              through the gate at kickoff. The VALUE only: no "has a projection" column (the archive drops late scratches, so
              presence is partly a post-hoc P(plays)); NaN when there is no row.

  script      (Phase 5 candidate, 2026-10-16) game script and coaching context, top-down: the team's last 4 REG games (this
              season and last) for tempo, no-huddle rate, pass rate over expected, xpass, early-down and win-probability-neutral
              pass rates, red-zone plays, EPA/play and success; its opponent's defense over its last 4 (EPA allowed, pass / rush
              split); Vegas x tendency (implied total x pass rate, spread x PROE); expected team plays (SCRIPT_PLAYS_RULE), pass
              and rush attempts; the player's trailing-3 shares applied to them; head-coach change and tenure (`coaches` table).
              Only what `pbp_team` lacks is a column.

  ngs         (Phase 5 candidate, 2026-10-16) Next Gen Stats over his last published game (l1) and last three (l3): receiving
              separation, cushion, intended air yards and their team share, catch %, YAC over expected; rushing efficiency, RYOE per
              attempt, 8+ box share; passing time to throw, aggressiveness, CPOE, air-yards differential. Thresholded by the NFL,
              so NaN is common (about 70 receivers a week qualify). Weekly release, Friday refresh (`_charting_features`).
  pfr         (same) PFR weekly advanced stats, l1 and l3: drops, drop %, receiving broken tackles, passer rating when targeted;
              yards before / after contact per carry, broken tackles, carries per broken tackle; QB pressure %, blitzes, hurries,
              hits, sacks, drop %, bad-throw %. pfr ids map to GSIS ids through players.parquet. Thursday refresh.
  ftn         (same) FTN charting joined to play-by-play, l3 ratios: his target share on motion / play-action / RPO / screen
              plays, catchable and contested target rates, drops per game; light-box (<= 6) share and mean box on his carries;
              QB blitzed / out-of-pocket / play-action rates, mean pass rushers, interception-worthy rate. FTN starts in 2022: every
              2021 row is NaN (the 2021 asset is a 404), as is week 1. Wednesday refresh.
  dvp_split   (Phase 5 candidate, 2026-10-27) the opponent's defense SPLIT by stat, where `dvp` is whole-position PPR only
              (week 7 miss review: the model under-predicts against easy defenses, +1.6 overall, WR +2.3). Per game allowed to
              the target's position over the defense's last 4 / all completed REG games this season (same window as `dvp`:
              ends at week-1): receiving yards / targets / TDs, rushing yards / attempts / TDs, passing yards / TDs, sacks; from
              play-by-play, EPA per dropback and per rush allowed, dropback rate and explosive-play (20+ yards) rate faced; from
              FTN by defense (`ftn_def`, Wednesday release), light-box rate, blitz rate and pass rushers over the last 4 charted
              games; and the defense's season-to-date rank (1 = most allowed) in receiving and rushing yards to the position
              among the defenses with a game. A row carries the stems that bear on his position (DVPS_POS) and NaN for the rest.

Cross-team aggregates (DvP, teammate context) only ever use GAMES THAT ENDED: the opponent's and our own
previous games are days old. Nothing here sums across the league for the current week (finding 7).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from model.point_in_time import TargetRow, as_of_join

KNOWN_AT = "known_at"
NAN = float("nan")
DAY = pd.Timedelta(days=1)
LISTED = ("Out", "Doubtful", "Questionable")
OUT_LIKE = ("Out", "Doubtful")
POSITIONS = ("QB", "RB", "WR", "TE")
USAGE_WINDOW = 3                       # team games a player stays a spine candidate after appearing in one
# Game rows are stamped with the game's KICKOFF (a lower bound on availability, finding 7). A game that
# kicked off less than this long before the target is not known to have finished (the 1pm slot feeding a
# 4:25pm row, or a player who was waived after an early game and claimed for a late one), so the four result
# tables (player_games, snap_counts, xfp, game_results) are read at kickoff - RESULT_LAG. Thursday -> Sunday,
# and every same-team pair of games, is untouched.
RESULT_LAG = pd.Timedelta(hours=4)
DEPTH_MAX_AGE = pd.Timedelta(days=21)  # a chart older than this is not a current-roster signal
CAREER_CUTOFF_SEASON = 2020            # pinned equal to point_in_time.CAREER_CUTOFF_SEASON by a test
# nflverse roof codes drift: 2021-2023 split retractable roofs into 'closed'/'open' (decided on game day, i.e. observed),
# 2024+ code every retractable-roof game 'closed'. Only the structure is comparable across seasons and knowable
# before the day, so the feature is the three-way structure, and indoor means a FIXED dome.
ROOF_STRUCTURE = {"dome": "dome", "closed": "retractable", "open": "retractable", "outdoors": "outdoors"}


# --------------------------------------------------------------------------- registry
def _lag_cols(names: tuple[str, ...]) -> list[str]:
    return [f"{n}_l{k}" for k in (1, 2, 3) for n in names]


LAG_STATS = ("pts_ppr", "opps", "carries", "targets", "pass_att", "target_share", "carry_share",
             "snap_pct", "xfp")
STD_STATS = ("pts_ppr", "opps", "carries", "targets", "target_share", "carry_share", "snap_pct", "xfp")
PREV_STATS = ("ppg", "opps_pg", "carries_pg", "targets_pg", "target_share", "snap_pct", "xfp_pg")

PBP_WINDOWS = ("l1", "t3", "std")
PBP_USAGE_STATS = ("rz_tgt_share", "i10_tgt_share", "rz_car_share", "i10_car_share", "ay_share", "adot", "wopr",
                   "rz_opps_pg", "i10_opps_pg")
PBP_PREV_STATS = ("rz_tgt_share", "rz_car_share", "ay_share", "adot", "wopr")
PART_STATS = ("pass_snap_share", "run_snap_share", "rz_snap_share")
PBP_TEAM_COLUMNS = ["team_plays_pg_t3", "team_plays_pg_std", "team_pass_rate_t3", "team_pass_rate_std",
                    "team_neutral_pass_rate_t3", "team_neutral_pass_rate_std", "team_plays_pg_prev", "team_pass_rate_prev",
                    "opp_plays_faced_pg_std", "opp_pass_rate_faced_std"]
# Efficiency priors: 2015-2019 regular seasons, skill positions (nflverse stats_player_week), rounded to 3 places. Fixed
# constants known before the first backtest season, never refit on later data (a test recomputes them). A rate is
# (sum of numerator + k * prior) / (sum of denominator + k); k is the number of opportunities the prior is worth.
EFF_PRIORS = {
    "QB": {"catch_rate": 0.663, "rec_ypt": 7.330, "rec_td_rate": 0.054, "rush_ypc": 4.169, "rush_td_rate": 0.040,
           "pass_ypa": 7.196, "pass_td_rate": 0.045, "pass_int_rate": 0.023, "pass_cmp_rate": 0.633},
    "RB": {"catch_rate": 0.752, "rec_ypt": 6.080, "rec_td_rate": 0.030, "rush_ypc": 4.184, "rush_td_rate": 0.029,
           "pass_ypa": 7.196, "pass_td_rate": 0.045, "pass_int_rate": 0.023, "pass_cmp_rate": 0.633},
    "WR": {"catch_rate": 0.605, "rec_ypt": 7.839, "rec_td_rate": 0.048, "rush_ypc": 5.996, "rush_td_rate": 0.031,
           "pass_ypa": 7.196, "pass_td_rate": 0.045, "pass_int_rate": 0.023, "pass_cmp_rate": 0.633},
    "TE": {"catch_rate": 0.663, "rec_ypt": 7.330, "rec_td_rate": 0.054, "rush_ypc": 5.996, "rush_td_rate": 0.031,
           "pass_ypa": 7.196, "pass_td_rate": 0.045, "pass_int_rate": 0.023, "pass_cmp_rate": 0.633},
}
EFF_K = {"catch_rate": 20.0, "rec_ypt": 20.0, "rec_td_rate": 60.0, "rush_ypc": 30.0, "rush_td_rate": 80.0,
         "pass_ypa": 150.0, "pass_cmp_rate": 150.0, "pass_td_rate": 300.0, "pass_int_rate": 300.0}
EFF_WINDOWS = ("career", "std")
# `script` (game script and coaching context). Windows: the team's (or its opponent's) last SCRIPT_WINDOW regular-season games
# of this season and last season (so week 1 reads last season's final games). Only what `pbp_team` lacks is a column: its plays
# per game and dropback rate (t3/std) are reused internally for the expected-volume chain, not repeated.
SCRIPT_WINDOW = 4
SCRIPT_TEAM_COLUMNS = ["scr_sec_per_play_l4", "scr_no_huddle_rate_l4", "scr_proe_l4", "scr_xpass_l4",
                       "scr_early_down_pass_rate_l4", "scr_wp_neutral_pass_rate_l4", "scr_rz_plays_pg_l4",
                       "scr_epa_per_play_l4", "scr_success_rate_l4"]
SCRIPT_OPP_COLUMNS = ["scr_opp_epa_allowed_l4", "scr_opp_pass_epa_allowed_l4", "scr_opp_rush_epa_allowed_l4"]
SCRIPT_GAME_COLUMNS = ["scr_itt_x_pass_rate", "scr_spread_x_proe", "expected_team_plays", "expected_pass_att",
                       "expected_rush_att"]
SCRIPT_PLAYER_COLUMNS = ["exp_player_targets", "exp_player_carries", "exp_player_pass_att"]
SCRIPT_COACH_COLUMNS = ["hc_changed_this_season", "hc_games_in_charge"]
# expected_team_plays = a + b * trailing plays/game (l4) + c * |spread| + d * total. OLS on every 2021-2024 regular-season
# team-game with a trailing window and a line (`python -m model.script_rule` refits and prints them; fixed constants, never
# refit inside a backtest: 2025 is out of sample, 2024 is in sample like the tuned tree counts).
# Fit 2026-10-16 on 2,174 team-games: RMSE 8.25 plays against 8.30 for a constant and 9.11 for the trailing mean alone (team plays
# in one game are mostly noise; the rule shrinks the trailing mean ~86% toward the league). |spread| carries ~nothing; a signed
# spread would (+0.15 plays per point for the favourite), which `scr_spread_x_proe` and the vegas family already see.
SCRIPT_PLAYS_RULE = {"intercept": 47.6941, "trail_plays": 0.1366, "abs_spread": -0.0411, "total": 0.1361}

# `pepa` (Phase 5 candidate, 2026-10-16): per-player EPA and efficiency the matrix lacks. Per-game means over l1 / l3 / std of the
# counts, ratio-of-sums for the ratios, and per-opportunity rates over the last 3. Not added because already a feature: aDOT
# (pbp_adot_*), WOPR (pbp_wopr_*), air-yards share (pbp_ay_share_*), and every eff_* yards/TD/catch/completion rate.
PEPA_WINDOWS = ("l1", "l3", "std")
PEPA_COUNTS = ("pass_epa", "rush_epa", "rec_epa", "pass_air_yards", "rec_air_yards", "rec_yac", "rec_fd", "rush_fd", "pass_fd",
               "sacks")
PEPA_RATIOS = ("cpoe", "pacr", "racr")
PEPA_RATES = ("epa_per_opp", "pass_epa_per_db", "rec_epa_per_tgt", "rush_epa_per_car", "yac_per_rec")
# `nfelo` (Phase 5 candidate, 2026-10-16): greerreNFL team ratings, QB values, WEPA, market line movement, film margin
# (model/nfelo.py has the sources and the known_at rules). All team-game level.
NFELO_COLUMNS = ["nf_tm_pass_off", "nf_tm_rush_off", "nf_opp_pass_def", "nf_opp_rush_def", "nf_tm_total", "nf_opp_total",
                 "nf_pass_edge", "nf_rush_edge", "nf_elo", "nf_opp_elo", "nf_elo_dif", "nf_qb_adj", "nf_opp_qb_adj",
                 "nf_qb_value", "nf_opp_qb_value", "nf_wepa_net", "nf_opp_wepa_net", "nf_spread_move", "nf_total_move",
                 "nf_tickets_pct", "nf_film_margin_l4"]
FILM_WINDOW = 4
# Weekly charting releases (repeated from point_in_time.NGS_STATS / PFR_COLUMNS / FTN_COUNTS; a test pins them equal).
NGS_STATS = ("avg_separation", "avg_cushion", "avg_intended_air_yards", "percent_share_of_intended_air_yards", "catch_percentage",
             "avg_yac_above_expectation", "efficiency", "rush_yards_over_expected_per_att", "percent_attempts_gte_eight_defenders",
             "avg_time_to_throw", "aggressiveness", "completion_percentage_above_expectation", "avg_air_yards_differential")
PFR_RAW = ("drops", "drop_pct", "rec_brk_tkl", "rat", "rush_att", "rush_ybc", "rush_yac", "brk_tkl", "pressure_pct", "blitz",
           "hurry", "hits", "sacks", "pass_drop_pct", "bad_throw_pct")
# Per-carry rushing rates are ratios of the window's sums (`ybc_att` = yards before contact / carries, `att_br` = carries per
# broken tackle, NaN with none); every other pfr stat is the table's own per-game value, averaged.
PFR_RATIOS = {"ybc_att": ("rush_ybc", "rush_att"), "yac_att": ("rush_yac", "rush_att"), "att_br": ("rush_att", "brk_tkl")}
PFR_STATS = ("drops", "drop_pct", "rec_brk_tkl", "rat", "ybc_att", "yac_att", "brk_tkl", "att_br", "pressure_pct", "blitz",
             "hurry", "hits", "sacks", "pass_drop_pct", "bad_throw_pct")
FTN_C = ("tgt", "tgt_motion", "tgt_pa", "tgt_rpo", "tgt_screen", "tgt_catchable", "tgt_contested", "drops",
         "car", "car_box", "car_light", "box_sum", "db", "db_blitz", "db_oop", "db_pa", "db_rush_n", "rushers_sum", "int_worthy")
FTN_T = ("T_tgt", "T_tgt_motion", "T_tgt_pa", "T_tgt_rpo", "T_tgt_screen")
# name: (numerator, denominator) over the window's sums; a denominator from the team table is prefixed "T_"; None = per game.
FTN_SPEC = {"tgt_share_motion": ("tgt_motion", "T_tgt_motion"), "tgt_share_pa": ("tgt_pa", "T_tgt_pa"),
            "tgt_share_rpo": ("tgt_rpo", "T_tgt_rpo"), "tgt_share_screen": ("tgt_screen", "T_tgt_screen"),
            "catchable_rate": ("tgt_catchable", "tgt"), "contested_rate": ("tgt_contested", "tgt"), "drops_pg": ("drops", None),
            "light_box_car_share": ("car_light", "car_box"), "avg_box_car": ("box_sum", "car_box"),
            "qb_blitz_rate": ("db_blitz", "db"), "qb_oop_rate": ("db_oop", "db"), "qb_pa_rate": ("db_pa", "db"),
            "qb_avg_rushers": ("rushers_sum", "db_rush_n"), "qb_int_worthy_rate": ("int_worthy", "db")}
FTN_STATS = tuple(FTN_SPEC)
# ---- `dvp_split`: the opponent's defense by stat (2026-10-27)
DVPS_WINDOW = 4
DVPS_WINDOWS = ("l4", "std")
# stem -> player_games column, summed per (game, position) over the rows whose opponent_team is the defense; 0 when the
# position had no row in a game the defense played (as in `dvp`)
DVPS_ALLOWED = {"rec_yd": "receiving_yards", "rec_tgt": "targets", "rec_td": "receiving_tds", "rush_yd": "rushing_yards",
                "rush_att": "carries", "rush_td": "rushing_tds", "pass_yd": "passing_yards", "pass_td": "passing_tds",
                "sacks": "sacks_suffered"}
DVPS_PBP = ("epa_pass", "epa_rush", "pass_rate_faced", "explosive_rate")        # pbp_team rows with the defense as `opponent`
DVPS_FTN = ("light_box_rate", "blitz_rate", "avg_rushers")                     # ftn_def rows, l4 only
FTN_D = ("car_box", "car_light", "box_sum", "db", "db_blitz", "db_rush_n", "rushers_sum")   # ftn_def columns (pinned by a test)
DVPS_RANKS = {"rank_rec_yd": "receiving_yards", "rank_rush_yd": "rushing_yards"}
DVPS_COLUMNS = ([f"dvps_{s}_{w}" for w in DVPS_WINDOWS for s in (*DVPS_ALLOWED, *DVPS_PBP)]
                + [f"dvps_{s}_l4" for s in DVPS_FTN] + [f"dvps_{s}_std" for s in DVPS_RANKS])
DVPS_STEM = {c: c[len("dvps_"):].rsplit("_", 1)[0] for c in DVPS_COLUMNS}
# the stems a position's row carries; every other dvp_split column is NaN on it (the trees already get `position`)
DVPS_POS: dict[str, frozenset] = {
    "QB": frozenset({"pass_yd", "pass_td", "sacks", "rush_yd", "rush_att", "rush_td", "epa_pass", "epa_rush", "pass_rate_faced",
                     "explosive_rate", "blitz_rate", "avg_rushers", "rank_rush_yd"}),
    "RB": frozenset({"rec_yd", "rec_tgt", "rec_td", "rush_yd", "rush_att", "rush_td", "epa_pass", "epa_rush", "pass_rate_faced",
                     "explosive_rate", "light_box_rate", "rank_rec_yd", "rank_rush_yd"}),
    "WR": frozenset({"rec_yd", "rec_tgt", "rec_td", "epa_pass", "pass_rate_faced", "explosive_rate", "blitz_rate", "avg_rushers",
                     "rank_rec_yd"}),
}
DVPS_POS["TE"] = DVPS_POS["WR"]
# `opp_db_out` (Phase 5 candidate, 2026-10-27; miss review 2026 wk7: McMillan wk4 vs DET missing both starting safeties, and
# the opponent's defensive injuries were loaded then discarded). The opponent's injury report for THIS week, known before kickoff,
# by defensive position group (INJ_DEF_GROUPS maps the report's roster position), joined to the opponent's newest defensive depth
# chart before kickoff (`depth_charts_def`, no older than DEPTH_MAX_AGE) for who is a starter (rank 1 at his slot).
#   opp_db_out_n / opp_lb_out_n / opp_dl_out_n   Out or Doubtful (OUT_LIKE) DBs / LBs / DL on the report
#   opp_db_q_n                                   Questionable DBs
#   opp_db_starter_out_n                         of the Out/Doubtful DBs, how many are rank 1 at a DB slot on that chart
#   opp_def_out_n                                every defensive Out/Doubtful
#   opp_def_out_l4wk                             defensive 'Out' rows the opponent carried in weeks N-4..N-1 of this season
# A missing report and a clean report differ: every count is NaN when the opponent has no report rows for the week at all (the
# meta column `opp_report_present` says which), and opp_def_out_l4wk is NaN when it has none in the window (week 1).
# opp_db_starter_out_n is NaN without a fresh chart. IR is not a report designation (an IR player is off the report), so
# "IR-equivalent" is Out/Doubtful only.
OPP_DEF_COLUMNS = ["opp_db_out_n", "opp_db_q_n", "opp_db_starter_out_n", "opp_lb_out_n", "opp_dl_out_n", "opp_def_out_n",
                   "opp_def_out_l4wk"]
INJ_DEF_GROUPS = {"CB": "DB", "S": "DB", "SS": "DB", "FS": "DB", "DB": "DB", "SAF": "DB",
                  "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
                  "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "EDGE": "DL"}
OPP_DEF_WINDOW = 4

FAMILIES: dict[str, list[str]] = {
    "lags": _lag_cols(LAG_STATS) + [f"lag{k}_weeks_ago" for k in (1, 2, 3)] + ["games_std", "weeks_since_last_game"]
            + [f"{n}_std_mean" for n in STD_STATS],
    "prev_season": ["prev_season_games"] + [f"prev_season_{n}" for n in PREV_STATS],
    "td_luck": ["td_luck_rush_std", "td_luck_rec_std", "td_luck_pass_std", "td_luck_total_std", "td_luck_total_prev"],
    "role": ["role_score", "role_rank_pos", "tm_group_size", "tm_out_group_n", "tm_out_above_n", "tm_q_group_n",
             "tm_out_group_opps", "tm_out_targets_all", "tm_out_carries_all"],
    "injury": ["inj_report_status", "inj_practice_status", "inj_weeks_since_report", "inj_listed_l4wk"],
    "dvp": [f"dvp_ppr_l{n}" for n in (2, 4)] + ["dvp_ppr_std"],
    "vegas": ["team_spread", "total_line", "implied_team_total", "implied_opp_total"],
    "context": ["is_home", "rest_days", "opp_rest_days", "div_game", "neutral_site", "week", "season"],
    "weather": ["wx_temp_obs", "wx_wind_obs", "wx_roof_obs", "wx_indoor_obs"],
    "static": ["age_years", "exp_years", "is_rookie", "career_games_prior", "draft_round", "draft_overall",
               "is_undrafted", "height_in", "weight_lb", "combine_forty", "combine_bench", "combine_vertical",
               "combine_broad_jump", "combine_cone", "combine_shuttle"],
    "adp": ["adp_ppr", "adp_std", "adp_ppr_pos_rank", "adp_std_pos_rank"],
    "college": ["college_rec_market_share", "college_rec_td_share", "college_rec_pg", "college_ypr",
                "college_rush_share", "college_car_pg", "college_ypc", "college_dominator",
                "college_pass_att_pg", "college_pass_ypa", "college_pass_td_rate", "college_power_conf",
                "college_breakout_age"],
    # ---- Phase 2.5 (opt-in for the models)
    "pbp_usage": [f"pbp_{n}_{w}" for w in PBP_WINDOWS for n in PBP_USAGE_STATS] + [f"pbp_prev_{n}" for n in PBP_PREV_STATS],
    "pbp_team": PBP_TEAM_COLUMNS,
    "pbp_part": [f"part_{n}_{w}" for w in PBP_WINDOWS for n in PART_STATS] + ["part_prev_pass_snap_share"],
    "eff": [f"eff_{n}_{w}" for w in EFF_WINDOWS for n in EFF_K],
    # ---- Phase 5 candidate (opt-in; `python -m model.trial --candidate +family:streak`)
    "streak": ["resid_xfp_l1", "resid_xfp_l2", "resid_xfp_l3", "over_xfp_streak", "resid_xfp_sum_l4",
               "over_trail_streak", "resid_opps_l1", "pts_vs_prev_season_l2"],
    # ---- Phase 5 candidate (opt-in; `python -m model.trial --candidate +family:shape`; 2026-10-27, from the miss review)
    "shape": ["shape_tgt_share_std_l6", "shape_car_share_std_l6", "shape_pts_cv_l8", "shape_td_per_opp_std",
              "shape_td_pts_share_l8", "shape_boom_rate_l8", "shape_opps_trend_l4"],
    # ---- Phase 5 candidate (opt-in; `python -m model.trial --candidate +family:ecr`)
    "ecr": ["ecr_rank", "ecr_pos_rank", "ecr_sd", "ecr_best", "ecr_worst", "ecr_spread", "ecr_owned_pct", "ecr_delta_l1"],
    # ---- Phase 5 candidates (2026-10-16; opt-in): `+family:nfelo`, `+family:pepa`
    "nfelo": NFELO_COLUMNS,
    "pepa": [f"pepa_{n}_{w}" for w in PEPA_WINDOWS for n in PEPA_COUNTS + PEPA_RATIOS] + [f"pepa_{n}_l3" for n in PEPA_RATES],
    # ---- Phase 5 candidate (opt-in; `python -m model.trial --candidate +family:market`)
    "market": ["mkt_pts_ppr", "mkt_pts_std", "mkt_targets", "mkt_carries", "mkt_pass_att", "mkt_rec_yd", "mkt_rush_yd",
               "mkt_pass_yd", "mkt_td_total", "mkt_pos_rank", "mkt_vs_xfp_l3"],
    # ---- Phase 5 candidate (opt-in; `python -m model.trial --candidate +family:script`)
    "script": SCRIPT_TEAM_COLUMNS + SCRIPT_OPP_COLUMNS + SCRIPT_GAME_COLUMNS + SCRIPT_PLAYER_COLUMNS + SCRIPT_COACH_COLUMNS,
    # ---- Phase 5 candidates, weekly charting releases (opt-in; `--candidate +family:ngs` / `pfr` / `ftn`)
    "ngs": [f"ngs_{s}_{w}" for s in NGS_STATS for w in ("l1", "l3")],
    "pfr": [f"pfr_{s}_{w}" for s in PFR_STATS for w in ("l1", "l3")],
    "ftn": [f"ftn_{s}_l3" for s in FTN_STATS],
    # ---- Phase 5 candidate (2026-10-27; opt-in): `python -m model.trial --candidate +family:dvp_split`
    "dvp_split": DVPS_COLUMNS,
    # ---- Phase 5 candidate (2026-10-27; opt-in; `--candidate +family:opp_db_out`): the OPPONENT's defensive injury report
    "opp_db_out": OPP_DEF_COLUMNS,
}
# market column -> sleeper_proj column (a stat Sleeper omits is a projected 0 when the row exists)
MARKET_STATS = {"mkt_pts_ppr": "pts_ppr", "mkt_pts_std": "pts_std", "mkt_targets": "rec_tgt", "mkt_carries": "rush_att",
                "mkt_pass_att": "pass_att", "mkt_rec_yd": "rec_yd", "mkt_rush_yd": "rush_yd", "mkt_pass_yd": "pass_yd",
                "mkt_pos_rank": "pos_rank"}
STREAK_CAP = 6                         # a run of over/under games is reported as min(run, 6), signed
# `shape` (2026-10-27): TDs per opportunity, shrunk toward a position prior worth SHAPE_TD_K opportunities. The priors are the
# 2015-2019 EFF_PRIORS rates: a QB's pass TDs per attempt; for the others rush + rec TDs per carry + target, the rec rate for
# receivers and the midpoint of the rush and rec rates for a back (both 0.03). Fixed constants, never refit.
SHAPE_TD_PRIOR = {"QB": 0.045, "RB": 0.030, "WR": 0.048, "TE": 0.054}
SHAPE_TD_K = {"QB": 150.0, "RB": 60.0, "WR": 60.0, "TE": 60.0}
SHAPE_BOOM = 1.5                       # a boom game scores at least this multiple of the window's mean
SHAPE_TD_PTS = {"rush": 6.0, "rec": 6.0, "pass": 4.0}
# Produced, but never a model input.
IDENTITY = ["player_id", "season", "week", "team", "opponent", "position", "game_id", "kickoff_utc"]
META = ["lag1_week", "lag2_week", "lag3_week", "dvp_l2_games", "dvp_l4_games", "dvp_l2_last_week",
        "dvp_l4_last_week", "inj_last_report_week", "spine_src",
        "opp_report_present"]          # `opp_db_out`: 1 if the opponent filed any report row this week (its counts are NaN when 0)
EXCLUDED_FROM_MATRIX = {
    "inj_days_since_report": "real report timestamp <=2024, derived kickoff-24h from 2025 (always ~1 day): train/test skew",
}


def feature_columns() -> list[str]:
    seen: list[str] = []
    for cols in FAMILIES.values():
        for c in cols:
            if c not in seen:
                seen.append(c)
    return seen


def family_of(col: str) -> str | None:
    for fam, cols in FAMILIES.items():
        if col in cols:
            return fam
    return None


# --------------------------------------------------------------------------- small helpers
def _f(x: Any) -> float:
    try:
        return NAN if x is None or pd.isna(x) else float(x)
    except (TypeError, ValueError):
        return NAN


def _s(x: Any) -> Any:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) or x is pd.NA else x


def _nm(a: np.ndarray) -> float:
    """NaN-ignoring mean; NaN for an empty or all-NaN slice (no warnings)."""
    a = a[~np.isnan(a)]
    return float(a.mean()) if a.size else NAN


def _ns(a: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    return float(a.sum()) if a.size else NAN


class _Row:
    """Lazy view of the newest-known row of a gate result (truthy iff there is one). Column access only,
    because materialising every column of a wide row (to_dict / iloc) dominated the build time."""
    __slots__ = ("_df",)

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def __getitem__(self, col: str):
        return self._df[col].iloc[-1]


def _last(df: pd.DataFrame) -> "_Row | None":
    return _Row(df) if len(df) else None


def _cols(df: pd.DataFrame, *names: str):
    """Iterate selected columns row-wise without building namedtuples (itertuples is slow on wide frames)."""
    return zip(*(df[n].tolist() for n in names))


def _isnan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _rk(ts: pd.Timestamp) -> pd.Timestamp:
    """The gate cutoff for result tables (see RESULT_LAG)."""
    return ts - RESULT_LAG


def _memo(store, trace, key, fn):
    """Cache per (store, key) unless a trace was requested (a trace must record every gate call)."""
    return fn() if trace is not None else store.memo(key, fn)


# --------------------------------------------------------------------------- per-player history
# value-matrix columns of Hist.v (12 volume/points columns, then the Phase 2.5 component columns)
PTS, OPPS, CAR, TGT, ATT, TSH, CSH, SNAP, XFP, TDR, TDC, TDP = range(12)
REC, RECY, RECTD, RUSHY, RUSHTD, PASSY, PASSTD, INT, CMP = range(12, 21)
# `pepa` inputs (nflverse weekly stats; NaN EPA on a stats row means no play of that kind, stored as 0; cpoe stays NaN without
# attempts). The stats column feeding each, in order, is PEPA_SRC.
EPA_P, EPA_R, EPA_C, CPOE, PAY, RAY, RYAC, RFD, UFD, PFD, SACK = range(21, 32)
N_VALUES = 32
PEPA_SRC = ("passing_epa", "rushing_epa", "receiving_epa", "passing_cpoe", "passing_air_yards", "receiving_air_yards",
            "receiving_yards_after_catch", "receiving_first_downs", "rushing_first_downs", "passing_first_downs", "sacks_suffered")


class Hist:
    """Every game the player appeared in with known_at < kickoff, all seasons in the store, oldest first."""
    __slots__ = ("season", "week", "is_reg", "has_stats", "game_id", "team", "v")

    def __init__(self, rows: list):
        self.game_id = [r[0] for r in rows]
        self.season = np.array([r[1] for r in rows], dtype="int64")
        self.week = np.array([r[2] for r in rows], dtype="int64")
        self.is_reg = np.array([r[3] for r in rows], dtype=bool)
        self.has_stats = np.array([r[4] for r in rows], dtype=bool)
        self.team = [r[5] for r in rows]
        self.v = np.array([r[6:] for r in rows], dtype="float64").reshape(len(rows), N_VALUES)

    def __len__(self) -> int:
        return len(self.game_id)


def _history(store, player_id: str, t: TargetRow, trace) -> Hist:
    """An appearance is a stats row OR offense_snaps > 0. Snap-only appearances carry 0 for every volume
    stat (the player recorded none). xFP: the ffopportunity row if there is one; otherwise 0 when the
    stats row shows no pass attempt / carry / target (that is exactly who ffopportunity omits), else NaN."""
    def build() -> Hist:
        rk = _rk(t.kickoff)
        pg = as_of_join(store, "player_games", rk, player_id=player_id, trace=trace)
        sn = as_of_join(store, "snap_counts", rk, player_id=player_id, trace=trace)
        xf = as_of_join(store, "xfp", rk, player_id=player_id, trace=trace)
        snap = dict(zip(sn["game_id"], sn["offense_pct"]))
        xcol = {c: dict(zip(xf["game_id"], xf[c])) for c in
                ("total_fantasy_points_exp", "rush_touchdown", "rush_touchdown_exp", "rec_touchdown",
                 "rec_touchdown_exp", "pass_touchdown", "pass_touchdown_exp")}
        xfp_by = xcol["total_fantasy_points_exp"]
        rows = []
        # pepa inputs: absent from a table that predates them (synthetic test stores) -> NaN
        pep = [pg[c].tolist() if c in pg.columns else [NAN] * len(pg) for c in PEPA_SRC]
        pep = list(zip(*pep)) if len(pg) else []
        for i, (gid, season, week, stype, team, ppr, car, tgt, att_, tsh, csh, rec, recy, rectd, rushy, rushtd, passy,
                passtd, ints, cmp_) in enumerate(_cols(
                pg, "game_id", "season", "week", "season_type", "team", "fantasy_points_ppr", "carries", "targets",
                "attempts", "target_share", "carry_share", "receptions", "receiving_yards", "receiving_tds",
                "rushing_yards", "rushing_tds", "passing_yards", "passing_tds", "passing_interceptions", "completions")):
            carries, targets, att = _f(car), _f(tgt), _f(att_)
            opps = (0.0 if math.isnan(carries) else carries) + (0.0 if math.isnan(targets) else targets)
            no_opp = opps + (0.0 if math.isnan(att) else att) == 0
            has_x = gid in xfp_by
            if has_x:
                tdr = _f(xcol["rush_touchdown"].get(gid)) - _f(xcol["rush_touchdown_exp"].get(gid))
                tdc = _f(xcol["rec_touchdown"].get(gid)) - _f(xcol["rec_touchdown_exp"].get(gid))
                tdp = _f(xcol["pass_touchdown"].get(gid)) - _f(xcol["pass_touchdown_exp"].get(gid))
                xfp = _f(xfp_by[gid])
            else:
                tdr = tdc = tdp = xfp = 0.0 if no_opp else NAN
            pv = [_f(x) for x in pep[i]]
            for j in (0, 1, 2):                              # EPA of plays he did not have is 0, not unknown
                if math.isnan(pv[j]) and "passing_epa" in pg.columns:
                    pv[j] = 0.0
            rows.append((gid, int(season), int(week), stype == "REG", True, team, _f(ppr), opps, carries, targets, att,
                         _f(tsh), _f(csh), _f(snap.get(gid)), xfp, tdr, tdc, tdp,
                         _f(rec), _f(recy), _f(rectd), _f(rushy), _f(rushtd), _f(passy), _f(passtd), _f(ints), _f(cmp_), *pv))
        seen = set(pg["game_id"])
        only = sn[(sn["offense_snaps"] > 0) & sn["position"].isin(POSITIONS) & ~sn["game_id"].isin(seen)]
        for gid, season, week, gtype, pct, team in _cols(only, "game_id", "season", "week", "game_type", "offense_pct", "team"):
            xfp = _f(xfp_by[gid]) if gid in xfp_by else 0.0
            rows.append((gid, int(season), int(week), gtype == "REG", False, team, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, _f(pct), xfp, 0.0, 0.0, 0.0, *([0.0] * 9), 0.0, 0.0, 0.0, NAN, *([0.0] * 7)))
        rows.sort(key=lambda x: x[0])                       # placeholder order; real order is kickoff (below)
        kick = {**dict(zip(pg["game_id"], pg[KNOWN_AT])), **dict(zip(only["game_id"], only[KNOWN_AT]))}
        rows.sort(key=lambda x: kick[x[0]])                 # stable: ties keep game_id order
        return Hist(rows)
    return _memo(store, trace, ("hist", player_id, t.game_id), build)


def _cur_prev(h: Hist, season: int) -> tuple[np.ndarray, np.ndarray]:
    """Value matrices of this season's appearances and last season's REG appearances."""
    return h.v[h.season == season], h.v[(h.season == season - 1) & h.is_reg]


def _trailing(cur: np.ndarray, prev: np.ndarray, col: int, n: int = 3) -> float:
    """Mean of the last n current-season appearances; last season's mean when there is none."""
    if len(cur):
        return _nm(cur[-n:, col])
    return _nm(prev[:, col]) if len(prev) else NAN


LAG_COLS = {"pts_ppr": PTS, "opps": OPPS, "carries": CAR, "targets": TGT, "pass_att": ATT,
            "target_share": TSH, "carry_share": CSH, "snap_pct": SNAP, "xfp": XFP}
STD_COLS = {n: LAG_COLS[n] for n in STD_STATS}
PREV_COLS = {"ppg": PTS, "opps_pg": OPPS, "carries_pg": CAR, "targets_pg": TGT, "target_share": TSH,
             "snap_pct": SNAP, "xfp_pg": XFP}


def _lag_and_history_features(h: Hist, t: TargetRow) -> dict[str, Any]:
    m = h.season == t.season
    cur, prev = h.v[m], h.v[(h.season == t.season - 1) & h.is_reg]
    weeks = h.week[m]
    out: dict[str, Any] = {}
    for k in (1, 2, 3):
        have = len(cur) >= k
        out[f"lag{k}_week"] = int(weeks[-k]) if have else None
        out[f"lag{k}_weeks_ago"] = float(t.week - weeks[-k]) if have else NAN
        for feat, col in LAG_COLS.items():
            out[f"{feat}_l{k}"] = float(cur[-k, col]) if have else NAN
    out["weeks_since_last_game"] = int(t.week - weeks[-1]) if len(cur) else None
    out["games_std"] = int(len(cur))
    for feat, col in STD_COLS.items():
        out[f"{feat}_std_mean"] = _nm(cur[:, col]) if len(cur) else NAN
    out["prev_season_games"] = int(len(prev))
    for feat, col in PREV_COLS.items():
        out[f"prev_season_{feat}"] = _nm(prev[:, col]) if len(prev) else NAN
    have = len(cur) > 0
    out["td_luck_rush_std"] = _ns(cur[:, TDR]) if have else NAN
    out["td_luck_rec_std"] = _ns(cur[:, TDC]) if have else NAN
    out["td_luck_pass_std"] = _ns(cur[:, TDP]) if have else NAN
    out["td_luck_total_std"] = _ns(np.nansum(cur[:, [TDR, TDC, TDP]], axis=1)[~np.isnan(cur[:, [TDR, TDC, TDP]]).all(axis=1)]) if have else NAN
    out["td_luck_total_prev"] = (_ns(np.nansum(prev[:, [TDR, TDC, TDP]], axis=1)[~np.isnan(prev[:, [TDR, TDC, TDP]]).all(axis=1)])
                                 if len(prev) else NAN)
    return out


def _run_length(x: np.ndarray, cap: int = STREAK_CAP) -> float:
    """Signed length of the run of same-sign values at the END of `x`: +n for n consecutive positives, -n for negatives,
    0 when there is no history or the newest value is NaN or exactly 0. A NaN or a 0 inside the run ends it. Capped at +-cap."""
    if not len(x) or math.isnan(x[-1]) or x[-1] == 0:
        return 0.0
    sign = 1.0 if x[-1] > 0 else -1.0
    n = 0
    for v in x[::-1]:
        if math.isnan(v) or v == 0 or (v > 0) != (sign > 0):
            break
        n += 1
        if n == cap:
            break
    return sign * n


def _streak_features(h: Hist, t: TargetRow) -> dict[str, float]:
    """The `streak` family. The window is last season's REG appearances followed by this season's (both strictly before
    kickoff, `_cur_prev`), so a week-1 row sees last season's final games and a week-3 row's four-game sum reaches back.
    Residuals: actual PPR minus the game's xFP (NaN when the game has no xFP), and opps minus the trailing-3 mean of opps
    entering that game (NaN for the first game in the window)."""
    cur, prev = _cur_prev(h, t.season)
    seq = np.vstack([prev, cur]) if len(prev) and len(cur) else (cur if len(cur) else prev)
    n = len(seq)
    out: dict[str, float] = {}
    resid = seq[:, PTS] - seq[:, XFP] if n else np.empty(0)
    for k in (1, 2, 3):
        out[f"resid_xfp_l{k}"] = float(resid[-k]) if n >= k else NAN
    out["over_xfp_streak"] = _run_length(resid)
    out["resid_xfp_sum_l4"] = _ns(resid[-4:]) if n else NAN
    opps = seq[:, OPPS] if n else np.empty(0)
    vol = np.full(n, NAN)
    for i in range(1, n):
        vol[i] = opps[i] - _nm(opps[max(0, i - 3):i])
    out["over_trail_streak"] = _run_length(vol)
    out["resid_opps_l1"] = float(vol[-1]) if n else NAN
    out["pts_vs_prev_season_l2"] = _nm(seq[-2:, PTS]) - _nm(prev[:, PTS]) if n and len(prev) else NAN
    return out


def _nstd(a: np.ndarray) -> float:
    """NaN-ignoring population standard deviation; NaN with fewer than two values."""
    a = a[~np.isnan(a)]
    return float(a.std()) if a.size >= 2 else NAN


def _slope(a: np.ndarray) -> float:
    """OLS slope of the values against their order (0, 1, ...), NaN entries dropped; NaN with fewer than two values."""
    x = np.arange(len(a), dtype="float64")
    ok = ~np.isnan(a)
    if ok.sum() < 2:
        return NAN
    xs, ys = x[ok], a[ok]
    dx = xs - xs.mean()
    return float((dx * (ys - ys.mean())).sum() / (dx * dx).sum())


def _shape_features(h: Hist, t: TargetRow) -> dict[str, float]:
    """The `shape` family (player archetype). Windows: the last 6 / 8 / 4 games of last season's REG appearances followed by this
    season's (`_cur_prev`, both strictly before kickoff); `_std` = this season's appearances only, like the lag `_std` columns.
      shape_tgt_share_std_l6 / shape_car_share_std_l6   std of his weekly target / carry share over the last 6
      shape_pts_cv_l8                                   std / mean of PPR over the last 8 (NaN when the mean is not positive)
      shape_td_per_opp_std                              (TDs + k * prior) / (opps + k): pass TDs per attempt for a QB (k 150), rush +
                                                        rec TDs per carry + target otherwise (k 60); NaN with no opportunity this season
      shape_td_pts_share_l8                             6 * (rush + rec TDs) + 4 * pass TDs over PPR points, sums over the last 8
      shape_boom_rate_l8                                share of the last 8 at or above 1.5x the window's mean PPR
      shape_opps_trend_l4                               OLS slope of opportunities over the last 4 (per game; a QB's opportunities
                                                        add his pass attempts, as the retrospective counts volume)
    NaN where the window is empty or a ratio has no denominator; never a model input unless `+family:shape` is adopted."""
    cur, prev = _cur_prev(h, t.season)
    seq = np.vstack([prev, cur]) if len(prev) and len(cur) else (cur if len(cur) else prev)
    n = len(seq)
    out: dict[str, float] = {c: NAN for c in FAMILIES["shape"]}
    if n:
        w6, w8, w4 = seq[-6:], seq[-8:], seq[-4:]
        out["shape_tgt_share_std_l6"] = _nstd(w6[:, TSH])
        out["shape_car_share_std_l6"] = _nstd(w6[:, CSH])
        mean8 = _nm(w8[:, PTS])
        if not math.isnan(mean8) and mean8 > 0:
            out["shape_pts_cv_l8"] = _nstd(w8[:, PTS]) / mean8
            pts8 = w8[:, PTS][~np.isnan(w8[:, PTS])]
            out["shape_boom_rate_l8"] = float((pts8 >= SHAPE_BOOM * mean8).mean())
        td_pts = (SHAPE_TD_PTS["rush"] * np.nansum(w8[:, RUSHTD]) + SHAPE_TD_PTS["rec"] * np.nansum(w8[:, RECTD])
                  + SHAPE_TD_PTS["pass"] * np.nansum(w8[:, PASSTD]))
        pts_sum = float(np.nansum(w8[:, PTS]))
        out["shape_td_pts_share_l8"] = float(td_pts / pts_sum) if pts_sum > 0 else NAN
        out["shape_opps_trend_l4"] = _slope(w4[:, ATT] + w4[:, OPPS] if t.position == "QB" else w4[:, OPPS])
    if len(cur):
        if t.position == "QB":
            td, opp = float(np.nansum(cur[:, PASSTD])), float(np.nansum(cur[:, ATT]))
        else:
            td, opp = float(np.nansum(cur[:, RUSHTD]) + np.nansum(cur[:, RECTD])), float(np.nansum(cur[:, OPPS]))
        k = SHAPE_TD_K.get(t.position, 60.0)
        prior = SHAPE_TD_PRIOR.get(t.position, 0.045)
        out["shape_td_per_opp_std"] = (td + k * prior) / (opp + k) if opp > 0 else NAN
    return out


def _pepa_ratios(v: np.ndarray) -> dict[str, float]:
    """cpoe: attempt-weighted mean over games that have one; pacr / racr: yards over air yards, ratio of sums (NaN with no air yards)."""
    out = {}
    w = v[:, ATT]
    ok = ~np.isnan(v[:, CPOE]) & (w > 0)
    out["cpoe"] = float((v[ok, CPOE] * w[ok]).sum() / w[ok].sum()) if ok.any() else NAN
    out["pacr"] = _ratio(float(np.nansum(v[:, PASSY])), float(np.nansum(v[:, PAY])))
    out["racr"] = _ratio(float(np.nansum(v[:, RECY])), float(np.nansum(v[:, RAY])))
    return out


def _pepa_features(h: Hist, t: TargetRow) -> dict[str, float]:
    """The `pepa` family over this season's appearances strictly before the row's result cutoff (the `Hist` rows): l1 = his last
    game, l3 = his last three, std = all. NaN with no appearance this season (week 1), like the lags."""
    cur = h.v[h.season == t.season]
    counts = {"pass_epa": EPA_P, "rush_epa": EPA_R, "rec_epa": EPA_C, "pass_air_yards": PAY, "rec_air_yards": RAY,
              "rec_yac": RYAC, "rec_fd": RFD, "rush_fd": UFD, "pass_fd": PFD, "sacks": SACK}
    out: dict[str, float] = {}
    for w, v in (("l1", cur[-1:]), ("l3", cur[-3:]), ("std", cur)):
        for n, col in counts.items():
            out[f"pepa_{n}_{w}"] = _nm(v[:, col]) if len(v) else NAN
        r = _pepa_ratios(v) if len(v) else {n: NAN for n in PEPA_RATIOS}
        for n in PEPA_RATIOS:
            out[f"pepa_{n}_{w}"] = r[n]
    v = cur[-3:]
    n = (lambda c: _ns(v[:, c])) if len(v) else (lambda c: NAN)                    # noqa: E731  numerators: NaN if unknown
    s = (lambda c: float(np.nansum(v[:, c]))) if len(v) else (lambda c: 0.0)       # noqa: E731  denominators
    out["pepa_epa_per_opp_l3"] = _ratio(n(EPA_P) + n(EPA_R) + n(EPA_C), s(ATT) + s(SACK) + s(CAR) + s(TGT))
    out["pepa_pass_epa_per_db_l3"] = _ratio(n(EPA_P), s(ATT) + s(SACK))
    out["pepa_rec_epa_per_tgt_l3"] = _ratio(n(EPA_C), s(TGT))
    out["pepa_rush_epa_per_car_l3"] = _ratio(n(EPA_R), s(CAR))
    out["pepa_yac_per_rec_l3"] = _ratio(n(RYAC), s(REC))
    return out


# --------------------------------------------------------------------------- the spine
def _add(cand: dict, pid: str, pos: str | None, src: str, stamp: pd.Timestamp, rank: int) -> None:
    c = cand.setdefault(pid, {"src": set(), "pos": None, "stamp": None, "prio": 99})
    c["src"].add(src)
    # position comes from the freshest signal (ties: usage > injury > depth > draft)
    if pos and (c["pos"] is None or (stamp, -rank) > (c["stamp"], -c["prio"])):
        c["pos"], c["stamp"], c["prio"] = pos, stamp, rank


def _spine(store, tg: TargetRow, trace) -> tuple[list[TargetRow], list[str]]:
    def build():
        ko, rk = tg.kickoff, _rk(tg.kickoff)
        cand: dict[str, dict] = {}
        # (1) recent usage: appeared for this team in one of its last USAGE_WINDOW games (crosses seasons)
        gr = as_of_join(store, "game_results", rk, team=tg.team, trace=trace)
        last = gr.tail(USAGE_WINDOW)
        last_gids = list(last["game_id"])
        for gid, stamp in zip(last["game_id"], last[KNOWN_AT]):
            pgg = as_of_join(store, "player_games", rk, game_id=gid, trace=trace)
            for pid, pos in _cols(pgg[pgg["team"] == tg.team], "player_id", "position"):
                _add(cand, pid, pos, "usage", stamp, 0)
            snn = as_of_join(store, "snap_counts", rk, game_id=gid, trace=trace)
            snn = snn[(snn["team"] == tg.team) & (snn["offense_snaps"] > 0) & snn["position"].isin(POSITIONS)]
            for pid, pos in _cols(snn, "player_id", "position"):
                _add(cand, pid, pos, "usage", stamp, 0)
        # (2) this week's injury report (any designation, incl. Out: they are roster members with no stats row)
        inj = as_of_join(store, "injuries", ko, team=tg.team, trace=trace)
        inj = inj[(inj["season"] == tg.season) & (inj["week"] == tg.week) & inj["position"].isin(POSITIONS)]
        for pid, pos, at in _cols(inj, "player_id", "position", KNOWN_AT):
            _add(cand, pid, pos, "injury", at, 1)
        # (3) the newest depth chart known before kickoff, if it is recent enough to mean anything
        dc = as_of_join(store, "depth_charts", ko, team=tg.team, trace=trace)
        if len(dc):
            newest = dc[KNOWN_AT].max()
            if ko - newest <= DEPTH_MAX_AGE:
                for pid, pos in _cols(dc[dc[KNOWN_AT] == newest], "player_id", "pos"):
                    _add(cand, pid, pos, "depth", newest, 2)
        # (4) the draft class, for the opener only (rookies have no usage, injury row or 2024- chart yet)
        if tg.week == 1:
            dp = as_of_join(store, "draft_picks", ko, season=tg.season, team=tg.team, trace=trace)
            for pid, pos, at in _cols(dp[dp["position"].isin(POSITIONS)], "player_id", "position", KNOWN_AT):
                _add(cand, pid, pos, "draft", at, 3)
        # usage-only candidates are dropped when a fresher pre-kickoff signal puts them on ANOTHER team
        for pid, c in list(cand.items()):
            if c["src"] != {"usage"}:
                continue
            dpl = as_of_join(store, "depth_charts", ko, player_id=pid, trace=trace)
            if len(dpl) and ko - dpl[KNOWN_AT].max() <= DEPTH_MAX_AGE and dpl["team"].iloc[-1] != tg.team:
                del cand[pid]
                continue
            ij = as_of_join(store, "injuries", ko, player_id=pid, trace=trace)
            ij = ij[(ij["season"] == tg.season) & (ij["week"] == tg.week)]
            if len(ij) and ij["team"].iloc[-1] != tg.team:
                del cand[pid]
        rows = [TargetRow(pid, tg.season, tg.week, tg.team, tg.opponent, c["pos"], tg.game_id, ko, tg.is_home,
                          tuple(sorted(c["src"])))
                for pid, c in sorted(cand.items()) if c["pos"] in POSITIONS]
        return rows, last_gids
    return _memo(store, trace, ("spine", tg.game_id, tg.team), build)


def spine_for(store, tg: TargetRow, *, trace: list | None = None) -> list[TargetRow]:
    """Eligible rows for one team-game (`tg` is a TargetRow with an empty player_id), from pre-kickoff
    information only. Each row's `src` says which signals admitted him: usage / injury / depth / draft."""
    return list(_spine(store, tg, trace)[0])


# --------------------------------------------------------------------------- team-game context
def _team_context(store, t: TargetRow, trace) -> dict[str, Any]:
    """Everything that is the same for every player on this team in this game. Computed once per
    (store, game, team) through the gate at this game's kickoff."""
    def build() -> dict[str, Any]:
        ko = t.kickoff
        ctx: dict[str, Any] = {}
        fx = _last(as_of_join(store, "fixtures", ko, game_id=t.game_id, trace=trace))
        ln = as_of_join(store, "lines", ko, game_id=t.game_id, trace=trace)
        line = _last(ln)
        spread_home = _f(line["spread_line"]) if line else NAN               # + = home favored
        total = _f(line["total_line"]) if line else NAN
        spread = spread_home if t.is_home else -spread_home                  # + = this team favored
        ok = not (math.isnan(total) or math.isnan(spread))
        ctx["vegas"] = {"team_spread": spread, "total_line": total,
                        "implied_team_total": total / 2 + spread / 2 if ok else NAN,
                        "implied_opp_total": total / 2 - spread / 2 if ok else NAN}
        ctx["context"] = {"is_home": int(t.is_home),
                          "rest_days": _f(fx["home_rest"] if t.is_home else fx["away_rest"]),
                          "opp_rest_days": _f(fx["away_rest"] if t.is_home else fx["home_rest"]),
                          "div_game": int(fx["div_game"]), "neutral_site": int(fx["location"] == "Neutral"),
                          "week": t.week, "season": t.season}
        wx = as_of_join(store, "weather_obs", ko, game_id=t.game_id, trace=trace)
        w = _last(wx)
        roof = ROOF_STRUCTURE.get(_s(w["roof"])) if w else None      # '' (not yet set, 2026) -> None
        ctx["weather"] = {"wx_temp_obs": _f(w["temp"]) if w else NAN,
                          "wx_wind_obs": _f(w["wind"]) if w else NAN,
                          "wx_roof_obs": roof,
                          "wx_indoor_obs": (int(roof == "dome") if roof is not None else None)}
        # opponent DvP: the opponent's completed REG games this season, PPR it allowed per position
        gr = as_of_join(store, "game_results", _rk(ko), team=t.opponent, trace=trace)
        gr = gr[(gr["season"] == t.season) & (gr["game_type"] == "REG")]
        allowed = as_of_join(store, "player_games", _rk(ko), opponent_team=t.opponent, trace=trace)
        allowed = allowed[allowed["season"] == t.season]
        per_game = allowed.groupby(["game_id", "position"])["fantasy_points_ppr"].sum()
        dvp: dict[str, dict[str, Any]] = {}
        for pos in POSITIONS:
            d: dict[str, Any] = {}
            for n in (2, 4):
                last = gr.iloc[-n:]
                pts = [float(per_game.get((gid, pos), 0.0)) for gid in last["game_id"]]   # 0 if the position scored nothing
                d[f"dvp_ppr_l{n}"] = float(sum(pts) / len(pts)) if pts else NAN
                d[f"dvp_l{n}_games"] = int(len(last))
                d[f"dvp_l{n}_last_week"] = int(last["week"].max()) if len(last) else None
            allp = [float(per_game.get((gid, pos), 0.0)) for gid in gr["game_id"]]
            d["dvp_ppr_std"] = float(sum(allp) / len(allp)) if allp else NAN
            dvp[pos] = d
        ctx["dvp"] = dvp
        ctx["pbp_team"] = _team_pbp_features(store, t, trace)
        ctx["nfelo"] = _nfelo_features(store, t, trace)
        ctx["script"] = _script_team(store, t, ctx["vegas"], trace)
        # teammates: the spine, each with trailing usage and this week's report
        rows, last_gids = _spine(store, t, trace)
        inj = as_of_join(store, "injuries", ko, team=t.team, trace=trace)
        inj = inj[(inj["season"] == t.season) & (inj["week"] == t.week)]
        status = dict(_cols(inj, "player_id", "report_status"))            # last known wins (sorted)
        tab = {}
        for s in rows:
            tab[s.player_id] = _teammate_row(_history(store, s.player_id, t, trace), t, s.position, last_gids,
                                             status.get(s.player_id))
        ctx["team_table"], ctx["last_gids"], ctx["status"] = tab, last_gids, status
        return ctx
    return _memo(store, trace, ("ctx", t.game_id, t.team), build)


def _teammate_row(h: Hist, t: TargetRow, pos: str, last_gids: list[str], status) -> dict[str, Any]:
    cur, prev = _cur_prev(h, t.season)
    return {"pos": pos, "recent": any(g in last_gids for g in h.game_id),
            "score": _trailing(cur, prev, SNAP), "opps3": _trailing(cur, prev, OPPS),
            "tgt3": _trailing(cur, prev, TGT), "car3": _trailing(cur, prev, CAR),
            "status": _s(status)}


def _role_features(ctx: dict, h: Hist, t: TargetRow) -> dict[str, Any]:
    tab = ctx["team_table"]
    me = tab.get(t.player_id) or _teammate_row(h, t, t.position, ctx["last_gids"], ctx["status"].get(t.player_id))
    mine = -1.0 if _isnan(me["score"]) else me["score"]
    score = lambda r: -1.0 if _isnan(r["score"]) else r["score"]           # noqa: E731  (no history ranks last)
    others = {pid: r for pid, r in tab.items() if pid != t.player_id and r["recent"]}
    same = [r for r in others.values() if r["pos"] == t.position]
    out_same = [r for r in same if r["status"] in OUT_LIKE]
    out_all = [r for r in others.values() if r["status"] in OUT_LIKE]
    z = lambda x: 0.0 if _isnan(x) else float(x)                           # noqa: E731
    return {
        "role_score": me["score"],
        "role_rank_pos": float(1 + sum(score(r) > mine for r in same)) if not _isnan(me["score"]) else NAN,
        "tm_group_size": len(same),
        "tm_out_group_n": len(out_same),
        "tm_out_above_n": sum(score(r) > mine for r in out_same),
        "tm_q_group_n": sum(r["status"] == "Questionable" for r in same),
        "tm_out_group_opps": sum(z(r["opps3"]) for r in out_same),
        "tm_out_targets_all": sum(z(r["tgt3"]) for r in out_all),
        "tm_out_carries_all": sum(z(r["car3"]) for r in out_all),
    }


# --------------------------------------------------------------------------- play-by-play families (Phase 2.5)
PBP_U = ("tgt", "rz_tgt", "i10_tgt", "ay", "ay_n", "car", "rz_car", "i10_car")            # pbp_usage columns
PBP_T = ("tgt", "rz_tgt", "i10_tgt", "ay", "car", "rz_car", "i10_car",                    # pbp_team columns, then
         "part_dropbacks", "part_rush", "part_rz", "part_i10")                            # the participation coverage
PBP_P = ("pass_on", "run_on", "rz_on", "i10_on")                                          # pbp_part columns


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else NAN


def _pbp_team_games(store, team: str, t: TargetRow, trace) -> dict[str, np.ndarray]:
    """{game_id: team totals} for the team's games known before this row's result cutoff (kickoff - RESULT_LAG)."""
    def build() -> dict[str, np.ndarray]:
        tm = as_of_join(store, "pbp_team", _rk(t.kickoff), team=team, trace=trace)
        return {g: np.array(v, dtype="float64") for g, *v in zip(tm["game_id"], *[tm[c] for c in PBP_T])}
    return _memo(store, trace, ("pbp_team_games", team, t.game_id), build)


def _pbp_arrays(store, player_id: str, t: TargetRow, h: Hist, trace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-appearance arrays aligned with `h` (same games, same order): the player's play-by-play counts (U), his team's
    totals for that game (T, plus how many of its plays the participation feed covers) and his on-field counts (P).
    A game with no play-by-play row for its team is all-NaN (excluded from every window); a player with no row in a game
    that HAS play-by-play had zero of that stat (U = 0), and P is NaN where the feed does not cover the game."""
    def build():
        n = len(h)
        U = np.full((n, len(PBP_U)), np.nan)
        T = np.full((n, len(PBP_T)), np.nan)
        P = np.full((n, len(PBP_P)), np.nan)
        if not n or not store.has("pbp_usage"):
            return U, T, P
        rk = _rk(t.kickoff)
        u = as_of_join(store, "pbp_usage", rk, player_id=player_id, trace=trace)
        used = {g: np.array(v, dtype="float64") for g, *v in zip(u["game_id"], *[u[c] for c in PBP_U])}
        teams = {tm: _pbp_team_games(store, tm, t, trace) for tm in dict.fromkeys(h.team)}
        pp = {}
        if store.has("pbp_part"):
            q = as_of_join(store, "pbp_part", rk, player_id=player_id, trace=trace)
            pp = {g: np.array(v, dtype="float64") for g, *v in zip(q["game_id"], *[q[c] for c in PBP_P])}
        for i, (g, tm) in enumerate(zip(h.game_id, h.team)):
            row = teams[tm].get(g)
            if row is None:
                continue
            T[i] = row
            U[i] = used.get(g, 0.0)
            if row[7] > 0:                                   # the participation feed covers this game
                P[i] = pp.get(g, 0.0)
        return U, T, P
    return _memo(store, trace, ("pbp_arr", player_id, t.game_id), build)


def _usage_stats(U: np.ndarray, T: np.ndarray, idx: np.ndarray) -> dict[str, float]:
    idx = idx[~np.isnan(T[idx, 0])] if len(idx) else idx
    if not len(idx):
        return {n: NAN for n in PBP_USAGE_STATS}
    u, tt = U[idx].sum(axis=0), T[idx].sum(axis=0)
    tsh, ash = _ratio(u[0], tt[0]), _ratio(u[3], tt[3])
    return {"rz_tgt_share": _ratio(u[1], tt[1]), "i10_tgt_share": _ratio(u[2], tt[2]),
            "rz_car_share": _ratio(u[6], tt[5]), "i10_car_share": _ratio(u[7], tt[6]),
            "ay_share": ash, "adot": _ratio(u[3], u[4]),
            "wopr": 1.5 * tsh + 0.7 * ash if not (math.isnan(tsh) or math.isnan(ash)) else NAN,
            "rz_opps_pg": float((U[idx, 1] + U[idx, 6]).mean()), "i10_opps_pg": float((U[idx, 2] + U[idx, 7]).mean())}


def _part_stats(P: np.ndarray, T: np.ndarray, idx: np.ndarray) -> dict[str, float]:
    idx = idx[~np.isnan(P[idx, 0])] if len(idx) else idx
    if not len(idx):
        return {n: NAN for n in PART_STATS}
    p, tt = P[idx].sum(axis=0), T[idx, 7:11].sum(axis=0)
    return {"pass_snap_share": _ratio(p[0], tt[0]), "run_snap_share": _ratio(p[1], tt[1]),
            "rz_snap_share": _ratio(p[2], tt[2])}


def _pbp_features(store, t: TargetRow, h: Hist, ctx: dict, trace) -> dict[str, float]:
    """pbp_usage, pbp_team and pbp_part columns. Windows: l1 = his last appearance this season, t3 = his last three, std =
    all of them; `prev` = last season's regular-season appearances. Every input is a game that ended before the row's result
    cutoff (the arrays come through the gate); week 1 has no current-season window, so those columns are NaN."""
    out: dict[str, float] = {c: NAN for fam in ("pbp_usage", "pbp_part") for c in FAMILIES[fam]}
    out.update(ctx["pbp_team"])
    U, T, P = _pbp_arrays(store, t.player_id, t, h, trace)
    cur = np.flatnonzero(h.season == t.season)
    prev = np.flatnonzero((h.season == t.season - 1) & h.is_reg)
    for w, idx in (("l1", cur[-1:]), ("t3", cur[-3:]), ("std", cur)):
        for k, v in _usage_stats(U, T, idx).items():
            out[f"pbp_{k}_{w}"] = v
        for k, v in _part_stats(P, T, idx).items():
            out[f"part_{k}_{w}"] = v
    pv = _usage_stats(U, T, prev)
    for k in PBP_PREV_STATS:
        out[f"pbp_prev_{k}"] = pv[k]
    out["part_prev_pass_snap_share"] = _part_stats(P, T, prev)["pass_snap_share"]
    return out


def _team_pbp_features(store, t: TargetRow, trace) -> dict[str, float]:
    """The team's own pace and dropback rates over its last three / all games this season and last season, and the plays
    and dropback rate its opponent's DEFENSE has faced (the opponent's rows as `opponent`). Regular-season games that ended
    before the row's result cutoff only."""
    out = {c: NAN for c in PBP_TEAM_COLUMNS}
    if not store.has("pbp_team"):
        return out
    rk = _rk(t.kickoff)
    mine = as_of_join(store, "pbp_team", rk, team=t.team, trace=trace)
    mine = mine[mine["game_type"] == "REG"]
    cur, prev = mine[mine["season"] == t.season], mine[mine["season"] == t.season - 1]

    def rates(df: pd.DataFrame) -> tuple[float, float, float]:
        if not len(df):
            return NAN, NAN, NAN
        return (float(df["plays"].mean()), _ratio(df["dropbacks"].sum(), df["plays"].sum()),
                _ratio(df["neutral_dropbacks"].sum(), df["neutral_plays"].sum()))
    for w, df in (("t3", cur.tail(3)), ("std", cur)):
        out[f"team_plays_pg_{w}"], out[f"team_pass_rate_{w}"], out[f"team_neutral_pass_rate_{w}"] = rates(df)
    out["team_plays_pg_prev"], out["team_pass_rate_prev"], _ = rates(prev)
    faced = as_of_join(store, "pbp_team", rk, opponent=t.opponent, trace=trace)
    faced = faced[(faced["game_type"] == "REG") & (faced["season"] == t.season)]
    out["opp_plays_faced_pg_std"], out["opp_pass_rate_faced_std"], _ = rates(faced)
    return out


def _script_window(store, t: TargetRow, trace, **key) -> pd.DataFrame:
    """The last SCRIPT_WINDOW regular-season pbp_team rows matching `key` (team=... for an offense, opponent=... for the defense
    it faced) from this season and last season, of games that ended before the row's result cutoff."""
    df = as_of_join(store, "pbp_team", _rk(t.kickoff), trace=trace, **key)
    df = df[(df["game_type"] == "REG") & df["season"].isin([t.season - 1, t.season])]
    return df.tail(SCRIPT_WINDOW)


def _sr(df: pd.DataFrame, num: str, den: str, scale: float = 1.0) -> float:
    """Ratio of column sums (NaN-ignoring; NaN when the denominator is 0 or the window is empty)."""
    if not len(df):
        return NAN
    return _ratio(scale * float(np.nansum(df[num].to_numpy(dtype="float64"))), float(np.nansum(df[den].to_numpy(dtype="float64"))))


def expected_team_plays(trail_plays: float, spread: float, total: float, rule: dict | None = None) -> float:
    """SCRIPT_PLAYS_RULE applied: NaN if any input is missing."""
    r = rule or SCRIPT_PLAYS_RULE
    if any(math.isnan(x) for x in (trail_plays, spread, total)):
        return NAN
    return r["intercept"] + r["trail_plays"] * trail_plays + r["abs_spread"] * abs(spread) + r["total"] * total


def _coach_features(store, t: TargetRow, trace) -> dict[str, float]:
    """hc_changed_this_season: this game's head coach differs from the coach of the team's first game this season (for that
    first game itself: from the coach of the team's last game of the previous season). hc_games_in_charge: games (any type, since
    2000) this coach coached this team before this one. NaN without a `coaches` table or a row for this game."""
    out = {c: NAN for c in SCRIPT_COACH_COLUMNS}
    if not store.has("coaches"):
        return out
    co = as_of_join(store, "coaches", t.kickoff, team=t.team, trace=trace)
    me = co[co["game_id"] == t.game_id]
    if not len(me):
        return out
    coach = me["coach"].iloc[-1]
    before = co[co["game_id"] != t.game_id]
    before = before[before[KNOWN_AT] < me[KNOWN_AT].iloc[-1]]          # earlier games only (no later fixture can be known)
    this = before[before["season"] == t.season]
    ref = this["coach"].iloc[0] if len(this) else (
        before.loc[before["season"] == t.season - 1, "coach"].iloc[-1] if (before["season"] == t.season - 1).any() else None)
    out["hc_changed_this_season"] = NAN if ref is None else float(coach != ref)
    out["hc_games_in_charge"] = float((before["coach"] == coach).sum())
    return out


def _script_team(store, t: TargetRow, vegas: dict, trace) -> dict[str, float]:
    """Team-game half of the `script` family (every column but the player's own expected volume), plus `_trail_pass_rate`,
    `_exp_pass`, `_exp_rush` for the player half. Memoised per (game, team) through `_team_context`."""
    out: dict[str, float] = {c: NAN for c in SCRIPT_TEAM_COLUMNS + SCRIPT_OPP_COLUMNS + SCRIPT_GAME_COLUMNS}
    out.update(_coach_features(store, t, trace))
    out["_trail_plays"] = out["_trail_pass_rate"] = NAN
    if not store.has("pbp_team"):
        return out
    mine = _script_window(store, t, trace, team=t.team)
    if len(mine):
        out["scr_sec_per_play_l4"] = _sr(mine, "sc_secs", "sc_secs_n")
        out["scr_no_huddle_rate_l4"] = _sr(mine, "sc_nohuddle", "plays")
        out["scr_proe_l4"] = _sr(mine, "sc_pass_oe", "sc_xp_n")
        out["scr_xpass_l4"] = _sr(mine, "sc_xpass", "sc_xp_n")
        out["scr_early_down_pass_rate_l4"] = _sr(mine, "sc_early_db", "sc_early_plays")
        out["scr_wp_neutral_pass_rate_l4"] = _sr(mine, "sc_wp_db", "sc_wp_plays")
        out["scr_rz_plays_pg_l4"] = _nm(mine["sc_rz_plays"].to_numpy(dtype="float64"))
        out["scr_epa_per_play_l4"] = _sr(mine, "sc_epa", "sc_epa_n")
        out["scr_success_rate_l4"] = _sr(mine, "sc_succ", "sc_epa_n")
        out["_trail_plays"] = _nm(mine["plays"].to_numpy(dtype="float64"))
        out["_trail_pass_rate"] = _sr(mine, "dropbacks", "plays")
    faced = _script_window(store, t, trace, opponent=t.opponent)
    if len(faced):
        out["scr_opp_epa_allowed_l4"] = _sr(faced, "sc_epa", "sc_epa_n")
        out["scr_opp_pass_epa_allowed_l4"] = _sr(faced, "sc_pass_epa", "sc_pass_n")
        out["scr_opp_rush_epa_allowed_l4"] = _sr(faced, "sc_rush_epa", "sc_rush_n")
    spread, total, itt = vegas["team_spread"], vegas["total_line"], vegas["implied_team_total"]
    rate = out["_trail_pass_rate"]
    out["scr_itt_x_pass_rate"] = itt * rate
    out["scr_spread_x_proe"] = spread * out["scr_proe_l4"]
    plays = expected_team_plays(out["_trail_plays"], spread, total)
    out["expected_team_plays"] = plays
    out["expected_pass_att"] = plays * rate
    out["expected_rush_att"] = plays * (1.0 - rate)
    return out


def _team_dropbacks(store, team: str, t: TargetRow, trace) -> dict[str, float]:
    """{game_id: dropbacks} for a team's games known before the row's result cutoff."""
    def build() -> dict[str, float]:
        tm = as_of_join(store, "pbp_team", _rk(t.kickoff), team=team, trace=trace)
        return dict(zip(tm["game_id"], tm["dropbacks"].astype("float64")))
    return _memo(store, trace, ("pbp_team_db", team, t.game_id), build)


def _script_player(store, t: TargetRow, h: Hist, sc: dict, trace) -> dict[str, float]:
    """The player's trailing-3 share applied to the team's expected volume: target share x expected dropbacks, carry share x
    expected rushes, (his pass attempts / team dropbacks) x expected dropbacks. Shares are his last three appearances this season
    (last season's REG appearances when he has none), like the other trailing-3 columns."""
    cur, prev = _cur_prev(h, t.season)
    out = {"exp_player_targets": _trailing(cur, prev, TSH) * sc["expected_pass_att"],
           "exp_player_carries": _trailing(cur, prev, CSH) * sc["expected_rush_att"],
           "exp_player_pass_att": NAN}
    if not store.has("pbp_team") or not len(h):
        return out
    idx = np.flatnonzero(h.season == t.season)
    if not len(idx):
        idx = np.flatnonzero((h.season == t.season - 1) & h.is_reg)
    idx = idx[-3:]
    att = db = 0.0
    for i in idx:
        d = _team_dropbacks(store, h.team[i], t, trace).get(h.game_id[i])
        a = h.v[i, ATT]
        if d is None or math.isnan(d) or math.isnan(a):
            continue
        att, db = att + a, db + d
    out["exp_player_pass_att"] = _ratio(att, db) * sc["expected_pass_att"] if db > 0 else NAN
    return out


def _eff_features(h: Hist, t: TargetRow) -> dict[str, float]:
    """Shrunken efficiency: (sum of numerator + k * prior) / (sum of denominator + k), the prior fixed by position (EFF_PRIORS,
    2015-2019), over his regular-season appearances since 2020 (`career`) or this season's (`std`). NaN when he had no
    opportunity of that kind in the window (no denominator): a prior with no data is a constant per position."""
    spec = {"catch_rate": (REC, TGT), "rec_ypt": (RECY, TGT), "rec_td_rate": (RECTD, TGT), "rush_ypc": (RUSHY, CAR),
            "rush_td_rate": (RUSHTD, CAR), "pass_ypa": (PASSY, ATT), "pass_td_rate": (PASSTD, ATT),
            "pass_int_rate": (INT, ATT), "pass_cmp_rate": (CMP, ATT)}
    pri = EFF_PRIORS[t.position]
    out: dict[str, float] = {}
    for w, mask in (("career", h.is_reg), ("std", h.is_reg & (h.season == t.season))):
        v = h.v[mask]
        for name, (num, den) in spec.items():
            d = float(np.nansum(v[:, den])) if len(v) else 0.0
            n = float(np.nansum(v[:, num])) if len(v) else 0.0
            out[f"eff_{name}_{w}"] = (n + EFF_K[name] * pri[name]) / (d + EFF_K[name]) if d > 0 else NAN
    return out


def _nfelo_features(store, t: TargetRow, trace) -> dict[str, float]:
    """The `nfelo` family for one team-game (memoised with the team context). Ratings are this (season, week)'s PRE rows, read
    at kickoff (they are stamped at the week's first kickoff - 1h); WEPA and film margin are the teams' earlier games, read at
    the result cutoff; line movement is the newest open/last value scraped strictly before kickoff."""
    out = {c: NAN for c in NFELO_COLUMNS}
    ko, rk = t.kickoff, _rk(t.kickoff)

    def weekly(table: str, team: str):
        return _last(as_of_join(store, table, ko, team=team, season=t.season, week=t.week, trace=trace))

    if store.has("nf_units"):
        me, op = weekly("nf_units", t.team), weekly("nf_units", t.opponent)
        if me:
            out["nf_tm_pass_off"], out["nf_tm_rush_off"] = _f(me["pass_off_value_pre"]), _f(me["rush_off_value_pre"])
            out["nf_tm_total"] = _f(me["total_value_pre"])
        if op:
            out["nf_opp_pass_def"], out["nf_opp_rush_def"] = _f(op["pass_def_value_pre"]), _f(op["rush_def_value_pre"])
            out["nf_opp_total"] = _f(op["total_value_pre"])
        out["nf_pass_edge"] = out["nf_tm_pass_off"] - out["nf_opp_pass_def"]
        out["nf_rush_edge"] = out["nf_tm_rush_off"] - out["nf_opp_rush_def"]
    if store.has("nf_elo"):
        me, op = weekly("nf_elo", t.team), weekly("nf_elo", t.opponent)
        if me:
            out["nf_elo"], out["nf_qb_adj"] = _f(me["elo"]), _f(me["qb_adj"])
        if op:
            out["nf_opp_elo"], out["nf_opp_qb_adj"] = _f(op["elo"]), _f(op["qb_adj"])
        out["nf_elo_dif"] = out["nf_elo"] - out["nf_opp_elo"]
    if store.has("nf_qb"):
        q = as_of_join(store, "nf_qb", ko, game_id=t.game_id, trace=trace)
        for team, col in ((t.team, "nf_qb_value"), (t.opponent, "nf_opp_qb_value")):
            r = _last(q[q["team"] == team])
            if r:
                out[col] = _f(r["qb_value_pre"])
    if store.has("nf_wepa"):
        for team, col in ((t.team, "nf_wepa_net"), (t.opponent, "nf_opp_wepa_net")):
            w = as_of_join(store, "nf_wepa", rk, team=team, season=t.season, trace=trace)
            out[col] = _nm(w["wepa_net"].to_numpy("float64")) if len(w) else NAN
    if store.has("nf_film"):
        f = as_of_join(store, "nf_film", rk, team=t.team, trace=trace)
        f = f[f["season"].isin([t.season - 1, t.season])].tail(FILM_WINDOW)
        # staleness guard: the source lags (no 2026 rows yet). If the team's newest completed REG game is missing from it, the
        # window would silently be old games, so the value is NaN rather than stale.
        newest = None
        if store.has("game_results"):
            gr = as_of_join(store, "game_results", rk, team=t.team, trace=trace)
            gr = gr[(gr["game_type"] == "REG") & gr["season"].isin([t.season - 1, t.season])]
            newest = gr["game_id"].iloc[-1] if len(gr) else None
        if len(f) and (newest is None or newest in set(f["game_id"])):
            out["nf_film_margin_l4"] = _nm(f["film_margin_predictive"].to_numpy("float64"))
    if store.has("nf_lines"):
        ln = as_of_join(store, "nf_lines", ko, game_id=t.game_id, trace=trace)
        val = {k: _f(v) for k, v in zip(ln["kind"], ln["value"])}          # oldest known first: the newest of each kind wins
        sign = 1.0 if t.is_home else -1.0
        so, sl = val.get("spread_open", NAN), val.get("spread_last", NAN)
        if not math.isnan(so):
            # newest admissible line minus the open; a last line scraped at/after kickoff never passed the gate -> the open
            out["nf_spread_move"] = sign * ((so if math.isnan(sl) else sl) - so)
        to, tl = val.get("total_open", NAN), val.get("total_last", NAN)
        if not math.isnan(to):
            out["nf_total_move"] = (to if math.isnan(tl) else tl) - to
        tk = val.get("tickets", NAN)
        out["nf_tickets_pct"] = tk if t.is_home or math.isnan(tk) else 1.0 - tk
    return out


# --------------------------------------------------------------------------- weekly charting releases (Phase 5 candidates)
def _published(store, table: str, t: TargetRow, h: Hist, trace) -> np.ndarray:
    """Indices into `h` of this season's appearances whose game the release `table` (a coverage table: one row per team-game it
    has published) shows before kickoff, oldest first. The coverage rows carry the release's own known_at (a weekly refresh
    after the week ends), so a game that ended but is not yet published is simply not in the window."""
    cur = np.flatnonzero(h.season == t.season)
    if not len(cur) or not store.has(table):
        return cur[:0]
    seen: set = set()
    for tm in dict.fromkeys(h.team[i] for i in cur):
        seen |= _memo(store, trace, (table, tm, t.game_id),
                      lambda tm=tm: set(as_of_join(store, table, t.kickoff, team=tm, trace=trace)["game_id"]))
    return np.array([i for i in cur if h.game_id[i] in seen], dtype="int64")


def _player_rows(store, table: str, t: TargetRow, cols: tuple, trace) -> dict[str, np.ndarray]:
    if not store.has(table):
        return {}
    d = as_of_join(store, table, t.kickoff, player_id=t.player_id, trace=trace)
    return {g: np.array(v, dtype="float64") for g, *v in zip(d["game_id"], *[d[c] for c in cols])}


def _charting_features(store, t: TargetRow, h: Hist, trace) -> dict[str, float]:
    """The `ngs`, `pfr` and `ftn` families. Window = his appearances this season (h, read at kickoff - RESULT_LAG) whose game the
    release had published before kickoff (`_published`); l1 = the newest such game, l3 = the newest three.

      ngs  NaN in a published game where he did not qualify (NGS is thresholded); l3 = mean of the non-NaN games
      pfr  NaN in a published game with no PFR line for him; per-game values averaged, per-carry rates as ratios of sums (PFR_RATIOS)
      ftn  a charted game with no row for him = zero of everything (he was not targeted / did not carry / drop back); rates are
           ratios of the window's sums (FTN_SPEC). FTN starts in 2022: 2021 rows (and any season FTN never published) are NaN.

    Week 1 has no current-season window, so every column is NaN, like the lags."""
    out: dict[str, float] = {c: NAN for fam in ("ngs", "pfr", "ftn") for c in FAMILIES[fam]}
    # ---- NGS
    idx = _published(store, "ngs_games", t, h, trace)
    if len(idx):
        rows = _player_rows(store, "ngs", t, NGS_STATS, trace)
        V = np.array([rows.get(h.game_id[i], np.full(len(NGS_STATS), np.nan)) for i in idx]).reshape(len(idx), len(NGS_STATS))
        for j, s in enumerate(NGS_STATS):
            out[f"ngs_{s}_l1"] = float(V[-1, j])
            out[f"ngs_{s}_l3"] = _nm(V[-3:, j])
    # ---- PFR
    idx = _published(store, "pfr_games", t, h, trace)
    if len(idx):
        rows = _player_rows(store, "pfr", t, PFR_RAW, trace)
        V = np.array([rows.get(h.game_id[i], np.full(len(PFR_RAW), np.nan)) for i in idx]).reshape(len(idx), len(PFR_RAW))
        col = {c: j for j, c in enumerate(PFR_RAW)}
        for s in PFR_STATS:
            for w, sl in (("l1", V[-1:]), ("l3", V[-3:])):
                if s in PFR_RATIOS:
                    num, den = PFR_RATIOS[s]
                    ok = ~np.isnan(sl[:, col[num]]) & ~np.isnan(sl[:, col[den]])
                    out[f"pfr_{s}_{w}"] = _ratio(float(sl[ok, col[num]].sum()), float(sl[ok, col[den]].sum())) if ok.any() else NAN
                else:
                    out[f"pfr_{s}_{w}"] = _nm(sl[:, col[s]])
    # ---- FTN
    idx = _published(store, "ftn_team", t, h, trace)
    if len(idx):
        rows = _player_rows(store, "ftn", t, FTN_C, trace)
        last = idx[-3:]
        mine = np.array([rows.get(h.game_id[i], np.zeros(len(FTN_C))) for i in last]).reshape(len(last), len(FTN_C)).sum(axis=0)
        team: dict[str, float] = dict.fromkeys(FTN_T, 0.0)
        for i in last:
            tm = as_of_join(store, "ftn_team", t.kickoff, team=h.team[i], game_id=h.game_id[i], trace=trace)
            for c in FTN_T:
                team[c] += float(tm[c].iloc[-1])
        have = {c: float(mine[j]) for j, c in enumerate(FTN_C)} | team
        for s, (num, den) in FTN_SPEC.items():
            out[f"ftn_{s}_l3"] = have[num] / len(last) if den is None else _ratio(have[num], have[den])
    return out


# --------------------------------------------------------------------------- opponent DvP split by stat (Phase 5 candidate)
def _dvps_league(store, t: TargetRow, trace) -> dict[tuple, float]:
    """{(rank stem, position, defense): rank} over every defense with a completed REG game this season before this game's result
    cutoff: 1 = most yards allowed per game to that position (ties share the better rank). The one league-wide read of the
    family; memoised per game (both teams share the kickoff, so the cutoff)."""
    def build() -> dict[tuple, float]:
        rk = _rk(t.kickoff)
        gr = as_of_join(store, "game_results", rk, trace=trace)
        gr = gr[(gr["season"] == t.season) & (gr["game_type"] == "REG")]
        games = gr.groupby("team")["game_id"].nunique()
        games = games[games > 0]
        pg = as_of_join(store, "player_games", rk, trace=trace)
        pg = pg[pg["season"] == t.season]
        cols = [c for c in DVPS_RANKS.values() if c in pg.columns]
        sums = pg.groupby(["opponent_team", "position"])[cols].sum()
        out: dict[tuple, float] = {}
        for stem, col in DVPS_RANKS.items():
            if col not in cols:
                continue
            for pos in POSITIONS:
                allowed = {tm: float(sums[col].get((tm, pos), 0.0)) / float(n) for tm, n in games.items()}
                for tm, v in allowed.items():
                    out[(stem, pos, tm)] = 1.0 + sum(o > v for o in allowed.values())
        return out
    return _memo(store, trace, ("dvps_league", t.game_id), build)


def _dvp_split_table(store, t: TargetRow, trace) -> dict[str, dict[str, float]]:
    """{position: {column: value}} for the opponent's defense: its completed REG games this season before this game's result
    cutoff (kickoff - RESULT_LAG, as `dvp`), l4 = the last four, std = all of them. Per-game means of what it allowed to each
    position (DVPS_ALLOWED), ratios of sums for the play-by-play rates (pbp_team rows with the defense as `opponent`) and the
    FTN rates (`ftn_def`, read at kickoff: a Wednesday release), ranks from `_dvps_league`. NaN with no history (week 1) or
    without the table. Memoised per (game, defense)."""
    def build() -> dict[str, dict[str, float]]:
        rk = _rk(t.kickoff)
        out = {pos: {c: NAN for c in DVPS_COLUMNS} for pos in POSITIONS}
        if not (store.has("game_results") and store.has("player_games")):
            return out
        gr = as_of_join(store, "game_results", rk, team=t.opponent, trace=trace)
        gr = gr[(gr["season"] == t.season) & (gr["game_type"] == "REG")]
        gids = list(gr["game_id"])                               # oldest known first: chronological
        pg = as_of_join(store, "player_games", rk, opponent_team=t.opponent, trace=trace)
        pg = pg[pg["season"] == t.season]
        have = {s: c for s, c in DVPS_ALLOWED.items() if c in pg.columns}
        sums = pg.groupby(["game_id", "position"])[list(have.values())].sum()
        for pos in POSITIONS:
            d = out[pos]
            for w, sel in (("l4", gids[-DVPS_WINDOW:]), ("std", gids)):
                if not sel:
                    continue
                for s, c in have.items():
                    d[f"dvps_{s}_{w}"] = float(sum(float(sums[c].get((g, pos), 0.0)) for g in sel) / len(sel))
        if store.has("pbp_team"):
            faced = as_of_join(store, "pbp_team", rk, opponent=t.opponent, trace=trace)
            faced = faced[(faced["game_type"] == "REG") & (faced["season"] == t.season)]
            for w, df in (("l4", faced.tail(DVPS_WINDOW)), ("std", faced)):
                rates = {f"dvps_epa_pass_{w}": _sr(df, "sc_pass_epa", "sc_pass_n"),
                         f"dvps_epa_rush_{w}": _sr(df, "sc_rush_epa", "sc_rush_n"),
                         f"dvps_pass_rate_faced_{w}": _sr(df, "dropbacks", "plays"),
                         f"dvps_explosive_rate_{w}": _sr(df, "sc_explosive", "plays") if "sc_explosive" in df.columns else NAN}
                for pos in POSITIONS:
                    out[pos].update(rates)
        if store.has("ftn_def"):
            fd = as_of_join(store, "ftn_def", t.kickoff, team=t.opponent, trace=trace)
            fd = fd[fd["season"] == t.season].tail(DVPS_WINDOW)
            rates = {"dvps_light_box_rate_l4": _sr(fd, "car_light", "car_box"), "dvps_blitz_rate_l4": _sr(fd, "db_blitz", "db"),
                     "dvps_avg_rushers_l4": _sr(fd, "rushers_sum", "db_rush_n")}
            for pos in POSITIONS:
                out[pos].update(rates)
        if gids:
            league = _dvps_league(store, t, trace)
            for pos in POSITIONS:
                for stem in DVPS_RANKS:
                    out[pos][f"dvps_{stem}_std"] = league.get((stem, pos, t.opponent), NAN)
        return out
    return _memo(store, trace, ("dvps", t.game_id, t.opponent), build)


def _dvp_split_features(store, t: TargetRow, trace) -> dict[str, float]:
    """The `dvp_split` family for his row: the defense table's column for his position where the stem bears on it (DVPS_POS),
    NaN elsewhere."""
    row = _dvp_split_table(store, t, trace).get(t.position, {})
    keep = DVPS_POS.get(t.position, frozenset())
    return {c: (row.get(c, NAN) if DVPS_STEM[c] in keep else NAN) for c in DVPS_COLUMNS}


# --------------------------------------------------------------------------- the row
def build_features(store, t: TargetRow, *, trace: list | None = None) -> dict:
    """Feature row for one target. Deterministic; NaN where history does not exist."""
    ko = t.kickoff
    ctx = _team_context(store, t, trace)
    h = _history(store, t.player_id, t, trace)
    out: dict[str, Any] = {
        "player_id": t.player_id, "season": t.season, "week": t.week, "team": t.team,
        "opponent": t.opponent, "position": t.position, "game_id": t.game_id,
        "kickoff_utc": ko.isoformat(), "spine_src": ",".join(t.src) if t.src else None,
    }

    # ---- lags, season-to-date, prior season, TD luck ---------------------------------
    out.update(_lag_and_history_features(h, t))
    out.update(_streak_features(h, t))                # opt-in family; same history, nothing new read
    out.update(_shape_features(h, t))                 # opt-in family (2026-10-27); same history, nothing new read

    # ---- role + teammate-injury context ----------------------------------------------
    out.update(_role_features(ctx, h, t))

    # ---- opponent DvP vs position ----------------------------------------------------
    out.update(ctx["dvp"].get(t.position) or {k: (NAN if "ppr" in k else None) for k in ctx["dvp"]["QB"]})
    out.update(_dvp_split_features(store, t, trace))     # opt-in `dvp_split`: the same defense, split by stat

    # ---- Vegas + context + weather (team-game level) ----------------------------------
    out.update(ctx["vegas"])
    out.update(ctx["context"])
    out.update(ctx["weather"])

    # ---- own injury / practice designation -------------------------------------------
    inj = as_of_join(store, "injuries", ko, player_id=t.player_id, trace=trace)
    inj = inj[inj["season"] == t.season]
    if len(inj):
        last = _last(inj)
        out["inj_report_status"] = _s(last["report_status"])
        out["inj_practice_status"] = _s(last["practice_status"])
        out["inj_last_report_week"] = int(last["week"])
        out["inj_days_since_report"] = float((ko - last["known_at"]) / DAY)   # EXCLUDED from matrix: skewed 2025+
        out["inj_weeks_since_report"] = int(t.week - last["week"])           # timestamp-free twin
        out["inj_listed_l4wk"] = int(((inj["week"] >= t.week - 4) & inj["report_status"].isin(LISTED)).sum())
    else:
        out.update(inj_report_status=None, inj_practice_status=None, inj_last_report_week=None,
                   inj_days_since_report=NAN, inj_weeks_since_report=None, inj_listed_l4wk=0)

    # ---- static (season-start known_at) ----------------------------------------------
    p = _last(as_of_join(store, "players_static", ko, player_id=t.player_id, trace=trace))
    dp = _last(as_of_join(store, "draft_picks", ko, player_id=t.player_id, trace=trace))
    c = _last(as_of_join(store, "combine", ko, player_id=t.player_id, trace=trace))
    cg = _last(as_of_join(store, "career_pre_cutoff", ko, player_id=t.player_id, trace=trace))
    if p:
        out["age_years"] = (ko.tz_convert(None) - pd.Timestamp(p["birth_date"])).total_seconds() / (365.25 * 86400)
        rookie = _f(p["rookie_season"])
        out["exp_years"] = float(t.season - rookie) if not math.isnan(rookie) else NAN
        out["is_rookie"] = int(rookie == t.season) if not math.isnan(rookie) else None
        out["is_undrafted"] = int(pd.isna(p["draft_year"]))
        out["height_in"], out["weight_lb"] = _f(p["height"]), _f(p["weight"])
        fallback_round, fallback_pick = _f(p["draft_round"]), _f(p["draft_pick"])
    else:
        out.update(age_years=NAN, exp_years=NAN, is_rookie=None, is_undrafted=None, height_in=NAN, weight_lb=NAN)
        fallback_round = fallback_pick = NAN
    out["draft_round"] = _f(dp["round"]) if dp else fallback_round
    out["draft_overall"] = _f(dp["pick"]) if dp else fallback_pick
    for name in ("forty", "bench", "vertical", "broad_jump", "cone", "shuttle"):
        out[f"combine_{name}"] = _f(c[name]) if c else NAN
    # career games: the pre-cutoff table + stats-row games in the store's completed earlier seasons
    later = (h.season < t.season) & (h.season >= CAREER_CUTOFF_SEASON) & h.is_reg & h.has_stats
    out["career_games_prior"] = float(later.sum()) + (_f(cg["games"]) if cg else 0.0)

    # ---- ADP (season prior) ----------------------------------------------------------
    out.update(_adp_features(store, t, trace))

    # ---- college production (rookie priors; season-start known_at) ---------------------
    out.update(_college_features(store, t, trace))

    # ---- FantasyPros weekly ECR (opt-in family; NaN unless the table was fetched) ----------
    out.update(_ecr_features(store, t, trace))

    # ---- Phase 2.5: play-by-play usage / team / participation, and shrunken efficiency -----
    out.update(_pbp_features(store, t, h, ctx, trace))
    out.update(_eff_features(h, t))

    # ---- Phase 5 candidates (opt-in): nfelo team-game context, player EPA / efficiency -----
    out.update(ctx["nfelo"])
    out.update(_pepa_features(h, t))
    # ---- Phase 5 candidate: game script and coaching context (opt-in `script`) ----------
    out.update({k: v for k, v in ctx["script"].items() if not k.startswith("_")})
    out.update(_script_player(store, t, h, ctx["script"], trace))
    # ---- Phase 5 candidates: NGS / PFR / FTN weekly charting (opt-in) ----------------
    out.update(_charting_features(store, t, h, trace))
    # ---- Phase 5 candidate: the market's projection for this game (opt-in) --------------
    out.update(_market_features(store, t, h, trace))
    # ---- Phase 5 candidate: the opponent's defensive injury report (opt-in `opp_db_out`) ----
    out.update(_opp_def_features(store, t, trace))
    return out


def _opp_def_features(store, t: TargetRow, trace) -> dict[str, Any]:
    """The `opp_db_out` family (OPP_DEF_COLUMNS and the meta `opp_report_present`), once per (store, game, opponent): the
    opponent's injury rows for this season and week known before kickoff, grouped by INJ_DEF_GROUPS, and its newest defensive
    depth chart before kickoff for who starts."""
    def build() -> dict[str, Any]:
        ko = t.kickoff
        inj = as_of_join(store, "injuries", ko, team=t.opponent, trace=trace)
        inj = inj[inj["season"] == t.season]
        wk = inj[inj["week"] == t.week]
        out: dict[str, Any] = {c: NAN for c in OPP_DEF_COLUMNS}
        out["opp_report_present"] = int(len(wk) > 0)
        # chronic: defensive 'Out' rows in the prior OPP_DEF_WINDOW weeks of this season (NaN with no report rows in the window)
        prior = inj[(inj["week"] >= t.week - OPP_DEF_WINDOW) & (inj["week"] < t.week)]
        if len(prior):
            grp = prior["position"].map(INJ_DEF_GROUPS)
            out["opp_def_out_l4wk"] = float(((prior["report_status"] == "Out") & grp.notna()).sum())
        if not len(wk):
            return out
        status = dict(_cols(wk, "player_id", "report_status"))              # last known wins (sorted by known_at)
        pos = dict(_cols(wk, "player_id", "position"))
        grp = {pid: INJ_DEF_GROUPS.get(_s(p)) for pid, p in pos.items()}
        out_like = [pid for pid, s in status.items() if s in OUT_LIKE and grp[pid]]
        for g, col in (("DB", "opp_db_out_n"), ("LB", "opp_lb_out_n"), ("DL", "opp_dl_out_n")):
            out[col] = float(sum(grp[pid] == g for pid in out_like))
        out["opp_def_out_n"] = float(len(out_like))
        out["opp_db_q_n"] = float(sum(s == "Questionable" and grp[pid] == "DB" for pid, s in status.items()))
        if store.has("depth_charts_def"):
            dc = as_of_join(store, "depth_charts_def", ko, team=t.opponent, trace=trace)
            if len(dc):
                newest = dc[KNOWN_AT].max()
                if ko - newest <= DEPTH_MAX_AGE:
                    dc = dc[dc[KNOWN_AT] == newest]
                    starters = set(dc.loc[(dc["grp"] == "DB") & (dc["rank"] == 1), "player_id"])
                    out["opp_db_starter_out_n"] = float(sum(pid in starters for pid in out_like if grp[pid] == "DB"))
        return out
    return _memo(store, trace, ("opp_def", t.game_id, t.opponent), build)


def _market_features(store, t: TargetRow, h: Hist, trace) -> dict[str, float]:
    """The `market` family: this game's archived Sleeper/Rotowire projection, through the gate at kickoff (the row is stamped
    kickoff - 1h, so it is the one table whose own-game row is a legitimate input). NaN when there is no row: the value is used,
    its presence never becomes a column."""
    out = {c: NAN for c in FAMILIES["market"]}
    if not store.has("sleeper_proj"):
        return out
    m = as_of_join(store, "sleeper_proj", t.kickoff, player_id=t.player_id, trace=trace)
    m = m[(m["season"] == t.season) & (m["week"] == t.week)]
    if not len(m):
        return out
    r = _last(m)
    for col, src in MARKET_STATS.items():
        v = _f(r[src])
        out[col] = 0.0 if math.isnan(v) and col != "mkt_pos_rank" else v
    out["mkt_td_total"] = sum(0.0 if math.isnan(x) else x for x in (_f(r["pass_td"]), _f(r["rush_td"]), _f(r["rec_td"])))
    cur, prev = _cur_prev(h, t.season)
    out["mkt_vs_xfp_l3"] = out["mkt_pts_ppr"] - _trailing(cur, prev, XFP, 3)
    return out


def _college_features(store, t: TargetRow, trace) -> dict[str, float]:
    """Rookie priors from the `college` table, through the gate. Shown only in the season the player was drafted
    into: a veteran row is NA, so coverage never depends on how many draft classes the CFBD pull holds."""
    out = {c: NAN for c in FAMILIES["college"]}
    if not store.has("college"):
        return out
    c = _last(as_of_join(store, "college", t.kickoff, player_id=t.player_id, trace=trace))
    if c is None or _f(c["draft_season"]) != t.season:
        return out
    for col in out:
        out[col] = _f(c[col])
    return out


def _ecr_features(store, t: TargetRow, trace) -> dict[str, float]:
    """This week's newest pre-kickoff ECR row, and last week's position rank for the delta. Week-matched on purpose: the gate
    alone would hand a Thursday game last week's ranking (scraped the Friday before), which ranked a different game."""
    out = {c: NAN for c in FAMILIES["ecr"]}
    if not store.has("ecr"):
        return out
    e = as_of_join(store, "ecr", t.kickoff, player_id=t.player_id, trace=trace)
    e = e[e["season"] == t.season]
    cur, prev = _last(e[e["week"] == t.week]), _last(e[e["week"] == t.week - 1])
    if cur is None:
        return out
    out["ecr_rank"] = _f(cur["ecr_sf_rank"])
    out["ecr_pos_rank"] = _f(cur["ecr_pos_rank"])
    out["ecr_sd"], out["ecr_best"], out["ecr_worst"] = _f(cur["ecr_sd"]), _f(cur["ecr_best"]), _f(cur["ecr_worst"])
    out["ecr_spread"] = out["ecr_worst"] - out["ecr_best"]
    out["ecr_owned_pct"] = _f(cur["ecr_owned_pct"])
    if prev is not None:
        out["ecr_delta_l1"] = out["ecr_pos_rank"] - _f(prev["ecr_pos_rank"])
    return out


def _adp_features(store, t: TargetRow, trace) -> dict[str, float]:
    out = {"adp_ppr": NAN, "adp_std": NAN, "adp_ppr_pos_rank": NAN, "adp_std_pos_rank": NAN}
    if not store.has("adp"):
        return out
    a = as_of_join(store, "adp", t.kickoff, player_id=t.player_id, trace=trace)
    a = a[a["season"] == t.season]
    for fmt, suffix in (("ppr", "ppr"), ("standard", "std")):
        x = a[a["format"] == fmt]
        if len(x):
            out[f"adp_{suffix}"] = _f(x["adp"].iloc[-1])
            out[f"adp_{suffix}_pos_rank"] = _f(x["pos_rank"].iloc[-1])
    return out


def fingerprint(row: dict) -> str:
    """Exact, order-stable text of a feature row: floats via repr (round-trips bits), NaN
    spelled out. Equal fingerprints == byte-identical features."""
    parts = []
    for k, v in row.items():
        if isinstance(v, float):
            v = "nan" if math.isnan(v) else repr(v)
        parts.append(f"{k}={type(v).__name__}:{v}")
    return "|".join(parts)
