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
}
# Produced, but never a model input.
IDENTITY = ["player_id", "season", "week", "team", "opponent", "position", "game_id", "kickoff_utc"]
META = ["lag1_week", "lag2_week", "lag3_week", "dvp_l2_games", "dvp_l4_games", "dvp_l2_last_week",
        "dvp_l4_last_week", "inj_last_report_week", "spine_src"]
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
# value-matrix columns of Hist.v
PTS, OPPS, CAR, TGT, ATT, TSH, CSH, SNAP, XFP, TDR, TDC, TDP = range(12)


class Hist:
    """Every game the player appeared in with known_at < kickoff, all seasons in the store, oldest first."""
    __slots__ = ("season", "week", "is_reg", "has_stats", "game_id", "v")

    def __init__(self, rows: list):
        self.game_id = [r[0] for r in rows]
        self.season = np.array([r[1] for r in rows], dtype="int64")
        self.week = np.array([r[2] for r in rows], dtype="int64")
        self.is_reg = np.array([r[3] for r in rows], dtype=bool)
        self.has_stats = np.array([r[4] for r in rows], dtype=bool)
        self.v = np.array([r[5:] for r in rows], dtype="float64").reshape(len(rows), 12)

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
        for gid, season, week, stype, ppr, car, tgt, att_, tsh, csh in _cols(
                pg, "game_id", "season", "week", "season_type", "fantasy_points_ppr", "carries", "targets",
                "attempts", "target_share", "carry_share"):
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
            rows.append((gid, int(season), int(week), stype == "REG", True, _f(ppr), opps, carries, targets, att,
                         _f(tsh), _f(csh), _f(snap.get(gid)), xfp, tdr, tdc, tdp))
        seen = set(pg["game_id"])
        only = sn[(sn["offense_snaps"] > 0) & sn["position"].isin(POSITIONS) & ~sn["game_id"].isin(seen)]
        for gid, season, week, gtype, pct in _cols(only, "game_id", "season", "week", "game_type", "offense_pct"):
            xfp = _f(xfp_by[gid]) if gid in xfp_by else 0.0
            rows.append((gid, int(season), int(week), gtype == "REG", False, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, _f(pct), xfp, 0.0, 0.0, 0.0))
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

    # ---- role + teammate-injury context ----------------------------------------------
    out.update(_role_features(ctx, h, t))

    # ---- opponent DvP vs position ----------------------------------------------------
    out.update(ctx["dvp"].get(t.position) or {k: (NAN if "ppr" in k else None) for k in ctx["dvp"]["QB"]})

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
