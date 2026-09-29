"""First small feature builder: proves the gate, not feature completeness.

Every read of raw data goes through `as_of_join` (model/point_in_time.py); this module
must not import a loader, read a file, or touch a store's private frames. The AST test in
tests/test_no_leakage.py fails the build if it does.

Family definitions (all history is WITHIN the target's season in v1; cross-season lag and
prior-season priors are a Phase 1 decision):

  lags        last 1/2/3 games the player appeared in before the target game, and the
              week each came from (`lag{k}_week`, asserted < target week by the audit)
  dvp         mean PPR points the opponent allowed to the player's position over its last
              2 / 4 completed regular-season games (window ends at target week - 1 by
              construction: only games with known_at < kickoff exist)
  vegas       closing spread / total from the player's team's view, implied team total
  context     home/away, rest days, divisional, roof
  weather     OBSERVED temp/wind (v1 proxy for a forecast; NaN indoors). Declared in the
              weather_obs table's `proxy` field.
  injury      the most recent injury/practice designation known before kickoff and its age.
              `inj_days_since_report` uses known_at, which is a real timestamp <=2024 but derived
              (kickoff - 24h) from 2025, so its distribution shifts between train and test;
              `inj_weeks_since_report` is the timestamp-free twin Phase 1 should prefer.
  static      age, experience, draft capital (season-start known_at)
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from model.point_in_time import TargetRow, as_of_join

NAN = float("nan")
DAY = pd.Timedelta(days=1)
LISTED = ("Out", "Doubtful", "Questionable")


def _f(x: Any) -> float:
    try:
        return NAN if x is None or pd.isna(x) else float(x)
    except (TypeError, ValueError):
        return NAN


def _s(x: Any) -> Any:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) or x is pd.NA else x


def build_features(store, t: TargetRow, *, trace: list | None = None) -> dict:
    """Feature row for one target. Deterministic; NaN where history does not exist."""
    ko = t.kickoff
    out: dict[str, Any] = {
        "player_id": t.player_id, "season": t.season, "week": t.week, "team": t.team,
        "opponent": t.opponent, "position": t.position, "game_id": t.game_id,
        "kickoff_utc": ko.isoformat(),
    }

    # ---- lags: prior games of this player, oldest first ------------------------------
    pg = as_of_join(store, "player_games", ko, player_id=t.player_id, trace=trace)
    pg = pg[pg["season"] == t.season]
    sn = as_of_join(store, "snap_counts", ko, player_id=t.player_id, trace=trace)
    xf = as_of_join(store, "xfp", ko, player_id=t.player_id, trace=trace)
    snap_pct = dict(zip(sn["game_id"], sn["offense_pct"]))
    xfp = dict(zip(xf["game_id"], xf["total_fantasy_points_exp"]))
    for k in (1, 2, 3):
        g = pg.iloc[-k] if len(pg) >= k else None
        out[f"lag{k}_week"] = _s(int(g["week"])) if g is not None else None
        out[f"pts_ppr_l{k}"] = _f(g["fantasy_points_ppr"]) if g is not None else NAN
        out[f"carries_l{k}"] = _f(g["carries"]) if g is not None else NAN
        out[f"targets_l{k}"] = _f(g["targets"]) if g is not None else NAN
        out[f"pass_att_l{k}"] = _f(g["attempts"]) if g is not None else NAN
        out[f"target_share_l{k}"] = _f(g["target_share"]) if g is not None else NAN
        out[f"snap_pct_l{k}"] = _f(snap_pct.get(g["game_id"])) if g is not None else NAN
        out[f"xfp_l{k}"] = _f(xfp.get(g["game_id"])) if g is not None else NAN   # lagged xFP only
    out["weeks_since_last_game"] = int(t.week - pg.iloc[-1]["week"]) if len(pg) else None
    out["games_std"] = int(len(pg))
    out["pts_ppr_std_mean"] = _f(pg["fantasy_points_ppr"].mean()) if len(pg) else NAN

    # ---- opponent DvP vs position ----------------------------------------------------
    gr = as_of_join(store, "game_results", ko, team=t.opponent, trace=trace)
    gr = gr[(gr["season"] == t.season) & (gr["game_type"] == "REG")]
    allowed = as_of_join(store, "player_games", ko, opponent_team=t.opponent, position=t.position,
                         trace=trace)
    per_game = allowed.groupby("game_id")["fantasy_points_ppr"].sum()
    for n in (2, 4):
        last = gr.iloc[-n:]
        pts = [float(per_game.get(gid, 0.0)) for gid in last["game_id"]]   # 0 if position scored nothing
        out[f"dvp_ppr_l{n}"] = float(sum(pts) / len(pts)) if pts else NAN
        out[f"dvp_l{n}_games"] = int(len(last))
        out[f"dvp_l{n}_last_week"] = int(last["week"].max()) if len(last) else None

    # ---- Vegas + context -------------------------------------------------------------
    fx = as_of_join(store, "fixtures", ko, game_id=t.game_id, trace=trace).iloc[-1]
    ln = as_of_join(store, "lines", ko, game_id=t.game_id, trace=trace)
    spread_home = _f(ln.iloc[-1]["spread_line"]) if len(ln) else NAN     # + = home favored
    total = _f(ln.iloc[-1]["total_line"]) if len(ln) else NAN
    team_spread = spread_home if t.is_home else -spread_home             # + = this team favored
    out["team_spread"] = team_spread
    out["total_line"] = total
    out["implied_team_total"] = total / 2 + team_spread / 2 if not (math.isnan(total) or math.isnan(team_spread)) else NAN
    out["is_home"] = int(t.is_home)
    out["rest_days"] = _f(fx["home_rest"] if t.is_home else fx["away_rest"])
    out["div_game"] = int(fx["div_game"])
    out["neutral_site"] = int(fx["location"] == "Neutral")
    out["roof"] = _s(fx["roof"])

    # ---- weather (observed; v1 proxy) ------------------------------------------------
    wx = as_of_join(store, "weather_obs", ko, game_id=t.game_id, trace=trace)
    out["wx_temp_obs"] = _f(wx.iloc[-1]["temp"]) if len(wx) else NAN
    out["wx_wind_obs"] = _f(wx.iloc[-1]["wind"]) if len(wx) else NAN

    # ---- injury / practice designation -----------------------------------------------
    inj = as_of_join(store, "injuries", ko, player_id=t.player_id, trace=trace)
    inj = inj[inj["season"] == t.season]
    if len(inj):
        last = inj.iloc[-1]
        out["inj_report_status"] = _s(last["report_status"])
        out["inj_practice_status"] = _s(last["practice_status"])
        out["inj_last_report_week"] = int(last["week"])
        out["inj_days_since_report"] = float((ko - last["known_at"]) / DAY)   # skewed 2025+ (derived known_at)
        out["inj_weeks_since_report"] = int(t.week - last["week"])           # timestamp-free twin
        out["inj_listed_l4wk"] = int(((inj["week"] >= t.week - 4) & inj["report_status"].isin(LISTED)).sum())
    else:
        out.update(inj_report_status=None, inj_practice_status=None, inj_last_report_week=None,
                   inj_days_since_report=NAN, inj_weeks_since_report=None, inj_listed_l4wk=0)

    # ---- static (season-start known_at) ----------------------------------------------
    pl = as_of_join(store, "players_static", ko, player_id=t.player_id, trace=trace)
    dp = as_of_join(store, "draft_picks", ko, player_id=t.player_id, trace=trace)
    if len(pl):
        p = pl.iloc[-1]
        out["age_years"] = (ko.tz_convert(None) - pd.Timestamp(p["birth_date"])).total_seconds() / (365.25 * 86400)
        rookie = _f(p["rookie_season"])
        out["exp_years"] = float(t.season - rookie) if not math.isnan(rookie) else NAN
        out["is_rookie"] = int(rookie == t.season) if not math.isnan(rookie) else None
    else:
        out.update(age_years=NAN, exp_years=NAN, is_rookie=None)
    out["draft_round"] = _f(dp.iloc[-1]["round"]) if len(dp) else NAN
    out["draft_overall"] = _f(dp.iloc[-1]["pick"]) if len(dp) else NAN
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
