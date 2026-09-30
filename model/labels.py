"""Target and baseline columns: the ONLY place besides `make_target`/`team_games` that reads target-week rows.

Nothing here is a feature and nothing in features.py may import it (the AST test enforces that).
Columns it adds to the matrix:

  y_played          1 if the player appeared in the game: a stats row OR offense_snaps > 0
  y_has_stats_row   1 if nflverse has a stats row for him in that game (snap-only players do not)
  y_offense_snaps   PFR offensive snaps (NaN if no snap row)
  y_points_ppr      the target: PPR points; 0.0 for a snap-only appearance; NaN for a DNP
  base_xfp_sameweek same-week ffopportunity xFP (a baseline to BEAT, post-game information by construction)
  base_trail3_ppr   mean PPR over his last <=3 appearances this season (from the lag features)
  y_<component>     Phase 2.5: the scoring components behind y_points_ppr (see COMPONENTS): targets, receptions, receiving
                    yards and TDs, carries, rushing yards and TDs, pass attempts / completions / yards / TDs / INTs, fumbles
                    lost, two-point conversions, special-teams TDs. Same rows as y_points_ppr: NaN for a DNP, 0 for a
                    snap-only appearance. `ppr_from_components` rebuilds y_points_ppr from them exactly.

Philosophy (matches the tracker): the model predicts points-if-he-plays; whether he plays is a separate
concern, so Phase 2 trains on y_played == 1 rows and the availability layer handles the rest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model.point_in_time import RawStore

# label column -> player_games column. The PPR weights in `PPR_WEIGHTS` reproduce nflverse's fantasy_points_ppr exactly.
COMPONENTS = {"y_tgt": "targets", "y_rec": "receptions", "y_rec_yds": "receiving_yards", "y_rec_td": "receiving_tds",
              "y_car": "carries", "y_rush_yds": "rushing_yards", "y_rush_td": "rushing_tds", "y_att": "attempts",
              "y_cmp": "completions", "y_pass_yds": "passing_yards", "y_pass_td": "passing_tds",
              "y_int": "passing_interceptions", "y_fum_lost": "fumbles_lost", "y_two_pt": "two_pt_conversions",
              "y_st_td": "special_teams_tds"}
PPR_WEIGHTS = {"y_pass_yds": 0.04, "y_pass_td": 4.0, "y_int": -2.0, "y_rush_yds": 0.1, "y_rush_td": 6.0, "y_rec_yds": 0.1,
               "y_rec_td": 6.0, "y_rec": 1.0, "y_fum_lost": -2.0, "y_two_pt": 2.0, "y_st_td": 6.0}


def ppr_from_components(df: pd.DataFrame) -> pd.Series:
    """PPR points rebuilt from the y_<component> columns (nflverse weights); NaN where a component is NaN (a DNP)."""
    return sum(w * df[c] for c, w in PPR_WEIGHTS.items())


def attach_labels(store: RawStore, df: pd.DataFrame) -> pd.DataFrame:
    pg = store._tables["player_games"].df
    sn = store._tables["snap_counts"].df
    xf = store._tables["xfp"].df
    key = list(zip(df["player_id"], df["game_id"]))

    stats = pg.drop_duplicates(["player_id", "game_id"], keep="last").set_index(["player_id", "game_id"])
    pts = stats["fantasy_points_ppr"].to_dict()
    opp = (stats["carries"].fillna(0) + stats["targets"].fillna(0) + stats["attempts"].fillna(0)).to_dict()
    snaps = sn.drop_duplicates(["player_id", "game_id"], keep="last").set_index(["player_id", "game_id"])["offense_snaps"].to_dict()
    xfp = xf.drop_duplicates(["player_id", "game_id"], keep="last").set_index(["player_id", "game_id"])["total_fantasy_points_exp"].to_dict()

    has_stats = np.array([k in pts for k in key])
    snap_v = np.array([snaps.get(k, np.nan) for k in key], dtype="float64")
    played = has_stats | (snap_v > 0)
    y = np.array([pts.get(k, np.nan) for k in key], dtype="float64")
    y = np.where(has_stats, y, np.where(played, 0.0, np.nan))
    x = np.array([xfp.get(k, np.nan) for k in key], dtype="float64")
    zero_opp = np.array([opp.get(k, np.nan) == 0 for k in key])
    # no ffopportunity row: 0 for someone who played without an opportunity (that is who it omits), NaN for a DNP
    x = np.where(np.isnan(x) & played & (zero_opp | ~has_stats), 0.0, x)

    out = df.copy()
    for lab, col in COMPONENTS.items():
        val = stats[col].fillna(0.0).to_dict()
        v = np.array([val.get(k, np.nan) for k in key], dtype="float64")
        out[lab] = np.where(has_stats, v, np.where(played, 0.0, np.nan))
    out["y_played"] = played.astype("int8")
    out["y_has_stats_row"] = has_stats.astype("int8")
    out["y_offense_snaps"] = snap_v
    out["y_points_ppr"] = y
    out["base_xfp_sameweek"] = np.where(played, x, np.nan)
    out["base_trail3_ppr"] = out[["pts_ppr_l1", "pts_ppr_l2", "pts_ppr_l3"]].mean(axis=1, skipna=True)
    return out
