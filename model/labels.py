"""Target and baseline columns: the ONLY place besides `make_target`/`team_games` that reads target-week rows.

Nothing here is a feature and nothing in features.py may import it (the AST test enforces that).
Columns it adds to the matrix:

  y_played          1 if the player appeared in the game: a stats row OR offense_snaps > 0
  y_has_stats_row   1 if nflverse has a stats row for him in that game (snap-only players do not)
  y_offense_snaps   PFR offensive snaps (NaN if no snap row)
  y_points_ppr      the target: PPR points; 0.0 for a snap-only appearance; NaN for a DNP
  base_xfp_sameweek same-week ffopportunity xFP (a baseline to BEAT, post-game information by construction)
  base_trail3_ppr   mean PPR over his last <=3 appearances this season (from the lag features)

Philosophy (matches the tracker): the model predicts points-if-he-plays; whether he plays is a separate
concern, so Phase 2 trains on y_played == 1 rows and the availability layer handles the rest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model.point_in_time import RawStore


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
    out["y_played"] = played.astype("int8")
    out["y_has_stats_row"] = has_stats.astype("int8")
    out["y_offense_snaps"] = snap_v
    out["y_points_ppr"] = y
    out["base_xfp_sameweek"] = np.where(played, x, np.nan)
    out["base_trail3_ppr"] = out[["pts_ppr_l1", "pts_ppr_l2", "pts_ppr_l3"]].mean(axis=1, skipna=True)
    return out
