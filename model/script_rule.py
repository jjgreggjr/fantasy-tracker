"""Fit the `script` family's expected-plays rule (features.SCRIPT_PLAYS_RULE).

    python -m model.script_rule            # 2021-2024 regular-season team-games; prints the coefficients to paste

    plays = a + b * trailing plays/game (last 4 REG games, through the gate) + c * |spread| + d * total

One row per (team, game) with a trailing window and a closing line; the target is the team's own scrimmage plays in that game
(pbp_team, the label, read here only). The trailing input is computed by the same feature code the matrix uses, so the rule
is fitted on exactly what it will be applied to. The constants are fixed in features.py; nothing refits them in a backtest.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from model import features as F
from model import point_in_time as pit

FIT_SEASONS = (2021, 2022, 2023, 2024)
TERMS = ("intercept", "trail_plays", "abs_spread", "total")


def fit(frame: pd.DataFrame) -> dict[str, float]:
    """OLS of `plays` on trail_plays, |spread| and total. `frame` columns: plays, trail_plays, spread, total."""
    d = frame.dropna(subset=["plays", "trail_plays", "spread", "total"])
    X = np.column_stack([np.ones(len(d)), d["trail_plays"], d["spread"].abs(), d["total"]])
    beta, *_ = np.linalg.lstsq(X, d["plays"].to_numpy(dtype="float64"), rcond=None)
    return {k: float(v) for k, v in zip(TERMS, beta)}


def team_game_frame(store: pit.RawStore, seasons=FIT_SEASONS) -> pd.DataFrame:
    """(game, team) rows: actual plays (label), and the feature code's trailing plays, spread and total."""
    tm = store._tables["pbp_team"].df.set_index(["game_id", "team"])["plays"]
    rows = []
    for tg in pit.team_games(store, seasons):
        vegas = F._team_context(store, tg, None)["vegas"]
        sc = F._script_team(store, tg, vegas, None)
        rows.append({"season": tg.season, "game_id": tg.game_id, "team": tg.team,
                     "plays": float(tm.get((tg.game_id, tg.team), math.nan)), "trail_plays": sc["_trail_plays"],
                     "spread": vegas["team_spread"], "total": vegas["total_line"]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    store = pit.load_store(range(pit.CAREER_CUTOFF_SEASON, max(FIT_SEASONS) + 1), adp=False, college=False)
    df = team_game_frame(store)
    d = df.dropna()
    rule = fit(d)
    pred = rule["intercept"] + rule["trail_plays"] * d["trail_plays"] + rule["abs_spread"] * d["spread"].abs() + rule["total"] * d["total"]
    rmse = float(np.sqrt(((d["plays"] - pred) ** 2).mean()))
    naive = float(np.sqrt(((d["plays"] - d["trail_plays"]) ** 2).mean()))
    const = float(d["plays"].std(ddof=0))
    print(f"{len(d)} team-games ({len(df) - len(d)} without a window or a line)")
    print("SCRIPT_PLAYS_RULE = {" + ", ".join(f'"{k}": {v:.4f}' for k, v in rule.items()) + "}")
    print(f"RMSE: rule {rmse:.3f}, trailing-4 alone {naive:.3f}, constant {const:.3f}")
    signed = d.assign(abs_spread=d["spread"].abs())
    X = np.column_stack([np.ones(len(d)), d["trail_plays"], signed["abs_spread"], d["total"], d["spread"]])
    b, *_ = np.linalg.lstsq(X, d["plays"].to_numpy(), rcond=None)
    print(f"diagnostic, with signed spread too: {np.round(b, 4).tolist()} (intercept, trail, |spread|, total, spread)")


if __name__ == "__main__":
    main()
