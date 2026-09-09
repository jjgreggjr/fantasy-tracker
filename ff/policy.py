"""Roster-construction policy: how much a player is worth KEEPING.

Separate from PlayScore, which answers "will he produce this week." This
answers "should he occupy a roster spot," which in a dynasty league is a
different question with a different time horizon.

Age curves below come from published positional aging research. The
consistent findings across sources:

  RB  peak 24-26 (recent studies put the average peak season at ~24.8);
      decline begins at 27; ~93% of peak seasons happen before age 29;
      the commonly cited sell window is 25-26, i.e. before the cliff.
  WR  breakout usually in years 2-4; peak 26-28; leaves peak at 29;
      steep decline at 32-33. Deep threats age worse than possession WRs.
  TE  slowest to develop and slowest to fall off; peak 26-29, holds
      production within ~10-15% of peak into the early 30s.
  QB  nearly flat from 27 to 35; decline is late, abrupt and variable.

Sources: 4for4 production curves (2025); Apex Fantasy Leagues RB/WR peak-age
studies; fantasyhistorydata.com age curves; Fantasy Footballers dynasty
lifecycle series. Numbers below are our encoding of that consensus, not any
single article's table, and are deliberately coarse — these are population
averages and individual players vary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import upside as upside_mod

# age -> remaining dynasty value (1.0 = full runway, 0.0 = no future value)
AGE_CURVES = {
    "RB": [(23, 1.00), (24, 1.00), (25, 0.88), (26, 0.72), (27, 0.52),
           (28, 0.32), (29, 0.18), (30, 0.10), (99, 0.05)],
    "WR": [(24, 1.00), (25, 1.00), (26, 0.95), (27, 0.90), (28, 0.82),
           (29, 0.68), (30, 0.55), (31, 0.42), (32, 0.28), (99, 0.12)],
    "TE": [(25, 1.00), (26, 0.95), (27, 0.92), (28, 0.90), (29, 0.85),
           (30, 0.75), (31, 0.62), (32, 0.48), (99, 0.28)],
    "QB": [(26, 1.00), (30, 0.97), (33, 0.92), (34, 0.82), (35, 0.68),
           (99, 0.45)],
}


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric column as a Series, or an all-NaN Series if it is absent.

    pd.to_numeric(df.get(missing)) returns a bare NaN scalar, not a Series,
    which breaks every comparison downstream.
    """
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def age_score(position: str, age: float) -> float:
    """Remaining dynasty runway, 0-1, from the position's aging curve."""
    if pd.isna(age) or position not in AGE_CURVES:
        return 0.6                      # unknown age: assume middling runway
    for cutoff, val in AGE_CURVES[position]:
        if age <= cutoff:
            return val
    return AGE_CURVES[position][-1][1]


def weights(meta: dict) -> dict:
    """How to trade off production now against runway later.

    Two things move the dial. Dynasty leagues care about the future at all;
    redraft leagues do not. And a deep bench is what makes stashing young
    players possible — with 16 bench spots you can afford to hold upside,
    with 3 you cannot, so bench depth raises the weight on age.
    """
    if meta.get("type") not in ("dynasty", "keeper"):
        return {"pts": 0.85, "age": 0.00, "trend": 0.15, "label": "redraft"}

    starting = max(len([p for p in (meta.get("roster_positions") or [])
                        if p not in ("BN", "TAXI", "IR")]), 1)
    bench = len([p for p in (meta.get("roster_positions") or []) if p == "BN"])
    ratio = bench / starting
    if ratio < 1.0:
        w = {"pts": 0.60, "age": 0.25, "trend": 0.15, "label": "shallow bench"}
    elif ratio <= 1.5:
        w = {"pts": 0.50, "age": 0.30, "trend": 0.20, "label": "normal bench"}
    else:
        w = {"pts": 0.40, "age": 0.40, "trend": 0.20, "label": "deep bench"}
    w["bench_ratio"] = round(ratio, 2)
    return w


def replacement_level(available: pd.DataFrame) -> dict:
    """Best freely available player at each position.

    A roster spot is only worth what it buys you over the waiver wire. Cutting
    the WR6 when twenty similar receivers are unowned costs almost nothing;
    cutting your second startable TE can cost a lot even at the same E_pts.
    """
    if available is None or available.empty:
        return {}
    a = available[available.E_pts.notna()]
    if a.empty:
        return {}
    return a.groupby("position").E_pts.max().to_dict()


def keep_value(roster: pd.DataFrame, available: pd.DataFrame,
               meta: dict) -> pd.DataFrame:
    """Add keep_value and its components to a scored roster."""
    r = roster.copy()
    w = weights(meta)
    repl = replacement_level(available)

    r["replacement_pts"] = r.position.map(repl)
    # For roster decisions use keep_pts, not E_pts: a player who cannot play
    # THIS week still occupies a spot for a reason, and scoring him as zero
    # would punish him for an injury we have already accounted for.
    kp = [upside_mod.keep_points(x) for _, x in r.iterrows()]
    r["keep_pts"] = [v for v, _ in kp]
    r["keep_pts_basis"] = [b for _, b in kp]
    vor = (r.keep_pts - r.replacement_pts)
    r["vor"] = vor.round(1)
    span = max(vor.max() - min(vor.min(), 0), 1e-6)
    r["pts_score"] = ((vor - min(vor.min(), 0)) / span).clip(0, 1).fillna(0)

    r["age_score"] = [age_score(p, a) for p, a in zip(r.position, r.age)]

    r["trend_score"] = (_num(r, "trend_opps").fillna(0) / 4 + 0.5).clip(0, 1)

    r["keep_value"] = (w["pts"] * r.pts_score
                       + w["age"] * r.age_score
                       + w["trend"] * r.trend_score).round(3)

    # A young player who is not yet producing is a stash, not a cut — but only
    # if he has a PATH to snaps. "Young" alone protects fourth-stringers buried
    # behind three players, which is exactly the roster bloat we are trying to
    # clear. Require youth AND at least one credible route to a role.
    young = ((r.position.isin(["RB", "WR", "TE"]))
             & (pd.to_numeric(r.get("years_exp"), errors="coerce") <= 2)
             & (pd.to_numeric(r.age, errors="coerce") <= 24))
    depth = _num(r, "depth_rank")
    trend = _num(r, "trend_opps").fillna(0)
    proj = _num(r, "proj_opps").fillna(0)
    draft = _num(r, "draft_number")

    # A depth chart lists an injured player last, so it cannot be the only
    # evidence of role. "He has held a real role recently" rescues players who
    # are buried on today's chart purely because they are hurt.
    peak = _num(r, "peak_snap_3g")
    role_opps = _num(r, "role_opps")
    held_role = (peak.ge(0.40) | role_opps.ge(8)).fillna(False)

    path = (
        depth.le(2)                                   # starter or direct backup
        | (depth.eq(3) & (trend > 0))                 # third string but rising
        | (proj >= 5)                                 # already projected a role
        | draft.le(64)                                # day-1/2 draft capital
        | held_role                                   # has actually played the role
    ).fillna(False)
    r["held_role"] = held_role.astype(int)
    r["has_path"] = path.astype(int)
    r["stash"] = (young & path).astype(int)
    r["buried"] = (young & ~path).astype(int)
    r.attrs["weights"] = w
    return r


def trade_interest(row: pd.Series, all_rosters: pd.DataFrame,
                   depth: pd.DataFrame, my_owner: str) -> str:
    """Who else in the league would plausibly want this player.

    Two concrete cases, both checkable from data we already hold: he is the
    direct backup to somebody else's starter (handcuff value), or an owner is
    thin at his position.
    """
    if all_rosters is None or all_rosters.empty or depth is None or depth.empty:
        return ""
    bits = []
    dr = row.get("depth_rank")
    if pd.notna(dr) and dr >= 2:
        ahead = depth[(depth.team == row.team) & (depth.position == row.position)
                      & (depth["rank"] == dr - 1)]
        if not ahead.empty:
            starter = ahead.iloc[0]["name"]
            # depth charts and rosters disagree on suffixes ("Travis Etienne"
            # vs "Travis Etienne Jr."), so match on the normalized name
            from .build import name_key
            key = name_key(starter)
            own = all_rosters[all_rosters["name"].map(name_key) == key]
            if not own.empty and own.iloc[0].get("owner_name") != my_owner:
                bits.append(f"handcuff to {starter}, owned by "
                            f"{own.iloc[0]['owner_name']}")
    pos_rosters = all_rosters[all_rosters.position == row.position]
    if not pos_rosters.empty and "E_pts" in pos_rosters.columns:
        startable = (pos_rosters[pos_rosters.E_pts > 8]
                     .groupby("owner_name").size())
        thin = [o for o, c in startable.items()
                if c <= 3 and o != my_owner]
        if thin:
            bits.append(f"thin at {row.position}: {', '.join(sorted(thin)[:3])}")
    return " · ".join(bits)

def explain_keep(row: pd.Series, w: dict) -> str:
    bits = []
    if pd.notna(row.get("vor")):
        basis = row.get("keep_pts_basis", "")
        tag = f" ({basis})" if basis and basis != "this week" else ""
        bits.append(f"{row.vor:+.1f} pts vs best free agent at "
                    f"{row.position}{tag}")
    if pd.notna(row.get("peak_snap_3g")) and row.get("held_role"):
        bits.append(f"held a {row.peak_snap_3g:.0%} snap role "
                    f"({row.role_opps:.0f} opps/g)")
    if pd.notna(row.get("age")):
        bits.append(f"age {row.age:.0f} → {row.age_score:.0%} runway")
    if row.get("stash"):
        bits.append("young with a path — protected")
    elif row.get("buried"):
        dr = row.get("depth_rank")
        bits.append(f"young but buried at {row.position}"
                    + (f"{int(dr)}" if pd.notna(dr) else " (not on depth chart)"))
    f = row.get("flag")
    if pd.notna(f) and str(f).strip():
        bits.append(str(f))
    return " · ".join(bits)
