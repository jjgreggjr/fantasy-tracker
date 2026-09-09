"""What a player has shown when healthy — independent of today's status.

Current depth-chart rank and this week's projection both collapse for an
injured player: he is listed last and projected zero. Using those as evidence
of his role punishes him for being hurt, and does it several times over.

These metrics look at what he actually did when he was on the field, which is
the relevant question for whether he is worth a roster spot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROLE_SNAP = 0.40      # snap share at which a player is genuinely in the rotation


def _roll_best(vals: list[float], window: int = 3) -> float:
    v = [x for x in vals if pd.notna(x)]
    if not v:
        return np.nan
    if len(v) < window:
        return float(np.mean(v))
    return float(max(np.mean(v[i:i + window]) for i in range(len(v) - window + 1)))


def player_upside(pw: pd.DataFrame, points: pd.DataFrame | None = None
                  ) -> pd.DataFrame:
    """Per player: peak role held, opportunity when in that role, and ceiling."""
    if pw is None or pw.empty:
        return pd.DataFrame(columns=["gsis_id"])
    d = pw.sort_values(["season", "week"]).copy()
    pts = None
    if points is not None and not points.empty:
        pts = points[["season", "week", "gsis_id", "pts"]]
        d = d.merge(pts, on=["season", "week", "gsis_id"], how="left")

    rows = []
    for gid, g in d.groupby("gsis_id"):
        snaps = g.snap_pct.tolist()
        opps = g.opps.tolist()
        role = g[g.snap_pct >= ROLE_SNAP]
        p = g["pts"] if "pts" in g.columns else pd.Series(dtype=float)
        rows.append({
            "gsis_id": gid,
            "peak_snap_3g": _roll_best(snaps),
            "peak_opps_3g": _roll_best(opps),
            "role_games": int(len(role)),
            "role_opps": round(role.opps.mean(), 1) if len(role) else np.nan,
            "role_snap": round(role.snap_pct.mean(), 3) if len(role) else np.nan,
            "ceiling_pts": (round(float(p.quantile(0.90)), 1)
                            if p.notna().any() else np.nan),
            "median_pts": (round(float(p.median()), 1)
                           if p.notna().any() else np.nan),
        })
    out = pd.DataFrame(rows)
    for c in ("peak_snap_3g", "peak_opps_3g"):
        out[c] = out[c].round(3)
    return out


def keep_points(row: pd.Series) -> tuple[float, str]:
    """The points figure to use for ROSTER decisions, not lineup decisions.

    A player who cannot play this week has no E_pts — correctly, for setting a
    lineup. But "he is hurt this week" is not the same as "he is not worth a
    roster spot," so for keep-value we fall back to what he produced in the
    role he actually held, discounted for the uncertainty of a return.
    """
    if pd.notna(row.get("E_pts")):
        return float(row.E_pts), "this week"
    if pd.notna(row.get("median_pts")):
        return float(row.median_pts) * 0.85, "healthy baseline"
    if pd.notna(row.get("trail_pts")):
        return float(row.trail_pts) * 0.85, "recent form"
    return np.nan, "no data"
