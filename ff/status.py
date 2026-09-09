"""Player status, and — critically — the status of whoever is ahead of him.

The Charbonnet miss was not a math error. He was listed RB4 because he was on
PUP, and nothing in the pipeline asked *why* the depth chart said what it said.
This module makes that question mandatory: for every player we name, it reports
his own availability and the availability of everyone listed above him.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

OUT_STATES = {"Out", "IR", "PUP", "Sus", "NA", "DNR"}
DOUBTFUL = {"Doubtful"}
SOFT = {"Questionable"}

# practice participation -> how much it tells you about Sunday
PRACTICE_WEIGHT = {"DNP": 0.15, "Did Not Participate In Practice": 0.15,
                   "Limited": 0.65, "Limited Participation in Practice": 0.65,
                   "Full": 0.95, "Full Participation in Practice": 0.95}


def status_flag(row: pd.Series) -> str:
    s = row.get("injury_status")
    if pd.isna(s) or not str(s).strip():
        return "healthy"
    s = str(s)
    if s in OUT_STATES:
        return s.lower()
    if s in DOUBTFUL:
        return "doubtful"
    if s in SOFT:
        return "questionable"
    return s.lower()


def play_probability(row: pd.Series) -> float:
    """Rough chance he suits up, from designation plus practice participation.

    Practice is the stronger signal: a Questionable player who practiced fully
    on Friday is a very different bet from one who did not practice at all.
    """
    flag = status_flag(row)
    base = {"healthy": 0.97, "questionable": 0.72, "doubtful": 0.28}.get(flag, 0.02)
    pr = row.get("practice_participation")
    if pd.notna(pr) and str(pr) in PRACTICE_WEIGHT:
        # let practice pull the estimate, but never override a hard OUT
        if flag in ("healthy", "questionable", "doubtful"):
            base = 0.4 * base + 0.6 * PRACTICE_WEIGHT[str(pr)]
    return round(float(base), 2)


def _news_age_hours(v) -> float:
    if pd.isna(v):
        return np.nan
    try:
        ts = float(v)
    except (TypeError, ValueError):
        return np.nan
    if ts > 1e12:          # Sleeper sends milliseconds
        ts /= 1000.0
    age = (datetime.now(timezone.utc) - datetime.fromtimestamp(ts, timezone.utc))
    return round(age.total_seconds() / 3600, 1)


def build(players: pd.DataFrame, depth: pd.DataFrame,
          proj: pd.DataFrame | None = None, week: int | None = None
          ) -> pd.DataFrame:
    """One status row per player on a depth chart, including the blocker chain."""
    if depth is None or depth.empty:
        return pd.DataFrame()

    p = players.copy()
    p["status_flag"] = p.apply(status_flag, axis=1)
    p["play_prob"] = p.apply(play_probability, axis=1)
    p["news_age_hours"] = p.get("news_updated", pd.Series(index=p.index)).map(_news_age_hours)

    d = depth.merge(
        p[["gsis_id", "status_flag", "play_prob", "news_age_hours",
           "injury_body_part", "practice_participation", "sleeper_depth_order"]],
        on="gsis_id", how="left")
    # A player on a depth chart with no match in players.csv has unknown
    # availability, which is not the same as healthy — say so rather than
    # letting NaN propagate into the blocker chain.
    d["status_flag"] = d.status_flag.fillna("unknown")
    d["play_prob"] = pd.to_numeric(d.play_prob, errors="coerce").fillna(0.85)

    # two independent depth charts: ESPN (via nflverse) and Sleeper's own.
    # A big disagreement is the signature of a rank driven by availability
    # rather than by role — exactly the case that misled us.
    sd = pd.to_numeric(d.get("sleeper_depth_order"), errors="coerce")
    d["depth_disagreement"] = (pd.to_numeric(d["rank"], errors="coerce") - sd).abs()

    share = None
    if proj is not None and not proj.empty and week is not None:
        pw = proj[proj.week == week]
        share = pw.set_index("gsis_id")[["proj_carry_share", "proj_target_share"]]

    rows = []
    for (team, pos), g in d.groupby(["team", "position"]):
        g = g.sort_values("rank")
        for _, r in g.iterrows():
            ahead = g[g["rank"] < r["rank"]]
            blockers = [{"name": b["name"], "rank": int(b["rank"]),
                         "status": b.status_flag, "play_prob": b.play_prob,
                         "practice": b.practice_participation}
                        for _, b in ahead.iterrows()]
            # volume held by players ahead of him who probably will not play
            opp_ahead = 0.0
            if share is not None:
                col = "proj_carry_share" if pos == "RB" else "proj_target_share"
                for _, b in ahead.iterrows():
                    if b.gsis_id in share.index and pd.notna(b.play_prob):
                        v = share.loc[b.gsis_id, col]
                        if pd.notna(v):
                            opp_ahead += float(v) * (1 - float(b.play_prob))
            rows.append({
                "gsis_id": r.gsis_id, "name": r["name"], "team": team,
                "position": pos, "espn_rank": r["rank"],
                "sleeper_rank": r.sleeper_depth_order,
                "depth_disagreement": r.depth_disagreement,
                "status_flag": r.status_flag, "play_prob": r.play_prob,
                "injury_body_part": r.injury_body_part,
                "practice": r.practice_participation,
                "news_age_hours": r.news_age_hours,
                "n_blockers": len(blockers),
                "blockers_out": sum(1 for b in blockers
                                    if b["play_prob"] is not None
                                    and pd.notna(b["play_prob"])
                                    and b["play_prob"] < 0.5),
                "opportunity_ahead": round(opp_ahead, 3),
                "blocked_by": "; ".join(
                    f"{b['name']} ({b['status']}"
                    + (f", {b['practice']}"
                       if b["practice"] is not None and pd.notna(b["practice"])
                       else "")
                    + ")" for b in blockers) or "nobody",
            })
    return pd.DataFrame(rows)


def warnings(st: pd.DataFrame, roster_ids: set | None = None) -> list[str]:
    """Things a human should look at before trusting a recommendation."""
    if st is None or st.empty:
        return []
    s = st if roster_ids is None else st[st.gsis_id.isin(roster_ids)]
    out = []
    dis = s[s.depth_disagreement >= 2]
    for _, r in dis.iterrows():
        out.append(f"{r['name']}: depth charts disagree (ESPN {int(r.espn_rank)} "
                   f"vs Sleeper {int(r.sleeper_rank)}) — his listed rank may "
                   f"reflect availability, not role")
    opp = s[s.opportunity_ahead >= 0.15]
    for _, r in opp.iterrows():
        out.append(f"{r['name']}: {r.opportunity_ahead:.0%} of the volume ahead "
                   f"of him belongs to players unlikely to play ({r.blocked_by})")
    hurt = s[(s.status_flag.isin(["out", "ir", "pup", "doubtful"]))]
    for _, r in hurt.iterrows():
        bp = f" ({r.injury_body_part})" if pd.notna(r.injury_body_part) else ""
        out.append(f"{r['name']}: {r.status_flag.upper()}{bp} — his depth rank "
                   f"of {int(r.espn_rank)} reflects that, not his role")
    return out


def describe(st: pd.DataFrame, name: str) -> str:
    """One-line status summary for a named player, for use in any answer."""
    if st is None or st.empty:
        return ""
    hit = st[st["name"].str.contains(name, case=False, na=False)]
    if hit.empty:
        return ""
    r = hit.iloc[0]
    bits = [f"{r.status_flag}"]
    if pd.notna(r.practice):
        bits.append(f"practice: {r.practice}")
    if pd.notna(r.injury_body_part):
        bits.append(str(r.injury_body_part))
    bits.append(f"{r.position}{int(r.espn_rank)}, behind: {r.blocked_by}")
    if r.opportunity_ahead >= 0.10:
        bits.append(f"{r.opportunity_ahead:.0%} of volume ahead of him is in doubt")
    return " · ".join(bits)
