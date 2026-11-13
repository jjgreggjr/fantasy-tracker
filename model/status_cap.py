"""Status-layer cap on served points (rule-based; pandas/numpy).

The tree model predicts "points if he plays" from what it sees, and cannot see that every player ahead of a deep-bench player on
the depth chart is back healthy (2026 week 9: Emanuel Wilson, SEA RB4, model 4.4 and p90 10.6 vs E_pts 0.8, Charbonnet back). data/status.csv
(`ff.status.build`) does see it. A row is ELIGIBLE when it has `n_blockers >= 1`, `blockers_out == 0`, `opportunity_ahead <=
CAP_MAX_OPP_AHEAD`, and every blocker in `blocked_by` ("Name (status[, practice]); ...") is healthy. An eligible row's points are
capped, per league, at `max(e_pts_<slug>, sleeper_proj_<slug>) * CAP_SLACK`; the PPR columns (`pts_model`, `pts_model_components`)
use the first PPR league's E_pts and `sleeper_proj`. Numbers are never raised.

Adds `status_cap` (1 if any column of the row was lowered) and `pts_model_components_raw` (the PPR components number before the cap).
"""
from __future__ import annotations

CAP_SLACK = 1.25               # cap = reference * slack
CAP_MAX_OPP_AHEAD = 0.05       # eligible only when status.csv `opportunity_ahead` is at or below this
STATUS_COLUMNS = ("n_blockers", "blockers_out", "opportunity_ahead", "blocked_by")


def all_healthy(blocked_by) -> bool:
    """True when the string names blockers and every "(" group starts with "healthy". "nobody" / blank / malformed: False."""
    if not isinstance(blocked_by, str) or "(" not in blocked_by:
        return False
    return all(g.strip().lower().startswith("healthy") for g in blocked_by.split("(")[1:])


def eligible_ids(status, max_opportunity_ahead=CAP_MAX_OPP_AHEAD) -> set:
    import pandas as pd
    if status is None or any(c not in status.columns for c in ("gsis_id",) + STATUS_COLUMNS):
        return set()
    s = status.drop_duplicates("gsis_id")
    nb = pd.to_numeric(s["n_blockers"], errors="coerce")
    bo = pd.to_numeric(s["blockers_out"], errors="coerce")
    oa = pd.to_numeric(s["opportunity_ahead"], errors="coerce")
    ok = (nb >= 1) & (bo == 0) & (oa <= max_opportunity_ahead) & s["blocked_by"].map(all_healthy)
    return set(s.loc[ok.fillna(False), "gsis_id"])


def apply_cap(out, status, leagues, *, slack=CAP_SLACK, max_opportunity_ahead=CAP_MAX_OPP_AHEAD):
    """Returns (out, info). `out` is modified in place and returned; see the module docstring."""
    import numpy as np
    out["pts_model_components_raw"] = out["pts_model_components"] if "pts_model_components" in out else np.nan
    out["status_cap"] = 0
    info = {"eligible": 0, "capped": 0, "names": []}
    ids = eligible_ids(status, max_opportunity_ahead)
    if not ids:
        return out, info
    elig = out["gsis_id"].isin(ids).to_numpy()
    info["eligible"] = int(elig.sum())
    if not elig.any():
        return out, info

    def col(name):
        return out[name].astype("float64").to_numpy() if name in out else np.full(len(out), np.nan)

    def ref(e, s):
        return np.fmax(col(e), col(s))                             # nan-aware: the other one if one is NaN, NaN if both

    targets = [(f"pts_{ls.slug}", ref(f"e_pts_{ls.slug}", f"sleeper_proj_{ls.slug}")) for ls in leagues]
    ppr = next((ls for ls in leagues if ls.is_ppr), None)
    rp = ref(f"e_pts_{ppr.slug}" if ppr is not None else "__none__", "sleeper_proj")
    targets += [("pts_model_components", rp), ("pts_model", rp)]

    lowered = np.zeros(len(out), dtype=bool)
    for name, r in targets:
        if name not in out:
            continue
        cur = col(name)
        cap = r * slack
        hit = elig & ~np.isnan(cap) & (cur > cap)
        if hit.any():
            cur = cur.copy()
            cur[hit] = np.round(cap[hit], 3) + 0.0
            out[name] = cur
            lowered |= hit
    out["status_cap"] = lowered.astype(int)
    info["capped"] = int(lowered.sum())
    raw, now = col("pts_model_components_raw"), col("pts_model_components")
    for i in np.flatnonzero(lowered)[:10]:
        row = out.iloc[i]
        info["names"].append(f"{row.get('name', row['gsis_id'])} {row.get('team', '')} {row.get('position', '')} {raw[i]:.1f}->{now[i]:.1f}")
    return out, info
