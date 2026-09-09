"""Weekly projections.

Primary source is Sleeper, which publishes a projected stat line per player
per week on the same free API we already use. Field names are not formally
documented, so every key we read is resolved through an alias list and the
keys we actually saw are logged — if Sleeper renames something, the log says
so instead of the column silently going to zero.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE = "https://api.sleeper.com/projections/nfl"
UA = {"User-Agent": "ff-volume-tracker/1.0 (personal use)"}
POSITIONS = ["QB", "RB", "WR", "TE"]

# our column -> possible keys in Sleeper's stats blob, in preference order
ALIASES = {
    "proj_pass_att": ["pass_att"],
    "proj_pass_yds": ["pass_yd", "pass_yds"],
    "proj_pass_tds": ["pass_td", "pass_tds"],
    "proj_pass_int": ["pass_int"],
    "proj_carries": ["rush_att", "rush_atts", "carries"],
    "proj_rush_yds": ["rush_yd", "rush_yds"],
    "proj_rush_tds": ["rush_td", "rush_tds"],
    "proj_targets": ["rec_tgt", "targets", "tgt"],
    "proj_rec": ["rec", "receptions"],
    "proj_rec_yds": ["rec_yd", "rec_yds"],
    "proj_rec_tds": ["rec_td", "rec_tds"],
    "proj_pts_ppr": ["pts_ppr"],
    "proj_pts_half": ["pts_half_ppr"],
    "proj_pts_std": ["pts_std"],
}


def _pick(stats: dict, names: list[str]) -> float:
    for n in names:
        if n in stats and stats[n] is not None:
            try:
                return float(stats[n])
            except (TypeError, ValueError):
                continue
    return np.nan


def fetch_sleeper(season: int, week: int, raw_dir: Path, today: str) -> list[dict]:
    """One call per week, cached to disk by date."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / f"projections_{season}_wk{week:02d}_{today}.json"
    if cache.exists():
        log.info("using cached Sleeper projections for %s wk%s", season, week)
        return json.loads(cache.read_text(encoding="utf-8"))

    params = [("season_type", "regular"), ("order_by", "pts_ppr")]
    params += [("position[]", p) for p in POSITIONS]
    r = requests.get(f"{BASE}/{season}/{week}", params=params, headers=UA, timeout=120)
    r.raise_for_status()
    data = r.json()
    cache.write_text(json.dumps(data), encoding="utf-8")
    for old in sorted(raw_dir.glob(f"projections_{season}_wk{week:02d}_*.json"))[:-2]:
        old.unlink()
    return data


def build(season: int, week: int, players: pd.DataFrame, raw_dir: Path,
          today: str) -> pd.DataFrame:
    """Return projections.csv rows for one week from Sleeper."""
    try:
        raw = fetch_sleeper(season, week, raw_dir, today)
    except Exception as e:
        log.warning("Sleeper projections unavailable for wk%s: %s", week, e)
        return pd.DataFrame()
    if not raw:
        log.warning("Sleeper returned no projections for wk%s", week)
        return pd.DataFrame()

    seen_keys: set = set()
    rows = []
    for item in raw:
        stats = item.get("stats") or {}
        seen_keys.update(stats.keys())
        pid = str(item.get("player_id"))
        rec = {"sleeper_id": pid,
               "proj_team": item.get("team"),
               "proj_opponent": item.get("opponent")}
        for col, names in ALIASES.items():
            rec[col] = _pick(stats, names)
        rows.append(rec)

    log.info("Sleeper projection stat keys seen (%d): %s",
             len(seen_keys), ", ".join(sorted(seen_keys)[:40]))

    df = pd.DataFrame(rows)
    from .build import norm_id
    df["sleeper_id"] = df.sleeper_id.map(norm_id)

    xw = players[["gsis_id", "sleeper_id", "position", "team"]].copy()
    xw["sleeper_id"] = xw.sleeper_id.map(norm_id)
    xw = xw.dropna(subset=["sleeper_id"]).drop_duplicates("sleeper_id")
    merged = df.merge(xw, on="sleeper_id", how="left")

    missed = int(merged.gsis_id.isna().sum())
    if missed:
        log.info("%d of %d Sleeper projections had no gsis_id match (dropped)",
                 missed, len(merged))
    out = merged.dropna(subset=["gsis_id"]).copy()
    if out.empty:
        return out

    out["team"] = out.team.fillna(out.proj_team)
    out["season"], out["week"] = season, week
    out["source"] = "sleeper"
    out["pulled_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for c in ["proj_carries", "proj_targets"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["proj_opps"] = out.proj_carries.fillna(0) + out.proj_targets.fillna(0)

    # committee share: a player's slice of his own team's projected volume
    tot_car = out.groupby("team").proj_carries.transform("sum")
    tot_tgt = out.groupby("team").proj_targets.transform("sum")
    out["proj_carry_share"] = (out.proj_carries / tot_car.replace(0, np.nan)).round(3)
    out["proj_target_share"] = (out.proj_targets / tot_tgt.replace(0, np.nan)).round(3)

    cols = (["season", "week", "gsis_id", "source", "pulled_at", "team", "position"]
            + list(ALIASES.keys())
            + ["proj_opps", "proj_carry_share", "proj_target_share"])
    return out[cols].sort_values(["position", "proj_opps"],
                                 ascending=[True, False]).reset_index(drop=True)
