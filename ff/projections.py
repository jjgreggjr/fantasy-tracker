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
IDP_POSITIONS = ["DL", "LB", "DB", "K"]

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


def fetch_sleeper(season: int, week: int, raw_dir: Path, today: str,
                  positions: list[str] = POSITIONS, prefix: str = "projections") -> list[dict]:
    """One call per week, cached to disk by date."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / f"{prefix}_{season}_wk{week:02d}_{today}.json"
    if cache.exists():
        log.info("using cached Sleeper projections for %s wk%s", season, week)
        return json.loads(cache.read_text(encoding="utf-8"))

    params = [("season_type", "regular"), ("order_by", "pts_ppr")]
    params += [("position[]", p) for p in positions]
    r = requests.get(f"{BASE}/{season}/{week}", params=params, headers=UA, timeout=120)
    r.raise_for_status()
    data = r.json()
    cache.write_text(json.dumps(data), encoding="utf-8")
    for old in sorted(raw_dir.glob(f"{prefix}_{season}_wk{week:02d}_*.json"))[:-2]:
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


# --------------------------------------------------------------------------
# Defenders and kickers
# --------------------------------------------------------------------------
# Sleeper's projected stat line uses the same keys as league scoring (`idp_sack`, `fgm_30_39`,
# `xpm`...), so a projected line is scored exactly like a played one. Checked on the live API
# (2026 week 8, position[]=DL/LB/DB/K): ~650 of ~4,500 defenders carry `idp_tkl_solo` and
# `idp_tkl_ast`; `idp_def_td` and `idp_blk_kick` are never projected (0 here); kickers carry
# `fgm`, `fga`, `fgm_20_29`..`fgm_50p`, `fgmiss_30_39`..`fgmiss_50p`, `xpm`, `xpa`, `xpmiss`
# (no 0-19 makes and no 0-29 misses are published). A key a player does not have is NaN here, so
# "Sleeper projected him" stays distinguishable from "projected zero".
IDP_ALIASES = {
    "proj_tkl_solo": ["idp_tkl_solo"], "proj_tkl_ast": ["idp_tkl_ast"],
    "proj_tkl_loss": ["idp_tkl_loss"], "proj_sacks": ["idp_sack"],
    "proj_qb_hits": ["idp_qb_hit"], "proj_ints": ["idp_int"],
    "proj_pass_def": ["idp_pass_def"], "proj_ff": ["idp_ff"],
    "proj_fum_rec": ["idp_fum_rec"], "proj_def_td": ["idp_def_td"],
    "proj_safeties": ["idp_safe"], "proj_blk_kick": ["idp_blk_kick"],
    "proj_fg_att": ["fga"], "proj_fg_made": ["fgm"],
    "proj_fgm_0_19": ["fgm_0_19"], "proj_fgm_20_29": ["fgm_20_29"],
    "proj_fgm_30_39": ["fgm_30_39"], "proj_fgm_40_49": ["fgm_40_49"],
    "proj_fgm_50p": ["fgm_50p"],
    "proj_fgmiss_0_19": ["fgmiss_0_19"], "proj_fgmiss_20_29": ["fgmiss_20_29"],
    "proj_fgmiss_30_39": ["fgmiss_30_39"], "proj_fgmiss_40_49": ["fgmiss_40_49"],
    "proj_fgmiss_50p": ["fgmiss_50p"],
    "proj_xpa": ["xpa"], "proj_xpm": ["xpm"], "proj_xpmiss": ["xpmiss"],
}
# the fields whose presence means Sleeper actually projected a line for him
IDP_PRESENCE = ["proj_tkl_solo", "proj_tkl_ast", "proj_sacks", "proj_ints", "proj_pass_def",
                "proj_fg_made", "proj_fg_att", "proj_xpm"]
_FG_MISS_BUCKETS = ["proj_fgmiss_0_19", "proj_fgmiss_20_29", "proj_fgmiss_30_39",
                    "proj_fgmiss_40_49", "proj_fgmiss_50p"]


def idp_from_raw(raw: list, idp_players: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """The pure half of `build_idp`: Sleeper's projection payload -> data/idp_projections.csv rows.

    `proj_fg_missed` is derived (Sleeper publishes no total): the larger of attempts minus makes and the
    sum of the miss buckets it does publish. Players with no projected stat line are not written."""
    rows = []
    for item in raw or []:
        stats = item.get("stats") or {}
        rec = {"sleeper_id": str(item.get("player_id")),
               "proj_team": item.get("team"), "proj_opponent": item.get("opponent")}
        for col, names in IDP_ALIASES.items():
            rec[col] = _pick(stats, names)
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    from .build import norm_id
    df["sleeper_id"] = df.sleeper_id.map(norm_id)
    df = df[df[IDP_PRESENCE].notna().any(axis=1)]
    xw = idp_players[["gsis_id", "sleeper_id", "position", "team"]].copy()
    xw["sleeper_id"] = xw.sleeper_id.map(norm_id)
    xw = xw.dropna(subset=["sleeper_id"]).drop_duplicates("sleeper_id")
    out = df.merge(xw, on="sleeper_id", how="left").dropna(subset=["gsis_id"]).copy()
    if out.empty:
        return out
    out["team"] = out.team.fillna(out.proj_team)
    buckets = out[_FG_MISS_BUCKETS].fillna(0).sum(axis=1)
    out["proj_fg_missed"] = np.maximum((out.proj_fg_att - out.proj_fg_made).clip(lower=0).fillna(0), buckets)
    out.loc[out.proj_fg_att.isna() & out.proj_fg_made.isna() & (buckets == 0), "proj_fg_missed"] = np.nan
    out["season"], out["week"] = season, week
    out["source"] = "sleeper"
    out["pulled_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cols = (["season", "week", "gsis_id", "source", "pulled_at", "team", "position"]
            + list(IDP_ALIASES.keys()) + ["proj_fg_missed"])
    return out[cols].sort_values(["position", "gsis_id"]).reset_index(drop=True)


def build_idp(season: int, week: int, idp_players: pd.DataFrame, raw_dir: Path,
              today: str) -> pd.DataFrame:
    """data/idp_projections.csv rows for one week: Sleeper's projected stat line for DL / LB / DB / K."""
    try:
        raw = fetch_sleeper(season, week, raw_dir, today, IDP_POSITIONS, "projections_idp")
    except Exception as e:
        log.warning("Sleeper IDP/K projections unavailable for wk%s: %s", week, e)
        return pd.DataFrame()
    return idp_from_raw(raw, idp_players, season, week)
