"""College production as rookie priors: CFBD final-college-season lines -> a gated `college` table.

    model/.venv/bin/python -m model.college            # match report per draft class

Source: `model/data/cfbd/player_stats_{Y-1}.csv` (CFBD /stats/player/season rows, long format:
season, playerId, player, position, team, conference, category, statType, stat) and
`team_stats_{Y-1}.csv` (season, team, conference, statName, statValue), written by model/fetch_cfbd.py in
GitHub Actions. College season Y-1 is the last college season of NFL draft class Y. The fetch kept only names
matching a drafted QB/RB/WR/TE, so it is over-inclusive: namesakes (Kevin Harris at South Carolina AND at
Texas Southern) are in the file and must be told apart here.

Identity (a wrong match is worse than NA, so every rule below can only shrink coverage):

  1. A CFBD row is a candidate for a drafted player if the normalised NAME agrees and the college season is
     the class year minus one (model.adp.norm_name; suffixes and punctuation removed).
  2. The COLLEGE must agree: nflverse `draft_picks.college` (Pro-Football-Reference spelling, "Ohio St.",
     "Mississippi") against CFBD `team` ("Ohio State", "Ole Miss"), through a normaliser plus a short, explicit
     alias table (every alias was read off a real disagreement in the data; see COLLEGE_ALIASES). Name alone is
     never enough: 3 of the namesake pairs in the 2021-2025 files would otherwise attach the wrong player's
     stats to a draftee.
  3. Tier 1 = name + position + college. Tier 2 = name + college with a different POSITION (a college RB drafted
     as a TE): college agreement is the identity check, position is not required. Neither tier guesses:
     more than one surviving candidate, two draftees claiming one CFBD id, an empty draft college, or a college
     that cannot be verified all give NA (no row in the table). Nickname cases ("Cam Ward" is "Cameron Ward" in
     CFBD) are NA: the fetch kept only rows whose name equals a drafted name, so those rows are not in the CSVs
     (a Phase 3 fetch fix: keep every FBS QB/RB/WR/TE row and match here).
  4. FBS only. CFBD's team totals (the market-share denominators) exist for FBS teams only, and FCS player lines
     are not reliably complete (a 2021 FCS quarterback shows 28 attempts). A matched player whose team has no
     team-stats row that season gets NA in every college column.

Features (final college season only; per-game = per TEAM game, so a missed game shows as lower production):

  college_rec_market_share  receiving yards / team passing yards (NCAA passing yards; sacks are not netted)
  college_rec_td_share      receiving TDs / team passing TDs
  college_rec_pg            receptions per team game
  college_ypr               receiving yards per reception (NA under MIN_REC receptions)
  college_rush_share        rushing yards / team rushing yards (NCAA nets sacks into team rushing, so slightly overstated
                            for pass-heavy teams)
  college_car_pg            carries per team game
  college_ypc               rushing yards per carry (NA under MIN_CAR carries)
  college_dominator         mean of (rec+rush yards / team pass+rush yards) and (rec+rush TDs / team pass+rush TDs);
                            NA for a drafted QB (his production is passing)
  college_pass_att_pg       pass attempts per team game (NA under MIN_PASS_ATT attempts: trick-play passers are not passers)
  college_pass_ypa, college_pass_td_rate   yards and TDs per attempt (same rule)
  college_power_conf        1 if the school was in the SEC / Big Ten / Big 12 / ACC / Pac-12 that season
  college_breakout_age      ALWAYS NA: it needs several college seasons and the fetch pulls only the last one.
                            (Phase 3: fetch earlier seasons, then age at the first 20%-dominator season.)

known_at (the leakage decision): the start of the NFL season the player was drafted into (first regular-season
kickoff - 7 days), the same rule as `draft_picks`. The identity match needs the draft (April) and the stat lines
are final at the end of bowl season, so nothing here is knowable later than that. The CSVs were fetched in
September 2026, i.e. as revised after the fact; college season totals do not move materially after the
bowls, and the table says so in `proxy`. features.py further restricts the columns to the player's rookie
season (draft_season == season), so a veteran row never sees them: coverage would otherwise depend on how many
draft classes the CSVs happen to hold (2021+ only), and that would be a train/test skew of its own.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from model import point_in_time as pit
from model.adp import norm_name
from model.point_in_time import KNOWN_AT, RawTable

CFBD_DIR = Path(__file__).resolve().parent / "data" / "cfbd"
SKILL = ("QB", "RB", "WR", "TE")
POWER_CONFERENCES = frozenset({"SEC", "Big Ten", "Big 12", "ACC", "Pac-12"})
MIN_PASS_ATT = 10      # under this a player is not a passer (trick plays)
MIN_REC = 10           # per-touch efficiency on a handful of touches is noise (a -8 YPR on one catch)
MIN_CAR = 20

COLLEGE_COLUMNS = ["college_rec_market_share", "college_rec_td_share", "college_rec_pg", "college_ypr",
                   "college_rush_share", "college_car_pg", "college_ypc", "college_dominator",
                   "college_pass_att_pg", "college_pass_ypa", "college_pass_td_rate", "college_power_conf",
                   "college_breakout_age"]

PLAYER_NEED = {"season", "playerId", "player", "position", "team", "conference", "category", "statType", "stat"}
TEAM_NEED = {"season", "team", "statName", "statValue"}
TEAM_STATS = ("games", "netPassingYards", "passingTDs", "rushingYards", "rushingTDs")

# PFR spelling (draft_picks.college) -> CFBD spelling, keyed by the lower-cased PFR string. Only pairs that were
# actually observed disagreeing in the 2021-2026 classes. "St." -> "State" is handled by the normaliser.
COLLEGE_ALIASES = {
    "mississippi": "ole miss", "boston col.": "boston college", "central florida": "ucf",
    "ala-birmingham": "uab", "connecticut": "uconn", "north carolina st.": "nc state",
    "se missouri st.": "southeast missouri state", "miami (fl)": "miami",
}


def norm_college(s) -> str:
    """Comparison key for a college name: alias, '&' -> 'and', 'St.' -> 'State', letters only."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).strip().lower()
    s = COLLEGE_ALIASES.get(s, s)
    s = s.replace("&", "and")
    s = re.sub(r"\bst\.", "state", s)
    return re.sub(r"[^a-z]", "", s)


# --------------------------------------------------------------------------- reading
def read_cfbd(cfbd_dir: Path = CFBD_DIR) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """(player rows, team rows) from the fetched CSVs, or None if none exist. Schema drift is an error."""
    pf, tf = sorted(cfbd_dir.glob("player_stats_*.csv")), sorted(cfbd_dir.glob("team_stats_*.csv"))
    if not pf or not tf:
        return None
    ps = pd.concat([pd.read_csv(f) for f in pf], ignore_index=True)
    ts = pd.concat([pd.read_csv(f) for f in tf], ignore_index=True)
    if not PLAYER_NEED <= set(ps.columns):
        raise ValueError(f"CFBD player CSV schema changed: need {sorted(PLAYER_NEED)}, got {sorted(ps.columns)}")
    if not TEAM_NEED <= set(ts.columns):
        raise ValueError(f"CFBD team CSV schema changed: need {sorted(TEAM_NEED)}, got {sorted(ts.columns)}")
    return ps, ts


def _player_wide(ps: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, playerId): identity plus `category_STAT` columns. A stat a player has no line for
    is absent in the source and is 0 here (CFBD lists only players with a stat in the category)."""
    ps = ps.drop_duplicates(["season", "playerId", "category", "statType"], keep="last").copy()
    ps["stat"] = pd.to_numeric(ps["stat"], errors="coerce")
    ps["col"] = ps["category"].str.lower() + "_" + ps["statType"]
    wide = ps.pivot(index=["season", "playerId"], columns="col", values="stat")
    ident = (ps.sort_values(["season", "playerId"]).drop_duplicates(["season", "playerId"])
             .set_index(["season", "playerId"])[["player", "position", "team", "conference"]])
    return ident.join(wide).reset_index()


def _team_wide(ts: pd.DataFrame) -> pd.DataFrame:
    t = ts[ts["statName"].isin(TEAM_STATS)].drop_duplicates(["season", "team", "statName"], keep="last").copy()
    t["statValue"] = pd.to_numeric(t["statValue"], errors="coerce")
    return t.pivot(index=["season", "team"], columns="statName", values="statValue").reset_index()


# --------------------------------------------------------------------------- identity
REASONS = ("matched", "no_draft_college", "no_name_match", "college_disagree", "ambiguous", "shared_cfbd_id",
           "non_fbs")


def match_draftees(cf: pd.DataFrame, draft: pd.DataFrame) -> pd.DataFrame:
    """One row per drafted QB/RB/WR/TE with a gsis id: class, name, position, `reason` (REASONS), `tier` and the
    matched CFBD (season, playerId). `cf` is `_player_wide` output (any columns), `draft` the nflverse draft
    picks (season, gsis_id, pfr_player_name, position, college)."""
    d = draft[draft["position"].isin(SKILL) & draft["gsis_id"].notna()].copy()
    d["key"] = d["pfr_player_name"].map(norm_name)
    d["ckey"] = d["college"].map(norm_college)
    c = cf[cf["position"].isin(SKILL)].copy()
    c["key"] = c["player"].map(norm_name)
    c["ckey"] = c["team"].map(norm_college)
    by_name: dict[tuple[int, str], pd.DataFrame] = {k: g for k, g in c.groupby(["season", "key"])}
    out = []
    for r in d.itertuples(index=False):
        cand = by_name.get((int(r.season) - 1, r.key))
        rec = {"draft_season": int(r.season), "player_id": r.gsis_id, "name": r.pfr_player_name,
               "position": r.position, "college": r.college, "cfbd_season": np.nan, "cfbd_player_id": np.nan,
               "cfbd_team": None, "tier": None, "reason": None}
        if not r.ckey:
            rec["reason"] = "no_draft_college"
        elif cand is None:
            rec["reason"] = "no_name_match"
        else:
            agree = cand[cand["ckey"] == r.ckey]
            t1 = agree[agree["position"] == r.position]
            t2 = agree[agree["position"] != r.position]
            pick, tier = (t1, "name_pos_college") if len(t1) else (t2, "name_college_posswitch")
            if not len(agree):
                rec["reason"] = "college_disagree"
            elif len(pick) != 1:
                rec["reason"] = "ambiguous"
            else:
                p = pick.iloc[0]
                rec.update(reason="matched", tier=tier, cfbd_season=int(p["season"]),
                           cfbd_player_id=int(p["playerId"]), cfbd_team=p["team"])
        out.append(rec)
    m = pd.DataFrame(out)
    ok = m["reason"] == "matched"
    dup = m[ok].duplicated(["cfbd_season", "cfbd_player_id"], keep=False)       # two draftees, one CFBD player
    bad = m[ok].index[dup.to_numpy()]
    m.loc[bad, ["reason", "tier"]] = ["shared_cfbd_id", None]
    return m


# --------------------------------------------------------------------------- features
def _div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.where(b > 0)).astype("float64")


def derive_features(wide: pd.DataFrame, team: pd.DataFrame, pos: pd.Series) -> pd.DataFrame:
    """Feature columns for matched rows. `wide` rows are _player_wide rows; `team` has the team totals for the
    same (season, team) and the same index; `pos` is the DRAFT position (dominator is NA for a QB)."""
    g = lambda c: wide[c].fillna(0.0) if c in wide.columns else pd.Series(0.0, index=wide.index)   # noqa: E731
    rec, rec_yds, rec_td = g("receiving_REC"), g("receiving_YDS"), g("receiving_TD")
    car, rush_yds, rush_td = g("rushing_CAR"), g("rushing_YDS"), g("rushing_TD")
    att, pass_yds, pass_td = g("passing_ATT"), g("passing_YDS"), g("passing_TD")
    games = team["games"]
    out = pd.DataFrame(index=wide.index)
    out["college_rec_market_share"] = _div(rec_yds, team["netPassingYards"])
    out["college_rec_td_share"] = _div(rec_td, team["passingTDs"])
    out["college_rec_pg"] = _div(rec, games)
    out["college_ypr"] = _div(rec_yds, rec).where(rec >= MIN_REC)
    out["college_rush_share"] = _div(rush_yds, team["rushingYards"])
    out["college_car_pg"] = _div(car, games)
    out["college_ypc"] = _div(rush_yds, car).where(car >= MIN_CAR)
    yd = _div(rec_yds + rush_yds, team["netPassingYards"] + team["rushingYards"])
    td = _div(rec_td + rush_td, team["passingTDs"] + team["rushingTDs"])
    dom = (yd + td) / 2
    out["college_dominator"] = dom.where(pos != "QB")
    passer = att >= MIN_PASS_ATT
    out["college_pass_att_pg"] = _div(att, games).where(passer)
    out["college_pass_ypa"] = _div(pass_yds, att).where(passer)
    out["college_pass_td_rate"] = _div(pass_td, att).where(passer)
    out["college_power_conf"] = wide["conference"].isin(POWER_CONFERENCES).astype("float64")
    out["college_breakout_age"] = np.nan
    return out[COLLEGE_COLUMNS]


def build_college_frame(ps: pd.DataFrame, ts: pd.DataFrame, draft: pd.DataFrame,
                        season_starts: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(table rows without known_at handling beyond the stamp, match report per draftee)."""
    wide = _player_wide(ps)
    team = _team_wide(ts)
    draft = draft[(draft["season"] - 1).isin(wide["season"].unique())]      # classes the CSVs can speak for
    m = match_draftees(wide, draft)
    ok = m["reason"] == "matched"
    j = m[ok].merge(wide, left_on=["cfbd_season", "cfbd_player_id"], right_on=["season", "playerId"], how="left",
                    suffixes=("", "_cf"))
    j = j.merge(team, left_on=["cfbd_season", "cfbd_team"], right_on=["season", "team"], how="left",
                suffixes=("", "_tm"))
    fbs = j["games"].notna() & j["netPassingYards"].notna()
    m.loc[m.index[ok.to_numpy()][~fbs.to_numpy()], ["reason", "tier"]] = ["non_fbs", None]
    j = j[fbs.to_numpy()].reset_index(drop=True)
    feats = derive_features(j, j[["games", "netPassingYards", "passingTDs", "rushingYards", "rushingTDs"]], j["position"])
    tab = pd.concat([j[["player_id", "draft_season", "position", "cfbd_player_id", "cfbd_team", "tier"]], feats], axis=1)
    tab = tab.rename(columns={"tier": "match_tier"})
    tab[KNOWN_AT] = pd.to_datetime(tab["draft_season"].map(season_starts), utc=True).astype("datetime64[ns, UTC]")
    return tab.reset_index(drop=True), m


def load_college_table(cfbd_dir: Path = CFBD_DIR) -> RawTable | None:
    """The gated `college` table, or None when no CFBD CSVs exist (the columns are then NA, nothing invented)."""
    raw = read_cfbd(cfbd_dir)
    if raw is None:
        return None
    draft = pit._nflverse("draft_picks", "draft_picks.parquet")
    draft = draft[["season", "gsis_id", "pfr_player_name", "position", "college"]]
    tab, m = build_college_frame(raw[0], raw[1], draft, pit._season_starts())
    if tab.empty:
        return None
    rule = ("start of the NFL season the player was drafted into (first REG kickoff - 7d), like draft_picks; "
            "features.py shows the columns only in that (rookie) season")
    return RawTable("college", tab, rule,
                    proxy="CFBD lines fetched Sep 2026 (as revised after the fact); college season totals are final at the bowls",
                    dropped=m[m["reason"] != "matched"]["reason"].value_counts().to_dict())


# --------------------------------------------------------------------------- reporting
def match_report(cfbd_dir: Path = CFBD_DIR) -> pd.DataFrame:
    """Per draft class: drafted skill players, how many got college features and why the rest did not."""
    raw = read_cfbd(cfbd_dir)
    if raw is None:
        raise SystemExit(f"no CFBD CSVs in {cfbd_dir}")
    draft = pit._nflverse("draft_picks", "draft_picks.parquet")
    draft = draft[["season", "gsis_id", "pfr_player_name", "position", "college"]]
    tab, m = build_college_frame(raw[0], raw[1], draft, pit._season_starts())
    t = m.groupby(["draft_season", "reason"]).size().unstack(fill_value=0).reindex(columns=list(REASONS), fill_value=0)
    t.insert(0, "drafted_skill", t.sum(axis=1))
    t["match_rate"] = (t["matched"] / t["drafted_skill"]).round(3)
    t["tier2"] = m[m["tier"] == "name_college_posswitch"].groupby("draft_season").size().reindex(t.index, fill_value=0)
    return t


def main() -> int:
    pd.set_option("display.width", 200)
    t = match_report()
    print("drafted QB/RB/WR/TE per class (with a gsis id) and what happened to each\n")
    print(t.to_string())
    tot = t.sum(numeric_only=True)
    print(f"\nall classes: {int(tot['matched'])} of {int(tot['drafted_skill'])} drafted skill players get college "
          f"features ({tot['matched'] / tot['drafted_skill']:.1%}); {int(tot['tier2'])} via the position-switch tier")
    return 0


if __name__ == "__main__":
    sys.exit(main())
