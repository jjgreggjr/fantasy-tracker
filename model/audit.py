"""The leakage audit as reusable machinery, plus a one-shot proof slice.

    model/.venv/bin/python -m model.audit            # prints the proof slice + the leaky-join catch

`tests/test_no_leakage.py` runs the same functions as assertions. Nothing here is
imported by feature code.

  audit_truncation  rebuild features from a store physically cut to known_at < kickoff and
                    compare with the normal path (a feature that reaches past the gate, or a
                    gate bug, shows up as a diff)
  inject_canaries   synthetic rows stamped after / at / just before kickoff
  perturb_own_game  the target game's own raw rows replaced with absurd values
  leaky_features    a deliberately wrong builder (the "ad-hoc merge") used to prove the audit bites
"""
from __future__ import annotations

import random
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from model import point_in_time as pit
from model.features import build_features, fingerprint
from model.point_in_time import KNOWN_AT, RawStore, TargetRow, make_target

Builder = Callable[[RawStore, TargetRow], dict]
MONSTER = 987654.0
# Phase 2.5 canary rows for the play-by-play tables: distinctive counts (the shares they imply are .4 / .3 / .2 ... not
# what any real game produces) so a row that leaks into a feature moves it.
PBP_CANARY_USAGE = {"tgt": 40.0, "rz_tgt": 12.0, "i10_tgt": 8.0, "ay": 800.0, "ay_n": 40.0, "car": 30.0, "rz_car": 15.0,
                    "i10_car": 9.0}
PBP_CANARY_TEAM = {"plays": 999.0, "dropbacks": 900.0, "neutral_plays": 500.0, "neutral_dropbacks": 450.0, "tgt": 100.0,
                   "rz_tgt": 40.0, "i10_tgt": 20.0, "ay": 1600.0, "car": 100.0, "rz_car": 50.0, "i10_car": 30.0,
                   "part_dropbacks": 100.0, "part_rush": 100.0, "part_rz": 50.0, "part_i10": 25.0}
PBP_CANARY_PART = {"pass_on": 90.0, "run_on": 80.0, "rz_on": 40.0, "i10_on": 20.0}
PBP_TABLES = ("pbp_usage", "pbp_team", "pbp_part")
COLLEGE_MONSTER = {"college_rec_market_share": 0.987, "college_rec_td_share": 0.987, "college_rec_pg": 98.7,
                   "college_ypr": 98.7, "college_rush_share": 0.987, "college_car_pg": 98.7, "college_ypc": 98.7,
                   "college_dominator": 0.987, "college_pass_att_pg": 98.7, "college_pass_ypa": 98.7,
                   "college_pass_td_rate": 0.987, "college_power_conf": 1.0}


# --------------------------------------------------------------------------- sampling
def sample_targets(store: RawStore, seed: int = 20240929, per_pos: int = 3,
                   weeks: Iterable[int] = range(1, 23)) -> list[TargetRow]:
    """Deterministic spread across the season: `per_pos` random players per skill position per
    week, plus one rostered-but-listed (Out/Doubtful/Questionable) player, plus one 2024 rookie."""
    rng = random.Random(seed)
    pg = store._tables["player_games"].df
    inj = store._tables["injuries"].df
    ps = store._tables["players_static"].df
    rookies = set(ps.loc[ps["rookie_season"] == 2024, "player_id"])
    out: list[TargetRow] = []
    for wk in weeks:
        cur = pg[pg["week"] == wk]
        ids: list[str] = []
        for pos in pit.SKILL_POSITIONS:
            pool = sorted(cur.loc[cur["position"] == pos, "player_id"])
            ids += rng.sample(pool, min(per_pos, len(pool)))
        listed = sorted(set(inj.loc[(inj["week"] == wk) & inj["position"].isin(pit.SKILL_POSITIONS)
                                    & inj["report_status"].isin(["Out", "Doubtful", "Questionable"]),
                                    "player_id"]))
        if listed:
            ids.append(rng.choice(listed))
        rk = sorted(set(cur["player_id"]) & rookies)
        if rk:
            ids.append(rng.choice(rk))
        for pid in dict.fromkeys(ids):
            out.append(make_target(store, pid, 2024, wk))
    return out


# --------------------------------------------------------------------------- audit
def diff_keys(a: dict, b: dict) -> list[str]:
    return [k for k in a if fingerprint({k: a[k]}) != fingerprint({k: b.get(k)})]


def audit_truncation(builder: Builder, store: RawStore, targets: list[TargetRow]) -> list[dict]:
    """[] when clean. Otherwise one record per target whose features change when every raw table
    is physically cut to known_at < that target's kickoff."""
    bad: list[dict] = []
    cut_ts, cut = None, None
    for t in sorted(targets, key=lambda x: (x.kickoff, x.player_id)):
        if t.kickoff != cut_ts:                     # one truncated copy alive at a time
            cut_ts, cut = t.kickoff, store.truncated_before(t.kickoff)
        full, trunc = builder(store, t), builder(cut, t)
        keys = diff_keys(full, trunc)
        if keys:
            bad.append({"player_id": t.player_id, "week": t.week, "keys": keys,
                        "full": {k: full[k] for k in keys[:3]}, "truncated": {k: trunc.get(k) for k in keys[:3]}})
    return bad


# --------------------------------------------------------------------------- canaries
def _template(df: pd.DataFrame, **over) -> pd.DataFrame:
    row = df.iloc[[0]].copy()
    for c in row.columns:
        if c not in over and pd.api.types.is_numeric_dtype(row[c]) and c not in ("season", "week"):
            row[c] = np.nan
        elif c not in over and c != KNOWN_AT:
            row[c] = None
    for k, v in over.items():
        row[k] = v
    return row


def _stamp(ts) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_convert("UTC").as_unit("ns")


def inject_canaries(store: RawStore, t: TargetRow, stamps: Iterable[pd.Timestamp],
                    week_offset: int = 1, result_lag: pd.Timedelta = pd.Timedelta(0)) -> RawStore:
    """Add synthetic monster rows for `t`, one full set per stamp: a monster game for the player
    (stats, snaps, xFP), a monster the opponent allowed at his position (and the opponent's game
    row that makes it a window member), a fake 'Out' injury report, a wild closing line, wild
    weather, a phantom draft pick. Everything stamped `known_at = stamp`, except the game-result tables
    (player_games, snap_counts, xfp, game_results), which are stamped `stamp - result_lag`: the builder
    reads them at kickoff - RESULT_LAG, so a control canary must sit just inside THAT cutoff."""
    tabs = {n: store._tables[n].df for n in store.names()}
    add: dict[str, list[pd.DataFrame]] = {n: [] for n in tabs}
    for i, stamp in enumerate(stamps):
        ts = _stamp(stamp)
        rts = _stamp(pd.Timestamp(stamp) - result_lag)       # result-table stamp
        gid = f"{t.season}_{t.week + week_offset:02d}_CANARY{i}"
        wk = t.week + week_offset
        add["player_games"].append(_template(
            tabs["player_games"], player_id=t.player_id, season=t.season, week=wk, season_type="REG",
            game_id=gid, team=t.team, opponent_team=t.opponent, position=t.position,
            fantasy_points_ppr=MONSTER, fantasy_points=MONSTER, carries=MONSTER, targets=MONSTER,
            attempts=MONSTER, target_share=0.99, carry_share=0.99, **{KNOWN_AT: rts}))
        add["player_games"].append(_template(          # a monster scored AGAINST the opponent
            tabs["player_games"], player_id="00-CANARY", season=t.season, week=wk, season_type="REG",
            game_id=gid, team="ZZZ", opponent_team=t.opponent, position=t.position,
            fantasy_points_ppr=MONSTER, fantasy_points=MONSTER, **{KNOWN_AT: rts}))
        add["snap_counts"].append(_template(tabs["snap_counts"], player_id=t.player_id, season=t.season,
                                            week=wk, game_id=gid, offense_pct=9.99, **{KNOWN_AT: rts}))
        add["xfp"].append(_template(tabs["xfp"], player_id=t.player_id, season=t.season, week=wk,
                                    game_id=gid, total_fantasy_points_exp=MONSTER, rush_touchdown=5.0,
                                    rush_touchdown_exp=0.0, rec_touchdown=5.0, rec_touchdown_exp=0.0,
                                    pass_touchdown=5.0, pass_touchdown_exp=0.0, **{KNOWN_AT: rts}))
        add["game_results"].append(_template(
            tabs["game_results"], game_id=gid, season=t.season, week=wk, game_type="REG",
            team=t.opponent, opponent="ZZZ", **{KNOWN_AT: rts}))
        add["injuries"].append(_template(
            tabs["injuries"], player_id=t.player_id, season=t.season, week=t.week, game_type="REG",
            team=t.team, position=t.position, report_status="Out",
            practice_status="Did Not Participate In Practice", known_at_source="canary", **{KNOWN_AT: ts}))
        add["lines"].append(_template(tabs["lines"], game_id=t.game_id, season=t.season, week=t.week,
                                      spread_line=99.0, total_line=99.0, **{KNOWN_AT: ts}))
        add["weather_obs"].append(_template(tabs["weather_obs"], game_id=t.game_id, season=t.season,
                                            week=t.week, temp=-40.0, wind=99.0, **{KNOWN_AT: ts}))
        add["draft_picks"].append(_template(tabs["draft_picks"], player_id=t.player_id, season=t.season,
                                            round=99.0, pick=999.0, **{KNOWN_AT: ts}))
        add["players_static"].append(_template(tabs["players_static"], player_id=t.player_id,
                                               birth_date=pd.Timestamp("1970-01-01"), rookie_season=t.season - 30.0,
                                               height=99.0, weight=999.0, **{KNOWN_AT: ts}))
        add["combine"].append(_template(tabs["combine"], player_id=t.player_id, season=t.season, forty=9.99,
                                        **{KNOWN_AT: ts}))
        add["career_pre_cutoff"].append(_template(tabs["career_pre_cutoff"], player_id=t.player_id, games=9999.0,
                                                  **{KNOWN_AT: ts}))
        if "college" in tabs:              # a rookie-season college line (the feature is shown only in the draft season)
            add["college"].append(_template(tabs["college"], player_id=t.player_id, draft_season=t.season,
                                            **COLLEGE_MONSTER, **{KNOWN_AT: ts}))
        if "pbp_usage" in tabs:            # Phase 2.5: the same fake game in the play-by-play tables (result-table stamp)
            common = dict(season=t.season, week=wk, game_type="REG", game_id=gid, **{KNOWN_AT: rts})
            add["pbp_usage"].append(_template(tabs["pbp_usage"], player_id=t.player_id, team=t.team,
                                              **PBP_CANARY_USAGE, **common))
            add["pbp_team"].append(_template(tabs["pbp_team"], team=t.team, opponent=t.opponent,
                                             **PBP_CANARY_TEAM, **common))
            add["pbp_team"].append(_template(tabs["pbp_team"], team="ZZZ", opponent=t.opponent,   # faced by the opponent's defense
                                             **{**PBP_CANARY_TEAM, "plays": 888.0, "dropbacks": 222.0}, **common))
            if "pbp_part" in tabs:
                add["pbp_part"].append(_template(tabs["pbp_part"], player_id=t.player_id, team=t.team,
                                                 **PBP_CANARY_PART, **common))
    out = store
    for name, frames in add.items():
        if frames:
            out = out.with_frame(name, pd.concat([tabs[name], *frames], ignore_index=True))
    return out


def perturb_own_game(store: RawStore, t: TargetRow) -> RawStore:
    """Replace every numeric value in the target game's own rows (known_at == kickoff) with
    absurd numbers. Features must not notice: the game being predicted is not an input."""
    out = store
    for name in ("player_games", "snap_counts", "xfp", "game_results", *[n for n in PBP_TABLES if store.has(n)]):
        df = store._tables[name].df.copy()
        m = df["game_id"] == t.game_id
        for c in df.columns[df.dtypes.map(pd.api.types.is_float_dtype)]:
            df.loc[m, c] = MONSTER
        out = out.with_frame(name, df)
    return out


# --------------------------------------------------------------------------- family canaries
def _append(store: RawStore, name: str, frames: list[pd.DataFrame]) -> RawStore:
    return store.with_frame(name, pd.concat([store._tables[name].df, *frames], ignore_index=True))


def _prev_game_id(store: RawStore, t: TargetRow) -> str:
    """The target team's most recent completed game before kickoff (raw read: audit code only)."""
    gr = store._tables["game_results"].df
    gr = gr[(gr["team"] == t.team) & (gr[KNOWN_AT] < t.kickoff)].sort_values(KNOWN_AT)
    return gr["game_id"].iloc[-1]


def inject_teammate_out(store: RawStore, t: TargetRow, stamp) -> tuple[RawStore, str | None]:
    """An 'Out' report for one REAL teammate at the target's position who was not already listed Out or
    Doubtful, for the target's own week, stamped `stamp`. Returns (store, teammate id or None)."""
    from model.features import spine_for
    tg = TargetRow("", t.season, t.week, t.team, t.opponent, "", t.game_id, t.kickoff, t.is_home)
    inj = store._tables["injuries"].df
    out_now = set(inj.loc[(inj["season"] == t.season) & (inj["week"] == t.week)
                          & inj["report_status"].isin(["Out", "Doubtful"]), "player_id"])
    mates = [r for r in spine_for(store, tg) if r.player_id != t.player_id and r.position == t.position
             and "usage" in r.src and r.player_id not in out_now]
    if not mates:
        return store, None
    mate = mates[0].player_id
    row = _template(inj, player_id=mate, season=t.season, week=t.week, game_type="REG", team=t.team,
                    position=t.position, report_status="Out", practice_status="Did Not Participate In Practice",
                    known_at_source="canary", **{KNOWN_AT: _stamp(stamp)})
    return _append(store, "injuries", [row]), mate


def inject_phantom_teammate(store: RawStore, t: TargetRow, stamp,
                            result_lag: pd.Timedelta = pd.Timedelta(0)) -> RawStore:
    """A teammate who does not exist: a monster-usage game in the team's previous game (so he is a
    'recent' spine candidate; a result row, stamped `stamp - result_lag`) and an 'Out' report for this
    week (stamped `stamp`)."""
    ts, gid = _stamp(stamp), _prev_game_id(store, t)
    rts = _stamp(pd.Timestamp(stamp) - result_lag)
    pg, inj = store._tables["player_games"].df, store._tables["injuries"].df
    prev = pg[pg["game_id"] == gid].iloc[0]
    row = _template(pg, player_id="00-CANARY-TM", season=int(prev["season"]), week=int(prev["week"]),
                    season_type="REG", game_id=gid, team=t.team, opponent_team=prev["opponent_team"],
                    position=t.position, carries=MONSTER, targets=MONSTER, fantasy_points_ppr=MONSTER,
                    **{KNOWN_AT: rts})
    ij = _template(inj, player_id="00-CANARY-TM", season=t.season, week=t.week, game_type="REG", team=t.team,
                   position=t.position, report_status="Out", known_at_source="canary", **{KNOWN_AT: ts})
    return _append(_append(store, "player_games", [row]), "injuries", [ij])


def inject_spine_phantoms(store: RawStore, t: TargetRow, stamp,
                          result_lag: pd.Timedelta = pd.Timedelta(0)) -> RawStore:
    """Players who do not exist, on the target's team, admitted (if the gate lets them in) by each of
    the four spine signals: this week's injury report, a depth snapshot, a game in the team's previous
    game (usage; a result row, stamped `stamp - result_lag`) and, in week 1, a draft pick.
    Ids: 00-CANARY-{INJ,DEPTH,USE,DRAFT}."""
    ts, gid = _stamp(stamp), _prev_game_id(store, t)
    rts = _stamp(pd.Timestamp(stamp) - result_lag)
    tabs = {n: store._tables[n].df for n in ("player_games", "injuries", "depth_charts", "draft_picks")}
    prev = tabs["player_games"][tabs["player_games"]["game_id"] == gid].iloc[0]
    out = _append(store, "injuries", [_template(
        tabs["injuries"], player_id="00-CANARY-INJ", season=t.season, week=t.week, game_type="REG", team=t.team,
        position=t.position, report_status="Questionable", known_at_source="canary", **{KNOWN_AT: ts})])
    out = _append(out, "depth_charts", [_template(
        tabs["depth_charts"], player_id="00-CANARY-DEPTH", team=t.team, season=t.season, pos=t.position,
        known_at_source="canary", **{KNOWN_AT: ts})])
    out = _append(out, "player_games", [_template(
        tabs["player_games"], player_id="00-CANARY-USE", season=int(prev["season"]), week=int(prev["week"]),
        season_type=prev["season_type"], game_id=gid, team=t.team, opponent_team=prev["opponent_team"],
        position=t.position, carries=3.0, fantasy_points_ppr=3.0, **{KNOWN_AT: rts})])
    if t.week == 1:
        out = _append(out, "draft_picks", [_template(
            tabs["draft_picks"], player_id="00-CANARY-DRAFT", season=t.season, team=t.team, position=t.position,
            round=1.0, pick=1.0, **{KNOWN_AT: ts})])
    return out


def inject_prev_season_game(store: RawStore, t: TargetRow, stamp,
                            result_lag: pd.Timedelta = pd.Timedelta(0)) -> RawStore:
    """A monster regular-season game for the target in LAST season, stamped `stamp` (a post-kickoff stamp
    is physically impossible, which is the point: the gate must still not read it)."""
    pg = store._tables["player_games"].df
    row = _template(pg, player_id=t.player_id, season=t.season - 1, week=17, season_type="REG",
                    game_id=f"{t.season - 1}_17_CANARY", team=t.team, opponent_team=t.opponent, position=t.position,
                    fantasy_points_ppr=MONSTER, carries=MONSTER, targets=MONSTER, attempts=MONSTER,
                    **{KNOWN_AT: _stamp(pd.Timestamp(stamp) - result_lag)})
    return _append(store, "player_games", [row])


def inject_college(store: RawStore, t: TargetRow, stamp, draft_offset: int = 0) -> RawStore:
    """A monster college line for the target, stamped `stamp`, for the draft class `t.season + draft_offset`.
    offset 0 = his rookie season (the feature shows it if the gate lets it in); -1 = a veteran (the feature
    must not show it at all)."""
    tab = store._tables["college"].df
    row = _template(tab, player_id=t.player_id, draft_season=t.season + draft_offset, **COLLEGE_MONSTER,
                    **{KNOWN_AT: _stamp(stamp)})
    return _append(store, "college", [row])


def drop_own_game(store: RawStore, t: TargetRow) -> RawStore:
    """Delete every stats/snap/xFP/result row of the target game, as if it had not been recorded (or
    everyone had been a DNP). The row universe and every feature must not notice: eligibility is not
    'has a stats row that week'."""
    out = store
    for name in ("player_games", "snap_counts", "xfp", "game_results", *[n for n in PBP_TABLES if store.has(n)]):
        df = store._tables[name].df
        out = out.with_frame(name, df[df["game_id"] != t.game_id].copy())
    return out


# --------------------------------------------------------------------------- multi-season samplers
def sample_team_games(store: RawStore, plan: dict[int, int], seed: int = 20260929) -> list[TargetRow]:
    """Deterministic team-games across seasons: `plan` = {season: n}. Always includes week-1 games,
    a post-bye game and the season's latest completed week; the rest are random regular-season games."""
    rng = random.Random(seed)
    out: list[TargetRow] = []
    for season, n in plan.items():
        tgs = pit.team_games(store, [season])
        by = {(g.team, g.week): g for g in tgs}
        weeks = sorted({g.week for g in tgs})
        wk1 = [g for g in tgs if g.week == 1]
        picks = rng.sample(wk1, min(2, len(wk1)))
        postbye = [g for g in tgs if g.week > 1 and (g.team, g.week - 1) not in by and (g.team, g.week - 2) in by]
        if postbye:
            picks.append(rng.choice(postbye))
        last = [g for g in tgs if g.week == weeks[-1]]
        picks.append(rng.choice(last))
        rest = [g for g in tgs if g not in picks]
        picks += rng.sample(rest, max(0, n - len(picks)))
        out += picks
    return out


def sample_spine_rows(store: RawStore, team_games: list[TargetRow]) -> list[TargetRow]:
    from model.features import spine_for
    return [t for tg in team_games for t in spine_for(store, tg)]


def sample_college_rows(store: RawStore, seasons=(2022, 2024, 2025, 2026), per_season: int = 5,
                        weeks=(1, 2), seed: int = 20260930) -> list[TargetRow]:
    """Rookies with a college line, on the team that drafted them, in `weeks` of their draft season: the rows
    the college family can actually move. Spine rows, so the players are eligible exactly as in the matrix."""
    from model.features import spine_for
    rng = random.Random(seed)
    col = store._tables["college"].df
    dp = store._tables["draft_picks"].df
    out: list[TargetRow] = []
    for season in seasons:
        ids = sorted(col.loc[col["draft_season"] == season, "player_id"])
        by = {(g.team, g.week): g for g in pit.team_games(store, [season])}
        kept = 0
        for pid in rng.sample(ids, len(ids)):
            team = dp.loc[(dp["player_id"] == pid) & (dp["season"] == season), "team"]
            if team.empty:
                continue
            rows = [r for wk in weeks if (team.iloc[0], wk) in by
                    for r in spine_for(store, by[(team.iloc[0], wk)]) if r.player_id == pid]
            if rows:
                out += rows
                kept += 1
            if kept == per_season:
                break
    return out


def audit_spine_truncation(store: RawStore, team_games: list[TargetRow]) -> list[dict]:
    """[] when the spine (who is eligible, at which position, admitted by which signals) is identical
    from a store physically cut to known_at < kickoff."""
    from model.features import spine_for
    bad, cut_ts, cut = [], None, None
    for tg in sorted(team_games, key=lambda x: (x.kickoff, x.team)):
        if tg.kickoff != cut_ts:
            cut_ts, cut = tg.kickoff, store.truncated_before(tg.kickoff)
        a = [(r.player_id, r.position, r.src) for r in spine_for(store, tg)]
        b = [(r.player_id, r.position, r.src) for r in spine_for(cut, tg)]
        if a != b:
            bad.append({"game_id": tg.game_id, "team": tg.team, "only_full": sorted(set(a) - set(b))[:3],
                        "only_truncated": sorted(set(b) - set(a))[:3]})
    return bad


# --------------------------------------------------------------------------- the leaky join
def leaky_features(store: RawStore, t: TargetRow) -> dict:
    """What an innocent ad-hoc merge looks like. Three classic bugs, deliberately:
      1. history filtered with `week <= target week` (the target game is in its own history)
      2. same-week ffopportunity xFP merged in as if it were a lag
      3. injury status merged on (player, season, week) without asking when it was reported
    Raw frames via the private handle (the bypass), then pandas merges. Same output keys as the
    gated builder for the columns it computes, so the audit can compare like with like."""
    pg = store._tables["player_games"].df
    hist = pg[(pg["player_id"] == t.player_id) & (pg["season"] == t.season) & (pg["week"] <= t.week)]
    xf = store._tables["xfp"].df[["player_id", "season", "week", "total_fantasy_points_exp"]]
    inj = store._tables["injuries"].df[["player_id", "season", "week", "report_status"]]
    hist = hist.merge(xf, on=["player_id", "season", "week"], how="left") \
               .merge(inj, on=["player_id", "season", "week"], how="left").sort_values("week")
    last = hist.iloc[-1] if len(hist) else None
    return {
        "player_id": t.player_id, "week": t.week,
        "pts_ppr_l1": float(last["fantasy_points_ppr"]) if last is not None else float("nan"),
        "xfp_l1": float(last["total_fantasy_points_exp"]) if last is not None else float("nan"),
        "inj_report_status": last["report_status"] if last is not None else None,
    }


def gated_subset(store: RawStore, t: TargetRow) -> dict:
    """The gated builder restricted to the keys leaky_features computes."""
    f = build_features(store, t)
    return {k: f[k] for k in ("player_id", "week", "pts_ppr_l1", "xfp_l1", "inj_report_status")}


# --------------------------------------------------------------------------- proof slice
PROOF_WEEK = 10
PROOF_PLAYERS = [                      # (label, gsis id)
    ("stud RB", "00-0034844"),         # Saquon Barkley
    ("WR1", "00-0036322"),             # Justin Jefferson
    ("QB", "00-0034796"),              # Lamar Jackson
    ("2024 rookie WR", "00-0039337"),  # Malik Nabers
    ("injured (Questionable, DNP)", "00-0036554"),  # Nico Collins
]


def proof_slice(store: RawStore, week: int = PROOF_WEEK) -> pd.DataFrame:
    rows = []
    for label, pid in PROOF_PLAYERS:
        t = make_target(store, pid, 2024, week)
        f = build_features(store, t)
        pg = store._tables["player_games"].df
        actual = pg[(pg["player_id"] == pid) & (pg["season"] == 2024) & (pg["week"] == week)]
        f = {"profile": label, "name": pg.loc[pg["player_id"] == pid, "player_display_name"].iloc[0], **f}
        f["TARGET_actual_ppr (not a feature)"] = float(actual["fantasy_points_ppr"].iloc[0]) if len(actual) else None
        rows.append(f)
    return pd.DataFrame(rows).set_index("name").T


def main() -> None:
    pd.set_option("display.width", 200, "display.max_rows", 200, "display.max_columns", 20,
                  "display.max_colwidth", 32)
    store = pit.load_store((2024,))
    print(f"=== proof slice: 2024 week {PROOF_WEEK}, every value through as_of_join ===")
    print(proof_slice(store).to_string())
    targets = sample_targets(store, per_pos=1, weeks=range(2, 19, 2))
    print(f"\n=== truncation audit, {len(targets)} sampled rows ===")
    good = audit_truncation(build_features, store, targets)
    print(f"gated builder : {len(good)} mismatches")
    bad = audit_truncation(leaky_features, store, targets)
    print(f"leaky builder : {len(bad)} mismatches of {len(targets)}")
    for b in bad[:5]:
        print("   ", b["player_id"], "wk", b["week"], "differs on", b["keys"], "full", b["full"],
              "truncated", b["truncated"])


if __name__ == "__main__":
    main()
