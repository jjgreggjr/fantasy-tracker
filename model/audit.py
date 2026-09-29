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
                    week_offset: int = 1) -> RawStore:
    """Add synthetic monster rows for `t`, one full set per stamp: a monster game for the player
    (stats, snaps, xFP), a monster the opponent allowed at his position (and the opponent's game
    row that makes it a window member), a fake 'Out' injury report, a wild closing line, wild
    weather, a phantom draft pick. Everything stamped `known_at = stamp`."""
    tabs = {n: store._tables[n].df for n in store.names()}
    add: dict[str, list[pd.DataFrame]] = {n: [] for n in tabs}
    for i, stamp in enumerate(stamps):
        ts = _stamp(stamp)
        gid = f"{t.season}_{t.week + week_offset:02d}_CANARY{i}"
        wk = t.week + week_offset
        add["player_games"].append(_template(
            tabs["player_games"], player_id=t.player_id, season=t.season, week=wk, season_type="REG",
            game_id=gid, team=t.team, opponent_team=t.opponent, position=t.position,
            fantasy_points_ppr=MONSTER, fantasy_points=MONSTER, carries=MONSTER, targets=MONSTER,
            attempts=MONSTER, target_share=0.99, **{KNOWN_AT: ts}))
        add["player_games"].append(_template(          # a monster scored AGAINST the opponent
            tabs["player_games"], player_id="00-CANARY", season=t.season, week=wk, season_type="REG",
            game_id=gid, team="ZZZ", opponent_team=t.opponent, position=t.position,
            fantasy_points_ppr=MONSTER, fantasy_points=MONSTER, **{KNOWN_AT: ts}))
        add["snap_counts"].append(_template(tabs["snap_counts"], player_id=t.player_id, season=t.season,
                                            week=wk, game_id=gid, offense_pct=9.99, **{KNOWN_AT: ts}))
        add["xfp"].append(_template(tabs["xfp"], player_id=t.player_id, season=t.season, week=wk,
                                    game_id=gid, total_fantasy_points_exp=MONSTER, **{KNOWN_AT: ts}))
        add["game_results"].append(_template(
            tabs["game_results"], game_id=gid, season=t.season, week=wk, game_type="REG",
            team=t.opponent, opponent="ZZZ", **{KNOWN_AT: ts}))
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
    out = store
    for name, frames in add.items():
        if frames:
            out = out.with_frame(name, pd.concat([tabs[name], *frames], ignore_index=True))
    return out


def perturb_own_game(store: RawStore, t: TargetRow) -> RawStore:
    """Replace every numeric value in the target game's own rows (known_at == kickoff) with
    absurd numbers. Features must not notice: the game being predicted is not an input."""
    out = store
    for name in ("player_games", "snap_counts", "xfp", "game_results"):
        df = store._tables[name].df.copy()
        m = df["game_id"] == t.game_id
        for c in df.columns[df.dtypes.map(pd.api.types.is_float_dtype)]:
            df.loc[m, c] = MONSTER
        out = out.with_frame(name, df)
    return out


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
