"""A one-off look-back: how would the model have scored against Sleeper and E_pts in 2026 weeks 1-3, before any week was frozen?

    python3.12 -m model.retro                 # needs the full git history (a depth-1 clone cannot do it), about 4 minutes

NOT the scoreboard and NOT part of the adoption gate. `data/model_pts.csv` starts with the first week served by the weekly run, so
the gate counts only weeks whose predictions were frozen before kickoff. This replays the earlier weeks so the scoreboard code can
be run on real results today, and says plainly what each comparator is:

  model      a REPLAY: `model.serve`'s own functions on a completed week (spine and features through the gate at each game's kickoff,
             fits on strictly earlier games). It uses the final pre-kickoff injury report and closing lines, like a Friday run, and
             applies no availability filter. It is what the model would have said; it was not frozen at the time.
  E_pts and Sleeper's projection in the league's scoring: read out of git, per game, from the newest committed
             `leagues/<slug>/{roster,all_rosters,available}.csv` and `data/projections.csv` whose commit is BEFORE that game's kickoff
             (a commit follows the data it holds, so this can only be conservative). A league with no snapshot before a game (the ESPN
             league was first committed in week 2) has no E_pts or Sleeper value there, and the game drops out of the identical rows.
  actual     the platform's points from data/lineups_played.csv, as in the live scoreboard.

The scoring itself is `model.scoreboard` run on a temporary copy of the repo's data with the replayed rows as model_pts.csv: nothing
is written to the repository.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from model import components as C
from model import leaguescore as L
from model import point_in_time as pit
from model import scoreboard as S
from model import serve as SV

REPO = SV.ROOT
WEEKS = (1, 2, 3)
SEASON = 2026
_SNAP: dict = {}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout


def history(path: str) -> list[tuple[str, pd.Timestamp]]:
    """(commit, commit time) of every commit touching `path`, oldest first."""
    rows = [line.split() for line in git("log", "--format=%H %cI", "--", path).splitlines()]
    return sorted(((sha, pd.Timestamp(ts).tz_convert("UTC")) for sha, ts in rows), key=lambda x: x[1])


def snapshot(sha: str, path: str) -> pd.DataFrame:
    if (sha, path) not in _SNAP:
        _SNAP[(sha, path)] = pd.read_csv(io.StringIO(git("show", f"{sha}:{path}")), low_memory=False, dtype={"gsis_id": str},
                                         usecols=lambda c: c in ("gsis_id", "season", "week", "source", "E_pts", "proj_pts_league", "proj_pts_ppr"))
    return _SNAP[(sha, path)]


def newest_before(path: str, when: pd.Timestamp, season: int, week: int) -> pd.DataFrame | None:
    """The newest committed version of `path` from before `when` that holds rows of (season, week)."""
    for sha, ts in reversed(history(path)):
        if ts < when:
            d = snapshot(sha, path)
            if len(d) and int(d["week"].iloc[0]) == week and int(d["season"].iloc[0]) == season:
                return d
    return None


def league_pool(slug: str, when: pd.Timestamp, week: int) -> pd.DataFrame:
    frames = [d for d in (newest_before(f"leagues/{slug}/{n}", when, SEASON, week) for n in ("roster.csv", "all_rosters.csv", "available.csv"))
              if d is not None]
    if not frames:
        return pd.DataFrame({"E_pts": [], "proj_pts_league": []}, index=pd.Index([], name="gsis_id"))
    return pd.concat(frames, ignore_index=True).dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")


def sleeper_ppr(when: pd.Timestamp, week: int) -> pd.Series | None:
    d = newest_before("data/projections.csv", when, SEASON, week)
    if d is None:
        return None
    d = d[d["source"] == "sleeper"]
    return d.drop_duplicates("gsis_id").set_index("gsis_id")["proj_pts_ppr"] if len(d) else None


def replay_week(store, cfg: dict, leagues: list, week: int) -> pd.DataFrame:
    """model_pts-shaped rows for one completed week: replayed predictions, git-frozen comparators, per game."""
    tgs = [g for g in pit.team_games(store, [SEASON], completed_only=False) if g.week == week]
    hist = SV.training_frame(store, SEASON, week, jobs=4, log=lambda *_: None)
    served = SV.serve_frame(store, tgs)
    df = SV.join_frames(hist, served, leagues)
    served_df = df.loc[df.index[len(hist):]]
    pred, comp, bands = SV.predict_week(df, SEASON, week, cfg, leagues, log=lambda *_: None)
    out = pd.DataFrame({"season": SEASON, "week": week, "gsis_id": served_df["player_id"], "position": served_df["position"],
                        "kickoff": pd.to_datetime(served_df["kickoff_utc"], utc=True), "pts_model": pred["pts_model"],
                        "pts_model_components": pred["pts_model_components"], "sleeper_proj": np.nan})
    for ls in leagues:
        out[f"pts_{ls.slug}"] = C.compose(comp, served_df["position"], ls.scoring)
        out[f"sleeper_proj_{ls.slug}"] = np.nan
        out[f"e_pts_{ls.slug}"] = np.nan
    for kick, grp in out.groupby("kickoff"):
        proj = sleeper_ppr(kick, week)
        if proj is not None:
            out.loc[grp.index, "sleeper_proj"] = grp["gsis_id"].map(proj)
        for ls in leagues:
            pool = league_pool(ls.slug, kick, week)
            out.loc[grp.index, f"e_pts_{ls.slug}"] = grp["gsis_id"].map(pool["E_pts"])
            out.loc[grp.index, f"sleeper_proj_{ls.slug}"] = grp["gsis_id"].map(pool["proj_pts_league"])
    return out.drop(columns="kickoff")


def main(argv=None) -> int:
    try:
        git("rev-parse", "--verify", "HEAD")
        if git("rev-parse", "--is-shallow-repository").strip() == "true":
            raise RuntimeError("shallow clone")
    except Exception as e:
        print(f"model.retro needs the full git history ({e}): `git fetch --unshallow`", file=sys.stderr)
        return 1
    cfg = SV.load_config()
    store = SV.load_store(SEASON, refresh=False)
    leagues = L.load_leagues(REPO)
    frames = []
    for week in WEEKS:
        frames.append(replay_week(store, cfg, leagues, week))
        print(f"week {week}: {len(frames[-1])} replayed rows", flush=True)
    tmp = Path(tempfile.mkdtemp())
    try:
        shutil.copytree(REPO / "data", tmp / "data", ignore=shutil.ignore_patterns("model_*"))
        shutil.copytree(REPO / "leagues", tmp / "leagues")
        pd.concat(frames, ignore_index=True).to_csv(tmp / "data" / "model_pts.csv", index=False)
        print("=== RETRO: replayed model rows, git-frozen comparators, weeks that were never frozen. NOT the adoption gate. ===\n")
        print(S.run(tmp, write=False))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
