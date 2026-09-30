"""Build the leakage-proof feature matrix.

    model/.venv/bin/python -m model.build_features                 # 2021-2026 -> model/cache/features.parquet
    model/.venv/bin/python -m model.build_features --seasons 2024 --out /tmp/f.parquet

One row per eligible (player, season, week), where eligibility comes from `features.spine_for` (usage in
the team's last 3 games, this week's injury report, the newest depth chart, the draft class in week 1),
never from whether the player has a stats row that week. Every feature passes through the `as_of_join`
gate; `labels.attach_labels` then adds y_* (target) and base_* (baselines, NOT features). Only games with a
result are built (an upcoming week would score everyone as a DNP).

Outputs (git-ignored, under model/cache/): features.parquet, features_schema.json (column -> role/family),
and a printed report. The build is deterministic: same raw cache, same bytes.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import resource
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from model import features as F
from model import labels
from model import point_in_time as pit

CACHE = pit.CACHE_DIR
DEFAULT_OUT = CACHE / "features.parquet"
LABELS = ["y_played", "y_has_stats_row", "y_offense_snaps", "y_points_ppr", *labels.COMPONENTS]
BASELINES = ["base_xfp_sameweek", "base_trail3_ppr"]
SPINE_FLAGS = ["spine_usage", "spine_injury", "spine_depth", "spine_draft", "spine_depth_only"]


def parse_seasons(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return sorted(set(out))


_WORK: dict = {}          # the store a forked worker reads: set by the parent just before it forks, never pickled


def _rows_for(chunk: list) -> list[dict]:
    store = _WORK["store"]
    return [F.build_features(store, t) for tg in chunk for t in F.spine_for(store, tg)]


def _chunks(tgs: list) -> list[list]:
    """Consecutive team-games of one (season, week), so a week's teams share the per-game context caches."""
    out: list[list] = []
    for tg in tgs:
        if out and (out[-1][0].season, out[-1][0].week) == (tg.season, tg.week):
            out[-1].append(tg)
        else:
            out.append([tg])
    return out


def build_rows(store: pit.RawStore, tgs: list, *, verbose: bool = True, jobs: int = 1) -> list[dict]:
    """One feature row per eligible (player, game) of the team-games `tgs`, in the order of `tgs`. Every row is a pure function
    of (store, target), so `jobs` only changes how long it takes: forked workers read the same in-memory store (copy on write)
    and their results are concatenated in order, which a test proves equal to the serial build."""
    rows: list[dict] = []
    t0, last = time.time(), None
    if jobs > 1 and len(tgs) >= 2 * jobs and "fork" in multiprocessing.get_all_start_methods():
        chunks = _chunks(tgs)
        _WORK["store"] = store
        try:
            with multiprocessing.get_context("fork").Pool(jobs) as pool:
                for chunk, part in zip(chunks, pool.imap(_rows_for, chunks)):
                    rows += part
                    if verbose and chunk[0].season != last:
                        last = chunk[0].season
                        print(f"  season {last}: {len(rows):,} rows so far, {time.time() - t0:5.0f}s ({jobs} workers)", flush=True)
        finally:
            _WORK.clear()
        return rows
    for tg in tgs:
        for t in F.spine_for(store, tg):
            rows.append(F.build_features(store, t))
        if verbose and tg.season != last:
            last = tg.season
            print(f"  season {tg.season}: {len(rows):,} rows so far, {time.time() - t0:5.0f}s", flush=True)
    return rows


def frame_from_rows(rows: list[dict], store: pit.RawStore, *, with_labels: bool = True) -> pd.DataFrame:
    """The matrix frame from feature rows: spine flags, optionally the y_* / base_* columns (which READ the target games' rows, so a
    frame of upcoming games is built with `with_labels=False`), sorted the way the matrix always is."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop(columns=list(F.EXCLUDED_FROM_MATRIX))
    src = df["spine_src"].fillna("")
    for s in ("usage", "injury", "depth", "draft"):
        df[f"spine_{s}"] = src.str.contains(s).astype("int8")
    df["spine_depth_only"] = (src == "depth").astype("int8")
    if with_labels:
        df = labels.attach_labels(store, df)
    return df.sort_values(["season", "week", "kickoff_utc", "team", "player_id"], kind="mergesort").reset_index(drop=True)


def build_games(store: pit.RawStore, tgs: list, *, verbose: bool = False, jobs: int = 1, with_labels: bool = True) -> pd.DataFrame:
    """The matrix rows of an explicit list of team-games (`pit.team_games`), upcoming or played."""
    return frame_from_rows(build_rows(store, tgs, verbose=verbose, jobs=jobs), store, with_labels=with_labels)


def build(seasons: list[int], *, store: pit.RawStore | None = None, verbose: bool = True,
          weeks: Iterable[int] | None = None, jobs: int = 1) -> pd.DataFrame:
    """Feature matrix for `seasons` (REG weeks with a result; only `weeks` if given), with labels and
    baselines attached. `jobs` > 1 builds the weeks in forked workers (same rows, same order)."""
    if store is None:
        hist = range(pit.CAREER_CUTOFF_SEASON, max(seasons) + 1)   # career_games_prior needs every season since the cutoff
        store = pit.load_store(hist)
    have = set(store._tables["player_games"].df["season"])
    missing = set(range(pit.CAREER_CUTOFF_SEASON, max(seasons) + 1)) - have
    if missing:
        raise ValueError(f"store lacks seasons {sorted(missing)}: career_games_prior and prev_season_* would be wrong")
    tgs = pit.team_games(store, seasons)
    if weeks is not None:
        keep = set(weeks)
        tgs = [g for g in tgs if g.week in keep]
    return build_games(store, tgs, verbose=verbose, jobs=jobs)


def schema(df: pd.DataFrame) -> dict:
    fam = {f: [c for c in cols if c in df.columns] for f, cols in F.FAMILIES.items()}
    known = set(F.IDENTITY) | set(F.META) | set(LABELS) | set(BASELINES) | set(SPINE_FLAGS) | set(F.feature_columns())
    stray = [c for c in df.columns if c not in known]
    if stray:
        raise ValueError(f"unclassified matrix columns (add them to features.FAMILIES/META): {stray}")
    return {"identity": F.IDENTITY, "features": fam, "meta": F.META + SPINE_FLAGS, "labels": LABELS,
            "baselines": BASELINES, "excluded_from_matrix": F.EXCLUDED_FROM_MATRIX,
            "notes": {"weather": "OBSERVED post-game values: backtest-only (weather_is_backtest_only); ablate wx_* columns",
                      "depth_rank": "not a feature: depth-chart semantics differ between <=2024 and 2025+",
                      "adp": "NA unless model/data/adp/*.csv exist (NA also means undrafted)",
                      "college": "CFBD final-college-season priors, rookie season only (model/college.py); breakout age is always NA",
                      "spine_depth_only": "rows admitted only by the newest depth chart; ~absent in <=2024 (see PLAN_MODEL.md)"}}


def report(df: pd.DataFrame, seconds: float) -> str:
    out = []
    out.append(f"shape {df.shape[0]:,} rows x {df.shape[1]} cols; build {seconds:,.0f}s; peak RSS "
               f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:,.0f} MB")
    t = df.groupby(["season", "position"]).agg(rows=("player_id", "size"), played=("y_played", "sum")).unstack("position")
    out.append("\nrows / played by season x position\n" + t.to_string())
    tot = df.groupby("season").agg(rows=("player_id", "size"), played=("y_played", "sum"),
                                   depth_only=("spine_depth_only", "sum"))
    tot["played_share"] = (tot["played"] / tot["rows"]).round(3)
    out.append("\nper season\n" + tot.to_string())
    prof = []
    for fam, cols in F.FAMILIES.items():
        cols = [c for c in cols if c in df.columns]
        nul = df[cols].isna().mean()
        prof.append({"family": fam, "cols": len(cols), "mean_null": round(float(nul.mean()), 3),
                     "cols_all_null": int((nul == 1).sum()), "cols_gt50pct_null": int((nul > 0.5).sum())})
    out.append("\nnull profile by family (all rows)\n" + pd.DataFrame(prof).to_string(index=False))
    played = df[df["y_played"] == 1]
    prof = []
    for fam, cols in F.FAMILIES.items():
        cols = [c for c in cols if c in played.columns]
        row = {"family": fam}
        for s, g in played.groupby("season"):
            row[int(s)] = round(float(g[cols].isna().mean().mean()), 3)
        prof.append(row)
    out.append("\nmean null rate by family x season (played rows only)\n" + pd.DataFrame(prof).to_string(index=False))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", default="2021-2026")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    seasons = parse_seasons(a.seasons)
    t0 = time.time()
    df = build(seasons)
    sch = schema(df)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    a.out.with_name(a.out.stem + "_schema.json").write_text(json.dumps(sch, indent=1, default=str))
    txt = report(df, time.time() - t0)
    a.out.with_name(a.out.stem + "_report.txt").write_text(txt)
    print(txt)
    print(f"\nwrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
