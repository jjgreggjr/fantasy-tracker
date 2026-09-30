"""Serve the model for the upcoming NFL week, beside E_pts.

    python -m model.serve                       # the week that has not kicked off yet (what the weekly workflow runs)
    python -m model.serve --dry-run             # compute and print, write nothing
    python -m model.serve --freeze              # recompute model/serving_config.json (preseason, or after a spec change)
    python -m model.serve --columns             # the column reference for data/model_pts.csv

What it does, in order (every step reuses training code; nothing is re-implemented for serving):

  1. Refresh the raw nflverse files that change in season (stats, snaps, injuries, depth charts, xFP, schedule, players) and
     load the store (`point_in_time.load_store`, no play-by-play / ADP / college: the shipped inputs do not use them).
  2. Pick the week: the first regular-season week with a game that has not kicked off. Only games that have NOT kicked off are
     (re)predicted; rows for games already under way stay exactly as an earlier run wrote them (the frozen record the
     scoreboard judges). A completed week is never touched.
  3. Rows: the spine and every feature for those games come from `build_features.build_games`, i.e. `features.spine_for` and
     `features.build_features` through the `as_of_join` gate at each game's real kickoff. The training frame is the matrix
     builder's own output for every earlier game (2021 on), cached on a hash of the feature code.
  4. Fit and predict with the backtest's own `backtest.walk_forward` (its masks and look-ahead assertions), for the week:
       * PPR point estimate: the plain average of LightGBM, XGBoost and CatBoost (`tuned_params.json`);
       * 14 LightGBM component models (own frozen tree counts), composed under each league's linear scoring
         (`leaguescore`): `pts_<slug>`; under PPR: `pts_model_components`;
       * LightGBM p10 / p50 / p90 with `underage_only` recalibration from trailing residuals.
     Inputs are Phase 2's 71 columns (`serving_config.json`): no new features.
  5. Who gets no row: a player the model predicts "points if he plays", so availability is the status layer's, not the model's.
     A row is WITHHELD when Sleeper's `injury_status` (data/players.csv, the pipeline's own status layer, `ff.status.OUT_STATES`)
     says Out / IR / PUP / Sus, when the nflverse report for the week says Out, or when he is cut / retired / reserve.
     Doubtful and Questionable players DO get a row (the model does not see the designation; `report_status` carries it).
  6. `data/model_pts.csv` via `ff.build.replace_partition` on (season, week), then `E_pts_model`, `p10`, `p90` appended to each
     league's `roster.csv`.

It must never break the pipeline: every failure (a missing library, a dead host, bad data) becomes a WARN row in logs/runs.csv
(check `model.serve`), writes NOTHING, and exits 0. Computation finishes before the first file is touched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL_FILE = DATA / "model_pts.csv"
CONFIG_PATH = Path(__file__).with_name("serving_config.json")
SERVE_CHECK = "model.serve"
FIRST_SEASON = 2021
CODE_FILES = ("features.py", "point_in_time.py", "labels.py", "build_features.py")
UNAVAILABLE_NFL_STATUS = {"CUT", "RET", "EXE", "RES"}          # released, retired, exempt, reserve: no row

COLUMN_DOCS = {
    "season": "NFL season",
    "week": "NFL week",
    "game_id": "nflverse game id of the player's game",
    "kickoff_utc": "that game's real kickoff (UTC, from the schedule). Rows of a game are frozen once this has passed",
    "gsis_id": "nflverse player id: the join key for every other file in the repo",
    "name": "display name (data/players.csv, else nflverse)",
    "position": "QB / RB / WR / TE",
    "team": "the team he plays for in that game",
    "opponent": "the opponent in that game",
    "report_status": "his own injury-report designation FOR THIS WEEK when served (Questionable / Doubtful); blank if there is no report "
                     "yet this week (last week's designation is not carried over). Players the report or the status layer calls Out "
                     "have no row. The model itself does not read it",
    "pts_model": "PPR point estimate: the plain average of the LightGBM, XGBoost and CatBoost flat models",
    "pts_model_components": "PPR points from the 14 component models (receptions, yards, TDs, carries, ...), same inputs",
    "pts_<slug>": "points in that league's own scoring: the component models composed with its league.json weights, LINEAR TERMS "
                  "ONLY. Not scored (see the ignored keys below): yardage bonuses (100/200-yard rush/rec, 300/400-yard pass), "
                  "long-touchdown bonuses (40+/50+ yard TDs), first downs, per-distance reception bins, per-incompletion and "
                  "all-fumble scoring; kicking, team defence and IDP never score (skill positions only). The IDP league "
                  "'where-you-at' is half-PPR here; the dynasty league is PPR + 0.5 per TE catch with a -1 interception",
    "p10": "PPR floor: LightGBM 10th-percentile model, recalibrated (underage_only) from trailing out-of-sample residuals",
    "p50": "PPR median from the LightGBM 50th-percentile model (not recalibrated)",
    "p90": "PPR ceiling: LightGBM 90th-percentile model, recalibrated like p10",
    "p10_<slug>": "floor in that league's own scoring, only for a league whose scoring is not PPR (the dynasty and IDP leagues): LightGBM "
                  "10th-percentile model fit on that league's composed actual points (same linear terms as pts_<slug>), recalibrated "
                  "(underage_only) from that league's trailing residuals. A PPR-scoring league uses p10",
    "p90_<slug>": "ceiling in that league's own scoring, for the same leagues, built like p10_<slug>",
    "sleeper_proj": "Sleeper's PPR projection for the player and week, read from data/projections.csv at serve time (no new "
                    "network call) and frozen with the row",
    "sleeper_proj_<slug>": "Sleeper's projection in that league's scoring (`proj_pts_league`), frozen at serve time",
    "e_pts_<slug>": "the pipeline's E_pts for that league, frozen at serve time (E_pts is overwritten every run)",
    "served_at": "UTC time this row was computed",
}


# --------------------------------------------------------------------------- small helpers (stdlib only at import time)
def now_utc():
    import pandas as pd
    return pd.Timestamp.now(tz="UTC")


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def spec_from(cfg: dict):
    from model import train as T
    s = cfg["spec"]
    return T.Spec(s["name"], drop_cols=tuple(s["drop_cols"]), drop_families=tuple(s["drop_families"]))


def default_cache() -> Path:
    """Where the matrix and quantile-history caches live: beside the raw files (model/cache/serve, or $MODEL_CACHE_DIR/serve)."""
    from model import point_in_time as pit
    return pit.CACHE_DIR / "serve"


def code_hash(season: int, extra: str = "") -> str:
    """Cache key: the feature code, the season, and anything else the cached frame depends on."""
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update((Path(__file__).parent / f).read_bytes())
    h.update(f"|{season}|{extra}".encode())
    return h.hexdigest()[:12]


def column_docs() -> str:
    return "\n".join(f"{k:22s} {v}" for k, v in COLUMN_DOCS.items())


# --------------------------------------------------------------------------- the run log (how integrity WARNs reach runs.csv)
def _last_context(root: Path) -> tuple[int, int]:
    """(season, week) of the newest row of logs/runs.csv: a model WARN sits next to the pipeline run it belongs to."""
    f = root / "logs" / "runs.csv"
    try:
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        if rows:
            return int(rows[-1]["season"]), int(rows[-1]["week"])
    except Exception:
        pass
    try:
        return int(json.loads((root / "config.json").read_text(encoding="utf-8")).get("season") or 0), 0
    except Exception:
        return 0, 0


def log_warns(root: Path, messages: list[str], season: int | None = None, week: int | None = None,
              check: str = SERVE_CHECK) -> None:
    """Append one WARN row (check `model.serve`, or `model.scoreboard`) through the pipeline's own `verify.log_run`, so it is
    indistinguishable from an integrity WARN. Best effort: if even that fails the message goes to stderr, and the pipeline
    carries on."""
    if not messages:
        return
    s0, w0 = _last_context(root)
    season, week = season or s0, week or w0
    try:
        from ff import verify
        verify.log_run(root / "logs", season, week, [verify._r(verify.WARN, check, m) for m in messages], note="model")
    except Exception as e:                                              # pragma: no cover - last resort
        print(f"model.serve: could not write the run-log WARN ({type(e).__name__}: {e}); messages: {messages}",
              file=sys.stderr)


# --------------------------------------------------------------------------- raw data
def volatile_files(season: int) -> list[Path]:
    """Cached raw files that change while the season is on; every serve re-downloads them (historical seasons never change)."""
    from model import point_in_time as pit
    nv, ff_ = pit.CACHE_DIR / "nflverse", pit.CACHE_DIR / "ffopportunity"
    return [nv / f"stats_player_week_{season}.parquet", nv / f"snap_counts_{season}.parquet",
            nv / f"injuries_{season}.parquet", nv / f"depth_charts_{season}.parquet", nv / "games.parquet",
            nv / "players.parquet", ff_ / f"ep_weekly_{season}.parquet"]


def refresh_raw(season: int) -> list[str]:
    """Delete the volatile cached files so `point_in_time._fetch` downloads them again. Returns the names removed."""
    gone = []
    for p in volatile_files(season):
        if p.exists():
            p.unlink()
            gone.append(p.name)
    return gone


def load_store(season: int, *, refresh: bool = True):
    from model import point_in_time as pit
    if refresh:
        refresh_raw(season)
    return pit.load_store(range(pit.CAREER_CUTOFF_SEASON, season + 1), adp=False, college=False, pbp=False)


# --------------------------------------------------------------------------- which week, which games
def pick_week(store, season: int, now, week: int | None = None):
    """(week, team-games of that week) . Default: the first regular-season week holding a game that has not kicked off."""
    from model import point_in_time as pit
    tgs = pit.team_games(store, [season], completed_only=False)
    by_week: dict[int, list] = {}
    for g in tgs:
        by_week.setdefault(g.week, []).append(g)
    if week is None:
        week = next((w for w in sorted(by_week) if any(g.kickoff > now for g in by_week[w])), None)
        if week is None:
            return None, []
    return week, by_week.get(week, [])


def unpublished_games(store, season: int, week: int) -> list[str]:
    """Completed games of the week before `week` whose player stats are not in the stats file yet (nflverse publishes Monday night's
    a few hours after the final whistle). Their players' lags would be a game short, so the serve says so."""
    gr = store._tables["game_results"].df
    have = set(store._tables["player_games"].df["game_id"])
    last = gr[(gr["season"] == season) & (gr["week"] == week - 1) & (gr["game_type"] == "REG")]
    return sorted(set(last["game_id"]) - have)


# --------------------------------------------------------------------------- frames
def _concat(parts):
    """pd.concat of matrix frames. Some columns are all-NA in a slice (ADP and college here, empty label columns in the served
    rows); pandas warns that a future version will type them differently, which changes nothing the models read."""
    import warnings
    import pandas as pd
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation", category=FutureWarning)
        return pd.concat(parts, ignore_index=True)


def prune_cache(cache_dir: Path, prefix: str, keep: Path) -> None:
    """A cache keyed on a hash of the code: once a new one is written the older ones can never be read again."""
    for old in cache_dir.glob(f"{prefix}_*.parquet"):
        if old != keep:
            old.unlink()


def training_frame(store, season: int, week: int, *, jobs: int = 1, cache_dir: Path | None = None, rebuild: bool = False, log=print):
    """Matrix rows (labels attached) of every completed game strictly before (season, week): the earlier seasons from a cache
    keyed on the feature code, this season's completed weeks rebuilt fresh every run."""
    import pandas as pd
    from model import build_features as BF
    cache_dir = cache_dir or default_cache()
    f = cache_dir / f"hist_{code_hash(season)}.parquet"
    if f.exists() and not rebuild:
        hist = pd.read_parquet(f)
        log(f"  history matrix: cache hit ({len(hist):,} rows)")
    else:
        t0 = time.time()
        hist = BF.build(list(range(FIRST_SEASON, season)), store=store, verbose=True, jobs=jobs)
        cache_dir.mkdir(parents=True, exist_ok=True)
        hist.to_parquet(f)
        prune_cache(cache_dir, "hist", f)
        log(f"  history matrix: built {len(hist):,} rows in {time.time() - t0:,.0f}s")
    cur = BF.build([season], store=store, verbose=False, jobs=jobs)
    cur = cur[cur["week"] < week] if len(cur) else cur
    return _concat([hist, cur]) if len(cur) else hist


def serve_frame(store, tgs: list):
    """The upcoming games' rows in the matrix's own schema, with every label column present and empty."""
    import numpy as np
    from model import build_features as BF
    df = BF.build_games(store, tgs, with_labels=False)
    if df.empty:
        return df
    for c in BF.LABELS:
        df[c] = 0 if c in ("y_played", "y_has_stats_row") else np.nan      # nothing is known about a game that has not been played
    for c in ("y_played", "y_has_stats_row"):
        df[c] = df[c].astype("int8")
    df["base_xfp_sameweek"] = np.nan
    df["base_trail3_ppr"] = df[["pts_ppr_l1", "pts_ppr_l2", "pts_ppr_l3"]].mean(axis=1, skipna=True)
    return df


def league_target(slug: str) -> str:
    return f"y_pts_{slug}"


def add_league_targets(df, leagues):
    """One label column per league whose scoring is not PPR: that league's linear-term points from the ACTUAL components
    (`y_pts_<slug>`, NaN where the components are NaN: a DNP, or a game not yet played). The floor and ceiling of such a league are
    quantile models fit on this, because a mean does not compose into a quantile."""
    from model import components as C
    out = df
    for ls in leagues:
        if ls.is_ppr:
            continue
        out = out.assign(**{league_target(ls.slug): C.compose({c: out[c] for c in ls.scoring.needs()}, out["position"], ls.scoring)})
    return out


def join_frames(hist, served, leagues=()):
    """One frame: earlier completed games first (matrix order), then the served rows, which no fit can train on. The misc label and a
    points label per non-PPR league are derived from the component labels."""
    from model import components as C
    return add_league_targets(C.add_misc(_concat([hist, served])), leagues)


# --------------------------------------------------------------------------- models
def _preds(df, spec, lib, params, season, week, **kw):
    """The backtest's own walk-forward for ONE week: its train mask, its look-ahead assertion, its encoder."""
    from model import backtest as B
    return B.walk_forward(df, spec, lib, params, season=season, weeks=(week,), **kw)["pred"]


def flat_predictions(df, spec, season: int, week: int, params: dict) -> dict:
    from model import train as T
    return {lib: _preds(df, spec, lib, params["point"][lib], season, week) for lib in T.LIBS}


def component_predictions(df, spec, season: int, week: int, params: dict, cfg: dict) -> dict:
    from model import components as C
    base = {k: v for k, v in params["point"]["lightgbm"].items() if k != "n_estimators"}
    return {t: _preds(df, spec, "lightgbm", {**base, "n_estimators": int(cfg["components"][t]["n_estimators"])},
                      season, week, target=t) for t in C.TARGETS}


def quantile_walk(df, spec, season: int, weeks, params: dict, target: str | None = None):
    """Sorted p10/p50/p90 walk-forward rows of `season` for `target` (PPR points unless a league's own target is named): the backtest's
    own `walk_forward_quantiles`, one LightGBM per alpha with the tuned quantile parameters."""
    from model import backtest as B
    from model import train as T
    target = target or T.TARGET
    res = None
    for a in B.QUANTILES:
        r = B.walk_forward(df, spec, "lightgbm", params["quantile"][str(a)], season=season, weeks=tuple(weeks), quantile=a,
                           target=target).rename(columns={"pred": f"q{int(a * 100)}"})
        res = r if res is None else res.assign(**{f"q{int(a * 100)}": r[f"q{int(a * 100)}"]})
    return B.sort_quantiles(res)


def quantile_history(df, spec, season: int, params: dict, *, target: str | None = None, cache_dir: Path | None = None, log=print):
    """Last season's out-of-sample quantile rows for `target`: what the recalibration is measured against. Cached like the matrix, keyed
    on the feature code, the quantile parameters, the spec and the labels themselves."""
    import pandas as pd
    from model import train as T
    target = target or T.TARGET
    cache_dir = cache_dir or default_cache()
    labels = int(pd.util.hash_pandas_object(df.loc[df["season"] == season - 1, target], index=False).sum() % (2 ** 63))
    key = code_hash(season, json.dumps([params["quantile"], list(spec.drop_cols), list(spec.drop_families), target, labels], sort_keys=True))
    tag = target.replace("y_", "").replace("_", "-")
    f = cache_dir / f"qhist-{tag}_{key}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    t0 = time.time()
    from model import backtest as B
    h = quantile_walk(df, spec, season - 1, B.WEEKS, params, target)
    cache_dir.mkdir(parents=True, exist_ok=True)
    h.to_parquet(f)
    prune_cache(cache_dir, f"qhist-{tag}", f)              # one file per target: the league and PPR histories live side by side
    log(f"  quantile history {season - 1} ({target}): {len(h):,} rows in {time.time() - t0:,.0f}s")
    return h


def band_predictions(df, spec, season: int, week: int, params: dict, cfg: dict, *, target: str | None = None,
                     cache_dir: Path | None = None, log=print):
    """q10 / q50 / q90 for the served week (raw, sorted) and the recalibrated p10 / p90, exactly as the backtest recalibrates a
    week: the shift comes from last season's out-of-sample rows plus this season's weeks before `week`, never `week` itself."""
    import numpy as np
    import pandas as pd
    from model import backtest as B
    from model import ensemble as E
    from model import train as T
    target = target or T.TARGET
    q = {a: _preds(df, spec, "lightgbm", params["quantile"][str(a)], season, week, quantile=a, target=target) for a in B.QUANTILES}
    served = pd.DataFrame({f"q{int(a * 100)}": s for a, s in q.items()})
    raw = served[["q10", "q50", "q90"]].to_numpy()
    served[["q10", "q50", "q90"]] = np.sort(raw, axis=1)
    served["crossed"] = ((raw[:, 0] > raw[:, 1]) | (raw[:, 1] > raw[:, 2])).astype(float)
    hist = quantile_history(df, spec, season, params, target=target, cache_dir=cache_dir, log=log)
    cols = ["position", "week", "y", "y_played", "q10", "q50", "q90"]
    cur = quantile_walk(df, spec, season, range(1, week), params, target)[cols] if week > 1 else pd.DataFrame(columns=cols)
    srv = pd.DataFrame({"position": df.loc[served.index, "position"], "week": week, "y": np.nan, "y_played": np.nan,
                        "q10": served["q10"], "q50": served["q50"], "q90": served["q90"]})
    rec = E.recalibrate(srv.reset_index(drop=True) if cur.empty else _concat([cur, srv]), hist, mode=cfg["recalibration"]["mode"])
    mine = rec.iloc[len(cur):]
    served["p10"], served["p90"] = mine["q10a"].to_numpy(), mine["q90a"].to_numpy()
    return served


def predict_week(df, season: int, week: int, cfg: dict, leagues=(), *, cache_dir: Path | None = None, log=print):
    """Everything the model says about the served rows of (season, week), indexed like `df`: (the PPR frame, the component
    predictions, {slug: band frame} for every league whose scoring is not PPR). `df` carries the league label columns
    (`join_frames(..., leagues)`)."""
    import pandas as pd
    from model import components as C
    from model import train as T
    spec, params = spec_from(cfg), T.load_params()
    t0 = time.time()
    flat = flat_predictions(df, spec, season, week, params)
    comp = component_predictions(df, spec, season, week, params, cfg)
    bands = band_predictions(df, spec, season, week, params, cfg, cache_dir=cache_dir, log=log)
    league_bands = {ls.slug: band_predictions(df, spec, season, week, params, cfg, target=league_target(ls.slug), cache_dir=cache_dir, log=log)
                    for ls in leagues if not ls.is_ppr}
    log(f"  fits and predictions: {time.time() - t0:,.0f}s")
    out = pd.DataFrame({"pts_model": pd.concat(list(flat.values()), axis=1).mean(axis=1)})
    for lib, s in flat.items():
        out[f"lib_{lib}"] = s
    pos = df.loc[out.index, "position"]
    out["pts_model_components"] = C.compose(comp, pos, C.PPR)
    out = out.join(bands[["q10", "q50", "q90", "p10", "p90"]])
    return out, comp, league_bands


# --------------------------------------------------------------------------- availability
def current_report(frame):
    """His own injury-report designation if it is THIS week's report, else None. The feature is 'the newest report this season', so
    on a Wednesday a player whose last report was last week's final 'Out' still carries it; that is last week's news."""
    import pandas as pd
    fresh = pd.to_numeric(frame["inj_weeks_since_report"], errors="coerce").eq(0)
    return frame["inj_report_status"].astype(object).where(frame["inj_report_status"].notna() & fresh, None)


def availability(frame, players):
    """(mask of rows to WITHHOLD, reason per row). The status layer owns availability; the model only predicts "if he plays". Three
    signals, any one withholds: the nflverse report for THIS week says Out; Sleeper's `injury_status` is one of the status layer's
    OUT_STATES (Out, IR, PUP, Sus: Sleeper also uses 'Out' for a coach's-decision inactive); his nflverse roster status is cut,
    retired, exempt or reserve. Last week's report does not withhold him: Sleeper's status, which is live, speaks for this week."""
    import pandas as pd
    from ff.status import OUT_STATES
    why = pd.Series("", index=frame.index, dtype=object)
    rep = current_report(frame).fillna("")
    why[rep.eq("Out")] = "nflverse report: Out"
    if players is not None and len(players):
        p = players.drop_duplicates("gsis_id").set_index("gsis_id")
        inj = frame["player_id"].map(p["injury_status"]) if "injury_status" in p else pd.Series(index=frame.index, dtype=object)
        nfl = frame["player_id"].map(p["nfl_status"]) if "nfl_status" in p else pd.Series(index=frame.index, dtype=object)
        why[inj.isin(OUT_STATES) & why.eq("")] = "status layer: " + inj.astype(str)
        why[nfl.isin(UNAVAILABLE_NFL_STATUS) & why.eq("")] = "status layer: nfl_status " + nfl.astype(str)
    return why.ne(""), why


# --------------------------------------------------------------------------- frozen side-by-side columns
def read_players(root: Path):
    import pandas as pd
    p = root / "data" / "players.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, low_memory=False, dtype={"gsis_id": str})


def sleeper_projection(root: Path, season: int, week: int):
    """(Series gsis_id -> PPR projection for the week from data/projections.csv, or None when the week is not there)."""
    import pandas as pd
    p = root / "data" / "projections.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, low_memory=False, usecols=lambda c: c in ("season", "week", "gsis_id", "source", "proj_pts_ppr"),
                    dtype={"gsis_id": str})
    d = d[(d["season"] == season) & (d["week"] == week) & (d["source"] == "sleeper")].dropna(subset=["gsis_id"])
    return d.drop_duplicates("gsis_id").set_index("gsis_id")["proj_pts_ppr"] if len(d) else None


def league_pool(root: Path, slug: str, season: int, week: int):
    """gsis_id -> (E_pts, proj_pts_league) for that league and week, from its roster / all_rosters / available files."""
    import pandas as pd
    frames = []
    for name in ("roster.csv", "all_rosters.csv", "available.csv"):
        p = root / "leagues" / slug / name
        if not p.exists():
            continue
        d = pd.read_csv(p, low_memory=False, usecols=lambda c: c in ("gsis_id", "season", "week", "E_pts", "proj_pts_league"),
                        dtype={"gsis_id": str})
        if {"season", "week", "gsis_id"} <= set(d.columns):
            frames.append(d[(d["season"] == season) & (d["week"] == week)])
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True).dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")
    return d if len(d) else None


def display_names(store, players, ids):
    """gsis_id -> name: the pipeline's own players table first, nflverse's display name for anyone it lacks."""
    import pandas as pd
    names = pd.Series(dtype=object)
    if players is not None and len(players):
        names = players.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")["name"]
    nv = store._tables["players_static"].df.drop_duplicates("player_id").set_index("player_id")["display_name"]
    return pd.Series([names.get(i, nv.get(i)) for i in ids], index=ids.index)


# --------------------------------------------------------------------------- the output frame
def assemble(store, served, pred, comp, leagues, *, root: Path, season: int, week: int, served_at, warnings: list,
             league_bands: dict | None = None):
    """One row per predicted (and not withheld) player: the columns of COLUMN_DOCS. Returns (frame, reason Series keyed by gsis_id
    for every withheld player)."""
    import numpy as np
    import pandas as pd
    from model import components as C
    players = read_players(root)
    if players is None:
        warnings.append("data/players.csv is missing: no availability filter and no names from the pipeline's table")
    withhold, why = availability(served, players)
    out = pd.DataFrame({"season": season, "week": week, "game_id": served["game_id"], "kickoff_utc": served["kickoff_utc"],
                        "gsis_id": served["player_id"], "position": served["position"], "team": served["team"],
                        "opponent": served["opponent"], "report_status": current_report(served)}, index=served.index)
    out.insert(5, "name", display_names(store, players, out["gsis_id"]))
    out["pts_model"] = pred["pts_model"]
    out["pts_model_components"] = pred["pts_model_components"]
    for ls in leagues:
        out[f"pts_{ls.slug}"] = C.compose(comp, served["position"], ls.scoring)
    out["p10"], out["p50"], out["p90"] = pred["p10"], pred["q50"], pred["p90"]
    for ls in leagues:                                             # a league whose scoring is not PPR has a band of its own points
        if not ls.is_ppr and league_bands and ls.slug in league_bands:
            out[f"p10_{ls.slug}"], out[f"p90_{ls.slug}"] = league_bands[ls.slug]["p10"], league_bands[ls.slug]["p90"]
    proj = sleeper_projection(root, season, week)
    if proj is None:
        warnings.append(f"no Sleeper projections for {season} week {week} in data/projections.csv: sleeper_proj left blank "
                        "(the scoreboard needs it frozen at serve time)")
    out["sleeper_proj"] = out["gsis_id"].map(proj) if proj is not None else np.nan
    pools = {ls.slug: league_pool(root, ls.slug, season, week) for ls in leagues}
    for ls in leagues:
        if pools[ls.slug] is None:
            warnings.append(f"{ls.slug}: no week {week} rows in its roster / all_rosters / available files: "
                            "sleeper and E_pts comparators left blank")
    for ls in leagues:
        pool = pools[ls.slug]
        out[f"sleeper_proj_{ls.slug}"] = out["gsis_id"].map(pool["proj_pts_league"]) if pool is not None and "proj_pts_league" in pool else np.nan
    for ls in leagues:
        pool = pools[ls.slug]
        out[f"e_pts_{ls.slug}"] = out["gsis_id"].map(pool["E_pts"]) if pool is not None and "E_pts" in pool else np.nan
    out["served_at"] = served_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    num = [c for c in out.columns if c.startswith(("pts_", "sleeper_proj", "e_pts_", "p10", "p90")) or c == "p50"]
    out[num] = out[num].astype("float64").round(3) + 0.0                      # + 0.0: no "-0.0" in the file
    reasons = why[withhold]
    reasons.index = served.loc[reasons.index, "player_id"].to_numpy()          # keyed by gsis_id
    out = out[~withhold].drop_duplicates(["season", "week", "gsis_id"]).reset_index(drop=True)
    return out, reasons


# --------------------------------------------------------------------------- writing (computation is finished by now)
def merge_frozen(old, new, season: int, week: int, now):
    """`new` plus the OLD rows of (season, week) whose game had already kicked off at `now`: those are the frozen record."""
    import pandas as pd
    if old is None or old.empty:
        return new
    sel = old[(old["season"] == season) & (old["week"] == week)]
    frozen = sel[pd.to_datetime(sel["kickoff_utc"], utc=True) <= now]
    if frozen.empty:
        return new
    return pd.concat([frozen, new[~new["gsis_id"].isin(frozen["gsis_id"])]], ignore_index=True)


def write_model_pts(path: Path, out, season: int, week: int, now) -> int:
    """Replace the (season, week) partition of model_pts.csv through ff.build.replace_partition, keeping rows of games that have
    kicked off. Writes a copy first and renames, so a crash cannot leave a half-written file."""
    import pandas as pd
    from ff import build
    old = pd.read_csv(path, low_memory=False, dtype={"gsis_id": str}) if Path(path).exists() else None
    merged = merge_frozen(old, out, season, week, now)
    tmp = Path(str(path) + ".tmp")
    if Path(path).exists():
        tmp.write_bytes(Path(path).read_bytes())
    elif tmp.exists():
        tmp.unlink()
    build.replace_partition(tmp, merged, ["season", "week"], ["season", "week", "gsis_id"])
    os.replace(tmp, path)
    return len(merged)


def roster_text(path: Path, table) -> str:
    """roster.csv with E_pts_model, p10, p90 appended, every existing byte of every existing column untouched (the csv module keeps
    each field's text; pandas would rewrite '1' as '1.0' in columns that have gaps). Re-running replaces the three columns."""
    from ff.modelcols import MODEL_COLS
    lookup = {r.gsis_id: (r.E_pts_model, r.p10, r.p90) for r in table.itertuples(index=False)}
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    drop = [i for i, h in enumerate(header) if h in MODEL_COLS]
    gi = header.index("gsis_id")
    fmt = lambda v: "" if v != v else repr(float(v) + 0.0)            # NaN -> empty, like pandas; + 0.0: never "-0.0"
    out = [[c for i, c in enumerate(header) if i not in drop] + list(MODEL_COLS)]
    for r in rows[1:]:
        v = lookup.get(r[gi], (float("nan"),) * 3)
        out.append([c for i, c in enumerate(r) if i not in drop] + [fmt(x) for x in v])
    import io
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(out)
    return buf.getvalue()


def roster_texts(root: Path, frame, leagues, season: int, week: int, warnings: list) -> dict:
    """{roster.csv path: its new text} for every league whose roster.csv belongs to the served week. Pure computation: nothing is
    written, so a malformed file raises here, before any output is touched."""
    import pandas as pd
    from ff import modelcols
    todo = {}
    for ls in leagues:
        p = root / "leagues" / ls.slug / "roster.csv"
        if not p.exists():
            continue
        head = pd.read_csv(p, low_memory=False, usecols=["season", "week"])
        if not len(head) or int(head["week"].iloc[0]) != week or int(head["season"].iloc[0]) != season:
            warnings.append(f"{ls.slug}: roster.csv is for week {int(head['week'].iloc[0]) if len(head) else '?'}, not the served "
                            f"week {week}: model columns not added")
            continue
        todo[p] = roster_text(p, modelcols.league_columns(frame, ls.slug))
    return todo


def write_outputs(root: Path, out_path: Path, out, leagues, season: int, week: int, now, warnings: list, *, rosters: bool = True):
    """Everything the serve writes, in the safe order: every roster text is computed first (a bad file raises before anything is
    touched), then model_pts.csv is replaced atomically, then each roster.csv is replaced atomically.
    Returns (rows now in model_pts.csv for the week, slugs whose roster.csv gained the columns)."""
    texts = roster_texts(root, out, leagues, season, week, warnings) if rosters else {}
    n = write_model_pts(out_path, out, season, week, now)
    for p, text in texts.items():
        tmp = Path(str(p) + ".tmp")
        tmp.write_text(text, encoding="utf-8", newline="")
        os.replace(tmp, p)
    return n, [p.parent.name for p in texts]


# --------------------------------------------------------------------------- the run
def summary_lines(out, reasons, leagues, root: Path, season: int, week: int) -> list[str]:
    """What was served, and for every league how much of its rostered skill-position pool has a row and why the rest does not."""
    import pandas as pd
    why = reasons.str.replace(r": .*", "", regex=True) if len(reasons) else reasons
    L = [f"model.serve: {season} week {week}: {len(out)} rows, {len(reasons)} withheld ("
         + (", ".join(f"{k} {v}" for k, v in why.value_counts().items()) or "none") + ")"]
    if len(out):
        L.append("  rows by position: " + ", ".join(f"{k} {v}" for k, v in out["position"].value_counts().sort_index().items()))
    have = set(out["gsis_id"])
    for ls in leagues:
        p = root / "leagues" / ls.slug / "all_rosters.csv"
        if not p.exists():
            continue
        r = pd.read_csv(p, low_memory=False, usecols=lambda c: c in ("gsis_id", "season", "week", "name", "position", "team"),
                        dtype={"gsis_id": str})
        r = r[(r["season"] == season) & (r["week"] == week)].drop_duplicates("gsis_id")
        miss = r[~r["gsis_id"].isin(have)]
        held = miss[miss["gsis_id"].isin(reasons.index)]
        line = f"  {ls.slug}: {len(r) - len(miss)} of {len(r)} rostered skill players have a model row"
        if len(miss):
            line += (f"; {len(miss)} without: {len(held)} withheld as unavailable, {len(miss) - len(held)} not in the week's spine "
                     "(bye, no role, not on a chart)")
        L.append(line)
    return L


def run(args, *, root: Path = ROOT) -> list[str]:
    """Compute, then write. Returns the printable summary; raises on any failure BEFORE a file is touched."""
    import pandas as pd
    from model import leaguescore as L
    out_path = Path(args.out) if args.out else root / "data" / "model_pts.csv"
    cfg = load_config()
    season = args.season or int(json.loads((root / "config.json").read_text(encoding="utf-8"))["season"])
    now = pd.Timestamp(args.now, tz="UTC") if args.now else now_utc()
    warnings: list[str] = []
    t0 = time.time()
    store = load_store(season, refresh=not args.no_refresh)
    week, tgs = pick_week(store, season, now, args.week)
    if week is None:
        return [f"model.serve: nothing to serve: every {season} regular-season game has kicked off"]
    upcoming = [g for g in tgs if g.kickoff > now]
    replay = not upcoming
    if replay and not (args.dry_run or args.out):
        raise RuntimeError(f"{season} week {week} has no game left to kick off: a completed week's predictions are never rewritten "
                           "(use --dry-run to replay it)")
    tgs = tgs if replay else upcoming
    print(f"model.serve: {season} week {week}, {len({g.game_id for g in tgs})} games"
          + (" (replay of a completed week)" if replay else "") + f", raw data ready in {time.time() - t0:,.0f}s", flush=True)
    jobs = args.jobs or min(4, os.cpu_count() or 1)
    late = unpublished_games(store, season, week)
    if late:
        warnings.append(f"player stats for {len(late)} completed week-{week - 1} game(s) are not published yet ({', '.join(late[:4])}"
                        f"{'...' if len(late) > 4 else ''}): their players' recent form is one game short in this serve")
    hist = training_frame(store, season, week, jobs=jobs, rebuild=args.rebuild)
    served = serve_frame(store, tgs)
    if served.empty:
        return [f"model.serve: no eligible players for {season} week {week}"]
    leagues = L.load_leagues(root)
    df = join_frames(hist, served, leagues)
    served_index = df.index[len(hist):]
    pred, comp, league_bands = predict_week(df, season, week, cfg, leagues)
    served_df = df.loc[served_index]
    out, reasons = assemble(store, served_df, pred, comp, leagues, root=root, season=season, week=week, served_at=now_utc(),
                            warnings=warnings, league_bands=league_bands)
    lines = summary_lines(out, reasons, leagues, root, season, week)
    if args.dry_run:
        return lines + ["  (dry run: nothing written)"] + [out.head(8).to_string(index=False)]
    live = out_path == root / "data" / "model_pts.csv"          # a custom --out is a scratch copy: no roster.csv, no run-log row
    n, done = write_outputs(root, out_path, out, leagues, season, week, now, warnings, rosters=live)
    lines.append(f"  wrote {out_path.name} ({n} rows for week {week}); model columns in roster.csv: {', '.join(done) or 'none'}")
    for w in warnings:
        lines.append(f"  WARN {w}")
    if warnings and live:
        log_warns(root, warnings, season, week)
    lines.append(f"  done in {time.time() - t0:,.0f}s")
    return lines


# --------------------------------------------------------------------------- freezing the serving config
def freeze(args, *, root: Path = ROOT) -> list[str]:
    """Write the component tree counts (the Phase 2.5 protocol: early stopping on 2024 after training 2021-2023, x1.1) for the serving
    spec into serving_config.json. Preseason, or after the spec changes."""
    import pandas as pd
    from model import components as C
    from model import train as T
    cfg = load_config()
    spec = spec_from(cfg)
    season = args.season or int(json.loads((root / "config.json").read_text(encoding="utf-8"))["season"])
    store = load_store(season, refresh=False)
    hist = training_frame(store, season, 1, jobs=args.jobs or min(4, os.cpu_count() or 1))
    df = C.add_misc(hist)
    dead = T.dead_columns(df)
    cols = T.feature_columns(df, spec, dead)
    base = {k: v for k, v in T.load_params()["point"]["lightgbm"].items() if k != "n_estimators"}
    played = df[df["y_played"] == 1]
    counts = {}
    for t in C.TARGETS:
        ok = played[played[t].notna() & (played["season"] >= spec.min_season)]
        tr, va = ok[ok["season"].isin(T.TRAIN_SEASONS_FOR_TUNING)], ok[ok["season"] == T.VALID_SEASON]
        Xtr, _ = T.encode(tr, cols)
        Xva, _ = T.encode(va, cols)
        _, n, loss = T._fit_with_early_stopping("lightgbm", base, Xtr, tr[t].to_numpy(), Xva, va[t].to_numpy())
        counts[t] = {"n_estimators": max(50, int(round(n * T.FOLLOW_UP_TREE_FACTOR))), "best_iteration": int(n),
                     "valid_rmse": round(float(loss), 4)}
    X, _ = T.encode(df.head(2), cols)
    cfg["components"] = counts
    cfg["inputs"] = list(X.columns)
    cfg["frozen_on"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    Path(CONFIG_PATH).write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return [f"wrote {CONFIG_PATH.name}: {len(cfg['inputs'])} inputs, component tree counts "
            + ", ".join(f"{k} {v['n_estimators']}" for k, v in counts.items())]


# --------------------------------------------------------------------------- entry point
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int, help="serve this week (a completed week needs --dry-run or --out)")
    ap.add_argument("--now", help="treat this UTC instant as now (tests, replays)")
    ap.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
    ap.add_argument("--out", help="write here instead of data/model_pts.csv (and leave roster.csv alone)")
    ap.add_argument("--jobs", type=int, default=0, help="workers for the feature build (default: up to 4)")
    ap.add_argument("--rebuild", action="store_true", help="ignore the cached history matrix")
    ap.add_argument("--no-refresh", action="store_true", help="do not re-download the volatile raw files")
    ap.add_argument("--freeze", action="store_true", help="recompute model/serving_config.json")
    ap.add_argument("--columns", action="store_true", help="print the column reference and exit")
    args = ap.parse_args(argv)
    if args.columns:
        print(column_docs())
        return 0
    try:
        for line in (freeze(args, root=ROOT) if args.freeze else run(args, root=ROOT)):
            print(line)
    except Exception as e:                      # ANY failure: a WARN row, no writes, exit 0. The pipeline never sees it.
        tb = traceback.extract_tb(e.__traceback__)
        where = f" at {Path(tb[-1].filename).name}:{tb[-1].lineno}" if tb else ""
        msg = f"{type(e).__name__}: {e}{where}"
        print(f"model.serve FAILED, outputs untouched: {msg}", file=sys.stderr)
        traceback.print_exc()
        log_warns(ROOT, [f"serve failed, nothing written: {msg}"[:400]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
