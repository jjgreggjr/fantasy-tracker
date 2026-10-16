"""The weekly post-mortem: what did each comparator get most wrong, and was the miss learnable, noise, or the status layer's?

    python -m model.retrospective             # recompute every scored week, update data/model_misses.csv and the report
    python -m model.retrospective --print     # also print the report
    python -m model.retrospective --dry-run   # compute and print, write nothing

Runs after `model.scoreboard` in the weekly step. It judges the SAME rows the scoreboard scores (`scoreboard.actuals` and
`scoreboard.comparators`, identical rows per league-week) and adds nothing to the gate: it explains misses, it does not grade them.

What it computes, per completed league-week:

  error         predicted minus actual, in the league's scoring, for `model_components`, `sleeper` and `e_pts` (the two model
                variants the scoreboard also carries are not post-mortemed; they move with `model_components`).
  one row per   a player rostered in several leagues has one error per league. The misses list keeps ONE row per player-week per
  player-week   comparator: the row of the first league in CANONICAL order that holds him (`gooma-s-family-league`, the PPR league
                the scoreboard lists first, then the others alphabetically), and says which league it came from.
  big miss      |error| >= 12 points (BIG_MISS). The week's 95th percentile of |error| is 14-17 points on the real data, so this is
                a little looser than "top 5%" and the report prints the percentile beside the count.
  reason        a rule on the PRE-KICKOFF record against the box score (data/player_weeks.csv), one primary and an optional secondary,
                with the evidence columns kept so a human can check each call:
                  inactive_missed   predicted > 5 and did not play (no stats row / zero snaps and zero points). A STATUS-LAYER miss,
                                    not a model miss: these rows are outside the scoreboard's identical rows and are counted apart.
                  left_early        snap share far below his trailing snap share (in-game injury or benching), on an over-prediction.
                  teammate_out      volume up and a same-team same-position player at or above him on that week's depth chart did not
                                    play (no stats row that week although he had one earlier this season, zero snaps, or a rostered
                                    zero-point lineup row).
                  blowout_script    game margin >= 17 and his volume dropped, on an over-prediction.
                  volume_up/down    actual volume off the expectation by more than 35% and at least 3 opportunities (6 for a QB),
                                    the sign agreeing with the error.
                  td_variance       volume as expected (within 25%, or within 3 opportunities; 6 for a QB) but the touchdowns far
                                    from it (2+ when fewer than one was expected, or none on heavy volume, or 1.5+ away from the
                                    trailing rate). With a volume call it is the secondary reason instead.
                  unexplained       none of the above.
                "Expected" volume, touchdowns and snap share are the player's trailing means over his earlier played weeks this season
                (the same opps_l1..l3 / snap_pct_l1..l3 record the model reads). E_opps is not frozen per week, so it cannot be used
                here; volume for a QB is pass attempts plus carries, for everyone else `opps` (carries plus targets).

Outputs: `data/model_misses.csv`, one row per big miss per comparator, the (season, week) partition replaced through
`ff.build.replace_partition` like model_pts.csv; and `model/reports/retrospective.md`: the drift table (pick accuracy by week per
comparator from data/model_eval.csv, league-pooled, with the cumulative mean and the week-to-week standard deviation), reason shares
per comparator, the latest week's top 15 misses with evidence, and a candidate-signals section DERIVED from the reason shares and the
frozen `inputs` of model/serving_config.json. It edits nothing else (not candidates.md, not the scoreboard's files).

Like `model.serve` and `model.scoreboard` it can never break the pipeline: any failure is one WARN row (check `model.retrospective`)
and nothing is written. Every number is a function of the committed inputs (no clock), so a rerun changes no byte.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from model import scoreboard as S

ROOT = Path(__file__).resolve().parents[1]
MISSES_FILE = ROOT / "data" / "model_misses.csv"
REPORT = ROOT / "model" / "reports" / "retrospective.md"
CONFIG_PATH = Path(__file__).with_name("serving_config.json")
CHECK = "model.retrospective"

COMPARATORS = ("model_components", "sleeper", "e_pts")
CANONICAL = "gooma-s-family-league"
BIG_MISS = 12.0                 # |error| in league points
INACTIVE_PRED = 5.0             # predicted more than this and did not play: a status-layer miss
VOLUME_OFF = 0.35               # actual volume off expected by more than this share: volume_up / volume_down
VOLUME_NEAR = 0.25              # within this share of expected: volume was as expected (td_variance territory)
HEAVY_VOLUME = {"QB": 30.0, "RB": 15.0, "WR": 15.0, "TE": 15.0}
MIN_VOLUME_DIFF = {"QB": 6.0, "RB": 3.0, "WR": 3.0, "TE": 3.0}   # a volume call also needs this many opportunities of difference
LEFT_EARLY_SHARE = 0.6          # snap share below this fraction of the trailing share
LEFT_EARLY_MIN_TRAIL = 0.4      # ...when the trailing share was at least this
BLOWOUT_MARGIN = 17.0
BLOWOUT_VOLUME = 0.8            # volume below this fraction of expected counts as "dropped" in a blowout
TD_FAR = 1.5
TRAIL_WEEKS = 3

REASONS = ("inactive_missed", "left_early", "teammate_out", "blowout_script", "volume_up", "volume_down", "td_variance", "unexplained")
LEARNABLE = ("volume_up", "volume_down", "teammate_out", "blowout_script")
NOISE = ("td_variance", "left_early", "unexplained")
STATUS = ("inactive_missed",)

KEYS = ["season", "week", "comparator", "gsis_id"]
EVIDENCE = ["exp_vol", "act_vol", "exp_td", "act_td", "trail_snap_pct", "snap_pct", "margin", "teammate_out_name", "played"]
MISS_COLS = ["season", "week", "gsis_id", "name", "position", "team", "league", "comparator", "predicted", "actual", "error",
             "reason", "secondary_reason"] + EVIDENCE

# reason -> the serving_config inputs that would carry (or already carry) the signal
SIGNAL_FEATURES = {
    "volume_up": ("opps_l1", "opps_l2", "opps_l3", "target_share_l1", "carry_share_l1", "role_score", "role_rank_pos", "opps_trend"),
    "volume_down": ("opps_l1", "opps_l2", "opps_l3", "target_share_l1", "carry_share_l1", "role_score", "role_rank_pos", "opps_trend"),
    "teammate_out": ("tm_out_above_n", "tm_out_group_opps", "tm_out_targets_all", "tm_out_carries_all", "tm_group_size"),
    "blowout_script": ("team_spread", "implied_team_total", "total_line"),
    "left_early": ("snap_pct_l1", "inj_weeks_since_report", "inj_report_status", "inj_practice_status"),
    "td_variance": ("td_luck_rush_std", "td_luck_rec_std", "td_luck_pass_std", "td_luck_total_std", "td_luck_total_prev"),
}


# --------------------------------------------------------------------------- inputs
def canonical_order(leagues: dict) -> list[str]:
    """League slugs, the canonical PPR league first, the rest alphabetically."""
    rest = sorted(s for s in leagues if s != CANONICAL)
    return ([CANONICAL] if CANONICAL in leagues else []) + rest


def load_inputs(root: Path) -> dict:
    """Every committed file the post-mortem reads (None where a file is missing)."""
    return {"mp": S.read_csv(root / "data" / "model_pts.csv"), "lp": S.read_csv(root / "data" / "lineups_played.csv"),
            "pw": S.read_csv(root / "data" / "player_weeks.csv"), "dc": S.read_csv(root / "data" / "depth_charts.csv"),
            "dvp": S.read_csv(root / "data" / "defense_vs_pos.csv"), "ev": S.read_csv(root / "data" / "model_eval.csv"),
            "leagues": S.league_index(root)}


def frozen_inputs(path: Path = CONFIG_PATH) -> set[str]:
    try:
        return set(json.loads(Path(path).read_text(encoding="utf-8")).get("inputs") or [])
    except Exception:
        return set()


# --------------------------------------------------------------------------- the box score and the pre-kickoff record
def _num(df, col: str):
    import numpy as np
    import pandas as pd
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def box_score(pw, season: int, week: int):
    """One row per player of that week: volume (QB: pass attempts + carries; else opps), touchdowns, snap share, team, opponent."""
    import pandas as pd
    if pw is None or pw.empty:
        return pd.DataFrame(columns=["gsis_id", "act_vol", "act_td", "snap_pct", "snaps", "team", "opponent"])
    w = pw[(pw["season"] == season) & (pw["week"] == week)].drop_duplicates("gsis_id").copy()
    w["gsis_id"] = w["gsis_id"].astype(str)
    w["act_vol"] = volume(w)
    w["act_td"] = _num(w, "rec_tds").fillna(0) + _num(w, "rush_tds").fillna(0) + _num(w, "pass_tds").fillna(0)
    w["snap_pct"], w["snaps"] = _num(w, "snap_pct"), _num(w, "snaps")
    for c in ("team", "opponent"):
        if c not in w.columns:
            w[c] = None
    return w[["gsis_id", "act_vol", "act_td", "snap_pct", "snaps", "team", "opponent"]]


def volume(df):
    """Opportunities as the classifier counts them: pass attempts plus carries for a QB, `opps` (carries + targets) otherwise."""
    import numpy as np
    qb = df["position"].astype(str).eq("QB") if "position" in df.columns else np.zeros(len(df), dtype=bool)
    opps = _num(df, "opps").fillna(_num(df, "carries").fillna(0) + _num(df, "targets").fillna(0))
    qb_vol = _num(df, "pass_att").fillna(0) + _num(df, "carries").fillna(0)
    return np.where(qb, qb_vol, opps)


def trailing(pw, season: int, week: int):
    """Per player, the pre-kickoff expectation: mean volume, touchdowns and snap share over his last TRAIL_WEEKS played weeks before
    `week` this season (a played week has snaps > 0, or a stats row when snaps are unknown). NaN with no earlier week."""
    import pandas as pd
    cols = ["gsis_id", "exp_vol", "exp_td", "trail_snap_pct"]
    if pw is None or pw.empty:
        return pd.DataFrame(columns=cols)
    h = pw[(pw["season"] == season) & (pw["week"] < week)].copy()
    snaps = _num(h, "snaps")
    h = h[snaps.isna() | (snaps > 0)]
    if h.empty:
        return pd.DataFrame(columns=cols)
    h["gsis_id"] = h["gsis_id"].astype(str)
    h["vol"] = volume(h)
    h["td"] = _num(h, "rec_tds").fillna(0) + _num(h, "rush_tds").fillna(0) + _num(h, "pass_tds").fillna(0)
    h["sp"] = _num(h, "snap_pct")
    h = h.sort_values(["gsis_id", "week"]).groupby("gsis_id").tail(TRAIL_WEEKS)
    g = h.groupby("gsis_id").agg(exp_vol=("vol", "mean"), exp_td=("td", "mean"), trail_snap_pct=("sp", "mean")).reset_index()
    return g[cols]


def margins(dvp, season: int, week: int) -> dict:
    """{team: final margin} from data/defense_vs_pos.csv: a team's points are what its opponent's defence allowed."""
    if dvp is None or dvp.empty or "points_allowed" not in dvp.columns:
        return {}
    d = dvp[(dvp["season"] == season) & (dvp["week"] == week)]
    if "position" in d.columns:
        d = d[d["position"].astype(str) == "ALL"]
    allowed = {str(t): float(p) for t, p in zip(d["defense"], _num(d, "points_allowed")) if p == p}
    return allowed


def team_margin(allowed: dict, team, opponent):
    """Margin for `team`: its points (what `opponent` allowed) minus the opponent's (what `team` allowed)."""
    import numpy as np
    if team in allowed and opponent in allowed:
        return allowed[opponent] - allowed[team]
    return np.nan


def absent_players(pw, lp, season: int, week: int) -> tuple[set, dict]:
    """(gsis_ids that did not play that week, {gsis_id: name}). Did not play = known this season (a stats row in an earlier week or a
    lineup row) and no stats row / zero snaps in `week`, or a rostered zero-point lineup row with no stats row."""
    import pandas as pd
    names: dict = {}
    out: set = set()
    seen_week: dict = {}
    if pw is not None and len(pw):
        w = pw[(pw["season"] == season) & (pw["week"] == week)]
        seen_week = dict(zip(w["gsis_id"].astype(str), _num(w, "snaps")))
        for g, n in zip(w["gsis_id"].astype(str), w.get("name", pd.Series(index=w.index, dtype=object))):
            names.setdefault(g, n)
        earlier = pw[(pw["season"] == season) & (pw["week"] < week)]
        for g, n in zip(earlier["gsis_id"].astype(str), earlier.get("name", pd.Series(index=earlier.index, dtype=object))):
            names.setdefault(g, n)
            if g not in seen_week:
                out.add(g)
    for g, s in seen_week.items():
        if s == 0:
            out.add(g)
    if lp is not None and len(lp):
        l = lp[(lp["season"] == season) & (lp["week"] == week) & lp["gsis_id"].notna()]
        pts = _num(l, "points").fillna(0)
        for g, n, p in zip(l["gsis_id"].astype(str), l.get("name", pd.Series(index=l.index, dtype=object)), pts):
            names.setdefault(g, n)
            if p == 0 and g not in seen_week:
                out.add(g)
    return out, names


def teammates_out(dc, absent: set, names: dict, season: int, week: int):
    """{gsis_id: name of a same-team same-position player at or above him on that week's depth chart who did not play}.
    A player missing from the chart is compared against the chart's top two at his position."""
    if dc is None or dc.empty:
        return {}
    d = dc[(dc["season"] == season) & (dc["week"] == week)].copy()
    if d.empty:
        return {}
    d["gsis_id"] = d["gsis_id"].astype(str)
    d["rank"] = _num(d, "rank")
    out: dict = {}
    label = lambda t: names.get(t["gsis_id"]) or t.get("name") or t["gsis_id"]
    for (team, pos), g in d.groupby(["team", "position"]):
        g = g.dropna(subset=["rank"]).sort_values("rank")
        gone = g[g["gsis_id"].isin(absent)]
        for gid, rank in zip(g["gsis_id"], g["rank"]):
            if gid in absent:
                continue
            above = gone[(gone["rank"] <= rank) & (gone["gsis_id"] != gid)]
            out[gid] = label(above.iloc[0]) if len(above) else ""         # charted: never falls back to the team rule below
        top2 = gone[gone["rank"] <= 2]
        if len(top2):
            out[(team, pos)] = label(top2.iloc[0])
    return out


def teammate_for(tm: dict, gsis_id: str, team, position) -> str:
    """The teammate-out name for a player: his own depth-chart entry when he is charted, else the team-position fallback."""
    if gsis_id in tm:
        return tm[gsis_id]
    return tm.get((team, position), "")


# --------------------------------------------------------------------------- the classifier
def classify(row: dict) -> tuple[str, str]:
    """(primary, secondary) reason for one big miss. `row` holds error (predicted - actual), position, exp_vol, act_vol, exp_td, act_td,
    trail_snap_pct, snap_pct, margin, teammate_out_name, played. Rules fire in priority order; the secondary is the next rule that fires."""
    import numpy as np
    f = lambda k: float(row.get(k)) if row.get(k) is not None and row.get(k) == row.get(k) else np.nan
    err, exp_v, act_v, exp_td, act_td = f("error"), f("exp_vol"), f("act_vol"), f("exp_td"), f("act_td")
    trail_sp, sp, margin = f("trail_snap_pct"), f("snap_pct"), f("margin")
    over, under = err > 0, err < 0
    if not row.get("played", True):
        return "inactive_missed", ""
    hits = []
    ratio = act_v / exp_v if exp_v == exp_v and exp_v > 0 else np.nan
    diff = act_v - exp_v if exp_v == exp_v and act_v == act_v else np.nan
    min_diff = MIN_VOLUME_DIFF.get(str(row.get("position")), 3.0)
    if over and trail_sp == trail_sp and sp == sp and trail_sp >= LEFT_EARLY_MIN_TRAIL and sp < LEFT_EARLY_SHARE * trail_sp:
        hits.append("left_early")
    vol_up = under and ratio == ratio and ratio > 1 + VOLUME_OFF and diff >= min_diff
    vol_down = over and ratio == ratio and ratio < 1 - VOLUME_OFF and -diff >= min_diff
    if vol_up and row.get("teammate_out_name"):
        hits.append("teammate_out")
    if over and margin == margin and abs(margin) >= BLOWOUT_MARGIN and ratio == ratio and ratio < BLOWOUT_VOLUME and -diff >= min_diff:
        hits.append("blowout_script")
    if vol_up:
        hits.append("volume_up")
    if vol_down:
        hits.append("volume_down")
    near = (ratio == ratio and abs(ratio - 1) <= VOLUME_NEAR) or (diff == diff and abs(diff) <= min_diff)
    heavy = act_v == act_v and act_v >= HEAVY_VOLUME.get(str(row.get("position")), 15.0)
    td_far = act_td == act_td and ((under and act_td >= 2 and exp_td == exp_td and exp_td < 1)
                                   or (over and act_td == 0 and heavy)
                                   or (exp_td == exp_td and abs(act_td - exp_td) >= TD_FAR))
    if td_far and (near or hits):                 # primary only when volume was as expected; otherwise a secondary beside the volume call
        hits.append("td_variance")
    if not hits:
        return "unexplained", ""
    return hits[0], (hits[1] if len(hits) > 1 else "")


# --------------------------------------------------------------------------- per week
def week_errors(inp: dict, season: int, week: int):
    """Every scored (player, league, comparator) of the week: the scoreboard's identical rows with predicted, actual and error; and the
    rostered players who did not play but had a prediction above INACTIVE_PRED. Columns: gsis_id, league, comparator, predicted,
    actual, error, played."""
    import pandas as pd
    mp, lp, pw = inp["mp"], inp["lp"], inp["pw"]
    frozen = mp[(mp["season"] == season) & (mp["week"] == week)]
    rows = []
    for slug in canonical_order(inp["leagues"]):
        lid, _ = inp["leagues"][slug]
        act = S.actuals(lp, pw, lid, season, week)
        if act.empty:
            continue
        pred = S.comparators(frozen, slug)
        both = act.merge(pred, on="gsis_id", how="left")
        played = both[both["played"]].dropna(subset=list(S.COMPARATORS) + ["actual"])        # the scoreboard's identical rows
        sat = both[~both["played"]]
        for comp in COMPARATORS:
            p = pd.to_numeric(played[comp], errors="coerce")
            rows.append(pd.DataFrame({"gsis_id": played["gsis_id"].astype(str), "league": slug, "comparator": comp, "predicted": p,
                                      "actual": pd.to_numeric(played["actual"], errors="coerce"), "played": True}))
            q = pd.to_numeric(sat[comp], errors="coerce")
            s = sat[q > INACTIVE_PRED]
            if len(s):
                rows.append(pd.DataFrame({"gsis_id": s["gsis_id"].astype(str), "league": slug, "comparator": comp,
                                          "predicted": q[q > INACTIVE_PRED], "actual": 0.0, "played": False}))
    cols = ["gsis_id", "league", "comparator", "predicted", "actual", "played"]
    if not rows:
        return pd.DataFrame(columns=cols + ["error"])
    e = pd.concat(rows, ignore_index=True)[cols]
    e["error"] = e["predicted"] - e["actual"]
    return e


def dedupe(errors):
    """One row per (comparator, player): the first league in canonical order (the frame is built in that order)."""
    return errors.drop_duplicates(["comparator", "gsis_id"], keep="first")


def evidence_frame(inp: dict, season: int, week: int, gsis_ids):
    """The evidence columns for the week's players: box score, trailing expectation, margin, teammate out."""
    import pandas as pd
    mp = inp["mp"]
    frozen = mp[(mp["season"] == season) & (mp["week"] == week)].drop_duplicates("gsis_id")
    ident = frozen[["gsis_id"] + [c for c in ("name", "position", "team", "opponent") if c in frozen.columns]].copy()
    ident["gsis_id"] = ident["gsis_id"].astype(str)
    ev = pd.DataFrame({"gsis_id": pd.Series(sorted(set(map(str, gsis_ids))), dtype=str)})
    ev = ev.merge(ident, on="gsis_id", how="left")
    box = box_score(inp["pw"], season, week).rename(columns={"team": "box_team", "opponent": "box_opp"})
    ev = ev.merge(box, on="gsis_id", how="left").merge(trailing(inp["pw"], season, week), on="gsis_id", how="left")
    for c in ("team", "opponent"):
        if c not in ev.columns:
            ev[c] = None
    ev["team"] = ev["team"].where(ev["team"].notna(), ev["box_team"])
    ev["opponent"] = ev["opponent"].where(ev["opponent"].notna(), ev["box_opp"])
    allowed = margins(inp["dvp"], season, week)
    ev["margin"] = [team_margin(allowed, t, o) for t, o in zip(ev["team"], ev["opponent"])]
    absent, names = absent_players(inp["pw"], inp["lp"], season, week)
    tm = teammates_out(inp["dc"], absent, names, season, week)
    ev["teammate_out_name"] = [teammate_for(tm, g, t, p) for g, t, p in zip(ev["gsis_id"], ev["team"], ev["position"])]
    for c in ("name", "position"):
        if c not in ev.columns:
            ev[c] = None
    return ev[["gsis_id", "name", "position", "team", "act_vol", "act_td", "snap_pct", "exp_vol", "exp_td", "trail_snap_pct",
               "margin", "teammate_out_name"]]


def big_misses(inp: dict, season: int, week: int):
    """The week's big misses (one row per comparator and player, MISS_COLS) and the week's |error| percentile per comparator."""
    import numpy as np
    import pandas as pd
    errors = dedupe(week_errors(inp, season, week))
    q95 = {c: float(np.percentile(errors[(errors["comparator"] == c) & errors["played"]]["error"].abs(), 95))
           if (errors["comparator"] == c).any() else np.nan for c in COMPARATORS}
    big = errors[(errors["error"].abs() >= BIG_MISS) | ~errors["played"]].copy()
    if big.empty:
        return pd.DataFrame(columns=MISS_COLS), q95
    ev = evidence_frame(inp, season, week, big["gsis_id"])
    big = big.merge(ev, on="gsis_id", how="left")
    reasons = [classify(r) for r in big.to_dict("records")]
    big["reason"] = [r[0] for r in reasons]
    big["secondary_reason"] = [r[1] for r in reasons]
    big["season"], big["week"] = season, week
    big = big.sort_values(["comparator", "error"], key=lambda s: -s.abs() if s.name == "error" else s)
    return big[MISS_COLS].reset_index(drop=True), q95


def completed_weeks(inp: dict) -> list[tuple[int, int]]:
    """(season, week) pairs with frozen model rows AND recorded lineups in any league."""
    mp, lp = inp["mp"], inp["lp"]
    if mp is None or mp.empty or lp is None or lp.empty:
        return []
    frozen = {(int(s), int(w)) for s, w in zip(mp["season"], mp["week"])}
    done = {(int(s), int(w)) for s, w in zip(lp["season"], lp["week"])}
    return sorted(frozen & done)


def evaluate(root: Path = ROOT):
    """(misses for every completed week, {(season, week): q95 per comparator}, inputs)."""
    import pandas as pd
    inp = load_inputs(root)
    frames, q = [], {}
    for season, week in completed_weeks(inp):
        m, q95 = big_misses(inp, season, week)
        q[(season, week)] = q95
        if len(m):
            frames.append(m)
    misses = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=MISS_COLS)
    return misses, q, inp


# --------------------------------------------------------------------------- the report
def _f(x, nd: int = 1) -> str:
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


def drift_table(ev):
    """{comparator: (by-week pick accuracy list [(label, acc)], cumulative mean, week-to-week std)} from model_eval.csv, league-pooled."""
    import numpy as np
    out = {}
    if ev is None or ev.empty:
        return out
    d = ev[ev["position"] == "ALL"]
    weeks = sorted({(int(s), int(w)) for s, w in zip(d["season"], d["week"])})
    for c in S.COMPARATORS:
        g = d[d["comparator"] == c]
        if g.empty:
            continue
        acc = []
        for s, w in weeks:
            x = g[(g["season"] == s) & (g["week"] == w)]
            acc.append((f"{s} wk{w:02d}", float(x["credit"].sum() / x["pairs"].sum()) if x["pairs"].sum() else np.nan))
        vals = np.array([a for _, a in acc if a == a])
        out[c] = (acc, float(vals.mean()) if len(vals) else np.nan, float(vals.std(ddof=1)) if len(vals) > 1 else np.nan)
    return out


def trend_sentence(drift: dict) -> str:
    """One sentence: is any comparator trending? Needs 4 weeks for a slope to mean anything; before that it says so."""
    import numpy as np
    weeks = max((len(v[0]) for v in drift.values()), default=0)
    if weeks < 4:
        return (f"{weeks} week(s) scored: too few to call a trend for any comparator (a slope needs 4+ weeks, and the week-to-week "
                "standard deviation above is the size of ordinary noise).")
    parts = []
    for c, (acc, mean, sd) in drift.items():
        y = np.array([a for _, a in acc if a == a])
        if len(y) < 4 or sd != sd or sd == 0:
            continue
        slope = np.polyfit(np.arange(len(y)), y, 1)[0]
        if abs(slope) * (len(y) - 1) > 2 * sd:
            parts.append(f"`{c}` is {'rising' if slope > 0 else 'falling'} ({slope * (len(y) - 1):+.3f} over the span, beyond twice the weekly noise)")
    return ("No comparator is trending: every span move is inside twice its week-to-week standard deviation." if not parts
            else "; ".join(parts) + ".")


def reason_shares(misses):
    """{comparator: {reason: count}} over the played big misses, plus the inactive count apart."""
    out = {}
    for c in COMPARATORS:
        g = misses[misses["comparator"] == c]
        counts = {r: int((g["reason"] == r).sum()) for r in REASONS}
        out[c] = counts
    return out


def registry_columns() -> set[str]:
    """Every column the feature registry can build (model.features.FAMILIES); empty if the registry cannot be imported."""
    try:
        from model import features as F
        return set(F.feature_columns())
    except Exception:
        return set()


def feature_status(feats, inputs: set[str], registry: set[str]) -> str:
    """`a`, `b` (served) ; pruned: `c` (built, not an input) ; not built: `d`."""
    have = [f for f in feats if f in inputs]
    pruned = [f for f in feats if f not in inputs and f in registry]
    absent = [f for f in feats if f not in inputs and f not in registry]
    parts = []
    if have:
        parts.append("served: " + ", ".join(f"`{f}`" for f in have))
    if pruned:
        parts.append("pruned: " + ", ".join(f"`{f}`" for f in pruned))
    if absent:
        parts.append("not built: " + ", ".join(f"`{f}`" for f in absent))
    return " ; ".join(parts) or "none"


def candidate_signals(misses, inputs: set[str], registry: set[str] | None = None) -> list[str]:
    """Derived lines: for each learnable reason, its share of the model's big misses, the features that would carry the signal, and
    whether each is served, pruned (in the registry, not an input) or not built. Nothing hand-written about what to do."""
    registry = registry_columns() if registry is None else registry
    L = ["| reason | share of model big misses | features that carry the signal | status in `serving_config.json` inputs |", "|---|---|---|---|"]
    g = misses[(misses["comparator"] == "model_components") & misses["played"]]
    n = len(g)
    for reason in LEARNABLE + ("left_early", "td_variance"):
        k = int((g["reason"] == reason).sum())
        share = k / n if n else float("nan")
        feats = SIGNAL_FEATURES.get(reason, ())
        L.append(f"| `{reason}` | {_f(100 * share, 0)}% ({k}) | " + (", ".join(f"`{f}`" for f in feats) or "n/a")
                 + f" | {feature_status(feats, inputs, registry)} |")
    inactive = int((misses[misses["comparator"] == "model_components"]["reason"] == "inactive_missed").sum())
    L.append(f"| `inactive_missed` | {inactive} rows, counted apart | status layer (`ff.status`, Sleeper `injury_status`, the nflverse report), not a feature | n/a |")
    return L


def render(misses, q: dict, inp: dict, inputs: set[str]) -> str:
    """The report. A function of the committed inputs only (no clock)."""
    import pandas as pd
    L = ["# Retrospective", "",
         "What each comparator got most wrong in every completed week, and why, judged on the scoreboard's identical rows "
         "(`python -m model.retrospective`, run after `model.scoreboard`). It explains misses; it is not the adoption gate.", ""]
    weeks = sorted(q)
    if not weeks:
        L += ["**Nothing to post-mortem yet.** No completed week has frozen model predictions and recorded lineups.", ""]
    else:
        wk = ", ".join(f"{s} wk{w}" for s, w in weeks)
        L += [f"Weeks post-mortemed: {wk}. A big miss is |predicted - actual| >= {BIG_MISS:.0f} points in the league's scoring, one row per "
              f"player-week per comparator (the `{CANONICAL}` row where he is rostered there, else the first other league alphabetically). "
              f"Comparators: `model_components` (E_pts_model), `sleeper`, `e_pts`.", ""]
    # (a) drift
    drift = drift_table(inp.get("ev"))
    L += ["## Drift: pick accuracy by week", "",
          "From `data/model_eval.csv`, pooled over leagues (the scoreboard's number). Cumulative mean and the week-to-week standard "
          "deviation of the weekly values.", ""]
    if drift:
        labels = [l for l, _ in next(iter(drift.values()))[0]]
        L += ["| comparator | " + " | ".join(labels) + " | mean | week sd |", "|---|" + "---|" * (len(labels) + 2)]
        for c, (acc, mean, sd) in drift.items():
            L.append(f"| `{c}` | " + " | ".join(_f(a, 3) for _, a in acc) + f" | {_f(mean, 3)} | {_f(sd, 3)} |")
        L += ["", trend_sentence(drift), ""]
    else:
        L += ["No scored weeks in `data/model_eval.csv` yet.", ""]
    # (b) reason shares
    L += ["## Why the big misses happened", "",
          "Share of each comparator's big misses by reason, all weeks pooled. `inactive_missed` (predicted > "
          f"{INACTIVE_PRED:.0f}, did not play) is outside the scored rows and is counted apart: it is the status layer's miss, not the model's. "
          "Learnable = the pre-kickoff record could in principle have carried the signal (" + ", ".join(f"`{r}`" for r in LEARNABLE) + "); "
          "noise = " + ", ".join(f"`{r}`" for r in NOISE) + ".", ""]
    if len(misses):
        shares = reason_shares(misses)
        L += ["| comparator | big misses | " + " | ".join(f"`{r}`" for r in REASONS if r not in STATUS) + " | learnable | noise | `inactive_missed` (apart) |",
              "|---|---|" + "---|" * (len(REASONS) - len(STATUS) + 3)]
        for c, counts in shares.items():
            n = sum(v for r, v in counts.items() if r not in STATUS)
            pct = lambda k: f"{counts[k]} ({_f(100 * counts[k] / n, 0)}%)" if n else "0"
            learn = sum(counts[r] for r in LEARNABLE)
            noise = sum(counts[r] for r in NOISE)
            L.append(f"| `{c}` | {n} | " + " | ".join(pct(r) for r in REASONS if r not in STATUS)
                     + f" | {_f(100 * learn / n, 0) if n else 'n/a'}% | {_f(100 * noise / n, 0) if n else 'n/a'}% | {counts['inactive_missed']} |")
        L += ["", "Per week, the 95th percentile of |error| on the scored rows (the `top 5%` rule would have cut here instead of at "
              f"{BIG_MISS:.0f}):", "", "| season / week | " + " | ".join(f"`{c}`" for c in COMPARATORS) + " |", "|---|" + "---|" * len(COMPARATORS)]
        L += [f"| {s} wk{w:02d} | " + " | ".join(_f(q[(s, w)].get(c)) for c in COMPARATORS) + " |" for s, w in weeks]
        L.append("")
    else:
        L += ["No big misses yet.", ""]
    # (c) latest week
    if weeks and len(misses):
        s, w = weeks[-1]
        L += [f"## Top 15 misses, {s} week {w}", "",
              "Each comparator's largest played misses that week with the evidence behind the reason (expected = trailing mean over his earlier "
              "played weeks this season; volume = pass attempts + carries for a QB, carries + targets otherwise; margin = his team's final margin).", ""]
        for c in COMPARATORS:
            g = misses[(misses["season"] == s) & (misses["week"] == w) & (misses["comparator"] == c) & misses["played"]]
            g = g.reindex(g["error"].abs().sort_values(ascending=False).index).head(15)
            L += [f"**`{c}`** ({len(g)} shown)", "",
                  "| player | pos | team | league | predicted | actual | error | reason | secondary | vol exp/act | TD exp/act | snap% trail/act | margin | teammate out |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in g.itertuples():
                L.append(f"| {r.name} | {r.position} | {r.team} | {r.league} | {_f(r.predicted)} | {_f(r.actual)} | {_f(r.error)} | `{r.reason}` | "
                         f"{('`' + r.secondary_reason + '`') if r.secondary_reason else ''} | {_f(r.exp_vol)}/{_f(r.act_vol, 0)} | {_f(r.exp_td)}/{_f(r.act_td, 0)} | "
                         f"{_f(r.trail_snap_pct, 2)}/{_f(r.snap_pct, 2)} | {_f(r.margin, 0)} | {r.teammate_out_name or ''} |")
            L.append("")
        ina = misses[(misses["season"] == s) & (misses["week"] == w) & (misses["comparator"] == "model_components") & ~misses["played"]]
        ina = ina.reindex(ina["predicted"].sort_values(ascending=False).index)
        L += [f"**Did not play, predicted > {INACTIVE_PRED:.0f} (`model_components`, {len(ina)} players; status layer, not scored)**", ""]
        if len(ina):
            L += ["| player | pos | team | league | predicted |", "|---|---|---|---|---|"]
            L += [f"| {r.name} | {r.position} | {r.team} | {r.league} | {_f(r.predicted)} |" for r in ina.itertuples()]
        else:
            L.append("none")
        L.append("")
    # (d) candidate signals
    L += ["## Candidate signals (derived)", "",
          "For each reason class, the share of the model's played big misses it explains, the features that would carry the signal, and whether "
          f"each is served (among the {len(inputs)} frozen `inputs` of `model/serving_config.json`), pruned (built by `model.features`, dropped "
          "by the Phase 2 SHAP prune) or not built. A served feature that still misses means the signal is in the inputs but the week's value "
          "was not (`tm_out_*` are read from the report as of serve time: a teammate ruled out after the last run is invisible). A reason whose "
          "features are all served is not a new-feature candidate; one whose share is high and whose features are pruned or not built is.", ""]
    L += candidate_signals(misses, inputs) if len(misses) else ["No big misses yet."]
    L += ["", "## Limits", "",
          f"- Rules, not judgment: thresholds are fixed ({int(VOLUME_OFF * 100)}% and 3+ opportunities (QB 6+) for a volume call, "
          f"{int(VOLUME_NEAR * 100)}% or 3 opportunities for 'as expected', snap share < "
          f"{int(LEFT_EARLY_SHARE * 100)}% of trailing, margin >= {BLOWOUT_MARGIN:.0f}, TDs {TD_FAR} from the trailing rate). Priority order: "
          "left_early, teammate_out, blowout_script, volume, td_variance. A row can satisfy two; the second is `secondary_reason`.",
          "- Expected volume is the trailing mean of his earlier played weeks this season, not E_opps (not frozen per week) and not the "
          "model's own component predictions (not frozen either). A player with no earlier week this season can only be `td_variance`-free "
          "`unexplained`, `left_early` (never: no trailing snaps) or `inactive_missed`.",
          "- Teammate out is read from that week's depth chart against `data/player_weeks.csv` and `data/lineups_played.csv`; `data/status.csv` "
          "is a snapshot of the upcoming week with no week column and is not used. A teammate who is simply untracked (never a stats row this "
          "season) is not counted as out.",
          "- Margins come from `data/defense_vs_pos.csv` (points allowed, `ALL` rows); a week without them has no `blowout_script`.",
          "- Two weeks of misses are a list, not a distribution; read the reason shares after six.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- entry point
def run(root: Path = ROOT, *, write: bool = True) -> str:
    from ff import build
    misses, q, inp = evaluate(root)
    text = render(misses, q, inp, frozen_inputs(root / "model" / "serving_config.json"))
    if write:
        if len(misses):
            build.replace_partition(root / "data" / "model_misses.csv", misses, ["season", "week"], KEYS)
        out = root / "model" / "reports" / "retrospective.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--print", action="store_true", dest="show", help="print the report")
    ap.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
    a = ap.parse_args(argv)
    try:
        text = run(ROOT, write=not a.dry_run)
        if a.show or a.dry_run:
            print(text)
        else:
            print(f"model.retrospective: wrote {REPORT.relative_to(ROOT)}")
    except Exception as e:                                        # never the pipeline's problem
        from model.serve import log_warns
        msg = f"{type(e).__name__}: {e}"[:400]
        print(f"model.retrospective FAILED, outputs untouched: {msg}", file=sys.stderr)
        traceback.print_exc()
        log_warns(ROOT, [f"retrospective failed, nothing written: {msg}"], check=CHECK)
    return 0


if __name__ == "__main__":
    sys.exit(main())
