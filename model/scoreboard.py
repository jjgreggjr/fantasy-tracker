"""The live scoreboard: does the model rank players better than Sleeper and E_pts, judged on real league points?

    python -m model.scoreboard                # score every completed week, update data/model_eval.csv and the report
    python -m model.scoreboard --print        # also print the report

Adoption gate, written down at Phase 3 and never moved: the model earns recipe preference only when it beats Sleeper's
projection on pick accuracy over at least 6 completed 2026 weeks. Until then E_pts stays authoritative.

What is scored, per league and completed week (a week is complete once the pipeline has recorded it in
data/lineups_played.csv, which only ever holds finished weeks):

  actual        the platform's own points for each rostered QB / RB / WR / TE, in that league's scoring
                (data/lineups_played.csv). Only players who played (an nflverse stats row in data/player_weeks.csv, or nonzero
                points) are scored: the model predicts points IF he plays, and availability is a separate layer.
  comparators   five predictions of those points, all frozen BEFORE kickoff, all in that league's scoring, taken from the
                row data/model_pts.csv froze at serve time (nothing is recomputed here):
                  model_components  E_pts_model: the 14 component models composed under the league's scoring (`pts_<slug>`)
                  model_flat        the 3-library PPR average, moved to the league's scoring by the same components' scoring
                                    difference (`pts_model + pts_<slug> - pts_model_components`; identical to `pts_model` in a
                                    full-PPR league)
                  model_blend       the 50/50 average of the two above
                  sleeper           Sleeper's projection in the league's scoring (`sleeper_proj_<slug>`)
                  e_pts             the pipeline's E_pts for the league (`e_pts_<slug>`); a ranking of opportunity, not a point
                                    projection, so only rank metrics are compared
  rows          every comparator is scored on the IDENTICAL rows (a player missing any comparator is dropped from all of them),
                and the coverage table says how many played players that costs.
  metrics       Spearman within position-week-league and pick accuracy: over every same-position pair with different actual
                points in the same league-week, the share where the higher prediction scored more (a tied prediction counts
                half). Pick accuracy is the head-to-head number the gate uses: it is what a start / sit decision is.

`data/model_eval.csv` keeps one row per (season, week, league, position, comparator) with the additive pieces (`credit`, `pairs`),
so any pooling can be recomputed. A week's rows are replaced when it is re-scored (`ff.build.replace_partition`), so the file
converges on the recorded results and never doubles up.

Like model.serve it can never break the pipeline: any failure is a WARN row (check `model.scoreboard`) and nothing is written.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = ROOT / "data" / "model_eval.csv"
REPORT = ROOT / "model" / "reports" / "live_scoreboard.md"
CHECK = "model.scoreboard"

GATE = ("Adoption gate: the model earns recipe preference only when it beats Sleeper's projection on pick accuracy "
        "over at least 6 completed 2026 weeks.")
GATE_WEEKS = 6
GATE_MODEL = "model_components"          # E_pts_model, the number roster.csv and the recipes show
COMPARATORS = ("model_components", "model_flat", "model_blend", "sleeper", "e_pts")
POSITIONS = ("QB", "RB", "WR", "TE")
KEYS = ["season", "week", "league", "position", "comparator"]
EVAL_COLS = KEYS + ["n", "spearman", "credit", "pairs", "pick_acc"]
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 2000, 20260930


# --------------------------------------------------------------------------- inputs
def read_csv(path: Path, **kw):
    import pandas as pd
    return pd.read_csv(path, low_memory=False, dtype={"gsis_id": str, "league_id": str}, **kw) if Path(path).exists() else None


def league_index(root: Path) -> dict:
    """{slug: (league_id, name)} from every leagues/*/league.json."""
    out = {}
    for p in sorted((root / "leagues").glob("*/league.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out[d.get("slug") or p.parent.name] = (str(d["league_id"]), d.get("name") or p.parent.name)
    return out


def comparators(mp, slug: str):
    """gsis_id + the five comparators for one league, from the frozen rows of one week of model_pts.csv (NaN where a column is
    missing, which drops the row from the identical-rows set)."""
    import numpy as np
    import pandas as pd
    col = lambda c: pd.to_numeric(mp[c], errors="coerce") if c in mp.columns else pd.Series(np.nan, index=mp.index)
    comp, flat, comp_ppr = col(f"pts_{slug}"), col("pts_model"), col("pts_model_components")
    out = pd.DataFrame({"gsis_id": mp["gsis_id"].astype(str),
                        "model_components": comp,
                        "model_flat": flat + (comp - comp_ppr),
                        "model_blend": comp + 0.5 * (flat - comp_ppr),
                        "sleeper": col(f"sleeper_proj_{slug}"),
                        "e_pts": col(f"e_pts_{slug}")})
    return out.drop_duplicates("gsis_id")


def actuals(lp, pw, league_id: str, season: int, week: int):
    """Played QB/RB/WR/TE of a league-week with the platform's points."""
    import pandas as pd
    a = lp[(lp["league_id"].astype(str) == league_id) & (lp["season"] == season) & (lp["week"] == week)]
    a = a[a["position"].isin(POSITIONS) & a["gsis_id"].notna()].drop_duplicates("gsis_id")
    seen = set()
    if pw is not None and len(pw):
        w = pw[(pw["season"] == season) & (pw["week"] == week)]
        seen = set(w["gsis_id"].dropna().astype(str))
    a = a.assign(played=a["gsis_id"].astype(str).isin(seen) | (pd.to_numeric(a["points"], errors="coerce").fillna(0) != 0))
    return a[["gsis_id", "position", "points", "played"]].rename(columns={"points": "actual"})


# --------------------------------------------------------------------------- metrics
def score_rows(e, season: int, week: int, league: str):
    """Eval rows (one per position and comparator, plus ALL) for the identical rows `e` (columns: position, actual, COMPARATORS)."""
    import numpy as np
    import pandas as pd
    from model import backtest as B
    rows = []
    for comp in COMPARATORS:
        per = []
        for pos in POSITIONS:
            g = e[e["position"] == pos]
            y, p = g["actual"].to_numpy(float), g[comp].to_numpy(float)
            credit, pairs = B._pairs(y, p)
            r = {"season": season, "week": week, "league": league, "position": pos, "comparator": comp, "n": len(g),
                 "spearman": B._rank_corr(p, y), "credit": credit, "pairs": pairs}
            r["pick_acc"] = credit / pairs if pairs else np.nan
            per.append(r)
        rows += per
        cr, pa = sum(r["credit"] for r in per), sum(r["pairs"] for r in per)
        sp = [r["spearman"] for r in per if r["spearman"] == r["spearman"]]
        rows.append({"season": season, "week": week, "league": league, "position": "ALL", "comparator": comp,
                     "n": sum(r["n"] for r in per), "spearman": float(np.mean(sp)) if sp else np.nan, "credit": cr,
                     "pairs": pa, "pick_acc": cr / pa if pa else np.nan})
    return pd.DataFrame(rows, columns=EVAL_COLS)


def evaluate(root: Path = ROOT):
    """(eval rows for every scorable league-week, coverage rows, availability rows). Reads only committed files."""
    import pandas as pd
    mp = read_csv(root / "data" / "model_pts.csv")
    lp = read_csv(root / "data" / "lineups_played.csv")
    pw = read_csv(root / "data" / "player_weeks.csv")
    leagues = league_index(root)
    empty = pd.DataFrame(columns=EVAL_COLS)
    if mp is None or mp.empty or lp is None or lp.empty:
        return empty, pd.DataFrame(), pd.DataFrame()
    ev, cov, avail = [], [], []
    for (season, week), frozen in mp.groupby(["season", "week"]):
        season, week = int(season), int(week)
        for slug, (lid, _) in leagues.items():
            act = actuals(lp, pw, lid, season, week)
            have = {"season": season, "week": week, "league": slug, "model_rows": len(frozen), "actual_rows": len(act)}
            for c in COMPARATORS:
                have[c] = int(comparators(frozen, slug)[c].notna().sum())
            avail.append(have)
            if act.empty:
                continue                                              # the week is not complete for this league yet
            played = act[act["played"]]
            pred = comparators(frozen, slug)
            both = played.merge(pred, on="gsis_id", how="left")
            e = both.dropna(subset=list(COMPARATORS) + ["actual"])
            cov.append({"season": season, "week": week, "league": slug, "played": len(played),
                        "with_model_row": int(both["model_components"].notna().sum()), "scored": len(e)})
            if len(e) >= 3:
                ev.append(score_rows(e, season, week, slug))
    evd = pd.concat(ev, ignore_index=True) if ev else empty
    return evd, pd.DataFrame(cov), pd.DataFrame(avail)


# --------------------------------------------------------------------------- summary and the gate
def pooled(ev, position: str = "ALL"):
    """{comparator: dict(n, pairs, pick_acc, spearman)} pooled over every scored league-week."""
    import numpy as np
    out = {}
    d = ev[ev["position"] == position]
    for c in COMPARATORS:
        g = d[d["comparator"] == c]
        if g.empty:
            continue
        out[c] = {"n": int(g["n"].sum()), "pairs": int(g["pairs"].sum()),
                  "pick_acc": float(g["credit"].sum() / g["pairs"].sum()) if g["pairs"].sum() else np.nan,
                  "spearman": float(np.nanmean(g["spearman"])) if g["spearman"].notna().any() else np.nan}
    return out


def gate_status(ev) -> dict:
    """The adoption gate, mechanically: weeks scored, the model's and Sleeper's pooled pick accuracy on identical rows, verdict."""
    import numpy as np
    d = ev[(ev["position"] == "ALL") & (ev["pairs"] > 0)]
    weeks = sorted({(int(s), int(w)) for s, w in zip(d["season"], d["week"])})
    p = pooled(ev)
    m, s = p.get(GATE_MODEL), p.get("sleeper")
    diff = (m["pick_acc"] - s["pick_acc"]) if m and s else np.nan
    met = bool(len(weeks) >= GATE_WEEKS and diff == diff and diff > 0)
    return {"weeks": len(weeks), "needed": GATE_WEEKS, "model": m["pick_acc"] if m else np.nan,
            "sleeper": s["pick_acc"] if s else np.nan, "diff": diff, "met": met, "week_list": weeks}


def week_bootstrap(ev, a: str, b: str):
    """95% interval of pooled pick accuracy (a minus b), resampling whole weeks (players within a week are correlated)."""
    import numpy as np
    d = ev[ev["position"] == "ALL"]
    wk = sorted({(int(s), int(w)) for s, w in zip(d["season"], d["week"])})
    if len(wk) < 3:
        return None
    by = {c: d[d["comparator"] == c].groupby(["season", "week"])[["credit", "pairs"]].sum().reindex(wk).fillna(0.0) for c in (a, b)}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(wk), size=(BOOTSTRAP_DRAWS, len(wk)))
    cr_a, cr_b, pr = by[a]["credit"].to_numpy(), by[b]["credit"].to_numpy(), by[a]["pairs"].to_numpy()
    den = pr[idx].sum(axis=1)
    ok = den > 0
    diff = (cr_a[idx].sum(axis=1)[ok] - cr_b[idx].sum(axis=1)[ok]) / den[ok]
    return float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


def _f(x, nd: int = 3) -> str:
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


def render(ev, cov, avail, gate: dict) -> str:
    """The report. Everything in it is a function of the committed inputs (no clock), so a rerun changes nothing."""
    L = ["# Live scoreboard", "", f"**{GATE}**", ""]
    if ev.empty:
        L += [f"**Gate status: not started.** No completed week has frozen model predictions yet (0 of {GATE_WEEKS} weeks).", ""]
    else:
        wk = ", ".join(f"{s} wk{w}" for s, w in gate["week_list"])
        L += [f"**Gate status: {'MET' if gate['met'] else 'NOT MET'}.** {gate['weeks']} of {GATE_WEEKS} completed weeks scored ({wk}). "
              f"Pooled pick accuracy on identical rows: model (E_pts_model) {_f(gate['model'])}, Sleeper {_f(gate['sleeper'])} "
              f"(difference {_f(gate['diff'])}). "
              + ("" if gate["weeks"] >= GATE_WEEKS else f"The gate cannot be met before {GATE_WEEKS} weeks are scored, whatever the difference. "),
              ""]
        ci = week_bootstrap(ev, GATE_MODEL, "sleeper")
        if ci:
            L += [f"Week-blocked 95% interval for that difference: [{ci[0]:+.3f}, {ci[1]:+.3f}] (informational; the gate is the point "
                  "estimate over 6+ weeks, as written).", ""]
    L += ["## What is scored", "",
          "Each completed week, each league, every rostered QB/RB/WR/TE who played, against the platform's own points in that league's "
          "scoring (`data/lineups_played.csv`). Five predictions, all frozen before kickoff in `data/model_pts.csv`, all in that league's "
          "scoring, scored on the **identical rows**:", "",
          "| comparator | what it is |", "|---|---|",
          "| `model_components` | E_pts_model: the 14 component models composed under the league's scoring (the gate's model) |",
          "| `model_flat` | the 3-library PPR average, moved to the league's scoring by the components' scoring difference |",
          "| `model_blend` | 50/50 average of the two |",
          "| `sleeper` | Sleeper's projection in the league's scoring, captured at serve time |",
          "| `e_pts` | the pipeline's E_pts for the league, captured at serve time (a ranking of opportunity, not a point projection) |", "",
          "Metrics: **pick accuracy** = over all same-position pairs in a league-week with different actual points, the share where the "
          "higher prediction scored more (ties count half); **Spearman** = rank correlation within position-week-league, averaged. "
          "Played = an nflverse stats row that week or nonzero points: the model predicts points if he plays, availability is a "
          "separate layer, and a player ruled Out has no model row.", ""]
    if ev.empty:
        L += ["## Results", "", "Nothing to score yet. The first frozen week is the one `data/model_pts.csv` holds; it is scored by the "
              "first run after its last game, when the pipeline records that week in `data/lineups_played.csv`.", ""]
    else:
        L += ["## Results, all scored weeks pooled", "", _table(pooled(ev)), "", "### By position", ""]
        for pos in POSITIONS:
            L += [f"**{pos}**", "", _table(pooled(ev, pos)), ""]
        L += ["### By week (pick accuracy, pooled over leagues)", "", _by(ev, "week"), "", "### By league (pick accuracy, pooled over weeks)", "",
              _by(ev, "league"), ""]
    if len(cov):
        c = cov.groupby("week")[["played", "with_model_row", "scored"]].sum()
        L += ["## Coverage", "",
              "Played rostered skill players per week (summed over leagues), how many had a model row, and how many made the identical-rows set "
              "(a player missing the model, Sleeper or E_pts is dropped from every comparator):", "",
              "| week | played | with a model row | scored |", "|---|---|---|---|"]
        L += [f"| {int(w)} | {int(r.played)} | {int(r.with_model_row)} | {int(r.scored)} |" for w, r in c.iterrows()]
        L.append("")
    if len(avail):
        L += ["## Which comparators exist", "",
              "Non-blank frozen values per league-week in `data/model_pts.csv` (the actual-points column is the recorded league rows):", "",
              "| season | week | league | model rows | actual rows | " + " | ".join(COMPARATORS) + " |",
              "|---|---|---|---|---|" + "---|" * len(COMPARATORS)]
        L += [f"| {r.season} | {r.week} | {r.league} | {r.model_rows} | {r.actual_rows} | " + " | ".join(str(getattr(r, c)) for c in COMPARATORS) + " |"
              for r in avail.sort_values(["season", "week", "league"]).itertuples()]
        L.append("")
    L += ["## Limits", "",
          "- Frozen means frozen at the last run before each game's kickoff; weeks served before this file existed have no rows and are not scored.",
          "- Few weeks are noise. Pick accuracy moves by several points from week to week; read the interval, not the ranking.",
          "- Snap-only appearances (a player who played and scored zero without a stats row) and non-appearances are not scored.",
          "- The IDP league is scored on skill positions only, like everything else in the repo; bonuses the composition cannot express "
          "(`python -m model.serve --columns`) are in the platform's actual points but not in the model's.", ""]
    return "\n".join(L)


def _table(p: dict) -> str:
    rows = ["| comparator | players | pairs | pick accuracy | Spearman |", "|---|---|---|---|---|"]
    rows += [f"| `{c}` | {v['n']} | {v['pairs']} | {_f(v['pick_acc'])} | {_f(v['spearman'])} |" for c, v in p.items()]
    return "\n".join(rows)


def _by(ev, key: str) -> str:
    d = ev[ev["position"] == "ALL"]
    groups = sorted(d[key].unique())
    rows = [f"| {key} | " + " | ".join(COMPARATORS) + " |", "|---|" + "---|" * len(COMPARATORS)]
    for g in groups:
        s = d[d[key] == g]
        cells = []
        for c in COMPARATORS:
            x = s[s["comparator"] == c]
            cells.append(_f(x["credit"].sum() / x["pairs"].sum()) if x["pairs"].sum() else "n/a")
        rows.append(f"| {g} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


# --------------------------------------------------------------------------- entry point
def run(root: Path = ROOT, *, write: bool = True) -> str:
    from ff import build
    ev, cov, avail = evaluate(root)
    text = render(ev, cov, avail, gate_status(ev) if not ev.empty else {})
    if write:
        if not ev.empty:
            build.replace_partition(root / "data" / "model_eval.csv", ev, ["season", "week"], KEYS)
        out = root / "model" / "reports" / "live_scoreboard.md"
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
            print(f"model.scoreboard: wrote {REPORT.relative_to(ROOT)}")
    except Exception as e:                                        # never the pipeline's problem
        from model.serve import log_warns
        msg = f"{type(e).__name__}: {e}"[:400]
        print(f"model.scoreboard FAILED, outputs untouched: {msg}", file=sys.stderr)
        traceback.print_exc()
        log_warns(ROOT, [f"scoreboard failed, nothing written: {msg}"], check=CHECK)
    return 0


if __name__ == "__main__":
    sys.exit(main())
