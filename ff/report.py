"""Markdown reports: one per league, plus a short cross-league digest."""
from __future__ import annotations

import pandas as pd

PCT = lambda v: "" if pd.isna(v) else f"{v:.0%}"
NUM = lambda v, d=1: "" if pd.isna(v) else f"{v:.{d}f}"
INT = lambda v: "" if pd.isna(v) else str(int(v))


def _table(df: pd.DataFrame, cols: dict, empty: str) -> str:
    if df is None or df.empty:
        return f"_{empty}_\n"
    lines = ["| " + " | ".join(cols.values()) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(
            "" if pd.isna(r.get(c, "")) else str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Pts"] = d.E_pts.map(lambda v: NUM(v, 1))
    d["Opps"] = d.E_opps.map(lambda v: NUM(v, 1))
    d["Tgt"] = d.E_targets.map(lambda v: NUM(v, 1))
    d["Car"] = d.E_carries.map(lambda v: NUM(v, 1))
    share = d.apply(lambda r: r.get("proj_carry_share") if r.position == "RB"
                    else r.get("proj_target_share"), axis=1)
    d["Share"] = share.map(PCT)
    d["DvP"] = d.dvp_rank.map(INT)
    d["Depth"] = d.depth_rank.map(INT)
    d["Opp"] = d.opponent.fillna("—")
    d["Age"] = d.age.map(lambda v: NUM(v, 1)) if "age" in d else ""
    return d


ROSTER_COLS = {"name": "Player", "position": "Pos", "team": "Tm", "Opp": "Opp",
               "Pts": "E_pts", "Opps": "Opps", "Share": "Share", "Depth": "Depth",
               "DvP": "DvP", "conf": "Conf", "flag": "Note"}


def render_league(lg: dict, ctx: dict) -> str:
    meta = lg["meta"]
    L, a = [], None
    a = L.append
    a(f"# {meta.get('name')} — {ctx['season']} Week {ctx['preview_week']}\n")
    a(f"{meta.get('type','?')} · {meta.get('total_slots','?')} roster slots"
      + (f" · IDP" if meta.get("idp") else "") + "  ")
    a(f"Generated {ctx['now']} · baselines: {ctx['baseline_note']}"
      f" · lineup as of {meta.get('roster_fetched_at') or '?'}\n")
    a("`E_pts` is expected points **in this league's scoring**: projected stat "
      "line blended with recent production, then adjusted ±12% for the matchup. "
      "**DvP rank 1 = softest matchup.** It is a ranking of opportunity, not a "
      "point projection.\n")

    if ctx["warnings"]:
        a("\n> " + " · ".join(ctx["warnings"]) + "\n")

    sw = lg.get("status_warnings") or []
    a("\n## Status warnings — read before setting a lineup\n")
    if sw:
        a("A depth-chart rank can reflect *availability* rather than role. "
          "These are the cases where that is happening on your roster.\n")
        for w in sw[:20]:
            a(f"- {w}")
        a("")
    else:
        a("_None. No depth-chart disagreements, no blocked players in doubt._\n")

    a("\n## Your roster\n")
    r = lg["roster"]
    a(_table(_prep(r) if not r.empty else r, ROSTER_COLS,
             "No roster loaded for this league."))

    if not r.empty:
        a("\n### Why each call\n")
        for _, p in _prep(r).head(20).iterrows():
            if p.get("why"):
                a(f"- **{p['name']}** — {p['why']}")
        a("")

    a("\n## Waiver wire — best available\n")
    av = lg["available"]
    if not av.empty:
        top = (av[av.E_pts.notna()].sort_values("E_pts", ascending=False)
               .groupby("position").head(6))
        top = _prep(top)
        if "trending_add" in top.columns:
            top["Hot"] = top.trending_add.map(lambda v: "🔥" if v else "")
        cols = dict(ROSTER_COLS)
        cols.pop("flag", None)
        cols["Hot"] = "Trending"
        a(_table(top.sort_values(["position", "Pts"], ascending=[True, False]),
                 cols, "none"))
    else:
        a("_No free-agent pool computed._\n")

    a("\n## Dynasty watchlist — young, ascending, unrostered here\n")
    wl = lg.get("watchlist", pd.DataFrame())
    if wl is not None and not wl.empty:
        w = wl.copy()
        w["Age"] = w.age.map(lambda v: NUM(v, 1))
        w["Depth"] = w.apply(lambda r: f"{INT(r.depth_rank)}"
                             + (f" (was {INT(r.prev_rank)})"
                                if pd.notna(r.prev_rank) else ""), axis=1)
        w["Snap%"] = w.avg_snap_pct.map(PCT)
        a(_table(w.head(15), {"name": "Player", "position": "Pos", "team": "Tm",
                              "Age": "Age", "years_exp": "Yrs", "Depth": "Depth",
                              "Snap%": "Snap%", "why": "Why"}, "none"))
    else:
        a("_None met the criteria._\n")

    a("\n## Ask me\n")
    a("From a Cowork session with your Projects folder connected, or in a terminal:\n")
    a("```")
    a(f"python -m ff.ask lineup  {meta.get('slug')}")
    a(f"python -m ff.ask cut     {meta.get('slug')} 3")
    a(f'python -m ff.ask options {meta.get("slug")} "Player Name"')
    a(f'python -m ff.ask explain {meta.get("slug")} "Player Name"')
    a(f'python -m ff.ask trade   {meta.get("slug")}')
    a(f'python -m ff.ask trade   {meta.get("slug")} "Player Name"')
    a("```")
    return "\n".join(L)


def render_digest(ctx: dict) -> str:
    L = []
    a = L.append
    a(f"# Week {ctx['preview_week']} digest — {ctx['season']}\n")
    a(f"Generated {ctx['now']}  ")
    sw = ctx.get("stats_week")
    a(f"Stats through: **{'Week ' + str(sw) if sw else 'no games yet'}** · "
      f"Baselines: **{ctx['baseline_note']}**\n")

    a("\n## Data freshness\n")
    for k, v in ctx["freshness"].items():
        a(f"- {k}: {v}")
    if ctx["warnings"]:
        a("\n### Warnings\n")
        for w in ctx["warnings"]:
            a(f"- {w}")

    checks = ctx.get("checks") or []
    bad = [c for c in checks if c["severity"] != "OK"]
    a("\n### Data integrity\n")
    if not checks:
        a("_not run_")
    elif not bad:
        a(f"All {len(checks)} checks passed.")
    else:
        for c in bad:
            a(f"- **{c['severity']}** {c['check']} — {c['detail']}")
    a("")

    for lg in ctx["leagues"]:
        meta = lg["meta"]
        a(f"\n## {meta.get('name')}\n")
        r = lg["roster"]
        if r is None or r.empty:
            a("_No roster loaded._\n")
            continue
        d = _prep(r)
        playable = d[d.E_pts.notna()]
        if not playable.empty:
            a("**Top of your roster this week**\n")
            a(_table(playable.head(5), ROSTER_COLS, "none"))
        flagged = d[d.flag.ne("") & d.flag.notna()]
        if not flagged.empty:
            a("\n**Needs a decision:** "
              + ", ".join(f"{r['name']} ({r.flag})" for _, r in flagged.iterrows())
              + "\n")
        av = lg.get("available", pd.DataFrame())
        if av is not None and not av.empty and "trending_add" in av.columns:
            hot = av[(av.trending_add == 1) & av.E_pts.notna()].head(3)
            if not hot.empty:
                a("**Trending and still free here:** "
                  + ", ".join(f"{r['name']} ({r.position}, {NUM(r.E_pts,1)})"
                              for _, r in hot.iterrows()) + "\n")
        a(f"\nFull report: `leagues/{meta.get('slug')}/report.md`\n")

    a("\n---\n\n# Defense vs position\n")
    a("Per-game production **allowed**. Rank 1 = softest.\n")
    for pos in ["RB", "WR", "TE", "QB"]:
        d = ctx["dvp"][ctx["dvp"].position == pos].copy()
        if d.empty:
            continue
        a(f"\n### vs {pos} — 8 softest\n")
        d = d.head(8)
        d["Rank"] = d.dvp_rank.map(INT)
        d["PPR"] = d.fpts.map(lambda v: NUM(v, 1))
        d["Tgt"] = d.tgts.map(lambda v: NUM(v, 1))
        d["RecYd"] = d.recyds.map(lambda v: NUM(v, 1))
        d["RecTD"] = d.rectds.map(lambda v: NUM(v, 2))
        d["Car"] = d.car.map(lambda v: NUM(v, 1))
        d["RushYd"] = d.rushyds.map(lambda v: NUM(v, 1))
        d["RushTD"] = d.rushtds.map(lambda v: NUM(v, 2))
        a(_table(d, {"Rank": "Rank", "defense": "Def", "PPR": "PPR/g",
                     "Tgt": "Tgt", "RecYd": "RecYd", "RecTD": "RecTD",
                     "Car": "Car", "RushYd": "RushYd", "RushTD": "RushTD"}, "none"))
    return "\n".join(L)
