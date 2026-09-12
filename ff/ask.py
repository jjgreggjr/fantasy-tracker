"""Question recipes.

These are what a Cowork session runs to answer James's standard questions.
Each takes a league slug, reads only that league's files plus shared data,
and returns a small frame plus a sentence. Read-only by design.

    python -m ff.ask live where-you-at      # current Sleeper lineup, live
    python -m ff.ask lineup where-you-at
    python -m ff.ask cut where-you-at 3
    python -m ff.ask options where-you-at "Kyle Pitts"
    python -m ff.ask explain where-you-at "Kyle Pitts"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import live as live_mod, policy, scoring, status as status_mod, trade as trade_mod

ROOT = Path(__file__).resolve().parent.parent


def _status():
    """The status table, or an empty frame if the pipeline has not built it."""
    p = ROOT / "data" / "status.csv"
    return pd.read_csv(p, low_memory=False) if p.exists() else pd.DataFrame()


def _status_block(st: pd.DataFrame, names) -> list[str]:
    """One status line per named player — printed with every recommendation."""
    if st is None or st.empty:
        return ["(no status data — run `python -m ff.run_weekly` to build it)"]
    out = []
    for n in names:
        line = status_mod.describe(st, n)
        if line:
            out.append(f"  {n}: {line}")
    return out


def _load(slug: str):
    d = ROOT / "leagues" / slug
    if not d.exists():
        avail = sorted(p.name for p in (ROOT / "leagues").glob("*")) if (ROOT / "leagues").exists() else []
        raise SystemExit(f"No league '{slug}'. Available: {', '.join(avail) or 'none'}")
    meta = json.loads((d / "league.json").read_text(encoding="utf-8"))
    get = lambda n: (pd.read_csv(d / n, low_memory=False)
                     if (d / n).exists() else pd.DataFrame())
    return meta, get("roster.csv"), get("available.csv"), get("all_rosters.csv"), d


def lineup(slug: str) -> tuple[pd.DataFrame, str]:
    """Fill every startable slot greedily by E_pts, honoring eligibility."""
    meta, roster, avail, _, _ = _load(slug)
    if roster.empty:
        return pd.DataFrame(), "No roster data for this league."
    all_slots = scoring.starting_slots(meta)
    # slots we cannot fill because those positions are out of scope (IDP, K, DST)
    slots = [s for s in all_slots if scoring.SLOT_ELIGIBILITY.get(s)]
    unscored = [s for s in all_slots if not scoring.SLOT_ELIGIBILITY.get(s)]
    pool = roster[roster.E_pts.notna()].sort_values("E_pts", ascending=False).copy()
    used, picks = set(), []

    # scarcest slots first so a SUPER_FLEX doesn't eat the only eligible QB
    order = sorted(range(len(slots)),
                   key=lambda i: len(scoring.SLOT_ELIGIBILITY.get(slots[i], set())))
    for i in order:
        slot = slots[i]
        for _, p in pool.iterrows():
            if p.gsis_id in used or not scoring.slot_allows(slot, p.position):
                continue
            picks.append({"slot": slot, "name": p["name"], "position": p.position,
                          "team": p.team, "opponent": p.get("opponent"),
                          "E_pts": p.E_pts, "conf": p.get("conf"),
                          "why": p.get("why")})
            used.add(p.gsis_id)
            break

    start = pd.DataFrame(picks)
    bench = roster[~roster.gsis_id.isin(used)].copy()
    if not bench.empty and not start.empty:
        gaps = []
        for _, b in bench.iterrows():
            elig = start[start.slot.map(lambda s: scoring.slot_allows(s, b.position))]
            gaps.append(round(b.E_pts - elig.E_pts.min(), 1)
                        if not elig.empty and pd.notna(b.E_pts) else np.nan)
        bench["gap_to_starter"] = gaps
        bench = bench.sort_values("gap_to_starter", ascending=False)
    if "flag" in bench.columns:
        bench["flag"] = bench.flag.fillna("")

    shaky = start[start.conf.eq("Low")] if not start.empty else pd.DataFrame()
    note = f"{len(start)} of {len(slots)} scoreable slots filled."
    if unscored:
        note += (f" {len(unscored)} slot(s) not scored here ("
                 + ", ".join(sorted(set(unscored))) + ") — K, DST and IDP are out of scope.")
    if not shaky.empty:
        note += (" Low confidence: "
                 + ", ".join(shaky["name"].tolist())
                 + " — check the bench alternatives above them.")
    close = bench[bench.gap_to_starter.between(-1.5, 0)] if "gap_to_starter" in bench else pd.DataFrame()
    if not close.empty:
        note += " Near-ties on the bench: " + ", ".join(close["name"].head(3).tolist()) + "."
    return start, note, bench


def roster_overage(slug: str) -> tuple[int, dict]:
    """How many players must go to reach the league's limit."""
    meta, roster, avail, allr, d = _load(slug)
    import pandas as _pd
    full = None
    p = ROOT / "data" / "my_roster.csv"
    if p.exists():
        mr = _pd.read_csv(p, dtype={"sleeper_id": str})
        full = mr[mr.league_id.astype(str) == str(meta["league_id"])]
    if full is None or full.empty:
        return 0, {}
    ir = int(full.is_ir.sum()); taxi = int(full.is_taxi.sum())
    active = len(full) - ir - taxi
    cap = meta.get("total_slots") or 0
    info = {"rostered": len(full), "ir": ir, "taxi": taxi, "active": active,
            "cap": cap, "taxi_slots": meta.get("taxi_slots", 0),
            "ir_slots": meta.get("reserve_slots", 0),
            "non_skill": int(full.position.isna().sum())}
    return max(0, active - cap), info


def cut(slug: str, n: int | None = None) -> tuple[pd.DataFrame, str]:
    """Rank the roster by keep-value; the lowest N are the cut candidates.

    Keep-value is not the same as this week's score. It weighs production over
    the waiver wire, remaining age runway from the position's aging curve, and
    usage trend — with the mix set by league type and bench depth.
    """
    meta, roster, avail, allr, _ = _load(slug)
    if roster.empty:
        return pd.DataFrame(), "No roster data for this league."

    over, info = roster_overage(slug)
    if n is None:
        n = over if over else 1

    r = policy.keep_value(roster, avail, meta)
    w = r.attrs["weights"]

    # never suggest cutting a current starter
    starters = set()
    try:
        st, _, _ = lineup(slug)
        starters = set(st["name"]) if not st.empty else set()
    except Exception:
        pass
    r["in_lineup"] = r["name"].isin(starters).astype(int)

    pool = r[(r.in_lineup == 0)].sort_values(["stash", "keep_value"])
    cands = pool.head(n).copy()
    cands["why_cut"] = [policy.explain_keep(x, w) for _, x in cands.iterrows()]

    # who else in the league would want him — cut vs trade is a real choice
    depth_p = ROOT / "data" / "depth_charts.csv"
    depth_df = pd.read_csv(depth_p) if depth_p.exists() else pd.DataFrame()
    me = ""
    mr = ROOT / "data" / "my_roster.csv"
    if mr.exists():
        _m = pd.read_csv(mr, dtype={"sleeper_id": str})
        _m = _m[_m.league_id.astype(str) == str(meta["league_id"])]
        if not _m.empty:
            me = _m.owner_name.iloc[0]
    cands["trade_interest"] = [
        policy.trade_interest(x, allr, depth_df, me) for _, x in cands.iterrows()]
    reps = []
    for _, p_ in cands.iterrows():
        fa = avail[avail.position == p_.position] if not avail.empty else pd.DataFrame()
        best = fa.sort_values("E_pts", ascending=False).head(1)
        reps.append(f"{best.iloc[0]['name']} ({best.iloc[0].E_pts:.1f})"
                    if not best.empty else "—")
    cands["best_replacement"] = reps

    note = (f"{meta.get('name')} · {meta.get('type')} · "
            f"{info.get('cap','?')} active slots"
            + (f" (+{info['taxi_slots']} taxi, +{info['ir_slots']} IR)"
               if info else "") + ".")
    if info:
        note += (f" You have {info['rostered']} rostered "
                 f"({info['active']} needing an active slot"
                 + (f", {info['non_skill']} K/DEF" if info.get("non_skill") else "")
                 + f"). Over by {over}." if over else
                 f" You have {info['rostered']} rostered — within the limit.")
    note += (f" Keep-value mix: {int(w['pts']*100)}% value-over-replacement, "
             f"{int(w['age']*100)}% age runway, {int(w['trend']*100)}% usage trend "
             f"({w['label']}"
             + (f", bench/starters {w['bench_ratio']}" if "bench_ratio" in w else "")
             + ").")
    if meta.get("idp"):
        note += " IDP players count toward the limit but are not scored here."
    return cands, note


def options(slug: str, player: str, top: int = 8) -> tuple[pd.DataFrame, str]:
    """Free agents at the same position, plus other owners' benched players."""
    meta, roster, avail, allr, _ = _load(slug)
    hit = roster[roster["name"].str.contains(player, case=False, na=False)]
    if hit.empty and not allr.empty:
        hit = allr[allr["name"].str.contains(player, case=False, na=False)]
    if hit.empty:
        return pd.DataFrame(), f"No player matching '{player}' in this league."
    p = hit.iloc[0]
    fa = (avail[avail.position == p.position].sort_values("E_pts", ascending=False)
          .head(top).copy() if not avail.empty else pd.DataFrame())
    if not fa.empty:
        fa["vs_him"] = (fa.E_pts - p.E_pts).round(1)
        fa["source"] = "free agent"
    bench = pd.DataFrame()
    if not allr.empty and "is_starter" in allr.columns:
        bench = allr[(allr.position == p.position) & (allr.is_starter == 0)
                     & (allr.owner_name != p.get("owner_name"))].copy()
        bench = bench.sort_values("E_pts", ascending=False).head(5)
        if not bench.empty:
            bench["vs_him"] = (bench.E_pts - p.E_pts).round(1)
            bench["source"] = "on " + bench.owner_name.astype(str) + "'s bench"
    cols = [c for c in ["name", "position", "team", "opponent", "E_pts", "E_opps",
                        "conf", "vs_him", "source", "why"] if c in fa.columns
            or c in bench.columns]
    out = pd.concat([fa, bench], ignore_index=True)
    out = out[[c for c in cols if c in out.columns]] if not out.empty else out
    return out, (f"{p['name']} ({p.position}, {p.team}) projects "
                 f"{p.E_pts if pd.notna(p.E_pts) else 'n/a'} this week. {p.get('why','')}")


def trade(slug: str, player: str | None = None):
    """Who wants your player and what to ask for — or a survey of the market."""
    meta, roster, avail, allr, _ = _load(slug)
    if roster.empty or allr.empty:
        return None, "Need both a roster and league rosters for this league."
    depth_p = ROOT / "data" / "depth_charts.csv"
    depth = pd.read_csv(depth_p) if depth_p.exists() else pd.DataFrame()
    me = ""
    mr = ROOT / "data" / "my_roster.csv"
    if mr.exists():
        _m = pd.read_csv(mr, dtype={"sleeper_id": str})
        _m = _m[_m.league_id.astype(str) == str(meta["league_id"])]
        if not _m.empty:
            me = _m.owner_name.iloc[0]
    if player:
        p, deals = trade_mod.shop(player, roster, allr, avail, depth, meta, me)
        return (p, deals), meta
    sell, mine = trade_mod.survey(roster, allr, avail, depth, meta, me)
    return (sell, mine, trade_mod.owner_profiles(allr)), meta


def explain(slug: str, player: str) -> tuple[pd.DataFrame, str]:
    """Full component breakdown for one player — the 'that looks wrong' tool."""
    meta, roster, avail, allr, _ = _load(slug)
    frames = [f for f in (roster, avail, allr) if not f.empty]
    hit = pd.DataFrame()
    for f in frames:
        m = f[f["name"].str.contains(player, case=False, na=False)]
        if not m.empty:
            hit = m
            break
    if hit.empty:
        return pd.DataFrame(), f"No player matching '{player}'."
    p = hit.iloc[0]
    rows = [
        ("Expected opportunities", p.get("E_opps"),
         f"projection {p.get('proj_opps')} x {p.get('w_proj')} + recent {p.get('avg_opps')} x {round(1-(p.get('w_proj') or 0),2)}"),
        ("Expected targets", p.get("E_targets"), ""),
        ("Expected carries", p.get("E_carries"), ""),
        ("Share of team volume",
         p.get("proj_carry_share") if p.position == "RB" else p.get("proj_target_share"),
         "carries" if p.position == "RB" else "targets"),
        ("Depth chart rank", p.get("depth_rank"), f"was {p.get('prev_rank')}"),
        ("Opponent", p.get("opponent"), f"DvP rank {p.get('dvp_rank')} (1 = softest)"),
        ("Matchup multiplier", p.get("mult"), f"{round(((p.get('mult') or 1)-1)*100,1)}%"),
        ("Projected pts (league scoring)", p.get("proj_pts_league"), ""),
        ("Trailing pts (league scoring)", p.get("trail_pts"), "last 3 games played"),
        ("E_pts", p.get("E_pts"), "blended, then matchup-adjusted"),
        ("Confidence", p.get("confidence"), p.get("conf")),
        ("Flag", p.get("flag") or "—", ""),
    ]
    return (pd.DataFrame(rows, columns=["component", "value", "note"]),
            f"{p['name']} — {p.position}, {p.team} — in {meta.get('name')}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ask the tracker a question.")
    ap.add_argument("recipe",
                    choices=["live", "lineup", "cut", "options", "explain", "trade"])
    ap.add_argument("slug")
    ap.add_argument("arg", nargs="?", default=None)
    a = ap.parse_args(argv)
    pd.set_option("display.width", 200, "display.max_columns", 40)

    if a.recipe == "live":
        try:
            res = live_mod.live_league(a.slug, ROOT)
        except live_mod.LiveUnavailable as e:
            meta, *_ = _load(a.slug)
            print(f"LIVE READ FAILED: {e}")
            print(f"Falling back to the committed snapshot, lineup as of "
                  f"{meta.get('roster_fetched_at') or 'unknown'} — treat "
                  f"is_starter as possibly stale.")
            return
        print(live_mod.render(res))
        st = res["mine"][res["mine"].is_starter == 1]
        if not st.empty:
            print("\nStatus of every current starter (and who is ahead of him):")
            for line in _status_block(_status(), st["name"].tolist()):
                print(line)
        return

    if a.recipe == "lineup":
        meta, *_ = _load(a.slug)
        print(f"Optimal lineup from committed scores; lineup snapshot as of "
              f"{meta.get('roster_fetched_at') or 'unknown'} "
              f"(run `ff.ask live {a.slug}` for what Sleeper shows right now)\n")
        start, note, bench = lineup(a.slug)
        print(start.to_string(index=False) if not start.empty else "(no lineup)")
        print("\n" + note)
        if not start.empty:
            print("\nStatus of every starter (and who is ahead of him):")
            for line in _status_block(_status(), start["name"].tolist()):
                print(line)
        if not bench.empty:
            print("\nBench:")
            print(bench[["name", "position", "E_pts", "gap_to_starter", "conf", "flag"]]
                  .head(12).to_string(index=False))
    elif a.recipe == "cut":
        try:
            n = int(a.arg) if a.arg else None
        except ValueError:
            print(f"`cut` takes a number of players to drop, not a name "
                  f"(got '{a.arg}'). Omit it and I'll work out how many "
                  f"you're over by:\n    python -m ff.ask cut {a.slug}")
            return
        out, note = cut(a.slug, n)
        show = ["name", "position", "team", "age", "E_pts", "keep_pts",
                "peak_snap_3g", "role_opps", "vor", "age_score", "keep_value",
                "best_replacement"]
        show = [c for c in show if c in out.columns]
        print(out[show].to_string(index=False) if not out.empty
              else "(nothing to cut)")
        if not out.empty:
            print()
            for _, x in out.iterrows():
                print(f"  {x['name']}: {x.why_cut}")
                if x.get("trade_interest"):
                    print(f"      trade instead? {x.trade_interest}")
            print("\nStatus check — is his depth rank about role, or about health?")
            for line in _status_block(_status(), out["name"].tolist()):
                print(line)
        print("\n" + note)
    elif a.recipe == "options":
        out, note = options(a.slug, a.arg or "")
        print(note + "\n")
        print(out.to_string(index=False) if not out.empty else "(no options found)")
        names = ([a.arg] if a.arg else []) + (out["name"].head(5).tolist()
                                              if not out.empty else [])
        print("\nStatus:")
        for line in _status_block(_status(), names):
            print(line)
    elif a.recipe == "trade":
        res, meta = trade(a.slug, a.arg)
        if a.arg:
            p, deals = res
            if p is None:
                print(f"No player matching '{a.arg}'."); return
            print(f"Shopping {p['name']} ({p.position}, {p.team}, age "
                  f"{p.age:.0f}) — E_pts {p.E_pts}, "
                  f"{p.age_score:.0%} age runway, {p.vor:+.1f} vs replacement\n")
            if not deals:
                print("No owner shows a clear fit for him right now — nobody is "
                      "thin at his position, he is not a handcuff to anyone's "
                      "starter, and he would not slot into anyone's top 2.")
                if pd.notna(p.get("E_pts")) and p.E_pts >= 10:
                    print("He still produces, though, so this is a 'hold and "
                          "re-check' rather than a cut. Market fit changes "
                          "with every injury.")
                elif pd.notna(p.get("vor")) and p.vor < 0:
                    print("He also scores below the waiver wire at his "
                          "position, so cutting is reasonable.")
                return
            for d in deals:
                print(f"── {d['owner']}  ({d['posture']}, avg starter age "
                      f"{d['avg_age']})")
                for r in d["reasons"]:
                    print(f"     • {r}")
                print("   ask for:")
                for _, ask in d["asks"].iterrows():
                    gap = (ask.keep_value - p.keep_value)
                    tag = ("  [you gain — expect a counter]" if gap > 0.08 else
                           "  [roughly even]" if gap > -0.08 else "  [you give up value]")
                    print(f"     - {ask['name']:22s} {ask.position} {str(ask.team):3s} "
                          f"age {ask.age:.0f}  E_pts {ask.E_pts:5.1f}  "
                          f"keep {ask.keep_value:.2f}{tag}")
                print(f"\n   pitch: {trade_mod.pitch(p, d, d['asks'].iloc[0], meta)}\n")
            print("Status of everyone named:")
            names = [p["name"]] + [x["name"] for d in deals
                                   for _, x in d["asks"].iterrows()]
            for line in _status_block(_status(), names):
                print(line)
        else:
            sell, mine, prof = res
            print("Your team:")
            print(mine.to_string(index=False) if not mine.empty else "(unknown)")
            print("\nSell-high candidates (producing, but past the age curve):")
            print(sell[["name", "position", "team", "age", "E_pts", "vor",
                        "age_score", "keep_value"]].to_string(index=False)
                  if not sell.empty else "(none)")
            print("\nLeague:")
            print(prof[["owner_name", "startable", "avg_age_startable",
                        "posture", "thin_at"]].to_string(index=False))
            print("\nShop one: python -m ff.ask trade " + a.slug + ' "Player Name"')
    else:
        out, note = explain(a.slug, a.arg or "")
        print(note + "\n")
        print(out.to_string(index=False) if not out.empty else "(not found)")
        print("\nStatus:")
        for line in _status_block(_status(), [a.arg or ""]):
            print(line)


if __name__ == "__main__":
    main()
