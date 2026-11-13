"""Check-in helper: the deterministic, read-only parts of the /checkin skill.

    python -m ff.checkin [report]                 # health, last week, season efficiency, serve, miss gate
    python -m ff.checkin misses [--week W] [--json]   # the miss-research queue, one entry per player
    python -m ff.checkin gate [--week W]          # the GATE line; exit 0 = open, 3 = researched
    python -m ff.checkin record --week W --from f.json [--force]   # the ONLY writer

Reads committed files only, relative to the repo root. Never runs the pipeline, the model, or the network.
`record` writes `model/reports/miss_research.md` and `model/candidates.md`.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

LEAGUE_ORDER = ["gooma-s-family-league", "we-can-think-of-something-funny", "where-you-at", "james-gregg-espn"]
STALE_DAYS = 4
# Early-week WARNs that are normal and do not count against health.
IGNORED_WARN = ("no rows for", "no rows yet", "0% have practice reports")
REFIT_LINE = ("Refit: trees retrain on every serve (walk-forward through last week); "
              "hyperparameters and tree counts are frozen by design.")
COMP_LABEL = {"model_components": "model", "e_pts": "e_pts", "sleeper": "sleeper"}
COMP_ORDER = ["e_pts", "model", "sleeper"]
BOX_COLS = ["snaps", "snap_pct", "targets", "carries", "pass_att", "touches", "opps", "rec_yds", "rush_yds",
            "pass_yds", "rec_tds", "rush_tds", "pass_tds", "fpts_ppr"]
GAME_COLS = ["opponent", "gameday", "home", "spread_line", "total_line"]
RETRO_NOTE_MAX = 400


# ---------------------------------------------------------------- small helpers

def _csv(root: Path, rel: str) -> pd.DataFrame:
    p = Path(root) / rel
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=False, dtype={"gsis_id": str, "sleeper_id": str, "espn_id": str})


def _table(headers: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _esc(x) -> str:
    """Markdown table cell: pipes escaped, newlines flattened."""
    return " ".join(str(x).split()).replace("|", "\\|")


def _clean(v):
    """numpy / NaN to plain JSON values."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        v = v.item()
    return v


def _true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def _league_meta(root: Path, slug: str) -> dict:
    p = Path(root) / "leagues" / slug / "league.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _league_slugs(root: Path) -> list[str]:
    d = Path(root) / "leagues"
    have = sorted(p.name for p in d.glob("*") if p.is_dir()) if d.exists() else []
    return [s for s in LEAGUE_ORDER if s in have] + [s for s in have if s not in LEAGUE_ORDER]


def _latest_sw(df: pd.DataFrame):
    """(season, week) of the newest row, or None."""
    if df.empty or "week" not in df:
        return None
    d = df.dropna(subset=["season", "week"])
    if d.empty:
        return None
    s = int(d["season"].max())
    return s, int(d[d["season"] == s]["week"].max())


# ---------------------------------------------------------------- 1. health

def _warn_parts(detail: str) -> tuple[list[str], list[str]]:
    """(real, ignored) pieces of a detail string (pieces are separated by '; ')."""
    real, ign = [], []
    for part in [p.strip() for p in str(detail).split(";") if p.strip()]:
        (ign if any(k in part for k in IGNORED_WARN) else real).append(part)
    return real, ign


def _status_time(root: Path) -> datetime | None:
    """status.csv carries no timestamp column, so its age is the time of the commit that last touched it
    (the pipeline commits it), falling back to the file's mtime."""
    p = Path(root) / "data" / "status.csv"
    if not p.exists():
        return None
    try:
        r = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%cI", "--", "data/status.csv"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return datetime.fromisoformat(r.stdout.strip()).astimezone(timezone.utc)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)


def health(root: Path = ROOT, now: datetime | None = None) -> list[str]:
    now = now or datetime.now(timezone.utc)
    L = ["## Health", ""]
    verdict = None
    runs = _csv(root, "logs/runs.csv")
    if runs.empty:
        L.append("- runs.csv: missing or empty")
        verdict = "HEALTH: no run log — fix: press Run workflow on the repo's Actions tab"
    else:
        runs["_t"] = pd.to_datetime(runs["ran_at"], utc=True, errors="coerce")
        runs = runs.sort_values("_t", kind="stable")
        new = runs.iloc[-1]
        detail = "" if pd.isna(new.get("detail")) else str(new["detail"])
        L.append(f"- newest run: {new['ran_at']} · {new['season']} wk{new['week']} · {new['status']}"
                 + (f" · {detail[:160]}" if detail else ""))
        recent = runs[(runs["_t"] >= now - pd.Timedelta(days=7)) & runs["status"].isin(["WARN", "FAIL"])]
        if recent.empty:
            L.append("- WARN/FAIL rows, last 7 days: 0")
        else:
            L.append(f"- WARN/FAIL rows, last 7 days: {len(recent)}")
            seen = set()
            for st, d in zip(recent["status"], recent["detail"].fillna("")):
                pre = str(d)[:120]
                if (st, pre) in seen:
                    continue
                seen.add((st, pre))
                tag = " (early-week, ignored)" if st == "WARN" and not _warn_parts(d)[0] else ""
                L.append(f"  - {st}: {pre}{tag}")
        age_days = (now - new["_t"]).total_seconds() / 86400 if pd.notna(new["_t"]) else None
        if new["status"] == "FAIL":
            verdict = "HEALTH: pipeline run failed — fix: read the detail, then press Run workflow on the Actions tab"
        elif age_days is None or age_days > STALE_DAYS:
            age = "unknown age" if age_days is None else f"{age_days:.1f} days old"
            verdict = f"HEALTH: stale, newest run {age} — fix: press Run workflow on the repo's Actions tab"

    mp = _csv(root, "data/model_pts.csv")
    if mp.empty or "served_at" not in mp:
        L.append("- model_pts: missing")
    else:
        sw = _latest_sw(mp)
        wk = mp[(mp["season"] == sw[0]) & (mp["week"] == sw[1])]
        share = _true(wk["frozen_before_kickoff"]).mean() * 100 if "frozen_before_kickoff" in wk else float("nan")
        L.append(f"- model_pts: latest served_at {mp['served_at'].max()}; frozen_before_kickoff {share:.0f}% of "
                 f"{len(wk)} rows (wk{sw[1]})")
    pw = _latest_sw(_csv(root, "data/player_weeks.csv"))
    L.append(f"- player_weeks: latest {pw[0]} wk{pw[1]}" if pw else "- player_weeks: missing")
    st = _csv(root, "data/status.csv")
    t = _status_time(root)
    if st.empty or t is None:
        L.append("- status: missing")
    else:
        hrs = (now - t).total_seconds() / 3600
        flag = st["status_flag"].fillna("") if "status_flag" in st else pd.Series("", index=st.index)
        rep = st["report_status"].notna() if "report_status" in st else pd.Series(False, index=st.index)
        desig = rep | flag.isin(["out", "questionable", "doubtful"])
        ina = st["inactive"].fillna(0).astype(float) == 1 if "inactive" in st else pd.Series(False, index=st.index)
        known = st["inactive_known_at"].dropna().max() if "inactive_known_at" in st else None
        L.append(f"- status: {hrs:.0f}h old (last commit of status.csv; the file has no timestamp column); "
                 f"{int(desig.sum())} with an injury designation (out/questionable/doubtful or a report status); "
                 f"{int(ina.sum())} inactive=1; latest inactive_known_at {known if isinstance(known, str) else 'none yet'}")
    if verdict is None and not runs.empty:
        real = _warn_parts(detail)[0] if new["status"] == "WARN" else []
        if real:
            L.append(f"- warn: {real[0][:120]}")
        verdict = "HEALTH: healthy"
    L += ["", verdict or "HEALTH: healthy"]
    return L


# ---------------------------------------------------------------- scoreboard text

def _summary_lines(root: Path) -> list[str]:
    """The scoreboard's Summary block: from the committed report, else rebuilt from the committed data
    (the same function `ff.ask scoreboard` prints)."""
    p = Path(root) / "model" / "reports" / "live_scoreboard.md"
    if p.exists():
        out, on = [], False
        for ln in p.read_text(encoding="utf-8").splitlines():
            if ln.startswith("## "):
                if on:
                    break
                on = ln.strip() == "## Summary"
            elif on:
                out.append(ln)
        if out:
            return out
    try:
        from model import scoreboard as sb
        return sb.committed_summary(Path(root)).splitlines()
    except Exception:  # noqa: BLE001 - missing data must not break the report
        return []


def _gate_line(root: Path) -> str:
    for ln in _summary_lines(root):
        if ln.startswith("- Gate:"):
            return ln[2:].strip()
    return "Gate: not available (no scoreboard summary)"


# ---------------------------------------------------------------- 2. last week

def last_week(root: Path = ROOT) -> list[str]:
    le = _csv(root, "data/lineup_eval.csv")
    mr = _csv(root, "data/matchup_results.csv")
    sw = _latest_sw(le) or _latest_sw(mr)
    if not sw:
        return ["## Last week", "", "No scored week yet.", "", _gate_line(root)]
    season, wk = sw
    L = [f"## Last week ({season} week {wk})", ""]
    rows = []
    for slug in _league_slugs(root):
        meta = _league_meta(root, slug)
        pf = pa = res = "-"
        if not mr.empty and meta:
            m = mr[(mr["season"] == season) & (mr["week"] == wk)]
            m = m[m["league_id"].astype(str) == str(meta.get("league_id"))] if "league_id" in m else m.iloc[0:0]
            if m.empty and "league_name" in mr:
                m = mr[(mr["season"] == season) & (mr["week"] == wk) & (mr["league_name"] == meta.get("name"))]
            m = m[m["roster_id"].astype(str) == str(meta.get("my_roster_id"))]
            if len(m):
                r = m.iloc[0]
                pf, pa = f"{r['points_for']:.1f}", f"{r['points_against']:.1f}"
                res = "T" if r.get("tie") == 1 else ("W" if r.get("won") == 1 else "L")
        act = perf = eff = "-"
        if not le.empty:
            e = le[(le["season"] == season) & (le["week"] == wk) & (le["league"] == slug)]
            a, p = e[e["basis"] == "actual"], e[e["basis"] == "perfect"]
            # The optimizers and the hindsight-best lineup skip unscored slots (K, DEF), so
            # the comparable number is offense-only; the raw ratio can read above 100%.
            if len(a):
                act = f"{a.iloc[0]['offense_points']:.1f}"
                eff = f"{100 * a.iloc[0]['offense_efficiency']:.0f}%"
            if len(p):
                perf = f"{p.iloc[0]['offense_points']:.1f}"
        rows.append([meta.get("name", slug), pf, pa, res, act, perf, eff])
    L += _table(["league", "PF", "PA", "W/L", "lineup pts (offense)", "perfect pts (offense)", "offense efficiency"], rows)
    L.append("")
    me = _csv(root, "data/model_eval.csv")
    line = "Pick accuracy: no model_eval rows for that week."
    if not me.empty:
        m = me[(me["season"] == season) & (me["week"] == wk) & (me["position"] == "ALL")]
        parts = []
        for comp, lab in (("model_components", "model"), ("sleeper", "Sleeper"), ("e_pts", "E_pts")):
            c = m[m["comparator"] == comp]
            if len(c) and c["pairs"].sum() > 0:
                parts.append(f"{lab} {(c['pick_acc'] * c['pairs']).sum() / c['pairs'].sum():.3f}")
        if parts:
            line = f"Pick accuracy, week {wk}, position ALL, pooled over leagues (pair-weighted): " + " vs ".join(parts)
    L += [line, "", _gate_line(root)]
    return L


# ---------------------------------------------------------------- 3. season efficiency

def efficiency(root: Path = ROOT) -> list[str]:
    want = [("Lineup efficiency", "lineup efficiency"), ("Offense-only efficiency", "offense-only efficiency"),
            ("Lineup wins", "lineup wins")]
    cols = ["e_pts", "model", "sleeper", "call", "actual"]
    rows = []
    lines = _summary_lines(root)
    for key, label in want:
        for ln in lines:
            if ln.startswith(f"- {key}"):
                vals = dict(re.findall(r"(\w+) (\d[\d.]*\d%?|\d%?)", ln.rsplit(": ", 1)[-1]))
                rows.append([label] + [vals.get(c, "-") for c in cols])
                break
    L = ["## Season efficiency", ""]
    return L + (_table(["metric"] + cols, rows) if rows else ["No lineup weeks scored yet."])


# ---------------------------------------------------------------- 4. serve

def serve(root: Path = ROOT) -> list[str]:
    mp = _csv(root, "data/model_pts.csv")
    L = ["## Serve", ""]
    if mp.empty or "served_at" not in mp:
        return L + ["No model_pts.csv.", REFIT_LINE]
    sw = _latest_sw(mp)
    wk = mp[(mp["season"] == sw[0]) & (mp["week"] == sw[1])]
    share = _true(wk["frozen_before_kickoff"]).mean() * 100 if "frozen_before_kickoff" in wk else float("nan")
    L += [f"- week served: {sw[0]} week {sw[1]}", f"- latest served_at: {wk['served_at'].max()}",
          f"- frozen_before_kickoff: {share:.0f}% of {len(wk)} rows", REFIT_LINE]
    return L


# ---------------------------------------------------------------- 5. misses

def _misses_week(root: Path, week: int | None = None):
    """(season, week, frame of that week's rows) from data/model_misses.csv; week None = the latest."""
    mi = _csv(root, "data/model_misses.csv")
    if mi.empty:
        return None, week, mi
    season = int(mi["season"].max())
    if week is None:
        week = int(mi[mi["season"] == season]["week"].max())
    return season, week, mi[(mi["season"] == season) & (mi["week"] == week)].copy()


def _section(text: str, season: int, week: int) -> str | None:
    """The body of the `## <season> week <week>` section of miss_research.md, or None."""
    head = f"## {season} week {week}"
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.rstrip() == head:
            j = i + 1
            while j < len(lines) and not lines[j].startswith("## "):
                j += 1
            return "\n".join(lines[i + 1:j])
    return None


def gate(root: Path = ROOT, week: int | None = None) -> tuple[str, bool]:
    """(the GATE line, researched?)"""
    season, week, m = _misses_week(root, week)
    if season is None:
        return "GATE: open (no misses file): 0 big misses to research", False
    p = Path(root) / "model" / "reports" / "miss_research.md"
    body = _section(p.read_text(encoding="utf-8"), season, week) if p.exists() else None
    if body is not None:
        d = re.search(r"Researched:\s*(\S+?)\.?(?:\s|$)", body)
        return f"GATE: researched ({season} week {week}" + (f", {d.group(1)})" if d else ")"), True
    return f"GATE: open ({season} week {week}): {m['gsis_id'].nunique()} big misses to research", False


def _retro_notes(root: Path, season: int, week: int) -> dict[str, str]:
    """{player name: short note} from retrospective.md's Top-15 section for that week, one row per comparator."""
    p = Path(root) / "model" / "reports" / "retrospective.md"
    if not p.exists():
        return {}
    head, on, comp, notes = f"## Top 15 misses, {season} week {week}", False, "", {}
    seen = set()
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.startswith("## "):
            on = ln.strip() == head
            continue
        if not on:
            continue
        m = re.match(r"\*\*`?(\w+)`?\*\*", ln)
        if m:
            comp = m.group(1)
            continue
        if not ln.startswith("|"):
            continue
        c = [x.strip().strip("`") for x in re.split(r"(?<!\\)\|", ln.strip().strip("|"))]
        if len(c) < 13 or c[0] in ("player", "---") or c[0].startswith("-"):
            continue
        name = c[0]
        if (name, comp) in seen:
            continue
        seen.add((name, comp))
        sec = f"/{c[8]}" if c[8] else ""
        lab = COMP_LABEL.get(comp, comp)
        if name in notes:  # the evidence columns repeat across comparators; add only the prediction
            notes[name] += f" | {lab}: pred {c[4]}"
            continue
        notes[name] = (f"{lab} ({c[3]}): pred {c[4]}, act {c[5]}, {c[7]}{sec}; vol {c[9]}, TD {c[10]}, snap {c[11]}, "
                       f"margin {c[12]}" + (f", teammate out {c[13]}" if len(c) > 13 and c[13] else ""))
    return {k: (v if len(v) <= RETRO_NOTE_MAX else v[:RETRO_NOTE_MAX - 3] + "...") for k, v in notes.items()}


def _box_trail(root: Path, season: int, week: int, gsis: str):
    pw = _csv(root, "data/player_weeks.csv")
    if pw.empty:
        return None, None
    mine = pw[(pw["season"] == season) & (pw["gsis_id"] == gsis)]
    cols = [c for c in BOX_COLS if c in pw]
    this = mine[mine["week"] == week]
    box = {c: _clean(this.iloc[0][c]) for c in cols} if len(this) else None
    prev = mine[mine["week"] < week].sort_values("week").tail(3)
    trail = {c: (_clean(prev[c].mean()) if len(prev) else None) for c in cols}
    return box, trail


def _game(root: Path, season: int, week: int, team) -> dict | None:
    sc = _csv(root, "data/schedule.csv")
    if sc.empty:
        return None
    g = sc[(sc["season"] == season) & (sc["week"] == week) & (sc["team"] == team)]
    if g.empty:
        return None
    r = g.iloc[0]
    out = {c: _clean(r[c]) for c in GAME_COLS if c in sc}
    if out.get("home") is not None:
        out["home"] = bool(out["home"])
    return out


def misses(root: Path = ROOT, week: int | None = None) -> tuple[int | None, int | None, list[dict]]:
    """One entry per unique player for the week, worst error first."""
    season, week, m = _misses_week(root, week)
    if season is None or m.empty:
        return season, week, []
    notes = _retro_notes(root, season, week)
    out = []
    for gid, g in m.groupby("gsis_id", sort=False):
        g = g.assign(_abs=g["error"].abs()).sort_values("_abs", ascending=False, kind="stable")
        w = g.iloc[0]
        box, trail = _box_trail(root, season, week, gid)
        ms = [{"league": r["league"], "comparator": COMP_LABEL.get(r["comparator"], r["comparator"]),
               "predicted": _clean(r["predicted"]), "actual": _clean(r["actual"]), "error": _clean(r["error"]),
               "reason": _clean(r["reason"]), "secondary_reason": _clean(r["secondary_reason"])}
              for _, r in g.iterrows()]
        played, ina = _clean(w.get("played")), _clean(w.get("on_inactive_list"))
        out.append({
            "gsis_id": gid, "name": w["name"], "position": w["position"], "team": w["team"], "week": week,
            "leagues": list(dict.fromkeys(g["league"])), "misses": ms,
            "exp_vol": _clean(w.get("exp_vol")), "act_vol": _clean(w.get("act_vol")),
            "exp_td": _clean(w.get("exp_td")), "act_td": _clean(w.get("act_td")),
            "trail_snap_pct": _clean(w.get("trail_snap_pct")), "snap_pct": _clean(w.get("snap_pct")),
            "teammate_out_name": _clean(w.get("teammate_out_name")),
            "played": None if played is None else str(played).lower() == "true",
            "on_inactive_list": None if ina is None else bool(float(ina)),
            "retro_note": notes.get(w["name"], ""),
            "game": _game(root, season, week, w["team"]), "box": box, "trail": trail,
        })
    out.sort(key=lambda d: -max(abs(x["error"]) for x in d["misses"]))
    return season, week, out


def _comp_label(ms: list[dict]) -> str:
    s = {x["comparator"] for x in ms}
    if s == {"e_pts", "model"}:
        return "both"
    return "+".join(c for c in COMP_ORDER if c in s)


def _miss_rows(entries: list[dict]) -> list[list]:
    rows = []
    for d in entries:
        w = max(d["misses"], key=lambda x: abs(x["error"]))
        reason = "/".join(x for x in (w["reason"], w["secondary_reason"]) if x)
        rows.append([_esc(d["name"]), d["position"], d["team"], _comp_label(d["misses"]),
                     f"{w['predicted']:.1f}→{w['actual']:.1f}", reason, ", ".join(d["leagues"])])
    return rows


MISS_HEAD = ["name", "pos", "team", "missed by", "predicted→actual (worst)", "reason", "leagues"]


def miss_block(root: Path = ROOT) -> list[str]:
    line, done = gate(root)
    season, week, ents = misses(root)
    L = ["## Miss research", "", line, ""]
    if ents:
        L += _table(MISS_HEAD, _miss_rows(ents[:15]))
        if len(ents) > 15:
            L += ["", f"+{len(ents) - 15} more"]
    return L


# ---------------------------------------------------------------- record (the only writer)

HEADER = ("# Miss research\n\n"
          "Written by `python -m ff.checkin record` after the /checkin skill researches the previous week's big misses "
          "(`data/model_misses.csv`). One section per week; the `## 2026 week N` headings are the /checkin skip gate "
          "(a week with a section here is not researched again). Candidates raised go to `model/candidates.md` as `open` rows.\n")
CAND_HEAD = ["| date raised | candidate | evidence that raised it | status |", "|---|---|---|---|"]


def _is_row(ln: str) -> bool:
    return ln.lstrip().startswith("|")


def _build_section(season: int, week: int, date: str, misses_in: list[dict]) -> str:
    n_c = sum(1 for m in misses_in if str(m.get("candidate") or "").strip())
    L = [f"## {season} week {week}", "",
         f"Researched: {date}. Misses researched: {len(misses_in)}. Candidates raised: {n_c}.", "",
         "| player | pos/team | why | foreseeable? | candidate |", "|---|---|---|---|---|"]
    for m in misses_in:
        why = _esc(m.get("why", ""))
        # Web sources only: the agents also list the repo files they read, which say nothing to a reader.
        srcs = [_esc(s) for s in (m.get("sources") or []) if not str(s).strip().lower().startswith("local:")]
        if srcs:
            why += " Sources: " + "; ".join(srcs)
        fore = "yes" if m.get("foreseeable") else "no"
        if m.get("foreseeable_signal"):
            fore += f": {_esc(m['foreseeable_signal'])}"
        cand = str(m.get("candidate") or "").strip()
        L.append(f"| {_esc(m.get('name', ''))} | {_esc(m.get('position', ''))}/{_esc(m.get('team', ''))} | {why} | {fore} | "
                 f"{'`' + _esc(cand) + '`' if cand else '-'} |")
    return "\n".join(L) + "\n"


def _add_candidates(text: str, rows: list[str]) -> str:
    """Insert rows after the last line of the first table in candidates.md (append a table if none)."""
    if not rows:
        return text
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if _is_row(ln) and ln.lstrip().startswith("| date raised")), None)
    if start is None:
        base = text.rstrip("\n") + ("\n\n" if text.strip() else "")
        return base + "\n".join(CAND_HEAD + rows) + "\n"
    end = start
    while end + 1 < len(lines) and _is_row(lines[end + 1]):
        end += 1
    lines[end + 1:end + 1] = rows
    return "\n".join(lines) + "\n"


def record(root: Path, week: int, src: Path, force: bool = False) -> tuple[int, str]:
    """(exit code, message)."""
    try:
        data = json.loads(Path(src).read_text(encoding="utf-8"))
        misses_in = list(data["misses"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        return 2, f"cannot read {src}: {e}"
    if int(data.get("week", week)) != int(week):
        return 2, f"week mismatch: --week {week} but the file says {data.get('week')}"
    season = int(data.get("season", 2026))
    date = str(data.get("researched_at") or datetime.now(timezone.utc).date().isoformat())
    mr = Path(root) / "model" / "reports" / "miss_research.md"
    old = mr.read_text(encoding="utf-8") if mr.exists() else None
    have = old is not None and _section(old, season, week) is not None
    if have and not force:
        return 2, f"{season} week {week} is already in {mr.name}; pass --force to replace it"
    sec = _build_section(season, week, date, misses_in)
    if old is None:
        new = HEADER + "\n" + sec
    elif have:
        lines, head = old.splitlines(), f"## {season} week {week}"
        i = next(k for k, ln in enumerate(lines) if ln.rstrip() == head)
        j = i + 1
        while j < len(lines) and not lines[j].startswith("## "):
            j += 1
        tail = lines[j:]
        new = "\n".join(lines[:i]) + "\n" + sec + ("\n" + "\n".join(tail) + "\n" if tail else "")
    else:
        new = old.rstrip("\n") + "\n\n" + sec
    cp = Path(root) / "model" / "candidates.md"
    ctext = cp.read_text(encoding="utf-8") if cp.exists() else ""
    rows, skipped, raised = [], 0, 0
    for m in misses_in:
        cand = str(m.get("candidate") or "").strip()
        if not cand:
            continue
        raised += 1
        if cand in ctext or _esc(cand) in ctext or any(f"`{_esc(cand)}`" in r for r in rows):
            skipped += 1
            continue
        ev = _esc(m.get("candidate_evidence", ""))
        rows.append(f"| {date} | `{_esc(cand)}` | {ev} (wk {week}: {_esc(m.get('name', ''))}) | open |")
    mr.parent.mkdir(parents=True, exist_ok=True)
    mr.write_text(new, encoding="utf-8")
    if rows:
        cp.write_text(_add_candidates(ctext, rows), encoding="utf-8")
    return 0, f"recorded {season} week {week}: {len(misses_in)} misses, {raised} candidates ({skipped} skipped as duplicates)"


# ---------------------------------------------------------------- CLI

def report(root: Path = ROOT, now: datetime | None = None) -> str:
    blocks = [health(root, now), last_week(root), efficiency(root), serve(root), miss_block(root)]
    return "\n\n".join("\n".join(b) for b in blocks)


def main(argv=None, root: Path = ROOT) -> int:
    ap = argparse.ArgumentParser(description="Read-mostly check-in helper (health, last week, misses).")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("report")
    mp = sub.add_parser("misses")
    mp.add_argument("--week", type=int)
    mp.add_argument("--json", action="store_true")
    gp = sub.add_parser("gate")
    gp.add_argument("--week", type=int)
    rp = sub.add_parser("record")
    rp.add_argument("--week", type=int, required=True)
    rp.add_argument("--from", dest="src", required=True)
    rp.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd in (None, "report"):
        print(report(root))
        return 0
    if a.cmd == "gate":
        line, done = gate(root, a.week)
        print(line)
        return 3 if done else 0
    if a.cmd == "misses":
        season, week, ents = misses(root, a.week)
        if a.json:
            print(json.dumps(ents, indent=2, default=str))
        elif not ents:
            print(f"No big misses for week {week}.")
        else:
            print(f"## Big misses, {season} week {week} ({len(ents)} players)\n")
            print("\n".join(_table(MISS_HEAD, _miss_rows(ents))))
        return 0
    code, msg = record(root, a.week, Path(a.src), a.force)
    print(msg)
    return code


if __name__ == "__main__":
    sys.exit(main())
