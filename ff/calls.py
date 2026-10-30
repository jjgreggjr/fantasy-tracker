"""The call: which lineup to start when the E_pts optimum and the model optimum differ.

E_pts is authoritative (the model is on probation, POLICY.md). The model is a tiebreaker with
three exceptions, applied per differing pair (a = E_pts pick, b = model pick) in this order:

  1. status       one of the pair is shaky (Questionable/Doubtful and did not practice, REPEAT_DNP, or
                  play_prob <= 0.5) and the other is not, and the E_pts gap <= STATUS_GAP (2.0): the
                  healthier player.
  2. near tie     |E_pts gap| <= NEAR_TIE (0.75): the model's pick; if the model gap is also within
                  NEAR_TIE, the higher E_opps.
  3. model        model gap >= MODEL_OVERRIDE (1.5) x the E_pts gap, and the model's pick has E_opps >=
     override     the E_pts pick's: the model's pick.
  otherwise       a, the E_pts pick.

The chosen players are then re-slotted by E_pts so every slot is valid; a swap that cannot be
slotted is dropped (the E_pts pick stays). The model never scores K or IDP positions, so those
fill from E_pts in the model lineup and the two lineups differ only where the model has an opinion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import scoring

NEAR_TIE = 0.75          # E_pts points
MODEL_OVERRIDE = 1.5     # model gap must be >= this multiple of the E_pts gap
STATUS_GAP = 2.0         # the healthier player wins when the E_pts gap is within this

ROOT = Path(__file__).resolve().parent.parent
ID_COLS = ("gsis_id", "sleeper_id", "espn_id", "player_id")
BASE_COLS = ("slot", "name", "position", "team", "E_pts", "E_pts_model", "p10", "p90")
EXTRA_COLS = ("opponent", "conf", "why", "E_opps")
NOT_MODELED = {"K", "DL", "LB", "DB"}      # positions the model never covers: borrowed from E_pts


def _short(name: str) -> str:
    parts = [p for p in str(name).replace(",", "").split() if p.lower().strip(".") not in ("jr", "sr", "ii", "iii", "iv", "v")]
    return parts[-1] if parts else str(name)


def _num(x, default=np.nan) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return default if np.isnan(v) else v


def _with_status(roster: pd.DataFrame, status: pd.DataFrame | None) -> pd.DataFrame:
    """Add `practice` / `play_prob` from the status table when the roster does not carry them."""
    need = [c for c in ("practice", "play_prob") if c not in roster.columns]
    if not need or "gsis_id" not in roster.columns:
        return roster
    if status is None:
        p = ROOT / "data" / "status.csv"
        try:
            status = pd.read_csv(p, low_memory=False) if p.exists() else pd.DataFrame()
        except Exception:
            status = pd.DataFrame()
    if status is None or status.empty or "gsis_id" not in status.columns:
        return roster.assign(**{c: np.nan for c in need})
    look = status.drop_duplicates("gsis_id").set_index("gsis_id")
    return roster.assign(**{c: (roster["gsis_id"].map(look[c]) if c in look.columns else np.nan) for c in need})


def _shaky(r) -> tuple[bool, str]:
    """(is he a status risk, why in a few words)."""
    inj = str(r.get("injury_status") or "")
    prac = str(r.get("practice") or "")
    if inj in ("Questionable", "Doubtful") and prac.startswith("Did Not"):
        return True, "did not practice"
    if str(r.get("flag") or "") == "REPEAT_DNP" or _num(r.get("repeat_dnp"), 0) >= 1:
        return True, "has a repeat DNP"
    pp = _num(r.get("play_prob"))
    if not np.isnan(pp) and pp <= 0.5:
        return True, f"play_prob {pp:.2f}"
    return False, ""


def _practice_word(r) -> str:
    prac = str(r.get("practice") or "")
    if prac.startswith("Limited"):
        return "did (limited)"
    if prac.startswith("Full"):
        return "did (full)"
    return "is not flagged"


def _vol(r) -> str:
    opps = _num(r.get("E_opps"))
    s = f"{opps:.1f} opps" if not np.isnan(opps) else "? opps"
    share = _num(r.get("proj_carry_share")) if r.get("position") == "RB" else np.nan
    if not np.isnan(share):
        return f"{s} / {share:.0%} carries"
    snap = _num(r.get("avg_snap_pct"))
    return f"{s} / {snap:.0%} snaps" if not np.isnan(snap) else s


def _decide(a, b) -> tuple[str, str]:
    """Return ("a" | "b", reason) for E_pts pick a vs model pick b."""
    na, nb = _short(a["name"]), _short(b["name"])
    ea, eb = _num(a.get("E_pts"), 0.0), _num(b.get("E_pts"), 0.0)
    ma, mb = _num(a.get("E_pts_model"), ea), _num(b.get("E_pts_model"), eb)
    egap, mgap = ea - eb, mb - ma
    oa, ob = _num(a.get("E_opps"), 0.0), _num(b.get("E_opps"), 0.0)

    sa, wa = _shaky(a)
    sb, wb = _shaky(b)
    if sa != sb and abs(egap) <= STATUS_GAP:
        bad, good, why = (a, b, wa) if sa else (b, a, wb)
        reason = (f"{_short(bad['name'])} {why}; {_short(good['name'])} {_practice_word(good)}. "
                  f"E_pts gap {abs(egap):.1f}")
        return ("b" if sa else "a"), reason

    if abs(egap) <= NEAR_TIE:
        if mgap > NEAR_TIE:
            pick = "b"
            tail = f"model prefers {nb} by {mgap:.1f}"
        elif mgap < -NEAR_TIE:
            pick = "a"
            tail = f"model prefers {na} by {-mgap:.1f}"
        else:
            pick = "b" if ob > oa else "a"
            tail = f"model gap {abs(mgap):.1f} also a tie"
        reason = f"E_pts tie ({ea:.1f} vs {eb:.1f}); {tail}; {oa:.1f} vs {ob:.1f} opps"
        return pick, reason

    if mgap > 0 and mgap >= MODEL_OVERRIDE * egap and ob >= oa:
        return "b", (f"E_pts +{egap:.1f} {na} on {_vol(a)}; model +{mgap:.1f} {nb} on {_vol(b)}")

    if mgap >= MODEL_OVERRIDE * egap and mgap > 0:
        mtxt = f"model +{mgap:.1f} {nb} but on fewer opps ({ob:.1f} vs {oa:.1f})"
    elif abs(mgap) <= NEAR_TIE:
        mtxt = f"model calls it even ({abs(mgap):.1f}); volume similar ({oa:.1f} vs {ob:.1f} opps)"
    elif mgap > 0:
        mtxt = f"model only +{mgap:.1f} {nb}; volume {oa:.1f} vs {ob:.1f} opps"
    else:
        mtxt = f"model agrees (+{-mgap:.1f} {na}); volume {oa:.1f} vs {ob:.1f} opps"
    note = ""
    if sa != sb:
        note = f"; {_short(b['name'] if sb else a['name'])} {wb or wa}"
    return "a", f"E_pts +{egap:.1f} {na}; {mtxt}{note}"


def _frame(picks: list[dict], slots: list[str], rows: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    """One row per slot in `slots` order; a slot nobody could fill is a blank row."""
    by_slot: dict[str, list[str]] = {}
    for p in picks:
        by_slot.setdefault(p["slot"], []).append(p["name"])
    seen: dict[str, int] = {}
    out = []
    cols = [*BASE_COLS[1:], *[c for c in EXTRA_COLS if c in rows.columns], *id_cols]
    for s in slots:
        i = seen.get(s, 0)
        seen[s] = i + 1
        names = by_slot.get(s, [])
        if i < len(names):
            r = rows.loc[names[i]]
            out.append({"slot": s, **{c: (r[c] if c in r.index else np.nan) for c in cols}})
        else:
            out.append({"slot": s, "name": "", **{c: np.nan for c in cols if c != "name"}})
    df = pd.DataFrame(out)
    for c in ("E_pts", "E_pts_model", "p10", "p90"):
        if c not in df.columns:
            df[c] = np.nan
    return df[["slot", "name", "position", "team", "E_pts", "E_pts_model", "p10", "p90",
               *[c for c in df.columns if c not in BASE_COLS]]]


def three_lineups(roster: pd.DataFrame, slots: list[str], idp_on: bool,
                  status: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Keys "e_pts", "model", "call": each one row per slot in `slots` order (see module docstring).
    The "call" frame adds e_pick, m_pick ("" when the same player), reason ("" when nothing differed) and
    pair (index of the differing pair, -1 none). `model.attrs["borrowed"]` lists K / IDP names filled from E_pts."""
    from .ask import optimize_lineup           # the exact optimizer lives beside the recipes

    id_cols = [c for c in ID_COLS if c in roster.columns]
    r = _with_status(roster, status).drop_duplicates("name").reset_index(drop=True)
    has_model = "E_pts_model" in r.columns
    borrowed_names: set[str] = set()
    if has_model:
        fill = r["E_pts_model"].isna() & r["position"].isin(NOT_MODELED) & r["E_pts"].notna()
        borrowed_names = set(r.loc[fill, "name"])
        r.loc[fill, "E_pts_model"] = r.loc[fill, "E_pts"]
    rows = r.set_index("name", drop=False)

    e_picks = optimize_lineup(r, slots, "E_pts", idp_on)
    e = _frame(e_picks, slots, rows, id_cols)
    if not has_model:
        m, call = e.copy(), e.copy()
        for df in (call,):
            df["e_pick"], df["m_pick"], df["reason"], df["pair"] = "", "", "", -1
        m.attrs["borrowed"], call.attrs["pairs"], call.attrs["has_model"] = [], [], False
        return {"e_pts": e, "model": m, "call": call}

    m_picks = optimize_lineup(r, slots, "E_pts_model", idp_on)
    m = _frame(m_picks, slots, rows, id_cols)
    m.attrs["borrowed"] = sorted(n for n in m["name"] if n in borrowed_names)

    e_slot = dict(zip(e["name"], e["slot"]))
    m_slot = dict(zip(m["name"], m["slot"]))
    e_only = [n for n in e["name"] if n and n not in m_slot]
    m_only = [n for n in m["name"] if n and n not in e_slot]

    pairs: list[tuple[str, str]] = []
    left = list(m_only)
    unpaired = []
    for a in e_only:                                   # same position first
        hit = next((b for b in left if rows.loc[b, "position"] == rows.loc[a, "position"]), None)
        if hit is None:
            unpaired.append(a)
        else:
            pairs.append((a, hit))
            left.remove(hit)
    for a in unpaired:                                 # then anyone who can take a's slot, then anyone left:
        hit = next((b for b in left if scoring.slot_allows(e_slot[a], rows.loc[b, "position"], idp_on)),
                   left[0] if left else None)          # the re-slot below decides whether the swap is feasible
        if hit is not None:
            pairs.append((a, hit))
            left.remove(hit)

    base = {n for n in e["name"] if n}
    n_filled = len(base)
    decisions = []
    chosen = set(base)
    for i, (a, b) in enumerate(pairs):
        pick, reason = _decide(rows.loc[a], rows.loc[b])
        if pick == "b":                                # take the swap only if the set still slots completely
            trial = (chosen - {a}) | {b}
            got = optimize_lineup(r[r["name"].isin(trial)], slots, "E_pts", idp_on)
            if len(got) == n_filled:
                chosen = trial
            else:
                pick, reason = "a", reason + " (swap cannot be slotted; E_pts pick kept)"
        decisions.append({"a": a, "b": b, "pick": pick, "reason": reason})

    c_picks = optimize_lineup(r[r["name"].isin(chosen)], slots, "E_pts", idp_on)
    call = _frame(c_picks, slots, rows, id_cols)
    pair_of = {}
    for i, d in enumerate(decisions):
        pair_of[d["a"]] = pair_of[d["b"]] = i
    e_pick, m_pick, reason, pid = [], [], [], []
    for k in range(len(slots)):
        en, mn, cn = e.at[k, "name"], m.at[k, "name"], call.at[k, "name"]
        differ = en != mn
        e_pick.append(en if differ else "")
        m_pick.append(mn if differ else "")
        i = next((pair_of[x] for x in (en, mn, cn) if differ and x in pair_of), -1)
        pid.append(i)
        reason.append(decisions[i]["reason"] if i >= 0 else "")
    call["e_pick"], call["m_pick"], call["reason"], call["pair"] = e_pick, m_pick, reason, pid
    call.attrs["has_model"] = True
    call.attrs["pairs"] = [{**d, "slot": e_slot[d["a"]],
                            "chosen": d["b"] if d["pick"] == "b" else d["a"]} for d in decisions]
    return {"e_pts": e, "model": m, "call": call}
