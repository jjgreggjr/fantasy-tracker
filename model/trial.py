"""The standing trial harness: one candidate improvement, judged against the written adoption rule, every time the same way.

    python -m model.trial --list                                   # families, their columns, which are in the serving inputs
    python -m model.trial --candidate recent:4                     # weight the last 4 weeks of training rows x3
    python -m model.trial --candidate +family:pbp_usage            # add an opt-in family (train.OPT_IN_FAMILIES)
    python -m model.trial --candidate=-family:dvp                  # drop a family ('=' form: a value starting with '-' needs it)
    python -m model.trial --candidate=-cols:dvp_ppr_l2,is_home     # drop columns
    python -m model.trial --candidate +cols:tm_out_group_n         # un-prune columns the serving spec dropped
    python -m model.trial --candidate param:lightgbm.reg_lambda=50 # a hyper-parameter change
    python -m model.trial --candidate recent:4 --candidate param:lightgbm.num_leaves=31    # flags compose into ONE variant

What a run does (PLAN_MODEL.md "Phase 5"): the frozen primary (serving_config.json's spec, tuned_params.json) and the variant go
through `backtest.walk_forward` on 2025 (the test year), 2024 (the replication) and the completed 2026 weeks (the live year),
scored on the Phase 2.5 head-to-head rows (played, with an earlier game this season) with `backtest.score` and the week-blocked
`paired_bootstrap`. The verdict follows the Phase 2.5 adoption rule on pooled dRMSE (variant minus primary, negative = better):

    SHIPS             beats the primary on 2025 (95% interval below zero) AND the same sign on 2024
    DIRECTION ONLY    better in both years but the 2025 interval spans zero: noise until replicated again
    DOES NOT SHIP     not better on 2025, or the sign flips on 2024
    TIE (components)  --rule tie only (component models may tie): neither year significantly worse, not a SHIPS

2026 is reported beside the verdict and never decides it (a few weeks; the plan's ground rule). Pick accuracy is reported with
the same intervals. The row is appended to model/reports/trials.md. Nothing here edits serving_config.json or tuned_params.json:
the run prints the exact edit and `python -m model.serve --freeze` for a human to adopt a SHIPS.

A variant whose inputs or library parameters differ from the primary's gets its own tree count by the Phase 2.5 protocol (early
stopping on 2024 after training on 2021-2023, x1.1), so the comparison is fair to it; `recent:<n>` keeps the primary's count (the
weights change the loss scale, not the problem). Walk-forwards are cached under model/cache/trials/ keyed on the matrix fingerprint,
the variant and the parameters, so the primary is computed once and reused by every later trial.

The frame: `model/cache/features.parquet` (the Phase 2.5 matrix, `python -m model.build_features`) when it exists; otherwise the
serving path's own history (`serve.training_frame`: the cached 2021-2025 matrix plus this season's completed weeks, no download).
Either covers 2026 only through the weeks completed when it was built; the run says which weeks it scored.

Only `--list` and parsing run without the boosted-tree libraries; lightgbm & co. are imported when a model is fitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from model import backtest as B
from model import features as F
from model import p25run as P
from model import train as T
from model.train import Spec

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "trials.md"
CONFIG_PATH = ROOT / "serving_config.json"
OUT = T.pit.CACHE_DIR / "trials"
TEST_SEASON, REP_SEASON, LIVE_SEASON = 2025, 2024, 2026
RECENT_FACTOR = 3.0
LIBS = (*T.LIBS, "average")
VERDICTS = ("SHIPS", "DIRECTION ONLY", "DOES NOT SHIP", "TIE (components)")
# how `encode` renames the raw registry columns; everything else keeps its name
ENCODED = {"inj_report_status": ("inj_report_ord",), "inj_practice_status": ("inj_practice_ord",),
           "wx_roof_obs": ("wx_roof_dome", "wx_roof_retractable")}


def log(msg: str) -> None:
    print(msg, flush=True)


def _rel(path: Path) -> str:
    """The path as printed: relative to the repo when it is inside it."""
    try:
        return str(Path(path).relative_to(ROOT.parent))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- the serving primary
def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def serving_spec(cfg: dict) -> Spec:
    """The frozen spec, exactly as `serve.spec_from` builds it."""
    s = cfg["spec"]
    return Spec(s["name"], drop_cols=tuple(s["drop_cols"]), drop_families=tuple(s["drop_families"]))


def column_status(col: str, fam: str, spec: Spec, inputs: set[str]) -> str:
    """Why a registry column is in or out of the serving inputs."""
    if any(c in inputs for c in ENCODED.get(col, (col,))):
        return "IN"
    if fam == "weather":
        return "out: weather (backtest-only proxy)"
    if col == "season":
        return "out: season excluded (not extrapolable)"
    if fam in T.OPT_IN_FAMILIES:
        return "out: opt-in family (+family:%s)" % fam
    if fam in spec.drop_families:
        return "out: family dropped (%s)" % fam
    if col in spec.drop_cols:
        return "out: pruned (+cols:%s to restore)" % col
    return "out: not in the frozen inputs"


def list_families(cfg: dict) -> str:
    spec = serving_spec(cfg)
    inputs = set(cfg["inputs"])
    lines = []
    for fam, cols in F.FAMILIES.items():
        st = {c: column_status(c, fam, spec, inputs) for c in cols}
        n_in = sum(v == "IN" for v in st.values())
        tag = "serving" if n_in == len(cols) else ("partly in serving" if n_in else "not in serving")
        if fam in T.OPT_IN_FAMILIES:
            tag += ", opt-in"
        lines.append(f"{fam}  [{n_in}/{len(cols)} {tag}]")
        for c, v in st.items():
            lines.append(f"    {'*' if v == 'IN' else ' '} {c:28s} {v if v != 'IN' else ''}".rstrip())
    lines.append(f"\n{len(inputs)} serving inputs (incl. 4 position one-hots), frozen {cfg.get('frozen_on')}; '*' = in the inputs.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- candidates
@dataclass(frozen=True)
class Candidate:
    kind: str                 # add_family | drop_family | add_cols | drop_cols | param | recent
    text: str                 # the string as typed (runnable again)
    family: str = ""
    cols: tuple = ()
    lib: str = ""
    key: str = ""
    value: object = None
    n: int = 0
    factor: float = RECENT_FACTOR


def _value(s: str):
    """int, float, bool or str, whichever the text is."""
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def parse_candidate(text: str) -> Candidate:
    t = text.strip()
    if not t:
        raise ValueError("empty candidate")
    head, sep, rest = t.partition(":")
    if not sep or not rest:
        raise ValueError(f"{text!r}: expected <kind>:<value>, e.g. recent:4, +family:pbp_usage, param:lightgbm.reg_lambda=50")
    if head in ("+family", "-family"):
        if rest not in F.FAMILIES:
            raise ValueError(f"{text!r}: unknown family {rest!r}; families: {', '.join(F.FAMILIES)}")
        return Candidate("add_family" if head[0] == "+" else "drop_family", t, family=rest)
    if head in ("+cols", "-cols"):
        cols = tuple(c.strip() for c in rest.split(",") if c.strip())
        known = set(F.feature_columns())
        bad = [c for c in cols if c not in known]
        if bad:
            raise ValueError(f"{text!r}: not registry columns: {bad}")
        return Candidate("add_cols" if head[0] == "+" else "drop_cols", t, cols=cols)
    if head == "param":
        lk, eq, val = rest.partition("=")
        lib, dot, key = lk.partition(".")
        if not (eq and dot and key and val):
            raise ValueError(f"{text!r}: expected param:<lib>.<key>=<value>")
        if lib not in T.LIBS:
            raise ValueError(f"{text!r}: lib must be one of {T.LIBS}")
        return Candidate("param", t, lib=lib, key=key, value=_value(val))
    if head == "recent":
        n_s, x, f_s = rest.partition("x")
        try:
            n, factor = int(n_s), (float(f_s) if x else RECENT_FACTOR)
        except ValueError:
            raise ValueError(f"{text!r}: expected recent:<n> or recent:<n>x<factor>") from None
        if n < 1 or factor <= 0:
            raise ValueError(f"{text!r}: n >= 1 and factor > 0")
        return Candidate("recent", t, n=n, factor=factor)
    raise ValueError(f"{text!r}: unknown candidate kind {head!r} (+family, -family, +cols, -cols, param, recent)")


@dataclass(frozen=True)
class Variant:
    """What a trial compares against the primary: a spec, parameter overrides per library, an optional recency weighting."""
    spec: Spec
    params: dict = field(default_factory=dict)          # {lib: {key: value}}
    recent: tuple | None = None                         # (n weeks, factor)
    candidates: tuple = ()

    @property
    def text(self) -> str:
        return " ".join(c.text for c in self.candidates)

    def changes_inputs(self, base: Spec) -> bool:
        return replace(self.spec, name=base.name) != base

    def lib_params(self, lib: str) -> dict:
        """The tuned parameters of `lib` with this variant's overrides applied."""
        return {**T.load_params()["point"][lib], **self.params.get(lib, {})}


def compose(candidates, base: Spec) -> Variant:
    """Several candidates into one variant, each checked against the state the earlier ones left."""
    cands = tuple(c if isinstance(c, Candidate) else parse_candidate(c) for c in candidates)
    if not cands:
        raise ValueError("no candidate")
    spec, params, recent = base, {}, None
    for c in cands:
        if c.kind == "add_family":
            if c.family == "weather":
                raise ValueError(f"{c.text}: weather is observed post-game and cannot be served")
            if c.family in spec.drop_families:
                spec = replace(spec, drop_families=tuple(f for f in spec.drop_families if f != c.family))
            elif c.family in T.OPT_IN_FAMILIES:
                if c.family in spec.add_families:
                    raise ValueError(f"{c.text}: already added")
                spec = replace(spec, add_families=(*spec.add_families, c.family))
            else:
                raise ValueError(f"{c.text}: family {c.family!r} is already in the inputs (its pruned columns, if any, come back with +cols:)")
        elif c.kind == "drop_family":
            if c.family in spec.drop_families or (c.family in T.OPT_IN_FAMILIES and c.family not in spec.add_families):
                raise ValueError(f"{c.text}: family {c.family!r} is not in the inputs")
            spec = replace(spec, drop_families=(*spec.drop_families, c.family),
                           add_families=tuple(f for f in spec.add_families if f != c.family))
        elif c.kind == "add_cols":
            missing = [x for x in c.cols if x not in spec.drop_cols]
            if missing:
                raise ValueError(f"{c.text}: not in the spec's drop list (already in, or out for another reason): {missing}")
            spec = replace(spec, drop_cols=tuple(x for x in spec.drop_cols if x not in c.cols))
        elif c.kind == "drop_cols":
            dup = [x for x in c.cols if x in spec.drop_cols]
            if dup:
                raise ValueError(f"{c.text}: already dropped: {dup}")
            spec = replace(spec, drop_cols=(*spec.drop_cols, *c.cols))
        elif c.kind == "param":
            if c.key in params.get(c.lib, {}):
                raise ValueError(f"{c.text}: {c.lib}.{c.key} set twice")
            params.setdefault(c.lib, {})[c.key] = c.value
        elif c.kind == "recent":
            if recent is not None:
                raise ValueError(f"{c.text}: recent given twice")
            recent = (c.n, c.factor)
    name = "trial_" + hashlib.md5(" ".join(c.text for c in cands).encode()).hexdigest()[:8]
    return Variant(replace(spec, name=name), params, recent, cands)


# --------------------------------------------------------------------------- recency weights
def recent_weights(df: pd.DataFrame, train: np.ndarray, season: int, week: int, n: int, factor: float = RECENT_FACTOR) -> np.ndarray:
    """One weight per matrix row: `factor` for training rows in the n most recent distinct (season, week) keys among the training
    rows (so at week 1 the previous season's last weeks are the recent ones), 1.0 everywhere else. (season, week) is what
    `train_mask` already bounded strictly below the fit week, so nothing here can look ahead."""
    key = (df["season"].to_numpy() * 100 + df["week"].to_numpy()).astype("int64")
    keys = np.unique(key[train])
    recent = keys[-n:] if len(keys) else keys
    w = np.ones(len(df), dtype="float64")
    w[train & np.isin(key, recent)] = factor
    return w


def weight_fn_for(variant: Variant):
    if variant.recent is None:
        return None
    n, factor = variant.recent
    return lambda df, train, season, week: recent_weights(df, train, season, week, n, factor)


# --------------------------------------------------------------------------- the frame
def seasons_with_labels(df: pd.DataFrame) -> dict[int, tuple]:
    """{season: weeks that have at least one labelled played row}."""
    lab = df[(df["y_played"] == 1) & df[T.TARGET].notna()]
    return {int(s): tuple(int(w) for w in sorted(g["week"].unique())) for s, g in lab.groupby("season")}


def load_frame(matrix: Path | None, *, season: int = LIVE_SEASON, log=log) -> tuple[pd.DataFrame, str, str]:
    """(frame, matrix key, where it came from). The Phase 2.5 matrix if it exists, else the serving path's history."""
    path = Path(matrix) if matrix else T.MATRIX
    if path.exists():
        df = T.load_matrix(path)
        src = f"matrix {path}"
    else:
        log(f"  {path} is missing: building the frame from the serving path's history cache (no download)")
        from model import serve as S
        store = S.load_store(season, refresh=False)
        df = S.training_frame(store, season, 99, jobs=1, log=log).reset_index(drop=True)
        src = "serve.training_frame (history cache + this season's completed weeks)"
    return df, P.matrix_key(df), src


# --------------------------------------------------------------------------- fits
def _cached(name: str, key: str, fn, *, force: bool = False) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    f, k = OUT / f"{name}.parquet", OUT / f"{name}.key"
    if not force and f.exists() and k.exists() and k.read_text() == key:
        return pd.read_parquet(f)
    res = fn()
    res.to_parquet(f)
    k.write_text(key)
    return res


def tree_count(df: pd.DataFrame, mkey: str, spec: Spec, lib: str, params: dict, *, force: bool = False) -> int:
    """The Phase 2.5 protocol for a variant's tree count: early stopping on 2024 after training on 2021-2023, x1.1 (floor 50)."""
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "tree_counts.json"
    book = json.loads(f.read_text()) if f.exists() else {}
    shape = {k: v for k, v in params.items() if k not in ("n_estimators", "iterations")}
    key = P._key(mkey, repr(spec), lib, shape)
    if force or key not in book:
        dead = T.dead_columns(df)
        cols = T.feature_columns(df, spec, dead)
        played = df[(df["y_played"] == 1) & df[T.TARGET].notna() & (df["season"] >= spec.min_season)]
        tr, va = played[played["season"].isin(T.TRAIN_SEASONS_FOR_TUNING)], played[played["season"] == T.VALID_SEASON]
        Xtr, _ = T.encode(tr, cols)
        Xva, _ = T.encode(va, cols)
        _, n, loss = T._fit_with_early_stopping(lib, shape, Xtr, tr[T.TARGET].to_numpy(), Xva, va[T.TARGET].to_numpy())
        book[key] = {"spec": spec.name, "lib": lib, "best_iteration": int(n), "valid_rmse": round(float(loss), 4)}
        f.write_text(json.dumps(book, indent=1, sort_keys=True))
    return max(50, int(round(book[key]["best_iteration"] * T.FOLLOW_UP_TREE_FACTOR)))


def params_for(df, mkey, variant: Variant, base: Spec, lib: str, *, retree: bool = True) -> dict:
    """The variant's parameters for `lib`: tuned + overrides, with its own tree count when its inputs or shape differ."""
    p = variant.lib_params(lib)
    if retree and (variant.changes_inputs(base) or variant.params.get(lib)):
        key = "iterations" if lib == "catboost" else "n_estimators"
        if key not in variant.params.get(lib, {}):                   # an explicit tree-count candidate is taken as given
            p[key] = tree_count(df, mkey, variant.spec, lib, p)
    return p


def run_one(df, mkey, spec: Spec, lib: str, params: dict, season: int, weeks, *, weight_fn=None, tag: str,
            force: bool = False, log=log) -> pd.DataFrame:
    """One cached walk-forward (`backtest.walk_forward`, the same function serving uses)."""
    key = P._key(mkey, repr(spec), lib, params, season, tuple(weeks), tag)
    name = f"{spec.name}__{lib}__{season}" + (f"__{tag}" if tag else "")
    t0 = time.time()
    res = _cached(name, key, lambda: B.walk_forward(df, spec, lib, params, season=season, weeks=tuple(weeks), weight_fn=weight_fn),
                  force=force)
    log(f"    {name}: {len(res):,} rows ({time.time() - t0:,.0f}s)")
    return res


def _average(runs: list[pd.DataFrame]) -> pd.DataFrame:
    out = runs[0].copy()
    out["pred"] = np.mean([r["pred"].to_numpy() for r in runs], axis=0)
    return out


def run_variant(df, mkey, variant: Variant, base: Spec, lib: str, season: int, weeks, *, primary: bool = False,
                retree: bool = True, force: bool = False, log=log) -> pd.DataFrame:
    libs = T.LIBS if lib == "average" else (lib,)
    runs = []
    for lb in libs:
        if primary:
            spec, params, wfn, tag = base, T.load_params()["point"][lb], None, ""
        else:
            spec, params, wfn = variant.spec, params_for(df, mkey, variant, base, lb, retree=retree), weight_fn_for(variant)
            tag = f"recent{variant.recent[0]}x{variant.recent[1]:g}" if variant.recent else ""
        runs.append(run_one(df, mkey, spec, lb, params, season, weeks, weight_fn=wfn, tag=tag, force=force, log=log))
    return runs[0] if len(runs) == 1 else _average(runs)


# --------------------------------------------------------------------------- verdicts
def verdict(d25: pd.Series | dict, d24: pd.Series | dict, *, rule: str = "rmse") -> str:
    """The adoption rule on pooled dRMSE (variant minus primary). `rule='rmse'`: SHIPS / DIRECTION ONLY / DOES NOT SHIP.
    `rule='tie'` (component models): DOES NOT SHIP only if a year is significantly worse; SHIPS if it beats on both; else TIE."""
    if rule == "tie":
        if d25["dRMSE_lo"] > 0 or d24["dRMSE_lo"] > 0:
            return "DOES NOT SHIP"
        if d25["dRMSE_hi"] < 0 and d24["dRMSE"] < 0:
            return "SHIPS"
        return "TIE (components)"
    if rule != "rmse":
        raise ValueError(rule)
    if d25["dRMSE"] < 0 and d24["dRMSE"] < 0:
        return "SHIPS" if d25["dRMSE_hi"] < 0 else "DIRECTION ONLY"
    return "DOES NOT SHIP"


def pm(d, metric: str, fmt: str = "{:+.3f}") -> str:
    return f"{fmt.format(d[metric])} [{fmt.format(d[metric + '_lo'])}, {fmt.format(d[metric + '_hi'])}]"


def adoption_steps(variant: Variant, base_cfg: dict, lib: str) -> list[str]:
    """What a human edits to adopt the variant; the harness never edits these files itself."""
    steps = []
    s = base_cfg["spec"]
    if variant.spec.drop_cols != tuple(s["drop_cols"]) or variant.spec.drop_families != tuple(s["drop_families"]) or variant.spec.add_families:
        steps.append(f"edit {_rel(CONFIG_PATH)} 'spec': drop_cols -> {sorted(variant.spec.drop_cols)}; "
                     f"drop_families -> {list(variant.spec.drop_families)}"
                     + (f"; add_families {list(variant.spec.add_families)} (serve.spec_from must learn to read add_families: "
                        f"it builds the Spec from drop_cols/drop_families only)" if variant.spec.add_families else ""))
    for lb, kv in variant.params.items():
        steps.append(f"edit {_rel(T.PARAMS_PATH)} point.{lb}: {kv} (and its tree count: this trial's is in "
                     f"{_rel(OUT / 'tree_counts.json')})")
    if variant.recent:
        steps.append(f"recent:{variant.recent[0]}x{variant.recent[1]:g} is not a config switch: serve.predict_week must pass "
                     f"backtest.walk_forward(weight_fn=trial.weight_fn_for(variant)); add it, then a serve-equals-backtest test")
    steps.append("then: python -m model.serve --freeze   (rewrites the frozen inputs and component tree counts; review the diff)")
    steps.append("then: one line in PLAN_MODEL.md 'Phase 5' and the status column of model/candidates.md")
    if lib != "average":
        steps.append(f"judged on {lib} alone (the serving point estimate is the 3-library average: re-run with --lib average before adopting)")
    return steps


# --------------------------------------------------------------------------- the trial
def trial(df, mkey, variant: Variant, base: Spec, *, lib: str = "lightgbm", draws: int = 2000, seasons=(TEST_SEASON, REP_SEASON, LIVE_SEASON),
          retree: bool = True, force: bool = False, log=log) -> dict:
    """{season: p25run.compare result (or None when the frame has no labelled weeks)}, plus 'weeks'."""
    have = seasons_with_labels(df)
    out = {"weeks": {}, "compare": {}}
    for season in seasons:
        weeks = have.get(season, ())
        out["weeks"][season] = weeks
        if not weeks:
            log(f"  {season}: no completed weeks in the frame, skipped")
            out["compare"][season] = None
            continue
        log(f"  {season}: weeks {weeks[0]}-{weeks[-1]} ({len(weeks)})")
        res = {"primary": run_variant(df, mkey, variant, base, lib, season, weeks, primary=True, force=force, log=log),
               "variant": run_variant(df, mkey, variant, base, lib, season, weeks, retree=retree, force=force, log=log)}
        out["compare"][season] = P.compare(res, "primary", draws=draws)
    return out


def verdict_of(result: dict, *, rule: str = "rmse") -> str:
    c25, c24 = result["compare"].get(TEST_SEASON), result["compare"].get(REP_SEASON)
    if c25 is None or c24 is None:
        return "NO VERDICT (2025 or 2024 missing from the frame)"
    return verdict(c25["delta"].loc["variant"], c24["delta"].loc["variant"], rule=rule)


def table_row(result: dict, variant: Variant, lib: str, rule: str, *, today: str | None = None) -> dict:
    row = {"date": today or date.today().isoformat(), "candidate": f"`{variant.text}`", "lib": lib}
    for season, label in ((TEST_SEASON, "2025"), (REP_SEASON, "2024"), (LIVE_SEASON, "2026")):
        c = result["compare"].get(season)
        if c is None:
            row[f"{label} dRMSE"] = row[f"{label} dPick"] = "n/a"
            continue
        d = c["delta"].loc["variant"]
        wk = result["weeks"][season]
        suffix = f" ({len(wk)} wk, n={c['n']:,})" if season == LIVE_SEASON else ""
        row[f"{label} dRMSE"] = pm(d, "dRMSE") + suffix
        row[f"{label} dPick"] = pm(d, "dPick")
    row["verdict"] = verdict_of(result, rule=rule) + (" [tie rule]" if rule == "tie" else "")
    return row


def md_table(rows: list[dict], header: bool = True) -> str:
    cols = list(rows[0])
    lines = []
    if header:
        lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


REPORT_HEAD = """# Trials

One row per `python -m model.trial` run (PLAN_MODEL.md "Phase 5"). dRMSE / dPick are variant minus primary on the Phase 2.5
head-to-head rows with 95% week-blocked bootstrap intervals; negative dRMSE and positive dPick mean the variant is better. The verdict
is the Phase 2.5 adoption rule on pooled dRMSE: SHIPS (beats on 2025, interval below zero, same sign on 2024), DIRECTION ONLY (better
both years, 2025 interval spans zero), DOES NOT SHIP, or TIE (components) under `--rule tie`. 2026 is the live year, reported only: it
never decides. Adopting a SHIPS is a human edit of `model/serving_config.json` or `model/tuned_params.json`, then
`python -m model.serve --freeze`.

| date | candidate | lib | 2025 dRMSE | 2025 dPick | 2024 dRMSE | 2024 dPick | 2026 dRMSE | 2026 dPick | verdict |
|---|---|---|---|---|---|---|---|---|---|
"""


def append_report(row: dict, path: Path | None = None) -> None:
    path = Path(path or REPORT)                 # resolved at call time, so a patched REPORT is honoured
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else REPORT_HEAD
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text + md_table([row], header=False) + "\n", encoding="utf-8")


def per_position_lines(result: dict) -> list[str]:
    lines = []
    for season in (TEST_SEASON, REP_SEASON, LIVE_SEASON):
        c = result["compare"].get(season)
        if c is None:
            continue
        bt = c["delta_pos"]["variant"]
        lines.append(f"  {season} by position, dRMSE / dPick: " + "; ".join(
            f"{pos} {bt.loc[pos, 'dRMSE']:+.3f} / {bt.loc[pos, 'dPick']:+.3f}" for pos in bt.index))
        sc = c["scores"]
        lines.append(f"  {season} pooled RMSE / pick: primary {sc.loc['primary', 'RMSE']:.3f} / {sc.loc['primary', 'pick_acc']:.3f}, "
                     f"variant {sc.loc['variant', 'RMSE']:.3f} / {sc.loc['variant', 'pick_acc']:.3f}  (n={c['n']:,})")
    return lines


# --------------------------------------------------------------------------- entry point
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidate", action="append", default=[], help="repeatable; all flags compose into one variant")
    ap.add_argument("--list", action="store_true", help="families, columns, and which are in the serving inputs")
    ap.add_argument("--lib", default="lightgbm", choices=LIBS, help="the library judged (average = the serving 3-library mean, 3x the time)")
    ap.add_argument("--rule", default="rmse", choices=("rmse", "tie"), help="tie: the component-model rule (a tie within noise ships)")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--matrix", type=Path, default=None, help=f"the frame (default {T.MATRIX}, else the serve history cache)")
    ap.add_argument("--no-live", action="store_true", help="skip the 2026 walk-forward")
    ap.add_argument("--no-retree", action="store_true", help="keep the primary's tree count even when inputs or shape change")
    ap.add_argument("--force", action="store_true", help="ignore cached walk-forwards")
    ap.add_argument("--no-report", action="store_true", help="print only; do not append to model/reports/trials.md")
    a = ap.parse_args(argv)
    cfg = load_config()
    if a.list:
        print(list_families(cfg))
        print("\nscreen every column outside the inputs at once (rho with the primary's residual, single-fit gain): "
              "python -m model.screen  -> model/reports/screen.md")
        return 0
    if not a.candidate:
        ap.error("give at least one --candidate, or --list")
    base = serving_spec(cfg)
    variant = compose(a.candidate, base)
    log(f"variant {variant.spec.name}: {variant.text}")
    log(f"  spec: drop_cols {len(variant.spec.drop_cols)} (primary {len(base.drop_cols)}), drop_families {list(variant.spec.drop_families)}, "
        f"add_families {list(variant.spec.add_families)}, params {variant.params or '-'}, recent {variant.recent or '-'}")
    t0 = time.time()
    df, mkey, src = load_frame(a.matrix)
    log(f"frame {df.shape} from {src}, key {mkey}")
    seasons = (TEST_SEASON, REP_SEASON) if a.no_live else (TEST_SEASON, REP_SEASON, LIVE_SEASON)
    result = trial(df, mkey, variant, base, lib=a.lib, draws=a.draws, seasons=seasons, retree=not a.no_retree, force=a.force)
    row = table_row(result, variant, a.lib, a.rule)
    print()
    print(md_table([row]))
    for line in per_position_lines(result):
        print(line)
    if not a.no_report:
        append_report(row)
        print(f"\nappended to {_rel(REPORT)}")
    v = row["verdict"]
    print(f"\nverdict: {v}")
    if v.startswith("SHIPS") or v.startswith("TIE"):
        print("to adopt (by hand; this harness edits no config):")
        for s in adoption_steps(variant, cfg, a.lib):
            print(f"  - {s}")
    print(f"{time.time() - t0:,.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
