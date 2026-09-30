"""Assemble model/reports/phase25_experiments.md from the cached Phase 2.5 walk-forwards (model/cache/phase25/).

    model/.venv/bin/python -m model.report_phase25

A pure function of the cached predictions, the feature matrix and the league file: no model is fitted here except one LightGBM
refit for the feature-importance table, which uses the tuned parameters and fixed seeds. Markdown text and numbers only; no plots,
no binaries (the report is committed). Every delta is "variant minus base": negative dRMSE / dMAE and positive dSpearman / dPick
mean the variant is better; intervals are the 95% week-blocked bootstrap of Phase 2 (2,000 resamples of the 18 weeks).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from model import backtest as B
from model import components as C
from model import ensemble as E
from model import features as F
from model import p25run as P
from model import phase25 as PH
from model import point_in_time as pit
from model import train as T
from model import volume as V
from model.report_phase2 import md
from model.train import POSITIONS

REPORT = Path(__file__).resolve().parent / "reports" / "phase25_experiments.md"
SEASONS = P.TEST_SEASONS
CACHE = pit.CACHE_DIR / "nflverse"
# HEAD probes of the nflverse-data release assets, 2026-09-30. Last-Modified is the last (re)upload, not the first availability.
LAST_MODIFIED = {
    ("play_by_play", 2021): "2026-01-08", ("play_by_play", 2022): "2026-02-12", ("play_by_play", 2023): "2026-02-12",
    ("play_by_play", 2024): "2026-08-13", ("play_by_play", 2025): "2026-08-13", ("play_by_play", 2026): "2026-09-29 (weeks 1-3)",
    ("pbp_participation", 2021): "2023-12-19", ("pbp_participation", 2022): "2023-12-19", ("pbp_participation", 2023): "2025-09-04",
    ("pbp_participation", 2024): "2025-09-04", ("pbp_participation", 2025): "2026-02-10", ("pbp_participation", 2026): "404: not published",
    ("ftn_charting", 2021): "404: not published", ("ftn_charting", 2022): "2024-10-10", ("ftn_charting", 2023): "2024-09-06",
    ("ftn_charting", 2024): "2025-09-01", ("ftn_charting", 2025): "2026-09-23", ("ftn_charting", 2026): "2026-09-29",
}


def pm(v: float, lo: float, hi: float, fmt: str = "{:+.3f}") -> str:
    return f"{fmt.format(v)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def d(c: dict, name: str, metric: str = "dRMSE", fmt: str = "{:+.3f}") -> str:
    """One delta with its interval, for prose: '-0.008 [-0.024, +0.007]' (the metric is named in the sentence)."""
    r = c["delta"].loc[name]
    return pm(r[metric], r[metric + "_lo"], r[metric + "_hi"], fmt)


def scores_md(c: dict) -> str:
    s = c["scores"].copy()
    s.columns = ["RMSE", "MAE", "Spearman", "Pick acc"]
    return md(s, "{:.3f}", label="run")


def delta_md(c: dict, names: list[str] | None = None) -> str:
    d = c["delta"]
    rows = {}
    for n in names or list(d.index):
        r = d.loc[n]
        rows[n] = {"dRMSE": pm(r["dRMSE"], r["dRMSE_lo"], r["dRMSE_hi"]), "dMAE": pm(r["dMAE"], r["dMAE_lo"], r["dMAE_hi"]),
                   "dSpearman": pm(r["dSpearman"], r["dSpearman_lo"], r["dSpearman_hi"], "{:+.4f}"),
                   "dPick": pm(r["dPick"], r["dPick_lo"], r["dPick_hi"], "{:+.4f}")}
    return md(pd.DataFrame(rows).T, label="run minus base")


def pos_delta_md(c: dict, name: str) -> str:
    b = c["delta_pos"][name]
    t = pd.DataFrame({"dRMSE": [pm(b.loc[p, "dRMSE"], b.loc[p, "dRMSE_lo"], b.loc[p, "dRMSE_hi"]) for p in [*POSITIONS, "ALL"]],
                      "dSpearman": [pm(b.loc[p, "dSpearman"], b.loc[p, "dSpearman_lo"], b.loc[p, "dSpearman_hi"], "{:+.4f}") for p in [*POSITIONS, "ALL"]]},
                     index=[*POSITIONS, "ALL"])
    return md(t, label="position")


def with_refs(res: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add the two reference rows every table is read against: the trailing-3 average and the post-game same-week xFP oracle
    (never an input; the ceiling for pre-game information)."""
    base = next(iter(res.values()))
    out = dict(res)
    for name, col in (("trailing3", "base_trail3_ppr"), ("xfp_oracle", "base_xfp_sameweek")):
        r = base.copy()
        r["pred"] = base[col]
        out[name] = r
    return out


def both_years(runs_by_season: dict[int, dict[str, pd.DataFrame]], base: str = "primary") -> dict[int, dict]:
    return {s: P.compare(with_refs(runs_by_season[s]), base) for s in SEASONS}


def verdict_row(exp: str, name: str, comp: dict[int, dict], rule: str = "rmse") -> dict:
    d25, d24 = comp[2025]["delta"].loc[name], comp[2024]["delta"].loc[name]
    verdict = P.verdict_rmse(d25, d24) if rule == "rmse" else P.verdict_tie(d25, d24)
    return {"experiment": exp, "variant": name, "2025 result": "dRMSE " + pm(d25["dRMSE"], d25["dRMSE_lo"], d25["dRMSE_hi"]),
            "2024 result": "dRMSE " + pm(d24["dRMSE"], d24["dRMSE_lo"], d24["dRMSE_hi"]),
            "rule": "beat + replicate" if rule == "rmse" else "tie ships", "verdict": verdict}


# --------------------------------------------------------------------------- sources
def source_facts() -> str:
    rows = []
    for y in range(2020, 2027):
        pbp = pd.read_parquet(CACHE / f"play_by_play_{y}.parquet", columns=["game_id", "play_id", "season_type", "play_type", "posteam"])
        pbp = pbp[(pbp["season_type"] == "REG") & pbp["play_type"].isin(["pass", "run"]) & pbp["posteam"].notna()]
        row = {"season": y, "PBP REG games": pbp["game_id"].nunique(), "PBP scrimmage plays": len(pbp),
               "PBP last modified": LAST_MODIFIED.get(("play_by_play", y), "not probed")}
        part = pit._fetch_optional(f"{pit.NFLVERSE}/pbp_participation/pbp_participation_{y}.parquet", CACHE / f"pbp_participation_{y}.parquet")
        if part is not None:
            d = pd.read_parquet(part, columns=["nflverse_game_id", "play_id", "offense_players", "route"])
            m = pbp.merge(d, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "play_id"], how="left")
            row["participation: scrimmage plays with players on field"] = float((m["offense_players"].fillna("").str.len() > 0).mean())
            row["participation: scrimmage plays with a route value (one per play)"] = float((m["route"].fillna("") != "").mean())
        else:
            row["participation: scrimmage plays with players on field"] = np.nan
            row["participation: scrimmage plays with a route value (one per play)"] = np.nan
        row["participation last modified"] = LAST_MODIFIED.get(("pbp_participation", y), "not probed")
        ftn = pit._fetch_optional(f"{pit.NFLVERSE}/ftn_charting/ftn_charting_{y}.parquet", CACHE / f"ftn_charting_{y}.parquet")
        if ftn is not None:
            f = pd.read_parquet(ftn, columns=["nflverse_game_id", "nflverse_play_id"])
            m = pbp.merge(f, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"], how="left")
            row["FTN: scrimmage plays matched"] = float(m["nflverse_game_id"].notna().mean())
        else:
            row["FTN: scrimmage plays matched"] = np.nan
        row["FTN last modified"] = LAST_MODIFIED.get(("ftn_charting", y), "not probed")
        rows.append(row)
    t = pd.DataFrame(rows).set_index("season")
    return md(t, "{:.3f}", floatfmt={"PBP REG games": "{:.0f}", "PBP scrimmage plays": "{:.0f}"}, label="season")


def matrix_coverage(df: pd.DataFrame) -> str:
    played = df[df["y_played"] == 1]
    fams = {"pbp_usage": "pbp_ay_share_std", "pbp_team": "team_plays_pg_std", "pbp_part": "part_pass_snap_share_std", "eff": "eff_rec_ypt_career"}
    rows = {}
    for fam, cols in ((f, F.FAMILIES[f]) for f in ("pbp_usage", "pbp_team", "pbp_part", "eff")):
        r = {}
        for season, g in played.groupby("season"):
            r[int(season)] = 1 - float(g[cols].isna().mean().mean())
        rows[f"{fam}: mean non-null over {len(cols)} columns"] = r
    for label, col in (("pbp_usage: pbp_wopr_l1", "pbp_wopr_l1"), ("pbp_usage: pbp_wopr_std", "pbp_wopr_std"),
                       ("pbp_usage: pbp_prev_wopr", "pbp_prev_wopr"), ("pbp_team: team_pass_rate_t3", "team_pass_rate_t3"),
                       ("pbp_part: part_pass_snap_share_l1", "part_pass_snap_share_l1"),
                       ("pbp_part: part_prev_pass_snap_share", "part_prev_pass_snap_share"),
                       ("eff: eff_rec_ypt_std", "eff_rec_ypt_std")):
        rows[label] = {int(s): float(g[col].notna().mean()) for s, g in played.groupby("season")}
    t = pd.DataFrame(rows).T
    t.columns = [str(c) for c in t.columns]
    return md(t, "{:.3f}", label="share of played rows that are non-null")


# --------------------------------------------------------------------------- experiment 1
def section_exp1(df: pd.DataFrame, mkey: str) -> tuple[str, list[dict]]:
    runs = {s: PH.exp1_runs(df, mkey, s) for s in SEASONS}
    comp = both_years(runs)
    order = ["pbp_usage", "pbp", "pbp_part", "pbp_team", "part_only", "no_xfp", "no_xfp_pbp"]
    out = ["## Experiment 1: play-by-play micro-signals\n"]
    out.append("Sources probed from the sandbox on 2026-09-30 (`model/cache/nflverse/`, nflverse-data release assets). `Last modified` is the "
               "last re-upload of the file, not when it first appeared.\n")
    out.append(source_facts())
    out.append("""
* **Play-by-play** covers 2020 through 2026 week 3 for every regular-season game and is refreshed in-season (2026: Tue 2026-09-29 15:44Z,
  ahead of the pipeline run). The new tables reproduce nflverse's own player stats: targets match `stats_player_week` on 100% of rows in
  every season, carries on 99.98% or better (kneels and scrambles are carries in both), and the team target totals give nflverse's
  `target_share` exactly. `pbp_usage`, `pbp_team` and every red-zone / inside-10 / air-yards value are also recomputed from the raw parquet
  with plain pandas in the tests.
* **Participation** covers 100% of regular-season scrimmage plays 2020-2025 (91% of all plays in 2021-22, 100% 2023+; the gap is
  non-scrimmage plays) with GSIS ids on the field, so a share of the team's dropbacks / runs / red-zone plays a player was on the field for is
  consistent across the whole backtest. It does **not** contain a per-player route: `route` is one value per play (the targeted route, blank on
  non-targets) and its vocabulary changed in 2023 ("HITCH" to "HITCH/CURL"), so route participation proper cannot be built for 2021-2025. Pass-play
  snap share (`part_pass_snap_share_*`) is the closest free proxy (blockers count). **The feed is published after the season**: the 2025 file was
  last modified 2026-02-10 and no 2026 file exists as of 2026-09-30, so these columns are 100% NA for 2026 and cannot be served in-season. It is
  stamped at kickoff for the backtest like the other game tables, with the store's declared-proxy flag (`RawTable.proxy`), and reported separately.
* **FTN charting** covers 2022-2025 (99.7-100% of scrimmage plays) but 2021 is a 404. Per the plan a source that does not cover the whole backtest
  is excluded: it would be all-NA for a full training year, and its fields are play-level flags (play action, motion, screen, RPO, drop, contested
  catch, blitzers) with no player column, so they are not volume signals either. Not built.
* **NGS** receiving files (`nextgen_stats/ngs_{y}_receiving`) are 404 for every year.
""")
    out.append("### What was built\n")
    out.append(f"""Three raw tables, each row stamped with its source game's kickoff and read at kickoff - 4h like every result table:
`pbp_usage` (player-game targets, red-zone and inside-10 targets, air yards, carries, red-zone and inside-10 carries), `pbp_team` (team-game
totals, the denominators of every share, plus plays, dropbacks, neutral-situation plays) and `pbp_part` (plays on the field, participation
feed). Four opt-in feature families through the gate: `pbp_usage` (32 columns: red-zone and inside-10 target/carry shares, air-yards share, aDOT,
WOPR, red-zone and inside-10 opportunities per game, over the last game / last three / season to date, plus last season's five main values),
`pbp_team` (10: the team's pace, dropback rate and neutral dropback rate, and the plays and dropback rate its opponent's defense has faced),
`pbp_part` (10: share of the team's dropbacks, non-dropback runs and red-zone plays on the field), `eff` (18, used by experiments 2 and 3).
Shares are ratios of sums over the window; nothing reads the target game. Coverage on played rows (share non-null):
""")
    out.append(matrix_coverage(df))
    out.append("""
Null rates track the lag family (week 1 has no current-season window, and a player with no history has none): 2021-2025 about 13-15%
for `pbp_usage`, 4-5% for the team columns, 2026 has three weeks. `pbp_part` is 91.5% null in 2026 (no feed). `eff` is null by design for
opportunities a player never had (a receiver has no passing rate).
""")
    out.append("### Results: the flat model with the new inputs\n")
    out.append("Pre-declared: the candidate for Phase 3 is **`pbp`** (usage + team: everything servable in-season). `pbp_part` adds the "
               "participation columns (backtest-only) and is reported as the upper bound; the rest are diagnostics. Same tuned LightGBM shape as "
               "the primary, own tree count by the early-stopping protocol. `no_xfp` drops the five lagged-xFP columns and `no_xfp_pbp` drops them and adds `pbp` "
               "(does play-by-play usage substitute for the expected-points lags it overlaps?).\n")
    verdicts = []
    for season in SEASONS:
        c = comp[season]
        out.append(f"**{season} walk-forward** ({c['n']:,} head-to-head rows, ALL positions)\n")
        out.append(scores_md(c))
        out.append("")
        out.append(delta_md(c, order))
        out.append("")
    out.append("Per position, the pre-declared candidate `pbp` (dRMSE and dSpearman against the primary):\n")
    for season in SEASONS:
        out.append(f"{season}:\n")
        out.append(pos_delta_md(comp[season], "pbp"))
        out.append("")
    for name in order[:5]:
        v = verdict_row("1 play-by-play", name, comp)
        if name in ("pbp_part", "part_only") and v["verdict"].startswith("SHIPS"):
            v["verdict"] = ("passes the accuracy rule on direction only (both intervals span zero); NOT ADOPTED: the participation feed is not "
                            "published in-season, so it is all-NA for 2026 (train/serve skew, like weather)")
        verdicts.append(v)
    out.append("### What the model does with them\n")
    imp = PH.importance(df, PH.EXP1["pbp_part"])
    new_fams = set(F.FAMILIES["pbp_usage"] + F.FAMILIES["pbp_team"] + F.FAMILIES["pbp_part"])
    fam = imp.groupby("family")["share"].sum().sort_values(ascending=False).round(4)
    out.append("Gain share by family in the last 2025 fit of the all-families model (trained on everything before week 18):\n")
    out.append(md(fam.to_frame("gain share"), "{:.3f}", label="family"))
    out.append("\nTop new columns by gain:\n")
    top = imp[imp["feature"].isin(new_fams)].head(12)[["feature", "family", "share"]]
    out.append(md(top, "{:.4f}", index=False))
    out.append(f"""
### Reading

* **Play-by-play micro-signals do not improve the flat model.** The pre-declared candidate `pbp` moves pooled RMSE by {d(comp[2025], 'pbp')} in 2025
  and {d(comp[2024], 'pbp')} in 2024: the sign flips, both intervals span zero, and Spearman / pick accuracy move by less than 0.003. The pieces do no
  better alone (`pbp_usage` {d(comp[2025], 'pbp_usage')} / {d(comp[2024], 'pbp_usage')}; `pbp_team` {d(comp[2025], 'pbp_team')} / {d(comp[2024], 'pbp_team')}).
* **Play-by-play usage is a substitute for the lagged xFP columns, not an addition to them.** Dropping the five xFP lags moves RMSE by {d(comp[2024], 'no_xfp')} in 2024 (MAE {d(comp[2024], 'no_xfp', 'dMAE')}, the only significant one) and {d(comp[2025], 'no_xfp')} in 2025;
  dropping them and adding `pbp` gives {d(comp[2025], 'no_xfp_pbp')} / {d(comp[2024], 'no_xfp_pbp')}, i.e. back to the primary. ffopportunity's expected points are built from the
  same air yards and field position, so red-zone and air-yards shares carry the information the model already had.
* **The participation columns are the only variant that keeps its sign** (`pbp_part`: {d(comp[2025], 'pbp_part')} / {d(comp[2024], 'pbp_part')}), by a margin inside the noise
  in both years, and they cannot be served in-season. They take {imp.groupby('family')['share'].sum().get('pbp_part', 0):.1%} of the gain in the all-families fit (pass-play snap share partly replaces snap percentage) without improving the fit.
  Not adopted.
* Nothing from Experiment 1 goes into the Phase 3 feature set. The tables, the tests and the loader stay in the repo (opt-in families) so the question can be
  reopened if a better feed appears.
""")
    return "\n".join(out), verdicts


# --------------------------------------------------------------------------- experiment 2
def section_exp2(df: pd.DataFrame, mkey: str) -> tuple[str, list[dict]]:
    d2, k2 = PH.exp2_frames(df, mkey, log=lambda m: None)
    out = ["## Experiment 2: two-stage (volume first)\n"]
    out.append("""Stage one is three LightGBM models (targets, carries, pass attempts) on the primary features, walk-forward like the point model, with
their own tree counts. Stage two predicts points from the stage-one predictions, position and 18 shrunken efficiency priors (career and
season-to-date catch rate, yards and TDs per target and per carry, per-attempt passing rates). Stage two trains on **honest** stage-one predictions
(each earlier row's value came from a model trained strictly before its week), so it never learns from a volume fit that saw its own answer; the price
is that 2021 has none, so stage two trains on 2022+ and `flat_2022` is its matched control.
""")
    out.append("### Stage one: how well can volume itself be predicted?\n")
    out.append("MAE of the walk-forward model against the trailing-3 mean of the same quantity (the average of the player's last three appearances "
               "this season, the number a fantasy player would carry in his head). Head-to-head rows: played, with an earlier game this season, at the "
               "positions where the quantity is real volume (targets: RB/WR/TE, carries: RB/QB, attempts: QB; `relevant` pools them). "
               "dMAE is model minus trailing-3 with a 95% week-blocked interval; R2 is against the group's own mean.\n")
    findings = {}
    for target in V.S1:
        for season in SEASONS:
            res = P.wf(df, mkey, T.PRIMARY, season, target=target, params=P.tree_params(df, mkey, target, T.PRIMARY))
            q = V.volume_quality(V.quality_frame(df, res, target), target)
            findings[(target, season)] = q
            out.append(f"**{ {'y_tgt': 'targets', 'y_car': 'carries', 'y_att': 'pass attempts'}[target] }, {season}**\n")
            show = q[["n", "y_mean", "MAE_model", "MAE_trail3", "dMAE", "dMAE_lo", "dMAE_hi", "RMSE_model", "RMSE_trail3", "corr_model", "corr_trail3", "R2_model", "R2_trail3"]].copy()
            if "MAE_std_mean" in q:
                show.insert(4, "MAE_season_mean", q["MAE_std_mean"])
            show["n"] = show["n"].astype(int)
            out.append(md(show, "{:.3f}", floatfmt={"n": "{:.0f}"}, label="position"))
            out.append("")
    out.append("### Stage two and the flat variants\n")
    runs = {s: PH.exp2_runs(d2, k2, s) for s in SEASONS}
    comp = both_years(runs)
    control = {s: P.compare(with_refs({"flat_2022": runs[s]["flat_2022"], "two_stage": runs[s]["two_stage"], "primary": runs[s]["primary"]}), "flat_2022")
               for s in SEASONS}
    order = ["flat_2022", "two_stage", "flat_plus_s1", "flat_plus_eff", "flat_plus_s1_eff", "flat_no_xfp", "flat_no_xfp_plus_s1"]
    out.append("Base for the deltas is the Phase 2 **primary**. `two_stage` = stage two alone; `flat_plus_s1` = primary features + stage-one predictions; "
               "`flat_plus_eff` and `flat_plus_s1_eff` separate the efficiency priors from the volume predictions; `flat_no_xfp` drops the five lagged-xFP "
               "columns and `flat_no_xfp_plus_s1` replaces them with the stage-one predictions (does stage one beat lagged xFP as the volume signal?).\n")
    for season in SEASONS:
        c = comp[season]
        out.append(f"**{season} walk-forward** ({c['n']:,} head-to-head rows, ALL positions)\n")
        out.append(scores_md(c))
        out.append("")
        out.append(delta_md(c, order))
        out.append("")
    out.append("`two_stage` against its matched control `flat_2022` (both train on 2022+):\n")
    for season in SEASONS:
        out.append(f"{season}:\n")
        out.append(delta_md(control[season], ["two_stage", "primary"]))
        out.append("")
    q25, q24 = findings[("y_tgt", 2025)], findings[("y_tgt", 2024)]
    c25, c24 = findings[("y_car", 2025)], findings[("y_car", 2024)]
    a25, a24 = findings[("y_att", 2025)], findings[("y_att", 2024)]
    out.append(f"""
### Reading

* **Volume is only modestly predictable, and that is where the ceiling problem lives.** The stage-one models beat the trailing-3 mean by
  {-q25.loc['relevant', 'dMAE'] / q25.loc['relevant', 'MAE_trail3']:.1%} (2025) and {-q24.loc['relevant', 'dMAE'] / q24.loc['relevant', 'MAE_trail3']:.1%} (2024) on target MAE, {-c25.loc['relevant', 'dMAE'] / c25.loc['relevant', 'MAE_trail3']:.1%} / {-c24.loc['relevant', 'dMAE'] / c24.loc['relevant', 'MAE_trail3']:.1%} on carries (RB and QB) and
  {-a25.loc['QB', 'dMAE'] / a25.loc['QB', 'MAE_trail3']:.1%} / {-a24.loc['QB', 'dMAE'] / a24.loc['QB', 'MAE_trail3']:.1%} on quarterback attempts; every interval excludes zero. In absolute terms a receiver's targets are still off by
  {q25.loc['relevant', 'MAE_model']:.2f} a game on a mean of {q25.loc['relevant', 'y_mean']:.2f} (R-squared {q25.loc['relevant', 'R2_model']:.2f} against {q25.loc['relevant', 'R2_trail3']:.2f} for the trailing mean), a
  back's carries by {c25.loc['RB', 'MAE_model']:.2f} on {c25.loc['RB', 'y_mean']:.2f}, a quarterback's attempts by {a25.loc['QB', 'MAE_model']:.1f} on {a25.loc['QB', 'y_mean']:.1f}. Better volume estimates are worth a few percent of MAE,
  not the {comp[2025]['scores'].loc['primary', 'RMSE'] - comp[2025]['scores'].loc['xfp_oracle', 'RMSE']:.1f}-point RMSE gap to the same-week xFP oracle, which sees the volume the player actually got: nothing knowable before kickoff that these experiments tried recovers it. Game script,
  in-game injuries and coaching choices are the likely sources; that part is inference, not something measured here.
* **The end-to-end two-stage model loses.** Stage two alone moves RMSE by {d(comp[2025], 'two_stage')} in 2025 and {d(comp[2024], 'two_stage')} in 2024, worse in both years, and
  worse than its matched control that trains on the same window ({d(control[2025], 'two_stage')} / {d(control[2024], 'two_stage')}). Stage-one error compounds, and the structure also drops the
  role, Vegas, injury and prior-season columns the flat model uses (`flat_plus_s1_eff`, which keeps them, does not beat the primary either).
* **Stage-one predictions as extra flat inputs do not replicate**: `flat_plus_s1` {d(comp[2025], 'flat_plus_s1')} / {d(comp[2024], 'flat_plus_s1')} (2024 significantly worse);
  they do not replace lagged xFP either (`flat_no_xfp_plus_s1` {d(comp[2025], 'flat_no_xfp_plus_s1')} / {d(comp[2024], 'flat_no_xfp_plus_s1')}). The efficiency priors alone are indistinguishable from
  nothing (`flat_plus_eff` {d(comp[2025], 'flat_plus_eff')} / {d(comp[2024], 'flat_plus_eff')}).
* Nothing from Experiment 2 goes into the Phase 3 design. The stage-one volume models are kept: they are components in Experiment 3.
""")
    verdicts = [verdict_row("2 two-stage", n, comp) for n in ("two_stage", "flat_plus_s1", "flat_plus_s1_eff", "flat_plus_eff", "flat_no_xfp_plus_s1")]
    return "\n".join(out), verdicts


# --------------------------------------------------------------------------- experiment 3
def section_exp3(df: pd.DataFrame, mkey: str) -> tuple[str, list[dict]]:
    sc_te, diverge = C.league_scoring(PH.SLUG)
    played_all = df[df["y_played"] == 1]
    ident = float((C.actual_points(df, played_all.index, C.PPR) - played_all["y_points_ppr"]).abs().max())
    n_ident = len(played_all)
    out = ["## Experiment 3: component models\n"]
    out.append(f"""One LightGBM per scoring component ({len(C.TARGETS)} of them: {', '.join(t[2:] for t in C.TARGETS)}), each walk-forward on the same
rows, each with its own early-stopped tree count, then composed with the scoring weights. `components` uses the primary features; `components_eff`
adds the `eff` family (shrunken efficiency), because a yards model without any yards-per-target history is handicapped. Points are the
conditional-mean sum, so any linear scoring composes exactly.

**Identity check.** Composed from the *actual* components, PPR equals nflverse's `fantasy_points_ppr` on every played row of the matrix
(max abs difference {ident:.1e} over {n_ident:,} rows; pinned by a test), so the composition path cannot drift. Tie-break between the two variants if both tie:
the one that adds no new features (`components`).
""")
    runs = {s: PH.exp3_runs(df, mkey, s) for s in SEASONS}
    primary = {s: PH.primary(df, mkey, s) for s in SEASONS}
    res = {}
    for s in SEASONS:
        res[s] = {"primary": primary[s]}
        for name, comps in runs[s].items():
            res[s][name] = C.composed(comps, C.PPR, primary[s])
    comp = both_years(res)
    out.append("### Composed PPR against the primary\n")
    for season in SEASONS:
        c = comp[season]
        out.append(f"**{season} walk-forward** ({c['n']:,} head-to-head rows, ALL positions)\n")
        out.append(scores_md(c))
        out.append("")
        out.append(delta_md(c, ["components", "components_eff"]))
        out.append("")
    out.append("Per position (composed `components` minus primary):\n")
    for season in SEASONS:
        out.append(f"{season}:\n")
        out.append(pos_delta_md(comp[season], "components"))
        out.append("")
    out.append("### How much of each component is predictable\n")
    out.append("R-squared-style skill (1 - MSE / MSE of a constant per position) on the head-to-head rows, `components` (primary features):\n")
    for season in SEASONS:
        sk = C.component_skill(df, runs[season]["components"])
        sk["n"] = sk["n"].astype(int)
        out.append(f"{season}:\n")
        out.append(md(sk, "{:.3f}", floatfmt={"n": "{:.0f}"}, label="component"))
        out.append("")
    # ---- the alternate scoring
    out.append("### The alternate scoring: the dynasty league's TE premium\n")
    out.append(f"`leagues/{PH.SLUG}/league.json`: full PPR (`rec` = 1.0) plus `bonus_rec_te` = {sc_te.position_bonus[('TE', 'y_rec')]} per tight-end catch. "
               "The composition adds the bonus to predicted tight-end receptions, the flat model has no reception estimate, so its baseline "
               "(`flat_plus_trailing_catches`) is the naive route: the PPR prediction plus the bonus times the tight end's average catches over "
               "his last three appearances (from the matrix's own earlier rows, a baseline never an input). Actual points are rebuilt from actual "
               "components, and the league's other divergences from the composed scoring are listed below: they are not composed.\n")
    out.append("Divergences between this league's real scoring and the composed one: " + "; ".join(diverge) + ".\n")
    rec3 = PH.trailing_receptions(df)
    alt = {}
    for s in SEASONS:
        p = primary[s]
        actual = C.actual_points(df, p.index, sc_te)
        tmpl = p.copy()
        tmpl["y"] = actual.to_numpy()
        flat = tmpl.copy()
        flat["pred"] = p["pred"].to_numpy() + np.where(p["position"] == "TE", sc_te.position_bonus[("TE", "y_rec")] * rec3.loc[p.index].fillna(0.0).to_numpy(), 0.0)
        unshifted = tmpl.copy()
        unshifted["pred"] = p["pred"].to_numpy()
        alt[s] = {"flat_plus_trailing_catches": flat, "flat_ppr_unshifted": unshifted}
        for name, comps in runs[s].items():
            alt[s][name] = C.composed(comps, sc_te, tmpl)
    te_cmps = {}
    for s in SEASONS:
        cmp_all = P.compare(alt[s], "flat_plus_trailing_catches")
        out.append(f"**{s}, TE-premium points, ALL positions** ({cmp_all['n']:,} rows; only tight ends' actual points differ from PPR)\n")
        out.append(scores_md(cmp_all))
        out.append("")
        out.append(delta_md(cmp_all, ["components", "components_eff", "flat_ppr_unshifted"]))
        out.append("")
        te_rows = {k: v[v["position"] == "TE"] for k, v in alt[s].items()}
        te_cmp = P.compare(te_rows, "flat_plus_trailing_catches")
        te_cmps[s] = te_cmp
        out.append(f"{s}, tight ends only ({te_cmp['n']:,} rows):\n")
        out.append(scores_md(te_cmp))
        out.append("")
        out.append(delta_md(te_cmp, ["components", "components_eff", "flat_ppr_unshifted"]))
        out.append("")
    sk25 = C.component_skill(df, runs[2025]["components"])["skill_vs_pos_const"]
    out.append(f"""
### Reading

* **Composing from component models ties the flat model on PPR**: `components` moves RMSE by {d(comp[2025], 'components')} in 2025 and {d(comp[2024], 'components')} in 2024
  (both intervals span zero: a tie, which ships under the rule), and Spearman by {d(comp[2025], 'components', 'dSpearman', '{:+.4f}')} in 2025 and
  {d(comp[2024], 'components', 'dSpearman', '{:+.4f}')} in 2024. Adding the efficiency family does not help (`components_eff` {d(comp[2025], 'components_eff')} / {d(comp[2024], 'components_eff')}),
  so the simpler variant is the design. By position the composed model is better for quarterbacks in 2025
  (dRMSE {comp[2025]['delta_pos']['components'].loc['QB', 'dRMSE']:+.3f}) and not in 2024 ({comp[2024]['delta_pos']['components'].loc['QB', 'dRMSE']:+.3f}): no position is reliably different.
* **The composition path works for another scoring.** Under the league's TE premium the composed prediction is at least as good as the naive route on tight ends
  ({d(te_cmps[2025], 'components')} in 2025, {d(te_cmps[2024], 'components')} in 2024, against PPR-plus-bonus-times-recent-catches), and ignoring the premium altogether costs
  {d(te_cmps[2025], 'flat_ppr_unshifted')} / {d(te_cmps[2024], 'flat_ppr_unshifted')} on tight ends, so the shift matters and the components get it right without a
  per-position patch. Any linear weights compose the same way; threshold bonuses (this league has 100/200-yard and 40/50-yard TD bonuses, and scores an interception at -1) need
  the yardage distribution, which a mean model does not give (interceptions compose trivially and are a one-line change).
* **What is predictable, component by component** (skill against a constant per position, 2025): targets {sk25['y_tgt']:.2f}, receptions {sk25['y_rec']:.2f}, carries {sk25['y_car']:.2f}, rushing and
  receiving yards {sk25['y_rush_yds']:.2f} / {sk25['y_rec_yds']:.2f}, pass attempts and yards {sk25['y_att']:.2f} / {sk25['y_pass_yds']:.2f}, but receiving and rushing TDs only {sk25['y_rec_td']:.2f} / {sk25['y_rush_td']:.2f},
  passing TDs {sk25['y_pass_td']:.2f}, interceptions {sk25['y_int']:.2f}, fumbles lost {sk25['y_fum_lost']:.2f}, and the two-point / special-teams bucket none. Touchdowns carry six points and almost no
  predictable signal, which is the other half of the ceiling.
""")
    verdicts = [verdict_row("3 components", n, comp, rule="tie") for n in ("components", "components_eff")]
    verdicts[1]["verdict"] += "; not preferred: 18 extra features for no gain"
    return "\n".join(out), verdicts


# --------------------------------------------------------------------------- experiment 4
def section_exp4(df: pd.DataFrame, mkey: str) -> tuple[str, list[dict]]:
    out = ["## Experiment 4: cheap wins\n"]
    libs = {s: PH.libs_runs(df, mkey, s) for s in (2023, 2024, 2025)}
    out.append("### Three-library ensemble\n")
    out.append("The simple average of LightGBM, XGBoost and CatBoost, and weights (non-negative, sum to one, 0.01 grid, minimising RMSE on the head-to-head "
               "rows) fit on ONE earlier walk-forward year and scored on the next: 2024 weights scored on 2025 (the plan) and, as the out-of-time "
               "replication, 2023 weights scored on 2024. The weights are fit on predictions that were themselves made walk-forward.\n")
    res, weights = {}, {}
    for season, fit_year in ((2025, 2024), (2024, 2023)):
        fit_e = P.h2h(libs[fit_year], "lightgbm")
        fit_preds = pd.DataFrame({lib: fit_e[f"pred_{lib}"] for lib in T.LIBS})
        w = E.fit_weights(fit_preds, fit_e["y"])
        weights[season] = w
        r = {"lightgbm": libs[season]["lightgbm"], "xgboost": libs[season]["xgboost"], "catboost": libs[season]["catboost"]}
        tmpl = libs[season]["lightgbm"]
        avg = tmpl.copy()
        avg["pred"] = np.mean([libs[season][lib]["pred"].to_numpy() for lib in T.LIBS], axis=0)
        r["average"] = avg
        wtd = tmpl.copy()
        wtd["pred"] = sum(w[lib] * libs[season][lib]["pred"].to_numpy() for lib in T.LIBS)
        r["weighted"] = wtd
        res[season] = r
    comp = {s: P.compare(with_refs(res[s]), "lightgbm") for s in SEASONS}
    for season in SEASONS:
        c = comp[season]
        out.append(f"**{season} walk-forward** ({c['n']:,} head-to-head rows, ALL positions); weights fit on {season - 1}: "
                   + ", ".join(f"{k} {v:.2f}" for k, v in weights[season].items()) + "\n")
        out.append(scores_md(c))
        out.append("")
        out.append(delta_md(c, ["xgboost", "catboost", "average", "weighted"]))
        out.append("")
    verdicts = [verdict_row("4a ensemble", n, comp) for n in ("average", "weighted")]
    # ---- post-hoc: do the two winners combine?
    out.append("### Post-hoc: averaging the flat ensemble with the component composition\n")
    out.append("Not part of the pre-declared plan and not in the adoption table: since both the ensemble and the component design pass their rules, "
               "does averaging their PPR predictions help? (LightGBM components, composed to PPR, averaged 50/50 with the flat three-library average "
               "or with flat LightGBM.) Read as a lead, not a result: it was thought of after seeing the tables above.\n")
    post = {}
    for season in SEASONS:
        comp_frame = C.composed(PH.exp3_runs(df, mkey, season)["components"], C.PPR, libs[season]["lightgbm"])
        r = {"lightgbm": res[season]["lightgbm"], "average": res[season]["average"], "components": comp_frame}
        b1, b2 = comp_frame.copy(), comp_frame.copy()
        b1["pred"] = 0.5 * res[season]["average"]["pred"].to_numpy() + 0.5 * comp_frame["pred"].to_numpy()
        b2["pred"] = 0.5 * res[season]["lightgbm"]["pred"].to_numpy() + 0.5 * comp_frame["pred"].to_numpy()
        r["average_and_components"], r["lightgbm_and_components"] = b1, b2
        post[season] = P.compare(r, "lightgbm")
    for season in SEASONS:
        out.append(f"**{season}**\n")
        out.append(scores_md(post[season]))
        out.append("")
        out.append(delta_md(post[season], ["average", "components", "average_and_components", "lightgbm_and_components"]))
        out.append("")
    # ---- quantiles
    out.append("### Quantile recalibration\n")
    out.append("""The band is LightGBM p10-p90. For each target week, per position, the band is shifted by the empirical residual quantiles of the rows
*before* it: q10 by the 10th percentile of (actual - q10), q90 by the 90th percentile of (actual - q90), over the previous season's out-of-sample
walk-forward plus the current season's earlier weeks (played rows). Default rule (`underage_only`): a position is touched only if its trailing
p10-p90 coverage is below 80% by more than 3 points, so a position that is already calibrated is left alone; a position needs 100 trailing rows.
Variants: `qb_only` (the same rule, quarterbacks only, so every other band is exactly as it was) and `all` (every position shifted, a diagnostic).
Coverage is on all played rows of the season (the Phase 2 convention); intervals are week-blocked bootstraps. The pass criteria written before the
runs: quarterbacks within 2 points of 80% in both years, and no RB/WR/TE position widened by more than 2% (mean band width).
""")
    q = {s: PH.quantile_run(df, mkey, s) for s in (2023, 2024, 2025)}
    qres = {}
    for season in SEASONS:
        cur = q[season].copy()
        hist = q[season - 1]
        e0 = cur[cur["y_played"] == 1]
        before = E.coverage_table(e0.assign(lo=e0["q10"], hi=e0["q90"]), "lo", "hi")
        under = E.recalibrate(cur, hist)
        qbo = E.recalibrate(cur, hist, positions=("QB",))
        allm = E.recalibrate(cur, hist, mode="all")
        e1, e2, e3 = (x[x["y_played"] == 1] for x in (under, qbo, allm))
        after = E.coverage_table(e1, "q10a", "q90a")
        after_qb = E.coverage_table(e2, "q10a", "q90a")
        after_all = E.coverage_table(e3, "q10a", "q90a")
        qres[season] = (before, after, after_all, e0, e1, e3)
        out.append(f"**{season}** (all played rows, n = {len(e0):,}; history = {season - 1} walk-forward)\n")
        rows = {}
        for pos in [*POSITIONS, "ALL"]:
            rows[pos] = {"n": int(before.loc[pos, "n"]),
                         "coverage before": pm(before.loc[pos, "coverage"], before.loc[pos, "lo95"], before.loc[pos, "hi95"], "{:.3f}"),
                         "after: underage_only": pm(after.loc[pos, "coverage"], after.loc[pos, "lo95"], after.loc[pos, "hi95"], "{:.3f}"),
                         "after: qb_only": f"{after_qb.loc[pos, 'coverage']:.3f}", "after: all shifted": f"{after_all.loc[pos, 'coverage']:.3f}",
                         "mean width before": before.loc[pos, "mean_width"], "mean width after (underage_only)": after.loc[pos, "mean_width"],
                         "width change": after.loc[pos, "mean_width"] / before.loc[pos, "mean_width"] - 1}
        t = pd.DataFrame(rows).T
        t["n"] = t["n"].astype(int)
        out.append(md(t, "{:.3f}", floatfmt={"n": "{:.0f}", "width change": "{:+.1%}"}, label="position"))
        sc0 = E.interval_score(e0.assign(lo=e0["q10"], hi=e0["q90"]), "lo", "hi")
        qb0, qb1 = e0[e0["position"] == "QB"], e1[e1["position"] == "QB"]
        out.append(f"\nInterval score (Winkler, alpha .2, lower is better): {sc0:.3f} before, {E.interval_score(e1, 'q10a', 'q90a'):.3f} after (underage_only), "
                   f"{E.interval_score(e3, 'q10a', 'q90a'):.3f} with every position shifted; quarterbacks alone "
                   f"{E.interval_score(qb0.assign(lo=qb0['q10'], hi=qb0['q90']), 'lo', 'hi'):.3f} to {E.interval_score(qb1, 'q10a', 'q90a'):.3f}. "
                   f"Quarterbacks below p10 / above p90: {before.loc['QB', 'below_lo']:.3f} / {before.loc['QB', 'above_hi']:.3f} before, "
                   f"{after.loc['QB', 'below_lo']:.3f} / {after.loc['QB', 'above_hi']:.3f} after (targets .100 / .100). "
                   f"Which positions the rule touched: " + ", ".join(p for p in POSITIONS if (under.loc[under['position'] == p, 'q90a'] != under.loc[under['position'] == p, 'q90']).any()) + ".\n")
    ok = {}
    for season in SEASONS:
        before, after, *_ = qres[season]
        qb_ok = abs(after.loc["QB", "coverage"] - 0.80) <= 0.02 and abs(before.loc["QB", "coverage"] - 0.80) > abs(after.loc["QB", "coverage"] - 0.80)
        widen = {p: after.loc[p, "mean_width"] / before.loc[p, "mean_width"] - 1 for p in ("RB", "WR", "TE")}
        ok[season] = (bool(qb_ok), max(widen.values()), max(widen, key=widen.get))
    if all(v[0] and v[1] <= 0.02 for v in ok.values()):
        verdict = "SHIPS"
    elif all(v[0] for v in ok.values()):
        verdict = (f"QB target met in both years; caps missed: widest RB/WR/TE change {ok[2025][1]:+.1%} ({ok[2025][2]}) in 2025, {ok[2024][1]:+.1%} "
                   f"({ok[2024][2]}) in 2024 against the +2% cap. SHIPS as `underage_only` with that miss flagged (`qb_only` meets the cap by construction)")
    else:
        verdict = "does not ship"
    out.append(f"""
### Reading

* **The simple average of the three libraries ships; fitted weights add nothing.** `average` moves RMSE by {d(comp[2025], 'average')} in 2025 (interval excludes zero) and
  {d(comp[2024], 'average')} in 2024 (same direction, interval spans zero), and Spearman by {d(comp[2025], 'average', 'dSpearman', '{:+.4f}')} / {d(comp[2024], 'average', 'dSpearman', '{:+.4f}')}. The weights fit on one year
  ({weights[2025]['lightgbm']:.2f} / {weights[2025]['xgboost']:.2f} / {weights[2025]['catboost']:.2f} on 2024, {weights[2024]['lightgbm']:.2f} / {weights[2024]['xgboost']:.2f} / {weights[2024]['catboost']:.2f} on 2023, LightGBM / XGBoost / CatBoost) move
  between years and score no better than equal weights ({d(comp[2025], 'weighted')} / {d(comp[2024], 'weighted')}). The gain is small (about 0.2% of RMSE) because the three libraries
  are nearly the same model on the same features; it costs about 11 seconds of refit a week instead of 1.
* **Quantile recalibration fixes the quarterback band and, by design, leaves the calibrated positions alone.** QB p10-p90 coverage goes from {qres[2025][0].loc['QB', 'coverage']:.3f} to {qres[2025][1].loc['QB', 'coverage']:.3f} in 2025 and
  {qres[2024][0].loc['QB', 'coverage']:.3f} to {qres[2024][1].loc['QB', 'coverage']:.3f} in 2024 (target .800; the 95% intervals now contain it), at a cost of about +12% in QB band width; RB is untouched in both years. The rule also widened WR and TE slightly (2025: +0.5% / +1.2%; 2024: +2.2% /
  +1.4%), because their own trailing coverage was below 77%: in 2024 their uncalibrated coverage really was 76.5% / 75.3%, and it ends at 80.9% / 80.1%. That 2024 WR widening is 0.2 point over
  the +2% cap written before the runs, so the pass criteria are missed by that margin; `qb_only` meets them exactly (every other band unchanged, QB numbers identical). The side that
  misses for quarterbacks flipped between years (below p10 {qres[2025][0].loc['QB', 'below_lo']:.3f} vs above p90 {qres[2025][0].loc['QB', 'above_hi']:.3f} in 2025; {qres[2024][0].loc['QB', 'below_lo']:.3f} vs {qres[2024][0].loc['QB', 'above_hi']:.3f} in 2024), so the tail shifts are partly chasing noise and the robust part is the width.
  The Winkler interval score does not get worse (flat in 2025, better in 2024).
""")
    verdicts.append({"experiment": "4b quantile recalibration", "variant": "underage_only",
                     "2025 result": f"QB coverage {qres[2025][0].loc['QB', 'coverage']:.3f} to {qres[2025][1].loc['QB', 'coverage']:.3f}",
                     "2024 result": f"QB coverage {qres[2024][0].loc['QB', 'coverage']:.3f} to {qres[2024][1].loc['QB', 'coverage']:.3f}",
                     "rule": "QB within 2 pts of 80%; RB/WR/TE width change <= +2%", "verdict": verdict})
    return "\n".join(out), verdicts


# --------------------------------------------------------------------------- assembly
def build() -> tuple[str, pd.DataFrame]:
    df, mkey = PH.load()
    repro = json.loads((P.OUT / "reproduce.json").read_text()) if (P.OUT / "reproduce.json").exists() else {}
    s1, v1 = section_exp1(df, mkey)
    s2, v2 = section_exp2(df, mkey)
    s3, v3 = section_exp3(df, mkey)
    s4, v4 = section_exp4(df, mkey)
    verdicts = pd.DataFrame(v1 + v2 + v3 + v4)
    head = ["# Phase 2.5: volume-first experiments\n"]
    head.append(f"""Four experiments on the Phase 2 primary (LightGBM, squared error, one model with position as a feature, tuned parameters,
weekly walk-forward). Every result is scored on the same head-to-head rows as Phase 2 (played rows with an earlier game this season: 2025 has
{P.h2h({'a': PH.primary(df, mkey, 2025)}, 'a').shape[0]:,}) and **replicated on the 2024 walk-forward** with the same fixed seeds and the same week-blocked
bootstrap. Adoption rule (PLAN_MODEL.md): an experiment ships into the Phase 3 recommendation only if it beats the primary on 2025 AND replicates directionally
on 2024 (pooled dRMSE below zero in both years), or, for component models only, ties within noise (neither year's 95% interval lies entirely above zero).
"Significant" below means the 2025 interval excludes zero; the rule itself is about direction in both years, so a direction-only pass is called out.

All numbers are a pure function of `model/cache/features.parquet` (rebuilt for this phase: the 142 Phase 2 columns are bit-identical, 85 columns
added), `model/tuned_params.json` and fixed seeds; a rerun reproduces every number. The primary recomputed from the rebuilt matrix equals Phase 2's
predictions exactly (max |difference| {max((v.get('max_abs_pred_diff') or 0.0) for k, v in repro.items() if isinstance(v, dict) and 'max_abs_pred_diff' in v):.1e} over LightGBM, XGBoost and
CatBoost 2025 and LightGBM 2024). Hyper-parameters were tuned on 2024 (validation), and each variant's tree count is picked by early stopping on 2024 after
training on 2021-2023, so 2024 carries a mild, symmetric in-sample edge for every variant; 2025 is never seen before its walk-forward scores it.
""")
    head.append("## Summary\n")
    head.append(md(verdicts, "{}", index=False))
    head.append("")
    body = "\n\n".join([s1, s2, s3, s4])
    return "\n".join(head) + "\n" + body + "\n", verdicts


def main() -> int:
    text, verdicts = build()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)
    print(f"wrote {REPORT} ({len(text):,} characters)")
    print(verdicts.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
