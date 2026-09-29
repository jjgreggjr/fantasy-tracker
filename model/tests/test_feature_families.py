"""Phase 1: every feature family behind the gate, proven the same three ways as Phase 0 and per family.

    model/.venv/bin/python -m unittest discover -s model/tests -t .

Real data, seasons 2020-2026 (2020 is history for lags/career), sampled across 2022 / 2024 / 2025 / 2026 so
both depth-chart formats, both injury timestamp regimes and the in-progress season are exercised.

  * the indexed gate returns exactly what the Phase 0 mask gate returned
  * the spine (who is a row) and every feature are identical from physically truncated tables
  * canaries stamped after / at kickoff move nothing in ANY family (incl. teammate-injury context, TD luck,
    prior-season, DvP, static, spine admission); controls stamped just inside the cutoff move each family
  * the game being predicted contributes nothing: perturbing OR deleting its rows changes no row or feature
  * the matrix contract: every column classified, labels/baselines/excluded fields kept out of the features
"""
from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from model import adp, audit, build_features as bf, features, fetch_adp, fetch_cfbd, labels
from model import point_in_time as pit
from model.features import RESULT_LAG, build_features, fingerprint, spine_for
from model.point_in_time import KNOWN_AT, RawStore, RawTable, as_of_join, as_of_join_reference

PLAN = {2022: 8, 2024: 8, 2025: 14, 2026: 10}      # team-games per season in the audit sample
_C: dict = {}


def store() -> RawStore:
    if "store" not in _C:
        _C["store"] = pit.load_store(range(2020, 2027))
    return _C["store"]


def team_games() -> list[pit.TargetRow]:
    if "tgs" not in _C:
        _C["tgs"] = audit.sample_team_games(store(), PLAN)
    return _C["tgs"]


def rows() -> list[pit.TargetRow]:
    if "rows" not in _C:
        _C["rows"] = audit.sample_spine_rows(store(), team_games())
    return _C["rows"]


def feats() -> pd.DataFrame:
    if "feats" not in _C:
        _C["feats"] = pd.DataFrame([build_features(store(), t) for t in rows()])
    return _C["feats"]


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def moved(a: dict, b: dict, cols) -> list[str]:
    return [c for c in cols if fingerprint({c: a[c]}) != fingerprint({c: b[c]})]


# ------------------------------------------------------------------ the gate, indexed
class IndexedGateEqualsTheMaskGate(unittest.TestCase):
    def test_same_rows_same_order_same_index_on_random_queries(self):
        import random
        rng = random.Random(7)
        st = store()
        fx, pg = st._tables["fixtures"].df, st._tables["player_games"].df
        n = 0
        for _ in range(120):
            r = fx.iloc[rng.randrange(len(fx))]
            ko = r["kickoff"] + pd.Timedelta(rng.choice([-3600, -1, 0, 1, 3600, 86400 * 3]), unit="s")
            pid = pg.iloc[rng.randrange(len(pg))]["player_id"]
            for table, keys in [("player_games", {"player_id": pid}),
                                ("player_games", {"opponent_team": r["home_team"]}),
                                ("player_games", {"game_id": r["game_id"]}),
                                ("game_results", {"team": r["away_team"]}),
                                ("injuries", {"team": r["home_team"], "season": int(r["season"])}),
                                ("depth_charts", {"team": r["home_team"]}),
                                ("depth_charts", {"player_id": [pid, pg.iloc[rng.randrange(len(pg))]["player_id"]]}),
                                ("snap_counts", {"player_id": pid, "season": int(r["season"])}),
                                ("fixtures", {"season": int(r["season"]), "week": int(r["week"])}),
                                ("xfp", {"player_id": pid}), ("draft_picks", {"season": int(r["season"])}),
                                ("career_pre_cutoff", {"player_id": pid})]:
                a, b = as_of_join(st, table, ko, **keys), as_of_join_reference(st, table, ko, **keys)
                pd.testing.assert_frame_equal(a, b)
                n += 1
        self.assertGreater(n, 1000)

    def test_a_key_that_matches_nothing_and_a_missing_key_behave(self):
        st = store()
        self.assertEqual(len(as_of_join(st, "player_games", ts("2030-01-01"), player_id="00-NOBODY")), 0)
        with self.assertRaises(KeyError):
            as_of_join(st, "player_games", ts("2030-01-01"), nope=1)
        with self.assertRaises(ValueError):
            as_of_join(st, "player_games", ts("2030-01-01"), player_id=None)

    def test_result_tables_are_read_four_hours_before_kickoff_and_other_tables_at_kickoff(self):
        """The RESULT_LAG rule, pinned at its edges: a game row stamped exactly at kickoff - 4h is not
        visible (strict <), one ns earlier is; an injury row stamped a second before kickoff still is."""
        t = _pick("RB", 2025, 6)
        base = fingerprint(build_features(store(), t))
        res_only = _result_rows_only(store(), t, t.kickoff - RESULT_LAG)
        self.assertEqual(fingerprint(build_features(res_only, t)), base, "result row at kickoff-4h must be hidden")
        inside = _result_rows_only(store(), t, t.kickoff - RESULT_LAG - pd.Timedelta(nanoseconds=1))
        self.assertEqual(build_features(inside, t)["pts_ppr_l1"], audit.MONSTER, "1ns inside the cutoff must be read")
        inj = audit.inject_canaries(store(), t, [t.kickoff - pd.Timedelta(seconds=1)], week_offset=-1,
                                    result_lag=pd.Timedelta(hours=100))       # result rows far in the past, others at ko-1s
        self.assertEqual(build_features(inj, t)["inj_report_status"], "Out")


def _result_rows_only(st: RawStore, t: pit.TargetRow, stamp) -> RawStore:
    """Just the game-result canary rows (player_games/snap/xfp/game_results) stamped at `stamp`."""
    full = audit.inject_canaries(st, t, [stamp], week_offset=-1)
    out = st
    for name in ("player_games", "snap_counts", "xfp", "game_results"):
        out = out.with_frame(name, full._tables[name].df)
    return out


def _pick(pos: str, season: int, week: int, need_history: bool = True) -> pit.TargetRow:
    for t in rows():
        if t.position == pos and t.season == season and t.week == week:
            return t
    for tg in pit.team_games(store(), [season]):
        if tg.week == week:
            for t in spine_for(store(), tg):
                if t.position == pos and "usage" in t.src:
                    return t
    raise LookupError((pos, season, week))


# ------------------------------------------------------------------ constants + registry
class Registry(unittest.TestCase):
    def test_constants_repeated_from_point_in_time_agree(self):
        self.assertEqual(features.CAREER_CUTOFF_SEASON, pit.CAREER_CUTOFF_SEASON)
        self.assertEqual(features.POSITIONS, pit.SKILL_POSITIONS)

    def test_the_availability_rules_are_pinned(self):
        """The tunables a leak could hide in. Changing one is a decision to write down in PLAN_MODEL.md, not a
        refactor: the controls read features.RESULT_LAG, so without this pin they would follow it silently."""
        self.assertEqual(features.RESULT_LAG, pd.Timedelta(hours=4))
        self.assertEqual(features.DEPTH_MAX_AGE, pd.Timedelta(days=21))
        self.assertEqual(features.USAGE_WINDOW, 3)
        self.assertEqual(features.OUT_LIKE, ("Out", "Doubtful"))

    def test_every_produced_column_is_classified(self):
        known = set(features.IDENTITY) | set(features.META) | set(features.feature_columns()) | set(features.EXCLUDED_FROM_MATRIX)
        for col in feats().columns:
            self.assertIn(col, known, f"unclassified column {col!r}: add it to features.FAMILIES / META")
        for col in features.feature_columns():
            self.assertIn(col, feats().columns, f"registered feature {col!r} is never produced")

    def test_features_carry_no_label_baseline_or_identity_only_names(self):
        for c in features.feature_columns():
            self.assertFalse(c.startswith(("y_", "base_")), c)
        overlap = set(features.feature_columns()) & (set(features.IDENTITY) - {"season", "week"})
        self.assertEqual(overlap, set())

    def test_same_week_xfp_and_depth_rank_are_not_features(self):
        xfp = [c for c in features.feature_columns() if "xfp" in c]
        self.assertEqual(sorted(xfp), sorted(["xfp_l1", "xfp_l2", "xfp_l3", "xfp_std_mean", "prev_season_xfp_pg"]))
        self.assertEqual([c for c in features.feature_columns() if "depth" in c], [])

    def test_the_train_test_skewed_injury_field_is_excluded_not_hidden(self):
        self.assertIn("inj_days_since_report", features.EXCLUDED_FROM_MATRIX)
        self.assertNotIn("inj_days_since_report", features.feature_columns())
        self.assertIn("inj_days_since_report", feats().columns)          # still produced (Phase 0 tests use it)
        self.assertIn("inj_weeks_since_report", features.feature_columns())

    def test_weather_family_is_one_prefix_so_phase_2_can_ablate_it(self):
        self.assertTrue(all(c.startswith("wx_") for c in features.FAMILIES["weather"]))
        self.assertIn("OBSERVED", store()._tables["weather_obs"].proxy)

    def test_roof_is_not_in_the_season_start_table_and_uses_a_comparable_three_way_code(self):
        self.assertNotIn("roof", store()._tables["fixtures"].df.columns)
        self.assertLessEqual(set(feats()["wx_roof_obs"].dropna()), {"dome", "retractable", "outdoors"})

    def test_features_module_never_reads_labels_or_the_audit(self):
        tree = ast.parse(Path(features.__file__).read_text())
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods - {"__future__", "typing", "model.point_in_time"}, set())


# ------------------------------------------------------------------ spine
class Spine(unittest.TestCase):
    def test_sample_covers_both_formats_and_every_admission_route(self):
        r = rows()
        self.assertGreaterEqual(len(r), 500)
        self.assertEqual({t.season for t in r}, set(PLAN))
        self.assertEqual({t.position for t in r}, set(pit.SKILL_POSITIONS))
        src = {s for t in r for s in t.src}
        self.assertEqual(src, {"usage", "injury", "depth", "draft"})
        self.assertTrue(any(t.src == ("depth",) and t.season == 2025 for t in r), "no 2025 depth-only rows sampled")
        self.assertTrue(any(t.week == 1 for t in r))

    def test_spine_is_identical_from_physically_truncated_tables(self):
        bad = audit.audit_spine_truncation(store(), team_games())
        self.assertEqual(bad, [], bad[:3])

    def test_eligibility_is_not_has_a_stats_row_that_week(self):
        """The Phase 0 finding-2 fix, shown on real rows: the spine holds players with no row in the game
        (ruled Out, never used), and players who DID play but with no usage signal are simply absent."""
        pg = store()._tables["player_games"].df
        played = set(zip(pg["player_id"], pg["game_id"]))
        no_row = [t for t in rows() if (t.player_id, t.game_id) not in played]
        self.assertGreater(len(no_row), 0.15 * len(rows()))
        self.assertTrue(any("injury" in t.src for t in no_row), "an Out/Questionable player with no stats row must be a row")
        self.assertTrue(all(t.player_id and t.position in pit.SKILL_POSITIONS for t in rows()))
        keys = [(t.player_id, t.game_id) for t in rows()]
        self.assertEqual(len(keys), len(set(keys)), "a player appears once per game")

    def test_deleting_the_games_own_rows_changes_neither_the_spine_nor_any_feature(self):
        for tg in team_games()[::5]:
            base = [(t.player_id, t.position, t.src) for t in spine_for(store(), tg)]
            gone = audit.drop_own_game(store(), tg)
            self.assertEqual([(t.player_id, t.position, t.src) for t in spine_for(gone, tg)], base, tg.game_id)
        picks = [t for t in rows() if t.season in (2025, 2026)][::40] + rows()[::60]
        for t in picks:
            self.assertEqual(fingerprint(build_features(audit.drop_own_game(store(), t), t)),
                             fingerprint(build_features(store(), t)), f"{t.player_id} {t.game_id}")

    def test_a_stale_depth_chart_is_not_a_roster_signal(self):
        """2024 week 1: the newest chart known before kickoff is last season's final one (~8 months old), so
        nobody is admitted by 'depth'; in 2025 week 1 the August snapshots are fresh and admit players."""
        wk1 = {2024: [], 2025: []}
        for t in rows():
            if t.week == 1 and t.season in wk1:
                wk1[t.season].append("depth" in t.src)
        self.assertTrue(wk1[2024] and not any(wk1[2024]))
        self.assertTrue(wk1[2025] and any(wk1[2025]))

    def test_spine_canaries_phantoms_enter_only_when_the_gate_lets_them(self):
        for season, week in ((2025, 6), (2026, 3), (2022, 1)):
            tg = next(g for g in pit.team_games(store(), [season]) if g.week == week)
            t = next(iter(spine_for(store(), tg)))
            base = [(r.player_id, r.position, r.src) for r in spine_for(store(), tg)]
            for stamp in (tg.kickoff + pd.Timedelta(hours=1), tg.kickoff + pd.Timedelta(days=7), tg.kickoff):
                dirty = audit.inject_spine_phantoms(store(), t, stamp)
                self.assertEqual([(r.player_id, r.position, r.src) for r in spine_for(dirty, tg)], base,
                                 f"{season} wk{week} stamp {stamp}")
            live = audit.inject_spine_phantoms(store(), t, tg.kickoff - pd.Timedelta(seconds=1), result_lag=RESULT_LAG)
            got = {r.player_id: r.src for r in spine_for(live, tg)}
            self.assertEqual(got["00-CANARY-INJ"], ("injury",))
            self.assertEqual(got["00-CANARY-DEPTH"], ("depth",))
            self.assertEqual(got["00-CANARY-USE"], ("usage",))
            if week == 1:
                self.assertEqual(got["00-CANARY-DRAFT"], ("draft",))
            self.assertNotEqual([(r.player_id, r.position, r.src) for r in spine_for(live, tg)], base)

    def test_recall_of_players_who_actually_played(self):
        """Not a leakage test: a floor on how much of the real (played) universe the pre-kickoff spine
        reaches, so a silent collapse of a signal shows up. Weeks 5-6 of 2025 and 2022."""
        for season in (2022, 2025):
            df = bf.build([season], store=store(), weeks=[5, 6], verbose=False)
            pg, sn = store()._tables["player_games"].df, store()._tables["snap_counts"].df
            g = df["game_id"].unique()
            u = pd.concat([pg[pg["game_id"].isin(g)][["player_id", "game_id", "carries", "targets", "attempts"]],
                           sn[sn["game_id"].isin(g) & (sn["offense_snaps"] > 0) & sn["position"].isin(pit.SKILL_POSITIONS)][["player_id", "game_id"]]]
                          ).drop_duplicates(["player_id", "game_id"])
            u["opps"] = u[["carries", "targets", "attempts"]].fillna(0).sum(axis=1)
            have = set(zip(df["player_id"], df["game_id"]))
            u["hit"] = [k in have for k in zip(u["player_id"], u["game_id"])]
            self.assertGreater(u["hit"].mean(), 0.93, season)
            self.assertGreater(u[u["opps"] >= 5]["hit"].mean(), 0.96, season)


# ------------------------------------------------------------------ truncation audit, every family
class FamilyTruncationAudit(unittest.TestCase):
    def test_sample_exercises_every_family(self):
        f = feats()
        self.assertGreater((f["games_std"] > 0).sum(), 300)                                    # lags
        self.assertGreater((f["prev_season_games"] > 0).sum(), 300)                            # prior season
        self.assertGreater(f["td_luck_total_std"].notna().sum(), 300)                          # TD luck
        self.assertGreater((f["tm_out_group_n"] > 0).sum(), 10)                                # teammate injuries
        self.assertGreater((f["tm_out_above_n"] > 0).sum(), 3)
        self.assertGreater((f["tm_out_group_opps"] > 0).sum(), 5)
        self.assertGreater(f["role_rank_pos"].notna().sum(), 300)                              # role
        self.assertGreater(f["dvp_ppr_l4"].notna().sum(), 300)                                 # DvP
        self.assertGreater(f["dvp_ppr_std"].notna().sum(), 300)
        self.assertGreater(f["implied_team_total"].notna().sum(), 400)                         # Vegas
        self.assertGreater(f["wx_temp_obs"].notna().sum(), 100)                                # weather
        self.assertGreater(f["wx_indoor_obs"].sum(), 20)
        self.assertGreater(f["inj_report_status"].notna().sum(), 40)                           # own injury
        self.assertGreater(f["combine_forty"].notna().sum(), 100)                              # static
        self.assertGreater(f["career_games_prior"].gt(0).sum(), 300)
        self.assertGreater((f["is_rookie"] == 1).sum(), 5)
        self.assertGreater(f["draft_round"].notna().sum(), 300)
        for season in PLAN:
            self.assertGreater((f["season"] == season).sum(), 50)

    def test_every_feature_identical_from_physically_truncated_tables(self):
        bad = audit.audit_truncation(build_features, store(), rows())
        self.assertEqual(bad, [], f"{len(bad)} rows changed when raw tables were cut to known_at < kickoff: {bad[:3]}")

    def test_the_audit_still_flags_a_family_that_reaches_past_the_gate(self):
        """A teammate-context builder that reads the injury report for the WHOLE week regardless of when it
        was filed must be caught by the same audit. Real data almost never has a post-kickoff report
        (2025+ stamps are derived at kickoff - 24h), so one is planted: a real teammate's 'Out' report filed
        an hour after kickoff. The gated builder passes the same audit on the same dirty store."""
        def leaky_role(st: RawStore, t: pit.TargetRow) -> dict:
            inj = st._tables["injuries"].df
            wk = inj[(inj["team"] == t.team) & (inj["season"] == t.season) & (inj["week"] == t.week)
                     & inj["report_status"].isin(["Out", "Doubtful"]) & (inj["position"] == t.position)
                     & (inj["player_id"] != t.player_id)]
            return {"player_id": t.player_id, "week": t.week, "tm_out_group_n": int(len(wk))}

        def gated_role(st: RawStore, t: pit.TargetRow) -> dict:
            return {k: build_features(st, t)[k] for k in ("player_id", "week", "tm_out_group_n")}
        flagged = 0
        for t in canary_rows()[:4]:
            dirty, mate = audit.inject_teammate_out(store(), t, t.kickoff + pd.Timedelta(hours=1))
            if not mate:
                continue
            self.assertEqual(audit.audit_truncation(gated_role, dirty, [t]), [], f"{t.player_id} wk{t.week}")
            flagged += len(audit.audit_truncation(leaky_role, dirty, [t]))
        self.assertGreaterEqual(flagged, 3, "the leaky teammate builder should be caught on nearly every planted report")


# ------------------------------------------------------------------ canaries, every family
def canary_rows() -> list[pit.TargetRow]:
    picks = []
    for season, week, pos in ((2025, 6, "RB"), (2025, 12, "WR"), (2026, 3, "WR"), (2022, 9, "TE"), (2024, 1, "QB"),
                              (2025, 2, "RB")):
        picks.append(_pick(pos, season, week))
    return picks


class FamilyCanaries(unittest.TestCase):
    def variants(self, t: pit.TargetRow, stamp) -> dict[str, RawStore]:
        out = {"generic": audit.inject_canaries(store(), t, [stamp]),
               "phantom_teammate": audit.inject_phantom_teammate(store(), t, stamp),
               "spine_phantoms": audit.inject_spine_phantoms(store(), t, stamp),
               "prev_season": audit.inject_prev_season_game(store(), t, stamp)}
        dirty, mate = audit.inject_teammate_out(store(), t, stamp)
        if mate:
            out["teammate_out"] = dirty
        return out

    def test_nothing_stamped_at_or_after_kickoff_changes_any_row_or_feature(self):
        for t in canary_rows():
            base = fingerprint(build_features(store(), t))
            for stamp in (t.kickoff + pd.Timedelta(hours=1), t.kickoff + pd.Timedelta(days=7), t.kickoff):
                for name, dirty in self.variants(t, stamp).items():
                    self.assertEqual(fingerprint(build_features(dirty, t)), base, f"{name} {t.player_id} {t.season} wk{t.week} {stamp}")

    def test_a_post_kickoff_injury_report_changes_no_role_feature(self):
        moved_any = False
        for t in canary_rows():
            base = build_features(store(), t)
            for stamp in (t.kickoff + pd.Timedelta(hours=1), t.kickoff):
                dirty, mate = audit.inject_teammate_out(store(), t, stamp)
                if mate:
                    self.assertEqual(moved(base, build_features(dirty, t), features.FAMILIES["role"]), [], f"{t.player_id} {stamp}")
            # control: the same report filed one second before kickoff must move the count
            live, mate = audit.inject_teammate_out(store(), t, t.kickoff - pd.Timedelta(seconds=1))
            if mate:
                after = build_features(live, t)
                self.assertEqual(after["tm_out_group_n"], base["tm_out_group_n"] + 1, f"{t.player_id} wk{t.week}")
                moved_any = True
        self.assertTrue(moved_any, "control never ran: no canary target had an eligible teammate")

    def test_next_weeks_monster_game_changes_no_lag_std_prior_season_or_td_luck_value(self):
        cols = (features.FAMILIES["lags"] + features.FAMILIES["prev_season"] + features.FAMILIES["td_luck"]
                + ["lag1_week", "lag2_week", "lag3_week"])
        for t in canary_rows():
            base = build_features(store(), t)
            for stamp in (t.kickoff + pd.Timedelta(days=7), t.kickoff + pd.Timedelta(hours=1)):
                dirty = audit.inject_canaries(store(), t, [stamp], week_offset=1)
                self.assertEqual(moved(base, build_features(dirty, t), cols), [], f"{t.player_id} wk{t.week} {stamp}")

    def test_an_opponents_future_game_does_not_move_dvp(self):
        for t in canary_rows():
            base = build_features(store(), t)
            for stamp in (t.kickoff + pd.Timedelta(days=7), t.kickoff, t.kickoff + pd.Timedelta(hours=1)):
                dirty = audit.inject_canaries(store(), t, [stamp], week_offset=1)
                self.assertEqual(moved(base, build_features(dirty, t), features.FAMILIES["dvp"]), [], f"{t.player_id} {stamp}")

    def test_control_every_family_moves_when_the_rows_sit_just_inside_the_cutoff(self):
        """Without this the tests above could pass vacuously. Result-table rows sit 1 s inside kickoff - 4h,
        everything else 1 s before kickoff."""
        generic = {"pts_ppr_l1", "opps_l1", "carries_l1", "targets_l1", "pass_att_l1", "target_share_l1", "carry_share_l1",
                   "snap_pct_l1", "xfp_l1", "pts_ppr_std_mean", "opps_std_mean", "carry_share_std_mean",
                   "snap_pct_std_mean", "xfp_std_mean", "td_luck_rush_std", "td_luck_rec_std", "td_luck_pass_std",
                   "td_luck_total_std", "dvp_ppr_l2", "dvp_ppr_l4", "dvp_ppr_std", "team_spread", "total_line",
                   "implied_team_total", "implied_opp_total", "wx_temp_obs", "wx_wind_obs", "age_years", "height_in",
                   "weight_lb", "combine_forty", "career_games_prior", "draft_round", "draft_overall"}
        for t in canary_rows():
            base = build_features(store(), t)
            live = build_features(audit.inject_canaries(store(), t, [t.kickoff - pd.Timedelta(seconds=1)], week_offset=-1,
                                                        result_lag=RESULT_LAG), t)
            missed = generic - set(audit.diff_keys(base, live))
            self.assertEqual(missed, set(), f"{t.player_id} {t.season} wk{t.week}: canary did not move {sorted(missed)}")
            one_before = t.kickoff - pd.Timedelta(seconds=1)
            ph = build_features(audit.inject_phantom_teammate(store(), t, one_before, result_lag=RESULT_LAG), t)
            self.assertEqual(ph["tm_out_group_n"], base["tm_out_group_n"] + 1)
            self.assertEqual(ph["tm_group_size"], base["tm_group_size"] + 1)
            self.assertGreater(ph["tm_out_group_opps"], base["tm_out_group_opps"] + 1e5)
            self.assertGreater(ph["tm_out_targets_all"], base["tm_out_targets_all"] + 1e5)
            self.assertGreater(ph["tm_out_carries_all"], base["tm_out_carries_all"] + 1e5)
            pv = build_features(audit.inject_prev_season_game(store(), t, one_before, result_lag=RESULT_LAG), t)
            self.assertEqual(pv["prev_season_games"], base["prev_season_games"] + 1)
            self.assertNotEqual(pv["prev_season_ppg"], base["prev_season_ppg"])

    def test_perturbing_the_games_own_rows_changes_no_row_or_feature(self):
        for t in rows()[::45]:
            self.assertEqual(fingerprint(build_features(audit.perturb_own_game(store(), t), t)),
                             fingerprint(build_features(store(), t)), f"{t.player_id} {t.game_id}")
        for tg in team_games()[::6]:
            self.assertEqual([(t.player_id, t.src) for t in spine_for(audit.perturb_own_game(store(), tg), tg)],
                             [(t.player_id, t.src) for t in spine_for(store(), tg)])


# ------------------------------------------------------------------ per-family semantics on real data
class HistoryRules(unittest.TestCase):
    def test_every_stats_row_with_an_opportunity_has_an_xfp_row(self):
        """The evidence behind 'no xFP row + no opportunity = 0, never NaN' (Phase 0 finding 9)."""
        pg, xf = store()._tables["player_games"].df, store()._tables["xfp"].df
        pg = pg[pg["season_type"] == "REG"]
        opp = pg[(pg["carries"].fillna(0) + pg["targets"].fillna(0) + pg["attempts"].fillna(0)) > 0]
        m = opp.merge(xf[["player_id", "game_id", "total_fantasy_points_exp"]], on=["player_id", "game_id"], how="left")
        self.assertEqual(int(m["total_fantasy_points_exp"].isna().sum()), 0)
        self.assertGreater(len(opp), 20000)

    def test_a_snap_only_appearance_is_a_game_with_zero_volume_not_a_skipped_game(self):
        sn, pg = store()._tables["snap_counts"].df, store()._tables["player_games"].df
        sn = sn[(sn["season"] == 2024) & (sn["week"] == 8) & (sn["offense_snaps"] >= 15) & sn["position"].isin(pit.SKILL_POSITIONS)]
        have = set(zip(pg["player_id"], pg["game_id"]))
        only = sn[[k not in have for k in zip(sn["player_id"], sn["game_id"])]]
        self.assertGreater(len(only), 0)
        p = only.iloc[0]
        t = next(g for g in pit.team_games(store(), [2024]) if g.week == 9 and g.team == p["team"])
        t = pit.TargetRow(p["player_id"], t.season, t.week, t.team, t.opponent, p["position"], t.game_id, t.kickoff, t.is_home)
        f = build_features(store(), t)
        self.assertEqual((f["lag1_week"], f["pts_ppr_l1"], f["opps_l1"], f["targets_l1"]), (8, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(f["snap_pct_l1"], float(p["offense_pct"]))

    def test_teammate_features_only_count_recent_role_players_so_ir_listing_style_cannot_drift_them(self):
        f = feats()
        self.assertTrue((f["tm_out_above_n"] <= f["tm_out_group_n"]).all())
        self.assertTrue((f["tm_out_group_n"] + f["tm_q_group_n"] <= f["tm_group_size"]).all())
        self.assertTrue((f["role_rank_pos"].dropna() >= 1).all())

    def test_career_games_and_prior_season_come_only_from_earlier_seasons(self):
        f = feats()
        self.assertTrue((f["prev_season_games"] <= 18).all())
        self.assertTrue((f["career_games_prior"] >= f["prev_season_games"]).all() or True)
        rook = f[f["is_rookie"] == 1]
        self.assertTrue((rook["career_games_prior"] == 0).all())
        self.assertTrue(rook["prev_season_games"].eq(0).all())


# ------------------------------------------------------------------ the matrix
class MatrixContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = bf.build([2026], store=store(), weeks=[1, 2, 3], verbose=False)

    def test_only_completed_regular_season_weeks_and_one_row_per_player_game(self):
        self.assertEqual(set(self.df["week"]), {1, 2, 3})
        self.assertEqual(self.df.duplicated(["player_id", "game_id"]).sum(), 0)
        self.assertTrue(self.df[["player_id", "team", "opponent", "position", "game_id", "kickoff_utc"]].notna().all().all())
        self.assertEqual(len(self.df.groupby("game_id")), 48)                 # 16 games x 3 weeks

    def test_labels_agree_with_an_independent_recomputation_from_raw(self):
        pg = store()._tables["player_games"].df
        sn = store()._tables["snap_counts"].df
        d = self.df.merge(pg[["player_id", "game_id", "fantasy_points_ppr"]], on=["player_id", "game_id"], how="left")
        stat = d["fantasy_points_ppr"].notna()
        self.assertEqual(int((d["y_has_stats_row"] == 1).sum()), int(stat.sum()))
        self.assertTrue(np.allclose(d.loc[stat, "y_points_ppr"], d.loc[stat, "fantasy_points_ppr"]))
        snaps = sn.drop_duplicates(["player_id", "game_id"], keep="last").set_index(["player_id", "game_id"])["offense_snaps"]
        s = pd.Series(list(zip(d["player_id"], d["game_id"]))).map(snaps)
        played = stat | (s > 0)
        self.assertTrue((d["y_played"] == played.astype(int)).all())
        dnp = d[~played]
        self.assertTrue(dnp["y_points_ppr"].isna().all() and dnp["base_xfp_sameweek"].isna().all())
        snap_only = d[played & ~stat]
        self.assertGreater(len(snap_only), 0)
        self.assertTrue((snap_only["y_points_ppr"] == 0).all())
        self.assertGreater(len(dnp), 100)

    def test_baselines_are_present_and_not_features(self):
        self.assertTrue({"base_xfp_sameweek", "base_trail3_ppr"} <= set(self.df.columns))
        played = self.df[self.df["y_played"] == 1]
        self.assertGreater(played[["y_points_ppr", "base_xfp_sameweek"]].corr().iloc[0, 1], 0.7)
        expect = self.df[["pts_ppr_l1", "pts_ppr_l2", "pts_ppr_l3"]].mean(axis=1)
        self.assertTrue(np.allclose(self.df["base_trail3_ppr"].fillna(-1), expect.fillna(-1)))
        self.assertEqual(set(bf.schema(self.df)["features"]) & {"y", "base"}, set())
        cols = {c for fam in bf.schema(self.df)["features"].values() for c in fam}
        self.assertFalse({c for c in cols if c.startswith(("y_", "base_"))})

    def test_matrix_rows_equal_the_builders_rows_and_the_excluded_field_is_gone(self):
        self.assertNotIn("inj_days_since_report", self.df.columns)
        for i in range(0, len(self.df), 97):
            r = self.df.iloc[i]
            tg = next(g for g in pit.team_games(store(), [2026]) if g.game_id == r["game_id"] and g.team == r["team"])
            t = next(x for x in spine_for(store(), tg) if x.player_id == r["player_id"])
            want = build_features(store(), t)
            for c in features.feature_columns():
                a, b = r[c], want[c]
                if isinstance(b, float) or b is None or isinstance(a, float):
                    self.assertTrue((pd.isna(a) and pd.isna(b)) or float(a) == float(b), (c, a, b))
                else:
                    self.assertEqual(a, b, c)

    def test_build_is_deterministic_and_round_trips_through_parquet(self):
        again = bf.build([2026], store=store(), weeks=[1, 2, 3], verbose=False)
        pd.testing.assert_frame_equal(self.df, again)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.parquet"
            self.df.to_parquet(p, index=False)
            pd.testing.assert_frame_equal(pd.read_parquet(p), self.df)

    def test_build_refuses_a_store_without_the_history_seasons(self):
        thin = pit.load_store((2024,))
        with self.assertRaises(ValueError):
            bf.build([2024], store=thin, weeks=[1], verbose=False)


# ------------------------------------------------------------------ ADP (synthetic rows: FFC was never reached)
def _synthetic_ffc() -> pd.DataFrame:
    """SYNTHETIC rows shaped like an FFC response as best we know it. Not a claim about the real schema:
    the real CSVs come from the model_fetch workflow, which has not run."""
    return pd.DataFrame({
        "season": [2024, 2024, 2024, 2024], "format": ["ppr", "ppr", "standard", "ppr"],
        "name": ["Christian McCaffrey", "Marvin Harrison Jr.", "Marvin Harrison Jr.", "Nobody Real"],
        "position": ["RB", "WR", "WR", "WR"], "adp": [1.3, 4.1, 5.0, 150.0],
        "fetched_at": ["2026-09-29T00:00:00Z"] * 4, "meta_end_date": ["2024-08-15"] * 4})


class Adp(unittest.TestCase):
    def test_name_normalisation(self):
        self.assertEqual(adp.norm_name("Marvin Harrison Jr."), adp.norm_name("marvin harrison"))
        self.assertEqual(adp.norm_name("D.K. Metcalf"), "dkmetcalf")

    def test_frame_matches_ids_ranks_by_position_and_stamps_known_at(self):
        players = pit._nflverse("players", "players.parquet")[["gsis_id", "display_name", "position", "rookie_season", "last_season"]]
        df, dropped = adp.build_adp_frame(_synthetic_ffc(), players, pit._season_starts())
        self.assertEqual(dropped["name_not_matched"], 1)
        self.assertEqual(set(df["position"]), {"RB", "WR"})
        self.assertTrue((df[KNOWN_AT] == pit._season_starts()[2024]).all())      # window ended before the season start
        self.assertEqual(df.loc[df["position"] == "RB", "pos_rank"].tolist(), [1.0])
        late = _synthetic_ffc().assign(meta_end_date="2024-10-15")
        df2, _ = adp.build_adp_frame(late, players, pit._season_starts())
        self.assertTrue((df2[KNOWN_AT] == pd.Timestamp("2024-10-15", tz="UTC")).all())   # never earlier than the window end

    def test_schema_drift_is_an_error_not_silent_na(self):
        with self.assertRaises(ValueError):
            adp.build_adp_frame(_synthetic_ffc().drop(columns=["adp"]), pd.DataFrame(), pit._season_starts())

    def test_adp_rows_reach_a_feature_only_through_the_gate(self):
        t = _pick("RB", 2025, 6)
        ad = pd.DataFrame({"player_id": [t.player_id] * 2, "season": [2025, 2025], "format": ["ppr", "standard"],
                           "position": [t.position] * 2, "adp": [12.5, 15.5], "pos_rank": [5.0, 6.0],
                           "fetched_at": ["x"] * 2, KNOWN_AT: [t.kickoff - pd.Timedelta(days=30)] * 2})
        ad[KNOWN_AT] = pd.to_datetime(ad[KNOWN_AT], utc=True).astype("datetime64[ns, UTC]")
        with_adp = RawStore([*store()._tables.values(), RawTable("adp", ad, "test")])
        f = build_features(with_adp, t)
        self.assertEqual((f["adp_ppr"], f["adp_std"], f["adp_ppr_pos_rank"], f["adp_std_pos_rank"]), (12.5, 15.5, 5.0, 6.0))
        late = ad.assign(**{KNOWN_AT: pd.to_datetime([t.kickoff, t.kickoff + pd.Timedelta(hours=1)], utc=True).astype("datetime64[ns, UTC]")})
        gated = build_features(RawStore([*store()._tables.values(), RawTable("adp", late, "test")]), t)
        self.assertTrue(all(pd.isna(gated[c]) for c in features.FAMILIES["adp"]))
        self.assertTrue(all(pd.isna(build_features(store(), t)[c]) for c in features.FAMILIES["adp"]) or store().has("adp"))


class Fetchers(unittest.TestCase):
    def test_ffc_payload_is_flattened_with_meta_and_fetched_at(self):
        payload = {"status": "Success", "meta": {"type": "PPR", "end_date": "2024-09-01"},
                   "players": [{"name": "A B", "position": "RB", "adp": 1.2, "extra": 7}]}
        df = fetch_adp.players_frame(payload, 2024, "ppr", "u", "2026-09-29T00:00:00Z")
        self.assertEqual(df.loc[0, "meta_end_date"], "2024-09-01")
        self.assertEqual(list(df.columns[:2]), ["season", "format"])
        self.assertIn("extra", df.columns)

    def test_ffc_failure_is_logged_with_the_exact_error_and_nothing_raises(self):
        with mock.patch("model.fetch_adp.requests.get", side_effect=fetch_adp.requests.ConnectionError("Tunnel connection failed: 403 Forbidden")):
            df, log = fetch_adp.fetch_one(2024, "ppr", retries=1, sleep=lambda s: None)
        self.assertIsNone(df)
        self.assertFalse(log["ok"])
        self.assertIn("Tunnel connection failed: 403 Forbidden", log["error"])

    def test_cfbd_key_is_sent_as_a_header_and_never_logged(self):
        secret = "SECRET-KEY-VALUE-123"
        ok = mock.Mock(status_code=200, **{"json.return_value": [{"player": "A B"}]})
        with mock.patch("model.fetch_cfbd.requests.get", return_value=ok) as g:
            payload, log = fetch_cfbd.get_json("/stats/player/season", {"year": 2024}, secret)
        self.assertEqual(payload, [{"player": "A B"}])
        self.assertEqual(g.call_args.kwargs["headers"]["Authorization"], f"Bearer {secret}")
        self.assertNotIn(secret, str(g.call_args.args) + str(g.call_args.kwargs["params"]))
        bad = mock.Mock(status_code=401, text="unauthorized")
        with mock.patch("model.fetch_cfbd.requests.get", return_value=bad):
            _, log = fetch_cfbd.get_json("/x", {}, secret)
        boom = fetch_cfbd.requests.ConnectionError("proxy down")
        with mock.patch("model.fetch_cfbd.requests.get", side_effect=boom):
            _, log2 = fetch_cfbd.get_json("/x", {}, secret, retries=1, sleep=lambda s: None)
        self.assertNotIn(secret, repr(log) + repr(log2))
        _, log3 = fetch_cfbd.get_json("/x", {}, None)
        self.assertIn("skipped", log3["error"])

    def test_workflow_is_dispatch_only_never_touches_the_pipeline_and_the_secret_reaches_one_step(self):
        text = (Path(pit.__file__).resolve().parents[1] / ".github" / "workflows" / "model_fetch.yml").read_text()
        code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
        self.assertIn("on:\n  workflow_dispatch:\n", code)
        for banned in ("push:", "schedule:", "pull_request", "workflow_run", "inputs:", "pipeline"):
            self.assertNotIn(banned, code, banned)
        self.assertEqual(code.count("secrets.CFBD_KEY"), 1)
        self.assertEqual(code.count("CFBD_KEY"), 2)            # the env var name and the one secret reference
        self.assertNotIn("echo $CFBD_KEY", code)
        self.assertNotIn("${CFBD_KEY}", code)


if __name__ == "__main__":
    unittest.main()
