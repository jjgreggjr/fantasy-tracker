"""Leakage harness: the gate, the truncation audit, the future-row canaries, schedule sanity,
and proof that the audit catches a deliberately leaky join.

    model/.venv/bin/python -m unittest discover -s model/tests -t .

Runs against REAL 2024 data (nflverse + ffopportunity release assets). The first run downloads
about 10 MB into model/cache/ (git-ignored); after that it is offline and deterministic. A failed
download is an error, never a skip: a leakage test that quietly does not run is worse than none.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import pandas as pd

from model import audit, features
from model import point_in_time as pit
from model.features import build_features, fingerprint
from model.point_in_time import KNOWN_AT, RawTable, as_of_join, make_target

_STORE: pit.RawStore | None = None
_TARGETS: list[pit.TargetRow] | None = None


def store() -> pit.RawStore:
    global _STORE
    if _STORE is None:
        _STORE = pit.load_store((2024,))
    return _STORE


def targets() -> list[pit.TargetRow]:
    global _TARGETS
    if _TARGETS is None:
        _TARGETS = audit.sample_targets(store())
    return _TARGETS


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


# ------------------------------------------------------------------ the gate itself
class GateSemantics(unittest.TestCase):
    def setUp(self):
        k = ts("2024-11-10 18:00")
        self.k = k
        df = pd.DataFrame({
            "player_id": ["a", "a", "a", "b"],
            "v": [1, 2, 3, 4],
            KNOWN_AT: pd.to_datetime([k - pd.Timedelta(days=7), k - pd.Timedelta(nanoseconds=1), k,
                                      k + pd.Timedelta(hours=1)], utc=True).astype("datetime64[ns, UTC]"),
        })
        self.store = pit.RawStore([RawTable("t", df, "test")])

    def test_strictly_before_kickoff(self):
        got = as_of_join(self.store, "t", self.k, player_id="a")
        self.assertEqual(got["v"].tolist(), [1, 2])       # == kickoff excluded, 1ns before included

    def test_oldest_known_first_and_membership_keys(self):
        got = as_of_join(self.store, "t", self.k + pd.Timedelta(days=1), player_id=["a", "b"])
        self.assertEqual(got["v"].tolist(), [1, 2, 3, 4])

    def test_returns_a_copy(self):
        as_of_join(self.store, "t", self.k, player_id="a")["v"] = 99
        self.assertEqual(as_of_join(self.store, "t", self.k, player_id="a")["v"].tolist(), [1, 2])

    def test_refuses_naive_timestamps_unknown_columns_and_none_keys(self):
        with self.assertRaises(ValueError):
            as_of_join(self.store, "t", pd.Timestamp("2024-11-10 18:00"))
        with self.assertRaises(KeyError):
            as_of_join(self.store, "t", self.k, nope=1)
        with self.assertRaises(ValueError):
            as_of_join(self.store, "t", self.k, player_id=None)

    def test_a_table_without_known_at_cannot_exist(self):
        with self.assertRaises(ValueError):
            RawTable("bad", pd.DataFrame({"x": [1]}), "n/a")
        with self.assertRaises(ValueError):       # NaT = availability unknown = refuse, do not guess
            RawTable("bad", pd.DataFrame({KNOWN_AT: pd.to_datetime([None], utc=True).astype("datetime64[ns, UTC]")}), "n/a")
        with self.assertRaises(ValueError):       # naive / wrong dtype
            RawTable("bad", pd.DataFrame({KNOWN_AT: pd.to_datetime(["2024-01-01"])}), "n/a")


# ------------------------------------------------------------------ the loaders
class LoaderContracts(unittest.TestCase):
    def test_every_table_has_a_rule_and_nonnull_utc_known_at(self):
        d = store().describe()
        self.assertEqual(set(d["table"]), {"fixtures", "lines", "weather_obs", "game_results", "player_games",
                                           "snap_counts", "xfp", "injuries", "depth_charts",
                                           "players_static", "draft_picks", "combine",
                                           "career_pre_cutoff"})       # Phase 1: pre-2020 career games
        for name in store().names():
            t = store()._tables[name]
            self.assertTrue(t.known_at_rule, name)
            self.assertEqual(str(t.df[KNOWN_AT].dtype), "datetime64[ns, UTC]", name)
            self.assertFalse(t.df[KNOWN_AT].isna().any(), name)
            self.assertGreater(len(t.df), 0, name)

    def test_known_at_matches_the_documented_rule_for_every_derived_table(self):
        """The audit trusts known_at; this pins the derivations so a mis-stamped table cannot hide."""
        t = store()._tables
        ko = t["fixtures"].df.set_index("game_id")["kickoff"]
        for name in ("player_games", "snap_counts", "xfp", "game_results"):
            df = t[name].df
            self.assertTrue((df[KNOWN_AT] == df["game_id"].map(ko)).all(), name)          # game rows: kickoff
        ln = t["lines"].df
        self.assertTrue((ln[KNOWN_AT] == ln["game_id"].map(ko) - pit.LINES_CLOSE_LEAD).all())
        wx = t["weather_obs"].df
        self.assertTrue((wx[KNOWN_AT] == wx["game_id"].map(ko) - pit.WEATHER_PROXY_LEAD).all())
        fx = t["fixtures"].df
        reg = fx["game_type"] == "REG"
        self.assertTrue((fx.loc[reg, KNOWN_AT] == fx.loc[reg, "kickoff"].min() - pit.SEASON_START_LEAD).all())
        self.assertTrue((fx.loc[~reg, KNOWN_AT] == fx.loc[~reg, "kickoff"] - pit.NON_REG_FIXTURE_LEAD).all())
        starts = set(pit._season_starts())
        for name in ("players_static", "draft_picks", "combine"):      # season-static: always exactly a season start
            self.assertLessEqual(set(t[name].df[KNOWN_AT]), starts, name)
        dp = t["draft_picks"].df
        rookies_2024 = dp[dp["season"] == 2024]
        self.assertTrue((rookies_2024[KNOWN_AT] == ko.min() - pit.SEASON_START_LEAD).all())

    def test_2025_formats_load_with_derived_or_snapshot_timestamps(self):
        """nflverse changed two schemas after 2024: injuries lost date_modified, depth charts became
        timestamped ESPN-style snapshots. Both paths must produce a valid known_at."""
        inj = pit.load_injuries([2024, 2025]).df
        self.assertEqual(set(inj.loc[inj["season"] == 2024, "known_at_source"]), {"reported"})
        d25 = inj[inj["season"] == 2025]
        self.assertEqual(set(d25["known_at_source"]), {"derived_kickoff_minus_24h"})
        g = pit._games()
        tk = pd.concat([g.rename(columns={"home_team": "team"})[["season", "week", "team", "kickoff"]],
                        g.rename(columns={"away_team": "team"})[["season", "week", "team", "kickoff"]]])
        tk = tk.set_index(["season", "week", "team"])["kickoff"]
        wk5 = d25[d25["week"] == 5]
        self.assertGreater(len(wk5), 100)
        want = [tk[(2025, 5, tm)] - pit.INJURY_DERIVED_LEAD for tm in wk5["team"]]
        self.assertTrue((wk5[KNOWN_AT].tolist() == want))
        dc = pit.load_depth_charts([2024, 2025]).df
        self.assertEqual(set(dc.loc[dc["season"] == 2024, "known_at_source"]), {"derived_team_kickoff"})
        self.assertEqual(set(dc.loc[dc["season"] == 2025, "known_at_source"]), {"snapshot_dt"})
        self.assertTrue(set(dc["pos"].dropna()) <= set(pit.OFFENSE_DEPTH_POS))

    def test_the_weather_proxy_is_declared_not_hidden(self):
        self.assertIn("OBSERVED", store()._tables["weather_obs"].proxy)

    def test_static_tables_carry_no_career_outcomes_or_mutable_current_fields(self):
        leaky_cols = {"w_av", "car_av", "dr_av", "games", "probowls", "allpro", "seasons_started", "to",
                      "pass_yards", "rush_yards", "rec_yards", "status", "latest_team", "last_season",
                      "years_of_experience", "ngs_status", "pff_status", "draft_ovr"}
        for name in ("players_static", "draft_picks", "combine"):
            self.assertFalse(leaky_cols & set(store()._tables[name].df.columns), name)

    def test_kickoffs_are_utc_and_survive_the_november_clock_change(self):
        fx = store()._tables["fixtures"].df.set_index("game_id")["kickoff"]
        self.assertEqual(fx["2024_01_BAL_KC"], ts("2024-09-06 00:20"))      # Thu 20:20 EDT
        eight = fx[[g for g in fx.index if g.startswith("2024_08_")]]
        nine = fx[[g for g in fx.index if g.startswith("2024_09_")]]
        self.assertIn(ts("2024-10-27 17:00"), set(eight))                   # 1pm EDT
        self.assertIn(ts("2024-11-03 18:00"), set(nine))                    # 1pm EST, after the change
        self.assertTrue((fx == fx.dt.tz_convert("UTC")).all())

    def test_real_late_injury_row_exists_and_is_gated(self):
        """A real 2024 row: Van Noy's week-1 report was modified after the Thursday opener kicked
        off. The gate must keep it out of that game's features (and only that one row is like it)."""
        inj = store()._tables["injuries"].df
        fx = store()._tables["fixtures"].df
        k = fx.set_index("game_id").loc["2024_01_BAL_KC", "kickoff"]
        late = inj[(inj["team"] == "BAL") & (inj["week"] == 1) & (inj[KNOWN_AT] >= k)]
        self.assertEqual(len(late), 1)
        got = as_of_join(store(), "injuries", k, player_id=late.iloc[0]["player_id"])
        self.assertTrue((got[KNOWN_AT] < k).all())


# ------------------------------------------------------------------ (a) truncation audit
class TruncationAudit(unittest.TestCase):
    def test_sample_is_broad(self):
        tg = targets()
        self.assertGreaterEqual(len(tg), 250)
        self.assertEqual({t.week for t in tg}, set(range(1, 23)))
        self.assertEqual({t.position for t in tg}, set(pit.SKILL_POSITIONS))
        played = set(zip(store()._tables["player_games"].df["player_id"], store()._tables["player_games"].df["week"]))
        self.assertTrue(any((t.player_id, t.week) not in played for t in tg),
                        "sample should include rows with no stats row (listed/Out players)")

    def test_gated_features_identical_from_physically_truncated_tables(self):
        bad = audit.audit_truncation(build_features, store(), targets())
        self.assertEqual(bad, [], f"{len(bad)} rows changed when raw tables were cut to known_at < kickoff: {bad[:3]}")


# ------------------------------------------------------------------ (b) future-row canaries
def canary_targets() -> list[pit.TargetRow]:
    s = store()
    picks = [("00-0034844", 10), ("00-0036322", 10), ("00-0034796", 10), ("00-0039337", 10),
             ("00-0036554", 10), ("00-0034796", 1), ("00-0039910", 3), ("00-0033906", 17)]
    return [make_target(s, p, 2024, w) for p, w in picks]


class FutureRowCanaries(unittest.TestCase):
    def test_rows_after_and_at_kickoff_change_nothing(self):
        for t in canary_targets():
            base = fingerprint(build_features(store(), t))
            stamps = [t.kickoff + pd.Timedelta(hours=1),        # an hour after kickoff
                      t.kickoff + pd.Timedelta(days=7),         # next week's monster game
                      t.kickoff]                                # exactly at kickoff: strict <
            dirty = audit.inject_canaries(store(), t, stamps)
            self.assertEqual(fingerprint(build_features(dirty, t)), base, f"{t.player_id} wk{t.week}")

    def test_the_target_games_own_rows_change_nothing(self):
        for t in canary_targets():
            base = fingerprint(build_features(store(), t))
            self.assertEqual(fingerprint(build_features(audit.perturb_own_game(store(), t), t)), base,
                             f"{t.player_id} wk{t.week}")

    def test_control_the_canaries_do_bite_when_stamped_before_kickoff(self):
        """Without this the two tests above could pass vacuously (e.g. a gate that drops everything).
        The same rows, stamped one second BEFORE kickoff, must move every feature family they touch."""
        expect = {"pts_ppr_l1", "carries_l1", "targets_l1", "snap_pct_l1", "xfp_l1",
                  "dvp_ppr_l2", "dvp_ppr_l4", "team_spread", "total_line", "implied_team_total",
                  "wx_temp_obs", "wx_wind_obs", "inj_days_since_report", "draft_round", "draft_overall"}
        for t in canary_targets():
            base = build_features(store(), t)
            want = set(expect)
            if base["inj_report_status"] != "Out":            # already-Out players cannot move status
                want.add("inj_report_status")
            if base["inj_practice_status"] != "Did Not Participate In Practice":
                want.add("inj_practice_status")
            # injury/line/weather/static rows sit 1 s before kickoff; the game-result tables are read at
            # kickoff - RESULT_LAG (Phase 1), so those canary rows sit 1 s inside THAT cutoff
            live = build_features(audit.inject_canaries(store(), t, [t.kickoff - pd.Timedelta(seconds=1)],
                                                        week_offset=-1, result_lag=features.RESULT_LAG), t)
            moved = set(audit.diff_keys(base, live))
            self.assertTrue(want <= moved, f"{t.player_id} wk{t.week}: did not move {sorted(want - moved)}")
            self.assertEqual(live["inj_report_status"], "Out")
            self.assertEqual(live["pts_ppr_l1"], audit.MONSTER)
            self.assertAlmostEqual(live["inj_days_since_report"], 1 / 86400, places=9)


# ------------------------------------------------------------------ (c) schedule sanity
RESULT_TABLES = ("player_games", "snap_counts", "xfp", "game_results")


class ScheduleSanity(unittest.TestCase):
    def test_no_input_is_the_target_game_or_later(self):
        # Phase 0 asserted "every consumed game is from an EARLIER WEEK". Phase 1 also reads teammates'
        # histories, and a teammate who changed teams between a Thursday game and a Sunday kickoff has a
        # same-week game that is legitimately finished. The invariant that matters is "the consumed game
        # had finished before this kickoff", so it is asserted directly (kickoff + 4h, finding 7), and a
        # same-day earlier slot (1pm game feeding a 4:25pm row) would fail it.
        kick = store()._tables["fixtures"].df.set_index("game_id")["kickoff"]
        for t in targets():
            trace: list = []
            f = build_features(store(), t, trace=trace)
            for rec in trace:
                if rec["max_known_at"] is not None:
                    self.assertLess(rec["max_known_at"], t.kickoff, f"{rec['table']} {t.player_id} wk{t.week}")
                if rec["table"] in RESULT_TABLES:
                    self.assertNotIn(t.game_id, rec["game_ids"], f"{rec['table']} fed the target game itself")
                    for g in rec["game_ids"]:
                        self.assertLessEqual(kick[g] + pd.Timedelta(hours=4), t.kickoff,
                                             f"{rec['table']} consumed {g}, which had not finished before {t.player_id} wk{t.week}")

            lag_weeks = [f[f"lag{k}_week"] for k in (1, 2, 3) if f[f"lag{k}_week"] is not None]
            self.assertTrue(all(w < t.week for w in lag_weeks), (t.player_id, t.week, lag_weeks))
            self.assertEqual(lag_weeks, sorted(lag_weeks, reverse=True))
            self.assertEqual(len(set(lag_weeks)), len(lag_weeks))
            self.assertGreaterEqual(f["games_std"], len(lag_weeks))
            for n in (2, 4):
                lw = f[f"dvp_l{n}_last_week"]
                self.assertTrue(lw is None or lw <= t.week - 1, (t.player_id, t.week, n, lw))
                self.assertLessEqual(f[f"dvp_l{n}_games"], n)
            if f["weeks_since_last_game"] is not None:
                self.assertGreaterEqual(f["weeks_since_last_game"], 1)
            if f["inj_days_since_report"] == f["inj_days_since_report"]:      # not NaN
                self.assertGreater(f["inj_days_since_report"], 0)

    def test_same_week_xfp_is_never_a_feature(self):
        """ffopportunity xFP for week N is computed from week N: only lagged xFP may exist. Phase 1 widened
        the exact list with two more LAGGED aggregates (season-to-date mean, last season's per-game mean);
        it is still an exact list, and the own-game-perturbation test proves none reads the target game."""
        t = make_target(store(), "00-0034844", 2024, 10)
        names = [k for k in build_features(store(), t) if "xfp" in k]
        self.assertEqual(names, ["xfp_l1", "xfp_l2", "xfp_l3", "xfp_std_mean", "prev_season_xfp_pg"])

    def test_week_one_has_no_history_and_says_so(self):
        for t in [x for x in targets() if x.week == 1]:
            f = build_features(store(), t)
            self.assertEqual(f["games_std"], 0)
            self.assertIsNone(f["lag1_week"])
            self.assertEqual(f["dvp_l2_games"], 0)


# ------------------------------------------------------------------ the module boundary
class FeatureCodeCannotBypassTheGate(unittest.TestCase):
    def test_features_module_only_imports_the_gate_and_never_touches_raw_frames(self):
        tree = ast.parse(Path(features.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "model.point_in_time":
                self.assertEqual({a.name for a in node.names}, {"TargetRow", "as_of_join"})
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("model"):
                self.assertEqual(node.module, "model.point_in_time")
            if isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, {"_tables", "df", "read_parquet", "read_csv", "read_json"},
                                 f"line {node.lineno}: raw access")
            if isinstance(node, ast.Name):
                self.assertNotIn(node.id, {"load_store", "make_target"} | {n for n in dir(pit) if n.startswith("load_")},
                                 f"line {node.lineno}: loaders are not for feature code")


# ------------------------------------------------------------------ the audit bites
class TheAuditCatchesALeakyJoin(unittest.TestCase):
    """Deliberate leak: model/audit.py::leaky_features does the ad-hoc merge (week <= target, same-week
    xFP, injury by (player, week) with no known_at). If these tests ever fail, the audit has gone blind."""

    def setUp(self):
        self.sample = audit.sample_targets(store(), per_pos=1, weeks=range(2, 19, 2))

    def test_truncation_audit_flags_the_leaky_builder(self):
        bad = audit.audit_truncation(audit.leaky_features, store(), self.sample)
        pg = store()._tables["player_games"].df
        played = set(zip(pg["player_id"], pg["week"]))
        own_row = [t for t in self.sample if (t.player_id, t.week) in played]
        flagged = {(b["player_id"], b["week"]) for b in bad}
        caught = sum((t.player_id, t.week) in flagged for t in own_row)
        self.assertGreaterEqual(caught, 0.95 * len(own_row),
                                f"leaky merge should be flagged on every row whose own game is in the raw table ({caught}/{len(own_row)})")
        keys = {k for b in bad for k in b["keys"]}
        self.assertTrue({"pts_ppr_l1", "xfp_l1"} <= keys, keys)
        clean = audit.audit_truncation(audit.gated_subset, store(), self.sample)
        self.assertEqual(clean, [], "the gated builder passes the very same audit")

    def test_leaky_builder_reads_the_answer(self):
        """Not just 'different': the leaky lag-1 IS the target game's actual score."""
        hits = total = 0
        for t in self.sample:
            pg = store()._tables["player_games"].df
            own = pg[(pg["player_id"] == t.player_id) & (pg["week"] == t.week)]
            if own.empty:
                continue
            total += 1
            hits += audit.leaky_features(store(), t)["pts_ppr_l1"] == float(own["fantasy_points_ppr"].iloc[0])
        self.assertGreater(total, 0)
        self.assertEqual(hits, total)

    def test_canary_flags_the_leaky_builder_too(self):
        t = make_target(store(), "00-0034844", 2024, 10)
        base = audit.leaky_features(store(), t)
        dirty = audit.leaky_features(audit.perturb_own_game(store(), t), t)
        self.assertNotEqual(fingerprint(base), fingerprint(dirty))          # own-game rows move it
        self.assertEqual(fingerprint(build_features(store(), t)),
                         fingerprint(build_features(audit.perturb_own_game(store(), t), t)))  # gated: immune


if __name__ == "__main__":
    unittest.main()
