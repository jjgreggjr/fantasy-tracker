"""The opt-in `script` family (game script and coaching context) on a synthetic store. No network, no cache.

    python -m unittest model.tests.test_script_family

  * the raw team-game sums (point_in_time._script_sums) from a hand-made play list, tempo included
  * trailing-4 windows, opponent splits, the Vegas interactions and the expected-volume chain, against hand arithmetic
  * the expected-plays rule: the constants are applied as documented, and `script_rule.fit` recovers planted coefficients
  * hc_changed_this_season / hc_games_in_charge, including week 1 against last season's final coach and a mid-season firing
  * leakage: the target game's own rows (and anything inside the result cutoff) never enter; the trace proves it
  * registry: no column shared with pbp_team or any other family; opt-in, so the primary's 71 inputs are unchanged
"""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from model import features as F
from model import point_in_time as pit
from model import script_rule
from model import train as T
from model.point_in_time import KNOWN_AT, RawStore, RawTable, TargetRow

SEASON = 2024
KICK0 = pd.Timestamp("2024-09-08 17:00", tz="UTC")


def ko(week: int, season: int = SEASON) -> pd.Timestamp:
    return KICK0 + pd.Timedelta(weeks=week - 1) - pd.Timedelta(weeks=52 * (SEASON - season))


def _ns(s) -> pd.Series:
    return pd.to_datetime(pd.Series(s), utc=True).astype("datetime64[ns, UTC]")


def team_row(team: str, opp: str, season: int, week: int, gid: str, plays: float, dropbacks: float, **sc) -> dict:
    r = {"team": team, "opponent": opp, "season": season, "week": week, "game_type": "REG", "game_id": gid,
         "plays": plays, "dropbacks": dropbacks}
    r.update({c: 0.0 for c in pit.PBP_SCRIPT_COUNTS})
    r.update(sc)
    r[KNOWN_AT] = ko(week, season)
    return r


def sc_for(i: int) -> dict:
    """Distinct, hand-checkable sums for game i."""
    return dict(sc_epa=0.1 * i * 60, sc_succ=0.4 * 60 + i, sc_epa_n=60.0, sc_pass_epa=2.0 * i, sc_pass_n=30.0, sc_rush_epa=-1.0 * i,
                sc_rush_n=30.0, sc_pass_oe=5.0 * i, sc_xpass=0.5 * 55, sc_xp_n=55.0, sc_nohuddle=float(i), sc_secs=28.0 * 50 + i,
                sc_secs_n=50.0, sc_early_plays=40.0, sc_early_db=20.0 + i, sc_wp_plays=30.0, sc_wp_db=15.0 + i, sc_rz_plays=float(i))


def synthetic_store(*, extra_team_rows: list[dict] = (), coaches: list[tuple] | None = None) -> RawStore:
    """KC plays weeks 1-6 of 2024 (week 6 is the target), and weeks 16-17 of 2023; BUF's defense faces 3 games before week 6."""
    rows = [team_row("KC", "XXX", 2023, 16, "2023_16_KC", 70.0, 40.0, **sc_for(9)),
            team_row("KC", "XXX", 2023, 17, "2023_17_KC", 50.0, 20.0, **sc_for(8))]
    for w, (plays, db) in enumerate([(60.0, 36.0), (62.0, 30.0), (64.0, 40.0), (66.0, 33.0), (68.0, 34.0)], start=1):
        rows.append(team_row("KC", "YYY", SEASON, w, f"2024_{w:02d}_KC", plays, db, **sc_for(w)))
    rows.append(team_row("KC", "BUF", SEASON, 6, "2024_06_KC_BUF", 999.0, 999.0, **{c: 999.0 for c in pit.PBP_SCRIPT_COUNTS}))
    rows.append(team_row("BUF", "KC", SEASON, 6, "2024_06_KC_BUF", 999.0, 999.0, **{c: 999.0 for c in pit.PBP_SCRIPT_COUNTS}))
    for w in (2, 3, 4):                                   # offenses that faced BUF
        rows.append(team_row(f"O{w}", "BUF", SEASON, w, f"2024_{w:02d}_O{w}_BUF", 60.0, 30.0,
                             sc_epa=0.2 * 60, sc_epa_n=60.0, sc_pass_epa=float(w), sc_pass_n=30.0, sc_rush_epa=-float(w), sc_rush_n=20.0))
    rows += list(extra_team_rows)
    tm = pd.DataFrame(rows)
    tm[KNOWN_AT] = _ns(tm[KNOWN_AT])
    if coaches is None:
        coaches = [(2023, 16, "Andy Reid"), (2023, 17, "Andy Reid")] + [(SEASON, w, "Andy Reid") for w in range(1, 7)]
    co = pd.DataFrame([{"game_id": f"g{s}_{w}", "season": s, "week": w, "game_type": "REG", "team": "KC", "coach": c,
                        "kickoff": ko(w, s)} for s, w, c in coaches])
    co.loc[(co["season"] == SEASON) & (co["week"] == 6), "game_id"] = "2024_06_KC_BUF"
    co[KNOWN_AT] = _ns(co["kickoff"] - pit.COACH_LEAD)
    return RawStore([RawTable("pbp_team", tm, "game kickoff"), RawTable("coaches", co, "kickoff - 24h")])


def target(week: int = 6, gid: str = "2024_06_KC_BUF", season: int = SEASON) -> TargetRow:
    return TargetRow("P1", season, week, "KC", "BUF", "WR", gid, ko(week, season), True)


VEGAS = {"team_spread": 3.0, "total_line": 48.0, "implied_team_total": 25.5, "implied_opp_total": 22.5}


def hist(games: list[tuple]) -> F.Hist:
    """games: (game_id, season, week, team, target_share, carry_share, attempts)."""
    rows = []
    for gid, s, w, team, tsh, csh, att in games:
        v = [0.0] * F.N_VALUES
        v[F.TSH], v[F.CSH], v[F.ATT] = tsh, csh, att
        rows.append((gid, s, w, True, True, team, *v))
    return F.Hist(rows)


# ------------------------------------------------------------------ raw sums
class ScriptSums(unittest.TestCase):
    def test_sums_and_tempo_from_a_hand_made_play_list(self):
        p = pd.DataFrame({
            "game_id": ["G"] * 6, "play_id": [1, 2, 3, 4, 5, 6], "posteam": ["KC"] * 6,
            "fixed_drive": [1, 1, 1, 2, 2, 2], "qtr": [1, 1, 1, 1, 1, 5],
            "game_seconds_remaining": [3600.0, 3570.0, 3530.0, 3400.0, 3380.0, 500.0],
            "epa": [0.5, -0.2, np.nan, 1.0, 0.0, 0.3], "success": [1.0, 0.0, np.nan, 1.0, 0.0, 1.0],
            "pass_oe": [10.0, -20.0, 5.0, np.nan, 30.0, 0.0], "xpass": [0.6, 0.3, 0.5, np.nan, 0.7, 0.5],
            "no_huddle": [0, 1, 1, 0, 0, 0], "down": [1, 2, 3, 1, 4, 1], "wp": [0.5, 0.9, 0.3, 0.1, 0.8, 0.5],
            "dropback": [True, False, True, True, False, False], "rz": [False, False, True, True, False, False],
            "yards_gained": [25.0, 3.0, np.nan, 20.0, -2.0, 19.0]})
        r = pit._script_sums(p, ["game_id", "posteam"]).iloc[0]
        self.assertEqual(r["sc_explosive"], 2.0, "20+ yards: 25 and exactly 20; NaN counts as 0")
        self.assertAlmostEqual(r["sc_epa"], 0.5 - 0.2 + 1.0 + 0.0 + 0.3)
        self.assertEqual(r["sc_epa_n"], 5.0)
        self.assertEqual(r["sc_succ"], 3.0)
        self.assertAlmostEqual(r["sc_pass_epa"], 1.5)
        self.assertEqual(r["sc_pass_n"], 2.0)
        self.assertAlmostEqual(r["sc_rush_epa"], 0.1)
        self.assertEqual((r["sc_pass_oe"], r["sc_xp_n"]), (25.0, 5.0))
        self.assertAlmostEqual(r["sc_xpass"], 2.6)
        self.assertEqual(r["sc_nohuddle"], 2.0)
        self.assertEqual((r["sc_early_plays"], r["sc_early_db"]), (4.0, 2.0))
        self.assertEqual((r["sc_wp_plays"], r["sc_wp_db"]), (4.0, 2.0))          # wp .5, .3, .8, .5
        self.assertEqual(r["sc_rz_plays"], 2.0)
        # tempo: 30 + 40 inside drive 1, 20 inside drive 2; the overtime play and the drive change are not gaps
        self.assertEqual((r["sc_secs"], r["sc_secs_n"]), (90.0, 3.0))


# ------------------------------------------------------------------ trailing windows and the expected-volume chain
class TrailingMath(unittest.TestCase):
    def setUp(self):
        self.st = synthetic_store()
        self.sc = F._script_team(self.st, target(), VEGAS, None)

    def test_team_window_is_the_last_four_regular_season_games_before_the_target(self):
        sc, w = self.sc, [2, 3, 4, 5]          # weeks 2-5 of 2024 (week 1 and 2023 fall out; week 6 is the target)
        g = [sc_for(i) for i in w]
        s = lambda k: sum(x[k] for x in g)     # noqa: E731
        self.assertAlmostEqual(sc["scr_epa_per_play_l4"], s("sc_epa") / s("sc_epa_n"), places=12)
        self.assertAlmostEqual(sc["scr_success_rate_l4"], s("sc_succ") / s("sc_epa_n"), places=12)
        self.assertAlmostEqual(sc["scr_proe_l4"], s("sc_pass_oe") / s("sc_xp_n"), places=12)
        self.assertAlmostEqual(sc["scr_xpass_l4"], 0.5, places=12)
        self.assertAlmostEqual(sc["scr_sec_per_play_l4"], s("sc_secs") / s("sc_secs_n"), places=12)
        self.assertAlmostEqual(sc["scr_no_huddle_rate_l4"], s("sc_nohuddle") / (62 + 64 + 66 + 68), places=12)
        self.assertAlmostEqual(sc["scr_early_down_pass_rate_l4"], s("sc_early_db") / s("sc_early_plays"), places=12)
        self.assertAlmostEqual(sc["scr_wp_neutral_pass_rate_l4"], s("sc_wp_db") / s("sc_wp_plays"), places=12)
        self.assertAlmostEqual(sc["scr_rz_plays_pg_l4"], s("sc_rz_plays") / 4, places=12)
        self.assertAlmostEqual(sc["_trail_plays"], (62 + 64 + 66 + 68) / 4, places=12)
        self.assertAlmostEqual(sc["_trail_pass_rate"], (30 + 40 + 33 + 34) / (62 + 64 + 66 + 68), places=12)

    def test_opponent_defense_window_and_splits(self):
        sc = self.sc
        self.assertAlmostEqual(sc["scr_opp_epa_allowed_l4"], 0.2, places=12)
        self.assertAlmostEqual(sc["scr_opp_pass_epa_allowed_l4"], (2 + 3 + 4) / 90, places=12)
        self.assertAlmostEqual(sc["scr_opp_rush_epa_allowed_l4"], -(2 + 3 + 4) / 60, places=12)

    def test_interactions_and_expected_volume(self):
        sc, r = self.sc, F.SCRIPT_PLAYS_RULE
        rate, trail = sc["_trail_pass_rate"], sc["_trail_plays"]
        self.assertAlmostEqual(sc["scr_itt_x_pass_rate"], 25.5 * rate, places=12)
        self.assertAlmostEqual(sc["scr_spread_x_proe"], 3.0 * sc["scr_proe_l4"], places=12)
        plays = r["intercept"] + r["trail_plays"] * trail + r["abs_spread"] * 3.0 + r["total"] * 48.0
        self.assertAlmostEqual(sc["expected_team_plays"], plays, places=12)
        self.assertAlmostEqual(sc["expected_pass_att"], plays * rate, places=12)
        self.assertAlmostEqual(sc["expected_rush_att"], plays * (1 - rate), places=12)
        dog = F._script_team(synthetic_store(), target(), {**VEGAS, "team_spread": -3.0}, None)
        self.assertAlmostEqual(dog["expected_team_plays"], plays, places=12, msg="the rule reads |spread|")
        self.assertAlmostEqual(dog["scr_spread_x_proe"], -sc["scr_spread_x_proe"], places=12)
        none = F._script_team(synthetic_store(), target(), {k: math.nan for k in VEGAS}, None)
        self.assertTrue(math.isnan(none["expected_team_plays"]) and math.isnan(none["scr_itt_x_pass_rate"]))
        self.assertFalse(math.isnan(none["scr_proe_l4"]))

    def test_week_one_reads_last_seasons_final_games(self):
        sc = F._script_team(self.st, target(1, "2024_01_KC"), VEGAS, None)
        self.assertAlmostEqual(sc["_trail_plays"], 60.0, places=12)               # 2023 wk16-17 only (2024 wk1 is the target)
        self.assertAlmostEqual(sc["_trail_pass_rate"], 60 / 120, places=12)
        self.assertTrue(math.isnan(sc["scr_opp_epa_allowed_l4"]), "BUF's defense has no game before week 1")

    def test_player_shares_times_expected_volume(self):
        sc = self.sc
        h = hist([("2024_01_KC", SEASON, 1, "KC", 0.10, 0.0, 0.0), ("2024_02_KC", SEASON, 2, "KC", 0.20, 0.5, 10.0),
                  ("2024_03_KC", SEASON, 3, "KC", 0.30, 0.3, 20.0), ("2024_05_KC", SEASON, 5, "KC", 0.40, 0.1, 30.0)])
        p = F._script_player(self.st, target(), h, sc, None)
        self.assertAlmostEqual(p["exp_player_targets"], (0.2 + 0.3 + 0.4) / 3 * sc["expected_pass_att"], places=12)
        self.assertAlmostEqual(p["exp_player_carries"], (0.5 + 0.3 + 0.1) / 3 * sc["expected_rush_att"], places=12)
        self.assertAlmostEqual(p["exp_player_pass_att"], (10 + 20 + 30) / (30 + 40 + 34) * sc["expected_pass_att"], places=12)
        rookie = F._script_player(self.st, target(), hist([]), sc, None)
        self.assertTrue(all(math.isnan(v) for v in rookie.values()))


class ExpectedPlaysRule(unittest.TestCase):
    def test_fit_recovers_planted_coefficients(self):
        rng = np.random.default_rng(7)
        n = 4000
        d = pd.DataFrame({"trail_plays": rng.normal(63, 4, n), "spread": rng.normal(0, 6, n), "total": rng.normal(45, 4, n)})
        d["plays"] = 40 + 0.2 * d["trail_plays"] - 0.3 * d["spread"].abs() + 0.15 * d["total"] + rng.normal(0, 0.01, n)
        got = script_rule.fit(d)
        for k, v in {"intercept": 40, "trail_plays": 0.2, "abs_spread": -0.3, "total": 0.15}.items():
            self.assertAlmostEqual(got[k], v, delta=0.02, msg=k)

    def test_the_rule_has_the_documented_terms_and_is_missing_on_missing_inputs(self):
        self.assertEqual(set(F.SCRIPT_PLAYS_RULE), set(script_rule.TERMS))
        self.assertTrue(math.isnan(F.expected_team_plays(math.nan, 3.0, 45.0)))
        rule = {"intercept": 1.0, "trail_plays": 2.0, "abs_spread": 3.0, "total": 4.0}
        self.assertEqual(F.expected_team_plays(10.0, -2.0, 5.0, rule), 1 + 20 + 6 + 20)


# ------------------------------------------------------------------ coaching
class Coaching(unittest.TestCase):
    def coach(self, coaches, week=6, season=SEASON):
        gid = "2024_06_KC_BUF" if (season, week) == (SEASON, 6) else f"g{season}_{week}"
        return F._coach_features(synthetic_store(coaches=coaches), target(week, gid, season), None)

    def test_same_coach_all_along(self):
        c = self.coach(None)
        self.assertEqual(c["hc_changed_this_season"], 0.0)
        self.assertEqual(c["hc_games_in_charge"], 7.0)          # 2023 wk16-17 + 2024 wk1-5; the target game is not counted

    def test_a_mid_season_firing(self):
        rows = [(2023, 16, "A"), (2023, 17, "A"), (SEASON, 1, "A"), (SEASON, 2, "A"), (SEASON, 3, "A"), (SEASON, 4, "B"),
                (SEASON, 5, "B"), (SEASON, 6, "B")]
        c = self.coach(rows)
        self.assertEqual((c["hc_changed_this_season"], c["hc_games_in_charge"]), (1.0, 2.0))
        c3 = self.coach(rows, week=3)
        self.assertEqual((c3["hc_changed_this_season"], c3["hc_games_in_charge"]), (0.0, 4.0))

    def test_week_one_compares_with_last_seasons_final_coach(self):
        rows = [(2023, 16, "A"), (2023, 17, "A"), (SEASON, 1, "B"), (SEASON, 2, "B")]
        c = self.coach(rows, week=1)
        self.assertEqual((c["hc_changed_this_season"], c["hc_games_in_charge"]), (1.0, 0.0))
        self.assertEqual(self.coach(rows, week=2)["hc_changed_this_season"], 0.0, "B is the week-1 coach of this season")
        kept = [(2023, 16, "A"), (2023, 17, "A"), (SEASON, 1, "A")]
        self.assertEqual(self.coach(kept, week=1)["hc_changed_this_season"], 0.0)

    def test_no_coach_row_for_the_game_is_missing_not_zero(self):
        c = self.coach([(2023, 16, "A")], week=1)
        self.assertTrue(math.isnan(c["hc_changed_this_season"]) and math.isnan(c["hc_games_in_charge"]))

    def test_a_coach_row_known_after_kickoff_is_invisible(self):
        st = synthetic_store()
        co = st._tables["coaches"].df.copy()
        co.loc[co["game_id"] == "2024_06_KC_BUF", KNOWN_AT] = ko(6) + pd.Timedelta(seconds=1)
        c = F._coach_features(st.with_frame("coaches", co), target(), None)
        self.assertTrue(math.isnan(c["hc_games_in_charge"]))


# ------------------------------------------------------------------ leakage
class Leakage(unittest.TestCase):
    def script_cols(self, st: RawStore) -> dict:
        sc = F._script_team(st, target(), VEGAS, None)
        h = hist([("2024_04_KC", SEASON, 4, "KC", 0.2, 0.2, 5.0), ("2024_05_KC", SEASON, 5, "KC", 0.3, 0.1, 5.0)])
        return {**{k: v for k, v in sc.items() if not k.startswith("_")}, **F._script_player(st, target(), h, sc, None)}

    def test_the_target_games_own_rows_never_enter(self):
        base = self.script_cols(synthetic_store())
        st = synthetic_store()
        tm = st._tables["pbp_team"].df
        gone = self.script_cols(st.with_frame("pbp_team", tm[tm["game_id"] != "2024_06_KC_BUF"]))
        self.assertEqual(F.fingerprint(base), F.fingerprint(gone))
        self.assertTrue(all(not math.isnan(v) for v in base.values()), base)

    def test_a_game_inside_the_result_cutoff_is_invisible_and_one_just_before_it_is_read(self):
        base = self.script_cols(synthetic_store())
        for stamp, seen in ((ko(6) - F.RESULT_LAG, False), (ko(6) - pd.Timedelta(hours=1), False),
                            (ko(6) - F.RESULT_LAG - pd.Timedelta(seconds=1), True)):
            extra = team_row("KC", "ZZZ", SEASON, 6, "LATE", 200.0, 200.0, **{c: 500.0 for c in pit.PBP_SCRIPT_COUNTS})
            extra[KNOWN_AT] = stamp
            got = self.script_cols(synthetic_store(extra_team_rows=[extra]))
            self.assertEqual(F.fingerprint(base) != F.fingerprint(got), seen, stamp)

    def test_the_trace_shows_only_finished_games(self):
        trace: list = []
        st = synthetic_store()
        sc = F._script_team(st, target(), VEGAS, trace)
        F._script_player(st, target(), hist([("2024_05_KC", SEASON, 5, "KC", 0.3, 0.1, 5.0)]), sc, trace)
        pbp = [r for r in trace if r["table"] == "pbp_team"]
        self.assertTrue(pbp)
        for r in pbp:
            self.assertNotIn("2024_06_KC_BUF", r["game_ids"])
            self.assertLessEqual(r["max_known_at"], ko(6) - F.RESULT_LAG)


# ------------------------------------------------------------------ registry
class Registry(unittest.TestCase):
    def test_columns_are_unique_and_share_nothing_with_pbp_team_or_any_family(self):
        cols = F.FAMILIES["script"]
        self.assertEqual(len(cols), len(set(cols)))
        self.assertEqual(len(cols), 22)
        self.assertFalse(set(cols) & set(F.FAMILIES["pbp_team"]))
        others = {c for f, cs in F.FAMILIES.items() if f != "script" for c in cs}
        self.assertFalse(set(cols) & others)
        self.assertFalse(set(pit.PBP_SCRIPT_COUNTS) & set(pit.PBP_TEAM_COUNTS))

    def test_opt_in_and_the_primary_keeps_its_71_inputs(self):
        self.assertIn("script", T.OPT_IN_FAMILIES)
        df = pd.DataFrame({c: [0.0, 1.0] for c in F.feature_columns()} | {"position": ["QB", "RB"]})
        prim = T.feature_columns(df, T.PRIMARY)
        self.assertFalse(set(prim) & set(F.FAMILIES["script"]))
        with_ = T.feature_columns(df, T.Spec("x", add_families=("script",)))
        self.assertEqual(set(with_) - set(prim), set(F.FAMILIES["script"]))
        cfg = json.loads((Path(T.__file__).resolve().parent / "serving_config.json").read_text())
        self.assertEqual(len(cfg["inputs"]), 71)
        self.assertFalse(set(cfg["inputs"]) & set(F.FAMILIES["script"]))

    def test_a_store_without_pbp_or_coaches_says_nan(self):
        st = RawStore([])
        sc = F._script_team(st, target(), VEGAS, None)
        self.assertTrue(all(math.isnan(sc[c]) for c in F.FAMILIES["script"] if c in sc))


if __name__ == "__main__":
    unittest.main()
