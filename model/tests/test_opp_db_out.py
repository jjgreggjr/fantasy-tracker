"""The opt-in `opp_db_out` family (model/features.py `_opp_def_features`): the OPPONENT's defensive injury report, known before
kickoff, joined to its newest defensive depth chart (`point_in_time.load_depth_charts_def`).

    python -m unittest model.tests.test_opp_db_out -v

No network, no real matrix: the loader reads synthetic depth-chart files (both formats) from a temporary cache, the features
read a synthetic store. Proves: an Out CB is counted and a Questionable one is not; a report stamped at or after kickoff never
enters (and a missing report is NaN, not 0); a starter is rank 1 at his slot on the newest fresh chart; the offense-only
`depth_charts` table is byte-identical with or without defensive rows in the file; the family is registered opt-in and the
frozen primary's inputs are untouched.
"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from model import features as F
from model import point_in_time as pit
from model import serve as S
from model import train as T
from model.features import NAN, DEPTH_MAX_AGE, OUT_LIKE, _opp_def_features
from model.point_in_time import KNOWN_AT, RawStore, RawTable, TargetRow

COLS = F.FAMILIES["opp_db_out"]
KO = pd.Timestamp("2025-10-05 17:00", tz="UTC")          # KC's 2025 week-5 kickoff in the synthetic schedule
SEC = pd.Timedelta(seconds=1)
DAY = pd.Timedelta(days=1)


def target(week: int = 5, season: int = 2025) -> TargetRow:
    return TargetRow("P1", season, week, "KC", "DET", "WR", f"{season}_{week:02d}_DET_KC", KO, True, ("usage",))


def same(a, b) -> bool:
    return (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)) or a == b


# ------------------------------------------------------------------ synthetic store
def inj(pid: str, pos: str, status, known_at: pd.Timestamp, *, week: int = 5, team: str = "DET", season: int = 2025) -> dict:
    return {"player_id": pid, "season": season, "week": week, "game_type": "REG", "team": team, "position": pos,
            "report_status": status, "report_primary_injury": "Knee", "practice_status": None, "practice_primary_injury": None,
            KNOWN_AT: known_at, "known_at_source": "test"}


def dc(pid: str, slot: str, grp: str, rank: int, known_at: pd.Timestamp, *, team: str = "DET") -> dict:
    return {"player_id": pid, "team": team, "season": 2025, "week": pd.NA, "slot": slot, "grp": grp, "rank": rank,
            KNOWN_AT: known_at, "known_at_source": "test"}


def _frame(rows: list[dict], cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=cols)
    df[KNOWN_AT] = pd.to_datetime(df[KNOWN_AT], utc=True).astype("datetime64[ns, UTC]")
    return df


def store(injuries: list[dict], charts: list[dict] | None = None) -> RawStore:
    tables = [RawTable("injuries", _frame(injuries, list(inj("x", "CB", "Out", KO))), "test")]
    if charts is not None:
        tables.append(RawTable("depth_charts_def", _frame(charts, list(dc("x", "LCB", "DB", 1, KO))), "test"))
    return RawStore(tables)


REPORT_AT = KO - DAY                                       # the derived 2025+ stamp: kickoff - 24h
CHART_AT = KO - 3 * DAY
# the opponent's chart: CB1 starts at LCB over CB2, S1 at FS, LB1 at MLB, DE1 at LDE
CHART = [dc("CB1", "LCB", "DB", 1, CHART_AT), dc("CB2", "LCB", "DB", 2, CHART_AT), dc("S1", "FS", "DB", 1, CHART_AT),
         dc("LB1", "MLB", "LB", 1, CHART_AT), dc("DE1", "LDE", "DL", 1, CHART_AT)]


def feats(injuries, charts=CHART, trace=None, week: int = 5) -> dict:
    return _opp_def_features(store(injuries, charts), target(week), trace)


# ------------------------------------------------------------------ counts
class Counts(unittest.TestCase):
    def test_an_out_cb_is_counted_and_a_questionable_one_is_not(self):
        f = feats([inj("CB1", "CB", "Out", REPORT_AT), inj("CB2", "CB", "Questionable", REPORT_AT)])
        self.assertEqual(f["opp_report_present"], 1)
        self.assertEqual(f["opp_db_out_n"], 1.0)
        self.assertEqual(f["opp_db_q_n"], 1.0)
        self.assertEqual(f["opp_def_out_n"], 1.0)
        self.assertEqual((f["opp_lb_out_n"], f["opp_dl_out_n"]), (0.0, 0.0))

    def test_doubtful_counts_as_out_and_every_group_lands_in_its_own_column(self):
        f = feats([inj("S1", "S", "Doubtful", REPORT_AT), inj("LB1", "LB", "Out", REPORT_AT), inj("DE1", "DE", "Out", REPORT_AT),
                   inj("DT1", "DT", "Questionable", REPORT_AT), inj("WR9", "WR", "Out", REPORT_AT)])    # an offensive Out: not defence
        self.assertEqual(OUT_LIKE, ("Out", "Doubtful"))
        self.assertEqual((f["opp_db_out_n"], f["opp_lb_out_n"], f["opp_dl_out_n"], f["opp_def_out_n"]), (1.0, 1.0, 1.0, 3.0))
        self.assertEqual(f["opp_db_q_n"], 0.0)

    def test_a_clean_report_is_zero_and_a_missing_report_is_nan(self):
        clean = feats([inj("WR9", "WR", "Questionable", REPORT_AT)])          # the opponent filed a report, no defender on it
        self.assertEqual(clean["opp_report_present"], 1)
        self.assertEqual([clean[c] for c in COLS if c != "opp_def_out_l4wk"], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        none = feats([inj("CB1", "CB", "Out", REPORT_AT, team="KC")])        # only OUR report exists: nothing for the opponent
        self.assertEqual(none["opp_report_present"], 0)
        self.assertTrue(all(math.isnan(none[c]) for c in COLS))
        other_week = feats([inj("CB1", "CB", "Out", REPORT_AT - 7 * DAY, week=4)])   # last week's report is not this week's
        self.assertEqual(other_week["opp_report_present"], 0)
        self.assertTrue(math.isnan(other_week["opp_db_out_n"]))
        self.assertEqual(other_week["opp_def_out_l4wk"], 1.0)                        # ...but it is the chronic window

    def test_the_last_known_designation_wins(self):
        f = feats([inj("CB1", "CB", "Questionable", REPORT_AT - DAY), inj("CB1", "CB", "Out", REPORT_AT)])
        self.assertEqual((f["opp_db_out_n"], f["opp_db_q_n"]), (1.0, 0.0))
        g = feats([inj("CB1", "CB", "Out", REPORT_AT - DAY), inj("CB1", "CB", "Questionable", REPORT_AT)])
        self.assertEqual((g["opp_db_out_n"], g["opp_db_q_n"]), (0.0, 1.0))

    def test_chronic_window_counts_defensive_outs_in_the_prior_four_weeks_only(self):
        rows = [inj("CB1", "CB", "Out", REPORT_AT)]
        for w, pid, pos, st in ((1, "S1", "S", "Out"), (2, "S1", "S", "Out"), (3, "LB1", "LB", "Questionable"),
                                (4, "WR9", "WR", "Out"), (0, "DE1", "DE", "Out")):          # week 0: outside the window
            rows.append(inj(pid, pos, st, REPORT_AT - 7 * (5 - w) * DAY, week=w))
        self.assertEqual(feats(rows)["opp_def_out_l4wk"], 2.0)
        self.assertTrue(math.isnan(feats([inj("CB1", "CB", "Out", REPORT_AT, week=1)], week=1)["opp_def_out_l4wk"]))
        self.assertTrue(math.isnan(feats(rows[:1])["opp_def_out_l4wk"]))                  # no prior report rows: unknown, not 0


# ------------------------------------------------------------------ starters
class Starters(unittest.TestCase):
    OUTS = [inj("CB1", "CB", "Out", REPORT_AT), inj("CB2", "CB", "Out", REPORT_AT), inj("S1", "S", "Doubtful", REPORT_AT),
            inj("LB1", "LB", "Out", REPORT_AT)]

    def test_rank_one_at_a_db_slot_is_a_starter(self):
        f = feats(self.OUTS)
        self.assertEqual(f["opp_db_out_n"], 3.0)
        self.assertEqual(f["opp_db_starter_out_n"], 2.0)          # CB1 (LCB #1) and S1 (FS #1); CB2 is the LCB #2; LB1 is not a DB

    def test_the_newest_chart_before_kickoff_decides(self):
        older = [dc("CB2", "LCB", "DB", 1, CHART_AT - 7 * DAY), dc("CB1", "LCB", "DB", 2, CHART_AT - 7 * DAY)]
        self.assertEqual(feats(self.OUTS, older + CHART)["opp_db_starter_out_n"], 2.0)
        flipped = [dc("CB2", "LCB", "DB", 1, CHART_AT + DAY), dc("S1", "FS", "DB", 2, CHART_AT + DAY)]      # newer: CB2 starts, S1 benched
        self.assertEqual(feats(self.OUTS, CHART + flipped)["opp_db_starter_out_n"], 1.0)

    def test_a_chart_at_or_after_kickoff_or_too_old_gives_no_starter_count(self):
        for late in (KO, KO + SEC, KO + 2 * DAY):
            f = feats(self.OUTS, [dc("CB1", "LCB", "DB", 1, late)])
            self.assertTrue(math.isnan(f["opp_db_starter_out_n"]), late)
            self.assertEqual(f["opp_db_out_n"], 3.0)                   # the report still counts
        stale = [dc("CB1", "LCB", "DB", 1, KO - DEPTH_MAX_AGE - SEC)]
        self.assertTrue(math.isnan(feats(self.OUTS, stale)["opp_db_starter_out_n"]))
        fresh = [dc("CB1", "LCB", "DB", 1, KO - DEPTH_MAX_AGE)]
        self.assertEqual(feats(self.OUTS, fresh)["opp_db_starter_out_n"], 1.0)
        self.assertEqual(feats(self.OUTS, [dc("CB1", "LCB", "DB", 1, KO - SEC)])["opp_db_starter_out_n"], 1.0)

    def test_without_the_defensive_depth_table_the_family_is_nan_and_nothing_is_read(self):
        """The serving store has no `depth_charts_def` unless the family is adopted (`load_store(defense=True)`): the family is
        then NaN without a read, as the charting families are, rather than the counts without the starters (a half-built row)."""
        trace: list = []
        f = feats(self.OUTS, charts=None, trace=trace)
        self.assertTrue(all(math.isnan(f[c]) for c in COLS))
        self.assertEqual(f["opp_report_present"], 0)
        self.assertEqual(trace, [])
        self.assertEqual(feats(self.OUTS, charts=[])["opp_db_out_n"], 3.0)          # the table, even empty: the report counts

    def test_another_teams_chart_is_not_the_opponents(self):
        ours = [dc("CB2", "LCB", "DB", 1, CHART_AT + DAY, team="KC")]          # newer, but KC's: neither newest-for-DET nor a DET starter
        self.assertEqual(feats(self.OUTS, CHART + ours)["opp_db_starter_out_n"], 2.0)
        self.assertTrue(math.isnan(feats(self.OUTS, ours)["opp_db_starter_out_n"]))


# ------------------------------------------------------------------ the gate
class Leakage(unittest.TestCase):
    def test_a_report_stamped_at_or_after_kickoff_never_enters(self):
        for late in (KO, KO + SEC, KO + 3 * DAY):
            f = feats([inj("CB1", "CB", "Out", late)])
            self.assertEqual(f["opp_report_present"], 0, late)
            self.assertTrue(all(math.isnan(f[c]) for c in COLS), late)
            mixed = feats([inj("CB2", "CB", "Questionable", REPORT_AT), inj("CB1", "CB", "Out", late)])
            self.assertEqual((mixed["opp_db_out_n"], mixed["opp_db_q_n"], mixed["opp_db_starter_out_n"]), (0.0, 1.0, 0.0), late)
        one_before = feats([inj("CB1", "CB", "Out", KO - SEC)])
        self.assertEqual((one_before["opp_db_out_n"], one_before["opp_db_starter_out_n"]), (1.0, 1.0))

    def test_every_read_goes_through_the_gate_and_the_trace_shows_it(self):
        trace: list = []
        feats([inj("CB1", "CB", "Out", REPORT_AT)], trace=trace)
        self.assertEqual([r["table"] for r in trace], ["injuries", "depth_charts_def"])
        for r in trace:
            self.assertLess(r["max_known_at"], KO)

    def test_the_context_is_memoised_per_game_and_opponent(self):
        st = store([inj("CB1", "CB", "Out", REPORT_AT)], CHART)
        a = _opp_def_features(st, target(), None)
        self.assertIs(_opp_def_features(st, target(), None), a)
        self.assertIn(("opp_def", target().game_id, "DET"), st._memo)


# ------------------------------------------------------------------ the loader: defensive rows as their own table, offense untouched
SCHEDULE = []           # (season, week, gameday, gametime ET, away, home): Thursday and Sunday each week; EEE never plays (bye)
for season, thu in ((2024, "2024-09-05"), (2025, "2025-09-04")):
    for w in (1, 2):
        d0 = pd.Timestamp(thu) + pd.Timedelta(days=7 * (w - 1))
        SCHEDULE += [(season, w, str(d0.date()), "20:15", "CCC", "DDD"),
                     (season, w, str((d0 + pd.Timedelta(days=3)).date()), "13:00", "AAA", "BBB")]


def _games_frame() -> pd.DataFrame:
    return pd.DataFrame([{"season": s, "week": w, "gameday": d, "gametime": t, "game_type": "REG", "away_team": a, "home_team": h,
                          "game_id": f"{s}_{w:02d}_{a}_{h}"} for s, w, d, t, a, h in SCHEDULE])


def _old_format(with_defense: bool) -> pd.DataFrame:
    # (club_code, week, formation, depth_position, depth_team, gsis_id)
    rows = [("BBB", 1, "Offense", "QB", "1", "00-Q1"), ("BBB", 1, "Offense", "WR", "1", "00-W1"),
            ("BBB", 1, "Offense", "WR", "2", "00-W2"), ("BBB", 1, "Offense", "LT", "1", "00-T1"),
            ("EEE", 1, "Offense", "RB", "1", "00-R9"),                                  # bye team: the week's last kickoff
            ("BBB", 2, "Offense", "QB", "1", "00-Q1"), ("BBB", None, "Offense", "QB", "1", "00-Q1"),   # SBBYE: no week
            ("BBB", 1, "Offense", "WR", "3", None)]                                    # no gsis id
    if with_defense:
        rows += [("BBB", 1, "Defense", "LCB", "1", "00-C1"), ("BBB", 1, "Defense", "LCB", "2", "00-C2"),
                 ("BBB", 1, "Defense", "RCB ", "1", "00-C3"), ("BBB", 1, "Defense", "FS", "1", "00-S1"),
                 ("BBB", 1, "Defense", "SS", "1", "00-S2"), ("BBB", 1, "Defense", "NB", "1", "00-C4"),
                 ("BBB", 1, "Defense", "MLB", "1", "00-L1"), ("BBB", 1, "Defense", "LDE", "1", "00-D1"),
                 ("BBB", 1, "Defense", "NT", "1", "00-D2"), ("BBB", 1, "Defense", "ZZ", "1", "00-X1"),     # unmapped code
                 ("EEE", 1, "Defense", "FS", "1", "00-S9"), ("BBB", 1, "Defense", "RCB", "2", None),
                 ("BBB", 1, "Special Teams", "PR", "1", "00-C1")]
    df = pd.DataFrame(rows, columns=["club_code", "week", "formation", "depth_position", "depth_team", "gsis_id"])
    df["season"] = 2024
    return df


def _new_format(with_defense: bool) -> pd.DataFrame:
    dt = pd.Timestamp("2025-09-02 15:00", tz="UTC")
    rows = [("BBB", dt, "Offense", "QB", 1, "00-Q1"), ("BBB", dt, "Offense", "WR", 1, "00-W1"),
            ("BBB", dt, "Offense", "WR", 2, "00-W2"), ("BBB", dt, "Offense", "LT", 1, "00-T1"),
            ("BBB", dt, "Offense", "WR", 3, None)]
    if with_defense:
        rows += [("BBB", dt, "Defense", "LCB", 1, "00-C1"), ("BBB", dt, "Defense", "LCB", 2, "00-C2"),
                 ("BBB", dt, "Defense", "FS", 1, "00-S1"), ("BBB", dt, "Defense", "WLB", 1, "00-L1"),
                 ("BBB", dt, "Defense", "DT", 1, "00-D1"), ("BBB", dt, "Defense", "ZZ", 1, "00-X1"),
                 ("BBB", dt, "Special Teams", "K", 1, "00-K1")]
    df = pd.DataFrame(rows, columns=["team", "dt", "pos_grp", "pos_abb", "pos_rank", "gsis_id"])
    df["season"] = 2025
    return df


class Loader(unittest.TestCase):
    def load(self, with_defense: bool):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            nv = root / "nflverse"
            nv.mkdir(parents=True)
            _games_frame().to_parquet(nv / "games.parquet")
            _old_format(with_defense).to_parquet(nv / "depth_charts_2024.parquet")
            _new_format(with_defense).to_parquet(nv / "depth_charts_2025.parquet")
            with mock.patch.object(pit, "CACHE_DIR", root), mock.patch.object(pit, "_GAMES", None), \
                    mock.patch("model.point_in_time.requests.get", side_effect=AssertionError("no network in tests")):
                off = pit.load_depth_charts([2024, 2025])
                de = pit.load_depth_charts_def([2024, 2025]) if with_defense else None
                return off, de

    def test_the_offense_table_is_byte_identical_with_or_without_defensive_rows_in_the_file(self):
        a, _ = self.load(with_defense=False)
        b, _ = self.load(with_defense=True)
        pd.testing.assert_frame_equal(a.df, b.df)
        self.assertEqual(a.dropped, b.dropped)
        self.assertEqual(a.known_at_rule, b.known_at_rule)
        self.assertTrue(set(a.df["pos"]) <= set(pit.OFFENSE_DEPTH_POS))
        self.assertEqual(len(a.df), 8)                 # 2024: QB, WR, WR, bye RB, week-2 QB; 2025: QB, WR, WR (LT and no-id rows dropped)

    def test_defensive_rows_carry_slot_group_rank_and_the_same_stamps(self):
        off, de = self.load(with_defense=True)
        d24 = de.df[de.df["season"] == 2024].set_index("player_id")
        self.assertEqual(d24.loc["00-C1", ["slot", "grp", "rank"]].tolist(), ["LCB", "DB", 1])
        self.assertEqual(d24.loc["00-C2", ["slot", "grp", "rank"]].tolist(), ["LCB", "DB", 2])
        self.assertEqual(d24.loc["00-C3", "slot"], "RCB")                       # stripped
        self.assertEqual(set(d24.loc[["00-S1", "00-S2", "00-C4"], "grp"]), {"DB"})
        self.assertEqual((d24.loc["00-L1", "grp"], d24.loc["00-D1", "grp"], d24.loc["00-D2", "grp"]), ("LB", "DL", "DL"))
        self.assertNotIn("00-X1", d24.index)                                     # unmapped code dropped, counted
        self.assertEqual(de.dropped["unmapped_def_position"], 2)
        self.assertEqual(de.dropped["no_gsis_id"], 1)
        self.assertEqual(set(d24["known_at_source"]), {"derived_team_kickoff"})
        o24 = off.df[(off.df["season"] == 2024) & (off.df["week"] == 1)].set_index("player_id")
        self.assertEqual(d24.loc["00-C1", KNOWN_AT], o24.loc["00-Q1", KNOWN_AT])       # BBB's week-1 kickoff, both tables
        self.assertEqual(d24.loc["00-S9", KNOWN_AT], o24.loc["00-R9", KNOWN_AT])       # the bye team's fallback, both tables
        self.assertEqual(d24.loc["00-C1", KNOWN_AT], pd.Timestamp("2024-09-08 17:00", tz="UTC"))
        d25 = de.df[de.df["season"] == 2025].set_index("player_id")
        self.assertEqual(set(d25.index), {"00-C1", "00-C2", "00-S1", "00-L1", "00-D1"})
        self.assertEqual((d25.loc["00-L1", "grp"], d25.loc["00-D1", "grp"]), ("LB", "DL"))
        self.assertEqual(set(d25["known_at_source"]), {"snapshot_dt"})
        self.assertTrue((d25[KNOWN_AT] == pd.Timestamp("2025-09-02 15:00", tz="UTC")).all())
        self.assertEqual(de.name, "depth_charts_def")
        self.assertEqual(set(de.df["grp"]), {"DB", "LB", "DL"})

    def test_the_store_loads_the_defensive_table_only_behind_the_defense_switch(self):
        """`load_store(pbp=False)` with the default switches (the serving store) parses neither the defensive depth chart nor
        FTN's `ftn_def`; `defense=True` (the matrix builder) adds the one, `charting=True` the other."""
        stamp = pd.Series([KO]).astype("datetime64[ns, UTC]")
        tiny = RawTable("player_games", pd.DataFrame({"player_id": ["x"], KNOWN_AT: stamp}), "t")
        sn = RawTable("snap_counts", pd.DataFrame({"player_id": ["x"], "position": ["WR"], KNOWN_AT: stamp}), "t")
        loaders = ("load_xfp", "load_injuries", "load_depth_charts", "load_depth_charts_def", "load_career_pre_cutoff", "load_coaches")
        lists = ("load_schedule_tables", "load_static_tables", "load_ngs_tables", "load_pfr_tables", "load_pbp_tables")
        with mock.patch.multiple(pit, load_player_games=mock.DEFAULT, load_snap_counts=mock.DEFAULT, load_ftn_tables=mock.DEFAULT,
                                 **{k: mock.DEFAULT for k in loaders + lists}) as m:
            m["load_player_games"].return_value, m["load_snap_counts"].return_value = tiny, sn
            for k in lists:
                m[k].return_value = []
            for k in loaders:
                m[k].return_value = RawTable(k.removeprefix("load_"), tiny.df, "t")
            m["load_ftn_tables"].return_value = [RawTable(n, tiny.df, "t") for n in ("ftn", "ftn_team", "ftn_def")]
            st = pit.load_store([2025], adp=False, college=False, pbp=False)
            self.assertFalse(m["load_depth_charts_def"].called or m["load_ftn_tables"].called)
            self.assertFalse(st.has("depth_charts_def") or st.has("ftn_def"))
            self.assertTrue(st.has("depth_charts") and st.has("injuries"))
            self.assertFalse(m["load_depth_charts_def"].called)
            st = pit.load_store([2025], adp=False, college=False, pbp=False, defense=True)
            self.assertEqual(m["load_depth_charts_def"].call_args.args, ([2025],))
            self.assertTrue(st.has("depth_charts_def") and not st.has("ftn_def"))
            st = pit.load_store([2025], adp=False, college=False, pbp=False, charting=True)
            self.assertTrue(st.has("ftn_def") and not st.has("depth_charts_def"))

    def test_the_group_constants_cover_every_group_once(self):
        self.assertEqual(set(pit.DEF_DEPTH_GROUPS), {"DB", "LB", "DL"})
        codes = [c for codes in pit.DEF_DEPTH_GROUPS.values() for c in codes]
        self.assertEqual(len(codes), len(set(codes)))
        self.assertFalse(set(codes) & set(pit.OFFENSE_DEPTH_POS))
        self.assertEqual(set(F.INJ_DEF_GROUPS.values()), {"DB", "LB", "DL"})


# ------------------------------------------------------------------ the registry and the frozen primary
class Registry(unittest.TestCase):
    def test_registered_opt_in_and_the_primary_and_serving_inputs_are_unchanged(self):
        self.assertEqual(COLS, ["opp_db_out_n", "opp_db_q_n", "opp_db_starter_out_n", "opp_lb_out_n", "opp_dl_out_n",
                                "opp_def_out_n", "opp_def_out_l4wk"])
        self.assertIn("opp_db_out", T.OPT_IN_FAMILIES)
        self.assertIn("opp_db_out", S.FAMILY_SOURCES)
        self.assertEqual((S.FAMILY_SOURCES["opp_db_out"].store_kwarg, S.FAMILY_SOURCES["opp_db_out"].table), ("defense", "depth_charts_def"))
        self.assertIn("opp_report_present", F.META)
        self.assertNotIn("opp_report_present", F.feature_columns())
        df = pd.DataFrame({c: [0.0, 1.0] for c in F.feature_columns()} | {"position": ["QB", "RB"]})
        prim = T.feature_columns(df, T.PRIMARY)
        self.assertEqual(set(prim) & set(COLS), set())
        both = T.feature_columns(df, T.Spec("x", add_families=("opp_db_out",)))
        self.assertTrue(set(COLS) <= set(both))
        self.assertEqual(set(both) - set(COLS), set(prim))
        self.assertEqual([c for c in COLS if "depth" in c], [])

    def test_a_store_without_the_tables_it_needs_is_all_nan(self):
        bare = RawStore([RawTable("injuries", _frame([], list(inj("x", "CB", "Out", KO))), "t")])
        f = _opp_def_features(bare, target(), None)
        self.assertEqual(f["opp_report_present"], 0)
        self.assertTrue(all(math.isnan(f[c]) for c in COLS))
        self.assertEqual(set(f) - {"opp_report_present"}, set(COLS))


if __name__ == "__main__":
    unittest.main()
