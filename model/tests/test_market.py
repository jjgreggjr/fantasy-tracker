"""The opt-in `market` family (Sleeper's archived Rotowire projections): the fetcher's parsing on a fixture response, the sleeper ->
gsis id map, the known_at rule (kickoff - 1h), value-only columns (no "has a projection" flag), the registry contract (the frozen
primary's inputs are untouched) and a leakage check on a synthetic store.

    python -m unittest model.tests.test_market

No network: the fixture is a trimmed API response and every store is built by hand.
"""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from model import features, train as T, trial as TR
from model import fetch_sleeper_proj as SP
from model.features import N_VALUES, PTS, XFP, Hist, _market_features
from model.point_in_time import KNOWN_AT, RawStore, RawTable, TargetRow, as_of_join

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sleeper_proj_2024_05.sample.json"
COLS = features.FAMILIES["market"]
KO = pd.Timestamp("2024-10-06 17:00", tz="UTC")             # KC's 2024 week-5 kickoff in the synthetic schedule
KO_LA = pd.Timestamp("2024-10-06 20:25", tz="UTC")


def fixture_frame() -> pd.DataFrame:
    return SP.parse_week(json.loads(FIXTURE.read_text()), 2024, 5)


def games() -> pd.DataFrame:
    return pd.DataFrame({"game_id": ["2024_05_NO_KC", "2024_05_GB_LA"], "season": [2024, 2024], "week": [5, 5],
                         "game_type": ["REG", "REG"], "home_team": ["KC", "LA"], "away_team": ["NO", "GB"],
                         "kickoff": pd.to_datetime([KO, KO_LA], utc=True).astype("datetime64[ns, UTC]")})


def id_map() -> pd.DataFrame:
    return pd.DataFrame({"sleeper_id": ["4046", "6794", "9999", "8888"],
                         "gsis_id": ["00-0033873", "00-0036900", "00-0099999", "00-0088888"],
                         "id_source": "test"})


# ------------------------------------------------------------------ the fetcher's parsing
class Parsing(unittest.TestCase):
    def test_rows_without_a_ppr_projection_are_dropped_and_columns_are_the_contract(self):
        df = fixture_frame()
        self.assertEqual(list(df.columns), SP.RAW_COLS)
        self.assertEqual(list(df["sleeper_id"]), ["4046", "6794", "8888", "7777"])       # 9999 has no stats.pts_ppr
        self.assertTrue((df["season"] == 2024).all() and (df["week"] == 5).all())
        self.assertEqual(set(df["company"]), {"rotowire"})

    def test_values_team_position_and_timestamps(self):
        df = fixture_frame().set_index("sleeper_id")
        qb, wr = df.loc["4046"], df.loc["6794"]
        self.assertEqual((qb["position"], qb["team"], qb["opponent"]), ("QB", "KC", "NO"))
        self.assertEqual((qb["pts_ppr"], qb["pass_att"], qb["pass_td"], qb["rush_att"]), (21.4, 36.1, 1.9, 4.2))
        self.assertTrue(math.isnan(qb["rec_tgt"]))                                        # omitted stat: NaN in the raw cache
        self.assertEqual(wr["team"], "LAR", "the row's team (projected FOR), not player.team (his current team)")
        self.assertEqual((wr["pts_ppr"], wr["pts_std"], wr["rec_tgt"], wr["rec_yd"]), (15.2, 9.6, 8.3, 71.0))
        self.assertEqual(qb["last_modified"], pd.Timestamp(1728604800000, unit="ms", tz="UTC"))
        self.assertTrue(pd.isna(wr["updated_at"]))
        self.assertEqual(str(fixture_frame()["last_modified"].dtype), "datetime64[ns, UTC]")

    def test_an_empty_response_is_an_empty_frame_with_the_contract(self):
        df = SP.parse_week([], 2021, 1)
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), SP.RAW_COLS)


# ------------------------------------------------------------------ id mapping
class IdMap(unittest.TestCase):
    def test_priority_order_and_validation(self):
        repo = pd.DataFrame({"sleeper_id": ["1", "2", None], "gsis_id": ["00-0000001", None, "00-0000009"]})
        dump = {"1": {"gsis_id": "00-0000111"}, "2": {"gsis_id": " 00-0000002 "}, "3": {"gsis_id": None}, "4": None,
                "DEF_KC": {"gsis_id": "00-0000004"}}
        dp = pd.DataFrame({"sleeper_id": ["3", "5", "2"], "gsis_id": ["00-0000003", "not-an-id", "00-0000222"]})
        m = SP.build_id_map(repo, dump, dp).set_index("sleeper_id")
        self.assertEqual(m.loc["1", "gsis_id"], "00-0000001")                     # data/players.csv wins
        self.assertEqual(m.loc["1", "id_source"], "data/players.csv")
        self.assertEqual(m.loc["2", "gsis_id"], "00-0000002")                     # repo had none -> Sleeper dump (stripped)
        self.assertEqual(m.loc["3", "gsis_id"], "00-0000003")                     # only DynastyProcess has it
        self.assertEqual(m.loc["3", "id_source"], "dynastyprocess")
        self.assertNotIn("5", m.index)                                            # malformed gsis id
        self.assertNotIn("DEF_KC", m.index)                                       # not a player id
        self.assertTrue(m.index.is_unique)

    def test_unmapped_rows_are_dropped_and_counted(self):
        df, dropped = SP.attach_known_at(fixture_frame(), games(), id_map())
        self.assertNotIn("7777", set(df["sleeper_id"]))
        self.assertEqual(dropped["no_gsis_id"], 1)
        self.assertEqual(dropped["not_skill_position"], 1)                        # the FB
        self.assertEqual(dict(zip(df["sleeper_id"], df["player_id"])), {"4046": "00-0033873", "6794": "00-0036900"})


# ------------------------------------------------------------------ known_at
class KnownAt(unittest.TestCase):
    def test_known_at_is_that_games_kickoff_minus_one_hour_never_the_sleeper_stamp(self):
        df, dropped = SP.attach_known_at(fixture_frame(), games(), id_map())
        self.assertEqual(SP.KNOWN_LEAD, pd.Timedelta(hours=1))
        k = df.set_index("sleeper_id")
        self.assertEqual(k.loc["4046", KNOWN_AT], KO - pd.Timedelta(hours=1))
        self.assertEqual(k.loc["6794", KNOWN_AT], KO_LA - pd.Timedelta(hours=1))   # LAR -> nflverse LA, the Rams' own kickoff
        self.assertEqual((k.loc["6794", "team"], k.loc["6794", "game_id"]), ("LA", "2024_05_GB_LA"))
        self.assertGreater(k.loc["4046", "last_modified"], KO, "the fixture's stamp is a post-game write, like the archive's")
        self.assertEqual(str(df[KNOWN_AT].dtype), "datetime64[ns, UTC]")
        RawTable("sleeper_proj", df, "test")                                         # the contract holds

    def test_a_team_without_a_regular_season_fixture_is_dropped_and_counted(self):
        g = games()
        g = g[g["home_team"] != "LA"]
        df, dropped = SP.attach_known_at(fixture_frame(), g, id_map())
        self.assertEqual(dropped["no_reg_fixture"], 1)
        self.assertEqual(list(df["sleeper_id"]), ["4046"])

    def test_pos_rank_is_within_season_week_position_over_every_projected_player(self):
        df, _ = SP.attach_known_at(fixture_frame(), games(), id_map())
        r = df.set_index("sleeper_id")["pos_rank"]
        self.assertEqual((r["4046"], r["6794"]), (1.0, 1.0))                       # 6794's 15.2 beats the unmapped WR's 6.0


# ------------------------------------------------------------------ registry: value only, opt-in, primary untouched
class Registry(unittest.TestCase):
    def test_market_is_registered_opt_in_and_the_primary_and_serving_inputs_are_unchanged(self):
        """Opt-in family, outside the Phase 2 primary; adopted into the serving spec on 2026-10-18 (its columns are in the served inputs, not the fallback)."""
        self.assertIn("market", features.FAMILIES)
        self.assertIn("market", T.OPT_IN_FAMILIES)
        self.assertEqual(len(COLS), 11)
        self.assertEqual(len(set(COLS)), 11)
        for c in COLS:
            self.assertEqual(features.family_of(c), "market")
        df = pd.DataFrame({c: [0.0, 1.0] for c in features.feature_columns()} | {"position": ["QB", "RB"],
                          "inj_report_status": [None, None], "inj_practice_status": [None, None], "wx_roof_obs": [None, None]})
        prim = T.feature_columns(df, T.PRIMARY)
        self.assertEqual(set(prim) & set(COLS), set())
        cfg = TR.load_config()
        serving = T.feature_columns(df, TR.serving_spec(cfg), dead=("college_breakout_age",))
        X, _ = T.encode(df, serving)
        self.assertEqual(set(X.columns), set(cfg["inputs"]))
        self.assertIn("market", TR.serving_spec(cfg).add_families, "market was adopted into the serving spec (2026-10-18)")
        self.assertTrue(set(COLS) <= set(cfg["inputs"]))
        self.assertFalse(set(COLS) & set(cfg["fallback"]["inputs"]), "the fallback is the spec without market")
        added = T.feature_columns(df, T.Spec("x", add_families=("market",)))
        self.assertEqual(set(added) - set(prim), set(COLS))

    def test_no_column_encodes_whether_a_projection_exists(self):
        for c in COLS:
            self.assertNotRegex(c, r"(has|have|is_|n_|count|cover|present|missing|avail|flag)", c)
        # with no row, every column is NaN: never 0, never a sentinel that could stand for 'absent'
        f = _market_features(store([proj_row(5, KO - pd.Timedelta(hours=1), 9.0, player_id="00-0000001")]), target(), hist([]), None)
        self.assertEqual(set(f), set(COLS))
        self.assertTrue(all(math.isnan(v) for v in f.values()))

    def test_trial_accepts_the_candidate(self):
        """market is already in the serving spec: the trial refuses to add it twice, drops it cleanly, and lists it as served."""
        base = TR.serving_spec(TR.load_config())
        with self.assertRaisesRegex(ValueError, "already added"):
            TR.compose(["+family:market"], base)
        v = TR.compose(["-family:market"], base)
        self.assertNotIn("market", v.spec.add_families)
        self.assertRegex(TR.list_families(TR.load_config()), r"market  \[11/11 serving, opt-in\]")


# ------------------------------------------------------------------ feature values and leakage on a synthetic store
def target(week: int = 5) -> TargetRow:
    return TargetRow("00-0033873", 2024, week, "KC", "NO", "QB", "2024_05_NO_KC", KO, True, ("usage",))


def hist(xfps: list[float], season: int = 2024) -> Hist:
    rows = []
    for i, x in enumerate(xfps, start=1):
        v = [0.0] * N_VALUES
        v[PTS], v[XFP] = 20.0, x
        rows.append((f"g{i}", season, i, True, True, "KC", *v))
    return Hist(rows)


def proj_row(week: int, known_at: pd.Timestamp, pts: float, **kw) -> dict:
    base = {"player_id": "00-0033873", "sleeper_id": "4046", "season": 2024, "week": week, "position": "QB", "team": "KC",
            "pts_ppr": pts, "pts_half_ppr": pts, "pts_std": pts - 1.0, "pass_att": 36.0, "pass_yd": 270.0, "pass_td": 1.9,
            "rush_att": 4.0, "rush_yd": 20.0, "rush_td": 0.1, "rec": np.nan, "rec_tgt": np.nan, "rec_yd": np.nan,
            "rec_td": np.nan, "pos_rank": 3.0, KNOWN_AT: known_at}
    base.update(kw)
    return base


def store(rows: list[dict]) -> RawStore:
    cols = list(proj_row(1, KO, 0.0))
    df = pd.DataFrame(rows, columns=cols)
    df[KNOWN_AT] = pd.to_datetime(df[KNOWN_AT], utc=True).astype("datetime64[ns, UTC]")
    for c in cols:
        if c not in ("player_id", "sleeper_id", "position", "team", KNOWN_AT):
            df[c] = df[c].astype("float64") if c not in ("season", "week") else df[c].astype("int64")
    return RawStore([RawTable("sleeper_proj", df, "test")])


class Values(unittest.TestCase):
    OWN = proj_row(5, KO - pd.Timedelta(hours=1), 21.4)

    def test_every_column_against_the_row(self):
        f = _market_features(store([self.OWN]), target(), hist([10.0, 14.0, 16.0, 18.0]), None)
        self.assertEqual(f["mkt_pts_ppr"], 21.4)
        self.assertEqual(f["mkt_pts_std"], 20.4)
        self.assertEqual((f["mkt_pass_att"], f["mkt_pass_yd"], f["mkt_carries"], f["mkt_rush_yd"]), (36.0, 270.0, 4.0, 20.0))
        self.assertEqual((f["mkt_targets"], f["mkt_rec_yd"]), (0.0, 0.0), "a stat Sleeper omits is a projected 0 when the row exists")
        self.assertTrue(abs(f["mkt_td_total"] - 2.0) < 1e-12)
        self.assertEqual(f["mkt_pos_rank"], 3.0)
        self.assertTrue(abs(f["mkt_vs_xfp_l3"] - (21.4 - 16.0)) < 1e-12)          # trailing-3 xFP = mean(14, 16, 18)

    def test_vs_xfp_is_nan_without_history_and_the_rest_still_present(self):
        f = _market_features(store([self.OWN]), target(), hist([]), None)
        self.assertTrue(math.isnan(f["mkt_vs_xfp_l3"]))
        self.assertEqual(f["mkt_pts_ppr"], 21.4)

    def test_a_store_without_the_table_is_all_nan(self):
        f = _market_features(RawStore([RawTable("other", pd.DataFrame({KNOWN_AT: pd.to_datetime([KO], utc=True)
                                                                         .astype("datetime64[ns, UTC]")}), "t")]),
                             target(), hist([10.0]), None)
        self.assertTrue(all(math.isnan(v) for v in f.values()))


class Leakage(unittest.TestCase):
    PAST = [proj_row(w, KO - pd.Timedelta(days=7 * (5 - w), hours=1), 10.0 + w) for w in (1, 2, 3, 4)]
    OWN = proj_row(5, KO - pd.Timedelta(hours=1), 21.4)

    def feats(self, rows, trace=None):
        return _market_features(store(rows), target(), hist([12.0]), trace)

    def test_the_own_games_row_is_read_only_because_it_is_stamped_before_kickoff(self):
        trace: list = []
        f = self.feats(self.PAST + [self.OWN], trace)
        self.assertEqual(f["mkt_pts_ppr"], 21.4)
        self.assertEqual(trace[0]["table"], "sleeper_proj")
        self.assertLess(trace[0]["max_known_at"], KO)
        # the same row stamped AT kickoff (or later) never enters: no market value at all for this game
        for late in (KO, KO + pd.Timedelta(hours=3), KO + pd.Timedelta(days=2)):
            g = self.feats(self.PAST + [proj_row(5, late, 99.0)])
            self.assertTrue(all(math.isnan(v) for v in g.values()), late)
        # 1 ns before kickoff it does
        self.assertEqual(self.feats(self.PAST + [proj_row(5, KO - pd.Timedelta(nanoseconds=1), 7.0)])["mkt_pts_ppr"], 7.0)

    def test_other_weeks_rows_never_stand_in_for_this_weeks(self):
        f = self.feats(self.PAST)                                                  # no week-5 row: earlier weeks are not used
        self.assertTrue(all(math.isnan(v) for v in f.values()))
        future = proj_row(6, KO + pd.Timedelta(days=7), 55.0)
        g = self.feats(self.PAST + [self.OWN, future])
        self.assertEqual(features.fingerprint(g), features.fingerprint(self.feats(self.PAST + [self.OWN])))

    def test_a_post_kickoff_correction_of_the_own_row_changes_nothing(self):
        corrected = proj_row(5, KO + pd.Timedelta(minutes=1), 40.0)
        a = self.feats(self.PAST + [self.OWN])
        b = self.feats(self.PAST + [self.OWN, corrected])
        self.assertEqual(features.fingerprint(a), features.fingerprint(b))

    def test_the_family_reads_raw_data_only_through_the_gate(self):
        import inspect
        src = inspect.getsource(_market_features)
        self.assertIn("as_of_join(store, \"sleeper_proj\", t.kickoff", src)
        self.assertNotIn("_tables", src)
        st = store(self.PAST + [self.OWN])
        self.assertEqual(len(as_of_join(st, "sleeper_proj", KO, player_id="00-0033873")), 5)


if __name__ == "__main__":
    unittest.main()
