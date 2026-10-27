"""The opt-in `dvp_split` family (model/features.py `_dvp_split_table` / `_dvp_split_features`; `ftn_def` from
model/point_in_time.py `load_ftn_tables`) on a synthetic store. No network, no cache.

    python -m unittest model.tests.test_dvp_split

  * the window: the defense's completed REG games this season before the result cutoff; l4 = the last four, std = all; the
    target game's own rows never enter; a game inside the cutoff is invisible, one just before it is read
  * per-game means allowed to each position (a game where the position scored nothing counts 0), the play-by-play ratios of sums,
    the FTN defense rates, the season-to-date ranks (1 = most allowed); l4 and std differ where the hand arithmetic says so
  * position conditioning: a row carries the stems in DVPS_POS for its position and NaN for the rest
  * week 1 is NaN; a store without pbp_team / ftn_def leaves those columns NaN and the rest intact
  * the trace: every read is at or before the cutoff and never holds the target game
  * registry: 31 columns, unique, opt-in, nothing shared with another family, FTN_D pinned to the loader
"""
from __future__ import annotations

import math
import unittest

import pandas as pd

from model import features as F
from model import point_in_time as pit
from model import train as T
from model.point_in_time import KNOWN_AT, RawStore, RawTable, TargetRow

SEASON = 2024
KICK0 = pd.Timestamp("2024-09-08 17:00", tz="UTC")
DEF = "BUF"
GAME6 = "2024_06_KC_BUF"


def ko(week: int, season: int = SEASON) -> pd.Timestamp:
    return KICK0 + pd.Timedelta(weeks=week - 1) - pd.Timedelta(weeks=52 * (SEASON - season))


def _ns(s) -> pd.Series:
    return pd.to_datetime(pd.Series(list(s)), utc=True).astype("datetime64[ns, UTC]")


def gid(week: int, team: str, season: int = SEASON) -> str:
    return f"{season}_{week:02d}_O{week}_{team}"


def pg_row(pid: str, pos: str, week: int, opp: str, *, game: str | None = None, stamp=None, season: int = SEASON, **stats) -> dict:
    r = {"player_id": pid, "position": pos, "season": season, "week": week, "season_type": "REG", "game_id": game or gid(week, opp, season),
         "team": f"O{week}", "opponent_team": opp, "receiving_yards": 0.0, "targets": 0.0, "receiving_tds": 0.0, "rushing_yards": 0.0,
         "carries": 0.0, "rushing_tds": 0.0, "passing_yards": 0.0, "passing_tds": 0.0, "sacks_suffered": 0.0, "fantasy_points_ppr": 0.0}
    r.update(stats)
    r[KNOWN_AT] = ko(week, season) if stamp is None else stamp
    return r


def gr_row(team: str, week: int, *, game: str | None = None, stamp=None, season: int = SEASON, game_type: str = "REG") -> dict:
    return {"game_id": game or gid(week, team, season), "season": season, "week": week, "game_type": game_type, "team": team,
            "opponent": f"O{week}", "points_for": 20, "points_against": 21, "is_home": True,
            KNOWN_AT: ko(week, season) if stamp is None else stamp}


def tm_row(opp: str, week: int, *, game: str | None = None, stamp=None, plays=60.0, dropbacks=30.0, pass_epa=0.0, rush_epa=0.0,
           explosive=0.0) -> dict:
    r = {"team": f"O{week}", "opponent": opp, "season": SEASON, "week": week, "game_type": "REG", "game_id": game or gid(week, opp),
         "plays": plays, "dropbacks": dropbacks}
    r.update({c: 0.0 for c in pit.PBP_SCRIPT_COUNTS})
    r.update(sc_pass_epa=pass_epa, sc_pass_n=30.0, sc_rush_epa=rush_epa, sc_rush_n=20.0, sc_explosive=explosive)
    r[KNOWN_AT] = ko(week) if stamp is None else stamp
    return r


def fd_row(team: str, week: int, *, stamp=None, **counts) -> dict:
    r = {"team": team, "season": SEASON, "week": week, "game_id": gid(week, team)}
    r.update({c: 0.0 for c in pit.FTN_DEF_COUNTS})
    r.update(counts)
    r[KNOWN_AT] = (ko(week) + pd.Timedelta(days=3)) if stamp is None else stamp          # the Wednesday release
    return r


def synthetic_store(*, pbp: bool = True, ftn: bool = True, extra_pg: list[dict] = (), extra_gr: list[dict] = (),
                    extra_tm: list[dict] = ()) -> RawStore:
    """BUF's defense has played weeks 1-5 of 2024 (week 6 vs KC is the target), one 2023 game and a week-5 playoff-coded game.
    What it allowed per week w: WR 10w receiving yards (w targets, a TD in week 5), RB 100-10w rushing yards (10 carries, w
    receiving yards), QB 200+w passing yards (w sacks, 1 TD), TE 10w receiving yards in odd weeks only. D1 allows WRs 100 a game
    (5 games), D2 allows them 5 (5 games); neither allows a rushing yard."""
    pg, gr, tm, fd = [], [], [], []
    for w in range(1, 6):
        gr.append(gr_row(DEF, w))
        pg += [pg_row(f"W{w}", "WR", w, DEF, receiving_yards=10.0 * w, targets=float(w), receiving_tds=1.0 if w == 5 else 0.0),
               pg_row(f"R{w}", "RB", w, DEF, rushing_yards=100.0 - 10.0 * w, carries=10.0, receiving_yards=float(w)),
               pg_row(f"Q{w}", "QB", w, DEF, passing_yards=200.0 + w, passing_tds=1.0, sacks_suffered=float(w))]
        if w % 2:
            pg.append(pg_row(f"T{w}", "TE", w, DEF, receiving_yards=10.0 * w))
        tm.append(tm_row(DEF, w, dropbacks=30.0 + w, pass_epa=float(w), rush_epa=-float(w), explosive=float(w)))
        fd.append(fd_row(DEF, w, car_box=10.0, car_light=float(w), db=30.0, db_blitz=float(w), db_rush_n=30.0, rushers_sum=120.0 + w))
        for d, yds in (("D1", 100.0), ("D2", 5.0)):
            gr.append(gr_row(d, w))
            pg.append(pg_row(f"X{d}{w}", "WR", w, d, receiving_yards=yds))
    # noise the window must ignore: last season, a playoff-coded game, and the target game itself (stamped at its kickoff)
    gr += [gr_row(DEF, 17, season=2023), gr_row(DEF, 5, game="2024_05_POST", game_type="POST")]
    pg += [pg_row("W0", "WR", 17, DEF, season=2023, receiving_yards=999.0), pg_row("W6", "WR", 6, DEF, game=GAME6, receiving_yards=999.0),
           pg_row("R6", "RB", 6, DEF, game=GAME6, rushing_yards=999.0)]
    gr.append(gr_row(DEF, 6, game=GAME6))
    tm.append(tm_row(DEF, 6, game=GAME6, plays=999.0, dropbacks=999.0, pass_epa=999.0, explosive=999.0))
    fd.append(fd_row(DEF, 6, stamp=ko(6) + pd.Timedelta(days=3), car_box=999.0, car_light=999.0))
    pg, gr, tm = pg + list(extra_pg), gr + list(extra_gr), tm + list(extra_tm)
    tabs = []
    for name, rows in (("player_games", pg), ("game_results", gr), ("pbp_team", tm if pbp else None), ("ftn_def", fd if ftn else None)):
        if rows is None:
            continue
        df = pd.DataFrame(rows)
        df[KNOWN_AT] = _ns(df[KNOWN_AT])
        tabs.append(RawTable(name, df, "t"))
    return RawStore(tabs)


def target(pos: str = "WR", week: int = 6, game: str = GAME6) -> TargetRow:
    return TargetRow("P1", SEASON, week, "KC", DEF, pos, game, ko(week), True)


def feats(st: RawStore, pos: str = "WR", trace=None, **kw) -> dict:
    return F._dvp_split_features(st, target(pos, **kw), trace)


def same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-12


# ------------------------------------------------------------------ the window and the arithmetic
class WindowMath(unittest.TestCase):
    def setUp(self):
        self.st = synthetic_store()

    def test_wr_receiving_l4_is_weeks_2_to_5_and_std_is_all_five(self):
        f = feats(self.st, "WR")
        self.assertTrue(same(f["dvps_rec_yd_l4"], (20 + 30 + 40 + 50) / 4))
        self.assertTrue(same(f["dvps_rec_yd_std"], (10 + 20 + 30 + 40 + 50) / 5))
        self.assertNotEqual(f["dvps_rec_yd_l4"], f["dvps_rec_yd_std"])
        self.assertTrue(same(f["dvps_rec_tgt_l4"], 14 / 4) and same(f["dvps_rec_tgt_std"], 15 / 5))
        self.assertTrue(same(f["dvps_rec_td_l4"], 1 / 4) and same(f["dvps_rec_td_std"], 1 / 5))

    def test_a_game_where_the_position_scored_nothing_counts_zero(self):
        f = feats(self.st, "TE")
        self.assertTrue(same(f["dvps_rec_yd_l4"], (0 + 30 + 0 + 50) / 4))
        self.assertTrue(same(f["dvps_rec_yd_std"], (10 + 0 + 30 + 0 + 50) / 5))

    def test_rb_rushing_and_receiving(self):
        f = feats(self.st, "RB")
        self.assertTrue(same(f["dvps_rush_yd_l4"], (80 + 70 + 60 + 50) / 4) and same(f["dvps_rush_yd_std"], 70.0))
        self.assertTrue(same(f["dvps_rush_att_l4"], 10.0) and same(f["dvps_rush_att_std"], 10.0))
        self.assertTrue(same(f["dvps_rec_yd_l4"], 14 / 4) and same(f["dvps_rec_yd_std"], 3.0))

    def test_qb_passing_and_sacks(self):
        f = feats(self.st, "QB")
        self.assertTrue(same(f["dvps_pass_yd_l4"], 200 + 14 / 4) and same(f["dvps_pass_yd_std"], 203.0))
        self.assertTrue(same(f["dvps_pass_td_std"], 1.0))
        self.assertTrue(same(f["dvps_sacks_l4"], 14 / 4) and same(f["dvps_sacks_std"], 3.0))
        self.assertTrue(same(f["dvps_rush_yd_std"], 0.0), "no QB ran: the position allowed 0 rushing yards, not NaN")

    def test_play_by_play_rates_are_ratios_of_sums_over_the_defenses_games(self):
        f = feats(self.st, "WR")
        self.assertTrue(same(f["dvps_epa_pass_l4"], (2 + 3 + 4 + 5) / 120) and same(f["dvps_epa_pass_std"], 15 / 150))
        self.assertTrue(same(f["dvps_pass_rate_faced_l4"], (32 + 33 + 34 + 35) / 240))
        self.assertTrue(same(f["dvps_pass_rate_faced_std"], (31 + 32 + 33 + 34 + 35) / 300))
        self.assertTrue(same(f["dvps_explosive_rate_l4"], 14 / 240) and same(f["dvps_explosive_rate_std"], 15 / 300))
        r = feats(self.st, "RB")
        self.assertTrue(same(r["dvps_epa_rush_l4"], -(2 + 3 + 4 + 5) / 80) and same(r["dvps_epa_rush_std"], -15 / 100))

    def test_ftn_defense_rates_over_the_last_four_charted_games(self):
        f = feats(self.st, "WR")
        self.assertTrue(same(f["dvps_blitz_rate_l4"], 14 / 120))
        self.assertTrue(same(f["dvps_avg_rushers_l4"], (480 + 14) / 120))
        self.assertTrue(same(feats(self.st, "RB")["dvps_light_box_rate_l4"], 14 / 40))

    def test_ranks_count_the_defenses_that_allowed_more_per_game(self):
        self.assertEqual(feats(self.st, "WR")["dvps_rank_rec_yd_std"], 2.0, "D1 (100 a game) allows more than BUF's 30; D2 less")
        r = feats(self.st, "RB")
        self.assertEqual(r["dvps_rank_rush_yd_std"], 1.0, "BUF is the only defense that allowed RBs a rushing yard")
        self.assertEqual(r["dvps_rank_rec_yd_std"], 1.0, "RB receiving: BUF 3 a game, D1 and D2 0")
        self.assertEqual(feats(self.st, "QB")["dvps_rank_rush_yd_std"], 1.0, "ties share the best rank (every defense 0)")


# ------------------------------------------------------------------ position conditioning
class PositionConditioning(unittest.TestCase):
    def test_each_position_carries_its_stems_and_nan_for_the_rest(self):
        st = synthetic_store()
        for pos, keep in F.DVPS_POS.items():
            f = feats(st, pos)
            for c, v in f.items():
                if F.DVPS_STEM[c] in keep:
                    self.assertFalse(math.isnan(v), f"{pos} {c}")
                else:
                    self.assertTrue(math.isnan(v), f"{pos} {c}")
        wr, qb, rb = feats(st, "WR"), feats(st, "QB"), feats(st, "RB")
        self.assertTrue(math.isnan(wr["dvps_rush_yd_l4"]) and math.isnan(wr["dvps_rank_rush_yd_std"]) and math.isnan(wr["dvps_light_box_rate_l4"]))
        self.assertTrue(math.isnan(qb["dvps_rec_yd_l4"]) and math.isnan(qb["dvps_rank_rec_yd_std"]))
        self.assertTrue(math.isnan(rb["dvps_blitz_rate_l4"]) and math.isnan(rb["dvps_pass_yd_l4"]))
        self.assertTrue(all(math.isnan(v) for v in feats(st, "K").values()), "an unknown position gets nothing")

    def test_the_stems_are_real_and_every_stem_belongs_to_some_position(self):
        stems = set(F.DVPS_STEM.values())
        for pos, keep in F.DVPS_POS.items():
            self.assertTrue(keep <= stems, pos)
        self.assertEqual(set().union(*F.DVPS_POS.values()), stems)


# ------------------------------------------------------------------ leakage and missing history
class Leakage(unittest.TestCase):
    def test_the_target_games_own_rows_never_enter(self):
        base = {pos: feats(synthetic_store(), pos) for pos in F.POSITIONS}
        st = synthetic_store()
        for name in ("player_games", "game_results", "pbp_team", "ftn_def"):
            df = st._tables[name].df
            st = st.with_frame(name, df[df["game_id"] != GAME6])
        for pos in F.POSITIONS:
            self.assertEqual(F.fingerprint(feats(st, pos)), F.fingerprint(base[pos]), pos)

    def test_a_game_inside_the_result_cutoff_is_invisible_and_one_just_before_it_is_read(self):
        base = feats(synthetic_store(), "WR")
        for stamp, seen in ((ko(6) - F.RESULT_LAG, False), (ko(6) - pd.Timedelta(hours=1), False),
                            (ko(6) - F.RESULT_LAG - pd.Timedelta(seconds=1), True)):
            st = synthetic_store(extra_pg=[pg_row("WL", "WR", 6, DEF, game="LATE", stamp=stamp, receiving_yards=400.0)],
                                 extra_gr=[gr_row(DEF, 6, game="LATE", stamp=stamp)],
                                 extra_tm=[tm_row(DEF, 6, game="LATE", stamp=stamp, pass_epa=50.0)])
            got = feats(st, "WR")
            self.assertEqual(F.fingerprint(base) != F.fingerprint(got), seen, stamp)
            if seen:
                self.assertTrue(same(got["dvps_rec_yd_l4"], (30 + 40 + 50 + 400) / 4))
                self.assertTrue(same(got["dvps_epa_pass_l4"], (3 + 4 + 5 + 50) / 120))

    def test_an_ftn_release_stamped_after_kickoff_is_invisible(self):
        st = synthetic_store()
        fd = st._tables["ftn_def"].df.copy()
        fd.loc[fd["week"] == 5, KNOWN_AT] = ko(6) + pd.Timedelta(seconds=1)        # the Thursday-game case: week 5 not yet out
        f = feats(st.with_frame("ftn_def", fd), "WR")
        self.assertTrue(same(f["dvps_blitz_rate_l4"], (1 + 2 + 3 + 4) / 120))

    def test_week_one_is_nan(self):
        st = synthetic_store()
        for pos in F.POSITIONS:
            f = feats(st, pos, week=1, game=f"2024_01_KC_{DEF}")
            self.assertTrue(all(math.isnan(v) for v in f.values()), pos)

    def test_a_store_without_pbp_or_ftn_leaves_those_columns_nan_and_the_rest_intact(self):
        f = feats(synthetic_store(pbp=False, ftn=False), "WR")
        for s in F.DVPS_PBP:
            self.assertTrue(math.isnan(f[f"dvps_{s}_l4"]) and math.isnan(f[f"dvps_{s}_std"]), s)
        self.assertTrue(math.isnan(f["dvps_blitz_rate_l4"]))
        self.assertTrue(same(f["dvps_rec_yd_std"], 30.0) and f["dvps_rank_rec_yd_std"] == 2.0)
        bare = F._dvp_split_features(RawStore([]), target(), None)
        self.assertTrue(all(math.isnan(v) for v in bare.values()))

    def test_a_defense_with_games_but_no_stat_rows_allowed_zero(self):
        st = synthetic_store()
        pg = st._tables["player_games"].df
        f = feats(st.with_frame("player_games", pg[pg["opponent_team"] != DEF]), "WR")
        self.assertTrue(same(f["dvps_rec_yd_std"], 0.0))
        self.assertEqual(f["dvps_rank_rec_yd_std"], 3.0)

    def test_the_trace_shows_only_finished_games_and_never_the_target(self):
        trace: list = []
        st = synthetic_store()
        feats(st, "WR", trace=trace)
        self.assertEqual({r["table"] for r in trace}, {"player_games", "game_results", "pbp_team", "ftn_def"})
        for r in trace:
            self.assertNotIn(GAME6, r["game_ids"], r["table"])
            if r["table"] == "ftn_def":
                self.assertLess(r["max_known_at"], ko(6))
            else:
                self.assertLessEqual(r["max_known_at"], ko(6) - F.RESULT_LAG)

    def test_memoised_per_game_and_defense_the_same_answer_as_an_uncached_build(self):
        st = synthetic_store()
        cached = feats(st, "WR")
        self.assertEqual(F.fingerprint(feats(st, "WR")), F.fingerprint(cached))
        self.assertEqual(F.fingerprint(feats(synthetic_store(), "WR", trace=[])), F.fingerprint(cached))


# ------------------------------------------------------------------ registry
class Registry(unittest.TestCase):
    def test_columns_are_the_registry_unique_opt_in_and_shared_with_no_family(self):
        cols = F.FAMILIES["dvp_split"]
        self.assertEqual(cols, F.DVPS_COLUMNS)
        self.assertEqual(len(cols), 31)
        self.assertEqual(len(cols), len(set(cols)))
        self.assertEqual(set(feats(synthetic_store())), set(cols))
        self.assertTrue(all(c.startswith("dvps_") for c in cols))
        others = {c for f, cs in F.FAMILIES.items() if f != "dvp_split" for c in cs}
        self.assertFalse(set(cols) & others)
        self.assertIn("dvp_split", T.OPT_IN_FAMILIES)
        for c in cols:
            self.assertEqual(F.family_of(c), "dvp_split")
        df = pd.DataFrame({c: [0.0, 1.0] for c in F.feature_columns()} | {"position": ["QB", "RB"]})
        prim = T.feature_columns(df, T.PRIMARY)
        self.assertFalse(set(prim) & set(cols))
        self.assertEqual(set(T.feature_columns(df, T.Spec("x", add_families=("dvp_split",)))) - set(prim), set(cols))

    def test_constants_repeated_from_the_loader_agree(self):
        self.assertEqual(F.FTN_D, pit.FTN_DEF_COUNTS)
        self.assertIn("sc_explosive", pit.PBP_SCRIPT_COUNTS)
        self.assertIn("yards_gained", pit.PBP_COLS)
        self.assertIn("defteam", pit.FTN_PBP_COLS)


if __name__ == "__main__":
    unittest.main()
