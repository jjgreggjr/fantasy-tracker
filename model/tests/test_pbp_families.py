"""Phase 2.5: the play-by-play tables and every new feature family behind the gate, proven the way Phase 0 and 1 proved theirs.

    model/.venv/bin/python -m unittest model.tests.test_pbp_families

Real data, seasons 2020-2026 (2020 is history for the prior-season columns), sampled across 2021 / 2022 / 2024 / 2025 / 2026 so
the season without participation (2026) and the first season with a prior-season anchor are exercised.

  * the raw tables reproduce nflverse's own player stats (targets, carries, target share), and every red-zone / inside-10 /
    air-yards / share value is recomputed from the raw play-by-play parquet with plain pandas (an independent path)
  * the new columns are identical from physically truncated tables, over a sample that exercises every family
  * canaries stamped at / after kickoff move no new column, canaries one second inside the result cutoff move every one that
    they touch to an exact expected value (so the tests above cannot pass vacuously), and the cutoff itself is pinned:
    result-table rows are read at kickoff - 4h, strictly
  * the game being predicted (perturbed or deleted) contributes nothing
  * the efficiency priors are the 2015-2019 numbers, the participation feed is a declared backtest proxy
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from model import audit, features, labels
from model import point_in_time as pit
from model import train as T
from model.features import RESULT_LAG, build_features, fingerprint
from model.point_in_time import KNOWN_AT, RawStore, as_of_join, make_target

PBP_TABLES = ("pbp_usage", "pbp_team", "pbp_part")
NEW_FAMILIES = ("pbp_usage", "pbp_team", "pbp_part", "eff")
NEW_COLS = [c for f in NEW_FAMILIES for c in features.FAMILIES[f]]
PLAN = {2021: 8, 2022: 8, 2024: 10, 2025: 14, 2026: 8}
CACHE = pit.CACHE_DIR / "nflverse"
_C: dict = {}


def store() -> RawStore:
    if "store" not in _C:
        _C["store"] = pit.load_store(range(2020, 2027))
    return _C["store"]


def rows() -> list[pit.TargetRow]:
    if "rows" not in _C:
        tgs = audit.sample_team_games(store(), PLAN, seed=20260930)
        _C["rows"] = audit.sample_spine_rows(store(), tgs)
    return _C["rows"]


def feats() -> pd.DataFrame:
    if "feats" not in _C:
        _C["feats"] = pd.DataFrame([build_features(store(), t) for t in rows()])
    return _C["feats"]


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def moved(a: dict, b: dict, cols) -> list[str]:
    return [c for c in cols if fingerprint({c: a[c]}) != fingerprint({c: b[c]})]


def pick(pos: str, season: int, week: int, *, exact: bool = False) -> pit.TargetRow:
    """A spine row with usage history at `pos` in (season, week), preferring a player with a full current-season history.
    `exact` also demands that every appearance be a stats-row game (no snap-only game), so a plain-pandas recomputation from
    the raw weekly stats sees the same games the store does."""
    best = None
    for tg in pit.team_games(store(), [season]):
        if tg.week != week:
            continue
        for t in features.spine_for(store(), tg):
            if t.position == pos and "usage" in t.src:
                f = build_features(store(), t)
                full = f["games_std"] >= min(week - 1, 3)
                if full and (not exact or f["games_std"] == len(_hand_appearances(t.player_id, season, week))):
                    return t
                best = best or t
    if best is None or exact:
        raise LookupError((pos, season, week))
    return best


# ------------------------------------------------------------------ the raw tables
class PbpTables(unittest.TestCase):
    def test_three_tables_each_with_a_rule_and_a_utc_known_at(self):
        for name in PBP_TABLES:
            t = store()._tables[name]
            self.assertTrue(t.known_at_rule, name)
            self.assertEqual(str(t.df[KNOWN_AT].dtype), "datetime64[ns, UTC]", name)
            self.assertFalse(t.df[KNOWN_AT].isna().any(), name)
            self.assertGreater(len(t.df), 10000 if name != "pbp_team" else 3000, name)

    def test_known_at_is_the_source_games_kickoff_for_every_row(self):
        ko = store()._tables["fixtures"].df.set_index("game_id")["kickoff"]
        for name in PBP_TABLES:
            df = store()._tables[name].df
            self.assertTrue((df[KNOWN_AT] == df["game_id"].map(ko)).all(), name)

    def test_the_participation_feed_is_a_declared_backtest_proxy_and_the_others_are_not(self):
        self.assertIn("NOT servable", store()._tables["pbp_part"].proxy)
        self.assertIsNone(store()._tables["pbp_usage"].proxy)
        self.assertIsNone(store()._tables["pbp_team"].proxy)

    def test_targets_carries_and_target_share_reproduce_nflverses_own_stats(self):
        pg = store()._tables["player_games"].df
        u = store()._tables["pbp_usage"].df
        tm = store()._tables["pbp_team"].df.set_index(["game_id", "team"])
        m = pg.merge(u[["player_id", "game_id", "tgt", "car"]], on=["player_id", "game_id"], how="left")
        for season in range(2021, 2027):
            g = m[(m["season"] == season) & (m["season_type"] == "REG")]
            self.assertGreater(len(g), 1000)
            self.assertEqual(float((g["targets"].fillna(0) != g["tgt"].fillna(0)).mean()), 0.0, f"targets {season}")
            self.assertLess(float((g["carries"].fillna(0) != g["car"].fillna(0)).mean()), 0.001, f"carries {season}")
        g = m[m["season"] >= 2021]
        T_tgt = pd.Series([tm["tgt"].get((a, b), np.nan) for a, b in zip(g["game_id"], g["team"])], index=g.index)
        ok = T_tgt > 0
        self.assertLess(float((g.loc[ok, "tgt"].fillna(0) / T_tgt[ok] - g.loc[ok, "target_share"].fillna(0)).abs().max()), 1e-9)

    def test_red_zone_inside_ten_and_air_yards_recomputed_from_the_raw_parquet(self):
        for season, week, team in ((2024, 10, "PHI"), (2022, 5, "KC"), (2025, 12, "LA")):
            p = pd.read_parquet(CACHE / f"play_by_play_{season}.parquet")
            g = p[(p["season_type"] == "REG") & (p["week"] == week) & (p["posteam"] == team)
                  & p["play_type"].isin(["pass", "run", "qb_kneel"]) & (p["two_point_attempt"].fillna(0) != 1)]
            tg = g[(g["play_type"] == "pass") & (g["pass_attempt"] == 1) & g["receiver_player_id"].notna()]
            cr = g[(g["rush_attempt"] == 1) & g["rusher_player_id"].notna()]
            row = store()._tables["pbp_team"].df
            row = row[(row["season"] == season) & (row["week"] == week) & (row["team"] == team)].iloc[0]
            self.assertEqual(row["tgt"], len(tg))
            self.assertEqual(row["rz_tgt"], int((tg["yardline_100"] <= 20).sum()))
            self.assertEqual(row["i10_tgt"], int((tg["yardline_100"] <= 10).sum()))
            self.assertEqual(row["ay"], float(tg["air_yards"].fillna(0).sum()))
            self.assertEqual(row["car"], len(cr))
            self.assertEqual(row["rz_car"], int((cr["yardline_100"] <= 20).sum()))
            self.assertEqual(row["i10_car"], int((cr["yardline_100"] <= 10).sum()))
            scrim = g[g["play_type"].isin(["pass", "run"])]
            self.assertEqual(row["plays"], len(scrim))
            self.assertEqual(row["dropbacks"], int((scrim["qb_dropback"] == 1).sum()))
            # one player, by the same independent route
            top = tg["receiver_player_id"].value_counts().index[0]
            u = store()._tables["pbp_usage"].df
            r = u[(u["player_id"] == top) & (u["game_id"] == row["game_id"])].iloc[0]
            mine = tg[tg["receiver_player_id"] == top]
            self.assertEqual((r["tgt"], r["rz_tgt"], r["i10_tgt"], r["ay"], r["ay_n"]),
                             (len(mine), int((mine["yardline_100"] <= 20).sum()), int((mine["yardline_100"] <= 10).sum()),
                              float(mine["air_yards"].fillna(0).sum()), int(mine["air_yards"].notna().sum())))

    def test_two_point_tries_and_defensive_or_special_teams_plays_are_not_counted(self):
        p = pd.read_parquet(CACHE / "play_by_play_2024.parquet")
        two = p[(p["two_point_attempt"] == 1) & (p["season_type"] == "REG")]
        self.assertGreater(len(two), 50)
        n_team_plays = store()._tables["pbp_team"].df
        n_team_plays = n_team_plays[(n_team_plays["season"] == 2024) & (n_team_plays["game_type"] == "REG")]["plays"].sum()
        scrim = p[(p["season_type"] == "REG") & p["play_type"].isin(["pass", "run"]) & (p["two_point_attempt"].fillna(0) != 1)
                  & p["posteam"].notna()]
        self.assertEqual(int(n_team_plays), len(scrim))

    def test_participation_covers_every_regular_season_scrimmage_play_2020_to_2025_and_zero_where_absent(self):
        tm = store()._tables["pbp_team"].df
        tm = tm[tm["game_type"] == "REG"]
        old = tm[tm["season"] <= 2025]
        self.assertTrue((old["part_dropbacks"] == old["dropbacks"]).all())
        self.assertTrue((old["part_rush"] == old["plays"] - old["dropbacks"]).all())
        self.assertTrue((tm["part_dropbacks"] <= tm["dropbacks"]).all())
        self.assertEqual(int(store()._tables["pbp_part"].df["season"].min()), 2020)

    def test_a_missing_release_asset_is_remembered_and_never_an_error(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "x.parquet"
            resp = mock.Mock(status_code=404)
            with mock.patch("model.point_in_time.requests.get", return_value=resp) as g:
                self.assertIsNone(pit._fetch_optional("http://x", dest))
                self.assertIsNone(pit._fetch_optional("http://x", dest))
                self.assertEqual(g.call_count, 1, "the 404 is remembered")
            boom = mock.Mock(status_code=500, raise_for_status=mock.Mock(side_effect=pit.requests.HTTPError("500")))
            with mock.patch("model.point_in_time.requests.get", return_value=boom):
                with self.assertRaises(pit.requests.HTTPError):
                    pit._fetch_optional("http://y", Path(d) / "y.parquet")


# ------------------------------------------------------------------ independent hand computation
def _hand_appearances(pid: str, season: int, week: int) -> list[tuple[int, str, str]]:
    """(week, game_id, team) of his stats-row appearances this season before `week`, from the raw weekly stats file."""
    s = pd.read_parquet(CACHE / f"stats_player_week_{season}.parquet", columns=["player_id", "week", "game_id", "team", "season_type"])
    s = s[(s["player_id"] == pid) & (s["season_type"] == "REG") & (s["week"] < week)].sort_values("week")
    return list(zip(s["week"], s["game_id"], s["team"]))


def _hand_usage(pid: str, season: int, games: list[tuple[int, str, str]]) -> dict:
    """The pbp_usage stats over `games`, in plain pandas from the raw play-by-play parquet."""
    p = pd.read_parquet(CACHE / f"play_by_play_{season}.parquet")
    p = p[(p["season_type"] == "REG") & p["play_type"].isin(["pass", "run", "qb_kneel"]) & (p["two_point_attempt"].fillna(0) != 1)]
    keys = ["tgt", "rz_tgt", "i10_tgt", "ay", "ay_n", "car", "rz_car", "i10_car", "T_tgt", "T_rz_tgt", "T_i10_tgt", "T_ay", "T_car",
            "T_rz_car", "T_i10_car"]
    s = dict.fromkeys(keys, 0.0)
    rz_opps = i10_opps = 0.0
    for _, gid, team in games:
        g = p[(p["game_id"] == gid) & (p["posteam"] == team)]
        t = g[(g["play_type"] == "pass") & (g["pass_attempt"] == 1) & g["receiver_player_id"].notna()]
        c = g[(g["rush_attempt"] == 1) & g["rusher_player_id"].notna()]
        mt, mc = t[t["receiver_player_id"] == pid], c[c["rusher_player_id"] == pid]
        add = {"tgt": len(mt), "rz_tgt": (mt["yardline_100"] <= 20).sum(), "i10_tgt": (mt["yardline_100"] <= 10).sum(),
               "ay": mt["air_yards"].fillna(0).sum(), "ay_n": mt["air_yards"].notna().sum(), "car": len(mc),
               "rz_car": (mc["yardline_100"] <= 20).sum(), "i10_car": (mc["yardline_100"] <= 10).sum(),
               "T_tgt": len(t), "T_rz_tgt": (t["yardline_100"] <= 20).sum(), "T_i10_tgt": (t["yardline_100"] <= 10).sum(),
               "T_ay": t["air_yards"].fillna(0).sum(), "T_car": len(c), "T_rz_car": (c["yardline_100"] <= 20).sum(),
               "T_i10_car": (c["yardline_100"] <= 10).sum()}
        for k, v in add.items():
            s[k] += float(v)
        rz_opps += float((mt["yardline_100"] <= 20).sum() + (mc["yardline_100"] <= 20).sum())
        i10_opps += float((mt["yardline_100"] <= 10).sum() + (mc["yardline_100"] <= 10).sum())
    r = lambda a, b: a / b if b > 0 else float("nan")    # noqa: E731
    tsh, ash = r(s["tgt"], s["T_tgt"]), r(s["ay"], s["T_ay"])
    n = len(games)
    return {"rz_tgt_share": r(s["rz_tgt"], s["T_rz_tgt"]), "i10_tgt_share": r(s["i10_tgt"], s["T_i10_tgt"]),
            "rz_car_share": r(s["rz_car"], s["T_rz_car"]), "i10_car_share": r(s["i10_car"], s["T_i10_car"]),
            "ay_share": ash, "adot": r(s["ay"], s["ay_n"]),
            "wopr": 1.5 * tsh + 0.7 * ash if not (math.isnan(tsh) or math.isnan(ash)) else float("nan"),
            "rz_opps_pg": rz_opps / n if n else float("nan"), "i10_opps_pg": i10_opps / n if n else float("nan")}


def _same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-12


class FeaturesMatchAnIndependentPandasComputation(unittest.TestCase):
    def check_usage(self, t: pit.TargetRow):
        f = build_features(store(), t)
        apps = _hand_appearances(t.player_id, t.season, t.week)
        # the hand path only covers stats-row appearances: a snap-only game would add a zero-usage game (and its team totals)
        self.assertEqual(f["games_std"], len(apps), f"{t.player_id} has snap-only appearances: pick another row")
        for w, games in (("l1", apps[-1:]), ("t3", apps[-3:]), ("std", apps)):
            want = _hand_usage(t.player_id, t.season, games)
            for k, v in want.items():
                self.assertTrue(_same(f[f"pbp_{k}_{w}"], v), f"{t.player_id} {t.season} wk{t.week} pbp_{k}_{w}: {f[f'pbp_{k}_{w}']} vs {v}")

    def test_usage_windows_for_a_running_back_a_receiver_and_a_tight_end(self):
        for pos, season, week in (("RB", 2024, 10), ("WR", 2025, 7), ("TE", 2022, 8)):
            self.check_usage(pick(pos, season, week, exact=True))

    def test_prior_season_columns_are_last_seasons_regular_season_appearances(self):
        t = pick("WR", 2025, 7, exact=True)
        f = build_features(store(), t)
        prev = pd.read_parquet(CACHE / f"stats_player_week_{t.season - 1}.parquet", columns=["player_id", "week", "game_id", "team", "season_type"])
        prev = prev[(prev["player_id"] == t.player_id) & (prev["season_type"] == "REG")].sort_values("week")
        games = list(zip(prev["week"], prev["game_id"], prev["team"]))
        if not games:
            self.skipTest("no prior season for the picked player")
        want = _hand_usage(t.player_id, t.season - 1, games)
        for k in features.PBP_PREV_STATS:
            self.assertTrue(_same(f[f"pbp_prev_{k}"], want[k]), k)

    def test_team_pace_and_dropback_rates_from_the_raw_parquet(self):
        t = pick("RB", 2024, 10)
        f = build_features(store(), t)
        p = pd.read_parquet(CACHE / "play_by_play_2024.parquet")
        p = p[(p["season_type"] == "REG") & p["play_type"].isin(["pass", "run"]) & (p["two_point_attempt"].fillna(0) != 1)]

        def rates(team: str, opp: bool = False, weeks=range(1, 10)):
            g = p[(p["defteam"] if opp else p["posteam"]) == team]
            g = g[g["week"].isin(list(weeks))]
            plays = g.groupby("game_id").size()
            db = g[g["qb_dropback"] == 1].groupby("game_id").size().reindex(plays.index).fillna(0)
            nt = g[g["wp"].between(0.2, 0.8) & g["down"].isin([1, 2]) & g["qtr"].isin([1, 2, 3]) & (g["half_seconds_remaining"] > 120)]
            nt_p = nt.groupby("game_id").size().reindex(plays.index).fillna(0)
            nt_d = nt[nt["qb_dropback"] == 1].groupby("game_id").size().reindex(plays.index).fillna(0)
            return plays, db, nt_p, nt_d
        plays, db, nt_p, nt_d = rates(t.team)
        my_weeks = sorted(p[(p["posteam"] == t.team) & (p["week"] < 10)]["week"].unique())
        self.assertAlmostEqual(f["team_plays_pg_std"], plays.mean(), places=12)
        self.assertAlmostEqual(f["team_pass_rate_std"], db.sum() / plays.sum(), places=12)
        self.assertAlmostEqual(f["team_neutral_pass_rate_std"], nt_d.sum() / nt_p.sum(), places=12)
        last3 = p[(p["posteam"] == t.team) & (p["week"].isin(my_weeks[-3:]))]
        plays3 = last3.groupby("game_id").size()
        self.assertAlmostEqual(f["team_plays_pg_t3"], plays3.mean(), places=12)
        fplays, fdb, _, _ = rates(t.opponent, opp=True)
        self.assertAlmostEqual(f["opp_plays_faced_pg_std"], fplays.mean(), places=12)
        self.assertAlmostEqual(f["opp_pass_rate_faced_std"], fdb.sum() / fplays.sum(), places=12)

    def test_participation_shares_from_the_raw_feeds(self):
        t = pick("WR", 2024, 9, exact=True)
        f = build_features(store(), t)
        apps = _hand_appearances(t.player_id, t.season, t.week)
        self.assertEqual(f["games_std"], len(apps))
        p = pd.read_parquet(CACHE / "play_by_play_2024.parquet", columns=["game_id", "play_id", "posteam", "play_type", "qb_dropback",
                                                                       "two_point_attempt", "season_type", "yardline_100"])
        d = pd.read_parquet(CACHE / "pbp_participation_2024.parquet", columns=["nflverse_game_id", "play_id", "offense_players"])
        m = p[(p["season_type"] == "REG") & p["play_type"].isin(["pass", "run"]) & (p["two_point_attempt"].fillna(0) != 1)].merge(
            d, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "play_id"])
        on_pass = on_run = tot_pass = tot_run = 0.0
        for _, gid, team in apps:
            g = m[(m["game_id"] == gid) & (m["posteam"] == team)]
            mine = g["offense_players"].str.contains(t.player_id)
            db = g["qb_dropback"] == 1
            on_pass += float((mine & db).sum())
            on_run += float((mine & ~db).sum())
            tot_pass += float(db.sum())
            tot_run += float((~db).sum())
        self.assertAlmostEqual(f["part_pass_snap_share_std"], on_pass / tot_pass, places=12)
        self.assertAlmostEqual(f["part_run_snap_share_std"], on_run / tot_run, places=12)

    def test_the_2026_season_has_no_participation_and_says_so_with_nan(self):
        t = pick("WR", 2026, 3)
        f = build_features(store(), t)
        self.assertTrue(all(math.isnan(f[c]) for c in features.FAMILIES["pbp_part"] if "prev" not in c))
        self.assertFalse(math.isnan(f["pbp_ay_share_std"]))           # play-by-play itself is there

    def test_week_one_has_no_current_season_window_but_a_prior_season_anchor(self):
        t = pick("WR", 2025, 1)
        f = build_features(store(), t)
        for w in ("l1", "t3", "std"):
            for k in features.PBP_USAGE_STATS:
                self.assertTrue(math.isnan(f[f"pbp_{k}_{w}"]), (k, w))
        self.assertTrue(all(math.isnan(f[c]) for c in ("team_plays_pg_t3", "team_plays_pg_std", "opp_plays_faced_pg_std")))
        self.assertFalse(math.isnan(f["team_plays_pg_prev"]))
        self.assertFalse(math.isnan(f["pbp_prev_wopr"]))

    def test_efficiency_from_raw_stats_with_the_documented_priors(self):
        t = pick("WR", 2024, 12)
        f = build_features(store(), t)
        s = pd.concat([pd.read_parquet(CACHE / f"stats_player_week_{y}.parquet") for y in range(2020, 2025)])
        s = s[(s["player_id"] == t.player_id) & (s["season_type"] == "REG") & ((s["season"] < 2024) | (s["week"] < 12))]
        pri, k = features.EFF_PRIORS[t.position], features.EFF_K
        career = lambda num, den: (s[num].sum() + k[den[0]] * pri[den[0]]) / (s[den[1]].sum() + k[den[0]])   # noqa: E731
        self.assertAlmostEqual(f["eff_catch_rate_career"], career("receptions", ("catch_rate", "targets")), places=12)
        self.assertAlmostEqual(f["eff_rec_ypt_career"], career("receiving_yards", ("rec_ypt", "targets")), places=12)
        self.assertAlmostEqual(f["eff_rec_td_rate_career"], career("receiving_tds", ("rec_td_rate", "targets")), places=12)
        cur = s[s["season"] == 2024]
        self.assertAlmostEqual(f["eff_rec_ypt_std"], (cur["receiving_yards"].sum() + k["rec_ypt"] * pri["rec_ypt"]) / (cur["targets"].sum() + k["rec_ypt"]), places=12)
        self.assertTrue(math.isnan(f["eff_pass_ypa_career"]))          # a receiver with no pass attempts: no denominator, NaN

    def test_a_quarterbacks_passing_rates_are_shrunk_toward_the_qb_prior(self):
        t = pick("QB", 2024, 10)
        f = build_features(store(), t)
        s = pd.concat([pd.read_parquet(CACHE / f"stats_player_week_{y}.parquet") for y in range(2020, 2025)])
        s = s[(s["player_id"] == t.player_id) & (s["season_type"] == "REG") & ((s["season"] < 2024) | (s["week"] < 10))]
        want = (s["passing_yards"].sum() + 150 * 7.196) / (s["attempts"].sum() + 150)
        self.assertAlmostEqual(f["eff_pass_ypa_career"], want, places=12)

    def test_the_efficiency_priors_are_the_2015_2019_regular_season_numbers(self):
        """Fixed constants known before the first backtest season. Recomputed here so a typo (or a refit on later data) fails."""
        fr = []
        for y in range(2015, 2020):
            f = CACHE / f"stats_player_week_{y}.parquet"
            if not f.exists():
                pit._fetch(f"{pit.NFLVERSE}/stats_player/stats_player_week_{y}.parquet", f)
            d = pd.read_parquet(f)
            fr.append(d[(d["season_type"] == "REG") & d["position"].isin(pit.SKILL_POSITIONS)])
        d = pd.concat(fr)
        for pos, g in d.groupby("position"):
            pri = features.EFF_PRIORS[pos]
            if pos != "QB":
                self.assertAlmostEqual(pri["catch_rate"], g["receptions"].sum() / g["targets"].sum(), delta=0.0006)
                self.assertAlmostEqual(pri["rec_ypt"], g["receiving_yards"].sum() / g["targets"].sum(), delta=0.0006)
                self.assertAlmostEqual(pri["rec_td_rate"], g["receiving_tds"].sum() / g["targets"].sum(), delta=0.0006)
            if pos in ("QB", "RB", "WR"):
                self.assertAlmostEqual(pri["rush_ypc"], g["rushing_yards"].sum() / g["carries"].sum(), delta=0.0006)
                self.assertAlmostEqual(pri["rush_td_rate"], g["rushing_tds"].sum() / g["carries"].sum(), delta=0.0006)
            if pos == "QB":
                self.assertAlmostEqual(pri["pass_ypa"], g["passing_yards"].sum() / g["attempts"].sum(), delta=0.0006)
                self.assertAlmostEqual(pri["pass_td_rate"], g["passing_tds"].sum() / g["attempts"].sum(), delta=0.0006)
                self.assertAlmostEqual(pri["pass_int_rate"], g["passing_interceptions"].sum() / g["attempts"].sum(), delta=0.0006)
                self.assertAlmostEqual(pri["pass_cmp_rate"], g["completions"].sum() / g["attempts"].sum(), delta=0.0006)
        # the two combinations with too little data to have their own prior borrow a documented neighbour
        self.assertEqual(features.EFF_PRIORS["QB"]["rec_ypt"], features.EFF_PRIORS["TE"]["rec_ypt"])
        self.assertEqual(features.EFF_PRIORS["TE"]["rush_ypc"], features.EFF_PRIORS["WR"]["rush_ypc"])
        for pos in pit.SKILL_POSITIONS:
            self.assertEqual(set(features.EFF_PRIORS[pos]), set(features.EFF_K))


# ------------------------------------------------------------------ leakage: truncation audit
class NewFamiliesTruncationAudit(unittest.TestCase):
    def test_the_sample_exercises_every_new_family(self):
        f = feats()
        self.assertGreater(len(f), 800)
        for col in ("pbp_rz_tgt_share_std", "pbp_i10_car_share_std", "pbp_ay_share_l1", "pbp_adot_t3", "pbp_wopr_std",
                    "pbp_rz_opps_pg_t3", "pbp_prev_ay_share", "team_plays_pg_t3", "team_neutral_pass_rate_std",
                    "team_pass_rate_prev", "opp_plays_faced_pg_std", "part_pass_snap_share_l1", "part_rz_snap_share_std",
                    "part_prev_pass_snap_share", "eff_rec_ypt_career", "eff_rush_ypc_std", "eff_pass_ypa_career"):
            self.assertGreater(int(f[col].notna().sum()), 150, col)
        self.assertEqual({int(s) for s in f["season"]}, set(PLAN))
        self.assertTrue(f.loc[f["season"] == 2026, [c for c in features.FAMILIES["pbp_part"] if "prev" not in c]].isna().all().all())

    def test_every_new_column_is_identical_from_physically_truncated_tables(self):
        cols = set(NEW_COLS)

        def builder(st: RawStore, t: pit.TargetRow) -> dict:
            f = build_features(st, t)
            return {k: v for k, v in f.items() if k in cols}
        bad = audit.audit_truncation(builder, store(), rows())
        self.assertEqual(bad, [], f"{len(bad)} rows changed when raw tables were cut to known_at < kickoff: {bad[:2]}")

    def test_the_tables_really_are_cut_by_the_audit(self):
        t = rows()[len(rows()) // 2]
        cut = store().truncated_before(t.kickoff)
        for name in PBP_TABLES:
            df = cut._tables[name].df
            self.assertLess(df[KNOWN_AT].max(), t.kickoff)
            self.assertLess(len(df), len(store()._tables[name].df))


# ------------------------------------------------------------------ leakage: canaries
def canary_rows() -> list[pit.TargetRow]:
    return [pick("RB", 2025, 6), pick("WR", 2025, 12), pick("TE", 2022, 9), pick("QB", 2024, 5), pick("RB", 2021, 8),
            pick("WR", 2026, 3)]


class NewFamilyCanaries(unittest.TestCase):
    def test_nothing_stamped_at_or_after_kickoff_changes_any_new_column(self):
        for t in canary_rows():
            base = build_features(store(), t)
            for stamp in (t.kickoff + pd.Timedelta(hours=1), t.kickoff + pd.Timedelta(days=7), t.kickoff):
                for offset in (1, -1):
                    dirty = build_features(audit.inject_canaries(store(), t, [stamp], week_offset=offset), t)
                    self.assertEqual(moved(base, dirty, NEW_COLS), [], f"{t.player_id} {t.season} wk{t.week} {stamp} offset {offset}")

    def test_control_one_second_inside_the_cutoff_moves_every_column_the_canary_touches_to_its_exact_value(self):
        """Without this the test above could pass vacuously (a gate that drops everything). The canary game is the player's
        newest appearance (l1) and the team's newest game; its counts are chosen so every ratio has a known value."""
        u, tt, pa = audit.PBP_CANARY_USAGE, audit.PBP_CANARY_TEAM, audit.PBP_CANARY_PART
        want_l1 = {"pbp_rz_tgt_share_l1": u["rz_tgt"] / tt["rz_tgt"], "pbp_i10_tgt_share_l1": u["i10_tgt"] / tt["i10_tgt"],
                   "pbp_rz_car_share_l1": u["rz_car"] / tt["rz_car"], "pbp_i10_car_share_l1": u["i10_car"] / tt["i10_car"],
                   "pbp_ay_share_l1": u["ay"] / tt["ay"], "pbp_adot_l1": u["ay"] / u["ay_n"],
                   "pbp_wopr_l1": 1.5 * u["tgt"] / tt["tgt"] + 0.7 * u["ay"] / tt["ay"],
                   "pbp_rz_opps_pg_l1": u["rz_tgt"] + u["rz_car"], "pbp_i10_opps_pg_l1": u["i10_tgt"] + u["i10_car"],
                   "part_pass_snap_share_l1": pa["pass_on"] / tt["part_dropbacks"],
                   "part_run_snap_share_l1": pa["run_on"] / tt["part_rush"], "part_rz_snap_share_l1": pa["rz_on"] / tt["part_rz"]}
        for t in canary_rows():
            base = build_features(store(), t)
            live = build_features(audit.inject_canaries(store(), t, [t.kickoff - pd.Timedelta(seconds=1)], week_offset=-1,
                                                        result_lag=RESULT_LAG), t)
            for c, v in want_l1.items():
                self.assertAlmostEqual(live[c], v, places=12, msg=f"{t.player_id} {t.season} wk{t.week} {c}")
            missed = [c for c in NEW_COLS if "prev" not in c and c not in audit.diff_keys(base, live)]
            self.assertEqual(missed, [], f"{t.player_id} {t.season} wk{t.week}: canary did not move {missed}")
            self.assertEqual(moved(base, live, [c for c in NEW_COLS if "prev" in c]), [], "last season's columns cannot see this season's game")
            self.assertNotEqual(live["team_plays_pg_t3"], base["team_plays_pg_t3"])
            self.assertNotEqual(live["opp_plays_faced_pg_std"], base["opp_plays_faced_pg_std"])
            self.assertNotEqual(live["eff_rec_ypt_std"], base["eff_rec_ypt_std"])

    def test_the_result_cutoff_is_kickoff_minus_four_hours_strictly_for_the_new_tables_too(self):
        t = pick("RB", 2025, 6)
        base = build_features(store(), t)
        at = audit.inject_canaries(store(), t, [t.kickoff - RESULT_LAG], week_offset=-1)
        self.assertEqual(moved(base, build_features(at, t), NEW_COLS), [], "a row stamped exactly at kickoff - 4h must be hidden")
        inside = audit.inject_canaries(store(), t, [t.kickoff - RESULT_LAG - pd.Timedelta(nanoseconds=1)], week_offset=-1)
        self.assertAlmostEqual(build_features(inside, t)["pbp_rz_tgt_share_l1"], 12 / 40, places=12)

    def test_each_new_table_is_gated_on_its_own(self):
        """A share needs a usage row AND the team's totals, so a leak in ONE table would be masked by the other's correct gating
        (a future usage row without its team row is simply excluded). This stamps the canary game correctly in every table but
        one, which is stamped an hour AFTER kickoff, and shows that table's rows stay invisible on their own."""
        for t in canary_rows()[:4]:
            base = build_features(store(), t)
            ok = audit.inject_canaries(store(), t, [t.kickoff - pd.Timedelta(seconds=1)], week_offset=-1, result_lag=RESULT_LAG)
            late = t.kickoff + pd.Timedelta(hours=1)

            def hide(name: str) -> RawStore:
                df = ok._tables[name].df.copy()
                df.loc[df["game_id"].str.contains("CANARY"), KNOWN_AT] = late
                return ok.with_frame(name, df)
            usage = build_features(hide("pbp_usage"), t)
            self.assertEqual(usage["pbp_rz_tgt_share_l1"], 0.0, "the game is visible, its usage row is not: he had no usage")
            self.assertEqual(usage["pbp_rz_opps_pg_l1"], 0.0)
            self.assertTrue(math.isnan(usage["pbp_adot_l1"]))
            team = build_features(hide("pbp_team"), t)
            self.assertTrue(all(math.isnan(team[f"pbp_{k}_l1"]) for k in features.PBP_USAGE_STATS), "no team totals: nothing to divide by")
            self.assertEqual(moved(base, team, features.FAMILIES["pbp_team"]), [], "the team canary rows are invisible")
            part = build_features(hide("pbp_part"), t)
            self.assertEqual(part["part_pass_snap_share_l1"], 0.0, "covered game, no participation row: on the field for nothing")
            self.assertAlmostEqual(part["pbp_rz_tgt_share_l1"], 12 / 40, places=12, msg="usage is untouched by the participation table")

    def test_a_game_with_no_team_totals_is_skipped_by_every_window_never_read_as_zero(self):
        t = pick("RB", 2025, 6)
        full = audit.inject_canaries(store(), t, [t.kickoff - RESULT_LAG - pd.Timedelta(seconds=1)], week_offset=-1)
        tm = full._tables["pbp_team"].df
        dirty = full.with_frame("pbp_team", tm[~tm["game_id"].str.contains("CANARY")])
        f = build_features(dirty, t)
        for k in features.PBP_USAGE_STATS:
            self.assertTrue(math.isnan(f[f"pbp_{k}_l1"]), k)          # his newest game has no team totals: nothing to divide by
        self.assertFalse(math.isnan(f["pbp_rz_opps_pg_t3"]), "the older games in the window are still used")

    def test_the_games_own_rows_change_no_new_column(self):
        for t in rows()[::60]:
            dirty = audit.perturb_own_game(store(), t)
            own = dirty._tables["pbp_team"].df
            self.assertTrue((own.loc[own["game_id"] == t.game_id, "plays"] == audit.MONSTER).all(), "the perturbation must actually reach the table")
            self.assertEqual(moved(build_features(store(), t), build_features(dirty, t), NEW_COLS), [], f"{t.player_id} {t.game_id}")
            gone = audit.drop_own_game(store(), t)
            self.assertNotIn(t.game_id, set(gone._tables["pbp_team"].df["game_id"]))
            self.assertEqual(moved(build_features(store(), t), build_features(gone, t), NEW_COLS), [], f"{t.player_id} {t.game_id}")

    def test_every_consumed_pbp_row_belongs_to_a_game_that_had_finished_and_never_the_target(self):
        kick = store()._tables["fixtures"].df.set_index("game_id")["kickoff"]
        seen = set()
        for t in rows()[::25]:
            trace: list = []
            build_features(store(), t, trace=trace)
            for rec in trace:
                if rec["table"] in PBP_TABLES:
                    seen.add(rec["table"])
                    if rec["max_known_at"] is not None:
                        self.assertLess(rec["max_known_at"], t.kickoff)
                    self.assertNotIn(t.game_id, rec["game_ids"], f"{rec['table']} fed the target game")
                    for g in rec["game_ids"]:
                        self.assertLessEqual(kick[g] + pd.Timedelta(hours=4), t.kickoff, f"{rec['table']} consumed {g} before it finished")
        self.assertEqual(seen, set(PBP_TABLES), "the trace must show every new table being read")

    def test_a_store_without_the_new_tables_still_builds_and_says_nan(self):
        t = pick("RB", 2024, 10)
        bare = RawStore([tab for n, tab in store()._tables.items() if n not in PBP_TABLES])
        f = build_features(bare, t)
        self.assertTrue(all(math.isnan(f[c]) for c in features.FAMILIES["pbp_usage"] + features.FAMILIES["pbp_team"] + features.FAMILIES["pbp_part"]))
        self.assertFalse(math.isnan(f["eff_rec_ypt_career"]))          # efficiency needs only the stats table


# ------------------------------------------------------------------ the registry and the models' view of it
class Registry(unittest.TestCase):
    def test_new_families_are_registered_and_opt_in_for_the_models(self):
        self.assertEqual(set(features.FAMILIES) & set(NEW_FAMILIES), set(NEW_FAMILIES))
        self.assertEqual(set(T.OPT_IN_FAMILIES), set(NEW_FAMILIES))
        self.assertEqual(len(NEW_COLS), len(set(NEW_COLS)))
        self.assertEqual((len(features.FAMILIES["pbp_usage"]), len(features.FAMILIES["pbp_team"]), len(features.FAMILIES["pbp_part"]),
                          len(features.FAMILIES["eff"])), (32, 10, 10, 18))
        df = pd.DataFrame({c: [0.0, 1.0] for c in features.feature_columns()} | {"position": ["QB", "RB"]})
        prim = T.feature_columns(df, T.PRIMARY)
        self.assertEqual(set(prim) & set(NEW_COLS), set(), "the Phase 2 primary must keep exactly its Phase 2 inputs")
        both = T.feature_columns(df, T.Spec("x", add_families=("pbp_usage", "eff")))
        self.assertTrue(set(features.FAMILIES["pbp_usage"] + features.FAMILIES["eff"]) <= set(both))
        self.assertFalse(set(features.FAMILIES["pbp_team"]) & set(both))
        self.assertEqual(set(both) - set(features.FAMILIES["pbp_usage"] + features.FAMILIES["eff"]), set(prim))

    def test_no_new_column_looks_like_a_label_a_baseline_or_a_same_week_value(self):
        for c in NEW_COLS:
            self.assertFalse(c.startswith(("y_", "base_", "spine_")), c)
        self.assertEqual([c for c in features.feature_columns() if "xfp" in c and c not in
                          ("xfp_l1", "xfp_l2", "xfp_l3", "xfp_std_mean", "prev_season_xfp_pg")], [])

    def test_features_module_still_imports_only_the_gate(self):
        import ast
        tree = ast.parse(Path(features.__file__).read_text())
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods - {"__future__", "typing", "model.point_in_time"}, set())

    def test_component_labels_rebuild_ppr_exactly_from_raw_stats(self):
        pg = store()._tables["player_games"].df
        d = pg[pg["season"] >= 2021].copy()
        ppr = (d["passing_yards"] * 0.04 + d["passing_tds"] * 4 - 2 * d["passing_interceptions"] + d["rushing_yards"] * 0.1
               + d["rushing_tds"] * 6 + d["receiving_yards"] * 0.1 + d["receiving_tds"] * 6 + d["receptions"]
               - 2 * d["fumbles_lost"] + 2 * d["two_pt_conversions"] + 6 * d["special_teams_tds"])
        self.assertLess(float((ppr - d["fantasy_points_ppr"]).abs().max()), 1e-9)
        self.assertGreater(len(d), 30000)


if __name__ == "__main__":
    unittest.main()
