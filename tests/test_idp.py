"""Individual defenders (DL / LB / DB) and kickers: positions, per-league scoring, E_pts, slots, and the guarantee that a
league without a defender slot is untouched.

    python -m unittest tests.test_idp

No network and no committed data is written: every frame is synthetic, except that the league.json files are read to
pin which committed leagues take the IDP path.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ff import ask, build, idp, leagues, projections, scoring, status

ROOT = Path(__file__).resolve().parent.parent

# the where-you-at scoring dict, as committed (print: leagues/where-you-at/league.json -> scoring)
WYA = json.loads((ROOT / "leagues" / "where-you-at" / "league.json").read_text(encoding="utf-8"))
SCORING = WYA["scoring"]


def stat_row(**kw) -> dict:
    """An idp_weeks row; unspecified stats are 0."""
    base = {c: 0.0 for c in build.IDP_WEEK_COLS}
    base.update(season=2026, week=1, gsis_id="x", name="X", position="LB", team="AAA", opponent="BBB")
    base.update(kw)
    return base


# --------------------------------------------------------------------------------------------- positions
class PositionMapping(unittest.TestCase):
    def test_codes_fold_to_league_positions(self):
        for code, want in [("DE", "DL"), ("DT", "DL"), ("NT", "DL"), ("DL", "DL"),
                           ("LB", "LB"), ("ILB", "LB"), ("OLB", "LB"), ("MLB", "LB"),
                           ("CB", "DB"), ("S", "DB"), ("SS", "DB"), ("FS", "DB"), ("SAF", "DB"), ("DB", "DB"),
                           ("K", "K")]:
            self.assertEqual(build.idp_position(code), want, code)
        for code in ("QB", "RB", "WR", "TE", "P", "LS", "OT", "G", "C", "FB", None, np.nan):
            self.assertIsNone(build.idp_position(code), code)

    def test_build_idp_weeks_from_a_nflverse_frame(self):
        def r(pid, pos, **kw):
            d = {"player_id": pid, "player_display_name": pid, "position": pos, "season": 2026, "week": 3,
                 "season_type": "REG", "team": "AAA", "opponent_team": "BBB"}
            d.update(kw)
            return d
        stats = pd.DataFrame([
            r("de", "DE", def_tackles_solo=3, def_tackle_assists=2, def_sacks=1.5, def_qb_hits=2,
              def_tackles_for_loss=1),
            r("cb", "CB", def_interceptions=1, def_pass_defended=2, def_tds=1, def_safeties=1),
            r("saf", "SAF", def_tackles_solo=5),
            r("lb", "OLB", def_fumbles_forced=1, fumble_recovery_opp=1, def_fg_blocks=1, def_punt_blocks=1),
            r("k", "K", fg_att=4, fg_made=3, fg_missed=1, fg_made_30_39=2, fg_made_50_59=1, fg_made_60_=1,
              fg_missed_40_49=1, pat_att=3, pat_made=2, pat_missed=1),
            r("qb", "QB"), r("p", "P"), r("ol", "OT"),
            r("post", "LB", season_type="POST", def_tackles_solo=9),
        ])
        out = build.build_idp_weeks(stats, 2026).set_index("gsis_id")
        self.assertEqual(sorted(out.index), ["cb", "de", "k", "lb", "saf"])          # QB, P, OT and the playoff row are out
        self.assertEqual(out.loc["de", "position"], "DL")
        self.assertEqual(out.loc["cb", "position"], "DB")
        self.assertEqual(out.loc["saf", "position"], "DB")
        self.assertEqual(out.loc["lb", "position"], "LB")
        self.assertEqual((out.loc["de", "tkl_solo"], out.loc["de", "tkl_ast"], out.loc["de", "sacks"]), (3, 2, 1.5))
        self.assertEqual(out.loc["saf", "tkl_ast"], 0)
        self.assertEqual((out.loc["cb", "ints"], out.loc["cb", "pass_def"], out.loc["cb", "def_td"],
                          out.loc["cb", "safeties"]), (1, 2, 1, 1))
        self.assertEqual((out.loc["lb", "ff"], out.loc["lb", "fum_rec"], out.loc["lb", "blk_kick"]), (1, 1, 2))
        k = out.loc["k"]
        self.assertEqual((k.fg_att, k.fg_made, k.fgm_30_39, k.fgm_50_59, k.fgm_60p, k.fgm_50p),
                         (4, 3, 2, 1, 1, 2))                                           # 50+ = 50-59 + 60+
        self.assertEqual((k.fgmiss_40_49, k.xpa, k.xpm, k.xpmiss), (1, 3, 2, 1))
        self.assertEqual(list(build.build_idp_weeks(stats.iloc[0:0], 2026).columns), build.IDP_WEEK_COLS)

    def test_assist_column_spelling_falls_back_only_when_the_column_is_absent(self):
        row = {"player_id": "s", "player_display_name": "s", "position": "SS", "season": 2026, "week": 1,
               "season_type": "REG", "team": "AAA", "opponent_team": "BBB", "def_tackles_solo": 2}
        old_name = build.build_idp_weeks(pd.DataFrame([{**row, "def_tackles_with_assist": 4}]), 2026)
        self.assertEqual(old_name.tkl_ast.iloc[0], 4)
        both = build.build_idp_weeks(pd.DataFrame([{**row, "def_tackles_with_assist": 4, "def_tackle_assists": 1}]), 2026)
        self.assertEqual(both.tkl_ast.iloc[0], 1)                                        # nflverse's real name wins

    def test_build_players_by_position_group(self):
        def r(gid, pos):
            return {"gsis_id": gid, "sleeper_id": f"s{gid}", "espn_id": None, "pfr_id": None, "full_name": gid,
                    "team": "AAA", "position": pos, "birth_date": "2000-01-01", "years_exp": 3, "rookie_year": 2023,
                    "draft_number": 10, "status": "ACT", "week": 1}
        ros = pd.DataFrame([r("qb", "QB"), r("dl", "DL"), r("de", "DE"), r("lb", "LB"), r("db", "DB"),
                            r("cb", "CB"), r("k", "K"), r("p", "P"), r("ol", "OL")])
        asof = pd.Timestamp("2026-10-29")
        skill = build.build_players(ros, None, asof)
        self.assertEqual(list(skill.gsis_id), ["qb"])                                   # default unchanged
        d = build.build_players(ros, None, asof, build.IDP_POS).set_index("gsis_id")
        self.assertEqual(sorted(d.index), ["cb", "db", "de", "dl", "k", "lb"])
        self.assertEqual(d.loc["de", "position"], "DL")
        self.assertEqual(d.loc["cb", "position"], "DB")


# --------------------------------------------------------------------------------------------- scoring
class LeagueScoring(unittest.TestCase):
    """One hand-computed row per group, against the committed where-you-at scoring dict."""

    def test_defender_row(self):
        # 6 solo (1) + 3 ast (.5) + 1 TFL (2) + 1 sack (4) + 2 QB hits (1) + 1 INT (6) + 2 PD (1) + 1 FF (4) + 1 FR (2)
        row = pd.DataFrame([stat_row(tkl_solo=6, tkl_ast=3, tkl_loss=1, sacks=1, qb_hits=2, ints=1, pass_def=2, ff=1,
                                     fum_rec=1)])
        self.assertEqual(scoring.score_idp(row, SCORING).iloc[0], 6 + 1.5 + 2 + 4 + 2 + 6 + 2 + 4 + 2)   # 29.5

    def test_rare_defensive_scores(self):
        # defensive TD 6 + safety 8 + blocked kick 2
        row = pd.DataFrame([stat_row(def_td=1, safeties=1, blk_kick=1)])
        self.assertEqual(scoring.score_idp(row, SCORING).iloc[0], 16.0)

    def test_kicker_row(self):
        # makes: 0-19 x1 (3) + 30-39 x2 (3 each) + 40-49 x1 (4) + 50+ x1 (5) = 18
        # misses: fgmiss -1 x2 + fgmiss_30_39 -2 x1 + fgmiss_40_49 -1 x1 = -5 (Sleeper charges every key a miss matches)
        # extra points: xpm +1 x3, xpmiss -1 x1 = +2
        row = pd.DataFrame([stat_row(position="K", fg_att=7, fg_made=5, fg_missed=2, fgm_0_19=1, fgm_30_39=2,
                                     fgm_40_49=1, fgm_50p=1, fgmiss_30_39=1, fgmiss_40_49=1,
                                     xpa=4, xpm=3, xpmiss=1)])
        self.assertEqual(scoring.score_idp(row, SCORING).iloc[0], 18 - 5 + 2)                   # 15.0

    def test_team_defence_keys_do_not_leak_in(self):
        # the dict also has team-defence `sack` 1, `ff` 1, `int` 2, `def_td` 6: a LB's 2 sacks are 2 x idp_sack only
        row = pd.DataFrame([stat_row(sacks=2, ints=1, ff=1, def_td=1)])
        self.assertEqual(scoring.score_idp(row, SCORING).iloc[0], 2 * 4 + 6 + 4 + 6)

    def test_missing_and_nan_columns_are_zero(self):
        self.assertEqual(scoring.score_idp(pd.DataFrame([{"sacks": np.nan, "tkl_solo": 2}]), SCORING).iloc[0], 2.0)

    def test_combined_tackle_key(self):
        row = pd.DataFrame([stat_row(tkl_solo=4, tkl_ast=2)])
        self.assertEqual(scoring.score_idp(row, {"idp_tkl": 1.0}).iloc[0], 6.0)

    def test_skill_scoring_ignores_defender_stats(self):
        row = pd.DataFrame([{**{c: 0 for c in scoring.STAT_KEYS.values() if c}, "position": "WR",
                             "receptions": 5, "rec_yds": 50, "tkl_solo": 9, "sacks": 3, "two_pt": 0}])
        self.assertEqual(scoring.score_frame(row, SCORING).iloc[0], 5 * 0.5 + 50 * 0.1)


class WhichLeaguesGetIdp(unittest.TestCase):
    def test_committed_leagues(self):
        got = {}
        for d in sorted(p for p in (ROOT / "leagues").iterdir() if (p / "league.json").exists()):
            got[d.name] = scoring.league_idp_positions(json.loads((d / "league.json").read_text(encoding="utf-8")))
        self.assertEqual(got["where-you-at"], ["DL", "LB", "DB", "K"])
        # gooma (K + DEF slots), the dynasty league and the ESPN league start no defender: no IDP path at all
        for slug in got:
            if slug != "where-you-at":
                self.assertEqual(got[slug], [], slug)

    def test_committed_non_idp_outputs_hold_skill_positions_only(self):
        for slug in ("gooma-s-family-league", "we-can-think-of-something-funny", "james-gregg-espn"):
            for name in ("roster.csv", "all_rosters.csv", "available.csv"):
                p = ROOT / "leagues" / slug / name
                if p.exists():
                    self.assertLessEqual(set(pd.read_csv(p, usecols=["position"]).position),
                                         {"QB", "RB", "WR", "TE"}, f"{slug}/{name}")
            self.assertFalse((ROOT / "leagues" / slug / "idp_points.csv").exists())

    def test_idp_flex_and_slot_rules(self):
        self.assertEqual(scoring.league_idp_positions({"roster_positions": ["QB", "IDP_FLEX", "BN"]}),
                         ["DL", "LB", "DB"])
        self.assertEqual(scoring.league_idp_positions({"roster_positions": ["QB", "K", "DEF", "BN"]}), [])
        self.assertFalse(scoring.slot_allows("DL", "DL"))                  # not scored unless the league is IDP
        self.assertTrue(scoring.slot_allows("DL", "DL", idp=True))
        self.assertFalse(scoring.slot_allows("DL", "LB", idp=True))
        self.assertTrue(scoring.slot_allows("IDP_FLEX", "DB", idp=True))
        self.assertTrue(scoring.slot_allows("FLEX", "WR"))                  # skill rules unchanged


# --------------------------------------------------------------------------------------------- E_pts
def idp_world():
    """Players A (LB, projected), B (DL, two team games), D (LB, one old game), K1 (K, projected), Z (DB, Out),
    C (DL, defines the team's games), BY (LB on a bye)."""
    players = pd.DataFrame([
        {"gsis_id": g, "sleeper_id": f"s{g}", "name": g, "team": t, "position": p, "age": 25.0, "years_exp": 3,
         "injury_status": inj, "nfl_status": "ACT", "draft_number": 50, "rookie_year": 2023}
        for g, t, p, inj in [("A", "AAA", "LB", None), ("B", "AAA", "DL", None), ("D", "AAA", "LB", None),
                             ("K1", "AAA", "K", None), ("Z", "AAA", "DB", "Out"), ("C", "AAA", "DL", None),
                             ("BY", "BYE", "LB", None)]])
    rows = []
    for w in (1, 2, 3):                      # C plays every game, so the team's last three games are 1, 2, 3
        rows.append(stat_row(week=w, gsis_id="C", name="C", position="DL", tkl_solo=1))
    rows.append(stat_row(week=1, gsis_id="A", name="A", tkl_solo=5))
    rows.append(stat_row(week=2, gsis_id="B", name="B", position="DL", tkl_solo=4))
    rows.append(stat_row(week=3, gsis_id="B", name="B", position="DL", tkl_solo=2, sacks=1))
    rows.append(stat_row(week=1, gsis_id="D", name="D", tkl_solo=3, tkl_ast=2))
    rows.append(stat_row(week=1, gsis_id="K1", name="K1", position="K", fg_att=2, fg_made=2, fgm_30_39=2, xpa=3, xpm=3))
    rows.append(stat_row(week=2, gsis_id="K1", name="K1", position="K", fg_att=1, fg_made=1, fgm_40_49=1, xpa=1, xpm=1))
    rows.append(stat_row(week=1, gsis_id="Z", name="Z", position="DB", tkl_solo=8))
    rows.append(stat_row(week=1, gsis_id="BY", name="BY", position="LB", team="BYE", tkl_solo=8))
    hist = pd.DataFrame(rows)
    raw = [
        {"player_id": "sA", "stats": {"idp_tkl_solo": 6, "idp_tkl_ast": 3, "idp_sack": 1}},
        {"player_id": "sK1", "stats": {"fga": 3, "fgm": 2, "fgm_30_39": 1, "fgm_40_49": 1, "fgmiss_50p": 1,
                                       "xpa": 2, "xpm": 2}},
        {"player_id": "sZ", "stats": {"idp_tkl_solo": 7}},
        {"player_id": "sC", "stats": {"adp_dd_ppr": 999.0}},          # no projected line: not a projection
    ]
    proj = projections.idp_from_raw(raw, players, 2026, 4)
    sched = pd.DataFrame([{"season": 2026, "week": 4, "team": "AAA", "opponent": "ZZZ", "total_line": 44.0, "home": 1},
                          {"season": 2026, "week": 4, "team": "BYE", "opponent": "BYE", "total_line": np.nan, "home": 0}])
    roster = players[["gsis_id", "name", "position", "team", "sleeper_id"]].assign(is_starter=0)
    return players, hist, proj, sched, roster


class ExpectedPoints(unittest.TestCase):
    def setUp(self):
        self.players, self.hist, self.proj, self.sched, self.roster = idp_world()
        self.meta = {"name": "T", "scoring": SCORING}
        self.pts = idp.league_points(self.hist, SCORING)

    def run_compute(self, proj=None, roster=None):
        out = idp.compute(self.roster if roster is None else roster, self.players, self.hist,
                          self.proj if proj is None else proj, self.pts, self.sched, self.meta, 4, 2026,
                          ["DL", "LB", "DB", "K"])
        return out.set_index("gsis_id")

    def test_projection_rows(self):
        self.assertEqual(sorted(self.proj.gsis_id), ["A", "K1", "Z"])
        k = self.proj.set_index("gsis_id").loc["K1"]
        self.assertEqual(k.proj_fg_missed, 1.0)                              # max(3 - 2, the 50+ bucket's 1)

    def test_projection_beats_trailing(self):
        o = self.run_compute()
        # A: 6 solo + 3 ast (x .5) + 1 sack (4) = 11.5 projected; trailing (5 pts in one of three games) is not used
        self.assertEqual(o.loc["A", "E_pts"], 11.5)
        self.assertEqual(o.loc["A", "pts_mix"], "projection only")
        # K1: 3 + 4 (made) - 1 (the derived miss) + 2 (xpm) = 8.0
        self.assertEqual(o.loc["K1", "E_pts"], 8.0)

    def test_trailing_fallback_and_window(self):
        o = self.run_compute()
        # B has rows in games 2 and 3 of the window (1, 2, 3): (4 + (2 + 4)) / 2 = 5.0, and 3.0 tackles a game
        self.assertAlmostEqual(o.loc["B", "trail_pts"], 5.0)
        self.assertEqual(o.loc["B", "E_pts"], 5.0)
        self.assertEqual(o.loc["B", "E_opps"], 3.0)
        self.assertEqual(o.loc["B", "pts_mix"], "recent only")
        # D has one row, in game 1; games 2 and 3 count as 0: (3 + 2 x .5) / 3 = 4 / 3
        self.assertAlmostEqual(o.loc["D", "trail_pts"], 4 / 3)
        self.assertEqual(o.loc["D", "E_pts"], 1.3)
        self.assertEqual(o.loc["D", "E_opps"], round(5 / 3, 1))             # tackles + assists, per game

    def test_opportunity_volume(self):
        o = self.run_compute()
        # A: 5 tackles in game 1 of the window, 0 in games 2 and 3 -> 5 / 3 a game
        self.assertEqual(o.loc["A", "E_opps"], round(5 / 3, 1))
        # K1: (2 FGA + 3 XPA) + (1 + 1) over the same three games
        self.assertEqual(o.loc["K1", "E_opps"], round(7 / 3, 1))
        self.assertTrue(o.loc["K1", "why"].startswith("K: "))

    def test_why_string(self):
        o = self.run_compute()
        self.assertIn("IDP: 3.0 tkl/g, 0.5 sacks/g", o.loc["B", "why"])
        self.assertIn("last 2 g avg 5.0 pts", o.loc["B", "why"])
        self.assertIn("proj 11.5 pts", o.loc["A", "why"])

    def test_out_and_bye_have_no_points(self):
        o = self.run_compute()
        self.assertTrue(np.isnan(o.loc["Z", "E_pts"]))                       # Out: projection ignored
        self.assertEqual(o.loc["Z", "flag"], "OUT")
        self.assertTrue(np.isnan(o.loc["BY", "E_pts"]))
        self.assertEqual(o.loc["BY", "flag"], "BYE")

    def test_without_any_projection_every_number_is_trailing(self):
        o = self.run_compute(proj=pd.DataFrame())
        self.assertEqual(o.loc["A", "E_pts"], round(5 / 3, 1))
        self.assertEqual(o.loc["A", "pts_mix"], "recent only")
        self.assertTrue(o.proj_pts_league.isna().all())

    def test_no_projection_and_no_history_is_no_baseline(self):
        players = pd.concat([self.players, self.players.iloc[[0]].assign(
            gsis_id="N", sleeper_id="sN", name="N")], ignore_index=True)
        roster = players[["gsis_id", "name", "position", "team", "sleeper_id"]]
        o = idp.compute(roster, players, self.hist, pd.DataFrame(), self.pts, self.sched, self.meta, 4, 2026,
                        ["DL", "LB", "DB", "K"]).set_index("gsis_id")
        self.assertTrue(np.isnan(o.loc["N", "E_pts"]))
        self.assertEqual(o.loc["N", "flag"], "NO BASELINE")

    def test_other_positions_and_missing_ids_are_ignored(self):
        mixed = pd.concat([self.roster, pd.DataFrame([
            {"gsis_id": "W", "name": "W", "position": "WR", "team": "AAA", "sleeper_id": "sW"},
            {"gsis_id": np.nan, "name": "N", "position": "LB", "team": "AAA", "sleeper_id": "sN"}])])
        o = self.run_compute(roster=mixed)
        self.assertNotIn("W", o.index)
        self.assertEqual(len(o), len(self.roster))


# --------------------------------------------------------------------------------------------- status
class StatusRows(unittest.TestCase):
    def test_idp_status_rows_use_the_injury_report(self):
        players, *_ = idp_world()
        players["practice_participation"] = np.nan
        players["news_updated"] = np.nan
        inj = pd.DataFrame([{"gsis_id": "A", "week": 4, "report_status": "Questionable",
                             "practice_status": "Limited Participation in Practice"}])
        st = status.build_idp(players, 4, inj, None)
        by = st.set_index("gsis_id")
        self.assertEqual(len(st), len(players))                                 # every rostered DL/LB/DB/K gets a row
        self.assertEqual(by.loc["A", "status_flag"], "questionable")
        self.assertEqual(by.loc["A", "practice"], "Limited Participation in Practice")
        self.assertEqual(by.loc["Z", "status_flag"], "out")                     # Sleeper's own designation
        self.assertEqual(by.loc["B", "status_flag"], "healthy")
        self.assertTrue(by.espn_rank.isna().all())

    def test_gameday_inactive_list_applies(self):
        players, *_ = idp_world()
        players["practice_participation"] = np.nan
        ina = pd.DataFrame([{"gsis_id": "B", "team": "AAA", "kickoff": pd.NaT, "inactive_known_at": "2026-10-25T16:00Z"}])
        st = status.build_idp(players, 4, None, ina).set_index("gsis_id")
        self.assertEqual(st.loc["B", "status_flag"], "inactive")
        self.assertEqual(st.loc["B", "play_prob"], 0.0)

    def test_describe_and_warnings_survive_rows_without_a_depth_chart(self):
        players, *_ = idp_world()
        players["practice_participation"] = np.nan
        st = status.build_idp(players, 4, None, None)
        self.assertIn("no depth-chart model", status.describe(st, "Z"))
        msgs = status.warnings(st, {"Z", "A"})
        self.assertTrue(any(m.startswith("Z: OUT") for m in msgs))


# --------------------------------------------------------------------------------------------- league files
def league_inputs(week=4, season=2026):
    """Everything write_league needs, small: two skill players on a roster + one free agent, a DL, a LB, a K."""
    players, hist, proj, sched, _ = idp_world()
    skill = pd.DataFrame([
        {"gsis_id": g, "sleeper_id": f"s{g}", "name": g, "team": "AAA", "position": p, "age": 25.0, "years_exp": 3,
         "injury_status": None, "nfl_status": "ACT", "draft_number": 50, "rookie_year": 2023}
        for g, p in [("W1", "WR"), ("R1", "RB"), ("W2", "WR")]])
    sched = pd.concat([sched, pd.DataFrame([{"season": 2026, "week": 3, "team": "AAA", "opponent": "ZZZ",
                                             "total_line": 44.0, "home": 1}])], ignore_index=True)
    cols = ["gsis_id", "name", "position", "team", "sleeper_id"]
    all_rosters = pd.concat([skill.iloc[:2][cols], players[players.gsis_id.isin(["A", "B", "K1"])][cols]],
                            ignore_index=True)
    all_rosters = all_rosters.assign(owner_name="me", is_starter=1, is_taxi=0, is_ir=0, roster_fetched_at="t",
                                     season=season, week=week)
    pw = pd.DataFrame([{"season": 2026, "week": w, "gsis_id": g, "name": g, "position": p, "team": "AAA",
                        "opponent": "ZZZ", "snaps": 50.0, "snap_pct": 0.8, "targets": 6, "receptions": 4,
                        "rec_yds": 40, "rec_tds": 0, "rec_fd": 0, "carries": 0, "rush_yds": 0, "rush_tds": 0,
                        "rush_fd": 0, "pass_att": 0, "pass_cmp": 0, "pass_yds": 0, "pass_tds": 0, "pass_int": 0,
                        "two_pt": 0, "fum": 0, "fum_lost": 0, "fum_rec": 0, "fum_rec_td": 0, "touches": 4,
                        "opps": 6, "fpts_ppr": 8.0}
                       for w in (1, 2, 3) for g, p in [("W1", "WR"), ("R1", "RB"), ("W2", "WR")]])
    base = pd.DataFrame([{"gsis_id": g, "games": 3, "avg_opps": 6.0, "avg_targets": 6.0, "avg_carries": 0.0,
                          "avg_snap_pct": 0.8, "avg_fpts": 8.0, "trend_opps": 0.0, "trend_snap": 0.0,
                          "trend_season": "current", "baseline_source": "current season"}
                         for g in ("W1", "R1", "W2")])
    dvp = pd.DataFrame([{"defense": "ZZZ", "position": p, "dvp_rank": 10} for p in ("WR", "RB")])
    depth = pd.DataFrame(columns=["gsis_id", "rank", "prev_rank"])
    return dict(my_roster=all_rosters, all_rosters=all_rosters, players=skill, pw_all=pw, base=base,
                proj=pd.DataFrame(), dvp=dvp, sched=sched, depth=depth, week=week, season=season,
                trending=pd.DataFrame(), cfg={}), {"players": players, "weeks": hist, "proj": proj}


def write(root: Path, positions: list, with_idp: bool, scoring_rules=None):
    args, bundle = league_inputs()
    meta = {"slug": "lg", "league_id": "1", "name": "LG", "platform": "smoke",
            "scoring": scoring_rules or {"rec": 0.5, "rec_yd": 0.1, **{k: v for k, v in SCORING.items() if k.startswith(("idp", "fg", "xp"))}},
            "roster_positions": positions}
    return leagues.write_league(root, meta, args["my_roster"], args["all_rosters"], args["players"], args["pw_all"],
                                args["base"], args["proj"], args["dvp"], args["sched"], args["depth"], args["week"],
                                args["season"], args["trending"], args["cfg"],
                                idp=bundle if with_idp else None)


class OtherLeaguesUnchanged(unittest.TestCase):
    def test_league_without_a_defender_slot_is_byte_identical(self):
        # the lineup has K and DEF slots (like gooma) and the rosters hold DL / LB / K rows, as the widened
        # crosswalk produces: none of it may reach the league's files
        pos = ["QB", "RB", "WR", "TE", "FLEX", "K", "DEF", "BN"]
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            write(Path(a), pos, with_idp=False)
            write(Path(b), pos, with_idp=True)
            da, db = Path(a) / "leagues" / "lg", Path(b) / "leagues" / "lg"
            self.assertEqual(sorted(p.name for p in da.iterdir()), sorted(p.name for p in db.iterdir()))
            self.assertNotIn("idp_points.csv", [p.name for p in db.iterdir()])
            for name in ("roster.csv", "all_rosters.csv", "available.csv", "player_points.csv", "league.json"):
                self.assertEqual((da / name).read_bytes(), (db / name).read_bytes(), name)
            r = pd.read_csv(db / "roster.csv")
            self.assertEqual(set(r.position), {"WR", "RB"})                    # skill only, as before

    def test_idp_league_adds_rows_and_leaves_the_skill_rows_alone(self):
        pos = ["QB", "RB", "WR", "TE", "FLEX", "K", "DL", "DL", "LB", "DB", "BN"]
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            write(Path(a), pos, with_idp=False)
            write(Path(b), pos, with_idp=True)
            for name in ("roster.csv", "all_rosters.csv", "available.csv"):
                old = pd.read_csv(Path(a) / "leagues" / "lg" / name)
                new = pd.read_csv(Path(b) / "leagues" / "lg" / name)
                self.assertTrue(set(old.columns) <= set(new.columns), name)       # additive columns only
                skill = new[new.position.isin(["QB", "RB", "WR", "TE"])].reset_index(drop=True)
                pd.testing.assert_frame_equal(old.sort_values("gsis_id").reset_index(drop=True)[old.columns],
                                              skill.sort_values("gsis_id").reset_index(drop=True)[old.columns],
                                              check_dtype=False)
            r = pd.read_csv(Path(b) / "leagues" / "lg" / "roster.csv")
            self.assertEqual(set(r.position), {"WR", "RB", "LB", "DL", "K"})
            idp_rows = r[r.position.isin(["LB", "DL", "K"])]
            self.assertTrue(idp_rows.E_pts.notna().all())
            self.assertTrue(idp_rows.E_opps.notna().all())
            self.assertTrue(idp_rows.why.notna().all())
            for col in ("E_targets", "E_carries", "dvp_rank", "depth_rank", "mult", "td_adj"):
                self.assertTrue(idp_rows[col].isna().all(), col)                  # skill-only columns stay NaN
            self.assertTrue((Path(b) / "leagues" / "lg" / "idp_points.csv").exists())
            av = pd.read_csv(Path(b) / "leagues" / "lg" / "available.csv")
            self.assertTrue(set(av[av.position.isin(["DL", "LB", "DB", "K"])].gsis_id).isdisjoint(set(r.gsis_id)))

    def test_free_agent_pool_is_trimmed_per_position(self):
        pos = ["QB", "RB", "WR", "TE", "K", "DL", "LB", "DB", "BN"]
        with tempfile.TemporaryDirectory() as b:
            args, bundle = league_inputs()
            free = pd.DataFrame([{"gsis_id": f"F{i}", "sleeper_id": f"sF{i}", "name": f"F{i}", "team": "AAA",
                                  "position": "DL", "age": 25.0, "years_exp": 3, "injury_status": None,
                                  "nfl_status": "ACT", "draft_number": 50, "rookie_year": 2023}
                                 for i in range(10)])
            bundle["players"] = pd.concat([bundle["players"], free], ignore_index=True)
            bundle["weeks"] = pd.concat([bundle["weeks"]] + [
                pd.DataFrame([stat_row(week=w, gsis_id=f"F{i}", name=f"F{i}", position="DL", tkl_solo=i + 1)])
                for i in range(10) for w in (1, 2, 3)], ignore_index=True)
            meta = {"slug": "lg", "league_id": "1", "name": "LG", "platform": "smoke", "scoring": SCORING,
                    "roster_positions": pos}
            leagues.write_league(Path(b), meta, args["my_roster"], args["all_rosters"], args["players"],
                                 args["pw_all"], args["base"], args["proj"], args["dvp"], args["sched"],
                                 args["depth"], 4, 2026, args["trending"], {"idp_pool_top": 1}, idp=bundle)
            av = pd.read_csv(Path(b) / "leagues" / "lg" / "available.csv")
            dl = av[av.position == "DL"]
            self.assertEqual(list(dl.gsis_id), ["F9"])                           # best of 11 free-agent linemen
            self.assertIn("W2", set(av.gsis_id))                                # skill rows are never trimmed


# --------------------------------------------------------------------------------------------- slots
class LineupSlots(unittest.TestCase):
    def write_league(self, root: Path, slug: str, positions: list, rows: list):
        d = root / "leagues" / slug
        d.mkdir(parents=True)
        (d / "league.json").write_text(json.dumps({"league_id": "9", "name": slug, "platform": "sleeper",
                                                   "roster_positions": positions, "scoring": SCORING}))
        df = pd.DataFrame([{"gsis_id": g, "name": g, "position": p, "team": "AAA", "opponent": "BBB", "E_pts": e,
                            "conf": "High", "why": "", "flag": np.nan, "is_starter": 0} for g, p, e in rows])
        df.to_csv(d / "roster.csv", index=False)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._root = ask.ROOT
        ask.ROOT = self.root

    def tearDown(self):
        ask.ROOT = self._root
        self.tmp.cleanup()

    def test_idp_slots_fill_from_the_new_rows(self):
        rows = [("qb", "QB", 20.0), ("k1", "K", 8.0), ("k2", "K", 6.0), ("dl1", "DL", 9.0), ("dl2", "DL", 7.0),
                ("dl3", "DL", 5.0), ("lb1", "LB", 8.5), ("lb2", "LB", 6.0), ("db1", "DB", 7.5),
                ("wr1", "WR", 12.0), ("wr2", "WR", 4.0), ("rb1", "RB", 11.0)]
        self.write_league(self.root, "idp", ["QB", "RB", "WR", "FLEX", "K", "DL", "DL", "LB", "DB", "BN"], rows)
        start, note, bench = ask.lineup("idp")
        got = list(zip(start.slot, start.name))
        self.assertIn(("K", "k1"), got)
        self.assertEqual(sorted(n for s, n in got if s == "DL"), ["dl1", "dl2"])        # the two best, not dl3
        self.assertIn(("LB", "lb1"), got)
        self.assertIn(("DB", "db1"), got)
        self.assertIn("9 of 9 scoreable slots filled", note)
        self.assertNotIn("not scored here", note)
        self.assertEqual(set(bench.name), {"k2", "dl3", "lb2"})
        # a bench defender is compared against the starter he could replace, not against a WR
        self.assertEqual(bench.set_index("name").loc["dl3", "gap_to_starter"], round(5.0 - 7.0, 1))

    def test_defenders_never_fill_skill_flex_slots(self):
        rows = [("wr1", "WR", 5.0), ("lb1", "LB", 30.0)]
        self.write_league(self.root, "idp", ["WR", "FLEX", "LB", "BN"], rows)
        start, _, _ = ask.lineup("idp")
        self.assertEqual(dict(zip(start.slot, start.name)), {"WR": "wr1", "LB": "lb1"})

    def test_league_without_a_defender_slot_still_skips_k(self):
        rows = [("qb", "QB", 20.0), ("k1", "K", 8.0), ("lb1", "LB", 9.0)]
        self.write_league(self.root, "plain", ["QB", "K", "DEF", "BN"], rows)
        start, note, _ = ask.lineup("plain")
        self.assertEqual(list(start.slot), ["QB"])
        self.assertIn("K, DST and IDP are out of scope", note)


if __name__ == "__main__":
    unittest.main()
