"""The model columns beside E_pts are display only.

    python -m unittest discover -s tests -t .

No network, no ML libraries, none of the committed data files are touched (everything runs in a temp copy). What is proven:

  * the league columns are that league's composed points and its own floor / ceiling: a non-PPR league uses its own quantile
    models' columns (`p10_<slug>`, `p90_<slug>`), a PPR league the PPR band, and a non-PPR league with no band of its own gets
    NaN, never a shifted guess; NaN everywhere the model has no row
  * the frame's week is its most common (season, week) PAIR, so a season boundary cannot name a week no row holds
  * with no data/model_pts.csv, or no row for the frame's week, the recipes print EXACTLY what they printed before
  * with it, `lineup` picks the same players in the same slots (E_pts still decides everything) and the printed header says
    once that E_pts stays authoritative
"""
from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from ff import ask, live, modelcols

REPO = Path(__file__).resolve().parent.parent
SLUG = "where-you-at"


def model_rows(week: int = 4) -> pd.DataFrame:
    """Model rows for three players in a league whose scoring differs from PPR (the TE premium) and one in full PPR."""
    return pd.DataFrame({
        "season": 2026, "week": week, "gsis_id": ["00-A", "00-B", "00-C", "00-D"],
        "pts_model": [15.0, 12.0, 9.0, 20.0], "pts_model_components": [14.0, 11.0, 8.0, 19.0],
        f"pts_{SLUG}": [13.0, 11.0, 8.5, 19.0], "pts_gooma-s-family-league": [14.0, 11.0, 8.0, 19.0],
        "p10": [7.0, 5.0, 2.0, 11.0], "p90": [24.0, 20.0, 15.0, 30.0]})


class LeagueColumns(unittest.TestCase):
    def test_points_are_the_composed_column_and_a_league_with_its_own_band_uses_it(self):
        m = model_rows().assign(**{f"p10_{SLUG}": [6.0, 4.0, 1.0, 10.0], f"p90_{SLUG}": [25.0, 21.0, 17.0, 31.0]})
        c = modelcols.league_columns(m, SLUG).set_index("gsis_id")
        self.assertEqual(c["E_pts_model"].tolist(), [13.0, 11.0, 8.5, 19.0])
        self.assertEqual(c["p10"].tolist(), [6.0, 4.0, 1.0, 10.0])          # the league's own quantile models, not the PPR band
        self.assertEqual(c["p90"].tolist(), [25.0, 21.0, 17.0, 31.0])

    def test_a_ppr_league_gets_the_ppr_band_unchanged(self):
        c = modelcols.league_columns(model_rows(), "gooma-s-family-league").set_index("gsis_id")
        self.assertEqual(c["p10"].tolist(), [7.0, 5.0, 2.0, 11.0])
        self.assertEqual(c["p90"].tolist(), [24.0, 20.0, 15.0, 30.0])

    def test_a_non_ppr_league_with_no_band_of_its_own_gets_no_band_not_a_shifted_guess(self):
        c = modelcols.league_columns(model_rows(), SLUG).set_index("gsis_id")     # its points differ from PPR and it has no p10_/p90_ columns
        self.assertEqual(c["E_pts_model"].tolist(), [13.0, 11.0, 8.5, 19.0])
        self.assertTrue(c["p10"].isna().all() and c["p90"].isna().all())

    def test_a_missing_value_stays_missing_and_an_unknown_league_is_empty(self):
        m = model_rows()
        m.loc[1, "p10"] = np.nan
        c = modelcols.league_columns(m, "gooma-s-family-league").set_index("gsis_id")
        self.assertTrue(np.isnan(c.loc["00-B", "p10"]) and c.loc["00-B", "p90"] == 20.0)
        self.assertTrue(modelcols.league_columns(m, "no-such-league").empty)
        self.assertTrue(modelcols.league_columns(pd.DataFrame(), SLUG).empty)

    def test_a_tiny_negative_band_edge_does_not_print_as_minus_zero(self):
        m = model_rows()
        m.loc[0, "p10"] = -0.01
        c = modelcols.league_columns(m, "gooma-s-family-league").set_index("gsis_id")
        self.assertEqual(repr(float(c.loc["00-A", "p10"])), "0.0")


class WeekOf(unittest.TestCase):
    def test_the_most_common_season_week_pair_not_the_two_modes_taken_apart(self):
        """(2025, 18) x3, (2026, 1) x2, (2026, 2) x2: the season mode is 2026 and the week mode is 18, a pair no row holds."""
        df = pd.DataFrame({"season": [2025] * 3 + [2026] * 4, "week": [18] * 3 + [1] * 2 + [2] * 2})
        mp = pd.DataFrame({"season": [2025, 2026], "week": [18, 1]})
        self.assertEqual(modelcols.week_of(df, mp), (2025, 18))

    def test_a_tie_takes_the_later_week_and_a_frame_without_weeks_takes_the_newest_in_the_file(self):
        df = pd.DataFrame({"season": [2025, 2026], "week": [18, 1]})
        self.assertEqual(modelcols.week_of(df, pd.DataFrame({"season": [2025], "week": [18]})), (2026, 1))
        mp = pd.DataFrame({"season": [2025, 2026, 2026], "week": [18, 1, 3]})
        self.assertEqual(modelcols.week_of(df[["season"]], mp), (2026, 3))
        self.assertIsNone(modelcols.week_of(df, pd.DataFrame()))

    def test_attach_at_a_season_boundary_joins_the_right_week(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "data").mkdir()
            pd.concat([model_rows(week=18).assign(season=2025, **{f"pts_{SLUG}": 1.0}), model_rows(week=1).assign(**{f"pts_{SLUG}": 2.0})]) \
                .to_csv(root / "data" / "model_pts.csv", index=False)
            frame = pd.DataFrame({"gsis_id": ["00-A"] * 3 + ["00-B"] * 4, "season": [2025] * 3 + [2026] * 4, "week": [18] * 3 + [1] * 2 + [2] * 2})
            out = modelcols.attach(frame, SLUG, root)
            self.assertEqual(set(out["E_pts_model"].dropna()), {1.0})           # (2025, 18), the pair most rows hold


class Attach(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "data").mkdir()
        self.df = pd.DataFrame({"gsis_id": pd.array(["00-A", "00-B", "00-X", None], dtype="string"), "season": 2026, "week": 4,
                                "name": list("abcd"), "E_pts": [10.0, 9.0, 8.0, 7.0]}, index=[5, 6, 7, 8])

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rows):
        rows.to_csv(self.root / "data" / "model_pts.csv", index=False)

    def test_no_model_file_means_the_frame_comes_back_untouched(self):
        out = modelcols.attach(self.df, SLUG, self.root)
        self.assertIs(out, self.df)
        self.assertEqual(list(out.columns), list(self.df.columns))

    def test_join_is_on_gsis_id_keeps_the_index_and_leaves_nan_where_the_model_has_no_row(self):
        self.write(model_rows())
        out = modelcols.attach(self.df, SLUG, self.root)
        self.assertEqual(list(out.index), [5, 6, 7, 8])
        self.assertEqual(list(out.columns), [*self.df.columns, *modelcols.MODEL_COLS])
        self.assertEqual(out["E_pts_model"].iloc[:2].tolist(), [13.0, 11.0])
        self.assertTrue(out["E_pts_model"].iloc[2:].isna().all())          # an unknown id and a missing id
        self.assertEqual(out["E_pts"].tolist(), self.df["E_pts"].tolist())  # nothing else moved

    def test_a_frame_from_another_week_gets_nothing_not_last_weeks_numbers(self):
        self.write(model_rows(week=3))
        self.assertIs(modelcols.attach(self.df, SLUG, self.root), self.df)

    def test_a_frame_with_no_week_column_takes_the_newest_week(self):
        self.write(pd.concat([model_rows(week=3), model_rows(week=4).assign(**{f"pts_{SLUG}": 99.0})]))
        out = modelcols.attach(self.df.drop(columns=["season", "week"]), SLUG, self.root)
        self.assertEqual(out["E_pts_model"].iloc[0], 99.0)

    def test_existing_model_columns_are_replaced_not_duplicated(self):
        self.write(model_rows())
        once = modelcols.attach(self.df, SLUG, self.root)
        twice = modelcols.attach(once, SLUG, self.root)
        self.assertEqual(list(once.columns), list(twice.columns))
        pd.testing.assert_frame_equal(once, twice)

    def test_an_unreadable_file_is_not_an_error(self):
        (self.root / "data" / "model_pts.csv").write_bytes(b"\x00\xff not a csv \x00")
        self.assertIs(modelcols.attach(self.df, SLUG, self.root), self.df)


def copy_league(root: Path, slug: str = SLUG) -> None:
    """A temp copy of one committed league folder as the PIPELINE wrote it: once the model step has run, the committed roster.csv
    also carries the three model columns, so they are stripped here (the tests add their own model file)."""
    shutil.copytree(REPO / "leagues" / slug, root / "leagues" / slug)
    (root / "data").mkdir(exist_ok=True)
    p = root / "leagues" / slug / "roster.csv"
    r = pd.read_csv(p, low_memory=False)
    r.drop(columns=[c for c in modelcols.MODEL_COLS if c in r.columns]).to_csv(p, index=False)


def model_rows_for(root: Path, slug: str, bump: float = 3.0) -> None:
    """Model rows for EVERY scored player of the league's roster.csv, deliberately reversing their E_pts order."""
    r = pd.read_csv(root / "leagues" / slug / "roster.csv", low_memory=False)
    mp = pd.DataFrame({"season": r["season"], "week": r["week"], "gsis_id": r["gsis_id"],
                       "pts_model": 30.0 - r["E_pts"].fillna(0), "pts_model_components": 29.0 - r["E_pts"].fillna(0),
                       f"pts_{slug}": 30.0 - r["E_pts"].fillna(0), "p10": 1.0, "p90": 40.0,
                       f"p10_{slug}": 2.0, f"p90_{slug}": 41.0})
    mp.to_csv(root / "data" / "model_pts.csv", index=False)


class Recipes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        copy_league(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def lineup(self):
        with mock.patch.object(ask, "ROOT", self.root):
            return ask.lineup(SLUG)

    def test_without_the_model_file_lineup_is_what_it_always_was(self):
        start, note, bench = self.lineup()
        self.assertFalse(any(c in start.columns for c in modelcols.MODEL_COLS))
        self.assertFalse(any(c in bench.columns for c in modelcols.MODEL_COLS))

    def test_with_it_lineup_picks_the_same_players_in_the_same_slots(self):
        before, note_before, bench_before = self.lineup()
        model_rows_for(self.root, SLUG)                 # a model that disagrees with E_pts about EVERYONE
        after, note_after, bench_after = self.lineup()
        self.assertEqual(before[["slot", "name", "E_pts"]].to_dict("records"), after[["slot", "name", "E_pts"]].to_dict("records"))
        self.assertEqual(note_before, note_after)
        self.assertEqual(list(bench_before["name"]), list(bench_after["name"]))
        self.assertEqual(list(after.columns[after.columns.get_loc("E_pts") + 1:][:3]), list(modelcols.MODEL_COLS))
        self.assertTrue(after["E_pts_model"].notna().all())

    def test_the_lineup_header_says_once_that_e_pts_stays_authoritative(self):
        model_rows_for(self.root, SLUG)
        buf = io.StringIO()
        with mock.patch.object(ask, "ROOT", self.root), redirect_stdout(buf):
            ask.main(["lineup", SLUG])
        out = buf.getvalue()
        self.assertEqual(out.count("E_pts stays authoritative"), 1)
        self.assertIn("E_pts_model", out)

    def test_no_note_and_no_columns_when_the_model_has_nothing(self):
        buf = io.StringIO()
        with mock.patch.object(ask, "ROOT", self.root), redirect_stdout(buf):
            ask.main(["lineup", SLUG])
        self.assertNotIn("authoritative", buf.getvalue())
        self.assertNotIn("E_pts_model", buf.getvalue())

    def test_live_render_prints_the_columns_beside_e_pts_and_the_note_once(self):
        mine = pd.DataFrame({"name": ["A", "B"], "position": ["RB", "WR"], "team": ["X", "Y"], "E_pts": [10.0, 9.0],
                             "E_pts_model": [11.0, 8.0], "p10": [4.0, 3.0], "p90": [20.0, 18.0], "E_opps": [9.0, 8.0],
                             "is_starter": [1, 0], "is_ir": [0, 0], "is_taxi": [0, 0]})
        res = {"league": "L", "platform": "sleeper", "live": True, "fetched_at": "t", "snapshot_at": "s", "unscored": [],
               "diff": {"started_since_snapshot": [], "benched_since_snapshot": []}, "mine": mine,
               "available": mine.assign(is_starter=0)}
        text = live.render(res)
        self.assertEqual(text.count("E_pts stays authoritative"), 1)
        header = next(l for l in text.splitlines() if l.split()[:1] == ["name"])
        cols = header.split()
        self.assertEqual(cols[cols.index("E_pts") + 1:cols.index("E_pts") + 4], list(modelcols.MODEL_COLS))
        plain = live.render({**res, "mine": mine.drop(columns=list(modelcols.MODEL_COLS)),
                             "available": mine.drop(columns=list(modelcols.MODEL_COLS)).assign(is_starter=0)})
        self.assertNotIn("authoritative", plain)
        self.assertNotIn("E_pts_model", plain)


if __name__ == "__main__":
    unittest.main()
