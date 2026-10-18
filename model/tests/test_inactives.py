"""`y_inactive`: the gameday inactive list as a label (model/inactives.py), and the promise that it is nobody's input.

    python -m unittest model.tests.test_inactives

Synthetic frames, no network. What is proven:

  * INA rows (status, not status_description_abbr) become a RawTable with known_at = team kickoff - 90 min; rows with no kickoff
    are dropped and counted
  * y_inactive is keyed on (player_id, season, week): another week's or season's list never marks a row
  * gated at the game's kickoff the list is visible (a pre-lock serve); gated at a Friday clock it is not: never a Friday feature
  * it is in no training family: not a matrix label, not among the 71 serving inputs, and the serve's cache key does not hash it
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from model import build_features as BF
from model import inactives as INA
from model import point_in_time as pit
from model import serve as SV

SUN = pd.Timestamp("2026-10-04 17:00", tz="UTC")
THU = pd.Timestamp("2026-10-02 00:15", tz="UTC")


def raw() -> pd.DataFrame:
    return pd.DataFrame({"season": 2026, "week": [4, 4, 4, 4, 3, 4],
                         "team": ["CAR", "CAR", "KC", "ZZZ", "CAR", "CAR"],
                         "gsis_id": ["coker", "active", "thu", "nokick", "lastwk", None],
                         "position": "WR", "game_type": "REG",
                         "status": ["INA", "ACT", " ina ", "INA", "INA", "INA"],
                         "status_description_abbr": ["A01", "A01", "I01", "A01", "A01", "A01"]})


def kickoffs() -> pd.Series:
    return pd.Series([SUN, THU, SUN - pd.Timedelta(days=7)],
                     index=pd.MultiIndex.from_tuples([(2026, 4, "CAR"), (2026, 4, "KC"), (2026, 3, "CAR")],
                                                     names=["season", "week", "team"]))


def table() -> pit.RawTable:
    df, n_bad = INA.inactive_rows(raw(), kickoffs())
    return pit.RawTable(INA.TABLE, df, INA.RULE, dropped={"no_team_kickoff": n_bad})


class Label(unittest.TestCase):
    def test_rows_and_known_at(self):
        t = table()
        d = t.df.set_index("player_id")
        self.assertEqual(sorted(d.index), ["coker", "lastwk", "thu"])            # ACT, null id and no-kickoff rows are out
        self.assertEqual(t.dropped, {"no_team_kickoff": 1})
        self.assertEqual(d.loc["coker", pit.KNOWN_AT], SUN - pd.Timedelta(minutes=90))
        self.assertEqual(str(t.df[pit.KNOWN_AT].dtype), "datetime64[ns, UTC]")

    def test_y_inactive_is_keyed_on_player_season_and_week(self):
        store = INA.with_inactives(pit.RawStore([]), [2026], table=table())
        spine = pd.DataFrame({"player_id": ["coker", "lastwk", "active", "coker"], "season": [2026, 2026, 2026, 2025],
                              "week": [4, 4, 4, 4]})
        out = INA.attach_inactive_label(store, spine)
        self.assertEqual(out["y_inactive"].tolist(), [1, 0, 0, 0])               # lastwk's list was week 3; 2025 is another season
        self.assertEqual(out["y_inactive"].dtype, np.int8)
        with self.assertRaises(LookupError):
            INA.attach_inactive_label(pit.RawStore([]), spine)

    def test_visible_to_a_pre_lock_serve_never_to_a_friday_one(self):
        store = INA.with_inactives(pit.RawStore([]), [2026], table=table())
        at_kick = INA.known_before(store, SUN, player_id="coker")
        self.assertEqual(at_kick["player_id"].tolist(), ["coker"])
        friday = pd.Timestamp("2026-10-02 22:11", tz="UTC")
        self.assertTrue(INA.known_before(store, friday, player_id="coker").empty)
        sunday_prelock = pd.Timestamp("2026-10-04 15:52", tz="UTC")             # 68 min before the 1 pm slate: posted 22 min earlier
        self.assertEqual(len(INA.known_before(store, sunday_prelock, player_id="coker")), 1)


class NobodysInput(unittest.TestCase):
    def test_not_a_matrix_label_and_not_a_serving_input(self):
        self.assertNotIn("y_inactive", BF.LABELS)
        inputs = json.loads(Path(SV.CONFIG_PATH).read_text(encoding="utf-8"))["inputs"]
        self.assertEqual(len(inputs), 71)
        self.assertFalse([c for c in inputs if "inactive" in c or c.startswith("ina_")])

    def test_the_serves_cache_key_does_not_hash_it(self):
        """labels.py / point_in_time.py are hashed into the history-matrix cache key; a new label there would force a full rebuild
        (~12 min) inside the 10-minute model step. This module is opt-in and outside the key."""
        self.assertNotIn("inactives.py", SV.CODE_FILES)


if __name__ == "__main__":
    unittest.main()
