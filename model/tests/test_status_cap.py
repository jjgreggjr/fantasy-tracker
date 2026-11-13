"""python -m unittest model.tests.test_status_cap

The status-layer cap (model/status_cap.py): synthetic frames only."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from model import serve as SV
from model import status_cap as SC

LEAGUES = [SimpleNamespace(slug="dyn", is_ppr=False), SimpleNamespace(slug="ppr", is_ppr=True)]


def frame(**over):
    row = {"gsis_id": "W", "name": "Emanuel Wilson", "team": "SEA", "position": "RB", "pts_model": 10.6, "pts_model_components": 10.2,
           "pts_dyn": 9.0, "pts_ppr": 10.4, "sleeper_proj": 1.0, "sleeper_proj_dyn": 0.9, "sleeper_proj_ppr": 1.0,
           "e_pts_dyn": 0.7, "e_pts_ppr": 0.8}
    row.update(over)
    return pd.DataFrame([row])


def status(**over):
    row = {"gsis_id": "W", "n_blockers": 3, "blockers_out": 0, "opportunity_ahead": 0.0,
           "blocked_by": "Zach Charbonnet (healthy); Kenneth Walker (healthy, full)"}
    row.update(over)
    return pd.DataFrame([row])


class StatusCap(unittest.TestCase):
    def test_wilson_capped_everywhere(self):
        out, info = SC.apply_cap(frame(), status(), LEAGUES)
        r = out.iloc[0]
        self.assertAlmostEqual(r["pts_ppr"], 1.25)            # max(0.8, 1.0) * 1.25
        self.assertAlmostEqual(r["pts_dyn"], 0.9 * 1.25)      # max(0.7, 0.9) * 1.25
        self.assertAlmostEqual(r["pts_model"], 1.25)
        self.assertAlmostEqual(r["pts_model_components"], 1.25)
        self.assertEqual(r["status_cap"], 1)
        self.assertEqual(r["pts_model_components_raw"], 10.2)
        self.assertEqual((info["eligible"], info["capped"]), (1, 1))
        self.assertIn("Emanuel Wilson SEA RB", info["names"][0])

    def test_overrides(self):
        out, _ = SC.apply_cap(frame(), status(), LEAGUES, slack=2.0)
        self.assertAlmostEqual(out.iloc[0]["pts_ppr"], 2.0)

    def test_ineligible_untouched(self):
        for st in (status(blocked_by="Zach Charbonnet (questionable, limited)"), status(blockers_out=1),
                   status(opportunity_ahead=0.3), status(blocked_by="A (healthy); B (doubtful)")):
            out, info = SC.apply_cap(frame(), st, LEAGUES)
            self.assertEqual(out.iloc[0]["pts_model"], 10.6)
            self.assertEqual(out.iloc[0]["status_cap"], 0)
            self.assertEqual(info["eligible"], 0)
        out, _ = SC.apply_cap(frame(), status(opportunity_ahead=0.05), LEAGUES)
        self.assertEqual(out.iloc[0]["status_cap"], 1)             # at the threshold is eligible

    def test_nobody_untouched(self):
        out, info = SC.apply_cap(frame(), status(n_blockers=0, blocked_by="nobody"), LEAGUES)
        self.assertEqual(out.iloc[0]["pts_model"], 10.6)
        self.assertEqual(info["eligible"], 0)

    def test_never_raised(self):
        f = frame(pts_model=0.5, pts_model_components=0.4, pts_dyn=0.3, pts_ppr=0.2)
        out, info = SC.apply_cap(f, status(), LEAGUES)
        self.assertEqual((out.iloc[0]["pts_model"], out.iloc[0]["pts_dyn"], out.iloc[0]["pts_ppr"]), (0.5, 0.3, 0.2))
        self.assertEqual(out.iloc[0]["status_cap"], 0)
        self.assertEqual((info["eligible"], info["capped"]), (1, 0))

    def test_no_status(self):
        f = frame()
        out, info = SC.apply_cap(f, None, LEAGUES)
        self.assertEqual(out.iloc[0]["pts_model"], 10.6)
        self.assertEqual(out.iloc[0]["status_cap"], 0)
        self.assertEqual(out.iloc[0]["pts_model_components_raw"], 10.2)
        self.assertEqual(info["capped"], 0)

    def test_nan_epts_uses_sleeper(self):
        out, _ = SC.apply_cap(frame(e_pts_dyn=np.nan, e_pts_ppr=np.nan), status(), LEAGUES)
        self.assertAlmostEqual(out.iloc[0]["pts_ppr"], 1.25)
        self.assertAlmostEqual(out.iloc[0]["pts_dyn"], 0.9 * 1.25)
        self.assertAlmostEqual(out.iloc[0]["pts_model"], 1.25)

    def test_both_nan_no_cap(self):
        out, _ = SC.apply_cap(frame(e_pts_dyn=np.nan, sleeper_proj_dyn=np.nan), status(), LEAGUES)
        self.assertEqual(out.iloc[0]["pts_dyn"], 9.0)
        self.assertEqual(out.iloc[0]["pts_ppr"], 1.25)

    def test_missing_status_columns_nan(self):
        out, info = SC.apply_cap(frame(), status(opportunity_ahead=np.nan), LEAGUES)
        self.assertEqual(info["eligible"], 0)

    def test_serve_helpers_warn_and_never_break(self):
        w = []
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(SV.read_status(Path(tmp), w))
            out = SV.status_cap_step(frame(), LEAGUES, Path(tmp), w, log=lambda *_: None)
        self.assertTrue(any("status cap skipped" in x for x in w))
        self.assertEqual(out.iloc[0]["status_cap"], 0)
        bad = frame().drop(columns=["gsis_id"])                       # apply_cap raises: WARN, frame uncapped
        w2 = []
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "data").mkdir()
            status().drop(columns=[]).to_csv(Path(tmp) / "data" / "status.csv", index=False)
            out = SV.status_cap_step(bad, LEAGUES, Path(tmp), w2, log=lambda *_: None)
        self.assertTrue(any("status cap skipped" in x for x in w2))
        self.assertEqual(out.iloc[0]["status_cap"], 0)
        self.assertEqual(out.iloc[0]["pts_model"], 10.6)


if __name__ == "__main__":
    unittest.main()
