"""Phase 5 adoption: an opt-in family (train.OPT_IN_FAMILIES) can be served from serving_config.json `spec.add_families`.

    python -m unittest model.tests.test_serve_families

Synthetic and fast (no model is fitted, nothing is downloaded):

  * `serve.spec_from` reads an optional `add_families` (absent = none: the frozen Phase 3 config behaves as before, 71 inputs)
  * `serve.FAMILY_SOURCES` maps each servable family to its `point_in_time.load_store` switch; an unlisted family is refused
  * the failure contract: a failed ECR download keeps the cached copy (WARN); an unreadable or absent table, or no scrape for the
    served week, serves the frozen `fallback` (the spec without the families) with a WARN naming why; no fallback frozen -> refused
  * the history and quantile-history cache keys change with the family set and the family's historical content, nothing else
  * `--freeze --dry-run` writes nothing; `--freeze` of an add_families spec writes the fallback beside it
"""
from __future__ import annotations

import argparse
import inspect
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from model import features as F
from model import point_in_time as pit
from model import serve as SV
from model import train as T
from model import trial as TR

ECR_COLS = tuple(F.FAMILIES["ecr"])


FIXTURE_71 = Path(__file__).parent / "fixtures" / "serving_config_71.json"


def base_cfg() -> dict:
    """The LIVE serving_config.json (now the 90-input market+ecr primary). Only the tests that pin it use this."""
    return SV.load_config()


def cfg71() -> dict:
    """The previous frozen primary (2026-09-30): 71 inputs, no add_families. Tests about 'the frozen primary with no families' use this."""
    return SV.load_config(FIXTURE_71)


def ecr_cfg() -> dict:
    """The frozen config with ecr adopted, as --freeze would leave it: the ecr columns appended and the 71 kept as the fallback."""
    c = json.loads(json.dumps(cfg71()))
    c["spec"]["add_families"] = ["ecr"]
    c["fallback"] = {"components": c["components"], "inputs": list(c["inputs"])}
    c["components"] = {k: {**v, "n_estimators": v["n_estimators"] + 1} for k, v in c["components"].items()}
    c["inputs"] = [*c["inputs"], *ECR_COLS]
    return c


class FakeStore:
    def __init__(self, **tables):
        self._tables = {k: SimpleNamespace(df=v) for k, v in tables.items()}

    def has(self, name):
        return name in self._tables


def ecr_table(weeks=((2025, 1), (2025, 2), (2026, 5), (2026, 6)), rank=1.0) -> pd.DataFrame:
    return pd.DataFrame({"player_id": ["A"] * len(weeks), "season": [s for s, _ in weeks], "week": [w for _, w in weeks],
                         "ecr_pos_rank": [rank] * len(weeks)})


def frame(with_ecr=True, rows=40) -> pd.DataFrame:
    """A matrix-shaped frame: every registry column that is not weather (opt-in families too when `with_ecr`), varying values."""
    cols = [x for x in F.feature_columns() if F.family_of(x) != "weather"
            and (F.family_of(x) not in T.OPT_IN_FAMILIES or (with_ecr and F.family_of(x) == "ecr"))]
    d = {x: np.linspace(0.0, 1.0, rows) + i for i, x in enumerate(cols)}
    d.update(position=(["QB", "RB", "WR", "TE"] * (rows // 4)), inj_report_status=[None] * rows, inj_practice_status=[None] * rows,
             season=[2021 + i % 4 for i in range(rows)], y_played=[1] * rows)
    d["college_breakout_age"] = np.nan
    return pd.DataFrame(d)


# =========================================================================================== the live config
class LiveConfig(unittest.TestCase):
    """model/serving_config.json as frozen 2026-10-18: the 71 + ecr + market, with the 71 kept as the fallback."""

    def test_the_live_config_adds_market_and_ecr(self):
        self.assertEqual(base_cfg()["spec"]["add_families"], ["market", "ecr"])
        self.assertEqual(SV.spec_from(base_cfg()).add_families, ("market", "ecr"))

    def test_the_live_inputs_are_the_71_then_ecr_then_market(self):
        live, old = base_cfg(), cfg71()
        self.assertEqual(len(old["inputs"]), 71)
        self.assertEqual(live["inputs"], old["inputs"] + list(F.FAMILIES["ecr"]) + list(F.FAMILIES["market"]))
        self.assertEqual(len(live["inputs"]), 90)

    def test_the_live_fallback_is_the_previous_71_primary(self):
        live, old = base_cfg(), cfg71()
        self.assertEqual(live["fallback"]["inputs"], old["inputs"])
        self.assertEqual(live["fallback"]["components"], old["components"])

    def test_the_live_config_was_frozen_on_or_after_the_adoption(self):
        self.assertGreaterEqual(base_cfg()["frozen_on"], "2026-10-18")

    def test_both_adopted_families_have_a_serving_source(self):
        self.assertIn("market", SV.FAMILY_SOURCES)
        self.assertIn("ecr", SV.FAMILY_SOURCES)


# =========================================================================================== the spec
class SpecFromConfig(unittest.TestCase):
    def test_the_frozen_config_has_no_add_families_and_reads_as_none(self):
        c = cfg71()
        self.assertNotIn("add_families", c["spec"])
        self.assertEqual(SV.spec_from(c).add_families, ())
        self.assertEqual(SV.spec_from(c), T.Spec(c["spec"]["name"], drop_cols=tuple(c["spec"]["drop_cols"]),
                                                 drop_families=tuple(c["spec"]["drop_families"])))   # the Phase 3 construction

    def test_add_families_is_read(self):
        s = SV.spec_from(ecr_cfg())
        self.assertEqual(s.add_families, ("ecr",))
        self.assertEqual(s.drop_cols, tuple(cfg71()["spec"]["drop_cols"]))
        self.assertEqual(SV.spec_from({**cfg71(), "spec": {**cfg71()["spec"], "add_families": []}}).add_families, ())

    def test_the_trial_primary_is_the_serve_spec(self):
        for c in (cfg71(), ecr_cfg()):
            self.assertEqual(TR.serving_spec(c), SV.spec_from(c))

    def test_the_frozen_config_still_serves_exactly_its_71_inputs_even_from_a_frame_that_carries_the_ecr_columns(self):
        c = cfg71()
        self.assertEqual(len(c["inputs"]), 71)
        self.assertEqual(SV.check_inputs(frame(with_ecr=True), c), c["inputs"])
        self.assertEqual(SV.check_inputs(frame(with_ecr=False), c), c["inputs"])

    def test_an_ecr_spec_serves_the_71_plus_the_ecr_columns_and_its_fallback_the_71(self):
        c = ecr_cfg()
        self.assertEqual(SV.check_inputs(frame(), c), c["inputs"])
        fb = SV.serving_config_for(c, FakeStore(), 2026, 6, [])
        self.assertEqual(SV.check_inputs(frame(), fb), cfg71()["inputs"])
        with self.assertRaises(SV.ServingSpecMismatch):              # the ecr inputs against a frame without them: refused, not dropped
            SV.check_inputs(frame(with_ecr=False), c)


# =========================================================================================== the registry
class Registry(unittest.TestCase):
    def test_every_servable_family_is_an_opt_in_family_with_a_real_store_switch(self):
        kwargs = set(inspect.signature(pit.load_store).parameters)
        for fam, src in SV.FAMILY_SOURCES.items():
            self.assertIn(fam, T.OPT_IN_FAMILIES)
            if src.store_kwarg is not None:
                self.assertIn(src.store_kwarg, kwargs)
                self.assertIsNotNone(src.table)
        self.assertEqual((SV.FAMILY_SOURCES["ecr"].store_kwarg, SV.FAMILY_SOURCES["ecr"].table), ("ecr", "ecr"))

    def test_an_unlisted_family_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            SV.family_source("pbp_part")
        self.assertIn("FAMILY_SOURCES", str(cm.exception))

    def load(self, families, *, refresh_ok=True, load_ok=True, refresh=True):
        src = SV.FAMILY_SOURCES["ecr"]
        warns: list[str] = []
        r = mock.Mock(return_value="ecr: refreshed") if refresh_ok else mock.Mock(side_effect=RuntimeError("could not fetch x: 503"))
        ld = mock.Mock(return_value=object()) if load_ok else mock.Mock(side_effect=OSError("Parquet magic bytes not found"))
        with mock.patch.object(src, "refresh", r), mock.patch.object(src, "load", ld), \
                mock.patch.object(pit, "load_store", return_value="STORE") as ls, mock.patch.object(SV, "refresh_raw"), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(SV.load_store(2026, refresh=refresh, families=families, warnings=warns), "STORE")
        return ls.call_args.kwargs, warns, r

    def test_the_base_store_is_loaded_exactly_as_before(self):
        kw, warns, r = self.load(())
        self.assertEqual(kw, dict(adp=False, college=False, pbp=False))
        self.assertEqual(warns, [])
        r.assert_not_called()

    def test_ecr_turns_on_the_ecr_table_and_refreshes_its_source(self):
        kw, warns, r = self.load(("ecr",))
        self.assertEqual(kw, dict(adp=False, college=False, pbp=False, ecr=True))
        self.assertEqual(warns, [])
        r.assert_called_once()

    def test_no_refresh_means_no_download(self):
        _, _, r = self.load(("ecr",), refresh=False)
        r.assert_not_called()

    def test_a_family_built_from_the_base_tables_needs_no_switch(self):
        kw, warns, _ = self.load(("streak",))
        self.assertEqual(kw, dict(adp=False, college=False, pbp=False))

    def test_a_failed_download_keeps_the_cached_table_with_a_warn(self):
        kw, warns, _ = self.load(("ecr",), refresh_ok=False)
        self.assertTrue(kw["ecr"])
        self.assertEqual(len(warns), 1)
        self.assertIn("could not refresh", warns[0])
        self.assertIn("503", warns[0])

    def test_an_unreadable_table_is_left_out_with_a_warn(self):
        kw, warns, _ = self.load(("ecr",), refresh_ok=False, load_ok=False)
        self.assertNotIn("ecr", kw)
        self.assertEqual(len(warns), 2)
        self.assertIn("cannot be loaded", warns[1])
        self.assertIn("Parquet magic bytes", warns[1])


# =========================================================================================== the fallback
class Fallback(unittest.TestCase):
    def test_the_frozen_config_is_served_as_is(self):
        c, warns = cfg71(), []
        self.assertIs(SV.serving_config_for(c, FakeStore(), 2026, 6, warns), c)
        self.assertEqual(warns, [])

    def test_an_ecr_table_covering_the_week_serves_the_ecr_spec(self):
        c, warns = ecr_cfg(), []
        self.assertIs(SV.serving_config_for(c, FakeStore(ecr=ecr_table()), 2026, 6, warns), c)
        self.assertEqual(warns, [])

    def test_no_ecr_table_serves_the_fallback_and_says_so(self):
        c, warns = ecr_cfg(), []
        got = SV.serving_config_for(c, FakeStore(), 2026, 6, warns)
        self.assertEqual(got["inputs"], c["fallback"]["inputs"])
        self.assertEqual(got["components"], c["fallback"]["components"])
        self.assertEqual(SV.spec_from(got).add_families, ())
        self.assertEqual(len(warns), 1)
        self.assertIn("no table in the store", warns[0])
        self.assertIn("WITHOUT ecr", warns[0])
        self.assertIn("71 inputs", warns[0])

    def test_no_scrape_for_the_served_week_serves_the_fallback_and_says_the_columns_would_be_empty(self):
        c, warns = ecr_cfg(), []
        got = SV.serving_config_for(c, FakeStore(ecr=ecr_table()), 2026, 7, warns)        # a Tuesday: week 7's Friday scrape is not out
        self.assertEqual(got["inputs"], c["fallback"]["inputs"])
        self.assertIn("no rows for 2026 week 7", warns[0])
        self.assertIn("ecr columns would be empty for every row of this serve", warns[0])
        self.assertIn("Fridays 00:03 UTC", warns[0])

    def test_a_fallback_that_was_never_frozen_is_refused(self):
        c = ecr_cfg()
        del c["fallback"]
        with self.assertRaises(RuntimeError) as cm:
            SV.serving_config_for(c, FakeStore(), 2026, 6, [])
        self.assertIn("--freeze", str(cm.exception))

    def test_a_failed_fetch_in_the_real_run_path_is_a_warn_and_the_serve_proceeds_with_the_primary(self):
        """run() end to end with the heavy steps stubbed: the ECR download raises and no cached table exists -> the store has no ecr
        table, predict_week receives the fallback config (the 71 inputs), the serve finishes and prints the WARN."""
        src, seen = SV.FAMILY_SOURCES["ecr"], {}
        kick = pd.Timestamp("2026-10-18 17:00", tz="UTC")

        def predict(df, season, week, cfg, leagues=(), **kw):
            seen["cfg"] = cfg
            return pd.DataFrame(index=df.index), {}, {}
        out = pd.DataFrame({"gsis_id": ["A"], "position": ["WR"], "report_status": [None], "practice_status": [None]})
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "config.json").write_text('{"season": 2026}')
            args = argparse.Namespace(season=None, week=None, now="2026-10-17 12:00", dry_run=True, out=None, jobs=1, rebuild=False,
                                      no_refresh=False, config=None)
            with mock.patch.object(SV, "load_config", return_value=ecr_cfg()), mock.patch.object(SV, "refresh_raw"), \
                    mock.patch.object(src, "refresh", side_effect=RuntimeError("could not fetch https://raw.githubusercontent.com/x: 503")), \
                    mock.patch.object(src, "load", return_value=None), \
                    mock.patch.object(pit, "load_store", return_value=FakeStore()) as ls, \
                    mock.patch.object(SV, "pick_week", return_value=(6, [SimpleNamespace(kickoff=kick, game_id="G")])), \
                    mock.patch.object(SV, "unpublished_games", return_value=[]), \
                    mock.patch.object(SV, "training_frame", return_value=pd.DataFrame({"x": [1.0]})), \
                    mock.patch.object(SV, "serve_frame", return_value=pd.DataFrame({"x": [2.0]})), \
                    mock.patch("model.leaguescore.load_leagues", return_value=[]), \
                    mock.patch.object(SV, "join_frames", side_effect=lambda h, s, l: pd.concat([h, s], ignore_index=True)), \
                    mock.patch.object(SV, "predict_week", side_effect=predict), \
                    mock.patch.object(SV, "assemble", return_value=(out, pd.Series(dtype=object))), redirect_stdout(io.StringIO()):
                lines = SV.run(args, root=root)
        self.assertNotIn("ecr", ls.call_args.kwargs)
        self.assertEqual(seen["cfg"]["inputs"], cfg71()["inputs"])
        self.assertEqual(SV.spec_from(seen["cfg"]).add_families, ())
        warn = [l for l in lines if l.startswith("  WARN")]
        self.assertTrue(any("could not refresh" in w and "503" in w for w in warn), warn)
        self.assertTrue(any("WITHOUT ecr" in w for w in warn), warn)
        self.assertIn("  (dry run: nothing written)", lines)

    def test_the_refresh_never_touches_the_cached_copy_when_the_download_fails(self):
        from model import fetch_ecr as FE
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            (cache / FE.OUT).write_bytes(b"the cached copy")
            with mock.patch.object(FE, "ECR_DIR", cache), \
                    mock.patch.object(FE, "download", side_effect=RuntimeError("could not fetch x: 503")):
                with self.assertRaises(RuntimeError):
                    SV.refresh_ecr()
            self.assertEqual((cache / FE.OUT).read_bytes(), b"the cached copy")
            self.assertEqual(sorted(p.name for p in cache.iterdir()), [FE.OUT])          # no scratch folder left behind

    def test_the_refresh_refuses_a_rebuild_that_lost_seasons(self):
        """A 2026-only rebuild once replaced the 2021-2026 table and the freeze dropped the whole family as empty."""
        import pandas as pd
        from model import fetch_ecr as FE
        from model import point_in_time as pit
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            full = pd.DataFrame({"season": [2021, 2025, 2026], "scrape_date": ["2021-09-10", "2025-09-12", "2026-09-11"]})
            full.to_parquet(cache / FE.OUT, index=False)
            only_now = pd.DataFrame({"season": [2026], "scrape_date": ["2026-10-16"]})
            with mock.patch.object(FE, "ECR_DIR", cache), \
                    mock.patch.object(FE, "download", side_effect=lambda ecr_dir: (ecr_dir / "a.parquet", ecr_dir / "b.csv")), \
                    mock.patch.object(FE, "build", return_value=(only_now, {})), \
                    mock.patch.object(pit, "_games", return_value=pd.DataFrame()), \
                    mock.patch("pandas.read_csv", return_value=pd.DataFrame()), \
                    mock.patch("pandas.read_parquet", side_effect=lambda p, **k: full[["season"]] if Path(p).name == FE.OUT else pd.DataFrame()):
                with self.assertRaises(ValueError) as cm:
                    SV.refresh_ecr()
            self.assertIn("lost seasons [2021, 2025]", str(cm.exception))
            self.assertEqual(pd.read_parquet(cache / FE.OUT)["season"].tolist(), [2021, 2025, 2026])   # untouched


# =========================================================================================== cache keys
class CacheKeys(unittest.TestCase):
    def test_the_base_history_file_keeps_its_phase_3_name(self):
        self.assertEqual(SV.history_file(FakeStore(), 2026), f"hist_{SV.code_hash(2026)}.parquet")

    def test_the_ecr_table_changes_the_history_file_and_only_its_history_rows_matter(self):
        base = SV.history_file(FakeStore(), 2026)
        with_ecr = SV.history_file(FakeStore(ecr=ecr_table()), 2026)
        self.assertNotEqual(base, with_ecr)
        self.assertTrue(with_ecr.startswith("hist+ecr_"))
        # a new scrape of THIS season (the weekly refresh) does not invalidate the cached 2021-2025 frame...
        self.assertEqual(SV.history_file(FakeStore(ecr=ecr_table(weeks=((2025, 1), (2025, 2), (2026, 5), (2026, 6), (2026, 7)))), 2026),
                         with_ecr)
        # ...a change to a cached season does
        self.assertNotEqual(SV.history_file(FakeStore(ecr=ecr_table(rank=2.0)), 2026), with_ecr)

    def test_the_two_histories_do_not_prune_each_other(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            a, b = cache / SV.history_file(FakeStore(), 2026), cache / SV.history_file(FakeStore(ecr=ecr_table()), 2026)
            for f in (a, b, cache / "hist_000000000000.parquet", cache / "hist+ecr_000000000000.parquet"):
                f.write_bytes(b"x")
            for f in (a, b):
                SV.prune_cache(cache, f.name.rsplit("_", 1)[0], f)
            self.assertEqual(sorted(p.name for p in cache.iterdir()), sorted([a.name, b.name]))

    def qhist_file(self, spec, df) -> str:
        params = {"quantile": {"0.1": {"n_estimators": 5}}}
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(SV, "quantile_walk", return_value=pd.DataFrame({"q10": [1.0]})):
            SV.quantile_history(df, spec, 2026, params, cache_dir=Path(d), log=lambda *_: None)
            return next(Path(d).iterdir()).name

    def test_the_quantile_history_key_follows_add_families_and_their_content_and_the_base_key_is_unchanged(self):
        df = frame().assign(season=lambda x: 2024 + x.index % 3, y_points_ppr=1.0)
        base, ecr = SV.spec_from(cfg71()), SV.spec_from(ecr_cfg())
        labels = int(pd.util.hash_pandas_object(df.loc[df["season"] == 2025, T.TARGET], index=False).sum() % (2 ** 63))
        old_key = SV.code_hash(2026, json.dumps([{"0.1": {"n_estimators": 5}}, list(base.drop_cols), list(base.drop_families), T.TARGET, labels],
                                                sort_keys=True))
        self.assertEqual(self.qhist_file(base, df), f"qhist-points-ppr_{old_key}.parquet")       # the formula before add_families
        f_ecr = self.qhist_file(ecr, df)
        self.assertTrue(f_ecr.startswith("qhist-points-ppr+ecr_"))
        self.assertNotEqual(f_ecr, self.qhist_file(ecr, df.assign(ecr_rank=df["ecr_rank"] + 1)))
        self.assertEqual(self.qhist_file(base, df.assign(ecr_rank=df["ecr_rank"] + 1)), f"qhist-points-ppr_{old_key}.parquet")


# =========================================================================================== freezing
class Freeze(unittest.TestCase):
    COUNTS = {"y_tgt": {"n_estimators": 100, "best_iteration": 91, "valid_rmse": 2.0}}

    def freeze(self, cfg: dict, dry: bool, refit: bool = False):
        self.fitted = []

        def comps(df, spec, dead):
            self.fitted.append(spec.add_families)
            return self.COUNTS, list(cfg71()["inputs"]) + (list(ECR_COLS) if spec.add_families else [])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "serving_config.json"
            p.write_text(json.dumps(cfg, indent=1))
            before = p.read_bytes()
            args = argparse.Namespace(season=2026, jobs=1, dry_run=dry, config=str(p), refit_fallback=refit)
            with mock.patch.object(SV, "load_store", return_value=FakeStore(ecr=ecr_table())) as ls, \
                    mock.patch.object(SV, "training_frame", return_value=frame().assign(y_misc_pts=0.0)), \
                    mock.patch("model.components.add_misc", side_effect=lambda x: x), \
                    mock.patch.object(SV, "freeze_components", side_effect=comps), \
                    mock.patch.object(SV, "CONFIG_PATH", Path(d) / "never_written.json"):
                lines = SV.freeze(args, root=Path(d))
            self.assertFalse((Path(d) / "never_written.json").exists())
            return lines, before, p.read_bytes(), ls.call_args.kwargs

    def test_a_dry_run_writes_nothing_and_prints_the_input_count(self):
        c = json.loads(json.dumps(cfg71()))
        c["spec"]["add_families"] = ["ecr"]
        lines, before, after, kw = self.freeze(c, dry=True)
        self.assertEqual(before, after)
        self.assertEqual(kw["families"], ("ecr",))
        self.assertEqual(kw["refresh"], False)                                   # freezing never downloads
        self.assertTrue(lines[0].startswith(f"would write serving_config.json: {71 + len(ECR_COLS)} inputs (71 in the fallback"), lines[0])
        self.assertTrue(any("dry run: nothing written" in l for l in lines))

    def test_freezing_an_ecr_spec_keeps_the_primary_that_served_as_its_fallback(self):
        c = json.loads(json.dumps(cfg71()))
        c["spec"]["add_families"] = ["ecr"]
        _, _, after, _ = self.freeze(c, dry=False)
        got = json.loads(after)
        self.assertEqual(got["inputs"], cfg71()["inputs"] + list(ECR_COLS))
        self.assertEqual(got["components"], self.COUNTS)
        self.assertEqual(got["fallback"]["inputs"], cfg71()["inputs"])
        self.assertEqual(got["fallback"]["components"], cfg71()["components"])     # carried over, not re-fit
        self.assertEqual(got["fallback"]["frozen_on"], cfg71()["frozen_on"])
        self.assertEqual(self.fitted, [("ecr",)])
        # a second freeze of the adopted config keeps that fallback too
        _, _, again, _ = self.freeze(got, dry=False)
        self.assertEqual(json.loads(again)["fallback"]["components"], cfg71()["components"])

    def test_refit_fallback_freezes_it_afresh(self):
        c = json.loads(json.dumps(cfg71()))
        c["spec"]["add_families"] = ["ecr"]
        _, _, after, _ = self.freeze(c, dry=False, refit=True)
        self.assertEqual(json.loads(after)["fallback"]["components"], self.COUNTS)
        self.assertEqual(sorted(self.fitted), [(), ("ecr",)])

    def test_freezing_the_base_spec_writes_no_fallback(self):
        c = ecr_cfg()
        c["spec"].pop("add_families")
        _, _, after, _ = self.freeze(c, dry=False)
        self.assertNotIn("fallback", json.loads(after))

    def test_a_family_whose_table_is_missing_cannot_be_frozen(self):
        c = json.loads(json.dumps(cfg71()))
        c["spec"]["add_families"] = ["ecr"]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text(json.dumps(c))
            with mock.patch.object(SV, "load_store", return_value=FakeStore()):
                with self.assertRaises(RuntimeError):
                    SV.freeze(argparse.Namespace(season=2026, jobs=1, dry_run=False, config=str(p)), root=Path(d))
            self.assertEqual(json.loads(p.read_text()), c)


# =========================================================================================== market (Sleeper/Rotowire projection)
from model import fetch_sleeper_proj as SP                                                    # noqa: E402

MKT_KO = pd.Timestamp("2026-10-18 17:00", tz="UTC")
MKT_TNF = pd.Timestamp("2026-10-16 00:15", tz="UTC")


def mkt_games() -> pd.DataFrame:
    return pd.DataFrame({"season": [2026, 2026, 2026], "week": [6, 6, 7], "game_type": ["REG"] * 3,
                         "home_team": ["KC", "PIT", "KC"], "away_team": ["BUF", "CIN", "DEN"],
                         "game_id": ["2026_06_BUF_KC", "2026_06_CIN_PIT", "2026_07_DEN_KC"],
                         "kickoff": [MKT_KO, MKT_TNF, MKT_KO + pd.Timedelta(days=7)]})


def projections_csv_fixture() -> pd.DataFrame:
    """data/projections.csv as ff/projections.py writes it (its exact columns), source 'sleeper', week 6 and an older week 5."""
    cols = ["season", "week", "gsis_id", "source", "pulled_at", "team", "position", "proj_pass_att", "proj_pass_yds", "proj_pass_tds",
            "proj_pass_int", "proj_carries", "proj_rush_yds", "proj_rush_tds", "proj_targets", "proj_rec", "proj_rec_yds",
            "proj_rec_tds", "proj_pts_ppr", "proj_pts_half", "proj_pts_std", "proj_opps", "proj_carry_share", "proj_target_share"]
    rows = [
        # a QB: passing and rushing lines
        [2026, 6, "00-0000001", "sleeper", "2026-10-17T12:00:00Z", "KC", "QB", 34.0, 260.0, 1.9, 0.6, 4.0, 22.0, 0.2,
         np.nan, np.nan, np.nan, np.nan, 21.5, 21.5, 21.5, 4.0, 0.2, np.nan],
        # a WR: receiving lines
        [2026, 6, "00-0000002", "sleeper", "2026-10-17T12:00:00Z", "BUF", "WR", np.nan, np.nan, np.nan, np.nan, 0.5, 3.0, 0.0,
         8.5, 6.1, 80.0, 0.5, 17.1, 14.05, 11.0, 9.0, 0.03, 0.27],
        # no pts_ppr: dropped, like parse_week
        [2026, 6, "00-0000003", "sleeper", "2026-10-17T12:00:00Z", "KC", "WR", *[np.nan] * 14, 0.0, np.nan, np.nan],
        # a Thursday-night player: his game kicked off before the pull, so the gate never hands him this row
        [2026, 6, "00-0000004", "sleeper", "2026-10-17T12:00:00Z", "PIT", "RB", np.nan, np.nan, np.nan, np.nan, 15.0, 65.0, 0.5,
         3.0, 2.4, 18.0, 0.1, 14.0, 12.8, 11.6, 18.0, 0.6, 0.1],
        [2026, 5, "00-0000001", "sleeper", "2026-10-10T12:00:00Z", "KC", "QB", 30.0, 230.0, 1.5, 0.7, 3.0, 15.0, 0.1,
         np.nan, np.nan, np.nan, np.nan, 18.0, 18.0, 18.0, 3.0, 0.2, np.nan],
        [2026, 6, "00-0000005", "fantasypros", "2026-10-17T12:00:00Z", "KC", "TE", *[1.0] * 17],             # not Sleeper: ignored
    ]
    return pd.DataFrame(rows, columns=cols)


class MarketSource(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "sleeper_proj"
        self.cache.mkdir()
        self.archive = self.cache / "proj_2026_05.parquet"
        self.archive.write_bytes(b"the archived week 5")
        self.csv = Path(self.tmp.name) / "projections.csv"
        projections_csv_fixture().to_csv(self.csv, index=False)
        self.patches = [mock.patch.object(SP, "CACHE", self.cache), mock.patch.object(pit, "_games", side_effect=mkt_games)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    # ---------------------------------------------------------------- the registry
    def test_market_is_servable_through_the_sleeper_proj_switch(self):
        src = SV.family_source("market")
        self.assertEqual((src.store_kwarg, src.table), ("sleeper_proj", "sleeper_proj"))
        self.assertIn("sleeper_proj", inspect.signature(pit.load_store).parameters)
        self.assertIs(src.refresh, SV.refresh_market)
        self.assertIsNotNone(src.extend)
        self.assertIn("market", T.OPT_IN_FAMILIES)

    def test_load_store_turns_on_sleeper_proj_refreshes_with_the_serve_clock_and_extends(self):
        src, warns = SV.FAMILY_SOURCES["market"], []
        r, ext = mock.Mock(return_value="market: ok"), mock.Mock(return_value="EXTENDED")
        now = pd.Timestamp("2026-10-17 12:00", tz="UTC")
        ecr = SV.FAMILY_SOURCES["ecr"]           # stubbed too: the real refresh would download and rebuild the real ECR cache
        with mock.patch.object(src, "refresh", r), mock.patch.object(src, "load", return_value=object()), \
                mock.patch.object(src, "extend", ext), mock.patch.object(pit, "load_store", return_value="STORE") as ls, \
                mock.patch.object(ecr, "refresh", mock.Mock(return_value="ecr: ok")), mock.patch.object(ecr, "load", return_value=object()), \
                mock.patch.object(SV, "refresh_raw"), redirect_stdout(io.StringIO()):
            got = SV.load_store(2026, families=("market", "ecr"), warnings=warns, now=now, week=None)
        self.assertEqual(got, "EXTENDED")
        self.assertTrue(ls.call_args.kwargs["sleeper_proj"])
        self.assertEqual(r.call_args.kwargs, dict(season=2026, now=now, week=None))
        self.assertEqual(ext.call_args.args[0], "STORE")
        self.assertEqual(warns, [])

    # ---------------------------------------------------------------- data/projections.csv -> the archive's columns
    def test_every_stat_the_market_family_reads_maps_from_a_projections_csv_column(self):
        self.assertEqual(set(SP.PROJECTIONS_CSV_STATS), set(SP.STAT_COLS))
        read = {v for k, v in F.MARKET_STATS.items() if k != "mkt_pos_rank"} | {"pass_td", "rush_td", "rec_td"}
        self.assertLessEqual(read, set(SP.PROJECTIONS_CSV_STATS))
        from ff.projections import ALIASES                      # the pipeline writes exactly these columns
        self.assertLessEqual(set(SP.PROJECTIONS_CSV_STATS.values()), set(ALIASES))

    def test_the_projections_csv_rows_map_onto_the_archive_columns(self):
        raw, pulled = SP.from_projections_csv(projections_csv_fixture(), 2026, 6)
        self.assertEqual(pulled, pd.Timestamp("2026-10-17 12:00", tz="UTC"))
        self.assertEqual(sorted(raw["player_id"]), ["00-0000001", "00-0000002", "00-0000004"])   # no pts / not Sleeper / week 5: out
        qb, wr = raw.set_index("player_id").loc["00-0000001"], raw.set_index("player_id").loc["00-0000002"]
        self.assertEqual((qb["pass_att"], qb["pass_yd"], qb["pass_td"], qb["rush_att"], qb["pts_ppr"]), (34.0, 260.0, 1.9, 4.0, 21.5))
        self.assertEqual((wr["rec_tgt"], wr["rec"], wr["rec_yd"], wr["rec_td"], wr["pts_std"], wr["pts_half_ppr"]),
                         (8.5, 6.1, 80.0, 0.5, 11.0, 14.05))
        self.assertTrue(np.isnan(wr["pass_att"]))
        self.assertIsNone(SP.from_projections_csv(projections_csv_fixture(), 2026, 7))            # week not pulled
        self.assertIsNone(SP.from_projections_csv(projections_csv_fixture().drop(columns=["proj_targets"]), 2026, 6))

    # ---------------------------------------------------------------- the current-week frame
    def test_the_current_week_is_stamped_with_its_read_time_and_never_written_into_the_archive(self):
        before = sorted(p.name for p in self.cache.iterdir())
        df, how = SP.fetch_current_week(2026, 6, projections=self.csv)
        self.assertIn("data/projections.csv", how)
        self.assertTrue((df[pit.KNOWN_AT] == pd.Timestamp("2026-10-17 12:00", tz="UTC")).all())     # the pull, not kickoff - 1h
        self.assertEqual(str(df[pit.KNOWN_AT].dtype), "datetime64[ns, UTC]")
        k = df.set_index("player_id")
        self.assertEqual(k.loc["00-0000001", "game_id"], "2026_06_BUF_KC")
        self.assertEqual(k.loc["00-0000001", "pos_rank"], 1.0)
        self.assertEqual(self.archive.read_bytes(), b"the archived week 5")
        self.assertEqual(sorted(p.name for p in self.cache.iterdir()), sorted(before + ["current_2026_06.parquet"]))
        self.assertFalse(list(self.cache.glob("proj_2026_06*")))
        self.assertFalse(SP.current_path(2026, 6).match("proj_*"))
        # overwritten by the next serve
        p2 = projections_csv_fixture().assign(proj_pts_ppr=lambda d: d["proj_pts_ppr"] + 1, pulled_at="2026-10-18T09:00:00Z")
        p2.to_csv(self.csv, index=False)
        SP.fetch_current_week(2026, 6, projections=self.csv)
        got = pd.read_parquet(SP.current_path(2026, 6)).set_index("player_id")
        self.assertEqual(got.loc["00-0000001", "pts_ppr"], 22.5)
        self.assertEqual(len(list(self.cache.glob("current_*"))), 1)

    def test_without_the_week_in_projections_csv_the_endpoint_is_read_and_stamped_now(self):
        now = pd.Timestamp("2026-10-17 15:30", tz="UTC")
        pd.DataFrame({"sleeper_id": ["4046"], "gsis_id": ["00-0000001"], "id_source": ["x"]}).to_parquet(self.cache / "id_map.parquet")
        resp = mock.Mock(**{"json.return_value": [{"player_id": "4046", "team": "KC", "opponent": "BUF", "company": "rotowire",
                                                   "player": {"position": "QB"}, "stats": {"pts_ppr": 20.0, "pass_att": 33.0}}],
                            "raise_for_status.return_value": None})
        sess = mock.Mock(**{"get.return_value": resp})
        df, how = SP.fetch_current_week(2026, 7, projections=self.csv, now=now, session=sess)   # the csv has no week 7
        self.assertIn("endpoint", how)
        self.assertIn("/2026/7?", sess.get.call_args.args[0])
        self.assertEqual(df[pit.KNOWN_AT].tolist(), [now])
        self.assertTrue(SP.current_path(2026, 7).exists())
        self.assertEqual(self.archive.read_bytes(), b"the archived week 5")

    def test_with_current_appends_the_week_the_archive_lacks_and_the_archive_wins_once_it_has_it(self):
        SP.fetch_current_week(2026, 6, projections=self.csv)
        arch = pd.DataFrame({"player_id": ["00-0000001"], "season": [2026], "week": [5], "pts_ppr": [18.0],
                             pit.KNOWN_AT: [pd.Timestamp("2026-10-11 16:00", tz="UTC")]})
        t = SP.with_current(pit.RawTable("sleeper_proj", arch, "test"), [2026])
        self.assertEqual(t.df.groupby("week").size().to_dict(), {5: 1, 6: 3})
        self.assertEqual(t.dropped["current_week_rows_added"], 3)
        self.assertTrue(t.df[pit.KNOWN_AT].is_monotonic_increasing)
        settled = pd.concat([arch, arch.assign(week=6, pts_ppr=30.0)], ignore_index=True)
        t2 = SP.with_current(pit.RawTable("sleeper_proj", settled, "test"), [2026])
        self.assertEqual(t2.df.loc[t2.df["week"] == 6, "pts_ppr"].tolist(), [30.0])                # the archived final version
        self.assertIsNone(SP.with_current(None, [2025]))                                           # nothing anywhere: still None
        self.assertFalse(inspect.signature(SP.load_sleeper_proj_table).parameters["current"].default)

    def test_the_gate_hands_the_current_row_to_a_game_after_the_read_and_never_to_one_before(self):
        df, _ = SP.fetch_current_week(2026, 6, projections=self.csv)
        st = pit.RawStore([pit.RawTable("sleeper_proj", df, "test")])
        self.assertEqual(len(pit.as_of_join(st, "sleeper_proj", MKT_KO, player_id="00-0000001")), 1)
        self.assertEqual(len(pit.as_of_join(st, "sleeper_proj", MKT_TNF, player_id="00-0000004")), 0)

    # ---------------------------------------------------------------- the failure contract
    def test_a_failed_read_raises_from_the_refresh_after_trying_the_archive(self):
        with mock.patch.object(SP, "fetch", return_value={"fetched": 0, "cached": 5}) as fa, \
                mock.patch.object(SP, "fetch_current_week", side_effect=RuntimeError("503 Service Unavailable")):
            with self.assertRaises(RuntimeError) as cm:
                SV.refresh_market(season=2026, now=pd.Timestamp("2026-10-17 12:00", tz="UTC"))
        self.assertEqual(list(fa.call_args.args[1]), [1, 2, 3, 4, 5])                    # settled weeks before the served one
        self.assertIn("week 6 projection failed", str(cm.exception))
        self.assertIn("503", str(cm.exception))
        self.assertFalse(list(self.cache.glob("current_*")))

    def test_a_failed_read_is_a_warn_and_the_serve_uses_the_fallback(self):
        """load_store end to end with pit.load_store stubbed: the refresh fails, the store has the archive only (no week 6 row),
        so serving_config_for serves the frozen 71 with a WARN naming market."""
        src, warns = SV.FAMILY_SOURCES["market"], []
        arch = pd.DataFrame({"player_id": ["A"], "season": [2026], "week": [5], "pts_ppr": [1.0],
                             pit.KNOWN_AT: [pd.Timestamp("2026-10-11 16:00", tz="UTC")]})
        store = pit.RawStore([pit.RawTable("sleeper_proj", arch, "test")])
        with mock.patch.object(SP, "fetch", side_effect=RuntimeError("could not fetch: 503")), \
                mock.patch.object(src, "load", return_value=object()), \
                mock.patch.object(pit, "load_store", return_value=store), mock.patch.object(SV, "refresh_raw"), \
                redirect_stdout(io.StringIO()):
            with mock.patch.object(SP, "fetch_current_week", side_effect=RuntimeError("503")):
                got = SV.load_store(2026, families=("market",), warnings=warns, now=pd.Timestamp("2026-10-17 12:00", tz="UTC"))
        self.assertEqual(len(warns), 1)
        self.assertIn("market: could not refresh", warns[0])
        c = json.loads(json.dumps(cfg71()))
        c["spec"]["add_families"] = ["market"]
        c["fallback"] = {"components": c["components"], "inputs": list(c["inputs"])}
        c["inputs"] = [*c["inputs"], *F.FAMILIES["market"]]
        cfg = SV.serving_config_for(c, got, 2026, 6, warns)
        self.assertEqual(cfg["inputs"], cfg71()["inputs"])
        self.assertEqual(len(cfg["inputs"]), 71)
        self.assertIn("market: no rows for 2026 week 6", warns[-1])
        self.assertIn("WITHOUT market", warns[-1])
        # with the week's current frame present the market spec is served as frozen
        SP.fetch_current_week(2026, 6, projections=self.csv)
        warns2 = []
        self.assertIs(SV.serving_config_for(c, SV._extend_market(store, [2026]), 2026, 6, warns2), c)
        self.assertEqual(warns2, [])

    def test_the_frozen_config_is_untouched_by_the_market_entry(self):
        c = cfg71()
        self.assertEqual(len(c["inputs"]), 71)
        self.assertEqual(SV.spec_from(c).add_families, ())
        self.assertFalse(any(x.startswith("mkt_") for x in c["inputs"]))
        self.assertEqual(SV.check_inputs(frame(with_ecr=True), c), c["inputs"])

    def test_the_market_table_joins_the_history_cache_key_by_its_cached_seasons_only(self):
        mk = lambda wk, v=1.0: pd.DataFrame({"player_id": ["A"] * len(wk), "season": [s for s, _ in wk],
                                             "week": [w for _, w in wk], "pts_ppr": [v] * len(wk)})
        a = SV.history_file(FakeStore(ecr=ecr_table(), sleeper_proj=mk([(2025, 1), (2026, 5)])), 2026)
        self.assertTrue(a.startswith("hist+ecr+market_"), a)
        self.assertEqual(SV.history_file(FakeStore(ecr=ecr_table(), sleeper_proj=mk([(2025, 1), (2026, 5), (2026, 6)])), 2026), a)
        self.assertNotEqual(SV.history_file(FakeStore(ecr=ecr_table(), sleeper_proj=mk([(2025, 1), (2026, 5)], 2.0)), 2026), a)


# =========================================================================================== the trial's printed steps
class AdoptionSteps(unittest.TestCase):
    def test_the_ecr_steps_are_the_json_edit_then_the_freeze(self):
        cfg = cfg71()
        steps = TR.adoption_steps(TR.compose(["+family:ecr"], TR.serving_spec(cfg)), cfg, "average")
        self.assertIn('"add_families": ["ecr"]', steps[0])
        self.assertNotIn("drop_cols", steps[0])                                 # nothing else in the spec changes
        self.assertTrue(any("python -m model.fetch_ecr" in s for s in steps))
        self.assertTrue(any(s.endswith("then python -m model.serve --freeze") for s in steps))
        self.assertFalse(any("must learn" in s for s in steps))

    def test_adopting_a_family_the_live_spec_already_has_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            TR.compose(["+family:ecr"], TR.serving_spec(base_cfg()))
        self.assertIn("already added", str(cm.exception))

    def test_the_json_edit_the_steps_name_is_one_spec_from_reads_back_as_the_variant(self):
        cfg = cfg71()
        v = TR.compose(["+family:ecr"], TR.serving_spec(cfg))
        edited = {**cfg, "spec": {**cfg["spec"], "add_families": ["ecr"]}}
        self.assertEqual(SV.spec_from(edited).add_families, v.spec.add_families)
        self.assertEqual(SV.spec_from(edited).drop_cols, v.spec.drop_cols)


if __name__ == "__main__":
    unittest.main()
