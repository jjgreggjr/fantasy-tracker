"""The weekly workflow runs the model beside the pipeline and can never be hurt by it.

    python3.12 -m unittest model.tests.test_workflow

The step list of .github/workflows/pipeline.yml, read as text (no YAML dependency): the model steps sit between the existing
pipeline step and the commit, use Python 3.12 with a pip cache keyed on model/requirements.txt, are failure-isolated and capped
in time, and the existing pipeline step, its failure gate and its checkout are untouched.
"""
from __future__ import annotations

import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class Workflow(unittest.TestCase):
    """The step list of .github/workflows/pipeline.yml, as text (no YAML dependency)."""

    @classmethod
    def setUpClass(cls):
        cls.text = (REPO / ".github" / "workflows" / "pipeline.yml").read_text(encoding="utf-8")
        cls.steps = cls.text.split("\n      - ")

    def step(self, needle: str) -> str:
        hits = [s for s in self.steps if needle in s]
        self.assertEqual(len(hits), 1, needle)
        return hits[0]

    def test_the_model_steps_sit_between_the_pipeline_and_the_commit(self):
        order = [i for i, s in enumerate(self.steps) if "run: python -m ff.run_weekly" in s or "run: python -m model.serve" in s
                 or "name: Commit results" in s]
        names = [self.steps[i] for i in order]
        self.assertIn("ff.run_weekly", names[0])
        self.assertIn("model.serve", names[1])
        self.assertIn("Commit results", names[2])

    def test_python_312_pip_cache_keyed_on_the_model_requirements_and_failure_isolated(self):
        setup = self.step("cache-dependency-path: model/requirements.txt")
        self.assertIn('python-version: "3.12"', setup)
        self.assertIn("cache: pip", setup)
        run = self.step("run: python -m model.serve")
        self.assertIn("run: python -m model.serve && python -m model.scoreboard", run)
        self.assertIn("continue-on-error: true", run)
        self.assertIn("timeout-minutes", run)
        self.assertIn("continue-on-error: true", self.step("pip install -r model/requirements.txt"))

    def test_the_existing_pipeline_step_and_its_failure_gate_are_unchanged(self):
        run = self.step("name: Run pipeline")
        self.assertIn("run: python -m ff.run_weekly --strict", run)
        self.assertIn("continue-on-error: true", run)
        self.assertIn("steps.pipeline.outcome == 'failure'", self.step("Fail the job if integrity checks failed"))

    def test_the_model_cache_is_keyed_on_content_and_week_not_on_the_run(self):
        """A run-id key misses every run and uploads a fresh multi-hundred-MB entry, evicting the pip caches from the 10 GB quota."""
        cache = self.step("path: model/cache")
        key = cache[cache.index("key:"):cache.index("restore-keys")]
        self.assertNotIn("run_id", key)
        self.assertIn("hashFiles('model/requirements.txt'", key)
        self.assertIn("model/features.py", key)
        self.assertIn("steps.modelweek.outputs.week", key)
        self.assertIn("model-cache-\n", cache[cache.index("restore-keys"):] + "\n")          # the prefix fallback is kept
        self.assertIn("date -u +%G-W%V", self.step("id: modelweek"))
        self.assertIn("continue-on-error: true", self.step("id: modelweek"))

    def test_the_manual_fetch_workflow_cannot_lose_the_adp_csvs_to_a_later_step(self):
        text = (REPO / ".github" / "workflows" / "model_fetch.yml").read_text(encoding="utf-8")
        steps = text.split("\n      - ")
        for needle in ("model.fetch_adp", "model.fetch_cfbd", "name: Step summary"):
            hit = [s for s in steps if needle in s]
            self.assertEqual(len(hit), 1, needle)
            self.assertIn("continue-on-error: true", hit[0], needle)

    def test_the_commit_step_also_adds_the_scoreboard_report(self):
        self.assertIn("model/reports/live_scoreboard.md", self.step("name: Commit results"))


class Phase4(unittest.TestCase):
    """The Sunday pre-lock run and the props step: the only two additions to pipeline.yml, both failure-isolated."""

    @classmethod
    def setUpClass(cls):
        cls.text = (REPO / ".github" / "workflows" / "pipeline.yml").read_text(encoding="utf-8")
        cls.steps = cls.text.split("\n      - ")

    def step(self, needle: str) -> str:
        hits = [s for s in self.steps if needle in s]
        self.assertEqual(len(hits), 1, needle)
        return hits[0]

    def crons(self) -> list[str]:
        return [ln.strip() for ln in self.text.splitlines() if ln.strip().startswith("- cron:")]

    def test_two_sunday_pre_lock_entries_and_the_other_four_unchanged(self):
        """Scheduled runs arrive 1.4-5.75 h late: the 13:52 slot is the one that can still beat the 17:00 UTC lock."""
        crons = self.crons()
        self.assertEqual(len(crons), 6)
        self.assertEqual([c.split("#")[0].strip() for c in crons],
                         ['- cron: "37 18 * * 2"', '- cron: "41 19 * * 2"', '- cron: "7 14 * * 3"', '- cron: "11 22 * * 5"',
                          '- cron: "52 13 * * 0"', '- cron: "52 15 * * 0"'])
        for c, hour in ((crons[-2], "13"), (crons[-1], "15")):
            minute, h, _, _, dow = c.split('"')[1].split()
            self.assertEqual((h, minute, dow), (hour, "52", "0"))                 # Sunday, before 17:00 UTC, off :00 and :30
            self.assertNotIn(minute, ("0", "30"))
        self.assertIn("workflow_dispatch:", self.text)
        why = self.text[self.text.index('- cron: "11 22 * * 5"'):self.text.index('- cron: "52 13 * * 0"')]
        for needle in ("late", "17:00 UTC", "40 h"):
            self.assertIn(needle, why)                                              # the comment says why, and about the props window

    def test_both_workflows_pin_the_runner_with_a_reason(self):
        for name in ("pipeline.yml", "model_fetch.yml"):
            text = (REPO / ".github" / "workflows" / name).read_text(encoding="utf-8")
            self.assertEqual(text.count("runs-on: ubuntu-24.04"), 1, name)
            self.assertNotIn("runs-on: ubuntu-latest", text, name)
            self.assertIn("Oct 19", text, name)

    def test_the_props_step_is_isolated_capped_keyed_by_the_secret_and_sits_between_the_model_and_the_commit(self):
        props = self.step("run: python -m model.fetch_props")
        self.assertIn("continue-on-error: true", props)
        self.assertRegex(props, r"timeout-minutes: \d+")
        self.assertIn("ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}", props)
        self.assertEqual(self.text.count("secrets.ODDS_API_KEY"), 1)                # the secret reaches this one step, nowhere else
        order = [i for i, s in enumerate(self.steps) if "run: python -m model.serve" in s or "run: python -m model.fetch_props" in s
                 or "name: Commit results" in s]
        self.assertEqual([("serve" in self.steps[i], "fetch_props" in self.steps[i]) for i in order[:2]], [(True, False), (False, True)])
        self.assertIn("Commit results", self.steps[order[2]])
        self.assertNotIn("steps.model", props)                                      # does not wait on, or depend on, the model step

    def test_it_runs_whether_or_not_the_model_install_worked_and_never_decides_the_jobs_colour(self):
        gate = self.step("Fail the job if integrity checks failed")
        self.assertIn("steps.pipeline.outcome == 'failure'", gate)
        self.assertNotIn("props", gate)
        self.assertNotIn("if:", self.step("run: python -m model.fetch_props"))     # no condition that could skip it for the wrong reason

    def test_the_commit_step_already_adds_the_data_directory_so_props_csv_is_committed_with_no_change_there(self):
        commit = self.step("name: Commit results")
        self.assertIn("git add data leagues reports config.json", commit)
        self.assertNotIn("fetch_props", commit)

    def test_the_props_step_is_the_only_new_step_and_the_old_steps_are_as_they_were(self):
        self.assertEqual(len(self.steps) - 1, 14)                                    # the 13 steps Phase 3 left, plus the props step
        self.assertEqual(sum(1 for s in self.steps if "fetch_props" in s), 1)
        self.assertIn("run: python -m ff.run_weekly --strict", self.step("name: Run pipeline"))
        self.assertIn("run: python -m model.serve && python -m model.scoreboard", self.step("id: model\n"))


if __name__ == "__main__":
    unittest.main()
