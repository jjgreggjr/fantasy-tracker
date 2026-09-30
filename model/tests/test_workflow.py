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


if __name__ == "__main__":
    unittest.main()
