"""
tests/test_paths.py
-------------------
Focused regression tests for config.paths.

These tests verify the canonical directory layout and the per-paper
artifact names used by the pipeline stages.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import get_paper_paths, get_project_paths


class TestProjectPaths(unittest.TestCase):

    def test_evaluation_lives_under_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch.dict("os.environ", {"PROJECT_ROOT": str(root)}, clear=False):
                paths = get_project_paths(create_dirs=False)

            self.assertEqual(paths["evaluation"], root / "data" / "evaluation")
            self.assertEqual(paths["results"], root / "data" / "evaluation" / "results")
            self.assertEqual(paths["reports"], root / "data" / "evaluation" / "reports")

    def test_project_dirs_are_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch.dict("os.environ", {"PROJECT_ROOT": str(root)}, clear=False):
                paths = get_project_paths(create_dirs=True)

            self.assertTrue(paths["evaluation"].exists())
            self.assertTrue(paths["xai_outputs"].exists())


class TestPaperPaths(unittest.TestCase):

    def test_canonical_artifact_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch.dict("os.environ", {"PROJECT_ROOT": str(root)}, clear=False):
                base_paths = get_project_paths(create_dirs=False)
                paper_paths = get_paper_paths("2403.02901", base_paths)

        self.assertEqual(paper_paths["metadata"], root / "data" / "extracted" / "2403.02901_metadata.json")
        self.assertEqual(paper_paths["layout_json"], root / "data" / "extracted" / "layout_2403.02901.json")
        self.assertEqual(paper_paths["text_json"], root / "data" / "extracted" / "text_2403.02901.json")
        self.assertEqual(paper_paths["tables_json"], root / "data" / "extracted" / "tables_2403.02901.json")
        self.assertEqual(paper_paths["layout"], paper_paths["layout_json"])
        self.assertEqual(paper_paths["text_elements"], paper_paths["text_json"])


if __name__ == "__main__":
    unittest.main(verbosity=2)