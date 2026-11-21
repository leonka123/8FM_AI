"""Tests for the resumable Kaggle training driver."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_DRIVER_PATH = Path(__file__).resolve().parents[1] / "kaggle" / "train_lu_sol.py"

_spec = importlib.util.spec_from_file_location("kaggle_train_lu_sol", _DRIVER_PATH)
driver = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(driver)


class DriverHelpersTests(unittest.TestCase):
    def test_finds_repository_root(self):
        root = driver._find_repo_root(Path(driver.__file__).resolve().parent)
        self.assertTrue((root / "eightfm" / "__init__.py").exists())

    def test_lerp_is_clamped(self):
        self.assertAlmostEqual(driver._lerp(0.0, 1.0, 0.25), 0.25)
        self.assertAlmostEqual(driver._lerp(0.0, 1.0, -3.0), 0.0)
        self.assertAlmostEqual(driver._lerp(0.0, 1.0, 3.0), 1.0)

    def test_state_roundtrip_and_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "driver_state.json"
            default = driver._load_state(path)
            self.assertEqual(default["global_step"], 0)
            default.update(global_step=42, best_score=0.5, chunks=3)
            driver._save_state(path, default)
            loaded = driver._load_state(path)
            self.assertEqual(loaded["global_step"], 42)
            self.assertEqual(loaded["chunks"], 3)
            self.assertAlmostEqual(loaded["best_score"], 0.5)

    def test_merge_metrics_appends_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            chunk = {
                "format": "eightfm-rl-training-metrics",
                "config": {"updates": 10},
                "updates": [{"update": 1}],
                "evaluations": [{"update": 1}],
                "best_score": 0.4,
            }
            driver._merge_metrics(path, chunk, 0)
            second = dict(chunk, updates=[{"update": 2}], evaluations=[], best_score=0.6)
            driver._merge_metrics(path, second, 10)
            merged = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(merged["chunks"]), 2)
            self.assertEqual(len(merged["updates"]), 2)
            self.assertAlmostEqual(merged["best_score"], 0.6)


if __name__ == "__main__":
    unittest.main()
