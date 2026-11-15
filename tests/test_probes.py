"""
Unit tests for Linear Probe and Baselines.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import numpy as np
from src.probes.linear_probe import ReversibilityLinearProbe
from src.probes.baselines import MajorityClassBaseline, LexicalBagOfWordsBaseline, ShuffledLabelProbeControl

class TestProbesAndBaselines(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X_train = np.random.randn(100, 64)
        self.y_train = np.random.choice([0, 1, 2], size=100)
        self.X_test = np.random.randn(30, 64)
        self.y_test = np.random.choice([0, 1, 2], size=30)

    def test_linear_probe_fit_and_eval(self):
        probe = ReversibilityLinearProbe(random_state=42)
        probe.fit(self.X_train, self.y_train)
        metrics = probe.evaluate(self.X_test, self.y_test)
        self.assertIn("balanced_accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("auroc", metrics)

    def test_majority_baseline(self):
        maj = MajorityClassBaseline()
        maj.fit(self.y_train)
        res = maj.evaluate(self.y_test)
        self.assertGreaterEqual(res["balanced_accuracy"], 0.0)

    def test_shuffled_control_near_chance(self):
        shuf = ShuffledLabelProbeControl.run(self.X_train, self.y_train, self.X_test, self.y_test, seed=42)
        # 3-class random chance is ~0.33
        self.assertAlmostEqual(shuf["balanced_accuracy"], 0.33, delta=0.25)

if __name__ == "__main__":
    unittest.main()
