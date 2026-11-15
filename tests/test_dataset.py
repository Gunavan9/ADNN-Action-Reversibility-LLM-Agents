"""
Unit tests for Dataset Integrity and Minimal Pair Constraints.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from src.dataset.generator import generate_full_dataset, partition_train_test

class TestDatasetIntegrity(unittest.TestCase):
    def setUp(self):
        self.dataset = generate_full_dataset()

    def test_minimum_sample_size(self):
        # Target >= 2,000 pairs
        self.assertGreaterEqual(len(self.dataset), 2000)

    def test_minimal_pair_ratio(self):
        # At least 30% of data must be minimal pairs
        minimal_count = sum(1 for d in self.dataset if d.get("is_minimal_pair", False))
        ratio = minimal_count / len(self.dataset)
        self.assertGreaterEqual(ratio, 0.30)

    def test_class_balance(self):
        # Balanced across 3 classes (0: rev, 1: part, 2: irrev)
        c0 = sum(1 for d in self.dataset if d["label"] == 0)
        c1 = sum(1 for d in self.dataset if d["label"] == 1)
        c2 = sum(1 for d in self.dataset if d["label"] == 2)
        total = len(self.dataset)
        self.assertGreater(c0 / total, 0.20)
        self.assertGreater(c1 / total, 0.20)
        self.assertGreater(c2 / total, 0.20)

    def test_scenario_disjoint_split(self):
        # Train and test splits must not share any scenario_id
        train, test = partition_train_test(self.dataset, test_ratio=0.20)
        train_scens = set(d["scenario_id"] for d in train)
        test_scens = set(d["scenario_id"] for d in test)
        self.assertEqual(len(train_scens.intersection(test_scens)), 0)

if __name__ == "__main__":
    unittest.main()
