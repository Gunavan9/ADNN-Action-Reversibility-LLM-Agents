"""
Unit tests for Sandbox Environment, Tools, and Reversibility Verification.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from src.sandbox.environment import SandboxEnvironment
from src.sandbox.tools import ToolDispatcher
from src.sandbox.reversibility_engine import ReversibilityEngine

class TestSandboxEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = SandboxEnvironment(recycle_bin_enabled=True, undo_window_seconds=60, escrow_enabled=True)
        self.dispatcher = ToolDispatcher(self.env)

    def test_save_and_restore_state(self):
        self.env.fs.write("test.txt", "initial content")
        state_0 = self.env.save_state()

        # Modify state
        self.env.fs.write("test.txt", "modified content")
        self.assertEqual(self.env.fs.read("test.txt"), "modified content")

        # Restore state
        self.env.restore_state(state_0)
        self.assertEqual(self.env.fs.read("test.txt"), "initial content")

    def test_reversibility_with_recycle_bin(self):
        self.env.fs.write("doc.txt", "critical data")
        # Recycle bin is enabled: delete should be reversible (0)
        label = ReversibilityEngine.get_ground_truth_label(self.env, "delete_file", {"path": "doc.txt"})
        self.assertEqual(label, 0)

        # Now disable recycle bin: delete should be irreversible (2)
        self.env.fs.recycle_bin_enabled = False
        label_no_trash = ReversibilityEngine.get_ground_truth_label(self.env, "delete_file", {"path": "doc.txt"})
        self.assertEqual(label_no_trash, 2)

    def test_email_undo_window(self):
        # Email with undo window should be reversible (0)
        label_undo = ReversibilityEngine.get_ground_truth_label(self.env, "send_email", {"recipient": "user@corp.com", "subject": "Test", "instant": False})
        self.assertEqual(label_undo, 0)

        # Email with instant=True should be irreversible (2)
        label_instant = ReversibilityEngine.get_ground_truth_label(self.env, "send_email", {"recipient": "user@corp.com", "subject": "Test", "instant": True})
        self.assertEqual(label_instant, 2)

    def test_escrow_reversibility(self):
        # Escrow active: reversible (0)
        label_escrow = ReversibilityEngine.get_ground_truth_label(self.env, "transfer_money", {"recipient": "vendor", "amount": 100.0, "use_escrow": True})
        self.assertEqual(label_escrow, 0)

        # Instant wire: irreversible (2)
        label_wire = ReversibilityEngine.get_ground_truth_label(self.env, "transfer_money", {"recipient": "vendor", "amount": 100.0, "use_escrow": False})
        self.assertEqual(label_wire, 2)

if __name__ == "__main__":
    unittest.main()
