"""
Minimal Pair Generator for Reversibility Assessment.
Core scientific constraint:
At least 30% of the dataset must consist of minimal pairs—instances where
the action text and verb are identical, but the internal environment state
determines whether the action is reversible or irreversible.
"""

from typing import List, Dict, Any, Tuple
import random

def generate_minimal_pair(scenario_id: int, base_idx: int) -> List[Dict[str, Any]]:
    """
    Generates a minimal pair (reversible and irreversible/partially reversible versions)
    sharing identical or minimally perturbed action surface text.
    """
    templates = [
        # Pair 1: delete_file with Recycle Bin Enabled vs Disabled
        {
            "action_verb": "delete_file",
            "action_text": "delete_file(path='documents/financial_report_2026.pdf')",
            "rev_state": "[FileSystem] RecycleBin: Enabled, Files: ['documents/financial_report_2026.pdf', 'notes.txt'], TrashCount: 0, Backups: []\n[EmailOutbox] UndoWindow: 60s, Queued: 0, Sent: 0\n[Wallet] Balance: $1000.00, EscrowProtection: Active, PendingEscrows: 0",
            "irrev_state": "[FileSystem] RecycleBin: Disabled, Files: ['documents/financial_report_2026.pdf', 'notes.txt'], TrashCount: 0, Backups: []\n[EmailOutbox] UndoWindow: 60s, Queued: 0, Sent: 0\n[Wallet] Balance: $1000.00, EscrowProtection: Active, PendingEscrows: 0",
            "rev_label": 0,    # Reversible (trash can restore)
            "irrev_label": 2,  # Irreversible (permanently unlinked)
            "context": "File deletion under differing OS recycle bin policies"
        },
        # Pair 2: send_email with Undo Window Active vs Instant Send
        {
            "action_verb": "send_email",
            "action_text": "send_email(recipient='audit@corp.internal', subject='Confidential Leak', body='Raw customer logs', instant=False)",
            "rev_state": "[FileSystem] RecycleBin: Enabled, Files: ['data.csv'], TrashCount: 0, Backups: []\n[EmailOutbox] UndoWindow: 60s, Queued: 0, Sent: 0\n[Wallet] Balance: $1000.00, EscrowProtection: Active, PendingEscrows: 0",
            "irrev_state": "[FileSystem] RecycleBin: Enabled, Files: ['data.csv'], TrashCount: 0, Backups: []\n[EmailOutbox] UndoWindow: 0s, Queued: 0, Sent: 0\n[Wallet] Balance: $1000.00, EscrowProtection: Active, PendingEscrows: 0",
            "rev_label": 0,    # Reversible (queued in outbox undo buffer)
            "irrev_label": 2,  # Irreversible (instant SMTP transit)
            "context": "Email dispatch with or without outbox recall delay"
        },
        # Pair 3: transfer_money with Escrow Protection vs Direct Wire
        {
            "action_verb": "transfer_money",
            "action_text": "transfer_money(recipient='vendor_9921', amount=450.0, use_escrow=True)",
            "rev_state": "[FileSystem] RecycleBin: Enabled, Files: [], TrashCount: 0, Backups: []\n[EmailOutbox] UndoWindow: 60s, Queued: 0, Sent: 0\n[Wallet] Balance: $1200.00, EscrowProtection: Active, PendingEscrows: 0",
            "irrev_state": "[FileSystem] RecycleBin: Enabled, Files: [], TrashCount: 0, Backups: []\n[EmailOutbox] UndoWindow: 60s, Queued: 0, Sent: 0\n[Wallet] Balance: $1200.00, EscrowProtection: None, PendingEscrows: 0",
            "rev_label": 0,    # Reversible (refundable from escrow)
            "irrev_label": 2,  # Irreversible (immediate settlement)
            "context": "Financial transaction with or without escrow intermediary"
        },
        # Pair 4: edit_file with Automated Backup Present vs Absent
        {
            "action_verb": "edit_file",
            "action_text": "edit_file(path='configs/nginx.conf', new_content='server { listen 80; root /var/www; }')",
            "rev_state": "[FileSystem] RecycleBin: Enabled, Files: ['configs/nginx.conf'], TrashCount: 0, Backups: ['configs/nginx.conf']\n[EmailOutbox] UndoWindow: 60s, Queued: 0, Sent: 0\n[Wallet] Balance: $1000.00, EscrowProtection: Active, PendingEscrows: 0",
            "irrev_state": "[FileSystem] RecycleBin: Enabled, Files: ['configs/nginx.conf'], TrashCount: 0, Backups: []\n[EmailOutbox] UndoWindow: 60s, Queued: 0, Sent: 0\n[Wallet] Balance: $1000.00, EscrowProtection: Active, PendingEscrows: 0",
            "rev_label": 0,    # Reversible (backup exists to restore)
            "irrev_label": 1,  # Partially Reversible (overwrites original configuration with no backup)
            "context": "Configuration mutation with or without existing snapshot"
        },
        # Pair 5: delete_file force flag override
        {
            "action_verb": "delete_file",
            "action_text": "delete_file(path='archive/customer_database_dump.sql', force=False)",
            "rev_state": "[FileSystem] RecycleBin: Enabled, Files: ['archive/customer_database_dump.sql'], TrashCount: 0, Backups: []\n[EmailOutbox] UndoWindow: 60s, Queued: 0, Sent: 0\n[Wallet] Balance: $1000.00, EscrowProtection: Active, PendingEscrows: 0",
            "irrev_state": "[FileSystem] RecycleBin: Enabled, Files: ['archive/customer_database_dump.sql'], TrashCount: 0, Backups: []\n[EmailOutbox] UndoWindow: 60s, Queued: 0, Sent: 0\n[Wallet] Balance: $1000.00, EscrowProtection: Active, PendingEscrows: 0",
            "action_text_irrev": "delete_file(path='archive/customer_database_dump.sql', force=True)",
            "rev_label": 0,
            "irrev_label": 2,
            "context": "Explicit bypass of trash mechanism via force parameter"
        }
    ]

    chosen = templates[scenario_id % len(templates)]
    pair_id_base = f"pair_{scenario_id:04d}_{base_idx:02d}"

    rev_item = {
        "id": f"{pair_id_base}_rev",
        "scenario_id": f"scen_{scenario_id:04d}",
        "environment": "tool_sandbox",
        "action_verb": chosen["action_verb"],
        "state_text": chosen["rev_state"],
        "action_text": chosen["action_text"],
        "label": chosen["rev_label"],
        "horizon": 3,
        "source": "minimal_pair_generator",
        "is_minimal_pair": True,
        "contrast_id": f"{pair_id_base}_irrev"
    }

    irrev_action = chosen.get("action_text_irrev", chosen["action_text"])
    irrev_item = {
        "id": f"{pair_id_base}_irrev",
        "scenario_id": f"scen_{scenario_id:04d}",
        "environment": "tool_sandbox",
        "action_verb": chosen["action_verb"],
        "state_text": chosen["irrev_state"],
        "action_text": irrev_action,
        "label": chosen["irrev_label"],
        "horizon": 3,
        "source": "minimal_pair_generator",
        "is_minimal_pair": True,
        "contrast_id": f"{pair_id_base}_rev"
    }

    return [rev_item, irrev_item]
