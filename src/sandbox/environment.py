"""
Simulated OS and Agent Sandbox Environment.
Provides a strictly controlled, reproducible execution environment with:
- Simulated Virtual FileSystem (with optional Recycle Bin and auto-backups)
- Email Outbox (with configurable Undo Window)
- Wallet System (with optional Escrow / reversible pending transactions)
- Exact state save, restore, and deep comparison
"""

import copy
import json
import time
from typing import Dict, Any, List, Optional, Tuple

class FileSystem:
    def __init__(self, recycle_bin_enabled: bool = True):
        self.recycle_bin_enabled = recycle_bin_enabled
        self.files: Dict[str, str] = {}
        self.trash: Dict[str, Tuple[str, float]] = {}  # path -> (content, timestamp)
        self.backups: Dict[str, str] = {}  # path -> backup_content

    def read(self, path: str) -> Optional[str]:
        return self.files.get(path, None)

    def write(self, path: str, content: str, create_backup: bool = False) -> Dict[str, Any]:
        if path in self.files and create_backup:
            self.backups[path] = self.files[path]
        had_previous = path in self.files
        old_content = self.files.get(path, None)
        self.files[path] = content
        return {"status": "success", "overwritten": had_previous, "path": path}

    def delete(self, path: str, force_permanent: bool = False) -> Dict[str, Any]:
        if path not in self.files:
            return {"status": "error", "message": f"File {path} not found"}
        
        content = self.files.pop(path)
        if self.recycle_bin_enabled and not force_permanent:
            self.trash[path] = (content, time.time())
            return {"status": "success", "destination": "recycle_bin", "path": path}
        else:
            return {"status": "success", "destination": "permanently_deleted", "path": path}

    def restore_from_trash(self, path: str) -> Dict[str, Any]:
        if path not in self.trash:
            return {"status": "error", "message": f"File {path} not found in trash"}
        content, _ = self.trash.pop(path)
        self.files[path] = content
        return {"status": "success", "restored_to": path}

    def restore_backup(self, path: str) -> Dict[str, Any]:
        if path not in self.backups:
            return {"status": "error", "message": f"No backup found for {path}"}
        self.files[path] = self.backups[path]
        return {"status": "success", "restored_backup": path}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "recycle_bin_enabled": self.recycle_bin_enabled,
            "files": dict(self.files),
            "trash": {k: v[0] for k, v in self.trash.items()},
            "backups": dict(self.backups)
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.recycle_bin_enabled = state.get("recycle_bin_enabled", True)
        self.files = dict(state.get("files", {}))
        self.trash = {k: (v, time.time()) for k, v in state.get("trash", {}).items()}
        self.backups = dict(state.get("backups", {}))


class EmailOutbox:
    def __init__(self, undo_window_seconds: int = 60):
        self.undo_window_seconds = undo_window_seconds
        self.queue: Dict[str, Dict[str, Any]] = {}
        self.sent: List[Dict[str, Any]] = []
        self._next_id = 1

    def send(self, recipient: str, subject: str, body: str, instant: bool = False) -> Dict[str, Any]:
        mail_id = f"mail_{self._next_id}"
        self._next_id += 1
        msg = {
            "id": mail_id,
            "recipient": recipient,
            "subject": subject,
            "body": body,
            "timestamp": time.time(),
            "status": "queued" if (self.undo_window_seconds > 0 and not instant) else "sent"
        }
        if msg["status"] == "queued":
            self.queue[mail_id] = msg
            return {"status": "success", "mail_id": mail_id, "mode": "queued_with_undo", "undo_window": self.undo_window_seconds}
        else:
            self.sent.append(msg)
            return {"status": "success", "mail_id": mail_id, "mode": "instant_sent"}

    def undo_send(self, mail_id: str) -> Dict[str, Any]:
        if mail_id in self.queue:
            recalled = self.queue.pop(mail_id)
            return {"status": "success", "recalled_id": mail_id, "subject": recalled["subject"]}
        return {"status": "error", "message": f"Message {mail_id} is not in undoable queue"}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "undo_window_seconds": self.undo_window_seconds,
            "queue": copy.deepcopy(self.queue),
            "sent": copy.deepcopy(self.sent),
            "next_id": self._next_id
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.undo_window_seconds = state.get("undo_window_seconds", 60)
        self.queue = copy.deepcopy(state.get("queue", {}))
        self.sent = copy.deepcopy(state.get("sent", []))
        self._next_id = state.get("next_id", 1)


class Wallet:
    def __init__(self, initial_balance: float = 1000.0, escrow_enabled: bool = True):
        self.balance = initial_balance
        self.escrow_enabled = escrow_enabled
        self.pending_escrows: Dict[str, Dict[str, Any]] = {}
        self.completed_transfers: List[Dict[str, Any]] = []
        self._tx_id = 1

    def transfer(self, recipient: str, amount: float, use_escrow: bool = True) -> Dict[str, Any]:
        if amount > self.balance:
            return {"status": "error", "message": "Insufficient funds"}
        
        tx_id = f"tx_{self._tx_id}"
        self._tx_id += 1
        self.balance -= amount

        if self.escrow_enabled and use_escrow:
            tx = {
                "id": tx_id,
                "recipient": recipient,
                "amount": amount,
                "status": "in_escrow"
            }
            self.pending_escrows[tx_id] = tx
            return {"status": "success", "tx_id": tx_id, "mode": "escrow_protected"}
        else:
            tx = {
                "id": tx_id,
                "recipient": recipient,
                "amount": amount,
                "status": "completed_wire"
            }
            self.completed_transfers.append(tx)
            return {"status": "success", "tx_id": tx_id, "mode": "instant_irreversible_wire"}

    def cancel_escrow(self, tx_id: str) -> Dict[str, Any]:
        if tx_id in self.pending_escrows:
            tx = self.pending_escrows.pop(tx_id)
            self.balance += tx["amount"]
            return {"status": "success", "refunded": tx["amount"], "tx_id": tx_id}
        return {"status": "error", "message": f"Transaction {tx_id} cannot be cancelled"}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "balance": self.balance,
            "escrow_enabled": self.escrow_enabled,
            "pending_escrows": copy.deepcopy(self.pending_escrows),
            "completed_transfers": copy.deepcopy(self.completed_transfers),
            "tx_id": self._tx_id
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.balance = state.get("balance", 1000.0)
        self.escrow_enabled = state.get("escrow_enabled", True)
        self.pending_escrows = copy.deepcopy(state.get("pending_escrows", {}))
        self.completed_transfers = copy.deepcopy(state.get("completed_transfers", []))
        self._tx_id = state.get("tx_id", 1)


class SandboxEnvironment:
    """
    Unified Sandbox Environment that coordinates FileSystem, EmailOutbox, and Wallet.
    Maintains full action history and supports exact save/restore points.
    """
    def __init__(self, recycle_bin_enabled: bool = True, undo_window_seconds: int = 60, escrow_enabled: bool = True):
        self.fs = FileSystem(recycle_bin_enabled=recycle_bin_enabled)
        self.email = EmailOutbox(undo_window_seconds=undo_window_seconds)
        self.wallet = Wallet(escrow_enabled=escrow_enabled)
        self.history: List[Dict[str, Any]] = []
        self.caution_events: List[str] = []

    def save_state(self) -> Dict[str, Any]:
        return {
            "fs": self.fs.state_dict(),
            "email": self.email.state_dict(),
            "wallet": self.wallet.state_dict(),
            "caution_events": list(self.caution_events)
        }

    def restore_state(self, state: Dict[str, Any]):
        self.fs.load_state_dict(state.get("fs", {}))
        self.email.load_state_dict(state.get("email", {}))
        self.wallet.load_state_dict(state.get("wallet", {}))
        self.caution_events = list(state.get("caution_events", []))

    def get_state_repr(self) -> str:
        """Compact textual summary of current state for prompt building."""
        lines = [
            f"[FileSystem] RecycleBin: {'Enabled' if self.fs.recycle_bin_enabled else 'Disabled'}, Files: {list(self.fs.files.keys())}, TrashCount: {len(self.fs.trash)}, Backups: {list(self.fs.backups.keys())}",
            f"[EmailOutbox] UndoWindow: {self.email.undo_window_seconds}s, Queued: {len(self.email.queue)}, Sent: {len(self.email.sent)}",
            f"[Wallet] Balance: ${self.wallet.balance:.2f}, EscrowProtection: {'Active' if self.wallet.escrow_enabled else 'None'}, PendingEscrows: {len(self.wallet.pending_escrows)}"
        ]
        return "\n".join(lines)

    def log_caution(self, caution_type: str):
        self.caution_events.append(caution_type)
