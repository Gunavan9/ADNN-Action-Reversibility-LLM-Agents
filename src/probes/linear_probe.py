"""
Linear Probe Implementation for Reversibility Decoding.
Uses L2-regularized Logistic Regression to guarantee that decoding capacity
reflects linear separability in the model's representation geometry rather
than expressive power of a complex probe.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from typing import Dict, Any, Tuple, Optional

class ReversibilityLinearProbe:
    def __init__(self, C: float = 1.0, max_iter: int = 100, random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            tol=1e-3,
            solver="lbfgs",
            random_state=self.random_state,
            class_weight="balanced"
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits linear probe on representation matrix X and labels y."""
        self.clf.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculates Balanced Accuracy, Macro-F1, and AUROC."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        bal_acc = balanced_accuracy_score(y, y_pred)
        macro_f1 = f1_score(y, y_pred, average="macro")
        
        # Calculate AUROC
        try:
            if len(np.unique(y)) == 2:
                auroc = roc_auc_score(y, y_proba[:, 1])
            else:
                auroc = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")
        except Exception:
            auroc = 0.5

        return {
            "balanced_accuracy": float(bal_acc),
            "macro_f1": float(macro_f1),
            "auroc": float(auroc)
        }
