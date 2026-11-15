"""
Compulsory Baselines and Controls for Probing.
Implements:
1. Majority-class baseline.
2. Lexical baseline (Bag-of-Words on action text only).
3. Shuffled-label probe (probe trained on randomly permuted labels).
4. Hewitt and Liang (2019) Control Task (Selectivity calculation).
5. Embedding-layer probe (Layer 0).
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from src.probes.linear_probe import ReversibilityLinearProbe

class MajorityClassBaseline:
    def __init__(self):
        self.majority_label = 0

    def fit(self, y_train: np.ndarray):
        values, counts = np.unique(y_train, return_counts=True)
        self.majority_label = values[np.argmax(counts)]

    def evaluate(self, y_test: np.ndarray) -> Dict[str, float]:
        preds = np.full_like(y_test, self.majority_label)
        return {
            "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
            "macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0))
        }


class LexicalBagOfWordsBaseline:
    """
    Logistic Regression on bag-of-words of action text only.
    Demonstrates whether probe learns pure lexical verb patterns (e.g. 'delete')
    or genuine state-dependent reversibility.
    """
    def __init__(self, random_state: int = 42):
        self.vectorizer = CountVectorizer()
        self.clf = LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=500)

    def fit(self, action_texts_train: List[str], y_train: np.ndarray):
        X_bow = self.vectorizer.fit_transform(action_texts_train)
        self.clf.fit(X_bow, y_train)

    def evaluate(self, action_texts_test: List[str], y_test: np.ndarray) -> Dict[str, float]:
        X_bow_test = self.vectorizer.transform(action_texts_test)
        preds = self.clf.predict(X_bow_test)
        return {
            "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
            "macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0))
        }


class ShuffledLabelProbeControl:
    """
    Trains the exact linear probe architecture on randomly permuted labels.
    Expected performance should be at chance level.
    """
    @staticmethod
    def run(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, seed: int = 42) -> Dict[str, float]:
        rng = np.random.RandomState(seed)
        y_train_shuffled = rng.permutation(y_train)
        probe = ReversibilityLinearProbe(random_state=seed)
        probe.fit(X_train, y_train_shuffled)
        return probe.evaluate(X_test, y_test)


class HewittLiangControlTask:
    """
    Control task in the style of Hewitt & Liang (EMNLP 2019).
    Assigns each action verb a random fixed label, trains probe on the control task,
    and computes: Selectivity = Probe Accuracy - Control Task Accuracy.
    High selectivity indicates that the probe reflects genuine model encoding
    rather than memorization capacity.
    """
    @staticmethod
    def run(
        action_verbs_train: List[str],
        action_verbs_test: List[str],
        X_train: np.ndarray,
        X_test: np.ndarray,
        probe_accuracy: float,
        num_classes: int = 3,
        seed: int = 42
    ) -> Dict[str, float]:
        rng = np.random.RandomState(seed)
        unique_verbs = list(set(action_verbs_train + action_verbs_test))
        # Assign fixed random pseudo-label to each verb
        verb_to_control_label = {v: rng.randint(0, num_classes) for v in unique_verbs}

        y_train_control = np.array([verb_to_control_label[v] for v in action_verbs_train])
        y_test_control = np.array([verb_to_control_label[v] for v in action_verbs_test])

        probe = ReversibilityLinearProbe(random_state=seed)
        probe.fit(X_train, y_train_control)
        control_metrics = probe.evaluate(X_test, y_test_control)

        control_acc = control_metrics["balanced_accuracy"]
        selectivity = probe_accuracy - control_acc

        return {
            "control_task_accuracy": float(control_acc),
            "selectivity": float(selectivity)
        }
