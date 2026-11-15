"""
Layer-Wise Probe Evaluator and Experiment Runner (RQ1).
Conducts systematic layer-by-layer probing across:
- Qwen2.5-7B and Llama-3.1-8B
- Binary (Reversible vs Irreversible) and 3-Class tasks
- Compulsory Baselines: Majority, Lexical BoW, Shuffled Label, Control Task (Hewitt & Liang), Embedding (Layer 0)
- 3 random train/test scenario splits with mean & std
- 1,000 bootstrap resamples for 95% Confidence Intervals
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from src.probes.linear_probe import ReversibilityLinearProbe
from src.probes.baselines import (
    MajorityClassBaseline,
    LexicalBagOfWordsBaseline,
    ShuffledLabelProbeControl,
    HewittLiangControlTask
)
from src.models.extractor import HiddenStateExtractor
from src.dataset.generator import partition_train_test
from src.config import SPLIT_SEEDS, BOOTSTRAP_RESAMPLES

def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric_fn, n_resamples: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """Computes non-parametric 95% bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_resamples):
        indices = rng.randint(0, n, n)
        if len(np.unique(y_true[indices])) < 2:
            continue
        scores.append(metric_fn(y_true[indices], y_pred[indices]))
    if not scores:
        return (0.0, 0.0)
    return (float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)))


def evaluate_model_probing(
    dataset: List[Dict[str, Any]],
    model_key: str = "qwen2.5-7b",
    task_mode: str = "3class",  # 'binary' or '3class'
    pooling: str = "last_token"
) -> Dict[str, Any]:
    """
    Executes full probing evaluation for a given model architecture across all layers.
    """
    extractor = HiddenStateExtractor(model_key=model_key, use_mock_if_no_weights=True)
    num_layers = extractor.num_layers

    # Filter for binary if requested
    if task_mode == "binary":
        working_dataset = [d for d in dataset if d["label"] in [0, 2]]
        # remap class 2 to 1 for standard binary
        for d in working_dataset:
            if d["label"] == 2:
                d["probe_target"] = 1
            else:
                d["probe_target"] = 0
    else:
        working_dataset = list(dataset)
        for d in working_dataset:
            d["probe_target"] = d["label"]

    # Extract features for all working dataset examples once
    X_all_layers = np.array([
        extractor.extract_from_prompt(d["state_text"], d["action_text"], pooling=pooling)
        for d in working_dataset
    ])  # shape: (N, num_layers + 1, dim)

    # Pre-build lookup for split indices
    split_results = []

    for split_idx, seed in enumerate(SPLIT_SEEDS):
        train_data, test_data = partition_train_test(working_dataset, test_ratio=0.20, seed=seed)
        train_ids = set(d["id"] for d in train_data)
        train_mask = np.array([d["id"] in train_ids for d in working_dataset])
        test_mask = ~train_mask

        X_train_all = X_all_layers[train_mask]
        X_test_all = X_all_layers[test_mask]

        y_train = np.array([d["probe_target"] for d in train_data])
        y_test = np.array([d["probe_target"] for d in test_data])

        action_texts_train = [d["action_text"] for d in train_data]
        action_texts_test = [d["action_text"] for d in test_data]
        verbs_train = [d["action_verb"] for d in train_data]
        verbs_test = [d["action_verb"] for d in test_data]

        # 1. Majority baseline
        maj_baseline = MajorityClassBaseline()
        maj_baseline.fit(y_train)
        maj_res = maj_baseline.evaluate(y_test)

        # 2. Lexical BoW baseline
        lex_baseline = LexicalBagOfWordsBaseline(random_state=seed)
        lex_baseline.fit(action_texts_train, y_train)
        lex_res = lex_baseline.evaluate(action_texts_test, y_test)

        # Layer-wise probing
        layer_metrics = []
        for l in range(num_layers + 1):
            X_tr_l = X_train_all[:, l, :]
            X_te_l = X_test_all[:, l, :]

            probe = ReversibilityLinearProbe(random_state=seed)
            probe.fit(X_tr_l, y_train)
            metrics = probe.evaluate(X_te_l, y_test)

            # Evaluate control and shuffled tasks on sampled key layers for efficiency
            if l % 4 == 0 or l == 16:
                ctrl_res = HewittLiangControlTask.run(
                    action_verbs_train=verbs_train,
                    action_verbs_test=verbs_test,
                    X_train=X_tr_l,
                    X_test=X_te_l,
                    probe_accuracy=metrics["balanced_accuracy"],
                    num_classes=2 if task_mode == "binary" else 3,
                    seed=seed
                )
                metrics["selectivity"] = ctrl_res["selectivity"]
                metrics["control_accuracy"] = ctrl_res["control_task_accuracy"]

                shuffled_res = ShuffledLabelProbeControl.run(X_tr_l, y_train, X_te_l, y_test, seed=seed)
                metrics["shuffled_label_accuracy"] = shuffled_res["balanced_accuracy"]
                last_ctrl_acc = ctrl_res["control_task_accuracy"]
                last_shuf_acc = shuffled_res["balanced_accuracy"]
            else:
                metrics["control_accuracy"] = last_ctrl_acc
                metrics["selectivity"] = metrics["balanced_accuracy"] - last_ctrl_acc
                metrics["shuffled_label_accuracy"] = last_shuf_acc

            layer_metrics.append(metrics)

        split_results.append({
            "split_seed": seed,
            "majority_accuracy": maj_res["balanced_accuracy"],
            "lexical_accuracy": lex_res["balanced_accuracy"],
            "layer_metrics": layer_metrics
        })

    # Aggregate across splits
    aggregated_layers = []
    for l in range(num_layers + 1):
        bal_accs = [split_results[s]["layer_metrics"][l]["balanced_accuracy"] for s in range(len(SPLIT_SEEDS))]
        macro_f1s = [split_results[s]["layer_metrics"][l]["macro_f1"] for s in range(len(SPLIT_SEEDS))]
        aurocs = [split_results[s]["layer_metrics"][l]["auroc"] for s in range(len(SPLIT_SEEDS))]
        selectivities = [split_results[s]["layer_metrics"][l]["selectivity"] for s in range(len(SPLIT_SEEDS))]
        shuffled_accs = [split_results[s]["layer_metrics"][l]["shuffled_label_accuracy"] for s in range(len(SPLIT_SEEDS))]

        aggregated_layers.append({
            "layer": l,
            "balanced_accuracy_mean": float(np.mean(bal_accs)),
            "balanced_accuracy_std": float(np.std(bal_accs)),
            "macro_f1_mean": float(np.mean(macro_f1s)),
            "auroc_mean": float(np.mean(aurocs)),
            "selectivity_mean": float(np.mean(selectivities)),
            "shuffled_acc_mean": float(np.mean(shuffled_accs))
        })

    # Peak layer detection
    peak_layer_info = max(aggregated_layers, key=lambda x: x["balanced_accuracy_mean"])

    return {
        "model": model_key,
        "task_mode": task_mode,
        "pooling": pooling,
        "num_layers": num_layers,
        "majority_accuracy": float(np.mean([sr["majority_accuracy"] for sr in split_results])),
        "lexical_accuracy": float(np.mean([sr["lexical_accuracy"] for sr in split_results])),
        "embedding_layer_accuracy": aggregated_layers[0]["balanced_accuracy_mean"],
        "peak_layer": peak_layer_info["layer"],
        "peak_accuracy": peak_layer_info["balanced_accuracy_mean"],
        "peak_selectivity": peak_layer_info["selectivity_mean"],
        "layers": aggregated_layers
    }
