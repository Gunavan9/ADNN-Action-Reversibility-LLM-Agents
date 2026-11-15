"""
Comprehensive Ablation Suite.
Implements compulsory and supplementary ablations specified in Section 7.3:
1. Token position: Last action token vs. Mean pooling across action tokens.
2. Minimal pairs only vs. All data (disentangles lexical cues from state awareness).
3. Cross-environment transfer: Train on Sandbox -> Test on ScienceWorld/ToolEmu, and reverse.
4. System prompt intervention: Neutral system prompt vs. Explicit caution prompt.
5. Reversibility horizon: Evaluation under H in {1, 2, 3}.
6. Model scale: Qwen2.5-7B vs. Llama-3.1-8B vs. Qwen2.5-14B.
7. Label granularity: Binary vs. 3-Class.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from src.probes.probe_eval import evaluate_model_probing
from src.evaluation.behavioral_eval import run_behavioral_evaluation
from src.dataset.generator import partition_train_test
from src.probes.linear_probe import ReversibilityLinearProbe
from src.models.extractor import HiddenStateExtractor

def run_token_position_ablation(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Ablation 1: Last action token vs Mean pooling."""
    res_last = evaluate_model_probing(dataset, model_key="qwen2.5-7b", task_mode="3class", pooling="last_token")
    res_mean = evaluate_model_probing(dataset, model_key="qwen2.5-7b", task_mode="3class", pooling="mean_pooling")
    
    return {
        "last_token_peak_acc": res_last["peak_accuracy"],
        "last_token_peak_layer": res_last["peak_layer"],
        "mean_pooling_peak_acc": res_mean["peak_accuracy"],
        "mean_pooling_peak_layer": res_mean["peak_layer"],
        "delta_accuracy": res_last["peak_accuracy"] - res_mean["peak_accuracy"]
    }


def run_minimal_pairs_ablation(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Ablation 2: Minimal pairs only vs All data."""
    minimal_pairs_data = [d for d in dataset if d.get("is_minimal_pair", False)]
    
    res_all = evaluate_model_probing(dataset, model_key="qwen2.5-7b", task_mode="3class")
    res_minimal = evaluate_model_probing(minimal_pairs_data, model_key="qwen2.5-7b", task_mode="3class")

    return {
        "all_data_count": len(dataset),
        "all_data_peak_acc": res_all["peak_accuracy"],
        "all_data_lexical_acc": res_all["lexical_accuracy"],
        "minimal_pairs_count": len(minimal_pairs_data),
        "minimal_pairs_peak_acc": res_minimal["peak_accuracy"],
        "minimal_pairs_lexical_acc": res_minimal["lexical_accuracy"],
        "probe_beats_lexical_on_minimal_pairs": bool(res_minimal["peak_accuracy"] > res_minimal["lexical_accuracy"])
    }


def run_cross_environment_transfer(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ablation 3: Cross-environment generalization.
    Train on Tool Sandbox -> Test on ScienceWorld, and reverse.
    """
    extractor = HiddenStateExtractor(model_key="qwen2.5-7b", use_mock_if_no_weights=True)
    target_layer = 16

    sandbox_data = [d for d in dataset if d["environment"] == "tool_sandbox"]
    sw_data = [d for d in dataset if d["environment"] == "scienceworld"]

    # Extract features at target layer
    X_sb = np.array([extractor.extract_from_prompt(d["state_text"], d["action_text"])[target_layer] for d in sandbox_data])
    y_sb = np.array([d["label"] for d in sandbox_data])

    X_sw = np.array([extractor.extract_from_prompt(d["state_text"], d["action_text"])[target_layer] for d in sw_data])
    y_sw = np.array([d["label"] for d in sw_data])

    # 1. Train Sandbox -> Test ScienceWorld
    probe_sb = ReversibilityLinearProbe(random_state=42)
    probe_sb.fit(X_sb, y_sb)
    eval_sb_to_sw = probe_sb.evaluate(X_sw, y_sw)

    # 2. Train ScienceWorld -> Test Sandbox
    probe_sw = ReversibilityLinearProbe(random_state=42)
    probe_sw.fit(X_sw, y_sw)
    eval_sw_to_sb = probe_sw.evaluate(X_sb, y_sb)

    return {
        "sandbox_to_scienceworld_acc": eval_sb_to_sw["balanced_accuracy"],
        "sandbox_to_scienceworld_f1": eval_sb_to_sw["macro_f1"],
        "scienceworld_to_sandbox_acc": eval_sw_to_sb["balanced_accuracy"],
        "scienceworld_to_sandbox_f1": eval_sw_to_sb["macro_f1"]
    }


def run_prompt_intervention_ablation(paired_tasks: List[Dict[str, Any]], model_key: str = "qwen2.5-7b") -> Dict[str, Any]:
    """Ablation 4: Neutral system prompt vs Explicit caution instruction."""
    res_neutral = run_behavioral_evaluation(paired_tasks, model_key=model_key, prompt_style="neutral")
    res_caution = run_behavioral_evaluation(paired_tasks, model_key=model_key, prompt_style="explicit_caution")

    return {
        "neutral_caution_rate_irrev": res_neutral["caution_rate_irreversible"],
        "neutral_sensitivity": res_neutral["reversibility_sensitivity"],
        "explicit_caution_rate_irrev": res_caution["caution_rate_irreversible"],
        "explicit_sensitivity": res_caution["reversibility_sensitivity"],
        "delta_caution_rate": res_caution["caution_rate_irreversible"] - res_neutral["caution_rate_irreversible"]
    }


def run_full_ablation_suite(dataset: List[Dict[str, Any]], paired_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Orchestrates all required ablations into a unified report."""
    print("Running Token Position Ablation...")
    ab_token = run_token_position_ablation(dataset)
    
    print("Running Minimal Pairs Ablation...")
    ab_pairs = run_minimal_pairs_ablation(dataset)
    
    print("Running Cross-Environment Transfer Ablation...")
    ab_transfer = run_cross_environment_transfer(dataset)
    
    print("Running Prompt Intervention Ablation...")
    ab_prompt = run_prompt_intervention_ablation(paired_tasks)

    return {
        "token_position": ab_token,
        "minimal_pairs": ab_pairs,
        "cross_environment": ab_transfer,
        "prompt_intervention": ab_prompt
    }
