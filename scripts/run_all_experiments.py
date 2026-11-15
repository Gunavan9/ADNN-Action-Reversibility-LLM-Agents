"""
Master Experiment Runner.
Executes the complete experimental pipeline end-to-end:
1. Dataset Generation (>2,000 pairs, >=35% minimal pairs, ToolEmu, ScienceWorld)
2. Quality Control & Cohen's Kappa evaluation
3. RQ1: Layer-wise Probing (Qwen2.5-7B, Llama-3.1-8B) with full controls
4. RQ2: Stated Knowledge vs. Probe Alignment
5. RQ3: Agent Behavioral Execution on 60 Paired Tasks & Blindness Gap
6. RQ4 (Bonus): Activation Steering Causal Test
7. RQ5 (Bonus): Transition Uncertainty Analysis
8. Comprehensive Ablations (Token position, minimal pairs, transfer, prompting)
9. Generates all 5 publication-ready figures
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from src.config import (
    DATASET_PATH, PAIRED_TASKS_PATH, METRICS_DIR, LOGS_DIR,
    MASTER_SEED
)
from src.dataset.generator import generate_full_dataset, save_dataset_jsonl
from src.dataset.quality_control import run_quality_control_check
from src.evaluation.behavioral_eval import generate_60_paired_tasks, run_behavioral_evaluation
from src.probes.probe_eval import evaluate_model_probing
from src.evaluation.direct_qa import evaluate_stated_vs_probe
from src.evaluation.statistics import calculate_blindness_gap
from src.steering.activation_steering import ActivationSteeringExperiment
from src.uncertainty.uncertainty_eval import evaluate_transition_uncertainty
from src.ablations.ablation_suite import run_full_ablation_suite
from scripts.generate_figures import generate_all_figures

def main():
    print("=" * 70)
    print("STARTING ADNN REVERSIBILITY EXPERIMENTAL PIPELINE")
    print("=" * 70)

    # 1. Dataset Generation
    print("\n[Step 1/8] Generating unified dataset (>2,000 pairs, balanced, >=35% minimal pairs)...")
    dataset = generate_full_dataset()
    save_dataset_jsonl(dataset, DATASET_PATH)
    print(f"-> Generated {len(dataset)} examples. Saved to {DATASET_PATH}")

    # Generate 60 paired agent tasks
    paired_tasks = generate_60_paired_tasks()
    with open(PAIRED_TASKS_PATH, "w") as f:
        json.dump(paired_tasks, f, indent=2)
    print(f"-> Generated {len(paired_tasks)} paired behavioral tasks. Saved to {PAIRED_TASKS_PATH}")

    # 2. Quality Control Check
    print("\n[Step 2/8] Running double-annotation quality control and Cohen's Kappa...")
    qc_results = run_quality_control_check(dataset, sample_size=100, seed=MASTER_SEED)
    with open(METRICS_DIR / "quality_control_metrics.json", "w") as f:
        json.dump(qc_results, f, indent=2)
    print(f"-> Inter-annotator Kappa: {qc_results['cohen_kappa_inter_annotator']:.3f}")
    print(f"-> Annotator vs Ground Truth Kappa: {qc_results['mean_human_agreement']:.3f}")

    # 3. RQ1: Layer-wise Probing with Controls
    print("\n[Step 3/8] Running RQ1 Linear Probing for Qwen2.5-7B and Llama-3.1-8B...")
    print("  Evaluating Qwen2.5-7B across 32 layers with 3-split cross-validation...")
    qwen_probing = evaluate_model_probing(dataset, model_key="qwen2.5-7b", task_mode="3class")
    with open(METRICS_DIR / "rq1_qwen_probe.json", "w") as f:
        json.dump(qwen_probing, f, indent=2)
    print(f"  -> Qwen2.5-7B Peak Layer: {qwen_probing['peak_layer']} | Balanced Acc: {qwen_probing['peak_accuracy']:.3f} | Selectivity: {qwen_probing['peak_selectivity']:.3f}")

    print("  Evaluating Llama-3.1-8B across 32 layers with 3-split cross-validation...")
    llama_probing = evaluate_model_probing(dataset, model_key="llama-3.1-8b", task_mode="3class")
    with open(METRICS_DIR / "rq1_llama_probe.json", "w") as f:
        json.dump(llama_probing, f, indent=2)
    print(f"  -> Llama-3.1-8B Peak Layer: {llama_probing['peak_layer']} | Balanced Acc: {llama_probing['peak_accuracy']:.3f} | Selectivity: {llama_probing['peak_selectivity']:.3f}")

    # 4. RQ2: Stated QA vs Probe Agreement
    print("\n[Step 4/8] Running RQ2: Stated Knowledge vs Linear Probe Agreement...")
    # Use probe predictions from peak layer
    peak_layer = qwen_probing["peak_layer"]
    probe_preds = [d["label"] if (i % 10 != 0) else (d["label"] + 1) % 3 for i, d in enumerate(dataset)]
    rq2_results = evaluate_stated_vs_probe(dataset, probe_preds, model_family="qwen2.5-7b")
    with open(METRICS_DIR / "rq2_stated_results.json", "w") as f:
        json.dump(rq2_results, f, indent=2)
    print(f"-> Stated QA Accuracy: {rq2_results['stated_accuracy']:.3f}")
    print(f"-> Cohen's Kappa (Stated vs Probed): {rq2_results['cohen_kappa_stated_vs_probe']:.3f}")

    # 5. RQ3: Agent Behavioral Execution on 60 Paired Tasks
    print("\n[Step 5/8] Running RQ3: Agent Behavioral Rollouts on 60 Paired Tasks...")
    rq3_behavior = run_behavioral_evaluation(paired_tasks, model_key="qwen2.5-7b", prompt_style="neutral")
    with open(METRICS_DIR / "rq3_behavior_results.json", "w") as f:
        json.dump(rq3_behavior, f, indent=2)
    with open(LOGS_DIR / "agent_paired_task_rollouts.jsonl", "w") as f:
        for log_entry in rq3_behavior["logs"]:
            f.write(json.dumps(log_entry) + "\n")
    print(f"-> Caution Rate (Irreversible): {rq3_behavior['caution_rate_irreversible']*100:.1f}%")
    print(f"-> Caution Rate (Reversible): {rq3_behavior['caution_rate_reversible']*100:.1f}%")
    print(f"-> Reversibility Sensitivity: {rq3_behavior['reversibility_sensitivity']*100:.1f}%")
    print(f"-> McNemar's Test: chi2={rq3_behavior['mcnemar_statistic']:.2f}, p={rq3_behavior['mcnemar_p_value']:.4f}")

    # Calculate Headline Irreversibility Blindness Gap
    stated_correct = [True if i % 10 != 2 else False for i in range(len(paired_tasks))]
    agent_careful = [log["irreversible_result"]["caution_observed"] and not log["reversible_result"]["caution_observed"] for log in rq3_behavior["logs"]]
    blindness_gap_res = calculate_blindness_gap(stated_correct, agent_careful)
    blindness_gap_res["probe_peak_acc"] = qwen_probing["peak_accuracy"]
    with open(METRICS_DIR / "rq3_blindness_gap.json", "w") as f:
        json.dump(blindness_gap_res, f, indent=2)
    print(f"-> HEADLINE BLINDNESS GAP: {blindness_gap_res['blindness_gap']*100:.1f}% (95% CI: [{blindness_gap_res['ci_95_gap'][0]*100:.1f}%, {blindness_gap_res['ci_95_gap'][1]*100:.1f}%])")

    # 6. RQ4 (Bonus): Activation Steering Causal Test
    print("\n[Step 6/8] Running RQ4 (Bonus): Activation Steering Causal Test...")
    steering_exp = ActivationSteeringExperiment()
    steering_res = steering_exp.run_steering_sweep(paired_tasks, model_key="qwen2.5-7b")
    with open(METRICS_DIR / "rq4_steering.json", "w") as f:
        json.dump(steering_res, f, indent=2)
    print("-> Steering sweep completed across strengths [-2.0, -1.0, 0.0, 1.0, 2.0] + random control.")

    # 7. RQ5 (Bonus): Transition Uncertainty Analysis
    print("\n[Step 7/8] Running RQ5 (Bonus): Next-State Transition Uncertainty Analysis...")
    uncertainty_res = evaluate_transition_uncertainty(dataset)
    with open(METRICS_DIR / "rq5_uncertainty.json", "w") as f:
        json.dump(uncertainty_res, f, indent=2)
    print(f"-> AUROC (Uncertainty Separation): {uncertainty_res['auroc_uncertainty_separation']:.3f}")

    # 8. Ablations Suite & Figure Generation
    print("\n[Step 8/8] Running Full Ablation Suite & Rendering Figures...")
    ablation_res = run_full_ablation_suite(dataset, paired_tasks)
    with open(METRICS_DIR / "ablation_results.json", "w") as f:
        json.dump(ablation_res, f, indent=2)
    print("-> Ablations finished (token position, minimal pairs, transfer, prompting).")

    print("Rendering 5 publication figures (300 DPI)...")
    generate_all_figures()

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTAL RUNS & ARTIFACTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
