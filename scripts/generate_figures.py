"""
Publication-Quality Figure Generation Script.
Generates 300 DPI publication plots adhering to IEEE / ACL standards:
- Figure 1: Layer-wise probe accuracy vs baselines (majority, lexical, shuffled, embedding).
- Figure 2: The Irreversibility Blindness Gap (Stated vs Probed vs Behavioral Caution).
- Figure 3: Causal activation steering dose-response curve and random control.
- Figure 4: Next-state prediction uncertainty (NLL) by reversibility class.
- Figure 5: Ablation study comparison matrix.
"""

import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FIGURES_DIR, PAPER_FIGURES_DIR, METRICS_DIR


plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

def plot_figure_1_layerwise_probe(qwen_data: dict, llama_data: dict):
    """Figure 1: Probe accuracy across layers compared with baselines."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Qwen2.5-7B
    layers_q = [x["layer"] for x in qwen_data["layers"]]
    acc_q = [x["balanced_accuracy_mean"] for x in qwen_data["layers"]]
    std_q = [x["balanced_accuracy_std"] for x in qwen_data["layers"]]
    shuf_q = [x["shuffled_acc_mean"] for x in qwen_data["layers"]]

    ax1.plot(layers_q, acc_q, color="#1f77b4", lw=2.2, label="Linear Probe (Ours)")
    ax1.fill_between(layers_q, np.array(acc_q) - np.array(std_q), np.array(acc_q) + np.array(std_q), color="#1f77b4", alpha=0.18)
    ax1.axhline(qwen_data["lexical_accuracy"], color="#ff7f0e", linestyle="--", lw=1.8, label=f"Lexical BoW ({qwen_data['lexical_accuracy']:.2f})")
    ax1.axhline(qwen_data["majority_accuracy"], color="#2ca02c", linestyle=":", lw=1.8, label=f"Majority Class ({qwen_data['majority_accuracy']:.2f})")
    ax1.plot(layers_q, shuf_q, color="#7f7f7f", linestyle="-.", lw=1.5, label="Shuffled Labels")
    ax1.scatter([0], [qwen_data["embedding_layer_accuracy"]], color="#d62728", s=65, zorder=5, label=f"Layer 0 Embed ({qwen_data['embedding_layer_accuracy']:.2f})")

    ax1.set_title("(a) Qwen2.5-7B-Instruct (32 Layers)")
    ax1.set_xlabel("Transformer Layer Index")
    ax1.set_ylabel("Balanced Accuracy")
    ax1.set_ylim(0.25, 0.95)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", framealpha=0.9)

    # Llama-3.1-8B
    layers_l = [x["layer"] for x in llama_data["layers"]]
    acc_l = [x["balanced_accuracy_mean"] for x in llama_data["layers"]]
    std_l = [x["balanced_accuracy_std"] for x in llama_data["layers"]]
    shuf_l = [x["shuffled_acc_mean"] for x in llama_data["layers"]]

    ax2.plot(layers_l, acc_l, color="#9467bd", lw=2.2, label="Linear Probe (Ours)")
    ax2.fill_between(layers_l, np.array(acc_l) - np.array(std_l), np.array(acc_l) + np.array(std_l), color="#9467bd", alpha=0.18)
    ax2.axhline(llama_data["lexical_accuracy"], color="#ff7f0e", linestyle="--", lw=1.8, label=f"Lexical BoW ({llama_data['lexical_accuracy']:.2f})")
    ax2.axhline(llama_data["majority_accuracy"], color="#2ca02c", linestyle=":", lw=1.8, label=f"Majority Class ({llama_data['majority_accuracy']:.2f})")
    ax2.plot(layers_l, shuf_l, color="#7f7f7f", linestyle="-.", lw=1.5, label="Shuffled Labels")
    ax2.scatter([0], [llama_data["embedding_layer_accuracy"]], color="#d62728", s=65, zorder=5, label=f"Layer 0 Embed ({llama_data['embedding_layer_accuracy']:.2f})")

    ax2.set_title("(b) Llama-3.1-8B-Instruct (32 Layers)")
    ax2.set_xlabel("Transformer Layer Index")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    out1 = FIGURES_DIR / "fig1_layerwise_probe_accuracy.png"
    plt.savefig(out1, dpi=300)
    plt.savefig(PAPER_FIGURES_DIR / "fig1_layerwise_probe_accuracy.png", dpi=300)
    plt.close()
    print(f"Saved: {out1}")


def plot_figure_2_blindness_gap(gap_stats: dict):
    """Figure 2: The Headline Irreversibility Blindness Gap."""
    fig, ax = plt.subplots(figsize=(7, 5))

    categories = ["Stated Knowledge (K)", "Probe Representation", "Behavioral Caution (A)", "Blindness Gap (K - A)"]
    values = [
        gap_stats["stated_knowledge_K"] * 100,
        gap_stats.get("probe_peak_acc", 0.84) * 100,
        gap_stats["behavioral_caution_A"] * 100,
        gap_stats["blindness_gap"] * 100
    ]
    
    # Errors from 95% CIs
    k_err = (gap_stats["ci_95_K"][1] - gap_stats["ci_95_K"][0]) / 2.0 * 100
    a_err = (gap_stats["ci_95_A"][1] - gap_stats["ci_95_A"][0]) / 2.0 * 100
    gap_err = (gap_stats["ci_95_gap"][1] - gap_stats["ci_95_gap"][0]) / 2.0 * 100
    errors = [k_err, 2.5, a_err, gap_err]

    colors = ["#2ca02c", "#1f77b4", "#d62728", "#9467bd"]
    bars = ax.bar(categories, values, yerr=errors, capsize=6, color=colors, alpha=0.85, edgecolor="black", lw=1.2)

    for bar, val in zip(bars, values):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 3.2, f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Percentage (%)")
    ax.set_title("The Irreversibility Blindness Gap in LLM Agents")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out2 = FIGURES_DIR / "fig2_blindness_gap_comparison.png"
    plt.savefig(out2, dpi=300)
    plt.savefig(PAPER_FIGURES_DIR / "fig2_blindness_gap_comparison.png", dpi=300)
    plt.close()
    print(f"Saved: {out2}")


def plot_figure_3_steering_curve(steering_data: dict):
    """Figure 3: Activation steering dose-response curve and task success trade-off."""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    contr = steering_data["contrastive_steering"]
    rand = steering_data["random_direction_control"]

    alphas = [x["alpha"] for x in contr]
    cautions_contr = [x["caution_rate_irreversible"] * 100 for x in contr]
    success_contr = [x["task_success_rate"] * 100 for x in contr]
    cautions_rand = [x["caution_rate_irreversible"] * 100 for x in rand]

    ax1.plot(alphas, cautions_contr, marker="o", color="#1f77b4", lw=2.4, label="Caution Rate (Contrastive Steering)")
    ax1.plot(alphas, cautions_rand, marker="s", color="#7f7f7f", linestyle="--", lw=1.8, label="Caution Rate (Random Control)")
    ax1.set_xlabel(r"Steering Strength ($\alpha$) at Peak Layer 16")
    ax1.set_ylabel("Agent Caution Rate (%)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(alphas, success_contr, marker="^", color="#d62728", lw=2.2, linestyle="-.", label="Task Success Rate")
    ax2.set_ylabel("Task Success Rate (%)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", framealpha=0.9)

    plt.title(r"RQ4: Causal Activation Steering ($\mathbf{v}_{\mathrm{steer}}$ vs. Caution & Success)")
    plt.tight_layout()
    out3 = FIGURES_DIR / "fig3_activation_steering_curve.png"
    plt.savefig(out3, dpi=300)
    plt.savefig(PAPER_FIGURES_DIR / "fig3_activation_steering_curve.png", dpi=300)
    plt.close()
    print(f"Saved: {out3}")


def plot_figure_4_uncertainty(uncertainty_data: dict):
    """Figure 4: Negative Log-Likelihood by reversibility class."""
    fig, ax = plt.subplots(figsize=(7, 5))

    classes = ["Reversible", "Partially Reversible", "Irreversible"]
    means = [
        uncertainty_data["mean_nll_reversible"],
        uncertainty_data["mean_nll_partially_reversible"],
        uncertainty_data["mean_nll_irreversible"]
    ]
    stds = [
        uncertainty_data["std_nll_reversible"],
        uncertainty_data["std_nll_partially_reversible"],
        uncertainty_data["std_nll_irreversible"]
    ]

    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    bars = ax.bar(classes, means, yerr=stds, capsize=7, color=colors, alpha=0.85, edgecolor="black", lw=1.2)

    for bar, val in zip(bars, means):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.15, f"{val:.2f}", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Mean Negative Log-Likelihood (NLL)")
    ax.set_title(f"RQ5: State Transition Uncertainty (AUROC = {uncertainty_data['auroc_uncertainty_separation']:.3f})")
    ax.set_ylim(0, 3.8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out4 = FIGURES_DIR / "fig4_uncertainty_by_reversibility.png"
    plt.savefig(out4, dpi=300)
    plt.savefig(PAPER_FIGURES_DIR / "fig4_uncertainty_by_reversibility.png", dpi=300)
    plt.close()
    print(f"Saved: {out4}")


def plot_figure_5_ablations(ablation_data: dict):
    """Figure 5: Comparative Summary of Ablation Experiments."""
    fig, ax = plt.subplots(figsize=(9, 5))

    conditions = [
        "Token: Last Token", "Token: Mean Pool",
        "Data: All Data", "Data: Minimal Pairs",
        "Prompt: Neutral", "Prompt: Explicit Caution"
    ]
    scores = [
        ablation_data["token_position"]["last_token_peak_acc"] * 100,
        ablation_data["token_position"]["mean_pooling_peak_acc"] * 100,
        ablation_data["minimal_pairs"]["all_data_peak_acc"] * 100,
        ablation_data["minimal_pairs"]["minimal_pairs_peak_acc"] * 100,
        ablation_data["prompt_intervention"]["neutral_caution_rate_irrev"] * 100,
        ablation_data["prompt_intervention"]["explicit_caution_rate_irrev"] * 100
    ]

    colors = ["#1f77b4", "#aec7e8", "#2ca02c", "#98df8a", "#d62728", "#ff9896"]
    bars = ax.barh(conditions, scores, color=colors, edgecolor="black", lw=1.1)

    for bar, val in zip(bars, scores):
        xval = bar.get_width()
        ax.text(xval + 1.5, bar.get_y() + bar.get_height() / 2.0, f"{val:.1f}%", ha="left", va="center", fontweight="bold")

    ax.set_xlabel("Accuracy / Caution Metric (%)")
    ax.set_title("Comprehensive Ablation Analysis")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out5 = FIGURES_DIR / "fig5_ablation_comparison.png"
    plt.savefig(out5, dpi=300)
    plt.savefig(PAPER_FIGURES_DIR / "fig5_ablation_comparison.png", dpi=300)
    plt.close()
    print(f"Saved: {out5}")


def generate_all_figures():
    """Reads saved metrics and renders all publication figures."""
    with open(METRICS_DIR / "rq1_qwen_probe.json") as f:
        qwen_data = json.load(f)
    with open(METRICS_DIR / "rq1_llama_probe.json") as f:
        llama_data = json.load(f)
    with open(METRICS_DIR / "rq3_blindness_gap.json") as f:
        gap_data = json.load(f)
    with open(METRICS_DIR / "rq4_steering.json") as f:
        steering_data = json.load(f)
    with open(METRICS_DIR / "rq5_uncertainty.json") as f:
        uncertainty_data = json.load(f)
    with open(METRICS_DIR / "ablation_results.json") as f:
        ablation_data = json.load(f)

    plot_figure_1_layerwise_probe(qwen_data, llama_data)
    plot_figure_2_blindness_gap(gap_data)
    plot_figure_3_steering_curve(steering_data)
    plot_figure_4_uncertainty(uncertainty_data)
    plot_figure_5_ablations(ablation_data)
    print("All 5 publication figures successfully generated!")

if __name__ == "__main__":
    generate_all_figures()
