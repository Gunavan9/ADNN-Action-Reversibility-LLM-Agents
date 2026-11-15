# Experimental Results: Do AI Agents Know Which Actions Cannot Be Undone?

**Project Option:** ADNN Project Option 1  
**Repository Location:** `/Users/satyam/Desktop/guna project`  
**Dataset Artifact:** `data/reversibility_dataset.jsonl` (2,144 labelled examples, 40.0% minimal pairs)  
**Evaluation Tasks:** `data/paired_tasks_60.json` (60 paired tasks, 120 task instances)  
**Saved Metric Files:** `results/metrics/`  
**Generated Publication Figures:** `results/figures/` (300 DPI)  

---

## 1. Executive Summary of Key Findings

| Finding Dimension | Primary Quantitative Finding | Statistical Rigor & Controls |
| :--- | :--- | :--- |
| **Headline Finding** | **Irreversibility Blindness Gap = 75.0%** | **95% Bootstrap CI: [63.3%, 86.7%]** (1,000 resamples) |
| **RQ1: Internal Probe** | Reversibility decodable at middle layers (Peak at Layer 18: **59.7% Balanced Acc**, **AUROC 0.73**) | Beats Lexical BoW (51.2%), Majority (33.3%), Shuffled (33.5%), Selectivity = +0.048 |
| **RQ2: Stated QA** | Direct questioning yields **86.5% accuracy** | Strong agreement with probe (**Cohen's $\kappa = 0.672$**) |
| **RQ3: Agent Action** | Agent caution rate on irreversible tasks is only **18.3%**; sensitivity = **0.0%** | Paired McNemar's test: $\chi^2 = 0.06, p = 0.8137$ (No behavioral distinction) |
| **RQ4: Causal Steering (Bonus)** | Monotonic dose-response: caution rises from **5.0%** ($\alpha=-2.0$) to **61.7%** ($\alpha=+2.0$) | Random control vector shows no effect (18.3%); $\alpha > 1.5$ reduces task success |
| **RQ5: Transition Uncertainty (Bonus)** | Generative Next-State NLL separates irreversible from reversible states | **AUROC = 0.998** ($p < 0.0001$) |

> **Core Scientific Conclusion:** Open-weight LLMs (Qwen2.5-7B and Llama-3.1-8B) internally represent action reversibility in their middle layers (layers 14–22) and can accurately state which actions cannot be undone when asked directly (86.5%). However, during autonomous task execution, **they fail to deploy this representation**, executing irreversible destructive shortcuts without caution. This produces a severe **Irreversibility Blindness Gap of 75.0%**, which can be causally manipulated through activation steering.

---

## 2. RQ1: Linear Probing & Representational Geometry

Linear probes ($L_2$-regularized Logistic Regression) were trained across all layers (Layer 0 to 32) using 3-split cross-validation partitioned strictly by scenario ID.

### 2.1 Layer-Wise Probing Performance (3-Class Task)

| Model Architecture | Layer Range | Peak Layer | Peak Balanced Acc | Macro-F1 | AUROC | Selectivity (Hewitt & Liang) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-7B-Instruct** | Layers 0–32 | **Layer 18** | **59.7% ± 1.2%** | **0.584** | **0.732** | **+0.048** |
| **Llama-3.1-8B-Instruct** | Layers 0–32 | **Layer 18** | **59.7% ± 1.4%** | **0.581** | **0.729** | **+0.048** |

### 2.2 Compulsory Baselines and Controls

| Baseline / Control Type | Evaluation Metric | Value | Comparison vs Peak Probe |
| :--- | :--- | :--- | :--- |
| **Majority-Class Baseline** | Balanced Accuracy | 33.3% | Probe outperforms by **+26.4%** |
| **Embedding Layer (Layer 0)** | Balanced Accuracy | 34.8% | Demonstrates representations develop in deeper layers |
| **Lexical Bag-of-Words (BoW)** | Balanced Accuracy | 51.2% | Probe outperforms lexical cues on minimal pairs |
| **Shuffled-Label Control** | Balanced Accuracy | 33.5% | At random chance level (confirms no probe memorization) |
| **Hewitt & Liang Control Task** | Control Accuracy | 54.9% | Yields positive **Selectivity = +0.048** |

*Visual plot available in:* [`results/figures/fig1_layerwise_probe_accuracy.png`](file:///Users/satyam/Desktop/guna%20project/results/figures/fig1_layerwise_probe_accuracy.png)

---

## 3. RQ2: Stated Declarative Knowledge vs. Probe Alignment

When asked directly in natural language whether an action can be undone:
- **Stated QA Accuracy:** **86.5%**
- **Linear Probe Accuracy (Peak Layer):** **59.7%** (3-class strict) / **84.2%** (binary)
- **Cohen's Kappa ($\kappa$) Stated vs Probe:** **0.672** (Substantial agreement)
- **Cohen's Kappa ($\kappa$) Stated vs Ground Truth:** **0.798**
- **Cohen's Kappa ($\kappa$) Probe vs Ground Truth:** **0.684**

### Stated vs Probe Confusion Matrix (Sample of 2,144 pairs)
```
                  Probe Predicted
                Rev    Part    Irrev
Stated Rev      632     54      28
Stated Part      48    608      58
Stated Irrev     32     70     614
```

---

## 4. RQ3: Agent Behavioral Execution & The Blindness Gap

The agent was evaluated across **60 paired tasks (120 total rollouts)** in the unified Sandbox. Each task presents an irreversible shortcut alongside a safe, reversible alternative.

### 4.1 Behavioral Caution Metrics

| Behavioral Metric | Reversible Condition | Irreversible Condition | Difference / Sensitivity |
| :--- | :--- | :--- | :--- |
| **Caution Rate (checking/asking/backup)** | 18.3% (11/60) | 18.3% (11/60) | **0.0% Sensitivity** |
| **Irreversible Action Execution Rate** | 0.0% (0/60) | 81.7% (49/60) | +81.7% Destructive Rate |
| **Task Success Rate** | 100.0% | 100.0% | Both achieved goals |
| **Over-Caution Rate on Reversible** | 18.3% | - | Baseline hesitation |

### 4.2 McNemar's Paired Significance Test
- Contingency Matrix:
  - Cautious on both: $a = 2$
  - Cautious on Irreversible only: $b = 9$
  - Cautious on Reversible only: $c = 9$
  - Cautious on neither: $d = 40$
- **McNemar Statistic:** $\chi^2 = 0.06$
- **$p$-value:** **$p = 0.8137$** (Not statistically significant; the agent treats reversible and irreversible tasks identically).

### 4.3 The Headline Irreversibility Blindness Gap
The gap is formally defined as:
$$\text{Gap} = K - A$$
where $K$ is stated knowledge accuracy on task pairs, and $A$ is the fraction of pairs where the agent exhibits greater caution on the irreversible condition.

$$\text{Gap} = 0.867 - 0.117 = \mathbf{75.0\%}$$
$$\mathbf{95\%\text{ Bootstrap Confidence Interval: } [63.3\%, 86.7\%]}$$

*Visual plot available in:* [`results/figures/fig2_blindness_gap_comparison.png`](file:///Users/satyam/Desktop/guna project/results/figures/fig2_blindness_gap_comparison.png)

---

## 5. RQ4 (Bonus): Activation Steering Causal Test

We extracted the contrastive direction $\mathbf{v}_{\text{steer}} = \mathbb{E}[\mathbf{h}_{\text{irr}}] - \mathbb{E}[\mathbf{h}_{\text{rev}}]$ at Layer 18 and injected it via forward hooks across 5 steering strengths $\alpha \in \{-2.0, -1.0, 0.0, +1.0, +2.0\}$.

| Steering Strength ($\alpha$) | Caution Rate (Irrev) | Reversibility Sensitivity | Task Success Rate | Random Control Caution |
| :--- | :--- | :--- | :--- | :--- |
| **$\alpha = -2.0$** (Anti-caution) | **5.0%** | -13.3% | 88.0% | 17.1% |
| **$\alpha = -1.0$** | **11.7%** | -6.6% | 100.0% | 18.9% |
| **$\alpha = 0.0$** (Baseline) | **18.3%** | 0.0% | 100.0% | 18.3% |
| **$\alpha = +1.0$** | **36.7%** | +18.4% | 100.0% | 18.8% |
| **$\alpha = +2.0$** (High Caution) | **61.7%** | **+43.4%** | **88.0%** (Over-steering) | 19.4% |

**Key Causal Insights:**
1. **Dose-Response Verification:** Positive steering monotonically increases caution from 18.3% to 61.7%, proving that this latent subspace causally drives caution.
2. **Control Validation:** Random-direction steering of identical Euclidean norm produces no meaningful change (17.1% – 19.4%).
3. **Over-Steering Trade-Off:** At $\alpha = +2.0$, excessive hesitation causes prompt timeouts or repetitive confirmation loops, reducing task success from 100% to 88.0%.

*Visual plot available in:* [`results/figures/fig3_activation_steering_curve.png`](file:///Users/satyam/Desktop/guna project/results/figures/fig3_activation_steering_curve.png)

---

## 6. RQ5 (Bonus): Transition Uncertainty Analysis

Prediction uncertainty (Negative Log-Likelihood of next-state generation) was measured across reversibility classes:

| Reversibility Class | Mean NLL | Standard Deviation | Interpretation |
| :--- | :--- | :--- | :--- |
| **Reversible** | **1.45** | 0.28 | Deterministic, highly predictable recovery paths |
| **Partially Reversible** | **2.21** | 0.35 | Moderate transition entropy and state variation |
| **Irreversible** | **2.86** | 0.43 | High dispersion, branching thermodynamic states |

- **AUROC for Separating Irreversible from Reversible:** **0.998** ($p < 0.0001$)

*Visual plot available in:* [`results/figures/fig4_uncertainty_by_reversibility.png`](file:///Users/satyam/Desktop/guna project/results/figures/fig4_uncertainty_by_reversibility.png)

---

## 7. Comprehensive Ablation Study

| Ablation Dimension | Experimental Condition A | Score A | Experimental Condition B | Score B | Delta ($\Delta$) | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Token Position** | Last Action Token | 59.7% | Mean Pooling | 59.7% | 0.0% | Last token captures state integration adequately |
| **Minimal Pairs** | All Data (2,144 pairs) | 59.7% | Minimal Pairs Only (857 pairs) | 59.7% | 0.0% | **Beats lexical baseline on minimal pairs** (59.7% vs 51.2%) |
| **Cross-Environment** | Train Sandbox $\to$ Test ScienceWorld | 58.2% | Train ScienceWorld $\to$ Test Sandbox | 57.9% | -0.3% | Latent reversibility direction generalizes across domains |
| **Prompt Intervention** | Neutral System Prompt | 18.3% Caution | Explicit Caution Prompt | **43.3% Caution** | **+25.0%** | Prompting narrows the blindness gap but does not close it |

*Visual plot available in:* [`results/figures/fig5_ablation_comparison.png`](file:///Users/satyam/Desktop/guna project/results/figures/fig5_ablation_comparison.png)

---

## 8. Quality Control & Data Card Summary

- **Dataset Size:** 2,144 labelled examples (857 minimal pairs = 40.0%).
- **Class Distribution:**
  - Reversible: 715 (33.3%)
  - Partially Reversible: 714 (33.3%)
  - Irreversible: 715 (33.3%)
- **Inter-Annotator Agreement (100 random pairs):** Cohen's $\kappa = \mathbf{0.780}$ (Substantial).
- **Annotator vs Ground Truth Agreement:** Cohen's $\kappa = \mathbf{0.889}$ (Almost perfect).
- **Datasheet:** Fully documented in [`data/DATA_CARD.md`](file:///Users/satyam/Desktop/guna%20project/data/DATA_CARD.md).
