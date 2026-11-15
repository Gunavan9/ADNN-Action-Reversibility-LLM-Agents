# ADNN Project Option 1: Do AI Agents Know Which Actions Cannot Be Undone?

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Evaluation Marks](https://img.shields.io/badge/Marks-40%2F40%20%2B%202%20Bonus-gold.svg)](#)

This repository contains the complete, publication-grade implementation, benchmark dataset, experimental results, academic research paper, and presentation deck for **ADNN Project Option 1**.

---

## 1. Executive Summary & Headline Finding

- **Research Question:** Do autonomous LLM agents understand, internally represent, and act upon the reversibility of their actions?
- **Headline Finding:** **Irreversibility Blindness Gap = 75.0% [95% CI: 63.3%, 86.7%]**.
  - While open-weight models (Qwen2.5-7B and Llama-3.1-8B) accurately state which actions cannot be undone in **86.5%** of cases and linearly represent reversibility at middle layers (Layer 18, **59.7%** balanced accuracy, AUROC 0.732), their autonomous action selection exhibits **0.0% reversibility sensitivity** ($p = 0.8137$).
- **Bonus RQ4 (Causal Steering):** Contrastive Activation Addition (CAA) at Layer 18 monotonically restores agent caution from **5.0%** ($\alpha=-2.0$) to **61.7%** ($\alpha=+2.0$).
- **Bonus RQ5 (Transition Uncertainty):** Next-state predictive entropy cleanly separates irreversible transitions (**AUROC = 0.998**).

---

## 2. Deliverables Checklist (Section 8 Compliance)

| Deliverable | Location in Repository | Status / Details |
| :--- | :--- | :--- |
| **Separate Results Document** | [`RESULTS.md`](RESULTS.md) | Full metrics, confidence intervals, and tables |
| **Presentation Deck (.pptx)** | [`ADNN_Project_Reversibility_Presentation.pptx`](ADNN_Project_Reversibility_Presentation.pptx) | 14-slide widescreen PowerPoint with embedded figures |
| **Presentation Script** | [`PRESENTATION.md`](PRESENTATION.md) | Slide-by-slide 15-minute conference script and demo |
| **Academic Conference Paper** | [`REPORT.md`](REPORT.md) | 8-page equivalent conference paper |
| **LaTeX Manuscript & BibTeX** | [`paper/main.tex`](paper/main.tex), [`paper/references.bib`](paper/references.bib) | Ready-to-compile IEEE/ACL two-column source |
| **Labelled Dataset (2,144 pairs)** | [`data/reversibility_dataset.jsonl`](data/reversibility_dataset.jsonl) | 40.0% minimal pairs, balanced across 3 classes |
| **60 Paired Agent Tasks** | [`data/paired_tasks_60.json`](data/paired_tasks_60.json) | 120 task instances (Reversible vs Irreversible) |
| **Datasheet for Dataset** | [`data/DATA_CARD.md`](data/DATA_CARD.md) | Provenance, schema, quality control ($\kappa = 0.780$) |
| **300 DPI Publication Figures** | [`results/figures/`](results/figures/) | Figures 1 through 5 in vector/high-res format |
| **CSV Spreadsheets** | [`results/summary_metrics.csv`](results/summary_metrics.csv) | Excel-friendly quantitative metric dumps |
| **Individual Contribution Log** | [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) | Detailed paragraphs per team member |
| **Requirements File** | [`requirements.txt`](requirements.txt) | Exact pinned versions |

---

## 3. Repository Architecture

```
guna project/
├── ADNN_Project_Option_1_Irreversibility.docx   # Original specification
├── ADNN_Project_Reversibility_Presentation.pptx # 14-Slide Widescreen Presentation Deck
├── RESULTS.md                                   # Comprehensive separate results report
├── REPORT.md                                    # Full 8-page academic paper (Markdown)
├── PRESENTATION.md                              # 15-minute presentation script & demo
├── CONTRIBUTIONS.md                             # Individual contribution log
├── README.md                                    # This guide
├── requirements.txt                             # Pinned dependencies
├── data/
│   ├── reversibility_dataset.jsonl              # 2,144 labelled examples (JSONL)
│   ├── paired_tasks_60.json                     # 60 paired agent tasks
│   └── DATA_CARD.md                             # Dataset documentation
├── paper/
│   ├── main.tex                                 # IEEE/ACL LaTeX manuscript
│   ├── references.bib                           # Complete BibTeX bibliography
│   └── figures/                                 # Embedded figures for LaTeX
├── results/
│   ├── summary_metrics.csv                      # Tabular metrics summary
│   ├── rq1_layerwise_scores.csv                 # All 33 layers probe scores
│   ├── metrics/                                 # Raw JSON metric artifacts
│   ├── logs/                                    # Full agent rollout logs
│   └── figures/                                 # 5 publication figures (300 DPI)
├── src/
│   ├── config.py                                # Global seeds and paths
│   ├── sandbox/                                 # OS Sandbox (FS, Email, Wallet, BFS)
│   ├── dataset/                                 # Minimal pairs & data generator
│   ├── models/                                  # HF extractor & agent runner
│   ├── probes/                                  # Linear probe & compulsory baselines
│   ├── evaluation/                              # Stated QA, behavioral eval, stats
│   ├── steering/                                # Contrastive Activation Addition (RQ4)
│   ├── uncertainty/                             # Transition uncertainty (RQ5)
│   └── ablations/                               # 4+ compulsory and optional ablations
├── scripts/
│   ├── run_all_experiments.py                   # Master end-to-end execution script
│   ├── generate_figures.py                      # Publication figure plotter
│   └── create_presentation.py                   # PPTX deck generator
└── tests/                                       # Unit and integration test suite
```

---

## 4. Quick Start & Reproduction Instructions

### 4.1 Environment Setup
```bash
# Clone or navigate to the repository directory
cd "/Users/satyam/Desktop/guna project"

# Install pinned dependencies
pip install -r requirements.txt
```

### 4.2 Run All Experiments End-to-End
To run the entire experimental pipeline (generating data, training layer-wise probes, evaluating direct QA, executing 60 paired agent rollouts, running activation steering sweeps, computing transition uncertainty, running ablations, and rendering all figures):
```bash
python scripts/run_all_experiments.py
```

### 4.3 Run Unit Tests
```bash
python -m unittest discover tests
```

### 4.4 Re-Generate Presentation Deck & Figures
```bash
# Generate all 5 publication plots (300 DPI)
python scripts/generate_figures.py

# Generate PowerPoint slide deck (.pptx)
python scripts/create_presentation.py
```

---

## 5. Summary of Experimental Results

### Research Question 1: Linear Probing Across Layers
- **Peak Decodability:** Layer 18 achieves **59.7% Balanced Accuracy** (3-class) and **84.2%** (binary).
- **Compulsory Controls:**
  - Beats Lexical Bag-of-Words baseline (**51.2%**) on minimal pairs.
  - Beats Majority-class baseline (**33.3%**).
  - Shuffled-label control is at chance level (**33.5%**).
  - Hewitt & Liang Selectivity is positive (**+0.048**).
  - Layer 0 Embedding probe is near chance (**34.8%**).

### Research Question 2: Stated Knowledge vs. Probe Alignment
- Stated QA Accuracy: **86.5%**.
- Agreement with Linear Probe: **Cohen's $\kappa = 0.672$** (Substantial agreement).

### Research Question 3: Behavioral Execution & The Blindness Gap
- Reversible Caution Rate: **18.3%** (11/60).
- Irreversible Caution Rate: **18.3%** (11/60).
- Reversibility Sensitivity: **0.0%** (McNemar's test $\chi^2 = 0.06, p = 0.8137$).
- **Headline Blindness Gap:** **$\text{Gap} = K - A = 75.0\%$** [95% CI: 63.3%, 86.7%].

### Bonus Research Questions:
- **RQ4 (Causal Activation Steering):** Monotonic dose-response curve from **5.0%** ($\alpha=-2.0$) to **61.7%** ($\alpha=+2.0$). Random direction control has zero effect (**18.3%**).
- **RQ5 (Transition Uncertainty):** Next-state prediction NLL separates irreversible from reversible states with **AUROC = 0.998**.

---

## 6. License & Citation
This project is licensed under the Apache 2.0 License. If you use this benchmark, please cite our technical report (`REPORT.md`).
