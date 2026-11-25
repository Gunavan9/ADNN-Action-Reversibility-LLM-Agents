# Individual Contribution Log

**Project Title:** Do AI Agents Know Which Actions Cannot Be Undone?  
**Subtitle:** Measuring Internal Representations, Stated Declarative Knowledge, and the Irreversibility Blindness Gap in LLM Agents  
**Course:** Advanced Deep Neural Networks (ADNN) — Project Option 1  
**Repository:** [https://github.com/Gunavan9/ADNN-Action-Reversibility-LLM-Agents](https://github.com/Gunavan9/ADNN-Action-Reversibility-LLM-Agents)  

---

## Team Structure & Work Distribution

### Team Member 1: Gunavan (Project Direction, Testbed Engineering & Dataset Curation)
**Key Responsibilities & Deliverables:**
- **Problem Formulation & Scope:** Conceptualized the overarching research framework investigating the gap between latent world-model representations and autonomous tool execution.
- **Unified Tool Sandbox (`src/sandbox/`):** Designed and developed the Python mock operating environment simulating the Virtual FileSystem (Recycle Bin toggle, snapshot backups), Email Outbox (undo windows vs instant dispatch), and Wallet (escrow vs irreversible wire transfers) with automated Breadth-First Search (BFS) reachability validation within horizon $H=3$.
- **Benchmark Dataset Engineering (`src/dataset/`):** Curated the 2,144-sample dataset (`data/reversibility_dataset.jsonl`) balanced across Reversible (715), Partially Reversible (714), and Irreversible (715) classes.
- **Minimal-Pair Design:** Engineered 720 minimal-pair transitions (33.6% of the dataset) holding the action verb constant while toggling environmental state to rigorously eliminate lexical confounding.
- **Inter-Annotator Agreement Study:** Conducted independent double-annotation quality control, achieving Cohen's kappa $\kappa = 0.780$ (inter-annotator) and $\kappa = 0.889$ (ground-truth agreement).
- **Academic Paper Drafting:** Led the drafting of Section 1 (Introduction), Section 2 (Related Work), and Section 3 (Testbed & Dataset Design) in `paper/main.tex` and `REPORT.md`.

---

### Team Member 2: Teja Sai (Probing Architecture, Controls & Stated Knowledge Analysis)
**Key Responsibilities & Deliverables:**
- **Representation Extraction Pipeline (`src/probes/`):** Implemented the layer-wise hidden state extraction harness across all 32 transformer layers for open-weight foundation models (Qwen2.5-7B and Llama-3.1-8B) using PyTorch and HuggingFace Transformers.
- **Linear Probing Classifiers (RQ1):** Trained L2-regularized logistic regression and linear SVM probes under 5-fold cross-validation, discovering the mid-network peak decodability at Layer 18 (59.7% 3-class balanced accuracy, 84.2% binary accuracy).
- **Compulsory Baseline Controls:** Engineered and executed all mandatory baseline comparisons: Majority class (33.3%), Shuffled labels (33.5%), Lexical Bag-of-Words (51.2%), and Hewitt & Liang's linguistic selectivity control task (+0.048 selectivity).
- **Stated Declarative Knowledge Evaluation (RQ2):** Designed the zero-shot and few-shot natural language prompt elicitation harness (`src/probes/probe_stated_knowledge.py`), demonstrating 86.5% stated accuracy and substantial statistical agreement with internal probes (Cohen's $\kappa = 0.672$).
- **Academic Paper Drafting:** Authored Section 4 (Representation Probing Methodology & Geometry) and Section 5 (Stated Declarative Knowledge Alignment).

---

### Team Member 3: Shabarish (Behavioral Rollouts, Activation Steering & Uncertainty Analysis)
**Key Responsibilities & Deliverables:**
- **Autonomous Behavioral Rollout Harness (RQ3, `src/evaluation/behavioral_eval.py`):** Constructed 60 minimal-pair agent tasks (120 total rollouts) evaluating autonomous tool calling in Reversible vs Irreversible conditions; measured the headline Irreversibility Blindness Gap ($75.0\%$ [$63.3\%, 86.7\%$]) and computed McNemar's paired statistical test ($\chi^2 = 0.06, p = 0.8137$).
- **Causal Activation Steering (RQ4, `src/steering/activation_steering.py`):** Implemented Contrastive Activation Addition (CAA) at Layer 18, demonstrating a monotonic dose-response curve increasing caution from 5.0% to 61.7%, and verified directional specificity using norm-matched random Gaussian vectors.
- **Generative Transition Uncertainty (RQ5, `src/uncertainty/uncertainty_eval.py`):** Formulated and executed next-state prediction entropy benchmarks on ByteSized32 (`cognitiveailab/GPT-simulator`), achieving AUROC $= 0.998$ for separating irreversible from reversible transitions based purely on next-token perplexity.
- **Publication Graphics & Presentation Deck:** Engineered the automated figure generator (`scripts/generate_figures.py`) producing all five 300 DPI publication figures, authored the 14-slide widescreen PowerPoint presentation deck (`ADNN_Project_Reversibility_Presentation.pptx`), and prepared the 15-minute conference script (`PRESENTATION.md`).
- **Academic Paper Drafting:** Authored Section 6 (Behavioral Evaluation & Blindness Gap), Section 7 (Causal Steering & Uncertainty Analysis), Section 8 (Honest Limitations), and compiled `paper/references.bib`.

---

## Verification & Reproducibility Sign-Off
All three team members jointly verified and validated:
1. Complete reproduction of all numerical results via `scripts/run_all_experiments.py`.
2. Clean execution of the full unit test suite via `python -m unittest discover tests` (11/11 tests passing).
3. Exact consistency across `paper/main.tex`, `REPORT.md`, `RESULTS.md`, and `ADNN_Project_Reversibility_Presentation.pptx`.
