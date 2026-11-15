# Individual Contribution Log

**Project Title:** Do AI Agents Know Which Actions Cannot Be Undone?  
**Course:** ADNN Project Option 1  
**Team Structure & Work Distribution:**  

---

### Team Member 1: Satyam (Lead Researcher & Architecture Engineering)
**Contribution:**  
Satyam conceptualized the experimental architecture, formalizing the tri-fold measurement framework comparing internal linear probes, stated declarative QA, and autonomous agent behavior. He designed and implemented the unified Tool Sandbox environment (`src/sandbox/`), including the virtual filesystem, email outbox, and wallet engines with exact state save/restore capabilities and automated BFS reversibility verification. Additionally, he authored the core behavioral evaluation harness (`src/evaluation/behavioral_eval.py`), orchestrated the 60 minimal-pair tasks, and led the drafting of the academic paper report and LaTeX manuscript.

---

### Team Member 2: Guna (Dataset Engineering, Quality Control & Probing)
**Contribution:**  
Guna led the dataset curation and quality control pipeline (`src/dataset/`), generating 2,144 labelled state-action transitions across the Sandbox, ScienceWorld, and ToolEmu environments. He ensured that over 40% of the dataset consisted of strict minimal pairs to control for lexical confounding. Guna conducted the double-annotation inter-annotator agreement study, computing Cohen's kappa scores ($\kappa = 0.780$). Furthermore, he implemented the layer-wise linear probing pipeline (`src/probes/`), integrating the compulsory majority, lexical Bag-of-Words, shuffled label, and Hewitt & Liang control task baselines.

---

### Team Member 3: Secondary Collaborator (Steering, Uncertainty & Visualization)
**Contribution:**  
This member implemented the bonus research components: Contrastive Activation Addition (CAA) steering (`src/steering/activation_steering.py`) and generative transition uncertainty analysis (`src/uncertainty/uncertainty_eval.py`). They executed the multi-strength steering sweeps across $\alpha \in \{-2.0, -1.0, 0.0, 1.0, 2.0\}$ along with the norm-matched random vector controls. They also engineered the publication figure generator (`scripts/generate_figures.py`), rendering all five 300 DPI conference figures, built the 14-slide widescreen presentation deck (`ADNN_Project_Reversibility_Presentation.pptx`), and verified reproducible environment scripts.
