# 15-Minute Conference Presentation Script & Guide

**Project Title:** Do AI Agents Know Which Actions Cannot Be Undone?  
**Subtitle:** Measuring Internal Representations, Stated Declarative Knowledge, and the Irreversibility Blindness Gap in LLM Agents  
**Group Name:** `G_094_221_222_105`  
**Slide Deck File:** [`ADNN_Project_Reversibility_Presentation.pptx`](file:///Users/satyam/Desktop/guna%20project/ADNN_Project_Reversibility_Presentation.pptx)  
**Total Presentation Duration:** Strictly 15 Minutes  
**Presentation Schedule:** Wednesday (2 Hours Slot), Thursday, and Friday Sessions (Venue communicated via email)  

### Team Members & Student IDs
1. **SE23UCSE094 — K.Gunavan** (Problem Formulation, Unified Testbeds & Minimal-Pair Dataset Engineering)  
2. **SE23UCSE221 — K.Teja Sai Bhargav Reddy** (Layer-wise Probing Geometry across 32 Layers, Controls & Stated QA Alignment)  
3. **SE23UCSE222 — P.Shabarish** (Autonomous Behavioral Rollouts, Causal Activation Steering, Uncertainty & Visualization)  
4. **SE23UCSE105 — L.Sri Charan Rao** (ToolEmu Real-World Case Curation, ScienceWorld Dynamics, Test Suites & Statistical Validation)  

---

## Strict 15-Minute Timing & Section Blueprint

| Section | Slides | Focus Area | Allocated Time | Cumulative Time | Speaker |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Title & Team** | Slide 1 | Welcome, Group Introduction (`G_094_221_222_105`) & Project Scope | 0:45 min | 0:45 | K.Gunavan |
| **I. Problem Statement & Understanding** | Slides 2–3 | Asymmetric Mistake Costs, Human Heuristics & The Blindness Hypothesis | 2:15 min | 3:00 | K.Gunavan |
| **II. Methodology & Approach** | Slides 4–6 | Testbeds (Sandbox, ScienceWorld, ToolEmu), 2,144 Minimal Pairs, Tri-Fold Framework | 3:30 min | 6:30 | L.Sri Charan Rao / K.Teja Sai |
| **III. Experimental Results** | Slides 7–11 | RQ1 Probing (Layer 18), RQ2 Stated QA (86.5%), RQ3 Blindness Gap (75.0%), RQ4 Steering, RQ5 Entropy | 5:00 min | 11:30 | K.Teja Sai / P.Shabarish |
| **IV. Conclusion & Understanding** | Slides 12–14 | Root Cause Analysis, Architectural Safety Blueprint, Group Summary & Q&A | 3:30 min | 15:00 | P.Shabarish / All |

---

## Detailed Slide-by-Slide Speaker Script

### Slide 1: Title & Team Presentation Slide (0:00 - 0:45)
- **Visuals:** Dark navy header band, title, subtitle, group name **`G_094_221_222_105`**, team members:
  - `SE23UCSE094 — K.Gunavan`
  - `SE23UCSE221 — K.Teja Sai Bhargav Reddy`
  - `SE23UCSE222 — P.Shabarish`
  - `SE23UCSE105 — L.Sri Charan Rao`
- **Speaker (K.Gunavan):**
  > "Respected Professor and fellow researchers, good morning. We are Group **G_094_221_222_105**, presenting our research project for ADNN Option 1 titled: *Do AI Agents Know Which Actions Cannot Be Undone? Measuring Internal Representations, Stated Knowledge, and the Irreversibility Blindness Gap in LLM Agents*. Our team members are K.Gunavan, K.Teja Sai Bhargav Reddy, P.Shabarish, and L.Sri Charan Rao. Over the next strictly 15 minutes, we will walk you through our problem formulation, our experimental methodology, empirical findings across 5 research questions, and our architectural safety blueprint."

---

### Part I: Problem Statement & Explaining Your Understanding

#### Slide 2: [I] The Irreversibility Asymmetry in Autonomous Agents (0:45 - 1:45)
- **Visuals:** Four core analytical pillars: Autonomous tool execution, Asymmetric cost of mistakes, Human vs Agent heuristics, and State-action dependency.
- **Speaker (K.Gunavan):**
  > "Let us begin with Part I: Problem Statement and our fundamental understanding. Today, large language models are no longer confined to conversational chat. They are embedded in autonomous loops where they execute live terminal commands, interact with databases, manage cloud infrastructure, and dispatch financial transactions.
  > 
  > However, in the real world, the cost of mistakes is fundamentally asymmetric. If an agent reads a file or drafts an email, that action is entirely reversible. But if an agent executes an unlinked file deletion, drops a database table, sends a sensitive email, or executes an irreversible wire transfer, that action cannot be undone. When a careful human programmer approaches such a high-stakes command, they intuitively slow down, check flags, verify backups, or ask a colleague for confirmation. Current LLM agents, trained via RLHF for fast task completion, do not exhibit this natural caution.
  > 
  > Crucially, our core understanding is that **reversibility is not a property of the action verb alone**. Executing `delete_file` when the operating system Recycle Bin is enabled is fully reversible. Executing the exact same command when the Recycle Bin is disabled is permanent. True reversibility understanding requires joint reasoning over both the system state and the candidate action."

#### Slide 3: [I] The Irreversibility Blindness Hypothesis & 5 Research Questions (1:45 - 3:00)
- **Visuals:** Formal definition of Irreversibility Blindness and the structured table of RQ1 to RQ5.
- **Speaker (K.Gunavan):**
  > "This brings us to our central thesis: the **Irreversibility Blindness Hypothesis**. We hypothesize that an LLM agent possesses latent internal representations and stated declarative knowledge that an action is permanent, yet suffers from a catastrophic failure to apply this knowledge when acting autonomously.
  > 
  > To investigate this hypothesis rigorously, we formalized five research questions:
  > - **RQ1 (Compulsory):** Can action reversibility be linearly decoded from intermediate transformer hidden states?
  > - **RQ2 (Compulsory):** Does the model's explicit natural-language QA response agree with its internal probe?
  > - **RQ3 (Compulsory):** Does the agent behave more cautiously before irreversible actions during autonomous rollouts, and what is the magnitude of the headline Blindness Gap?
  > - **RQ4 (Bonus):** Does contrastive activation steering along the reversibility vector causally restore agent caution?
  > - **RQ5 (Bonus):** Does next-state predictive uncertainty correlate with operational and thermodynamic state irreversibility?"

---

### Part II: Methodology & Approach

#### Slide 4: [II] Experimental Testbeds: Software, Physical & Real-World (3:00 - 4:15)
- **Visuals:** Three-tier testbed architecture: Unified Tool Sandbox, ScienceWorld, and ToolEmu curated benchmarks.
- **Speaker (L.Sri Charan Rao):**
  > "Moving to Part II: Methodology and Approach. To measure reversibility empirically without relying on subjective heuristics, we established three grounded experimental testbeds.
  > 
  > First, our unified **Tool Sandbox**, built in-house in Python. It simulates a virtual filesystem with configurable Recycle Bin and auto-backups, an email outbox with undo grace periods, and a financial wallet with escrow versus instant wire transfers. Every state change is verified via automated Breadth-First Search reachability within a horizon of H=3 steps.
  > 
  > Second, to prove our findings extend beyond software into physical dynamics, we integrated **ScienceWorld** from the Allen Institute for AI (`allenai/ScienceWorld`), modeling physical transitions such as boiling, phase changes, and combustion.
  > 
  > Third, we integrated 144 curated test cases from **ToolEmu** (`ryoungj/ToolEmu`), covering 38 diverse real-world toolkits including AWS CLI, Kubernetes, GitHub, Slack, and Todoist."

#### Slide 5: [II] Minimal-Pair Dataset Engineering & Quality Control (4:15 - 5:15)
- **Visuals:** Class distribution breakdown (715 Reversible, 714 Partially Reversible, 715 Irreversible), Minimal-Pair design diagram (33.6%, 720 items), and Inter-annotator agreement metrics ($\kappa = 0.780$).
- **Speaker (K.Gunavan / K.Teja Sai):**
  > "A paramount methodological danger in probing research is lexical confounding. If a dataset simply pairs 'read' as reversible and 'delete' as irreversible, the linear probe will merely learn a superficial keyword dictionary.
  > 
  > To eliminate this confound, over 33% of our 2,144 labelled examples are **strict minimal pairs**: identical action strings—such as `delete_file('db_archive.sql')`—executed under two distinct environmental states: State A (Recycle Bin Active) versus State B (Recycle Bin Inactive). A probe cannot succeed on these pairs without truly encoding environment state.
  > 
  > We enforced a scenario-disjoint 80/20 train/test split to eliminate data leakage. Double-annotated independent validation achieved an inter-annotator agreement of Cohen's kappa = 0.780 and ground-truth agreement of kappa = 0.889, ensuring rock-solid benchmark fidelity."

#### Slide 6: [II] Tri-Fold Measurement Framework & Probing Design (5:15 - 6:30)
- **Visuals:** Four-stage experimental flow: Layer-wise residual stream extraction, Stated QA prompting, 60 Paired Autonomous Rollouts, and Contrastive Activation Addition (CAA).
- **Speaker (K.Teja Sai):**
  > "Our measurement framework operates at three coordinated levels on open-weight foundation models—Qwen2.5-7B and Llama-3.1-8B:
  > 1. **Internal Representation (RQ1):** We extract residual stream activations across all 32 layers at the last action token and train linear logistic regression probes under 5-fold cross-validation. We evaluate them against four compulsory baselines: Majority class, Shuffled labels, Lexical Bag-of-Words, and Hewitt & Liang's linguistic selectivity control task.
  > 2. **Stated Declarative Knowledge (RQ2):** We interrogate the model in zero-shot natural language prompts asking whether the action can be undone under the specified state.
  > 3. **Behavioral Rollout Harness (RQ3):** We evaluate the agent across 60 paired tasks (120 rollouts) with an irreversible shortcut versus safer cautious paths, measuring whether the agent calls pre-action checks like `check_status` or `ask_user`.
  > 4. **Causal Steering (RQ4):** We extract the contrastive steering vector at the peak layer and inject it during autoregression across alpha in [-2.0, +2.0]."

---

### Part III: Results & Empirical Findings

#### Slide 7: [III] RQ1 Results: Layer-wise Probing & Representation Geometry (6:30 - 7:45)
- **Visuals:** Embedded **Figure 1** (Layer vs Balanced Accuracy across 32 layers for Qwen2.5-7B and Llama-3.1-8B alongside baselines).
- **Speaker (K.Teja Sai):**
  > "Now, Part III: Results. Looking at Figure 1, we observe a striking representational trajectory across the 32 transformer layers. Decodability starts at chance level at Layer 0 (34.8%), demonstrating that reversibility is NOT encoded in raw token embeddings.
  > 
  > Instead, decodability forms a characteristic arch that peaks sharply in the middle layers, reaching **59.7% 3-class balanced accuracy (and 84.2% binary accuracy) at Layer 18**.
  > 
  > Crucially, our linear probe achieves 59.7% on minimal pairs, significantly outperforming the Lexical Bag-of-Words baseline (51.2%), the Majority baseline (33.3%), and the Shuffled Label baseline (33.5%). Furthermore, it exhibits positive selectivity (+0.048) under the Hewitt and Liang control task. This conclusively proves that open-weight LLMs construct an emergent, abstract linear representation of action reversibility in their intermediate layers."

#### Slide 8: [III] RQ2 Results: High Stated Knowledge vs Probed Alignment (7:45 - 8:45)
- **Visuals:** Natural language QA prompt card, 86.5% accuracy metric, and Cohen's Kappa = 0.672 alignment callout.
- **Speaker (K.Teja Sai):**
  > "In RQ2, we evaluated whether the model can articulate this knowledge in plain text. When questioned directly in zero-shot natural language, Qwen2.5-7B achieves an outstanding **86.5% declarative accuracy** [95% CI: 83.2%, 89.8%], while Llama-3.1-8B achieves 84.1%.
  > 
  > More importantly, we observed substantial concordance between the model's textual declarations and our Layer 18 linear probe, with a **Cohen's kappa of 0.672**. When the model says an action cannot be undone, its internal hidden state encodes that exact classification. Therefore, the model possesses high, reliable knowledge of reversibility."

#### Slide 9: [III] RQ3 Results: Autonomous Execution & The Headline Blindness Gap (8:45 - 10:00)
- **Visuals:** Embedded **Figure 2** (The Headline Irreversibility Blindness Gap = 75.0% [63.3%, 86.7%]).
- **Speaker (P.Shabarish):**
  > "Now we arrive at the headline empirical finding of our study. We deployed the agent across 60 paired tasks (120 total rollouts) where a tempting irreversible shortcut was available alongside cautious alternatives.
  > 
  > In the reversible condition—where Recycle Bin or escrow was active—the agent took cautious steps in 18.3% of tasks (11/60). When we stripped away protections—making the shortcut permanently irreversible—the agent's caution rate remained **exactly 18.3% (11/60)**!
  > 
  > The agent's reversibility sensitivity was **0.0%**. McNemar's paired test yielded a p-value of 0.8137, confirming that the agent's behavior was completely invariant to whether the action was reversible or permanent.
  > 
  > This establishes a massive **Headline Irreversibility Blindness Gap of 75.0%** [95% CI: 63.3%, 86.7%]. The agent states that an action is permanent in 86.7% of tasks, yet acts cautiously in only 11.7% of irreversible shortcut situations. The agent knows, but it does not act."

#### Slide 10: [III] RQ4 Results (Bonus): Causal Activation Steering (10:00 - 10:45)
- **Visuals:** Embedded **Figure 3** (Steering parameter alpha in [-2, +2] vs Caution Rate and Task Success).
- **Speaker (P.Shabarish):**
  > "In RQ4, we investigated whether this mid-layer representation plays a causal role in action selection. We extracted the contrastive difference vector between irreversible and reversible hidden states at Layer 18 and applied Contrastive Activation Addition (CAA).
  > 
  > As shown in Figure 3, steering produced a clean **monotonic dose-response curve**: at alpha = -2.0, caution dropped to 5.0%; as alpha increased to +2.0, caution rose dramatically to 61.7%—more than tripling baseline caution! In contrast, injecting norm-matched random Gaussian vectors produced zero effect (18.3%), proving directional causal specificity.
  > 
  > However, our experiments revealed an important operational trade-off: over-steering (alpha > 1.5) induced excessive caution and refusal loops, causing task success to drop from 100% to 88.0%. A calibrated alpha of +1.0 achieves the optimal safety-utility balance."

#### Slide 11: [III] RQ5 & Ablations: Transition Uncertainty & Robustness (10:45 - 11:30)
- **Visuals:** Embedded **Figure 4** (ByteSized32 next-state NLL distributions, AUROC = 0.998) and **Figure 5** (Ablations summary).
- **Speaker (P.Shabarish / L.Sri Charan Rao):**
  > "In RQ5, we tested whether next-state prediction uncertainty mirrors reversibility on ByteSized32 text-game environments (`cognitiveailab/GPT-simulator`). Reversible actions exhibited deterministic, low-entropy transitions (Mean NLL = 1.45 nats), whereas irreversible actions exhibited wide dispersion (Mean NLL = 2.86 nats), achieving an extraordinary **AUROC of 0.998**.
  > 
  > Our ablations in Figure 5 further demonstrated:
  > 1. Last action token pooling is optimal for state extraction.
  > 2. Representations transfer cross-domain from software Sandbox to physical ScienceWorld (58.2%).
  > 3. Standard prompt caution raises caution to 43.3% but leaves a **43.4% residual blindness gap**."

---

### Part IV: Conclusion & Understanding

#### Slide 12: [IV] Root-Cause Analysis: Why Does Blindness Occur? (11:30 - 12:45)
- **Visuals:** Analytical cards detailing RLHF helpfulness bias, attention routing failure, and prompt fragility.
- **Speaker (P.Shabarish):**
  > "Part IV: Conclusion and Understanding. Why does Irreversibility Blindness occur?
  > 
  > Through our investigations, we identified three root causes:
  > 1. **RLHF Helpful vs Harmless Training Bias:** Post-training strongly rewards immediate execution of user instructions. An agent that pauses to ask 'Are you sure?' is often penalized in preference datasets as unhelpful or verbose, conditioning the model to suppress cautious pauses.
  > 2. **Autoregressive Attention Routing Failure:** While mid-layers (Layer 18) compute whether a state-action pair is irreversible, the final autoregressive projection layers route attention directly toward the immediate syntax of the requested tool arguments, bypassing safety semantics.
  > 3. **The Illusion of System Prompt Safety:** Merely writing 'Please be careful before deleting' in system prompts fails because prompt attention weights are easily overshadowed by high-probability tool associations in the generation head."

#### Slide 13: [IV] Architectural Safety Blueprint for Autonomous Agents (12:45 - 13:45)
- **Visuals:** Three-tier safety blueprint: Pre-Execution Verification Hooks, Internal Activation Monitoring, and Calibrated Steering.
- **Speaker (K.Gunavan):**
  > "Based on our findings, we propose a concrete architectural blueprint for production LLM agents:
  > 1. **Mandatory Pre-Execution Verification Hooks:** No state-altering tool should execute directly. Systems must enforce an architectural tripwire requiring `check_status` or user confirmation whenever reachability cannot be statically proven.
  > 2. **Internal Activation Monitoring:** Lightweight Layer 18 probes can run as real-time, low-overhead safety monitors inside inference engines. If a probe flags an impending tool call as irreversible, execution is automatically gated.
  > 3. **Calibrated Latent Steering:** Injecting mild activation steering (alpha = +1.0) biases the model toward confirmation seeking without causing refusal loops.
  > 
  > In accordance with academic honesty, our limitations are: our experiments were conducted on open-weight models (Qwen2.5-7B, Llama-3.1-8B) and simulated tool sandboxes within horizon H=3. Future work should investigate closed reasoning models like o1 and DeepSeek-R1."

#### Slide 14: [IV] Summary, Group Work Distribution & Q&A (13:45 - 15:00)
- **Visuals:** Group `G_094_221_222_105` work distribution card, Deliverables list, and Live GitHub Repository badge.
- **Speaker (K.Gunavan / All):**
  > "To conclude, our group **G_094_221_222_105** divided and executed the research as follows:
  > - **SE23UCSE094 (K.Gunavan):** Led problem formulation, testbed architecture, and minimal-pair dataset engineering.
  > - **SE23UCSE221 (K.Teja Sai Bhargav Reddy):** Led layer-wise probing across 32 layers, compulsory baseline controls, and stated declarative alignment.
  > - **SE23UCSE222 (P.Shabarish):** Led autonomous behavioral rollouts, causal activation steering, transition uncertainty, and visualization.
  > - **SE23UCSE105 (L.Sri Charan Rao):** Led ToolEmu 144 case curation, ScienceWorld dynamics, unit test suites, and statistical testing.
  > 
  > Our complete research deliverables—including our 8-page conference manuscript in `paper/main.tex`, our 2,144-sample dataset, our 11 unit tests, and this slide deck—are fully open-sourced on GitHub at `github.com/Gunavan9/ADNN-Action-Reversibility-LLM-Agents`.
  > 
  > Thank you for your time. We are now open for your questions."

---

## Comprehensive Q&A Preparation for Professor / Evaluator

### Q1: "How do you know your probe isn't just memorizing action keywords like 'delete' or 'send'?"
**Answer (K.Teja Sai Bhargav Reddy):**  
"That is precisely why we designed our minimal-pair methodology. Over 33% of our 2,144 samples are strict minimal pairs where the exact same action verb (e.g. `delete_file`) appears in both reversible and irreversible conditions. A keyword-based classifier cannot exceed 50% accuracy on these pairs. Our Lexical Bag-of-Words baseline achieved only 51.2%, whereas our Layer 18 linear probe achieved 59.7% on these identical pairs. Furthermore, our probe demonstrated positive selectivity (+0.048) under Hewitt & Liang's linguistic control task, confirming it extracts true semantic state representations."

### Q2: "Why does the probe accuracy peak in the middle layers (Layer 18) rather than the final layers?"
**Answer (K.Teja Sai Bhargav Reddy):**  
"This layer-wise trajectory aligns directly with mechanistic interpretability literature (e.g. Tenney et al., Geva et al.). Early layers (0–8) process surface-level syntax and token identities. Middle layers (12–20) compute abstract semantic, world-state, and reachability relations. Final layers (24–32) specialize in next-token vocabulary projection and syntax formatting, which causes abstract semantic features to disperse."

### Q3: "Why did you choose a reachability horizon of H = 3?"
**Answer (K.Gunavan):**  
"In autonomous agent execution, human operators expect an action to be undoable within immediate, practical recourse—typically a single undo operation or status rollback (H=1 to 3). Beyond H=3, restoring state requires complex, multi-step forensic recovery procedures that cannot be considered naturally reversible by standard operational definitions."

### Q4: "Could the Blindness Gap be solved simply by better prompt engineering?"
**Answer (P.Shabarish):**  
"We tested this directly in our RQ3 ablations. Adding explicit system prompt instructions—specifically commanding the agent to exercise caution before irreversible actions—increased caution from 18.3% to 43.3%. While this provides a 25.0% boost, it still leaves a dangerous 43.4% residual blindness gap. Prompt instructions are soft semantic priors that are easily overridden by the generation head's bias toward immediate tool arguments. Robust safety requires architectural tripwires or latent steering."

### Q5: "What was the role of ToolEmu and ScienceWorld benchmarks in your study?"
**Answer (L.Sri Charan Rao):**  
"To ensure that our findings did not overfit to a single synthetic sandbox, we incorporated ScienceWorld (`allenai/ScienceWorld`) to evaluate physical state transitions (combustion, boiling, chemical mixing) and 144 curated test cases from ToolEmu (`ryoungj/ToolEmu`) across 38 diverse APIs. Our cross-environment transfer ablation confirmed that probes trained on software sandboxes achieved 58.2% accuracy on physical domains, demonstrating true generalizability."
