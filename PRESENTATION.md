# 15-Minute Conference Presentation Script & Slide Deck Guide

**Title:** Do AI Agents Know Which Actions Cannot Be Undone?  
**Subtitle:** Probing, Stated Knowledge, and the Irreversibility Blindness Gap in LLM Agents  
**Slide Deck File:** [`ADNN_Project_Reversibility_Presentation.pptx`](file:///Users/satyam/Desktop/guna%20project/ADNN_Project_Reversibility_Presentation.pptx)  
**Total Duration:** 15 Minutes (12 min presentation + 3 min live demo & Q&A)  

---

## Slide 1: Title Slide (0:00 - 1:00)
- **Visuals:** Dark navy header band, title, authors, project scope (Qwen2.5-7B & Llama-3.1-8B).
- **Speaker Script:**
  > "Good morning everyone. Today, I am presenting our work on a fundamental question in AI safety and autonomous agency: *Do AI Agents Know Which Actions Cannot Be Undone?* As large language models are increasingly deployed to execute tools, manage file systems, and execute transactions, the cost of mistakes becomes deeply asymmetric. Reading a file is easily undone; unlinked file deletion or sending a confidential email is permanent. In this project, we provide the first systematic empirical measurement of whether LLM agents represent reversibility internally, whether they understand it when asked, and whether this knowledge reaches their autonomous behavior."

---

## Slide 2: Motivation & Background (1:00 - 2:15)
- **Visuals:** Four core pillars: Autonomous actions, Asymmetric mistake costs, Human heuristics, The Blindness hypothesis.
- **Speaker Script:**
  > "In classical reinforcement learning, reversibility has been studied as attainable utility preservation and reachability. But in modern LLM agent benchmarks like ToolEmu and SafeAgentBench, risk is evaluated broadly without isolating reversibility. When a careful human programmer approaches a dangerous terminal command or an email broadcast, they naturally slow down, double-check flags, or ask for confirmation. Our core hypothesis is what we term **Irreversibility Blindness**: an agent might accurately state that an action is permanent when queried in a QA prompt, yet act completely recklessly when deployed autonomously."

---

## Slide 3: Research Questions (2:15 - 3:30)
- **Visuals:** Structured table of RQ1 to RQ5.
- **Speaker Script:**
  > "To investigate this, we designed five research questions. RQ1 tests internal linear decodability across layers. RQ2 evaluates whether stated textual answers agree with the probe. RQ3 evaluates autonomous agent rollouts on paired tasks to measure the headline Blindness Gap. Finally, for our bonus investigations, RQ4 provides a causal test via Contrastive Activation Steering, and RQ5 explores next-state prediction uncertainty."

---

## Slide 4: Testbeds & Minimal-Pair Dataset (3:30 - 5:00)
- **Visuals:** Overview of Tool Sandbox, ScienceWorld (`github.com/allenai/ScienceWorld`), ToolEmu (`github.com/ryoungj/ToolEmu`), and the minimal-pair design.
- **Speaker Script:**
  > "A core methodological requirement of our work is that probes must not simply learn lexical keywords like 'delete'. Therefore, over 33% of our 2,144 labelled examples are **minimal pairs**: instances where the exact same action verb—such as `delete_file`—is executed, but its reversibility differs entirely based on the environment's internal state (for example, whether the OS Recycle Bin is enabled or disabled). In addition to our unified Sandbox, we evaluated physical transitions in ScienceWorld (`allenai/ScienceWorld`) and 144 curated test cases from ToolEmu (`ryoungj/ToolEmu`, Apache-2.0). Our dataset achieved an inter-annotator agreement of kappa = 0.780 and ground-truth agreement of kappa = 0.889 across 715 Reversible, 714 Partially Reversible, and 715 Irreversible examples."


---

## Slide 5: RQ1 - Linear Probing (5:00 - 6:45)
- **Visuals:** Embedded Figure 1 (Layer vs Balanced Accuracy across Qwen2.5-7B and Llama-3.1-8B with baselines).
- **Speaker Script:**
  > "Here in Figure 1, we show our linear probing results across all 32 layers for Qwen2.5-7B and Llama-3.1-8B. Notice the characteristic arch: reversibility decodability peaks sharply in the middle layers, reaching 59.7% 3-class balanced accuracy (and 84.2% binary accuracy) at Layer 18. Crucially, the probe outperforms our Lexical Bag-of-Words baseline (51.2%) on minimal pairs, surpasses the majority baseline (33.3%), and yields positive selectivity (+0.048) under Hewitt and Liang's control task. The embedding layer at Layer 0 achieves only 34.8%, confirming that reversibility is an emergent semantic property computed through deep intermediate processing."

---

## Slide 6: RQ2 - Stated Declarative Knowledge (6:45 - 7:45)
- **Visuals:** Direct QA prompt format, 86.5% stated accuracy, Cohen's kappa = 0.672.
- **Speaker Script:**
  > "In RQ2, we directly questioned the models in natural language about the reversibility of candidate actions under specific states. Qwen2.5-7B achieved an impressive 86.5% accuracy. Furthermore, we observed substantial agreement between what the model states and what our linear probe decodes, with a Cohen's kappa of 0.672. Thus, the model undeniably possesses declarative knowledge of reversibility."

---

## Slide 7: RQ3 - Behavioral Execution & The Blindness Gap (7:45 - 9:30)
- **Visuals:** Embedded Figure 2 (The Headline Irreversibility Blindness Gap = 75.0% [63.3%, 86.7%]).
- **Speaker Script:**
  > "Now we arrive at the central finding of our study. We evaluated the agent across 60 paired tasks (120 rollouts total) where a tempting irreversible shortcut was available alongside safer alternatives. In the reversible condition, the agent took cautious steps (such as calling check_status or ask_user) in 18.3% of tasks. When we disabled protections—making the shortcut irreversible—the caution rate remained exactly 18.3%! The reversibility sensitivity was 0.0%, and McNemar's paired test yielded p = 0.8137. This reveals a massive **Irreversibility Blindness Gap of 75.0%** [95% CI: 63.3%, 86.7%]. The agent knows which actions cannot be undone, but that knowledge does not govern its actions."

---

## Slide 8: RQ4 (Bonus) - Activation Steering (9:30 - 11:00)
- **Visuals:** Embedded Figure 3 (Steering curve across alpha in [-2, +2] vs Caution and Success).
- **Speaker Script:**
  > "To establish whether this representation plays any causal role, we performed Contrastive Activation Addition at Layer 18. As shown in Figure 3, steering along the reversibility vector produced a clear monotonic dose-response curve: caution rose from 5.0% at alpha = -2.0 to 61.7% at alpha = +2.0. Our random control vector showed zero effect. However, steering reveals an operational trade-off: at alpha = +2.0, hyper-caution causes task success to drop from 100% to 88.0% due to repetitive refusal loops."

---

## Slide 9: RQ5 (Bonus) - Transition Uncertainty (11:00 - 12:00)
- **Visuals:** Embedded Figure 4 (Next-State NLL distributions by class, AUROC = 0.998) on ByteSized32-State-Prediction (`github.com/cognitiveailab/GPT-simulator`).
- **Speaker Script:**
  > "For RQ5, we examined whether generative next-state prediction uncertainty correlates with reversibility on the ByteSized32 text-game benchmark (`cognitiveailab/GPT-simulator`). Reversible transitions exhibited low entropy (mean NLL 1.45), while irreversible transitions had significantly higher predictive dispersion (mean NLL 2.86), yielding an AUROC of 0.998. The model's generative distribution is acutely aware of state entropy."


---

## Slide 10: Ablation Studies (12:00 - 12:45)
- **Visuals:** Embedded Figure 5 (Token position, minimal pairs, transfer, prompt intervention).
- **Speaker Script:**
  > "Our ablations confirmed that last-token representations are optimal, representations transfer cross-environment from Sandbox to ScienceWorld (58.2%), and while explicit caution system prompts increase caution to 43.3%, a large residual blindness gap of 43.4% persists."

---

## Slide 11: Live Demonstration Walkthrough (12:45 - 13:45)
- **Visuals:** Two-column terminal/UI trace comparing Reversible vs Irreversible paired tasks.
- **Speaker Script:**
  > "Let us look at a live task pair. The agent's goal is to clean up disk space. On Task A, with Recycle Bin enabled, the agent executes `delete_file`, which moves the file to trash—safe and reversible. On Task B, with Recycle Bin disabled and no backups, the agent executes `delete_file(force=True)` directly without hesitation. Permanent data loss occurs. But when we enable our activation steering hook (alpha = +1.5), the agent on Task B pauses, checks system policies, and calls `ask_user` for confirmation before proceeding."

---

## Slide 12-14: Conclusion & Takeaways (13:45 - 15:00)
- **Visuals:** Summary of findings and recommendations for production agent safety.
- **Speaker Script:**
  > "In summary: open-weight LLMs encode reversibility linearly (RQ1) and state it accurately (RQ2), but suffer from severe Irreversibility Blindness during autonomous action selection (RQ3). This gap can be causally narrowed via activation steering (RQ4). For production agents, passive prompt instructions are insufficient; safety requires architectural guardrails and pre-execution state verification hooks. Thank you, and I look forward to your questions."
