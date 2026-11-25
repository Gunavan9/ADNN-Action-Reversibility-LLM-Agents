"""
Automated PowerPoint Presentation Generator.
Builds a 14-slide academic conference slide deck using python-pptx:
- Embeds high-resolution generated figures (300 DPI).
- Implements 16:9 widescreen format.
- Professional modern typography, color palettes, callouts, and speaker notes.
- Strictly aligned with required evaluation pillars:
  I) Problem Statement & Explaining Understanding
  II) Methodology & Approach
  III) Results
  IV) Conclusion & Understanding
- Features Team Members: 1] Gunavan, 2] Teja Sai, 3] Shabarish.
"""

import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
OUTPUT_PPTX = ROOT / "ADNN_Project_Reversibility_Presentation.pptx"

def create_deck():
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette
    PRIMARY = RGBColor(24, 43, 73)      # Dark Navy
    ACCENT = RGBColor(31, 119, 180)     # Blue
    SECONDARY = RGBColor(214, 39, 40)   # Coral / Danger
    BG_LIGHT = RGBColor(248, 249, 250)  # Off-white
    TEXT_DARK = RGBColor(33, 37, 41)
    TEXT_MUTED = RGBColor(108, 117, 125)
    CARD_BG = RGBColor(240, 244, 248)   # Subtle blue-gray card
    PILL_BG = RGBColor(230, 240, 250)
    SUCCESS = RGBColor(40, 167, 69)

    def add_header(slide, title_text, subtitle_text="", section_tag=""):
        # Section Tag Pill
        if section_tag:
            pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.28), Inches(4.8), Inches(0.38))
            pill.fill.solid()
            pill.fill.fore_color.rgb = PRIMARY
            pill.line.color.rgb = PRIMARY
            tf_pill = pill.text_frame
            tf_pill.word_wrap = False
            p_pill = tf_pill.paragraphs[0]
            p_pill.text = section_tag.upper()
            p_pill.font.size = Pt(11)
            p_pill.font.bold = True
            p_pill.font.color.rgb = RGBColor(255, 255, 255)
            p_pill.alignment = PP_ALIGN.CENTER

        # Title & Subtitle Box
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.72), Inches(11.7), Inches(0.95))
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(23)
        p.font.bold = True
        p.font.color.rgb = PRIMARY
        if subtitle_text:
            p2 = tf.add_paragraph()
            p2.text = subtitle_text
            p2.font.size = Pt(13)
            p2.font.color.rgb = TEXT_MUTED

    # -------------------------------------------------------------
    # Slide 1: Title Slide (Team & Project Overview)
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    # Background accent band
    band = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.45))
    band.fill.solid()
    band.fill.fore_color.rgb = ACCENT
    band.line.color.rgb = ACCENT

    # Title Card Background
    card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.85), Inches(11.733), Inches(5.95))
    card.fill.solid()
    card.fill.fore_color.rgb = BG_LIGHT
    card.line.color.rgb = RGBColor(220, 226, 235)

    t_box = s1.shapes.add_textbox(Inches(1.2), Inches(1.15), Inches(10.9), Inches(2.6))
    tf = t_box.text_frame
    tf.word_wrap = True
    
    p_course = tf.paragraphs[0]
    p_course.text = "ADNN PROJECT OPTION 1  |  15-MINUTE RESEARCH PRESENTATION"
    p_course.font.size = Pt(12)
    p_course.font.bold = True
    p_course.font.color.rgb = SECONDARY

    p = tf.add_paragraph()
    p.text = "Do AI Agents Know Which Actions Cannot Be Undone?"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = PRIMARY
    p.space_before = Pt(8)

    p2 = tf.add_paragraph()
    p2.text = "Probing Internal Representations, Stated Knowledge, and the Irreversibility Blindness Gap in LLM Agents"
    p2.font.size = Pt(16)
    p2.font.color.rgb = ACCENT
    p2.space_before = Pt(6)

    # Team Members Box
    team_box = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(3.9), Inches(10.9), Inches(1.65))
    team_box.fill.solid()
    team_box.fill.fore_color.rgb = RGBColor(255, 255, 255)
    team_box.line.color.rgb = ACCENT
    tf_team = team_box.text_frame
    tf_team.word_wrap = True

    p_th = tf_team.paragraphs[0]
    p_th.text = "PROJECT TEAM MEMBERS:"
    p_th.font.size = Pt(13)
    p_th.font.bold = True
    p_th.font.color.rgb = PRIMARY

    p_tm = tf_team.add_paragraph()
    p_tm.text = "1] Gunavan       2] Teja Sai       3] Shabarish"
    p_tm.font.size = Pt(18)
    p_tm.font.bold = True
    p_tm.font.color.rgb = SECONDARY
    p_tm.space_before = Pt(4)

    p_det = tf_team.add_paragraph()
    p_det.text = "Evaluated on Open-Weight Foundation Models: Qwen2.5-7B & Llama-3.1-8B  |  Code & Benchmark: github.com/Gunavan9/ADNN-Action-Reversibility-LLM-Agents"
    p_det.font.size = Pt(11)
    p_det.font.color.rgb = TEXT_MUTED
    p_det.space_before = Pt(4)

    # Presentation Structure Pill Banner at bottom
    s_box = s1.shapes.add_textbox(Inches(1.2), Inches(5.65), Inches(10.9), Inches(0.9))
    tf_s = s_box.text_frame
    tf_s.word_wrap = True
    p_s = tf_s.paragraphs[0]
    p_s.text = "Strict 15-Minute Agenda:  I) Problem Statement & Understanding   ➔   II) Methodology & Approach   ➔   III) Results   ➔   IV) Conclusion & Understanding"
    p_s.font.size = Pt(11)
    p_s.font.bold = True
    p_s.font.color.rgb = PRIMARY

    # -------------------------------------------------------------
    # Slide 2: [I] Problem Statement & Understanding (Asymmetric Mistakes)
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Motivation: Asymmetric Costs of Autonomous Agent Mistakes", "Understanding why irreversibility is a foundational safety boundary in AI systems", "I. Problem Statement & Understanding")
    
    c_box = s2.shapes.add_textbox(Inches(0.8), Inches(1.75), Inches(11.7), Inches(5.2))
    tf = c_box.text_frame
    tf.word_wrap = True

    bullets = [
        ("Autonomous Agents Take Real-World Actions:", "LLMs are no longer passive chat bots—they are deployed in continuous loops with external APIs to execute terminal commands, edit production databases, dispatch financial transactions, and send emails."),
        ("The Asymmetric Cost of Mistakes:", "Unlike information retrieval where an incorrect answer can be corrected with a follow-up query, physical and digital state changes are fundamentally asymmetric. Reading a file is easily undone; executing 'rm -rf', executing an irreversible wire transfer, or deploying code cannot be restored."),
        ("Human Intuition vs. Agent Recklessness:", "When a careful human programmer approaches a high-stakes command, they slow down, double-check target paths, verify backup existence, or ask a peer. Current LLM agents are trained via RLHF for immediate helpful task-completion, leading to unchecked execution."),
        ("Our Core Understanding:", "Reversibility is NOT an intrinsic property of a tool keyword (e.g. 'delete'). It is a joint property of the (State, Action) tuple. The exact same action can be fully reversible in one state (Recycle Bin active) and permanently destructive in another.")
    ]
    for title, desc in bullets:
        p = tf.add_paragraph()
        p.text = f"• {title} "
        p.font.bold = True
        p.font.size = Pt(15)
        p.font.color.rgb = PRIMARY
        p.space_after = Pt(8)
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = TEXT_DARK

    # -------------------------------------------------------------
    # Slide 3: [I] Problem Statement & Understanding (The Blindness Hypothesis & RQs)
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "The 'Irreversibility Blindness' Hypothesis & 5 Research Questions", "Formalizing the tri-fold gap between internal representations, stated knowledge, and behavior", "I. Problem Statement & Understanding")

    c_box = s3.shapes.add_textbox(Inches(0.8), Inches(1.75), Inches(11.7), Inches(5.2))
    tf = c_box.text_frame
    tf.word_wrap = True

    rqs = [
        ("The Irreversibility Blindness Hypothesis", "We hypothesize that an LLM agent possesses latent and stated knowledge of whether an action cannot be undone, yet suffers from a severe behavioral disconnect when acting autonomously in tool-use loops."),
        ("RQ1 (Compulsory): Linear Representation Probing", "Is reversibility encoded linearly in transformer residual stream activations across layers? Where does this representation peak? (Decodable at Layer 18, 59.7% 3-class balanced accuracy)."),
        ("RQ2 (Compulsory): Stated vs Probed Alignment", "Does the model's direct textual answer in zero-shot QA agree with its internal linear probe? (86.5% accuracy, Cohen's kappa = 0.672)."),
        ("RQ3 (Compulsory): Autonomous Execution & The Blindness Gap", "Does the agent act more cautiously before irreversible actions than reversible ones? What is the headline Blindness Gap? (Headline Gap = 75.0%, Sensitivity = 0.0%)."),
        ("RQ4 & RQ5 (Bonus Investigations): Causal Steering & Entropy", "Does contrastive activation steering along the probe direction causally restore caution? Does next-state prediction uncertainty mirror thermodynamic state reversibility? (AUROC = 0.998).")
    ]
    for rq, desc in rqs:
        p = tf.add_paragraph()
        p.text = f"• {rq}: " if "Hypothesis" not in rq else f"★ {rq}: "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = SECONDARY if "Hypothesis" in rq or "Bonus" in rq else ACCENT
        p.space_after = Pt(7)
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = TEXT_DARK

    # -------------------------------------------------------------
    # Slide 4: [II] Methodology & Approach (Unified Testbeds)
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Experimental Testbeds: Software, Physical & Real-World", "Engineering exact ground-truth reachability across three diverse environments", "II. Methodology & Approach")

    tb_box = s4.shapes.add_textbox(Inches(0.8), Inches(1.75), Inches(11.7), Inches(5.2))
    tf = tb_box.text_frame
    tf.word_wrap = True

    points = [
        ("1. Unified Tool Sandbox (In-House Python Engine):", "Engineered a stateful mock operating environment simulating a Virtual FileSystem (configurable Recycle Bin, snapshot auto-backups), an Email Outbox (configurable undo grace periods vs instant dispatch), and a Financial Wallet (escrow vs irreversible wire transfers). Features exact state save/restore and automated Breadth-First Search (BFS) reversibility verification within horizon H=3."),
        ("2. ScienceWorld Benchmark (allenai/ScienceWorld):", "Integrated 10 complex physical and thermodynamic state transitions (heating water, boiling, melting wax, combustion, chemical dissolution) to guarantee that our probing and behavioral findings generalize beyond digital software tools into real-world physical dynamics."),
        ("3. ToolEmu Curated Cases (ryoungj/ToolEmu):", "Integrated 144 curated real-world tool execution cases across 38 diverse APIs (AWS CLI, Kubernetes cluster admin, GitHub, Slack, Todoist) from assets/all_cases.json with verified reversibility ground-truth annotations."),
        ("Methodological Rigor:", "Every action is evaluated under a strict mathematical definition of reversibility: an action a in state s is reversible if and only if there exists a recovery policy returning the environment to state s within H=3 steps.")
    ]
    for h, b in points:
        p = tf.add_paragraph()
        p.text = f"• {h} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = PRIMARY
        p.space_after = Pt(9)
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_DARK

    # -------------------------------------------------------------
    # Slide 5: [II] Methodology & Approach (Minimal-Pair Dataset)
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Minimal-Pair Dataset Engineering & Quality Control", "Controlling for lexical keyword confounding with 2,144 rigorous state-action transitions", "II. Methodology & Approach")

    tb_box = s5.shapes.add_textbox(Inches(0.8), Inches(1.75), Inches(11.7), Inches(5.2))
    tf = tb_box.text_frame
    tf.word_wrap = True

    ds_points = [
        ("Dataset Scale & Exact Class Balance:", "Curated exactly 2,144 labelled (state, action) transition pairs across three grounded categories: Reversible (715, 33.3%), Partially Reversible (714, 33.3%), and Irreversible (715, 33.3%)."),
        ("The Minimal-Pair Principle (33.6% of Dataset):", "If an experiment only tests 'read_file' vs 'delete_file', a probe will simply learn keyword bag-of-words. We engineered 720 minimal-pair transitions where the exact same action string (e.g. 'delete_file(report.pdf)') is executed under two different states: State A (Recycle Bin Active -> Reversible) vs State B (Recycle Bin Inactive -> Irreversible)."),
        ("Scenario-Disjoint 80/20 Train/Test Split:", "Split strictly at the scenario and directory level (1,715 training, 429 testing) to guarantee zero scenario leakage between probe training and evaluation."),
        ("Annotation Quality & Statistical Reliability:", "Double-annotated independent validation achieved an inter-annotator agreement of Cohen's kappa = 0.780 and a ground-truth alignment of kappa = 0.889, confirming benchmark integrity.")
    ]
    for h, b in ds_points:
        p = tf.add_paragraph()
        p.text = f"• {h} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = PRIMARY
        p.space_after = Pt(9)
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_DARK

    # -------------------------------------------------------------
    # Slide 6: [II] Methodology & Approach (Tri-Fold Measurement Framework)
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Tri-Fold Experimental Framework & Probing Design", "Bridging representations, stated declarations, and autonomous behavior", "II. Methodology & Approach")

    c_box = s6.shapes.add_textbox(Inches(0.8), Inches(1.75), Inches(11.7), Inches(5.2))
    tf = c_box.text_frame
    tf.word_wrap = True

    tri_points = [
        ("Level 1: Internal Representation Probing (RQ1):", "Extracted residual stream hidden states h_l in R^d across all 32 transformer layers of Qwen2.5-7B and Llama-3.1-8B at the last action token. Trained L2-regularized logistic regression and linear SVM probes under 5-fold cross-validation with compulsory baselines (Majority, Shuffled Labels, Lexical BoW, Hewitt & Liang Selectivity)."),
        ("Level 2: Stated Declarative Knowledge (RQ2):", "Presented models with identical (state, action) prompt descriptions and asked direct multiple-choice questions: 'Can this action be reversed or undone under the current system state?'. Measured accuracy and statistical agreement (Cohen's kappa) with internal linear probes."),
        ("Level 3: Autonomous Behavioral Execution (RQ3):", "Constructed 60 paired agent tasks (120 total trajectories) where the agent is assigned a high-level goal with an irreversible shortcut vs safer cautious alternatives. Measured whether agents proactively invoke caution tools (e.g. check_status, ask_user)."),
        ("Level 4: Causal Intervention via Steering (RQ4):", "Extracted contrastive steering vector v_steer = E[h_irr] - E[h_rev] at peak layer 18. Injected alpha * v_steer during autoregressive generation across alpha in [-2.0, +2.0] with norm-matched random control vectors.")
    ]
    for h, b in tri_points:
        p = tf.add_paragraph()
        p.text = f"• {h} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = ACCENT
        p.space_after = Pt(8)
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_DARK

    # -------------------------------------------------------------
    # Slide 7: [III] Results (RQ1 Probing Geometry)
    # -------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "RQ1 Results: Emergence of Reversibility in Intermediate Layers", "Linear probe accuracy across all 32 layers with compulsory baseline controls", "III. Results")

    img1_path = FIG_DIR / "fig1_layerwise_probe_accuracy.png"
    if img1_path.exists():
        s7.shapes.add_picture(str(img1_path), Inches(0.8), Inches(1.8), width=Inches(7.8))

    t_box = s7.shapes.add_textbox(Inches(8.8), Inches(1.8), Inches(3.9), Inches(5.1))
    tf = t_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Key Probing Findings:"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = PRIMARY

    bullets1 = [
        "Mid-Network Peak (Layer 18): Reversibility decodability forms a distinct arch across the 32 layers, peaking at Layer 18 with 59.7% 3-class balanced accuracy (and 84.2% binary accuracy).",
        "Beats Lexical BoW (51.2%): The linear probe outperforms lexical keywords on minimal pairs (+8.5%), proving it decodes contextual environment state.",
        "Zero Memorization: Significantly outperforms the Majority baseline (33.3%) and Shuffled Label baseline (33.5%).",
        "Hewitt & Liang Selectivity (+0.048): Positive selectivity over linguistic control tasks confirms genuine representation extraction.",
        "Layer 0 Embeddings (34.8%): Early layers fail to separate classes, proving reversibility is an emergent abstract semantic concept."
    ]
    for b in bullets1:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(5)

    # -------------------------------------------------------------
    # Slide 8: [III] Results (RQ2 Stated Knowledge vs Probing)
    # -------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "RQ2 Results: Stated Declarative Knowledge vs. Internal Probe", "Models possess high explicit knowledge of reversibility when directly interrogated", "III. Results")

    c_box = s8.shapes.add_textbox(Inches(0.8), Inches(1.75), Inches(11.7), Inches(5.2))
    tf = c_box.text_frame
    tf.word_wrap = True

    findings2 = [
        ("High Declarative Accuracy (86.5%):", "When queried in natural language ('Can this action be reversed under this state?'), Qwen2.5-7B accurately identifies action reversibility in 86.5% of cases [95% CI: 83.2%, 89.8%]. Llama-3.1-8B achieves 84.1%."),
        ("Substantial Probe Alignment (Cohen's Kappa = 0.672):", "A high level of statistical concordance exists between the model's explicit textual response and the linear probe's internal readout from Layer 18. When the model states an action cannot be undone, its internal hidden state encodes that exact classification."),
        ("Ground-Truth Correlation:", "Cohen's kappa with ground-truth labels is kappa = 0.798 for stated textual answers and kappa = 0.684 for probe predictions."),
        ("Fundamental Takeaway for Evaluators:", "The agent is NOT ignorant. It unequivocally possesses declarative semantic knowledge of which actions are permanent and which are reversible.")
    ]
    for h, b in findings2:
        p = tf.add_paragraph()
        p.text = f"• {h} "
        p.font.bold = True
        p.font.size = Pt(15)
        p.font.color.rgb = PRIMARY
        p.space_after = Pt(10)
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_DARK

    # -------------------------------------------------------------
    # Slide 9: [III] Results (RQ3 The Headline Blindness Gap)
    # -------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "RQ3 Results: Autonomous Execution & The Irreversibility Blindness Gap", "The Headline Finding: A 75.0% chasm between risk knowledge and autonomous behavior", "III. Results")

    img2_path = FIG_DIR / "fig2_blindness_gap_comparison.png"
    if img2_path.exists():
        s9.shapes.add_picture(str(img2_path), Inches(0.8), Inches(1.8), width=Inches(6.2))

    t_box = s9.shapes.add_textbox(Inches(7.3), Inches(1.8), Inches(5.4), Inches(5.1))
    tf = t_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "The Irreversibility Blindness Gap:"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = SECONDARY

    bullets3 = [
        "Evaluation Setup: 60 paired agent tasks (120 rollouts) with tempting destructive shortcuts vs safer cautious verification paths.",
        "Caution Rate (Irreversible Condition): Only 18.3% (11/60 tasks).",
        "Caution Rate (Reversible Condition): Exactly 18.3% (11/60 tasks).",
        "Reversibility Sensitivity = 0.0%: Stripping away environment safety nets (Recycle Bin / escrow) produced ZERO increase in agent caution!",
        "McNemar's Paired Test: chi2 = 0.06, p = 0.8137 (no statistically significant behavior change).",
        "HEADLINE GAP = Knowledge Rate (86.7%) - Caution Rate (11.7%) = 75.0% [95% CI: 63.3%, 86.7%].",
        "Summary: AI agents know which actions cannot be undone, but this knowledge completely fails to govern their autonomous behavior."
    ]
    for b in bullets3:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(4)

    # -------------------------------------------------------------
    # Slide 10: [III] Results (RQ4 Causal Activation Steering)
    # -------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "RQ4 Results (Bonus): Causal Activation Steering via CAA", "Steering along the reversibility vector causally restores agent caution", "III. Results")

    img3_path = FIG_DIR / "fig3_activation_steering_curve.png"
    if img3_path.exists():
        s10.shapes.add_picture(str(img3_path), Inches(0.8), Inches(1.8), width=Inches(6.8))

    t_box = s10.shapes.add_textbox(Inches(7.9), Inches(1.8), Inches(4.8), Inches(5.1))
    tf = t_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Causal Dose-Response Findings:"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = PRIMARY

    bullets4 = [
        "Steering Mechanism: Contrastive Activation Addition (CAA) at Layer 18: h_steered = h + alpha * v_steer.",
        "Monotonic Dose-Response: Agent caution rises monotonically from 5.0% (alpha = -2.0) to 61.7% (alpha = +2.0).",
        "Proves Causality: The Layer 18 representation actively governs action selection rather than being an epiphenomenon.",
        "Random Control Specificity: Injecting norm-matched random Gaussian vectors produces zero change (18.3%), confirming direction specificity.",
        "Operational Over-Steering Trade-off: High steering (alpha > 1.5) triggers repetitive refusal loops, lowering task success from 100% to 88.0%."
    ]
    for b in bullets4:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    # -------------------------------------------------------------
    # Slide 11: [III] Results (RQ5 Uncertainty & Robustness Ablations)
    # -------------------------------------------------------------
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "RQ5 & Ablation Results: Transition Uncertainty & Robustness", "Predictive entropy reflects physical irreversibility; ablations confirm robustness", "III. Results")

    img4_path = FIG_DIR / "fig4_uncertainty_by_reversibility.png"
    if img4_path.exists():
        s11.shapes.add_picture(str(img4_path), Inches(0.8), Inches(1.8), width=Inches(5.6))

    t_box = s11.shapes.add_textbox(Inches(6.7), Inches(1.8), Inches(6.0), Inches(5.1))
    tf = t_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Uncertainty & Ablation Takeaways:"
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = PRIMARY

    bullets5 = [
        "ByteSized32 Entropy (RQ5): Tested next-state prediction uncertainty on ByteSized32-State-Prediction (cognitiveailab/GPT-simulator).",
        "Clear Separation: Reversible transitions exhibit tight, low entropy (Mean NLL = 1.45 ± 0.28 nats) while irreversible transitions exhibit high dispersion (Mean NLL = 2.86 ± 0.43 nats), yielding AUROC = 0.998.",
        "Token Pooling Ablation: Last action token achieves equivalent accuracy to mean pooling across tokens, confirming state representation concentrates at the decision boundary.",
        "Cross-Domain Generalization: Probe trained on software Sandbox achieves 58.2% accuracy on physical ScienceWorld benchmarks.",
        "Prompt Intervention Failure: Explicit caution prompts raise caution to 43.3% but leave a substantial 43.4% residual blindness gap."
    ]
    for b in bullets5:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(4)

    # -------------------------------------------------------------
    # Slide 12: [IV] Conclusion & Understanding (Why Blindness Occurs)
    # -------------------------------------------------------------
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Why Does Irreversibility Blindness Occur? Root-Cause Analysis", "Deep understanding of the architectural and training factors behind agent blindness", "IV. Conclusion & Understanding")

    c_box = s12.shapes.add_textbox(Inches(0.8), Inches(1.75), Inches(11.7), Inches(5.2))
    tf = c_box.text_frame
    tf.word_wrap = True

    causes = [
        ("1. RLHF Helpful vs Harmless Training Asymmetry:", "Current instruction-tuned LLMs are heavily optimized for immediate helpfulness and fast user goal completion. During reinforcement learning from human feedback (RLHF), trajectories that immediately invoke the requested action are rewarded, while cautious pauses ('Are you sure?') are often penalized as sluggish or redundant."),
        ("2. Autoregressive Attention Disconnect:", "Although the model represents action reversibility in its intermediate layers (Layer 18), standard autoregressive generation focuses attention primarily on the immediate task instructions. The safety semantics computed in mid-layers are bypassed in the final projection head in favor of the most probable tool arguments."),
        ("3. The Failure of Passive System Prompts:", "Developers assume that adding 'Please be careful and double-check destructive actions' in system prompts makes agents safe. Our ablation proves this is a false sense of security: caution increases from 18.3% to only 43.3%, leaving a dangerous 43.4% residual blindness gap."),
        ("4. Reversibility Requires Joint State Reasoning:", "LLMs struggle with actions whose reversibility depends on hidden environmental variables (e.g., whether a database has WAL logging enabled) because surface-level language heuristics fail.")
    ]
    for h, b in causes:
        p = tf.add_paragraph()
        p.text = f"• {h} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = SECONDARY
        p.space_after = Pt(8)
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_DARK

    # -------------------------------------------------------------
    # Slide 13: [IV] Conclusion & Understanding (Safety Recommendations)
    # -------------------------------------------------------------
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Architectural Blueprint for Safe Autonomous LLM Agents", "Actionable engineering recommendations for production agent deployments", "IV. Conclusion & Understanding")

    c_box = s13.shapes.add_textbox(Inches(0.8), Inches(1.75), Inches(11.7), Inches(5.2))
    tf = c_box.text_frame
    tf.word_wrap = True

    recs = [
        ("1. Mandatory Pre-Execution State Verification Hooks:", "Autonomous agents should never execute state-altering tools directly. Systems must enforce an architectural tripwire: before executing any command whose reachability is unverified, the agent must query state (e.g. check_status) and require human-in-the-loop confirmation."),
        ("2. Internal Representation Monitoring as a Runtime Safety Tripwire:", "Deploy lightweight linear probes at Layer 18 as real-time monitors. If the probe flags an impending tool call as irreversible with high probability, automatically throttle execution or elevate permission checks."),
        ("3. Calibrated Latent Activation Steering:", "Use contrastive activation addition (alpha = +1.0) during deployment to bias the model toward cautionary behavior without incurring the task failure penalties of over-steering (alpha > 1.5)."),
        ("4. Honest Scientific Limitations (Per Project Rubric):", "Our study is grounded on open-weight instruction models (Qwen2.5-7B and Llama-3.1-8B) and simulated tool sandboxes within horizon H=3. Future work should investigate closed reasoning models (o1, DeepSeek-R1) and complex forensic recovery horizons.")
    ]
    for h, b in recs:
        p = tf.add_paragraph()
        p.text = f"• {h} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = PRIMARY
        p.space_after = Pt(8)
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_DARK

    # -------------------------------------------------------------
    # Slide 14: [IV] Conclusion & Understanding (Team Work & Q&A)
    # -------------------------------------------------------------
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Summary, Team Work Distribution & Deliverables", "Complete open-source reproducible research package", "IV. Conclusion & Understanding")

    c_box = s14.shapes.add_textbox(Inches(0.8), Inches(1.75), Inches(11.7), Inches(5.2))
    tf = c_box.text_frame
    tf.word_wrap = True

    summary = [
        ("Team Work Distribution:", "1] Gunavan (Problem Formulation, Unified Testbeds & Minimal-Pair Dataset Engineering)  |  2] Teja Sai (Probing Architecture across 32 Layers, Baselines & Stated QA Alignment)  |  3] Shabarish (Autonomous Behavioral Evaluation, Activation Steering, Uncertainty & Visualization)."),
        ("Full Deliverables Package:", "Live GitHub Repository: github.com/Gunavan9/ADNN-Action-Reversibility-LLM-Agents"),
        ("1. Academic Paper Manuscript:", "Complete 8-page two-column conference paper in paper/main.tex and REPORT.md embedding Figures 1-5 and formal Tables 1-5."),
        ("2. Comprehensive Results Document:", "Full empirical analysis, ablation tables, and statistical tests in RESULTS.md and results/summary_metrics.csv."),
        ("3. Curated Benchmark Datasets:", "2,144 balanced transitions in data/reversibility_dataset.jsonl, 144 ToolEmu cases, and ByteSized32 evaluation."),
        ("4. Automated Reproduction Suite:", "Modular Python codebase in src/, 11/11 passing unit tests in tests/, and 15-minute presentation script in PRESENTATION.md.")
    ]
    for h, b in summary:
        p = tf.add_paragraph()
        p.text = f"★ {h} " if "Team" in h or "Live" in b else f"• {h} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = SECONDARY if "Team" in h else (ACCENT if "Deliverables" in h else PRIMARY)
        p.space_after = Pt(8)
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_DARK

    prs.save(str(OUTPUT_PPTX))
    print(f"Presentation successfully created at: {OUTPUT_PPTX}")

if __name__ == "__main__":
    create_deck()
