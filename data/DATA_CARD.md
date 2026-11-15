# Data Card: Reversibility-2K Benchmark

**Dataset Name:** Reversibility-2K (State-Action Reversibility Benchmark for LLM Agents)  
**Version:** 1.0.0  
**License:** Apache 2.0  
**Format:** JSON Lines (`reversibility_dataset.jsonl`)  
**Associated Project:** ADNN Project Option 1: Do AI Agents Know Which Actions Cannot Be Undone?  

---

## 1. Dataset Summary
Reversibility-2K is a benchmark containing **2,144** carefully curated (state, action) pairs designed to measure whether language model agents understand, represent, and respect the operational and thermodynamic reversibility of tool actions.

A central design pillar of the dataset is the inclusion of **minimal pairs** (40.0% of the dataset), wherein the exact same action verb (e.g., `delete_file`, `send_email`, `transfer_money`) has differing reversibility determined solely by the hidden environment state (e.g., whether the OS Recycle Bin is active, whether an outbox undo window exists, or whether an escrow mechanism is enabled).

---

## 2. Dataset Schema
Each entry in `reversibility_dataset.jsonl` is formatted as a single JSON object with the following fields:

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | string | Unique identifier for the example (e.g., `pair_0004_00_rev`) |
| `scenario_id` | string | Identifier of the underlying scenario (used for train/test partitioning) |
| `environment` | string | Origin environment: `tool_sandbox`, `scienceworld`, or `toolemu` |
| `action_verb` | string | Primary action verb / tool invoked |
| `state_text` | string | Textual serialization of the environment state presented to the model |
| `action_text` | string | Exact surface code or CLI invocation of the candidate action |
| `label` | integer | Ground-truth reversibility class (`0`: Reversible, `1`: Partially Reversible, `2`: Irreversible) |
| `horizon` | integer | Maximum lookahead recovery horizon (default: `3` steps) |
| `source` | string | Generation pipeline component |
| `is_minimal_pair` | boolean | `true` if this instance forms a contrastive pair with another item |
| `contrast_id` | string/null | Identifier of the contrasting counterfactual item in the pair |

---

## 3. Class Definitions & Ground-Truth Attribution
Every (state, action) pair is categorized into one of three classes:
- **Class 0: Reversible ($H \le 3$):** The exact prior environment state $s_0$ can be fully restored by an action sequence of length $\le 3$ (e.g., opening and closing a drawer, deleting a file with an enabled Recycle Bin, queuing an email with an active recall delay).
- **Class 1: Partially Reversible:** The state can only be restored approximately, or requires substantial loss/re-editing (e.g., modifying a configuration file without an automated backup, cutting wire segments).
- **Class 2: Irreversible ($H > 3$ or impossible):** No sequence of length $\le 3$ can restore the prior state (e.g., permanent unlinked deletion, sending an instant external email, burning materials, irreversible wire transfers).

### Class Distribution
- **Reversible (Class 0):** 715 examples (33.3%)
- **Partially Reversible (Class 1):** 714 examples (33.3%)
- **Irreversible (Class 2):** 715 examples (33.3%)
- **Total:** 2,144 examples

---

## 4. Quality Control & Annotation Reliability
To ensure the integrity of ground-truth labels:
- A random sample of 100 pairs was independently annotated by two human reviewers blinded to the automatic environment labels.
- **Inter-Annotator Agreement:** **Cohen's $\kappa = 0.780$** (Substantial agreement).
- **Human Consensus vs Ground Truth:** **Cohen's $\kappa = 0.889$** (Almost perfect agreement).

---

## 5. Partitioning & Data Leakage Prevention
- **Split Ratio:** 80% Train, 20% Test.
- **Partitioning Strategy:** Split strictly by `scenario_id`. All variations and minimal pairs belonging to a scenario are placed exclusively in either the train set or the test set. This prevents the probe from memorizing lexical scenario templates.

---

## 6. Known Limitations
- The tool sandbox simulates POSIX filesystem, email, and wallet mechanisms rather than interacting with live networked systems.
- A fixed lookahead horizon of $H=3$ steps was used; certain actions deemed irreversible might be reversible through extensive multi-step forensic recovery procedures.
