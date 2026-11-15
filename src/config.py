"""
Global Configuration and Constants for Reversibility in LLM Agents Project.
"""

from pathlib import Path
import os

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATASET_PATH = DATA_DIR / "reversibility_dataset.jsonl"
PAIRED_TASKS_PATH = DATA_DIR / "paired_tasks_60.json"
TOOLEMU_CASES_PATH = DATA_DIR / "toolemu_reversibility_cases.json"
SCIENCEWORLD_CASES_PATH = DATA_DIR / "scienceworld_reversibility_cases.json"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
LOGS_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = RESULTS_DIR / "figures"
PAPER_FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

for d in [DATA_DIR, RESULTS_DIR, METRICS_DIR, LOGS_DIR, FIGURES_DIR, PAPER_FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Random Seeds for reproducibility
MASTER_SEED = 42
SPLIT_SEEDS = [42, 1337, 2026]
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_ALPHA = 0.05  # 95% Confidence Interval

# Reversibility Class Definitions
LABEL_MAP = {
    0: "reversible",
    1: "partially_reversible",
    2: "irreversible"
}
LABEL_INV_MAP = {v: k for k, v in LABEL_MAP.items()}

# Model Configurations
SUPPORTED_MODELS = {
    "qwen2.5-7b": {
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "num_layers": 32,
        "hidden_dim": 3584,
        "family": "qwen"
    },
    "llama-3.1-8b": {
        "hf_name": "meta-llama/Llama-3.1-8B-Instruct",
        "num_layers": 32,
        "hidden_dim": 4096,
        "family": "llama"
    },
    "qwen2.5-14b": {
        "hf_name": "Qwen/Qwen2.5-14B-Instruct",
        "num_layers": 48,
        "hidden_dim": 5120,
        "family": "qwen"
    }
}

# Agent Evaluation Settings
MAX_AGENT_STEPS = 15
TEMPERATURE = 0.0  # Greedy decoding
NUM_PAIRED_TASKS = 60

# Steering Settings (RQ4)
STEERING_STRENGTHS = [-2.0, -1.0, 0.0, 1.0, 2.0]
TARGET_STEERING_LAYER = 16  # Middle-layer peak decodability

# Labeling Horizons
HORIZONS = [1, 2, 3]
DEFAULT_HORIZON = 3
