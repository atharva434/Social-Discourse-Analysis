# NLP Polarization Project 🚀

This repository contains the codebase for the NLP Polarization classification tasks (Subtasks 1, 2, and 3). It uses **`uv`** for strict dependency management and a modular architecture for reproducible experiments using **XLM-Roberta** and **BERT**.

## 📂 Project Structure

```text
nlp-project/
├── data/                       # [IGNORED BY GIT] Place your CSVs here
│   ├── subtask1/
│   │   ├── train/              # Place eng.csv, deu.csv, etc. here
│   │   └── dev/
│   ├── subtask2/               # Future placeholder
│   └── subtask3/               # Future placeholder
│
├── src/                        # Source Code
│   ├── common.py               # Shared utilities (metrics, saving)
│   ├── subtask1/               # Logic for Subtask 1
│   │   ├── data_setup.py       # Data loading & tokenization
│   │   ├── model_builder.py    # HF Model instantiation
│   │   └── engine.py           # Training loop (HF Trainer)
│
├── experiments.py              # 🧪 THE LAB NOTEBOOK (Config & Hyperparams)
├── train.py                    # 🎮 THE COMMANDER (Main entry point)
├── pyproject.toml              # Dependency definition
└── uv.lock                     # Exact version locking
```

---

## ⚡️ Setup & Installation

We use **`uv`** to manage dependencies. Do not use standard `pip` or `python` commands, or you will break the build.

### 1. Install `uv`
If you don't have it, install it once:
```bash
pip install uv
```

### 2. Sync Environment
Run this command in the project root. It will create a virtual environment (`.venv`) and install all dependencies (PyTorch, Transformers, WandB, etc.) locked to the correct versions.
```bash
uv sync
```

### 3. Setup Data
The `data/` folder is ignored by Git to keep the repo light. **You must manually create this structure:**

1.  Create a folder named `data` in the root.
2.  Inside `data`, create `subtask1`.
3.  Inside `subtask1`, create `train`.
4.  **Paste your CSV files** (`eng.csv`, `deu.csv`, etc.) into `data/subtask1/train/`.

**Visual check:**
```text
data/
└── subtask1/
    └── train/
        ├── eng.csv
        ├── deu.csv
        ├── ...
```

---

## 🏃‍♂️ Running Experiments

We use `uv run` to execute scripts. This ensures the virtual environment is automatically active.

### Run the Baseline
To run the standard XLM-Roberta baseline for Subtask 1:

```bash
uv run train.py --name st1_baseline
```

### Run All Experiments (Sequential)
To run every experiment listed in the registry (useful for overnight benchmarks):

```bash
uv run train.py --all
```

---

## 🧪 How to Create a New Experiment

Experiments are controlled by `experiments.py`. You do **not** need to modify the training code to try new hyperparameters.

1.  Open `experiments.py`.
2.  Add a new entry to the `experiment_registry` dictionary.
3.  Override the specific parameters you want to change (e.g., Learning Rate, Model, Epochs).

**Example:**
```python
"st1_high_lr": {
    "subtask": "subtask1",
    "model_name": "xlm-roberta-base",
    "learning_rate": 5e-5,  # <--- Changed parameter
    "dataset_path": "subtask1/train"
}
```

4.  Run it: 
```bash
uv run train.py --name st1_high_lr
```

---

## ❓ Troubleshooting

### "No GPU detected" / Running on CPU
* **Mac (M1/M2/M3):** The code supports MPS automatically.
* **Windows/Linux (NVIDIA):** If `uv` installed the CPU version of PyTorch by mistake, force a reinstall of the CUDA version:
    ```bash
    uv pip install torch --index-url