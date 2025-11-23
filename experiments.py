# experiments.py

# Global defaults
base_config = {
    "base_data_dir": "./data",  # Root data folder
    "output_dir": "./results",
    "use_wandb": True
}

experiment_registry = {
    # --- SUBTASK 1 EXPERIMENTS ---
    "st1_baseline": {
        "subtask": "subtask1",          # <--- Critical: Routes to src/subtask1
        "model_name": "xlm-roberta-base",
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "dataset_path": "subtask1/train" # Relative to base_data_dir
    },
    "st1_bert": {
        "subtask": "subtask1",
        "model_name": "bert-base-multilingual-cased",
        "epochs": 5,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "dataset_path": "subtask1/train"
    },

    # --- SUBTASK 2 EXPERIMENTS (Placeholders) ---
    "st2_baseline": {
        "subtask": "subtask2",
        "model_name": "xlm-roberta-base",
        "dataset_path": "subtask2/train"
        # Add subtask 2 specific params here later
    },
    
    # --- SUBTASK 3 EXPERIMENTS (Placeholders) ---
    "st3_baseline": {
        "subtask": "subtask3",
        "dataset_path": "subtask3/train"
    }
}