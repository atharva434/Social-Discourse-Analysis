# src/subtask1/data_setup.py
import pandas as pd
import glob
import os
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

class PolarizationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def load_and_prepare(data_root, dataset_relative_path, model_name, config):
    """
    Args:
        data_root: "./data"
        dataset_relative_path: "subtask1/train"
    """
    # Construct full path: ./data/subtask1/train
    target_dir = os.path.join(data_root, dataset_relative_path)
    
    print(f"[INFO] Loading Subtask 1 data from: {target_dir}")
    
    # 1. Load CSVs
    file_paths = glob.glob(os.path.join(target_dir, "*.csv"))
    if not file_paths:
        raise FileNotFoundError(f"No CSVs found in {target_dir}")

    all_dfs = []
    for fp in file_paths:
        try:
            df = pd.read_csv(fp, header=0, names=['text', 'polarization']) # Ensure columns match
            df['language'] = os.path.splitext(os.path.basename(fp))[0]
            all_dfs.append(df)
        except Exception as e:
            print(f"Skipping {fp}: {e}")

    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # 2. Split & Tokenize
    X = full_df['text'].astype(str)
    y = full_df['polarization'].astype(int)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_enc = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=128)
    val_enc = tokenizer(X_val.tolist(), truncation=True, padding=True, max_length=128)

    return (
        PolarizationDataset(train_enc, y_train),
        PolarizationDataset(val_enc, y_val),
        tokenizer,
        full_df.loc[X_val.index] # Return validation slice for analysis
    )