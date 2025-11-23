import os
import torch
from sklearn.metrics import f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    macro_f1 = f1_score(labels, preds, average='macro')
    return {'macro_f1': macro_f1}

def save_model(model, tokenizer, output_dir):
    """Saves model and tokenizer to disk"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[INFO] Model saved to {output_dir}")

def analyze_per_language(trainer, val_dataset, val_df_slice):
    """Runs prediction and prints F1 score per language"""
    print("[INFO] Running per-language analysis...")
    preds = trainer.predict(val_dataset).predictions.argmax(axis=1)
    
    analysis_df = val_df_slice.copy()
    analysis_df['prediction'] = preds
    
    # Calculate F1 per language
    results = analysis_df.groupby('language').apply(
        lambda x: f1_score(x['polarization'], x['prediction'], average='macro'),
        include_groups=False
    ).sort_values()
    
    return results