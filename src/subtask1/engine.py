from transformers import Trainer, TrainingArguments
from utils import compute_metrics

def train_model(model, train_dataset, val_dataset, output_dir, hyperparameters):
    """
    Sets up the HF Trainer and runs training.
    """
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=hyperparameters['epochs'],
        learning_rate=hyperparameters['learning_rate'],
        per_device_train_batch_size=hyperparameters['batch_size'],
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="wandb" if hyperparameters['use_wandb'] else "none",
        run_name=hyperparameters['run_name'],
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print("[INFO] Starting training...")
    trainer.train()
    return trainer