# train.py
import argparse
import os
import sys

# Add src to path so imports work cleanly
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from experiments import base_config, experiment_registry
import utils  # Your shared utils (metrics, saving)

def run_experiment(run_name):
    # 1. Config Setup
    if run_name not in experiment_registry:
        raise ValueError(f"Experiment {run_name} not found.")
    
    overrides = experiment_registry[run_name]
    config = {**base_config, **overrides}
    config["run_name"] = run_name
    
    subtask = config.get("subtask")
    print(f"--- Running {run_name} for {subtask.upper()} ---")

    # 2. Dynamic Routing based on Subtask
    if subtask == "subtask1":
        import subtask1.data_setup as data_mod
        import subtask1.model_builder as model_mod
        import subtask1.engine as engine_mod
    elif subtask == "subtask2":
        # import subtask2.data_setup as data_mod
        raise NotImplementedError("Subtask 2 not implemented yet")
    elif subtask == "subtask3":
        # import subtask3.data_setup as data_mod
        raise NotImplementedError("Subtask 3 not implemented yet")
    else:
        raise ValueError(f"Unknown subtask: {subtask}")

    # 3. Execution (Standardized Interface)
    # Note: We assume every subtask module has these specific function names
    
    train_data, val_data, tokenizer, val_df = data_mod.load_and_prepare(
        data_root=config["base_data_dir"],
        dataset_relative_path=config["dataset_path"],
        model_name=config["model_name"],
        config=config
    )

    model = model_mod.build_model(config["model_name"])
    
    trainer = engine_mod.train_model(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        output_dir=os.path.join(config["output_dir"], run_name),
        hyperparameters=config
    )

    # 4. Finalize
    utils.save_model(model, tokenizer, os.path.join(config["output_dir"], run_name))
    print(f"Experiment {run_name} Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Experiment name from experiments.py")
    args = parser.parse_args()
    
    run_experiment(args.name)