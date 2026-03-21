import optuna
import os
import torch
from datasets.GRURainSeqDataset import GRURainSeqDataset
from torch.utils.data import random_split
from models.gru_baseline import GRU
from utils import perfscores
import numpy as np

INPUT_DIR = "/store_new/mch/msrad/radar/radar_database_v2/rf_input_data/"
OUTPUT_DIR = "tuning_GRU"
STUDY_NAME = "tune_gru"
COLS_TO_USE = ["RADAR", "HEIGHT", "ISO0_HEIGHT", "ZH_mean", "ZV_mean", "KDP_mean", "RHOHV_mean"]
SUBSET = 0.3  # 30% of data for tuning
N_TRIALS = 30 # Number of parameter combinations to try
NUM_WORKERS = 4
MAX_EPOCHS = 25

def objective(trial):
    print(f"Running trial {trial.number}")
    # search space
    params = {
        "num_hidden_nodes": trial.suggest_int("num_hidden_nodes", 32, 256, step=32),
        "num_hidden_layers": trial.suggest_int("num_hidden_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "alpha_denseweight": trial.suggest_float("alpha_denseweight", 0.0, 1.0),
        "loss_function": trial.suggest_categorical("loss_function", ["mse", "mae", "huber"])
    }

    # Load Subset Dataset
    dataset = GRURainSeqDataset(input_dir=INPUT_DIR,
                                cols_to_use=COLS_TO_USE,
                                subset=SUBSET,
                                weighting="denseweight",
                                alpha_denseweight=params["alpha_denseweight"])
    
    # split dataset
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

    print(f"Size of dataset: {len(dataset)}")
    
    model = GRU(
        input_dim=dataset.gru_input_dim,
        num_hidden_nodes=params["num_hidden_nodes"],
        num_hidden_layers=params["num_hidden_layers"],
        dropout=params["dropout"],
        learning_rate=params["learning_rate"],
        tqdm_disabled = True
    )

    # Train and evaluate
    # maybe we should also define a different validation method that is more sensitive to less precipitation 
    model.fit(
        train_dataset, 
        batch_size=params["batch_size"], 
        max_epochs=MAX_EPOCHS, # Shorter epochs for tuning
        patience=5,
        frac_valid=0.1,
        num_workers=NUM_WORKERS
    )

    best_checkpoint_path = f"{OUTPUT_DIR}/trial_{trial.number}_best.pth"
    model.save_best_checkpoint(best_checkpoint_path)

    model = GRU.load(best_checkpoint_path, tqdm_disabled=True)
    print(f"Successfully loaded best checkpoint for trial {trial.number}")

    y_pred, y_true = model.predict_wLabels(test_dataset, log_name="val_tuning", num_workers=NUM_WORKERS)
    print(f"shapes y_pred/y_true: {y_pred.shape}, {y_true.shape}")

    if np.isnan(y_pred).any():
        print("WARNING: Model predicted NaNs!")
    
    # Calculate specialized QPE metrics
    metrics = perfscores(y_pred, y_true, bounds=[0, 1, 10, np.inf])
    
    scatter = metrics["all"]["scatter"]
    abs_log_bias = abs(metrics["all"]["logBias"])

    if np.isnan(scatter) or np.isnan(abs_log_bias): # is nan if predictions are all 0
        # Return a large penalty value (since we are minimizing)
        return 10.0, 10.0



    return scatter, abs_log_bias

if __name__ == "__main__":
    print("Starting train_gru_kfold.py script", flush=True)

    # Ensure the directory for checkpoints exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Is CUDA available? {torch.cuda.is_available()}")
    
    # Create a study to MINIMIZE scatter and log bias
    study = optuna.create_study(
        directions=["minimize", "minimize"], 
        study_name=STUDY_NAME,
        storage=f"sqlite:///{OUTPUT_DIR}/optuna_tune.db",
        load_if_exists=True
    ) 
    
    study.optimize(objective, n_trials=N_TRIALS, catch=(ValueError,))

    print("\n--- Optimization Finished ---")
    
    # For multi-objective, we look at the Pareto Front
    best_trials = study.best_trials
    print(f"Number of Pareto optimal trials: {len(best_trials)}")

    for i, trial in enumerate(best_trials):
        print(f"\nPareto Trial {i}:")
        print(f"  Trial Number: {trial.number}")
        print(f"  Values (Scatter, AbsLogBias): {trial.values}")
        print(f"  Params: {trial.params}")

    # Save all trial data to CSV
    df = study.trials_dataframe()
    df.to_csv(f"{OUTPUT_DIR}/optuna_tuning_results.csv", index=False)
    
    print(f"\nResults saved to {OUTPUT_DIR}/optuna_tuning_results.csv")