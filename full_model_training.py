from datasets.ScalerAwareGRUDataset import ScalerAwareGRUDataset
import os
from models.gru_baseline import GRU
import pickle
import torch

# Script to train the final model

print(f"Is CUDA available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Targeting GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: Training on CPU.")

INPUT_DIR = "/store_new/mch/msrad/radar/radar_database_v2/rf_input_data_qc/"
OUT_DIR = "./saved_models/GRU_Final_AllData"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(OUT_DIR, "gru_final_model.pth")
HYPERPARAM_SAVE_PATH = os.path.join(OUT_DIR, "hyperparams.pkl")
SCALER_SAVE_PATH = os.path.join(OUT_DIR, "standard_scaler.pkl")

hyperparam_dict = {'num_hidden_nodes': 96,
 'num_hidden_layers': 3,
 'dropout': 0.1676263814467597,
 'learning_rate': 0.00012631341749198117,
 'batch_size': 512,
 'alpha_denseweight': 0.7326962270498213,
 'loss_function': 'mse',
 'sort_by_height': True,
 'lr_scheduler_factor': 0.6988761757979692}


COLS_TO_USE = [
    "RADAR",
    "HEIGHT",
    "ISO0_HEIGHT",
    "ZH_mean",
    "ZV_mean",
    "KDP_mean",
    "RHOHV_mean",
    "SW_mean",
    "AH_mean",
    "VISIB_mean",
    "SWEEP"
]

# save of load hyperparameter dict and cols to use
os.makedirs(os.path.dirname(HYPERPARAM_SAVE_PATH), exist_ok=True)
with open(HYPERPARAM_SAVE_PATH, 'wb') as f:
    pickle.dump({'hyperparam_dict': hyperparam_dict, "cols_to_use": COLS_TO_USE}, f)


print("Initializing Dataset")
dataset = ScalerAwareGRUDataset(
    input_dir=INPUT_DIR,
    cols_to_use=COLS_TO_USE,
    scaler_path=SCALER_SAVE_PATH,
    is_training=True,         # Set to True to compute and save scaler
    subset=1,
    weighting="denseweight",
    alpha_denseweight=hyperparam_dict["alpha_denseweight"],
    sort_by_height=hyperparam_dict["sort_by_height"]
)

dataset.save()

print("\nInitializing GRU Model")
model = GRU(
    input_dim=dataset.gru_input_dim,
    num_hidden_layers=hyperparam_dict["num_hidden_layers"],
    num_hidden_nodes=hyperparam_dict["num_hidden_nodes"],
    dropout=hyperparam_dict["dropout"],
    learning_rate=hyperparam_dict["learning_rate"],
    loss_function=hyperparam_dict["loss_function"],
    lr_scheduler_factor=hyperparam_dict["lr_scheduler_factor"]
)

print("\nStarting Training")
# fit() internally takes frac_valid (default 10%) of the data for Early Stopping
model.fit(
    dataset=dataset,
    batch_size=hyperparam_dict["batch_size"],
    max_epochs=50,
    patience=7,
    frac_valid=0.1, 
    num_workers=4
)

print(f"\nTraining complete. Saving best model checkpoint to {MODEL_SAVE_PATH}")
model.save_best_checkpoint(MODEL_SAVE_PATH)
print("Done!")