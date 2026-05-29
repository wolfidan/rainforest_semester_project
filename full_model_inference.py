from datasets.ScalerAwareGRUDataset import ScalerAwareGRUDataset
import os
import pandas as pd
from models.gru_baseline import GRU

# Script to run inference with the final saved model

INPUT_DIR = "/store_new/mch/msrad/radar/radar_database_v2/rf_input_data_qc/"
OUT_DIR = "./saved_models/GRU_Final_AllData"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(OUT_DIR, "gru_final_model.pth")
SCALER_SAVE_PATH = os.path.join(OUT_DIR, "standard_scaler.pkl")
PREDICTIONS_SAVE_PATH = os.path.join(OUT_DIR, "inference_predictions.csv")


#TODO: load prediction dataset

print("Loading dataset metadata and scaler from disk")
dataset = ScalerAwareGRUDataset.load(
    input_dir=INPUT_DIR,
    scaler_path=SCALER_SAVE_PATH,
    is_training=False,
    subset=0.1
)

print("\nLoading trained GRU model")
if not os.path.exists(MODEL_SAVE_PATH):
    raise FileNotFoundError(f"Could not find trained model at {MODEL_SAVE_PATH}")

model = GRU.load(MODEL_SAVE_PATH, tqdm_disabled=True)

y_pred, y_true = model.predict_wLabels(
    dataset=dataset,
    num_workers=4
)

print(f"\nInference complete. Saving predictions to {PREDICTIONS_SAVE_PATH}")
predictions_df = pd.DataFrame({
    "group_id": dataset.selected_groups,
    "y_true": y_true,
    "y_pred": y_pred,
})
predictions_df.to_csv(PREDICTIONS_SAVE_PATH, index=False)
print("Saved prediction CSV successfully.")