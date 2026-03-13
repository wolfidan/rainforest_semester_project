from models.gru_baseline import GRU
from sklearn.model_selection import KFold
from torch.utils.data import Subset, random_split
from datasets.GRURainSeqDataset import GRURainSeqDataset
from helper.Logger import Logger
import torch
import os

"""
This script performs K-fold cross-validation for GRU
"""

##########################################################################################################
# USER INPUT
##########################################################################################################
MODEL_NAME = "GRU_Baseline_Denseweight_alpha10"
FILENAME_PREFIX = "cv_1"
INPUT_DIR = "/store_new/mch/msrad/radar/radar_database_v2/rf_input_data/"
OUTPUT_DIR = f"/scratch/mch/tkluser/rainforest_semester_project/saved_models/{MODEL_NAME}"
SUBSET = 1  # Use a subset of data for faster example running (max = 1.0)
MAX_EPOCHS = 100
N_SPLITS = 4 # K-fold split
NUM_WORKERS = 2 # adapt in slurm job accordingly
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
] # Which features to use (see rainforest paper for justification)

##########################################################################################################

print("Starting train_gru_kfold.py script", flush=True)

print(f"Is CUDA available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Targeting GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: Training on CPU.")

logger = Logger(outdir=OUTPUT_DIR, model_name=MODEL_NAME, filename_prefix=FILENAME_PREFIX)
print("Instantiating Logger successful", flush=True)

# load dataset
print("Load dataset", flush=True)
dataset = GRURainSeqDataset(input_dir=INPUT_DIR, cols_to_use=COLS_TO_USE, subset=SUBSET, weighting="denseweight")
print("Loading dataset successful", flush=True)

try:
    # K-fold
    kfold = KFold(n_splits=N_SPLITS, shuffle=True)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset), start=1):
        print(f"GRU Fold {fold}", flush=True)
        logger.set_curr_fold(fold)

        train_subset = Subset(dataset, train_ids)
        test_subset = Subset(dataset, test_ids)
        train_test_subset, _ = random_split(train_subset, [0.5, 0.5]) # test on a subset of the train set

        gru_model = GRU(input_dim=dataset.gru_input_dim, logger=logger)

        # fit model 
        gru_model.fit(train_subset, num_workers=NUM_WORKERS, max_epochs=MAX_EPOCHS, patience=5)

        # predict train set
        y_pred_train = gru_model.predict(train_test_subset, log_name="train", num_workers=NUM_WORKERS)

        # predict test set
        y_pred_test = gru_model.predict(test_subset, log_name="test", num_workers=NUM_WORKERS)
except Exception as e:
    raise e
finally:      
    # write logger to file
    logger.write_to_file(outdir=OUTPUT_DIR)

    # print and plot metrics

    logger.generate_training_metrics(output_dir=OUTPUT_DIR)

    combined_path = os.path.join(
            OUTPUT_DIR, "last_gru_checkpoint.pth"
        )
    gru_model.save(path=combined_path)




