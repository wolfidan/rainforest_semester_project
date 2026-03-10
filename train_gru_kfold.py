from models.gru_baseline import GRU
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from datasets.GRURainSeqDataset import GRURainSeqDataset
from helper.Logger import Logger

"""
This script performs K-fold cross-validation for GRU
"""

##########################################################################################################
# USER INPUT
##########################################################################################################
INPUT_DIR = "/store_new/mch/msrad/radar/radar_database_v2/rf_input_data/"
OUTPUT_DIR = "/scratch/mch/tkluser/rainforest_semester_project/saved_models/"
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
N_SPLITS = 4 # K-fold split
SUBSET = 1  # Use a subset of data for faster example running (max = 1.0)
FILENAME_PREFIX = "cv_1"
MODEL_NAME = "GRU_Baseline"
##########################################################################################################

print("Starting train_gru_kfold.py script", flush=True)

logger = Logger(outdir=OUTPUT_DIR, model_name=MODEL_NAME, filename_prefix=FILENAME_PREFIX)
print("Instantiating Logger successful", flush=True)

# load dataset
print("Load dataset", flush=True)
dataset = GRURainSeqDataset(input_dir=INPUT_DIR, cols_to_use=COLS_TO_USE, subset=SUBSET)
print("Loading dataset successful", flush=True)

try:
    # K-fold
    kfold = KFold(n_splits=N_SPLITS, shuffle=True)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset), start=1):
        print(f"GRU Fold {fold}", flush=True)
        logger.set_curr_fold(fold)

        train_subset = Subset(dataset, train_ids)
        test_subset = Subset(dataset, test_ids)

        gru_model = GRU(input_dim=dataset.gru_input_dim, logger=logger)

        # fit model 
        gru_model.fit(train_subset)

        # predict train set
        y_pred_train = gru_model.predict(train_subset, log_name="train")

        # predict test set
        y_pred_test = gru_model.predict(test_subset, log_name="test")
except Exception as e:
    raise e
finally:      
    # write logger to file
    logger.write_to_file(OUTPUT_DIR, model_name=MODEL_NAME, filename_prefix=FILENAME_PREFIX)

    # print and plot metrics

    logger.generate_training_metrics(output_dir=OUTPUT_DIR)




