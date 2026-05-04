# from models.gru_baseline import GRU
from models.gru_bidirectional import GRUBidirectional
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

# hyperparameters from optimized pareto front by optimizing for scatter and abs log bias, trial 8, which focuses more on log bias
# hyperparam_dict = {
#     'num_hidden_nodes': 64,
#     'num_hidden_layers': 2,
#     'dropout': 0.1,
#     'learning_rate': 0.001,
#     'batch_size': 512,
#     'alpha_denseweight': 0.8106510846663288,
#     'loss_function': 'mse'
# }

hyperparam_dict = {'num_hidden_nodes': 256,
                   'num_hidden_layers': 2,
                   'dropout': 0.23076003898617042,
                   'learning_rate': 0.00010156067242988125,
                   'batch_size': 128,
                   'alpha_denseweight': 0.9,
                   'loss_function': 'mse',
                   'layer_norm': False,
                   'lr_scheduler_factor': 0.647994463601564
            }

MODEL_NAME = "GRU_Bidirectional"
FILENAME_PREFIX = "cv_1"
#INPUT_DIR = "/store_new/mch/msrad/radar/radar_database_v2/rf_input_data/" # without qc
INPUT_DIR = "/store_new/mch/msrad/radar/radar_database_v2/rf_input_data_qc/" # with qc
OUTPUT_DIR = f"/scratch/mch/tkluser/rainforest_semester_project/saved_models/{MODEL_NAME}"
SUBSET = 1  # Use a subset of data for faster example running (max = 1.0)
MAX_EPOCHS = 60
NUM_WORKERS = 4 # adapt in slurm job accordingly
N_SPLITS = 4 # K-fold split
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
] # Which features to use (see rainforest paper for justification)



##########################################################################################################

print("Starting train_gru_kfold.py script", flush=True)
print(f"Is CUDA available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Targeting GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: Training on CPU.")

# Ensure the directory for checkpoints exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = Logger(outdir=OUTPUT_DIR, model_name=MODEL_NAME, filename_prefix=FILENAME_PREFIX)
print("Instantiating Logger successful", flush=True)


# load dataset
print("Load dataset", flush=True)
dataset = GRURainSeqDataset(input_dir=INPUT_DIR,
                            cols_to_use=COLS_TO_USE,
                            subset=SUBSET,
                            weighting="denseweight",
                            alpha_denseweight=hyperparam_dict["alpha_denseweight"],
                            sort_by_height=True
                            )
print(f"Loading dataset successful, length: {len(dataset)}", flush=True)

try:
    # K-fold
    kfold = KFold(n_splits=N_SPLITS, shuffle=True)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset), start=1):
        print(f"GRU Fold {fold}", flush=True)
        logger.set_curr_fold(fold)
        
        # load datasets
        train_subset = Subset(dataset, train_ids)
        test_subset = Subset(dataset, test_ids)
        train_test_subset, _ = random_split(train_subset, [0.5, 0.5]) # test on a subset of the train set for train accuracy

        # instatiate and fit model
        gru_model = GRUBidirectional(input_dim=dataset.gru_input_dim,
                        logger=logger,
                        num_hidden_layers=hyperparam_dict["num_hidden_layers"],
                        num_hidden_nodes=hyperparam_dict["num_hidden_nodes"],
                        dropout=hyperparam_dict["dropout"],
                        learning_rate=hyperparam_dict["learning_rate"],
                        loss_function=hyperparam_dict["loss_function"],
                        tqdm_disabled=True
                        )
        gru_model.fit(train_subset,
                      num_workers=NUM_WORKERS,
                      max_epochs=MAX_EPOCHS,
                      patience=10,
                      batch_size=hyperparam_dict["batch_size"])

        # save and load best checkpoint
        best_checkpoint_path = f"{OUTPUT_DIR}/best_val_checkpoint.pth"
        gru_model.save_best_checkpoint(best_checkpoint_path)
        
        # load best checkpoint, predict train set (subset of train set) and test set
        gru_model = GRUBidirectional.load(best_checkpoint_path, tqdm_disabled=True, logger=logger)
        y_pred_train = gru_model.predict(train_test_subset, log_name="train", num_workers=NUM_WORKERS)
        y_pred_test = gru_model.predict(test_subset, log_name="test", num_workers=NUM_WORKERS)
        
        logger.write_to_file(outdir=OUTPUT_DIR)
except Exception as e:
    raise e
finally:      
    # write logger to file
    logger.write_to_file(outdir=OUTPUT_DIR)

    # print and plot metrics
    logger.generate_training_metrics(output_dir=OUTPUT_DIR)

    # save last checkpoint
    combined_path = os.path.join(
            OUTPUT_DIR, "last_gru_checkpoint.pth"
        )
    gru_model.save(path=combined_path)




