# %%
import pandas as pd
import pickle
from rf import RandomForestRegressorBC
from gru import GRU
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error

# %%
"""
This script performs K-fold cross-validation for both GRU and RandomForestRegressorBC models
It is meant as an example, in practice hyperparameter tuning should be performed.
"""

##########################################################################################################
# USER INPUT
##########################################################################################################
INPUT_DIR = "/store_new/mch/msrad/radar/radar_database_v2/rf_input_data/"
OUTPUT_DIR = "/scratch/mch/wolfensb/rainforest_semester_project/saved_models/"
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
SUBSET = 0.6  # Use a subset of data for faster example running (max = 1.0)
##########################################################################################################


def prepare_for_gru(radar_db):
    radar_sub = pd.get_dummies(radar_db, columns=["RADAR"])
    dummies = [col for col in radar_sub.columns if col.startswith("RADAR_")]
    numeric = radar_sub.columns.difference(dummies)
    scaler = StandardScaler()
    radar_sub[numeric] = scaler.fit_transform(radar_sub[numeric])
    radar_sub = radar_sub.astype(
        {col: "int8" for col in radar_sub.select_dtypes("boolean").columns}
    )
    return radar_sub


def group_kfold_split(groups, n_splits=5, shuffle=True, random_state=None):
    """
    Picks group-wise train-test splits for cross-validation.

    Parameters
    ----------
    groups : array-like of shape (n_samples,)
        Group labels for each row.
    n_splits : int
        Number of folds.
    shuffle : bool
        Whether to shuffle group order.
    random_state : int or None
        Random seed.

    Yields
    ------
    train_groups : ndarray
        Groups used for training.
    test_groups : ndarray
        Groups used for testing.
    """
    rng = np.random.default_rng(random_state)

    unique_groups = np.unique(groups)

    if shuffle:
        rng.shuffle(unique_groups)

    folds = np.array_split(unique_groups, n_splits)

    for i in range(n_splits):
        test_groups = folds[i]
        train_groups = np.hstack(folds[:i] + folds[i + 1 :])
        yield train_groups, test_groups


def random_group_subset(groups, subset_percent, shuffle=True, random_state=None):
    """
    Randomly selects a subset of groups.

    Parameters
    ----------
    groups : array-like of shape (n_samples,)
        Group labels for each row.
    subset_percent : float
        Percentage of groups to select (between 0 and 1).
    shuffle : bool
        Whether to shuffle group order.
    random_state : int or None
        Random seed.

    Returns
    -------
    subset_mask : ndarray of shape (n_samples,)
        Boolean mask indicating which rows belong to the selected groups.
    """
    rng = np.random.default_rng(random_state)

    unique_groups = np.unique(groups)

    if shuffle:
        rng.shuffle(unique_groups)

    n_subset = int(len(unique_groups) * subset_percent)
    selected_groups = rng.choice(unique_groups, size=n_subset, replace=False)

    subset_mask = np.isin(groups, selected_groups)
    # In subset_mask we lose the original ordering of the selected groups
    # so we also need to reorder selected groups
    return sorted(selected_groups), subset_mask


# --- helpers to save predictions -------------------------------------------------


def _make_pred_df(
    *,
    model_name: str,
    fold: int,
    split: str,  # "train" or "test"
    vertgroup_ids: np.ndarray,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Create a tidy dataframe with observed vs predicted values.
    Indexing in this project:
      - y_true is indexed by vertgroup_id (group-level target)
      - y_pred is aligned with vertgroup_ids (same order)
    """
    y_true_aligned = y_true.loc[sorted(vertgroup_ids)].to_numpy()

    return pd.DataFrame(
        {
            "model": model_name,
            "fold": fold,
            "split": split,
            "vertgroup": vertgroup_ids,
            "y_obs": y_true_aligned,
            "y_pred": y_pred,
            "error": y_pred - y_true_aligned,
        }
    )


def save_cv_predictions(
    *,
    outdir: str,
    model_name: str,
    preds_per_fold: list[pd.DataFrame],
    filename_prefix: str,
) -> None:
    """
    Save:
      - per-fold parquet files
      - a combined parquet file
    """
    os.makedirs(outdir, exist_ok=True)

    # Save per-fold
    for df in preds_per_fold:
        fold = int(df["fold"].iloc[0])
        split = str(df["split"].iloc[0])
        fpath = os.path.join(
            outdir, f"{filename_prefix}_{model_name}_fold{fold:02d}_{split}.parquet"
        )
        df.to_parquet(fpath, index=False)

    # Save combined
    combined = pd.concat(preds_per_fold, ignore_index=True)
    combined_path = os.path.join(
        outdir, f"{filename_prefix}_{model_name}_all_folds.parquet"
    )
    combined.to_parquet(combined_path, index=False)

# --- Load data ------------------------------------------------------------------

radar = pd.read_parquet(os.path.join(INPUT_DIR, "radar_x0y0.parquet"))
gauge = pd.read_parquet(os.path.join(INPUT_DIR, "gauge.parquet"))
groups = pickle.load(open(os.path.join(INPUT_DIR, "grouping_idx_x0y0.p"), "rb"))

features = radar[COLS_TO_USE]
vertgroups = groups["grp_vertical"]
targets = gauge["RRE150Z0"]  # must be indexed by vertgroup ids for loc[] usage below

# Subset data for faster example running
selected_groups, subset_mask = random_group_subset(vertgroups, SUBSET, random_state=42)
features = features.loc[subset_mask]
vertgroups = vertgroups[subset_mask]
targets = targets.loc[selected_groups]

# Prepare features for GRU
features_gru = prepare_for_gru(features)

# Output folders for predictions
PRED_DIR = os.path.join(OUTPUT_DIR, "cv_predictions")
os.makedirs(PRED_DIR, exist_ok=True)

# %%
# ##########################################################
# # GRU Cross-validation + save preds (train & test)
# ##########################################################
print("=== GRU Cross-Validation ===")
gru_test_errors = []

gru_pred_dfs = []  # collect train+test dfs for all folds

kf = group_kfold_split(vertgroups, n_splits=N_SPLITS, random_state=42)
for fold, (vertgroups_train, vertgroups_test) in enumerate(kf, start=1):
    # Init model
    gru_model = GRU(input_dim=features_gru.shape[1])
    print(f"GRU Fold {fold}")

    # get row masks that correspond to train/test vertgroups
    train_mask = np.isin(vertgroups, vertgroups_train)
    test_mask = np.isin(vertgroups, vertgroups_test)

    X_train, X_test = features_gru.loc[train_mask], features_gru.loc[test_mask]
    # See remark at the end of random_group_subset to understand why we need sorted
    y_train, y_test = (
        targets.loc[sorted(vertgroups_train)],
        targets.loc[sorted(vertgroups_test)],
    )
    grp_train, grp_test = vertgroups[train_mask], vertgroups[test_mask]

    # Fit
    gru_model.fit(X_train, y_train, grp_train)

    # Predict on TRAIN (group-level)
    y_pred_train = gru_model.predict(X_train, grp_train)
    gru_pred_dfs.append(
        _make_pred_df(
            model_name="GRU",
            fold=fold,
            split="train",
            vertgroup_ids=np.asarray(vertgroups_train),
            y_true=targets,
            y_pred=np.asarray(y_pred_train),
        )
    )

    # Predict on TEST (group-level)
    y_pred_test = gru_model.predict(X_test, grp_test)
    gru_pred_dfs.append(
        _make_pred_df(
            model_name="GRU",
            fold=fold,
            split="test",
            vertgroup_ids=np.asarray(vertgroups_test),
            y_true=targets,
            y_pred=np.asarray(y_pred_test),
        )
    )

    # Metric on test
    mse = mean_squared_error(y_test, y_pred_test)
    gru_test_errors.append(mse)
    print(f"Fold {fold} MSE: {mse:.4f}")

print(f"GRU CV mean MSE: {np.mean(gru_test_errors):.4f}")

# Save all GRU CV preds
save_cv_predictions(
    outdir=PRED_DIR,
    model_name="GRU",
    preds_per_fold=gru_pred_dfs,
    filename_prefix="cv_pred",
)

# Fit final GRU on all data + save model
gru_model = GRU(input_dim=features_gru.shape[1])
gru_model.fit(features_gru, targets, vertgroups)
gru_model.save(os.path.join(OUTPUT_DIR, "gru_model_final.pth"))
print("GRU final model saved.")

# %%
##########################################################
# RF Cross-validation + save preds (train & test)
##########################################################
print("\n=== RandomForestRegressorBC Cross-Validation ===")
rf_test_errors = []

rf_pred_dfs = []  # collect train+test dfs for all folds

kf = group_kfold_split(vertgroups, n_splits=N_SPLITS, random_state=42)
for fold, (vertgroups_train, vertgroups_test) in enumerate(kf, start=1):
    # Init model
    rf_model = RandomForestRegressorBC(beta=-0.5, bctype="spline")
    print(f"RF Fold {fold}")

    train_mask = np.isin(vertgroups, vertgroups_train)
    test_mask = np.isin(vertgroups, vertgroups_test)

    X_train, X_test = features.loc[train_mask], features.loc[test_mask]
    # See remark at the end of random_group_subset to understand why we need sorted
    y_train, y_test = (
        targets.loc[sorted(vertgroups_train)],
        targets.loc[sorted(vertgroups_test)],
    )
    grp_train, grp_test = vertgroups[train_mask], vertgroups[test_mask]

    # Fit
    rf_model.fit(X_train, y_train, grp_train)

    # Predict on TRAIN (group-level)
    y_pred_train = rf_model.predict(X_train, grp_train)
    rf_pred_dfs.append(
        _make_pred_df(
            model_name="RF",
            fold=fold,
            split="train",
            vertgroup_ids=np.asarray(vertgroups_train),
            y_true=targets,
            y_pred=np.asarray(y_pred_train),
        )
    )

    # Predict on TEST (group-level)
    y_pred_test = rf_model.predict(X_test, grp_test)
    rf_pred_dfs.append(
        _make_pred_df(
            model_name="RF",
            fold=fold,
            split="test",
            vertgroup_ids=np.asarray(vertgroups_test),
            y_true=targets,
            y_pred=np.asarray(y_pred_test),
        )
    )

    mse = mean_squared_error(y_test, y_pred_test)
    rf_test_errors.append(mse)
    print(f"Fold {fold} MSE: {mse:.4f}")

print(f"RF CV mean MSE: {np.mean(rf_test_errors):.4f}")

# Save all RF CV preds
save_cv_predictions(
    outdir=PRED_DIR,
    model_name="RF",
    preds_per_fold=rf_pred_dfs,
    filename_prefix="cv_pred",
)

# Fit final RF on all data + save model
rf_model = RandomForestRegressorBC(beta=-0.5, bctype="spline")
rf_model.fit(features, targets, vertgroups)
rf_model.save(os.path.join(OUTPUT_DIR, "rf_model_final.pkl"))
print("RF final model saved.")
