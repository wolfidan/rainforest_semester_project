#%%
import pandas as pd
import pickle
from rf import RandomForestRegressorBC
from gru import GRU
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error
#%%
"""
This script performs 4-fold cross-validation for both GRU and RandomForestRegressorBC models
It is meant as an example, in practice hyperparameter tuning should be performed.
"""


def prepare_for_gru(radar_db):
    radar_sub = pd.get_dummies(radar_db, columns=["RADAR"])
    dummies = [col for col in radar_sub.columns if col.startswith("RADAR_")]
    numeric = radar_sub.columns.difference(dummies)
    scaler = StandardScaler()
    radar_sub[numeric] = scaler.fit_transform(radar_sub[numeric])
    radar_sub = radar_sub.astype({col: "int8" for col in radar_sub.select_dtypes("boolean").columns})
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
        train_groups = np.hstack(folds[:i] + folds[i+1:])
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
    return selected_groups, subset_mask  
    
        
##########################################################################################################
INPUT_DIR = "/store_new/mch/msrad/radar/radar_database_5min/rf_input_data/"
OUTPUT_DIR = "/store_new/mch/msrad/msradv/projects/rainforest_mark2/rainforest/saved_models/"
COLS_TO_USE = ["RADAR", "HEIGHT", "ISO0_HEIGHT", "ZH_mean", "ZV_mean", "KDP_mean", "RHOHV_mean", "SW_mean", "AH_mean", "VISIB_mean"]
N_SPLITS = 4
SUBSET = 0.4 # Use a subset of data for faster example running (max = 1.0)
##########################################################################################################

# Load data
radar = pd.read_parquet(os.path.join(INPUT_DIR, "radar_x0y0.parquet"))
gauge = pd.read_parquet(os.path.join(INPUT_DIR, "gauge.parquet"))
groups = pickle.load(open(os.path.join(INPUT_DIR, "grouping_idx_x0y0.p"), "rb"))

# Subset features
features = radar[COLS_TO_USE] 
vertgroups = groups["grp_vertical"]
targets = gauge["RRE005R0"]

# Subset data for faster example running
selected_groups, subset_mask = random_group_subset(vertgroups, SUBSET, random_state=42)
features = features.iloc[subset_mask]
vertgroups = vertgroups[subset_mask]
targets = targets.iloc[selected_groups]    

# Prepare features for GRU
features_gru = prepare_for_gru(features)

# Create model instances
# In practice, hyperparameter tuning should be performed
gru_model = GRU(input_dim=features_gru.shape[1])
rf_model = RandomForestRegressorBC(beta=-0.5, bctype='spline')

#%%
##########################################################
# GRU Cross-validation
##########################################################
print("=== GRU Cross-Validation ===")
gru_test_errors = []
kf = group_kfold_split(vertgroups, n_splits=N_SPLITS, random_state=42)
for fold, (vertgroups_train, vertgroups_test) in enumerate(kf):
    print(f"GRU Fold {fold+1}")
    
    # get row masks that correspond to train/test vertgroups
    train_mask = np.isin(vertgroups, vertgroups_train)
    test_mask  = np.isin(vertgroups, vertgroups_test)

    X_train, X_test = features_gru.iloc[train_mask], features_gru.iloc[test_mask]
    y_train, y_test = targets.loc[vertgroups_train], targets.loc[vertgroups_test]
    grp_train, grp_test = vertgroups[train_mask], vertgroups[test_mask]
    
    gru_model.fit(X_train, y_train, grp_train)
    y_pred = gru_model.predict(X_test, grp_test)
    mse = mean_squared_error(y_test, y_pred)
    gru_test_errors.append(mse)
    print(f"Fold {fold+1} MSE: {mse:.4f}")

print(f"GRU CV mean MSE: {np.mean(gru_test_errors):.4f}")

# Fit final GRU on all data
gru_model.fit(features_gru, targets, vertgroups)
gru_model.save(os.path.join(OUTPUT_DIR, "gru_model_final.pth"))
print("GRU final model saved.")

#%%
##########################################################
# RF Cross-validation
##########################################################
print("\n=== RandomForestRegressorBC Cross-Validation ===")
rf_test_errors = []
kf = group_kfold_split(vertgroups, n_splits=N_SPLITS, random_state=42)
 
for fold, (vertgroups_train, vertgroups_test) in enumerate(kf):
    print(f"RF Fold {fold+1}")
    
    # get row masks that correspond to train/test vertgroups
    train_mask = np.isin(vertgroups, vertgroups_train)
    test_mask  = np.isin(vertgroups, vertgroups_test)

    X_train, X_test = features.iloc[train_mask], features.iloc[test_mask]
    y_train, y_test = targets.loc[vertgroups_train], targets.loc[vertgroups_test]
    grp_train, grp_test = vertgroups[train_mask], vertgroups[test_mask]
    
    rf_model.fit(X_train, y_train, grp_train)
    y_pred = rf_model.predict(X_test, grp_test)
    mse = mean_squared_error(y_test, y_pred)
    rf_test_errors.append(mse)
    print(f"Fold {fold+1} MSE: {mse:.4f}")

print(f"RF CV mean MSE: {np.mean(rf_test_errors):.4f}")

# Fit final RF on all data
rf_model.fit(features, targets, vertgroups)
rf_model.save(os.path.join(OUTPUT_DIR, "rf_model_final.pkl"))
print("RF final model saved.")


# %%
