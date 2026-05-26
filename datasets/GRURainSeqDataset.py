import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from denseweight import DenseWeight
from datasets.datasetUtils import _find_group_indices, _process_groups_fast



class GRURainSeqDataset(Dataset):
    def __init__(self,
                 input_dir,
                 cols_to_use,
                 subset=1,
                 weighting=None, # "denseweight"
                 alpha_denseweight = 1.0,
                 sort_by_height = True
                 ):

        # --- Load data ------------------------------------------------------------------

        radar = pd.read_parquet(os.path.join(input_dir, "radar_x0y0.parquet"), columns=cols_to_use)
        gauge = pd.read_parquet(os.path.join(input_dir, "gauge.parquet"), columns=["RRE150Z0"])
        groups = pickle.load(open(os.path.join(input_dir, "grouping_idx_x0y0.p"), "rb"))

        print("loading parquet files successful")

        features = radar[cols_to_use]
        vertgroups = groups["grp_vertical"]
        targets = gauge["RRE150Z0"]  # must be indexed by vertgroup ids for loc[] usage below

        # Subset data for faster example running
        selected_groups, subset_mask = self.random_group_subset(vertgroups, subset, random_state=42)
        features = features.loc[subset_mask]
        vertgroups = vertgroups[subset_mask]
        targets = targets.loc[selected_groups]
        print("Subsetting dataset successful")

        if sort_by_height:
            features["tmp_grp"] = vertgroups
            features = features.sort_values(by=["tmp_grp", "HEIGHT"], ascending=[True, False]) # we want to sort descending by height
            vertgroups = features["tmp_grp"].values
            features = features.drop(columns=["tmp_grp"])

        # Prepare features for GRU
        features_gru = self.prepare_for_gru(features)
        self.gru_input_dim = features_gru.shape[1]

        features_gru.replace(False, 0, inplace=True)
        features_gru.replace(True, 1, inplace=True)
        features_gru = features_gru.to_numpy()


        # get sequences
        sequences = self.create_sequences(features_gru, vertgroups)

        targets = targets.to_numpy()

        if weighting == "denseweight":
            dw = DenseWeight(alpha=alpha_denseweight) # alpha = 0 for uniform sampling
            weights = dw.fit(targets)
            self.weights = torch.tensor(weights, dtype=torch.float32)
        else:
            self.weights = torch.ones(len(targets), dtype=torch.float32)
        
        self.targets = torch.tensor(targets, dtype=torch.float32)

        self.lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)  # Compute lengths

        # save vertgroups
        self.selected_groups =  selected_groups

        # Pad sequences to the maximum length
        self.padded_sequences = pad_sequence(sequences, batch_first=True)

        print("preparing features successful")

    def prepare_for_gru(self, radar_db):
        # one hot encoding
        radar_sub = pd.get_dummies(radar_db, columns=["RADAR"])
        dummies = [col for col in radar_sub.columns if col.startswith("RADAR_")]

        # standard scale other features
        numeric = radar_sub.columns.difference(dummies)
        scaler = StandardScaler()
        radar_sub[numeric] = scaler.fit_transform(radar_sub[numeric])
        radar_sub = radar_sub.astype(
            {col: "int8" for col in radar_sub.select_dtypes("boolean").columns}
        )
        return radar_sub
    
    def random_group_subset(self, groups, subset_percent, shuffle=True, random_state=None):
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
        return sorted(selected_groups), subset_mask # sorted by index
    
    def create_sequences(self, features, groups):
        # Reindex groups safely
        _, groups = np.unique(groups, return_inverse=True)
        groups = groups.astype(np.int32)

        unique_groups, counts = np.unique(groups, return_counts=True)
        max_seq_len = np.max(counts)

        start_idx, end_idx = _find_group_indices(groups)

        num_samples = len(unique_groups)
        num_features = features.shape[1]

        # if num_features != self.model.GRU.input_size:
        #     raise IndexError(
        #         f"Input features have {num_features} but model was defined for "
        #         f"{self.model.GRU.input_size} features (input_dim argument)"
        #     )

        padded_sequences = np.zeros(
            (num_samples, max_seq_len, num_features), dtype=np.float32
        )

        padded_sequences = _process_groups_fast(
            np.asarray(features), start_idx, end_idx, padded_sequences
        )

        return torch.tensor(padded_sequences, dtype=torch.float32)
    

    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return (
            self.padded_sequences[idx].float(),
            self.lengths[idx],
            self.targets[idx],
            self.weights[idx],
            self.selected_groups[idx] # correspond to vertgroups
        )
        