import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
from numba import njit
from torch.nn.utils.rnn import pad_sequence



# first some helper functions
@njit
def _find_group_indices(groups):
    """
    Compute start and end indices for contiguous groups.

    This function assumes that `groups` is a 1D array of integer group
    identifiers that are already sorted and contiguous. It returns the
    start and end indices for each unique group.

    Parameters
    ----------
    groups : ndarray of shape (n_samples,)
        Sorted array of group identifiers.

    Returns
    -------
    start_idx : ndarray of shape (n_groups,)
        Start indices of each group in `groups`.

    end_idx : ndarray of shape (n_groups,)
        End indices (exclusive) of each group in `groups`.
    """
    indices = np.searchsorted(groups, np.arange(groups[-1]+1)) # might have to be +1? because arange not inclusive 
    return indices[:-1], np.append(indices[1:], len(groups))

@njit
def _process_groups_fast(features, start_idx, end_idx, padded_sequences):
    """
    Populate a padded sequence array using precomputed group indices.

    For each group, feature vectors are copied into a preallocated
    padded array. Remaining entries are left as zero-padding.

    Parameters
    ----------
    features : ndarray of shape (n_samples, n_features)
        Input feature matrix sorted by group.
    start_idx : ndarray of shape (n_groups,)
        Start indices for each group.
    end_idx : ndarray of shape (n_groups,)
        End indices (exclusive) for each group.
    padded_sequences : ndarray of shape (n_groups, max_seq_len, n_features)
        Preallocated output array for padded sequences.

    Returns
    -------
    padded_sequences : ndarray
        The input array filled with grouped feature sequences.
    """
    for i in range(len(start_idx)):
        start, end = start_idx[i], end_idx[i]  # Get start and end indices
        group = features[start:end]  # Extract group (already sorted)
        seq_len = group.shape[0]  # Sequence length
        padded_sequences[i, :seq_len, :] = group  # Copy with padding

    return padded_sequences



class GRURainSeqDataset(Dataset):
    def __init__(self,
                 input_dir,
                 cols_to_use,
                 subset=1,
                 weighting=None # not yet implemented
                 ):

        # --- Load data ------------------------------------------------------------------

        radar = pd.read_parquet(os.path.join(input_dir, "radar_x0y0.parquet"))
        gauge = pd.read_parquet(os.path.join(input_dir, "gauge.parquet"))
        groups = pickle.load(open(os.path.join(input_dir, "grouping_idx_x0y0.p"), "rb"))

        features = radar[cols_to_use]
        vertgroups = groups["grp_vertical"]
        targets = gauge["RRE150Z0"]  # must be indexed by vertgroup ids for loc[] usage below

        # Subset data for faster example running
        selected_groups, subset_mask = self.random_group_subset(vertgroups, subset, random_state=42)
        features = features.loc[subset_mask]
        vertgroups = vertgroups[subset_mask]
        targets = targets.loc[selected_groups]

        # Prepare features for GRU
        features_gru = self.prepare_for_gru(features)
        self.gru_input_dim = features_gru.shape[1]

        features_gru.replace(False, 0, inplace=True)
        features_gru.replace(True, 1, inplace=True)
        features_gru = features_gru.to_numpy()


        # get sequences
        sequences = self.create_sequences(features_gru, vertgroups)

        self.targets = torch.tensor(targets.to_numpy(), dtype=torch.float32)

        self.weights = torch.ones(len(self.targets), dtype=torch.float32)

        self.lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)  # Compute lengths

        # save vertgroups
        self.selected_groups =  selected_groups

        # Pad sequences to the maximum length
        self.padded_sequences = pad_sequence(sequences, batch_first=True)

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
        