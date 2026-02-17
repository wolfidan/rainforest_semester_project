import pandas as pd
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from numba import njit

import logging

logging.basicConfig(level=logging.INFO)

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    indices = np.searchsorted(groups, np.arange(groups[-1]))
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


class RainDataset(Dataset):
    """
    PyTorch Dataset for variable-length precipitation sequences.

    This dataset stores padded sequences, their original lengths,
    target precipitation values, and optional sample weights.

    Parameters
    ----------
    sequences : list of torch.Tensor
        List of input sequences with shape (seq_len, n_features).
    targets : array-like of shape (n_samples,)
        Target precipitation values.
    weights : array-like of shape (n_samples,), optional
        Sample weights used for weighted sampling. If None, all
        samples receive unit weight.
    """

    def __init__(self, sequences, targets, weights=None):
        self.sequences = sequences
        self.targets = torch.tensor(targets, dtype=torch.float32)
        if weights is not None:
            self.weights = torch.tensor(weights, dtype=torch.float32)
        else:
            self.weights = torch.from_numpy(np.ones((len(self.targets))))
        self.lengths = [len(seq) for seq in sequences]  # Compute lengths

        # Pad sequences to the maximum length
        self.padded_sequences = pad_sequence(sequences, batch_first=True)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.targets)

    def __getitem__(self, idx):
        """
        Retrieve one sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the requested sample.

        Returns
        -------
        padded_sequence : torch.Tensor
            Padded input sequence of shape (max_seq_len, n_features).
        length : int
            Original (unpadded) sequence length.
        target : torch.Tensor
            Target precipitation value.
        weight : torch.Tensor
            Sample weight.
        """
        return (
            self.padded_sequences[idx].float(),
            self.lengths[idx],
            self.targets[idx],
            self.weights[idx],
        )


# --- Define GRU Model ---
class GRUmodel(nn.Module):
    """
    GRU-based neural network for precipitation prediction.

    The model processes variable-length input sequences using a GRU
    and outputs a single scalar prediction per sequence.

    Parameters
    ----------
    input_dim : int
        Number of input features per timestep.
    hidden_dim : int, default=64
        Number of hidden units in the GRU.
    num_layers : int, default=1
        Number of stacked GRU layers.
    dropout : float, default=0
        Dropout probability between GRU layers.
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0):
        super(GRUmodel, self).__init__()
        self.GRU = nn.GRU(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 1)  # Output single precipitation value

    def forward(self, x, lengths):
        """
        Forward pass of the GRU model.

        Parameters
        ----------
        x : torch.Tensor
            Padded input sequences of shape (batch_size, seq_len, input_dim).
        lengths : array-like of shape (batch_size,)
            Original sequence lengths.

        Returns
        -------
        output : torch.Tensor
            Predicted precipitation values of shape (batch_size,).
        """
        # Pack the padded sequences (ignores padding during GRU processing)
        x_packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.GRU(x_packed)  # GRU processes only valid timesteps
        out = self.fc(hidden[-1])  # Use last hidden state of last layer
        return out.squeeze()


class GRU(object):
    """
    High-level wrapper for training and inference of a GRU precipitation model.

    This class handles model construction, training, validation,
    early stopping, and prediction.

    Parameters
    ----------
    input_dim : int, default=18
        Number of input features. It must match the feature dimension of the input data.
    num_hidden_layers : int, default=2
        Number of GRU layers.
    num_hidden_nodes : int, default=64
        Number of hidden units per GRU layer.
    dropout : float, default=0.1
        Dropout probability.
    learning_rate : float, default=0.001
        Learning rate for the Adam optimizer.
    loss_function : {"mse", "mae", "huber"}, default="mse"
        Loss function used for training.
    """

    def __init__(
        self,
        input_dim=18,
        num_hidden_layers=2,
        num_hidden_nodes=64,
        dropout=0.1,
        learning_rate=0.001,
        loss_function="mse",
    ):

        # build model on device
        self.model = GRUmodel(
            input_dim, num_hidden_nodes, num_hidden_layers, dropout
        ).to(device)

        # loss
        if loss_function.upper() == "MSE":
            self.criterion = nn.MSELoss()
        elif loss_function.upper() == "MAE":
            self.criterion = nn.L1Loss()
        elif loss_function.upper() == "HUBER":
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss function {loss_function}")

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(
        self,
        features,
        targets,
        groups,
        weights=None,
        batch_size=512,
        max_epochs=100,
        patience=10,
        frac_valid=0.1,
    ):
        """
        Train the GRU model with early stopping.

        Parameters
        ----------
        features : ndarray or pandas.DataFrame of shape (n_samples, n_features)
            Input feature matrix.
        targets : array-like of shape (n_sequences,)
            Target precipitation values.
        groups : ndarray of shape (n_samples,)
            Group identifiers used to form sequences.
        weights : array-like of shape (n_sequences,), optional
            Sample weights for weighted sampling.
        batch_size : int, default=512
            Mini-batch size.
        max_epochs : int, default=100
            Maximum number of training epochs.
        patience : int, default=10
            Number of epochs without improvement before early stopping.
        frac_valid : float, default=0.1
            Fraction of samples used for validation.
        """
        if isinstance(features, pd.DataFrame):
            features.replace(False, 0, inplace=True)
            features.replace(True, 1, inplace=True)
            features = features.to_numpy()
        if isinstance(targets, pd.DataFrame) or isinstance(targets, pd.Series):
            targets = targets.to_numpy()
        if isinstance(weights, pd.DataFrame) or isinstance(weights, pd.Series):
            weights = weights.to_numpy()

        if len(groups) != len(features):
            logging.error("Length of groups and features must match!")
            sys.exit()
        if weights:
            if len(weights) != len(targets):
                logging.error("Length of weights and targets must match!")
                sys.exit()

        # get sequences
        train_sequences = self.create_sequences(features, groups)
        nsamples = len(train_sequences)
        # Define train-validation split
        train_size = int((1 - frac_valid) * nsamples)
        val_size = nsamples - train_size

        dataset = RainDataset(train_sequences, targets, weights)
        # Split dataset
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        weights_train = [w[3] for w in train_dataset]
        weights_valid = [w[3] for w in val_dataset]
        train_sampler = WeightedRandomSampler(weights_train, len(weights_train))
        valid_sampler = WeightedRandomSampler(weights_valid, len(weights_valid))

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler
        )
        valid_loader = DataLoader(
            val_dataset, batch_size=batch_size, sampler=valid_sampler
        )

        best_val_loss = np.inf  # Track best validation loss
        for epoch in range(max_epochs):
            print(f"Running Epoch {epoch+1}")
            self.model.train()
            train_loss = 0
            i = 0
            for X_batch, lengths_batch, y_batch, _ in train_loader:
                i += 1
                X_batch, lengths_batch, y_batch = (
                    X_batch.to(device),
                    lengths_batch.cpu(),
                    y_batch.to(device),
                )
                self.optimizer.zero_grad()
                outputs = self.model(X_batch, lengths_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)  # Average loss

            # --- Validation Step ---
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, lengths_batch, y_batch, _ in valid_loader:
                    X_batch, lengths_batch, y_batch = (
                        X_batch.to(device),
                        lengths_batch.cpu(),
                        y_batch.to(device),
                    )
                    outputs = self.model(X_batch, lengths_batch)
                    loss = self.criterion(outputs, y_batch)
                    val_loss += loss.item()

            val_loss /= len(valid_loader)  # Average validation loss
            print(
                f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
            )

            # --- Early Stopping Check ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break

    def predict(self, features, groups, batch_size=512):
        """
        Generate predictions for new data.

        Parameters
        ----------
        features : ndarray or pandas.DataFrame
            Input feature matrix.
        groups : ndarray
            Group identifiers defining sequences.
        batch_size : int, default=512
            Batch size used during inference.

        Returns
        -------
        predictions : ndarray of shape (n_sequences,)
            Model-predicted precipitation values.
        """
        # get sequences
        sequences = self.create_sequences(features, groups)
        dataset = RainDataset(sequences, np.zeros((len(sequences))))
        dataloader = DataLoader(dataset, batch_size=batch_size)
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for X_batch, lengths_batch, y_batch, _ in dataloader:
                X_batch, lengths_batch, y_batch = (
                    X_batch.to(device),
                    lengths_batch.cpu(),
                    y_batch.to(device),
                )
                y_pred = self.model(X_batch, lengths_batch)  # Forward pass
                all_predictions.append(y_pred)
        return torch.cat(all_predictions, dim=0).cpu().numpy()

    def create_sequences(self, features, groups):
        # Reindex groups safely
        _, groups = np.unique(groups, return_inverse=True)
        groups = groups.astype(np.int32)

        unique_groups, counts = np.unique(groups, return_counts=True)
        max_seq_len = np.max(counts)

        start_idx, end_idx = _find_group_indices(groups)

        num_samples = len(unique_groups)
        num_features = features.shape[1]

        if num_features != self.model.GRU.input_size:
            raise IndexError(
                f"Input features have {num_features} but model was defined for "
                f"{self.model.GRU.input_size} features (input_dim argument)"
            )

        padded_sequences = np.zeros(
            (num_samples, max_seq_len, num_features), dtype=np.float32
        )

        padded_sequences = _process_groups_fast(
            np.asarray(features), start_idx, end_idx, padded_sequences
        )

        return torch.tensor(padded_sequences, dtype=torch.float32)

    @classmethod
    def load(cls, path, map_location=None):
        """
        Load a GRU model from disk.

        Parameters
        ----------
        path : str
            Path to the saved model file.
        map_location : str or torch.device, optional
            Device mapping for loading (e.g. "cpu").

        Returns
        -------
        model : GRU
            Loaded GRU model instance.
        """
        checkpoint = torch.load(path, map_location=map_location)

        model = cls(
            input_dim=checkpoint["input_dim"],
            num_hidden_nodes=checkpoint["hidden_dim"],
            num_hidden_layers=checkpoint["num_layers"],
        )

        model.model.load_state_dict(checkpoint["model_state_dict"])
        model.model.eval()

        return model

    def save(self, path, save_optimizer=False):
        """
        Save the GRU model to disk.

        Parameters
        ----------
        path : str
            Output file path (e.g. "gru_model.pth").
        save_optimizer : bool, default=False
            Whether to also save the optimizer state.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.model.GRU.input_size,
            "hidden_dim": self.model.GRU.hidden_size,
            "num_layers": self.model.GRU.num_layers,
        }

        if save_optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        torch.save(checkpoint, path)
