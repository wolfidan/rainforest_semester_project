import pandas as pd
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from numba import njit
from helper.Logger import Logger
from tqdm import tqdm

import logging

logging.basicConfig(level=logging.INFO)

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        logger:Logger = None
    ):

        # build model on device
        self.model = GRUmodel(
            input_dim, num_hidden_nodes, num_hidden_layers, dropout
        ).to(device)

        self.loss_function_text = loss_function

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

        self.logger = logger


    def fit(
        self,
        dataset: Dataset, # dataset for training
        weights=None,
        batch_size=512,
        max_epochs=100,
        patience=10,
        frac_valid=0.1
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

        # Split dataset
        train_dataset, val_dataset = random_split(dataset, [1-frac_valid, frac_valid])

        # weights_train = [w[3] for w in train_dataset]
        # weights_valid = [w[3] for w in val_dataset]

        # train_sampler = WeightedRandomSampler(weights_train, len(weights_train))
        # valid_sampler = WeightedRandomSampler(weights_valid, len(weights_valid))

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, #sampler=train_sampler
        )
        valid_loader = DataLoader(
            val_dataset, batch_size=batch_size, #sampler=valid_sampler
        )

        best_val_loss = np.inf  # Track best validation loss
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            print(f"Running Epoch {epoch+1}")
            self.model.train()
            train_loss = 0
            i = 0

            loop_train = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
            for batch_idx, (X_batch, lengths_batch, y_batch, _, _) in loop_train:
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

                loop_train.set_description(f'Epoch [{epoch}/{max_epochs}] train step')

            train_loss /= len(train_loader)  # Average loss

            if self.logger:
                self.logger.log_training_epoch(split="train", epoch=epoch, mse = train_loss) 


            # --- Validation Step ---
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                loop_val = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=False)

                for batch_idx, (X_batch, lengths_batch, y_batch, _, _) in loop_val:
                    X_batch, lengths_batch, y_batch = (
                        X_batch.to(device),
                        lengths_batch.cpu(),
                        y_batch.to(device),
                    )
                    outputs = self.model(X_batch, lengths_batch)
                    loss = self.criterion(outputs, y_batch)
                    val_loss += loss.item()

                    loop_val.set_description(f'Epoch [{epoch}/{max_epochs}] val step')
            val_loss /= len(valid_loader)  # Average validation loss

            if self.logger:
                self.logger.log_training_epoch(split="val", epoch=epoch, mse = val_loss) 

            print(
                f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
            )

            

            # --- Early Stopping Check ---
            print(f'Current val loss: {val_loss}, best val loss: {best_val_loss}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break

    def predict(self, dataset, batch_size=512, log_name = None):
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
    
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)

            for batch_idx, (X_batch, lengths_batch, y_batch, _, idxs) in loop:
                X_batch, lengths_batch, y_batch = (
                    X_batch.to(device),
                    lengths_batch.cpu(),
                    y_batch.to(device),
                )
                y_pred = self.model(X_batch, lengths_batch)  # Forward pass
                all_predictions.append(y_pred)
            
                self.logger.log_eval_pred(split=log_name,
                                          vertgroup_ids=idxs,
                                          y_true=np.asarray(y_batch.cpu()),
                                          y_pred=np.asarray(y_pred.cpu())
                                          )
                loop.set_description("Test/Predict step")

        return torch.cat(all_predictions, dim=0).cpu().numpy()

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
