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
from models.gru_base import GRUBase


# --- Define GRU Model ---
class GRUmodelBidirectional(nn.Module):
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

    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0, layer_norm = False):
        super(GRUmodelBidirectional, self).__init__()
        self.GRU = nn.GRU(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)  # Output single precipitation value
        # self.relu = nn.ReLU() # as precipitation can't be negative --- we shouldn't use that, if output is negative at first, it will never start learning

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
        _, hidden = self.GRU(x_packed)  # GRU processes only valid timesteps, hidden shape is num_layers,batch_size,hidden_dim for 
        # forward hidden state of the last layer
        hidden_fwd = hidden[-2] 
        # backward hidden state of the last layer
        hidden_bwd = hidden[-1]

        last_hidden = torch.cat((hidden_fwd, hidden_bwd), dim=1)
        out = self.fc(last_hidden)  # Use last hidden state of last layer
        # Use squeeze(-1) to only drop the feature dimension (keep batch dim) - could lead to error if the batch only constist of one sample otherwise
        return out.squeeze(-1)


class GRUBidirectional(GRUBase):
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
        batch_size=None,
        layer_norm=False,
        lr_scheduler_factor=0.5,
        logger:Logger = None,
        tqdm_disabled=False,
    ):

        # build model on device
        self.model = GRUmodelBidirectional(
            input_dim, num_hidden_nodes, num_hidden_layers, dropout, layer_norm=layer_norm
        ).to(device)

        self.loss_function_text = loss_function

        # hyperparams
        self.hyperparams = {
            "input_dim": input_dim,
            "num_hidden_nodes": num_hidden_nodes,
            "num_hidden_layers": num_hidden_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "loss_function": loss_function,
            "batch_size": batch_size,
            "layer_norm": layer_norm,
            "lr_scheduler_factor": lr_scheduler_factor,
        }

        # Initialize Scaler for Mixed Precision
        self.scaler = torch.cuda.amp.GradScaler()

        # loss
        if loss_function.upper() == "MSE":
            self.criterion = nn.MSELoss(reduction='none')
        elif loss_function.upper() == "MAE":
            self.criterion = nn.L1Loss(reduction='none')
        elif loss_function.upper() == "HUBER":
            self.criterion = nn.HuberLoss(reduction='none')
        else:
            raise ValueError(f"Unknown loss function {loss_function}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=lr_scheduler_factor,     # Reduce LR by a factor -> 0.5 = half
            patience=3,      # nr of epochs with no improvement to wait before dropping
        )

        self.logger = logger
        self.tqdm_disabled = tqdm_disabled
