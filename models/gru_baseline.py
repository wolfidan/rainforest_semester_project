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
from models.gru_base import GRUBase

import logging

logging.basicConfig(level=logging.INFO)

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Define GRU Model ---
class GRUmodel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0, layer_norm=False):
        super(GRUmodel, self).__init__()
        self.GRU = nn.GRU(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.layer_norm = layer_norm

    def forward(self, x, lengths):
        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.GRU(x_packed)
        last_hidden = hidden[-1]
        if self.layer_norm:
            last_hidden = self.ln(last_hidden)
        out = self.fc(last_hidden)
        return out.squeeze(-1)


class GRU(GRUBase):
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
        logger: Logger = None,
        tqdm_disabled=False,
    ):
        self.model = GRUmodel(
            input_dim, num_hidden_nodes, num_hidden_layers, dropout, layer_norm=layer_norm
        ).to(device)

        self.loss_function_text = loss_function
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
            factor=lr_scheduler_factor,
            patience=3,
        )

        self.logger = logger
        self.tqdm_disabled = tqdm_disabled
