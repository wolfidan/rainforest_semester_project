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

    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0, layer_norm = False):
        super(GRUmodel, self).__init__()
        self.GRU = nn.GRU(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.ln = nn.LayerNorm(hidden_dim) # should help with the high variance in the outputs -> makes it easier for the linear layer as it doesn't have to handle big scale shifts
        self.fc = nn.Linear(hidden_dim, 1)  # Output single precipitation value
        # self.relu = nn.ReLU() # as precipitation can't be negative --- we shouldn't use that, if output is negative at first, it will never start learning

        self.layer_norm = layer_norm
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
        last_hidden = hidden[-1] # extract last layers hidden dim
        if self.layer_norm:
            last_hidden = self.ln(last_hidden)
        out = self.fc(last_hidden)  # Use last hidden state of last layer

        # Use squeeze(-1) to only drop the feature dimension (keep batch dim) - could lead to error if the batch only constist of one sample otherwise
        return out.squeeze(-1)


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
        layer_norm = False,
        lr_scheduler_factor = 0.5,
        logger:Logger = None,
        tqdm_disabled = False
    ):

        # build model on device
        self.model = GRUmodel(
            input_dim, num_hidden_nodes, num_hidden_layers, dropout, layer_norm=layer_norm
        ).to(device)

        self.loss_function_text = loss_function

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
            verbose=True
        )

        self.logger = logger
        self.tqdm_disabled = tqdm_disabled


    def fit(
        self,
        dataset: Dataset, # dataset for training
        batch_size=512,
        max_epochs=100,
        patience=10,
        frac_valid=0.1,
        num_workers = 4,
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
            train_dataset, batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,    # Matches SBATCH -c 2 request
            pin_memory=True, # makes loading to GPU faster
            #sampler=train_sampler
        )

        valid_loader = DataLoader(
            val_dataset,
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,    # Matches SBATCH -c 2 request
            pin_memory=True, # makes loading to GPU faster
            #sampler=train_sampler
        )

        best_val_loss = np.inf  # Track best validation loss
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            print(f"Running Epoch {epoch+1}")
            self.model.train()
            train_loss_unweighted = 0
            train_loss_weighted = 0
            i = 0

            loop_train = tqdm(enumerate(train_loader), total=len(train_loader), leave=False, disable=self.tqdm_disabled)
            for batch_idx, (X_batch, lengths_batch, y_batch, y_weights, _) in loop_train:
                i += 1
                X_batch, lengths_batch, y_batch, y_weights = (
                    X_batch.to(device, non_blocking=True),
                    lengths_batch.cpu(),
                    y_batch.to(device, non_blocking=True),
                    y_weights.to(device, non_blocking=True)
                )

                self.optimizer.zero_grad()

                with torch.autocast(device_type="cuda"):
                    outputs = self.model(X_batch, lengths_batch)
                    raw_loss = self.criterion(outputs, y_batch)

                    # weighted loss
                    weighted_loss = (raw_loss * y_weights).mean()
                    #non weighted loss
                    unweighted_loss = raw_loss.mean()
                
                # we need to do that with scaler beacause of autocast - otherwise, as it's cast to 16 bits, we sometimes would have negative values
                # Scale loss and call backward
                self.scaler.scale(weighted_loss).backward()
                
                # Step the optimizer via the scaler
                self.scaler.step(self.optimizer)
                
                # Update the scale for next iteration
                self.scaler.update()

                train_loss_unweighted += unweighted_loss.item() 
                train_loss_weighted += weighted_loss.item() 

                loop_train.set_description(f'Epoch [{epoch}/{max_epochs}] train step')

            train_loss_weighted /= len(train_loader)  # Average loss
            train_loss_unweighted /= len(train_loader)  # Average loss

            if self.logger:
                self.logger.log_training_epoch(split="train", epoch=epoch, mse = train_loss_unweighted, mse_weighted = train_loss_weighted) 


            # --- Validation Step ---
            self.model.eval()

            val_loss_unweighted = 0
            val_loss_weighted = 0
            with torch.no_grad():
                loop_val = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=False, disable=self.tqdm_disabled)

                for batch_idx, (X_batch, lengths_batch, y_batch, y_weights, _) in loop_val:
                    X_batch, lengths_batch, y_batch, y_weights = (
                        X_batch.to(device, non_blocking=True),
                        lengths_batch.cpu(),
                        y_batch.to(device, non_blocking=True),
                        y_weights.to(device, non_blocking=True)
                    )
                    with torch.autocast(device_type="cuda"):
                        outputs = self.model(X_batch, lengths_batch)
                        raw_loss = self.criterion(outputs, y_batch)

                        # weighted loss
                        weighted_loss = (raw_loss * y_weights).mean()

                        #non weighted loss
                        unweighted_loss = raw_loss.mean()

                    
                    val_loss_unweighted += unweighted_loss.item() 
                    val_loss_weighted += weighted_loss.item() 

                    loop_val.set_description(f'Epoch [{epoch}/{max_epochs}] val step')

            val_loss_unweighted /= len(valid_loader)  # Average validation loss
            val_loss_weighted /= len(valid_loader)  

            if self.logger:
                self.logger.log_training_epoch(split="val", epoch=epoch, mse = val_loss_unweighted, mse_weighted = val_loss_weighted) 

            print(
                f"Epoch {epoch+1}: Train Loss weighted/unweighted = {train_loss_weighted:.4f}/{train_loss_unweighted:.4f}, Val Loss weighted/unweighted = {val_loss_weighted:.4f}/{val_loss_unweighted}"
            )
            
            val_loss = val_loss_weighted # use weighted loss for validation

            # LR scheduler step
            self.scheduler.step(val_loss)

            # --- Early Stopping Check ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self.best_checkpoint = {
                    "model_state_dict": self.model.state_dict().copy(),
                    "input_dim": self.model.GRU.input_size,
                    "hidden_dim": self.model.GRU.hidden_size,
                    "num_layers": self.model.GRU.num_layers,
                }
            else:
                epochs_no_improve += 1

            print(f'Current val loss: {val_loss}, best val loss: {best_val_loss}')

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break

        return best_val_loss

    def predict(self, dataset, batch_size=512, log_name = None, num_workers = 4):
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
    
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,    # Matches SBATCH -c 2 request
                                pin_memory=True, # makes loading to GPU faster
                            )
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, disable=self.tqdm_disabled)

            for batch_idx, (X_batch, lengths_batch, y_batch, _, idxs) in loop:
                X_batch, lengths_batch, y_batch = (
                    X_batch.to(device, non_blocking=True),
                    lengths_batch.cpu(),
                    y_batch.to(device, non_blocking=True),
                )
                y_pred = self.model(X_batch, lengths_batch)  # Forward pass
                all_predictions.append(y_pred)

                if self.logger:
                    self.logger.log_eval_pred(split=log_name,
                                          vertgroup_ids=idxs,
                                          y_true=np.asarray(y_batch.cpu()),
                                          y_pred=np.asarray(y_pred.cpu())
                                          )
                loop.set_description("Test/Predict step")

        return torch.cat(all_predictions, dim=0).cpu().numpy()
    

    def predict_wLabels(self, dataset, batch_size=512, log_name = None, num_workers = 4):
        """
        Generate predictions for new data and returns predictions with actual value.

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
        labels : ndarray of shape (n_sequences,)
            True precipitation label
        """
        # get sequences
    
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,    # Matches SBATCH -c 2 request
                                pin_memory=True, # makes loading to GPU faster
                            )
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, disable=self.tqdm_disabled)

            for batch_idx, (X_batch, lengths_batch, y_batch, _, idxs) in loop:
                X_batch, lengths_batch, y_batch = (
                    X_batch.to(device, non_blocking=True),
                    lengths_batch.cpu(),
                    y_batch.to(device, non_blocking=True),
                )
                y_pred = self.model(X_batch, lengths_batch)  # Forward pass
                all_predictions.append(y_pred)
                all_labels.append(y_batch)

                if self.logger:
                    self.logger.log_eval_pred(split=log_name,
                                          vertgroup_ids=idxs,
                                          y_true=np.asarray(y_batch.cpu()),
                                          y_pred=np.asarray(y_pred.cpu())
                                          )
                loop.set_description("Test/Predict step")

        return torch.cat(all_predictions, dim=0).cpu().numpy(), torch.cat(all_labels, dim=0).cpu().numpy() # also return true labels

    @classmethod
    def load(cls, path, map_location=None, tqdm_disabled=False, logger:Logger = None):
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
            tqdm_disabled = tqdm_disabled,
            logger = logger
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

    def save_best_checkpoint(self, path):
        if self.best_checkpoint == None:
            print("Saving the best checkpoint without training doesn't work")
        else:
            torch.save(self.best_checkpoint, path)