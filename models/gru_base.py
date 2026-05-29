import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRUBase:
    """Shared GRU training/inference/save/load functionality.

    Subclasses must set up `self.model`, `self.hyperparams`, `self.logger` and
    `self.tqdm_disabled` in their __init__.
    """

    def fit(self, dataset, batch_size=512, max_epochs=100, patience=10, frac_valid=0.1, num_workers=4):
        n = len(dataset)
        if n < 2:
            raise ValueError(f"Dataset too small for training (len={n}). Provide more data or increase subset size.")

        n_val = int(round(n * frac_valid))
        # ensure at least one sample in validation and training
        if n_val < 1:
            n_val = 1

        n_train = n - n_val
        print(f"[DEBUG] Dataset size: {n}, Train: {n_train}, Val: {n_val}, Batch size: {batch_size}")
        
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        if len(train_loader) == 0 or len(valid_loader) == 0:
            raise ValueError(f"Empty DataLoader encountered (train batches={len(train_loader)}, val batches={len(valid_loader)}). Check dataset size and `batch_size`." )
        
        print(f"[DEBUG] Train batches: {len(train_loader)}, Val batches: {len(valid_loader)}")

        best_val_loss = np.inf
        epochs_no_improve = 0

        loop_epochs = tqdm(range(max_epochs), desc="Epoch", disable=self.tqdm_disabled)
        for epoch in loop_epochs:
            self.model.train()
            train_loss_unweighted = 0
            train_loss_weighted = 0

            for batch_idx, (X_batch, lengths_batch, y_batch, y_weights, _) in enumerate(train_loader):
                
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
                    weighted_loss = (raw_loss * y_weights).mean()
                    unweighted_loss = raw_loss.mean()
                    
                    # Debug first batch of first epoch
                    if epoch == 0 and batch_idx == 0:
                        print(f"[DEBUG] Epoch 0, Batch 0: outputs range=[{outputs.min():.6f}, {outputs.max():.6f}], y_batch range=[{y_batch.min():.6f}, {y_batch.max():.6f}]")
                        print(f"[DEBUG] raw_loss range=[{raw_loss.min():.6f}, {raw_loss.max():.6f}], weights range=[{y_weights.min():.6f}, {y_weights.max():.6f}]")

                self.scaler.scale(weighted_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_loss_unweighted += unweighted_loss.item()
                train_loss_weighted += weighted_loss.item()

            train_loss_weighted /= len(train_loader)
            train_loss_unweighted /= len(train_loader)

            if self.logger:
                self.logger.log_training_epoch(split="train", epoch=epoch, mse=train_loss_unweighted, mse_weighted=train_loss_weighted)

            # Validation
            self.model.eval()
            val_loss_unweighted = 0
            val_loss_weighted = 0
            with torch.no_grad():
                for batch_idx, (X_batch, lengths_batch, y_batch, y_weights, _) in enumerate(valid_loader):
                    lengths_batch = lengths_batch.cpu().view(-1).to(torch.int64)
                    X_batch, y_batch, y_weights = (
                        X_batch.to(device, non_blocking=True),
                        y_batch.to(device, non_blocking=True),
                        y_weights.to(device, non_blocking=True),
                    )
                    with torch.autocast(device_type="cuda"):
                        outputs = self.model(X_batch, lengths_batch)
                        raw_loss = self.criterion(outputs, y_batch)
                        weighted_loss = (raw_loss * y_weights).mean()
                        unweighted_loss = raw_loss.mean()

                    val_loss_unweighted += unweighted_loss.item()
                    val_loss_weighted += weighted_loss.item()

            val_loss_unweighted /= len(valid_loader)
            val_loss_weighted /= len(valid_loader)

            if self.logger:
                self.logger.log_training_epoch(split="val", epoch=epoch, mse=val_loss_unweighted, mse_weighted=val_loss_weighted)

            loop_epochs.set_postfix(
                train_loss=f"{train_loss_weighted:.6f}/{train_loss_unweighted:.6f}",
                val_loss=f"{val_loss_weighted:.6f}/{val_loss_unweighted:.6f}",
            )

            print(f"Epoch {epoch+1}: Train Loss weighted/unweighted = {train_loss_weighted:.6f}/{train_loss_unweighted:.6f}, Val Loss weighted/unweighted = {val_loss_weighted:.6f}/{val_loss_unweighted:.6f}")

            val_loss = val_loss_weighted
            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self.best_checkpoint = {
                    "model_state_dict": self.model.state_dict().copy(),
                    "input_dim": self.model.GRU.input_size if hasattr(self.model, 'GRU') else None,
                    "hidden_dim": self.model.GRU.hidden_size if hasattr(self.model, 'GRU') else None,
                    "num_layers": self.model.GRU.num_layers if hasattr(self.model, 'GRU') else None,
                    "hyperparams": self.hyperparams,
                }
            else:
                epochs_no_improve += 1

            print(f'Current val loss: {val_loss}, best val loss: {best_val_loss}')

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break

        return best_val_loss

    def predict(self, dataset, batch_size=512, log_name=None, num_workers=4):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
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
                y_pred = self.model(X_batch, lengths_batch)
                all_predictions.append(y_pred)
                if self.logger:
                    self.logger.log_eval_pred(split=log_name, vertgroup_ids=idxs, y_true=np.asarray(y_batch.cpu()), y_pred=np.asarray(y_pred.cpu()))
                loop.set_description("Test/Predict step")

        return torch.cat(all_predictions, dim=0).cpu().numpy()

    def predict_wLabels(self, dataset, batch_size=512, log_name=None, num_workers=4):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
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
                y_pred = self.model(X_batch, lengths_batch)
                all_predictions.append(y_pred)
                all_labels.append(y_batch)
                if self.logger:
                    self.logger.log_eval_pred(split=log_name, vertgroup_ids=idxs, y_true=np.asarray(y_batch.cpu()), y_pred=np.asarray(y_pred.cpu()))
                loop.set_description("Test/Predict step")

        return torch.cat(all_predictions, dim=0).cpu().numpy(), torch.cat(all_labels, dim=0).cpu().numpy()

    def save(self, path, save_optimizer=False):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.model.GRU.input_size if hasattr(self.model, 'GRU') else None,
            "hidden_dim": self.model.GRU.hidden_size if hasattr(self.model, 'GRU') else None,
            "num_layers": self.model.GRU.num_layers if hasattr(self.model, 'GRU') else None,
            "hyperparams": self.hyperparams,
        }
        if save_optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path, map_location=None, tqdm_disabled=False, logger=None):
        checkpoint = torch.load(path, map_location=map_location)
        hyperparams = checkpoint.get("hyperparams", {})
        model = cls(
            input_dim=checkpoint.get("input_dim", hyperparams.get("input_dim")),
            num_hidden_nodes=hyperparams.get("num_hidden_nodes", None),
            num_hidden_layers=hyperparams.get("num_hidden_layers", None),
            dropout=hyperparams.get("dropout", 0.0),
            learning_rate=hyperparams.get("learning_rate", 0.001),
            loss_function=hyperparams.get("loss_function", "mse"),
            batch_size=hyperparams.get("batch_size", None),
            layer_norm=hyperparams.get("layer_norm", False),
            lr_scheduler_factor=hyperparams.get("lr_scheduler_factor", 0.5),
            tqdm_disabled=tqdm_disabled,
            logger=logger,
        )
        model.model.load_state_dict(checkpoint["model_state_dict"])
        model.model.eval()
        return model

    def save_best_checkpoint(self, path):
        if getattr(self, 'best_checkpoint', None) is None:
            print("Saving the best checkpoint without training doesn't work")
        else:
            torch.save(self.best_checkpoint, path)
