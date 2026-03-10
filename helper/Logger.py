import numpy as np
import pandas as pd
import os
from helper.analyse_training_metrics import analyze_training_stats


class Logger():
    # helper class to log statistics about training and evaluation
    def __init__(self, outdir, model_name, filename_prefix):
        self.outdir = outdir
        self.model_name = model_name
        self.filename_prefix = filename_prefix
        self.curr_fold:int = -1

        self.eval_preds_per_fold = []
        self.training_stats = []
    
    def set_curr_fold(self, curr_fold: int):
        self.curr_fold = curr_fold

    def log_eval_pred(
        self,
        split: str,  # "train" or "test"
        vertgroup_ids: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fold: int = None,
    ) -> pd.DataFrame:
        """
        Create a tidy dataframe with observed vs predicted values - for evaluation metrics.
        Indexing in this project:
        - y_true is indexed by vertgroup_id (group-level target)
        - y_pred is aligned with vertgroup_ids (same order)
        """

        if fold == None:
            fold = self.curr_fold

        self.eval_preds_per_fold.append(pd.DataFrame(
            {
                "model": self.model_name,
                "fold": fold,
                "split": split,
                "vertgroup": vertgroup_ids,
                "y_obs": y_true,
                "y_pred": y_pred,
                "error": y_pred - y_true,
            }
        ))



    def log_training_epoch(
        self,
        split: str,  # "train" or "val",
        epoch: int,
        fold: int = None, # if None, take the objects curr fold state
        ** kwargs, # for all kinds of loss fuctions, e.g. mse = 0.5
    ) -> pd.DataFrame:
        """
        Create a tidy dataframe for training statistics.
    
        """

        if fold == None:
            fold = self.curr_fold

        data = {
                "model": [self.model_name],
                "fold": [fold],
                "split": [split],
                "epoch": [epoch],
            }
        
        for key, value in kwargs.items():
            data[key] = [value]

        self.training_stats.append(pd.DataFrame(
            data
        ))
        print(f'Finished epoch {epoch}, writing stats to file successful.')

    def write_to_file(
            self,
            outdir: str,
        ) -> None:
        """
        Save:
        - per-fold parquet files
        - a combined parquet file
        """
        os.makedirs(outdir, exist_ok=True)


        # save evaluation metrics
        # Save per-fold
        
        for df in self.eval_preds_per_fold:
            if len(df["fold"]) > 0:
                fold = int(df["fold"].iloc[0])
            else:
                fold = 0
            split = str(df["split"].iloc[0])
            fpath = os.path.join(
                outdir, f"{self.filename_prefix}_{self.model_name}_fold{fold:02d}_{split}.parquet"
            )
            df.to_parquet(fpath, index=False)

        # Save combined
        combined = pd.concat(self.eval_preds_per_fold, ignore_index=True)
        combined_path = os.path.join(
            outdir, f"{self.filename_prefix}_{self.model_name}_all_folds.parquet"
        )
        combined.to_parquet(combined_path, index=False)

        # save training metrics
        combined_train_df = pd.concat(self.training_stats, ignore_index=True)
        combined_path_train = os.path.join(
            outdir, f"{self.filename_prefix}_{self.model_name}_training_stats.parquet"
        )
        combined_train_df.to_parquet(combined_path_train, index=False)

        print(f"Training and evaluation statistics successfully saved to: {outdir}")

    def generate_training_metrics(self, output_dir):
        combined_path_train = os.path.join(
            self.outdir, f"{self.filename_prefix}_{self.model_name}_training_stats.parquet"
        )
        results = analyze_training_stats(combined_path_train, output_dir=output_dir)
        print("Summary Statistics:", results) 