import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datasets.GRURainSeqDataset import GRURainSeqDataset

# subclasses GRURainDataset
class ScalerAwareGRUDataset(GRURainSeqDataset):
    """
    Extends GRURainSeqDataset to save the StandardScaler and dataset hyperparameters
    during training, or load them during inference.
    """
    def __init__(self, input_dir, cols_to_use, scaler_path, is_training=True, subset=1, weighting=None, alpha_denseweight=1.0, sort_by_height=True, **kwargs):
        self.scaler_path = scaler_path
        self.is_training = is_training
        self.scaler = None
        self.numeric_cols = None
        self.cols_to_use = cols_to_use
        self.dataset_hyperparams = {
            # "subset": subset,
            "weighting": weighting,
            "alpha_denseweight": alpha_denseweight,
            "sort_by_height": sort_by_height,
            **kwargs,
        }

        super().__init__(input_dir, cols_to_use, subset=subset, weighting=weighting, alpha_denseweight=alpha_denseweight, sort_by_height=sort_by_height, **kwargs)

    def save(self, scaler_path=None):
        scaler_path = scaler_path or self.scaler_path
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(
                {
                    'scaler': self.scaler,
                    'numeric_cols': self.numeric_cols,
                    'cols_to_use': self.cols_to_use,
                    'dataset_hyperparams': self.dataset_hyperparams,
                },
                f,
            )

    @classmethod
    def load(cls, input_dir, scaler_path, is_training=False, **override_hyperparams):
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Could not find scaler metadata at {scaler_path}")

        with open(scaler_path, 'rb') as f:
            saved_data = pickle.load(f)

        cols_to_use = saved_data.get('cols_to_use')
        dataset_hyperparams = saved_data.get('dataset_hyperparams', {})
        dataset_hyperparams.update(override_hyperparams)

        return cls(
            input_dir=input_dir,
            cols_to_use=cols_to_use,
            scaler_path=scaler_path,
            is_training=is_training,
            **dataset_hyperparams,
        )

    def prepare_for_gru(self, radar_db):
        radar_sub = pd.get_dummies(radar_db, columns=["RADAR"])
        dummies = [col for col in radar_sub.columns if col.startswith("RADAR_")]

        self.numeric_cols = radar_sub.columns.difference(dummies)
        
        if self.is_training:
            print(f"Fitting new StandardScaler and saving to {self.scaler_path}")
            self.scaler = StandardScaler()
            radar_sub[self.numeric_cols] = self.scaler.fit_transform(radar_sub[self.numeric_cols])
            self.save()
        else:
            print(f"Loading existing StandardScaler from {self.scaler_path}")
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Could not find scaler at {self.scaler_path}")
                
            with open(self.scaler_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.scaler = saved_data['scaler']
                
            # Transform using loaded scaler
            radar_sub[self.numeric_cols] = self.scaler.transform(radar_sub[self.numeric_cols])

        # Convert booleans to int8
        radar_sub = radar_sub.astype(
            {col: "int8" for col in radar_sub.select_dtypes("boolean").columns}
        )
        return radar_sub