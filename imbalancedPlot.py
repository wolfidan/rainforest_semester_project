
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns
sns.set_theme(palette="colorblind", style="whitegrid")


# from utils import perfscores



### code from daniel
# -----------------------------
# Constants
# -----------------------------
BASE_DIR = "/scratch/mch/wolfensb/rainforest_semester_project/saved_models/cv_predictions"
OUT_DIR = os.path.abspath("/scratch/mch/tkluser/rainforest_semester_project/imblanced_plots")
# OUT_DIR = os.path.join(BASE_DIR, "fig_performance_GRU_vs_RF_allfolds_with_foldspread")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL = "RF"

EST_COL = "y_pred"
REF_COL = "y_obs"

ALLFOLD_FILES = {
    "GRU": os.path.join(BASE_DIR, "cv_pred_GRU_all_folds.parquet"),
    "RF":  os.path.join(BASE_DIR, "cv_pred_RF_all_folds.parquet"),
}

# -----------------------------
# Helpers
# -----------------------------
def load_allfold(method: str) -> pd.DataFrame:
    path = ALLFOLD_FILES[method]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_parquet(path)

    # --- split detection ---
    if "split" in df.columns:
        split_col = "split"
        df["__split__"] = df[split_col].astype(str).str.lower()
    elif "is_train" in df.columns:
        df["__split__"] = np.where(df["is_train"].astype(bool), "train", "test")
    elif "dataset" in df.columns:
        df["__split__"] = df["dataset"].astype(str).str.lower()
    else:
        raise ValueError(
            f"{path}: can't find a split indicator column. "
            "Expected one of: 'split', 'is_train', 'dataset'. "
            f"Columns are: {list(df.columns)}"
        )

    # enforce train/test presence (at least one)
    allowed = {"train", "test"}
    present = set(df["__split__"].dropna().unique())
    if not (present & allowed):
        raise ValueError(f"{path}: detected split values={present}, but no train/test found.")

    # --- fold column ---
    if "fold" not in df.columns:
        raise KeyError(f"{path}: missing required column 'fold'. Columns are: {list(df.columns)}")

    # normalize fold to string labels (keeps original values but consistent)
    df["__fold__"] = df["fold"].astype(str)

    # --- required columns ---
    for c in [EST_COL, REF_COL]:
        if c not in df.columns:
            raise KeyError(f"{path}: missing required column '{c}'. Columns are: {list(df.columns)}")

    return df


# Plot that shows imbalance in precipitation predictions
def precip_imbalance_plot(df:pd.DataFrame):

    # generate bins 
    bounds = (df["y_obs"].min()-0.01, df["y_obs"].max()+0.01)
    nr_bins = 10
    print(f'Min and max for generation of bins: {bounds}')

    bins = np.linspace(bounds[0], bounds[1], nr_bins)
    df["bin"] = np.digitize(df["y_obs"], bins)
    df["bin_center"] = df["bin"].map(lambda x: bins[x-1] + (bins[x] - bins[x-1])/2)
    
    # mean squared error over all the samples of every cross-val test split that fall into a bin
    # df = df.groupby("bin_center").agg(mse = ("squared_err", "mean")).reset_index()
    # plt.plot(df["bin_center"], df["mse"])

    sns.lineplot(df, x="bin_center", y="squared_err", estimator="mean")
    plt.title("MSE of predicted precipitation")
    plt.xlabel("True Precipitation [mm]")
    plt.ylabel("MSE")
    locs, _ = plt.xticks()
    plt.xticks(locs, [f'{x:.1f}' for x in locs])
    

    fpath = os.path.join(OUT_DIR, f"imbalancedPlot{MODEL}.png")
    plt.savefig(fpath, dpi=200)

    # df.to_csv(os.path.join(OUT_DIR, "temp.csv"))


def wind_imbalance_plot():
    # plot 
    pass



if __name__ == "__main__":
    df = load_allfold(MODEL)

    # df = df.sample(1000) # TODO:

    df = df[df["split"] == "train"]
    
    df["squared_err"] = pow(df["y_obs"]-df["y_pred"], 2)

    precip_imbalance_plot(df)
    


