import matplotlib.pyplot as plt
import pandas as pd
import os
from utils import load_allfold
import numpy as np


# script that compares the predictions of two models in a hexbin plot


MODEL_1_NAME = "GRU_Bidirectional"
MODEL_1_PATH = "/scratch/mch/tkluser/rainforest_semester_project/saved_models/GRU_Bidirectional/cv_1_GRU_Bidirectional_all_folds.parquet"

MODEL_2_NAME = "RF_noPostProcess"
MODEL_2_PATH = "/scratch/mch/tkluser/rainforest_semester_project/saved_models/cv_predictions/cv_pred_RF_noPostProcess_all_folds.parquet"

OUT_DIR = "/scratch/mch/tkluser/rainforest_semester_project/saved_models/GRU_Baseline_diverseOptim2/comparison_GRUOptim_BL"
OUT_DIR = "/scratch/mch/tkluser/rainforest_semester_project/saved_models/GRU_Bidirectional"
os.makedirs(OUT_DIR, exist_ok=True)

def split_and_group(df: pd.DataFrame, split="test"):
    df = df[df["__split__"] == split]
    df = df.groupby("vertgroup").agg(pred_mean = ("y_pred", "mean")).reset_index()
    return df

def save_method_scatter_allfolds(outdir, split):
    """
    ref-vs-est scatter for GRU/RF combined across folds (pooled),
    filtered by split and optional ref bound.
    """
    fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(10, 10))

    df_res1 = load_allfold(MODEL_1_PATH)
    df_res2 = load_allfold(MODEL_2_PATH)

    df_res1 = split_and_group(df_res1, split)
    df_res2 = split_and_group(df_res2, split)

    
    df_res = pd.merge(df_res1, df_res2, how="inner", on="vertgroup")

    pred_model1 = df_res["pred_mean_x"].to_numpy()
    pred_model2 = df_res["pred_mean_y"].to_numpy()

    # m2 = (est > 0) & (ref > 0) & np.isfinite(est) & np.isfinite(ref)
    # est = est[m2]
    # ref = ref[m2]

    hb = ax.hexbin(pred_model1, pred_model2, bins="log", mincnt=1)

    # 1:1 line
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lo = max(min(xlim[0], ylim[0]), 1e-3)
    hi = max(xlim[1], ylim[1])
    ax.plot([lo, hi], [lo, hi], "r--")
    
    ax.set_xlabel(f"{MODEL_1_NAME} (mm/10min)")
    ax.set_ylabel(f"{MODEL_2_NAME} (mm/10min)")
    ax.set_title(f"Predicted precip., {MODEL_1_NAME} vs. {MODEL_2_NAME}, {split.upper()} split")
    ax.grid(True, which="both", alpha=0.2)
    fig.colorbar(hb, ax=ax, label='counts')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.legend()

    fig.tight_layout()
    fpath = os.path.join(outdir, f"scatter_{MODEL_1_NAME}_vs_{MODEL_2_NAME}_{split}.png")
    fig.savefig(fpath, dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    save_method_scatter_allfolds(OUT_DIR, split="test")
    save_method_scatter_allfolds(OUT_DIR, split="train")