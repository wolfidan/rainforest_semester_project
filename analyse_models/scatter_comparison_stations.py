import matplotlib.pyplot as plt
import pandas as pd
import os
from utils import load_allfold
import numpy as np
import seaborn as sns

# script that compares the predictions of two models in a scatterplot, grouped by station

MODEL_1_QC = True
MODEL_1_NAME = "GRU_Bidirectional"
MODEL_1_PATH = "/scratch/mch/tkluser/rainforest_semester_project/saved_models/GRU_Bidirectional/cv_1_GRU_Bidirectional_all_folds.parquet"


MODEL_2_QC = True
MODEL_2_NAME = "RF_noPostProcess"
MODEL_2_PATH = "/scratch/mch/tkluser/rainforest_semester_project/saved_models/cv_predictions/cv_pred_RF_noPostProcess_all_folds.parquet"



# we have to be careful with the input data
INPUT_DIR_GAUGEDATA_QC = "/store_new/mch/msrad/radar/radar_database_v2/rf_input_data_qc/"
INPUT_DIR_GAUGEDATA_NoQC = "/store_new/mch/msrad/radar/radar_database_v2/rf_input_data/"

# OUT_DIR = "/scratch/mch/tkluser/rainforest_semester_project/saved_models/GRU_Baseline_Denseweight_alpha10/comparison_Bidirection"
OUT_DIR = "/scratch/mch/tkluser/rainforest_semester_project/saved_models/GRU_Bidirectional/comparison_RF_noPostProcess"

# OUT_DIR = "/scratch/mch/tkluser/rainforest_semester_project/saved_models/GRU_Baseline_diverseOptim2/comparison_GRUOptim_BL"
# OUT_DIR = "/scratch/mch/tkluser/rainforest_semester_project/saved_models/GRU_Baseline_optimized_logBias_scatter/comparison_GRU_Baseline_Denseweight_alpha10"
os.makedirs(OUT_DIR, exist_ok=True)

def merge_with_gauge(df1: pd.DataFrame, df2: pd.DataFrame):
    df_stations = pd.read_csv("/store_new/mch/msrad/radar/rainforest_data/references/metadata/data_stations.csv",
                          encoding='latin1',
                          sep=";")
    gauge_qc = pd.read_parquet(os.path.join(INPUT_DIR_GAUGEDATA_QC, "gauge.parquet"))
    gauge_Noqc = pd.read_parquet(os.path.join(INPUT_DIR_GAUGEDATA_NoQC, "gauge.parquet"))

    # use different gauge data depending on the dataset that the model was trained/tested on
    gauge1 = gauge_qc if MODEL_1_QC else gauge_Noqc
    gauge2 = gauge_qc if MODEL_2_QC else gauge_Noqc

    df1 = pd.merge(df1, gauge1, left_on="vertgroup", right_index=True)
    df2 = pd.merge(df2, gauge2, left_on="vertgroup", right_index=True)

    # agg by station
    df1 = df1.groupby("STATION").agg(pred_mean=("pred_mean", "mean"), ref_mean =("ref_mean", "mean")).reset_index()
    df2 = df2.groupby("STATION").agg(pred_mean=("pred_mean", "mean"), ref_mean =("ref_mean", "mean")).reset_index()

    df_res = pd.merge(df1, df2, on="STATION", how="inner")

    # mean of the measurements
    # df_res = df_res.groupby("STATION").agg(pred_mean_x=("pred_mean_x", "mean"), pred_mean_y=("pred_mean_y", "mean"), ref_mean =("ref_mean_x", "mean")).reset_index()
    df_res = pd.merge(df_res, df_stations, left_on="STATION", right_on="Abbrev")
    return df_res


def split_and_group(df: pd.DataFrame, split="test"):
    df = df[df["__split__"] == split]
    df = df.groupby("vertgroup").agg(pred_mean = ("y_pred", "mean"), ref_mean = ("y_obs", "mean")).reset_index()
    return df

def save_method_scatter_allfolds(outdir, split):
    """
    ref-vs-est scatter for GRU/RF combined across folds (pooled),
    filtered by split and optional ref bound.
    """

    # scatter one model against the other
    fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(10, 10))

    df_res1 = load_allfold(MODEL_1_PATH)
    df_res2 = load_allfold(MODEL_2_PATH)

    # print(df_res1.head())

    df_res1 = split_and_group(df_res1, split)
    df_res2 = split_and_group(df_res2, split)

    
    # df_res = pd.merge(df_res1, df_res2, how="inner", on="vertgroup")

    df_res = merge_with_gauge(df_res1, df_res2)  
    pred_model1 = df_res["pred_mean_x"].to_numpy()
    pred_model2 = df_res["pred_mean_y"].to_numpy()

    sc = ax.scatter(
        data=df_res, 
        x="pred_mean_x", 
        y="pred_mean_y", 
        c="ref_mean_x", 
        cmap="viridis", 
        edgecolor="w", 
        linewidth=0.5
    )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Reference Precip (mm/10min)")

    # ax.plot(pred_model1, pred_model2, 'o')
    # 1:1 line
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lo = max(min(xlim[0], ylim[0]), 1e-3)
    hi = max(xlim[1], ylim[1])
    ax.plot([lo, hi], [lo, hi], "r--")
    
    ax.set_xlabel(f"{MODEL_1_NAME} (mm/10min)")
    ax.set_ylabel(f"{MODEL_2_NAME} (mm/10min)")
    ax.set_title(f"Predicted precip., {MODEL_1_NAME} vs. {MODEL_2_NAME}, {split.upper()} split, mean over stations")
    ax.grid(True, which="both", alpha=0.2)
    # fig.colorbar(hb, ax=ax, label='counts')
    # ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')

    fig.tight_layout()
    fpath = os.path.join(outdir, f"scatter_station_{MODEL_1_NAME}_vs_{MODEL_2_NAME}_{split}.png")
    fig.savefig(fpath, dpi=200)
    plt.close(fig)

    print(f'successfully saved the plot to: {fpath}')



    # scatter models against the true values
    fig, ax = plt.subplots(2,1, sharex=True, sharey=True, figsize=(5, 10))

    for i,method in enumerate([MODEL_1_NAME, MODEL_2_NAME]):
        var_pd = ["x", "y"][i]
        pred_model = df_res[f"pred_mean_{var_pd}"].to_numpy()
        ref = df_res[f"ref_mean_{var_pd}"].to_numpy()

        ax[i].plot(pred_model, ref, 'o')
        # 1:1 line
        xlim = ax[i].get_xlim()
        ylim = ax[i].get_ylim()
        lo = max(min(xlim[0], ylim[0]), 1e-3)
        hi = max(xlim[1], ylim[1])
        ax[i].plot([lo, hi], [lo, hi], "r--")
        
        ax[i].set_xlabel(f"{method} (mm/10min)")
        ax[i].set_ylabel(f"Reference value (mm/10min)")
        ax[i].set_title(f"Predicted precip., {method} vs. Reference, {split.upper()} split, mean over station")
        ax[i].grid(True, which="both", alpha=0.2)
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')

    fig.tight_layout()
    fpath = os.path.join(outdir, f"scatter_station_{split}.png")
    fig.savefig(fpath, dpi=200)
    plt.close(fig)
    print(f'successfully saved the plot to: {fpath}')




if __name__ == "__main__":
    save_method_scatter_allfolds(OUT_DIR, split="test")
    save_method_scatter_allfolds(OUT_DIR, split="train")