import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import perfscores

sns.set_theme(palette="colorblind", style="whitegrid")

# -----------------------------
# USER INPUT
# -----------------------------
MODEL_NAME = "GRU_Baseline_Denseweight_alpha10"
INPUT_FILE = "saved_models/GRU_Baseline_Denseweight_alpha10/cv_1_GRU_Baseline_Denseweight_alpha10_all_folds.parquet"
OUT_DIR = "/scratch/mch/tkluser/rainforest_semester_project/saved_models/cv_1_GRU_Baseline_Denseweight_alpha10_all_folds/plots"

os.makedirs(OUT_DIR, exist_ok=True)

EST_COL = "y_pred"
REF_COL = "y_obs"

# -----------------------------
# Helpers
# -----------------------------
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_parquet(path)

    # Standardize split identification
    if "split" in df.columns:
        df["__split__"] = df["split"].astype(str).str.lower()
    else:
        # Fallback for your Logger.py structure
        df["__split__"] = "test" 
    
    df["__fold__"] = df["fold"].astype(str)
    return df

def compute_metrics(df, bounds):
    rows = []
    for split in df["__split__"].unique():
        dsplit = df[df["__split__"] == split]
        for fold, d in dsplit.groupby("__fold__"):
            metrics_by_bound = perfscores(
                d[EST_COL].to_numpy(),
                d[REF_COL].to_numpy(),
                bounds=bounds,
            )
            for bound_key, mdict in metrics_by_bound.items():
                for metric_name, val in mdict.items():
                    rows.append({
                        "fold": fold,
                        "split": split,
                        "bound": bound_key,
                        "metric": metric_name,
                        "value": float(val) if np.isfinite(val) else np.nan,
                    })
    return pd.DataFrame(rows)

# -----------------------------
# New/Updated Plots
# -----------------------------
def plot_fold_variance(dfm, split, metric, bounds_order):
    """Shows how the metric varies across folds using a swarm/box plot."""
    d = dfm[(dfm["split"] == split) & (dfm["metric"] == metric) & (dfm["bound"].isin(bounds_order))]
    if d.empty: return

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=d, x="bound", y="value", order=bounds_order, color="lightblue", showfliers=False)
    sns.swarmplot(data=d, x="bound", y="value", order=bounds_order, color="black", alpha=0.6)
    
    plt.title(f"{MODEL_NAME} - {metric} Stability across Folds ({split.upper()})")
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(os.path.join(OUT_DIR, f"fold_variance_{metric}_{split}.png"), dpi=200)
    plt.close()

def plot_residual_analysis(df, split):
    """New Plot: Residuals (Error) vs Reference Intensity."""
    d = df[df["__split__"] == split].copy()
    d["error"] = d[EST_COL] - d[REF_COL]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(d[REF_COL], d["error"], alpha=0.1, s=1)
    plt.axhline(0, color='red', linestyle='--')
    plt.xscale('log')
    plt.xlabel("Reference Precipitation (log scale)")
    plt.ylabel("Prediction Error (Pred - Obs)")
    plt.title(f"{MODEL_NAME} - Residual Analysis ({split.upper()})")
    plt.savefig(os.path.join(OUT_DIR, f"residuals_{split}.png"), dpi=200)
    plt.close()

def plot_density_scatter(df, split):
    """Heatmap scatter plot of Ref vs Pred."""
    d = df[df["__split__"] == split]
    plt.figure(figsize=(8, 8))
    plt.hexbin(d[REF_COL], d[EST_COL], bins='log', cmap='viridis', gridsize=50)
    
    max_val = max(d[REF_COL].max(), d[EST_COL].max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
    
    plt.xlabel("Observed (mm)")
    plt.ylabel("Predicted (mm)")
    plt.title(f"{MODEL_NAME} - Density Scatter ({split.upper()})")
    plt.colorbar(label='log10(count)')
    plt.savefig(os.path.join(OUT_DIR, f"density_scatter_{split}.png"), dpi=200)
    plt.close()

def plot_calibration(df, split):
    """
    Plots a Calibration/Reliability Diagram.
    Compares mean predicted vs. mean observed across intensity deciles.
    """
    d = df[df["__split__"] == split].copy()
    
    # Create 10 bins based on reference data quantiles (deciles)
    # We filter out 0s to focus on active precipitation
    active = d[d[REF_COL] > 0.1].copy()
    if active.empty: return
    
    active['bin'] = pd.qcut(active[REF_COL], q=10, duplicates='drop')
    
    # Calculate mean obs and mean pred per bin
    calibration = active.groupby('bin', observed=True).agg({
        REF_COL: 'mean',
        EST_COL: 'mean'
    }).reset_index()

    plt.figure(figsize=(8, 8))
    plt.plot(calibration[REF_COL], calibration[EST_COL], marker='o', linewidth=2, label='Model Performance')
    
    # 1:1 Reference Line
    max_val = max(calibration[REF_COL].max(), calibration[EST_COL].max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Calibration')
    
    plt.xlabel("Mean Observed Precipitation (mm)")
    plt.ylabel("Mean Predicted Precipitation (mm)")
    plt.title(f"{MODEL_NAME} - Calibration Plot ({split.upper()})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(OUT_DIR, f"calibration_{split}.png"), dpi=200)
    plt.close()

# -----------------------------
# Run Execution
# -----------------------------
print(f"Loading data for {MODEL_NAME}...")
df_all = load_data(INPUT_FILE)

bounds = [0, 1, 10, np.inf]
bounds_order = ["all", "0.0-1.0", "1.0-10.0", "10.0-inf"]

print("Computing per-fold metrics...")
df_metrics = compute_metrics(df_all, bounds)
df_metrics.to_csv(os.path.join(OUT_DIR, "detailed_metrics.csv"), index=False)

metrics_to_plot = ["RMSE", "scatter", "logBias", "ED"]

for split in df_all["__split__"].unique():
    print(f"Generating plots for split: {split}...")
    plot_density_scatter(df_all, split)
    plot_residual_analysis(df_all, split)
    plot_calibration(df_all, split)
    for m in metrics_to_plot:
        plot_fold_variance(df_metrics, split, m, bounds_order)

print(f"Success! Analysis saved to {OUT_DIR}")