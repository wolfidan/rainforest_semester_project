# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import perfscores

# -----------------------------
# Constants
# -----------------------------
BASE_DIR = "/scratch/mch/wolfensb/rainforest_semester_project/saved_models/cv_predictions"
OUT_DIR = os.path.join(BASE_DIR, "fig_performance_GRU_vs_RF_allfolds_with_foldspread")
os.makedirs(OUT_DIR, exist_ok=True)

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


# -----------------------------
# Compute metrics across folds (but from the pooled files)
# -----------------------------
def compute_metrics_table_allfolds(bounds):
    """
    Returns a tidy DataFrame with columns:
    method, split, fold, bound, metric, value
    """
    rows = []
    for method in ["GRU", "RF"]:
        df = load_allfold(method)

        for split in ["train", "test"]:
            dsplit = df[df["__split__"] == split]
            if dsplit.empty:
                continue

            # group by fold -> compute per-fold metrics (spread!)
            for fold, d in dsplit.groupby("__fold__", sort=True):
                metrics_by_bound = perfscores(
                    d[EST_COL].to_numpy(),
                    d[REF_COL].to_numpy(),
                    bounds=bounds,
                )
                for bound_key, mdict in metrics_by_bound.items():
                    for metric_name, val in mdict.items():
                        rows.append(
                            {
                                "method": method,
                                "split": split,
                                "fold": fold,
                                "bound": bound_key,
                                "metric": metric_name,
                                "value": float(val) if np.isfinite(val) else np.nan,
                            }
                        )

    return pd.DataFrame(rows)


# -----------------------------
# Plotting
# -----------------------------
def save_boxplot_compare(df, outdir, split, metric, bounds_order):
    """
    Boxplot of per-fold metric distribution, GRU vs RF, for each bound.
    """
    d = df[
        (df["split"] == split)
        & (df["metric"] == metric)
        & (df["bound"].isin(bounds_order))
    ].copy()
    if d.empty:
        return

    n = len(bounds_order)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, b in zip(axes, bounds_order):
        db = d[d["bound"] == b]
        vals_gru = db.loc[db["method"] == "GRU", "value"].dropna().to_numpy()
        vals_rf  = db.loc[db["method"] == "RF",  "value"].dropna().to_numpy()
        print(b, vals_gru, vals_rf, metric)
        ax.boxplot([vals_gru, vals_rf], labels=["GRU", "RF"], showmeans=True)
        ax.set_title(f"{split.upper()} • ref in [{b}]")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{metric} — per-fold distribution (GRU vs RF)")
    fig.tight_layout()
    fpath = os.path.join(outdir, f"box_{metric}_{split}.png")
    fig.savefig(fpath, dpi=200)
    plt.close(fig)


def save_summary_bar(df, outdir, split, metric, bounds_order):
    """
    Mean ± std across folds, GRU vs RF, grouped by bound.
    """
    d = df[
        (df["split"] == split)
        & (df["metric"] == metric)
        & (df["bound"].isin(bounds_order))
    ].copy()
    if d.empty:
        return

    g = (
        d.groupby(["method", "bound"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    g["bound"] = pd.Categorical(g["bound"], categories=bounds_order, ordered=True)
    g = g.sort_values(["bound", "method"])

    bounds_unique = list(pd.unique(g["bound"]))
    x = np.arange(len(bounds_unique))
    width = 0.35

    def extract(method):
        gg = g[g["method"] == method].set_index("bound").reindex(bounds_unique)
        return gg["mean"].to_numpy(), gg["std"].to_numpy(), gg["count"].to_numpy()

    m_gru, s_gru, c_gru = extract("GRU")
    m_rf,  s_rf,  c_rf  = extract("RF")

    fig, ax = plt.subplots(figsize=(5 + 1.2 * len(bounds_unique), 4))
    ax.bar(x - width / 2, m_gru, width, yerr=s_gru, capsize=3, label=f"GRU (nfold={int(np.nanmax(c_gru))})")
    ax.bar(x + width / 2, m_rf,  width, yerr=s_rf,  capsize=3, label=f"RF (nfold={int(np.nanmax(c_rf))})")

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bounds_unique])
    ax.set_title(f"{metric} — mean ± std across folds ({split.upper()})")
    ax.set_xlabel("Reference precip bound")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fpath = os.path.join(outdir, f"bar_{metric}_{split}.png")
    fig.savefig(fpath, dpi=200)
    plt.close(fig)


def save_method_scatter_allfolds(outdir, split, bound_key):
    """
    ref-vs-est scatter for GRU/RF combined across folds (pooled),
    filtered by split and optional ref bound.
    """
    fig, ax = plt.subplots(2,1, sharex=True, sharey=True, figsize=(5, 10))

    for i,method in enumerate(["GRU", "RF"]):
        df = load_allfold(method)
        d = df[df["__split__"] == split].copy()
        if d.empty:
            continue
        est = d[EST_COL].to_numpy()
        ref = d[REF_COL].to_numpy()

        # apply bound on ref
        if bound_key != "all":
            lo, hi = bound_key.split("-")
            lo = float(lo)
            hi = float(hi) if hi != "inf" else np.inf
            m = (ref >= lo) & (ref < hi)
            est = est[m]
            ref = ref[m]

        m2 = (est > 0) & (ref > 0) & np.isfinite(est) & np.isfinite(ref)
        est = est[m2]
        ref = ref[m2]

        ax[i].hexbin(ref, est, bins="log", mincnt=1)

        # 1:1 line
        xlim = ax[i].get_xlim()
        ylim = ax[i].get_ylim()
        lo = max(min(xlim[0], ylim[0]), 1e-3)
        hi = max(xlim[1], ylim[1])
        ax[i].plot([lo, hi], [lo, hi])
        
        ax[i].set_xlabel("Reference precip")
        ax[i].set_ylabel("Estimated precip")
        ax[i].set_title(f"{method}: {split.upper()} scatter (bound={bound_key})")
        ax[i].grid(True, which="both", alpha=0.2)
        ax[i].legend()

    fig.tight_layout()
    fpath = os.path.join(outdir, f"scatter_{split}_{bound_key.replace('inf','Inf')}.png")
    fig.savefig(fpath, dpi=200)
    plt.close(fig)

#%%
# -----------------------------
# Run
# -----------------------------
bounds = [0, 1, 10, np.inf]
bounds_order = ["all", "0.0-1.0", "1.0-10.0", "10.0-inf"]

dfm = compute_metrics_table_allfolds(bounds=bounds)

if dfm.empty:
    raise RuntimeError("No metrics computed. Check that split/fold columns exist and include train/test data.")

# Save metrics table
dfm.to_csv(os.path.join(OUT_DIR, "metrics_by_fold_from_allfold_files.csv"), index=False)

#%%
metrics_to_plot = ["RMSE", "scatter", "logBias", "ED"]

for split in ["train", "test"]:
    for metric in metrics_to_plot:
        save_boxplot_compare(dfm, OUT_DIR, split=split, metric=metric, bounds_order=bounds_order)
        save_summary_bar(dfm, OUT_DIR, split=split, metric=metric, bounds_order=bounds_order)

#%%
# Optional: pooled scatter per split/bound (aggregated across folds)
for split in ["train", "test"]:
    # We only do scatter for all values (no bounds)
    save_method_scatter_allfolds(OUT_DIR, split=split, bound_key="all")

print(f"Saved figures + metrics table to: {OUT_DIR}")
print("Generated:")
print(" - box_<metric>_<split>.png (per-fold distribution, fold from column)")
print(" - bar_<metric>_<split>.png (mean±std across folds)")
print(" - scatter_<split>_<bound>.png (pooled scatter)")
print(" - metrics_by_fold_from_allfold_files.csv (tidy table)")
