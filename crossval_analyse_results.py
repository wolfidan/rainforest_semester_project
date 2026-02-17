# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import perfscores

# -----------------------------
# Constants
# -----------------------------

BASE_DIR = "/scratch/mch/wolfensb/rainforest_semester_project/saved_models/cv_predictions"
OUT_DIR = os.path.join(BASE_DIR, "fig_performance_GRU_vs_RF")
os.makedirs(OUT_DIR, exist_ok=True)

EST_COL = "y_pred"  # e.g. "y_pred"
REF_COL = "y_obs"  # e.g. "y_true"

# -----------------------------
# Helpers: file discovery + column detection
# -----------------------------

def discover_files(base_dir):
    """
    Returns a list of dict entries: {method, fold, split, path}
    split in {"train","test"}; fold like "fold01".
    """
    pat = re.compile(r"cv_pred_(GRU|RF)_(fold\d{2})_(train|test)\.parquet$")
    entries = []
    for fn in os.listdir(base_dir):
        m = pat.match(fn)
        if m:
            method, fold, split = m.group(1), m.group(2), m.group(3)
            entries.append(
                {
                    "method": method,
                    "fold": fold,
                    "split": split,
                    "path": os.path.join(base_dir, fn),
                }
            )
    # Sort nicely
    entries.sort(key=lambda d: (d["method"], d["split"], d["fold"]))
    return entries


# -----------------------------
# Compute metrics across folds
# -----------------------------


def compute_metrics_table(entries, bounds):
    """
    Returns a tidy DataFrame with columns:
    method, split, fold, bound, metric, value
    """
    rows = []
    for e in entries:
        df = pd.read_parquet(e["path"])
        metrics_by_bound = perfscores(
            df[EST_COL].to_numpy(), df[REF_COL].to_numpy(), bounds=bounds
        )
        for bound_key, mdict in metrics_by_bound.items():
            for metric_name, val in mdict.items():
                rows.append(
                    {
                        "method": e["method"],
                        "split": e["split"],
                        "fold": e["fold"],
                        "bound": bound_key,
                        "metric": metric_name,
                        "value": float(val) if np.isfinite(val) else np.nan,
                    }
                )

    out = pd.DataFrame(rows)
    return out


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

    # Make a 1-row panel with one subplot per bound
    n = len(bounds_order)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, b in zip(axes, bounds_order):
        db = d[d["bound"] == b]
        # Values by method
        vals_gru = db.loc[db["method"] == "GRU", "value"].dropna().to_numpy()
        vals_rf = db.loc[db["method"] == "RF", "value"].dropna().to_numpy()

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

    # Aggregate
    g = (
        d.groupby(["method", "bound"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Ensure bound order
    g["bound"] = pd.Categorical(g["bound"], categories=bounds_order, ordered=True)
    g = g.sort_values(["bound", "method"])

    bounds_unique = list(pd.unique(g["bound"]))
    x = np.arange(len(bounds_unique))
    width = 0.35

    def extract(method):
        gg = g[g["method"] == method].set_index("bound").reindex(bounds_unique)
        return gg["mean"].to_numpy(), gg["std"].to_numpy()

    m_gru, s_gru = extract("GRU")
    m_rf, s_rf = extract("RF")

    fig, ax = plt.subplots(figsize=(5 + 1.2 * len(bounds_unique), 4))
    ax.bar(x - width / 2, m_gru, width, yerr=s_gru, capsize=3, label="GRU")
    ax.bar(x + width / 2, m_rf, width, yerr=s_rf, capsize=3, label="RF")

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


def save_method_scatter(df, outdir, split, bound_key, base_dir=None):
    """
    Optional: For the *test* split you might want a classic ref-vs-est scatter for GRU/RF combined across folds.
    This loads the parquets again (so needs base_dir) and plots log-log ref vs est for each method.
    """
    if base_dir is None:
        return

    # Identify files for this split/method for all folds
    entries = discover_files(base_dir)
    entries = [e for e in entries if e["split"] == split]

    fig, ax = plt.subplots(figsize=(5, 5))

    for method in ["GRU", "RF"]:
        est_all = []
        ref_all = []
        for e in entries:
            if e["method"] != method:
                continue
            dfp = pd.read_parquet(e["path"])
            est = dfp[EST_COL].to_numpy()
            ref = dfp[REF_COL].to_numpy()

            # apply bound on ref
            if bound_key != "all":
                lo, hi = bound_key.split("-")
                lo = float(lo)
                hi = float(hi) if hi != "inf" else np.inf
                m = (ref >= lo) & (ref < hi)
                est = est[m]
                ref = ref[m]

            m2 = (est > 0) & (ref > 0) & np.isfinite(est) & np.isfinite(ref)
            est_all.append(est[m2])
            ref_all.append(ref[m2])

        if len(est_all) == 0:
            continue
        est_all = np.concatenate(est_all) if est_all else np.array([])
        ref_all = np.concatenate(ref_all) if ref_all else np.array([])

        ax.scatter(ref_all, est_all, s=4, alpha=0.15, label=method)

    # 1:1 line
    lims = ax.get_xlim()
    lo = max(min(lims[0], ax.get_ylim()[0]), 1e-3)
    hi = max(lims[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reference precip")
    ax.set_ylabel("Estimated precip")
    ax.set_title(f"{split.upper()} scatter (bound={bound_key})")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()

    fig.tight_layout()
    fpath = os.path.join(
        outdir, f"scatter_{split}_{bound_key.replace('inf','Inf')}.png"
    )
    fig.savefig(fpath, dpi=200)
    plt.close(fig)


bounds = [0, 1, 10, np.inf]
# Labels produced by perfscores are "0-1", "1-10", "10-inf" (and always "all")
bounds_order = ["all", "0-1", "1-10", "10-inf"]

entries = discover_files(BASE_DIR)
if len(entries) == 0:
    raise FileNotFoundError(f"No fold train/test parquet files found in {BASE_DIR}")

dfm = compute_metrics_table(entries, bounds=bounds)

# Save metrics table for later
dfm.to_csv(os.path.join(OUT_DIR, "metrics_by_fold.csv"), index=False)

# Metrics to plot (focus on your key ones)
metrics_to_plot = ["RMSE", "scatter", "logBias", "ED"]

for split in ["train", "test"]:
    for metric in metrics_to_plot:
        save_boxplot_compare(
            dfm, OUT_DIR, split=split, metric=metric, bounds_order=bounds_order
        )
        save_summary_bar(
            dfm, OUT_DIR, split=split, metric=metric, bounds_order=bounds_order
        )

# Optional: ref-vs-est scatter for TEST for each bound (aggregated over folds)
for b in ["all", "0-1", "1-10", "10-inf"]:
    save_method_scatter(dfm, OUT_DIR, split="test", bound_key=b, base_dir=BASE_DIR)

print(f"Saved figures + metrics table to: {OUT_DIR}")
print("Generated:")
print(" - box_<metric>_<split>.png (per-fold distribution)")
print(" - bar_<metric>_<split>.png (mean±std across folds)")
print(" - scatter_test_<bound>.png (optional aggregated scatter)")
print(" - metrics_by_fold.csv (tidy table)")


# %%
