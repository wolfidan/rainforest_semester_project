import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_training_stats(stats_path, output_dir, metric="mse"):
    """
    Reads the training statistics parquet file, plots loss curves, 
    and returns summary metrics.
    """
    if not os.path.exists(stats_path):
        print(f"File not found: {stats_path}")
        return None

    # 1. Load the data
    df = pd.read_parquet(stats_path)
    
    # 2. Setup Plotting
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    # 3. Plot Training vs Validation Loss
    # We use lineplot which automatically calculates mean and confidence intervals across folds
    plot = sns.lineplot(
        data=df, 
        x="epoch", 
        y=metric, 
        hue="split", 
        style="split",
        markers=True, 
        dashes=False
    )
    
    plt.title(f"Training and Validation Loss across {df['fold'].nunique()} Folds")
    plt.xlabel("Epoch")

    ylabel = "Mean Squared Error (MSE)"
    if metric == "mse_weighted":
        ylabel = "Weighted Mean Squared Error (MSE)"
    plt.ylabel(ylabel)
    plt.yscale("log")  # Often helpful for loss curves
    plt.legend(title="Split")

    fpath = os.path.join(output_dir, f"train_val_stats_{metric}.png")
    plt.savefig(fpath, dpi=200)

    # 4. Calculate Interesting Values
    summary = {}
    
    # Final loss per split (averaged across folds)
    last_epoch = df['epoch'].max()
    final_stats = df[df['epoch'] == last_epoch].groupby('split')[metric].mean()
    summary['avg_final_train_loss'] = final_stats.get('train', None)
    summary['avg_final_val_loss'] = final_stats.get('val', None)

    # Best (minimum) validation loss achieved per fold
    best_val_per_fold = df[df['split'] == 'val'].groupby('fold')[metric].min()
    summary['mean_best_val_loss'] = best_val_per_fold.mean()
    summary['std_best_val_loss'] = best_val_per_fold.std()
    
    # Find the epoch where validation loss was lowest on average
    mean_val_by_epoch = df[df['split'] == 'val'].groupby('epoch')[metric].mean()
    summary['optimal_epoch'] = mean_val_by_epoch.idxmin()

    return summary

# Example Usage:
# stats_file = "/scratch/mch/tkluser/rainforest_semester_project/saved_models/cv_1_GRU_Baseline_training_stats.parquet"
# results = analyze_training_results(stats_file)
# print("Summary Statistics:", results)