import matplotlib.pyplot as plt
import numpy as np
import torch
from GRURainSeqDataset import GRURainSeqDataset # Assuming this is available in your environment
from denseweight import DenseWeight



def plot_label_weight_relationship(dataset, output_path="label_weight_dist.png", denseweight_alpha = 1):
    """
    Plots the relationship between labels (precipitation) and their training weights
    with an overlaid histogram of label frequency.
    """
    # 1. Extract targets and weights
    if hasattr(dataset, 'targets') and hasattr(dataset, 'weights'):
        targets = dataset.targets.cpu().numpy()
        weights = dataset.weights.cpu().numpy()
    else:
        targets = []
        weights = []
        print("Extracting weights from dataset...")
        for i in range(len(dataset)):
            data_tuple = dataset[i]
            targets.append(data_tuple[2])
            weights.append(data_tuple[3])
        
        targets = np.array([t.item() if torch.is_tensor(t) else t for t in targets])
        weights = np.array([w.item() if torch.is_tensor(w) else w for w in weights])

    # create weights directly with Denseweights

    dw = DenseWeight(alpha=denseweight_alpha) # alpha = 0 for uniform sampling
    weights = dw.fit(targets)
    precip_plot = np.arange(0, np.max(targets), 0.1)
    weigths_plot = dw(precip_plot)

    print(f'sum of all weights: {sum(weights)}')
    print(f'Mean of all weights: {np.mean(weights)}')
    print(f'Minimum weight value: {min(weights)}')

    
    # 2. Create the Plot
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- Plotting the Weights (Primary Y-axis) ---
    # color_weights = 'teal'
    # ax1.scatter(targets, weights, alpha=0.3, s=10, color=color_weights, label='Samples (Weights)')
    
    # Sort for the trend line
    # sorted_idx = np.argsort(targets)
    ax1.plot(precip_plot, weigths_plot, color='red', alpha=0.8, linewidth=2, label='Weighting Trend')
    
    ax1.set_xlabel("Label Value (Precipitation in $mm$)")
    ax1.set_ylabel("Weight Assigned")
    # ax1.tick_params(axis='y', labelcolor=color_weights)
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # --- Plotting the Histogram (Secondary Y-axis) ---
    ax2 = ax1.twinx()  # Create a second axes that shares the same x-axis
    color_hist = 'gray'
    # bins='auto' or a fixed number like 50
    ax2.hist(targets, bins=50, color=color_hist, alpha=0.2, label='Target Distribution (Freq)', log=True)
    ax2.set_ylabel("Frequency (Count)", color=color_hist)
    ax2.tick_params(axis='y', labelcolor=color_hist)

    # Title and Layout
    a = r'$\alpha$'
    plt.title(f"Precipitation Labels: Training Weights and Data Distribution (denseweight {a}={denseweight_alpha:.1f})")
    
    # Combined legend for both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='center right')

    # ax1.set_xscale('linear') 
    ax1.set_xscale('symlog')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Plot saved successfully to: {output_path}")
    plt.close()

if __name__ == "__main__":
    # Ensure paths and dataset class are correctly defined for your local environment
    INPUT_DIR = "/store_new/mch/msrad/radar/radar_database_v2/rf_input_data/"
    dataset = GRURainSeqDataset(INPUT_DIR, ["RADAR", "HEIGHT"], subset=1.0)
    plot_label_weight_relationship(dataset, output_path="plots/weighting_plot_alpha05.png", denseweight_alpha=0.5)