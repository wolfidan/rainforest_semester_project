# Rainforest Semester Project

## Project Overview

This repository compares Random Forest (RF) and Gated Recurrent Unit (GRU) models for quantitative precipitation estimation (QPE) using radar-based feature data. It contains training, inference, cross-validation, and analysis scripts for both classical machine learning and deep learning approaches.

The project is structured around:

- RF modeling and benchmarking
- GRU sequence modeling for vertical atmospheric columns
- Cross-validation and performance analysis
- Dataset preparation and scaling utilities
- Hyperparameter tuning and experiment tracking

## Repository Structure

- `full_model_training.py` - Train a final GRU model on the full dataset and save model artifacts.
- `full_model_inference.py` - Run inference using the final saved GRU model and save predictions.
- `train_gru_kfold.py` - Perform K-fold cross-validation for GRU-based sequence models.
- `requirements.txt` - Python dependencies for the project.
- `analyse_models/` - Analysis and plotting utilities for evaluation results.
- `datasets/` - Dataset loaders and preprocessing utilities used by GRU models.
- `helper/` - Helper utilities such as training loggers.
- `models/` - Model definitions for GRU variations.
- `saved_models/` - Directory used to store trained models, checkpoints, logs, and plots.
- `tuning_GRU/` - Hyperparameter tuning experiments and Optuna results.

## Core Components

### Models

- `models/gru_baseline.py` - Baseline GRU model implementation with training, validation, and prediction support.
- `models/gru_bidirectional.py` - Bidirectional GRU variant used for K-fold experiments.
- `daniel/rf.py` - Random Forest implementation used for RainForest baseline comparisons.

### Datasets

- `datasets/GRURainSeqDataset.py` - Dataset loader for GRU training with sequence handling, feature selection, and weighting.
- `datasets/ScalerAwareGRUDataset.py` - Dataset utility that stores scaling parameters and supports scaled training and inference.
- `datasets/datasetUtils.py` - Auxiliary dataset utilities used across scripts.

### Analysis

- `analyse_models/crossval_analyse_results_comparison.py` - Compare cross-validation results across models.
- `analyse_models/plot_*.py` - Plotting scripts for diagnostics, scatter plots, map visualizations, and precipitation distributions.

## Getting Started

### Requirements

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

### Data

Most scripts expect radar input data in an absolute path such as:

- `/store_new/mch/msrad/radar/radar_database_v2/rf_input_data_qc/`

Update `INPUT_DIR` in the relevant script before running if your data are stored elsewhere.


### Full model training

Train a final GRU model on the complete dataset:

```bash
python full_model_training.py
```

This script:
- prepares a scaled GRU-compatible dataset
- builds a GRU model with saved hyperparameters
- trains the model with early stopping
- saves the best checkpoint to `saved_models/GRU_Final_AllData`

### Inference

After training, run inference with the saved GRU model:

```bash
python full_model_inference.py
```

This will load the saved model and scaler, run predictions on the dataset, and write results to `saved_models/GRU_Final_AllData/inference_predictions.csv`.

## Notes

- Many scripts use hardcoded paths for input data and output directories. Update those paths to match your environment.
- Analysis scripts in `analyse_models/` expect prediction and log output from training jobs that are logged in the same format as the `helper/Logger`does.

## Contact

For questions about this project contact Tim Kluser (tim.kluser@gmail.com)
