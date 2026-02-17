# Comparison of RF and GRU for QPE

## List of scripts

- *rf.py*: implements the random forest class as used in RainForest
- *gru.py*: possible GRU implementation of the QPE problem in pytorch, using only vertical columns above the central pixel
- *crossval_example.py*: example of K-fold cross-validation of the RF and GRU models. It saves final trained models as well as test and train predictions in the directory *saved_models*.
- *crossval_example.job*: the slurm job for the cross-validation script, to be run with `sbatch crossval_example.job`
- *crossval_analyse_results.py*: script that analyses the results of the cross-validation and outputs scores and plots.
- *utils.py*: some utilities, currently only the methods to compute performance metrics.