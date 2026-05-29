import optuna

OUTPUT_DIR = "tuning_GRU"
STUDY_NAME = "tune_gru_diverseOptim3"

study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"], 
        study_name=STUDY_NAME,
        storage=f"sqlite:///{OUTPUT_DIR}/optuna_tune.db",
        load_if_exists=True
    ) 
    

# For multi-objective, we look at the Pareto Front
best_trials = study.best_trials
print(f"Number of Pareto optimal trials: {len(best_trials)}")

for i, trial in enumerate(best_trials):
    print(f"\nPareto Trial {i}:")
    print(f"  Trial Number: {trial.number}")
    print(f"  Values: {trial.values}")
    print(f"  Params: {trial.params}")

# fig = optuna.visualization.plot_pareto_front(
#     study,
#     targets=lambda t: (t.values[0], t.values[1], t.values[2]),
#     target_names=["Objective 0", "Objective 1", "Objective 2"],
# )

# fig.write_image("paretoStudy3.html")