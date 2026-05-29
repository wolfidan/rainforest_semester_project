import optuna
import plotly.graph_objects as go

# script that generates a parallel coordinates plots for the performance metrics and the hyperparameters

OUTPUT_DIR = "tuning_GRU"
STUDY_NAME = "tune_gru_diverseOptim3"

study = optuna.create_study(
    directions=["minimize", "minimize", "minimize"], 
    study_name=STUDY_NAME,
    storage=f"sqlite:///{OUTPUT_DIR}/optuna_tune.db",
    load_if_exists=True
)

pareto_trials = study.best_trials

obj0 = [t.values[0] for t in pareto_trials]
obj1 = [t.values[1] for t in pareto_trials]
obj2 = [t.values[2] for t in pareto_trials]

alpha_denseweight = [t.params.get("alpha_denseweight", None) for t in pareto_trials]
learning_rate = [t.params.get("learning_rate", None) for t in pareto_trials]
dropout = [t.params.get("dropout", None) for t in pareto_trials]
num_hidden_layers = [t.params.get("num_hidden_layers", None) for t in pareto_trials]

fig = go.Figure(data=
    go.Parcoords(
        line=dict(
            color=obj0, # Coloring the lines based on Objective 0
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Global rel.")
        ),
        # Define the exact order of the vertical axes here
        dimensions=[
            dict(label='Global reliablility', values=obj0),
            dict(label='Extreme precip.', values=obj1),
            dict(label='ED', values=obj2),
            dict(label='alpha_denseweight', values=alpha_denseweight),
            dict(label='learning_rate', values=learning_rate),
            dict(label='dropout', values=dropout),
            dict(label='num_hidden_layers', values=num_hidden_layers)
        ]
    )
)

fig.update_layout(
    title_text="Parallel Coordinates: Pareto Front Only",
    height=400,
    width=800
)


output_path = "tuning_GRU/parallel_coords.png"
fig.write_image(output_path, scale=3)
print(f"Saved parallel coordinates plot to {output_path}")