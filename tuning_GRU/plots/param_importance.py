import optuna
import plotly.graph_objects as go
from optuna.importance import get_param_importances

# script that generates plots for parameter importance

OUTPUT_DIR = "tuning_GRU"
STUDY_NAME = "tune_gru_diverseOptim3"

study = optuna.create_study(
    directions=["minimize", "minimize", "minimize"], 
    study_name=STUDY_NAME,
    storage=f"sqlite:///{OUTPUT_DIR}/optuna_tune.db",
    load_if_exists=True
) 


params = [
        "num_hidden_nodes",
        "num_hidden_layers",
        "dropout",
        "learning_rate",
        "batch_size",
        "alpha_denseweight",
        "loss_function",
        "sort_by_height",
        "lr_scheduler_factor",
]

imp0 = get_param_importances(study, target=lambda t: t.values[0], params=params)
imp1 = get_param_importances(study, target=lambda t: t.values[1], params=params)
imp2 = get_param_importances(study, target=lambda t: t.values[2], params=params)

all_params = set(imp0.keys()).union(set(imp1.keys())).union(set(imp2.keys()))
all_params = sorted(list(all_params))

val0 = [imp0.get(p, 0.0) for p in all_params]
val1 = [imp1.get(p, 0.0) for p in all_params]
val2 = [imp2.get(p, 0.0) for p in all_params]

fig = go.Figure(data=[
    go.Bar(name='Global reliablility', x=all_params, y=val0, marker_color='royalblue'),
    go.Bar(name='Extreme precip.', x=all_params, y=val1, marker_color='crimson'),
    go.Bar(name='ED', x=all_params, y=val2, marker_color='mediumseagreen')
])

fig.update_layout(
    title_text="Hyperparameter Importances",
    barmode='group', 
    xaxis_title="Hyperparameters",
    yaxis_title="Importance Score",
    legend_title="Target Objectives",
    height=600,
    width=1000
)

output_path = "tuning_GRU/param_importances.png"
fig.write_image(output_path, scale=2)
print(f"Saved grouped parameter importance plot to {output_path}")