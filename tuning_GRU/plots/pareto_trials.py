import optuna
from plotly.subplots import make_subplots
import os
import random
import kaleido

# script that generates plots for pareto trials

kaleido.get_chrome_sync()


OUTPUT_DIR = "tuning_GRU"
STUDY_NAME = "tune_gru_diverseOptim3"

study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"], 
        study_name=STUDY_NAME,
        storage=f"sqlite:///{OUTPUT_DIR}/optuna_tune.db",
        load_if_exists=True
    ) 

fig1 = optuna.visualization.plot_pareto_front(study, targets=lambda t: (t.values[0], t.values[1]), target_names=["Objective 0", "Objective 1"])
fig2 = optuna.visualization.plot_pareto_front(study, targets=lambda t: (t.values[1], t.values[2]), target_names=["Objective 1", "Objective 2"])
fig3 = optuna.visualization.plot_pareto_front(study, targets=lambda t: (t.values[0], t.values[2]), target_names=["Objective 0", "Objective 2"])

fig = make_subplots(rows=1, cols=3, subplot_titles=("Global reliablility vs Extreme precip", "Extreme precip vs ED", "Global reliablility vs ED"), horizontal_spacing=0.1)

for trace in fig1.data:
    trace.showlegend = False
    fig.add_trace(trace, row=1, col=1)
for trace in fig2.data:
    trace.showlegend = False
    fig.add_trace(trace, row=1, col=2)
for trace in fig3.data:
    trace.showlegend = False
    fig.add_trace(trace, row=1, col=3)

fig.update_xaxes(title_text="Global reliablility", row=1, col=1)
fig.update_yaxes(title_text="Extreme precip.", row=1, col=1)
fig.update_xaxes(title_text="Extreme precip", row=1, col=2)
fig.update_yaxes(title_text="ED", row=1, col=2)
fig.update_xaxes(title_text="Global reliablility", row=1, col=3)
fig.update_yaxes(title_text="ED", row=1, col=3)

fig.update_layout(title_text="Pairwise 2D Pareto Fronts", height=500, width=1400, hovermode="closest")

# Save as PNG
output_path = "tuning_GRU/paretoStudy3.png"
fig.write_image(output_path, scale=3)
print(f"Generated {output_path}")