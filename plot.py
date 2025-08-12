import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import math
import matplotlib.pyplot as plt
from yaml import safe_load
from pathlib import Path


with open('config/configs.yaml', 'r') as file:
    config = safe_load(file)


# Constants
WINDOW = config['window']
COLUMNS = [
    'run', 'window',
    'tp', 'tn', 'fp', 'fn',
    'accuracy', 'auc', 'bac',
    'micro_f1', 'macro_f1',
    'type_1_error', 'type_2_error',
    'rec_bankruptcy', 'pr_bankruptcy',
    'rec_healthy', 'pr_healthy',
    'train_time'
]
METRICS = COLUMNS[2:]


# Load CSV data
csv_files = list(Path('result/').glob('*.csv'))
if not csv_files:
    raise FileNotFoundError("No CSV files found in the 'result' directory.")

df: list = []
for file in csv_files:
    # Read CSV file
    df_tmp = pd.read_csv(file, names=COLUMNS, header=0)
    # Set model name from file name
    df_tmp['model'] = file.stem
    # Print number of rows
    print(f"{file.stem}: {len(df_tmp)} rows")
    # append to list
    df.append(df_tmp)

# Concatenate all dataframes
df = pd.concat(df, ignore_index=True)

# Filter by window
df = df[df['window'] == WINDOW]
df[METRICS] = df[METRICS].apply(pd.to_numeric, errors='coerce')

# Sort
scaler_prio = ['StandardScaler', 'RobustScaler']
arch_prio   = ['ANN', 'RNN', 'LSTM', 'GRU', 'CfC']

# Get unique models
unique_models = df['model'].unique().tolist()
models = sorted(
    unique_models,
    key=lambda m: (
        # Scaler
        scaler_prio.index(m.split('_results_')[1].split('_')[0])
            if '_results_' in m
               and m.split('_results_')[1].split('_')[0] in scaler_prio
            else len(scaler_prio),
        # Model name
        next(
            (arch_prio.index(a) for a in arch_prio if a in m.upper()),
            len(arch_prio)
        )
    )
)

# Best-by-median summary (print before plotting)
print(f"\n[Window={WINDOW}] Best model(s) by median for each metric\n" + "-"*70)
for metric in METRICS:
    # Get median values for each model
    medians = (
        df[df['model'].isin(models)]
        .groupby('model')[metric]
        .median()
        .dropna()
    )
    if medians.empty:
        print(f"{metric.upper():<15} | No data available")
        continue

    best_val = medians.max()
    best_models = medians[medians == best_val].index.tolist()
    best_models_str = ", ".join(best_models)

    # Print the best model(s) for the metric
    print(f"{metric.upper():<15} | best median = {best_val:.3f} | model(s): {best_models_str}")
print("-"*70 + "\n")

n_models = len(models)

# Generate pastel colors for each model
start_hex = '#FCB9AA'
end_hex   = '#35AAAB'
start_rgb = np.array(mcolors.to_rgb(start_hex))
end_rgb   = np.array(mcolors.to_rgb(end_hex))

if n_models > 1:
    pastel_colors = [
        mcolors.to_hex(start_rgb + (end_rgb - start_rgb) * i/(n_models-1))
        for i in range(n_models)
    ]
else:
    pastel_colors = [start_hex]


# Draw boxplots
ncols = 4
nrows = math.ceil(len(METRICS) / ncols)
fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(6*ncols, 6*nrows),
    constrained_layout=False
)
fig.suptitle(f'Model Comparison on Window={WINDOW} Across Metrics', fontsize=14)
axes = axes.flatten()

# Create boxplots for each metric
for idx, metric in enumerate(METRICS):
    ax = axes[idx]
    data = [df[df['model'] == m][metric].dropna() for m in models]
    bp = ax.boxplot(data, patch_artist=True)
    for patch, color in zip(bp['boxes'], pastel_colors):
        patch.set_facecolor(color)
    ax.set_title(metric.upper(), fontsize=10)
    ax.set_xticklabels(models, fontsize=8, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='y', linestyle='--', linewidth=0.4)

# Hide unused axes
for idx in range(len(METRICS), len(axes)):
    axes[idx].axis('off')


plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

