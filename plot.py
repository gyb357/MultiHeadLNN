import pandas as pd
import matplotlib.pyplot as plt
import math


# Set up column names and window size
columns = [
    'run', 'window',
    'tp', 'tn', 'fp', 'fn',
    'accuracy', 'auc', 'bac',
    'micro_f1', 'macro_f1',
    'type_1_error', 'type_2_error',
    'rec_bankruptcy', 'pr_bankruptcy',
    'rec_healthy', 'pr_healthy',
    'train_time'
]
window = 3


# Load CSV results
df_cfc_s = pd.read_csv('result/MultiHeadCfC_results_StandardScaler.csv')
df_cfc_s['model'] = 'CfC_standard'
df_cfc_r = pd.read_csv('result/MultiHeadCfC_results_RobustScaler.csv')
df_cfc_r['model'] = 'CfC_robust'
df_gru_s = pd.read_csv('result/MultiHeadGRU_results_StandardScaler.csv')
df_gru_s['model'] = 'GRU_standard'


# Ensure columns are consistent
dfs = [df_cfc_s, df_cfc_r, df_gru_s]
models = ['CfC_standard', 'CfC_robust', 'GRU_standard']


# Filter and concatenate dataframes
df = pd.concat(dfs, ignore_index=True)
df = df[df['window'] == window]
print(len(df))


# Convert columns to numeric
metrics = [
    'tp', 'tn', 'fp', 'fn',
    'accuracy', 'auc', 'bac',
    'micro_f1', 'macro_f1',
    'type_1_error', 'type_2_error',
    'rec_bankruptcy', 'pr_bankruptcy',
    'rec_healthy', 'pr_healthy',
    'train_time'
]
df[metrics] = df[metrics].apply(pd.to_numeric, errors='coerce') # type: ignore


# Create boxplots for each metric
ncols = 4
nrows = math.ceil(len(metrics) / ncols)
fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(4*ncols, 3*nrows),
    constrained_layout=False
)
fig.suptitle(f'Model Comparison on Window={window} Across Metrics', fontsize=14)
axes = axes.flatten()
pastel_colors = ['#FCB9AA', '#FFDBCC', '#ECEAE4', '#A2E1DB', '#55CBCD', '#35AAAB']


for idx, metric in enumerate(metrics):
    ax = axes[idx]
    data = [df[df['model'] == m][metric].dropna() for m in models]
    bp = ax.boxplot(data, patch_artist=True)
    for patch, color in zip(bp['boxes'], pastel_colors):
        patch.set_facecolor(color)
    ax.set_title(metric.upper(), fontsize=10)
    ax.set_xticklabels(models, fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='y', linestyle='--', linewidth=0.4)

for idx in range(len(metrics), len(axes)):
    axes[idx].axis('off')


# Add legend and adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

