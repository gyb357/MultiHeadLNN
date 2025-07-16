import pandas as pd
from yaml import safe_load
from typing import List


with open('config/configs.yaml', 'r') as file:
    config = safe_load(file)


def save_results(run: int, window: int, metrics: List[float], csv_path: str) -> None:
    results = pd.DataFrame({
        'run': [run],
        'window': [window],
        'tp': [metrics[6]],
        'tn': [metrics[3]],
        'fp': [metrics[4]],
        'fn': [metrics[5]],
        'accuracy': [metrics[0]],
        'auc': [metrics[1]],
        'bac': [metrics[2]],
        'micro_f1': [metrics[7]],
        'macro_f1': [metrics[8]],
        'type_1_error': [metrics[9]],
        'type_2_error': [metrics[10]],
        'rec_bankruptcy': [metrics[11]],
        'pr_bankruptcy': [metrics[12]],
        'rec_healthy': [metrics[13]],
        'pr_healthy': [metrics[14]],
    })
    results.to_csv(
        csv_path,
        mode='a',
        index=False,
        header=(run == 1 and window == config['window_start']),
        float_format='%.4f',
        lineterminator='\n'
    )

