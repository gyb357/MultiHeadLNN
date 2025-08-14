import pandas as pd
from yaml import safe_load
from typing import List


with open('config/configs.yaml', 'r') as file:
    config = safe_load(file)


def save_results(run: int, window: int, metrics: List[float], train_time: float, csv_path: str) -> None:
    results = pd.DataFrame({
        'run': [run],
        'window': [window],
        'tp': [metrics[8]],
        'tn': [metrics[5]],
        'fp': [metrics[6]],
        'fn': [metrics[7]],
        'acc': [metrics[0]],
        'roc_auc': [metrics[1]],
        'bac': [metrics[2]],
        'pr_auc1': [metrics[3]],
        'pr_auc2': [metrics[4]],
        'micro_f1': [metrics[9]],
        'macro_f1': [metrics[10]],
        'type_1_error': [metrics[11]],
        'type_2_error': [metrics[12]],
        'rec_bankruptcy': [metrics[13]],
        'pr_bankruptcy': [metrics[14]],
        'rec_healthy': [metrics[15]],
        'pr_healthy': [metrics[16]],
        'train_time': [train_time]
    })
    results.to_csv(
        csv_path,
        mode='a',
        index=False,
        header=(run == 1 and window == config['window_start']),
        float_format='%.4f',
        lineterminator='\n'
    )

