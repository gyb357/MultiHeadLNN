import pandas as pd
from yaml import safe_load
from typing import List


with open('config/configs.yaml', 'r') as file:
    config = safe_load(file)


def probability(probs: List[float], threshold: float) -> List[int]:
    return [1 if p >= threshold else 0 for p in probs]


def save_results(
        run: int,
        window: int,
        metrics: List[float],
        train_time: float,
        csv_path: str
) -> None:
    results = pd.DataFrame({'run': [run],
                            'window': [window],              # windows: 3, 4, 5
                            'tp': [metrics[8]],              # Bankruptcy (True Positive)
                            'tn': [metrics[5]],              # Healthy    (True Negative)
                            'fp': [metrics[6]],
                            'fn': [metrics[7]],
                            'acc': [metrics[0]],             # Accuracy
                            'roc_auc': [metrics[1]],         # ROC-AUC
                            'bac': [metrics[2]],             # Balanced Accuracy
                            'pr_auc1': [metrics[3]],         # PR-AUC (average_precision_score)
                            'pr_auc2': [metrics[4]],         # PR-AUC (precision_recall_curve + auc)
                            'micro_f1': [metrics[9]],
                            'macro_f1': [metrics[10]],
                            'type_1_error': [metrics[11]],
                            'type_2_error': [metrics[12]],
                            'rec_bankruptcy': [metrics[13]], # Recall
                            'pr_bankruptcy': [metrics[14]],  # Precision
                            'rec_healthy': [metrics[15]],    # Recall
                            'pr_healthy': [metrics[16]],     # Precision
                            'train_time': [train_time]})     # 1 Epoch per second
    
    # Save results to CSV
    results.to_csv(csv_path,
                   mode='a',
                   index=False,
                   header=(run == 1 and window == config['window_start']),
                   float_format='%.4f',
                   lineterminator='\n')

