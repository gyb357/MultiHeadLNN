import pandas as pd
import torch
import os
from yaml import safe_load
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from model.model import MultiHead, SingleHead, ANN
from dataset import (
    undersample_dataset,
    processing,
    extract_variable_train,
    to_tensor_dataset,
)
from typing import List
from torch.utils.data import DataLoader
from train.train import fit, eval
from model.utils import get_parameters
from train.utils import save_results


# Get configurations (pyyaml config)
with open('config/configs.yaml', 'r') as file:
    config = safe_load(file)


# Basic run and window settings
RUNS = config['runs']
WINDOW_START = config['window_start']
WINDOW_END = config['window_end']

# Undersampling settings
BAL_TRAIN = config['bal_train']
BAL_VALID = config['bal_valid']
BAL_TEST = config['bal_test']

# Scaler settings
_scalers = {'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()}
SCALER = _scalers[config['scaler']]

# DataLoader settings
BATCH_SIZE = config['batch_size']
DEVICE = config['device']

# Model settings
_head = {'multi': MultiHead,
         'single': SingleHead,
         'ann': ANN}
MODEL = _head[config['head']]
TYPE = config['type']
HIDDEN_SIZE = config['hidden_size']
NUM_CLASSES = config['num_classes']

# Training settings
LR = config['lr']
EPOCHS = config['epochs']
PATIENCE = config['patience']
THRESHOLD = config['threshold']


# Main
if __name__ == '__main__':
    for run in range(1, RUNS + 1):
        for window in range(WINDOW_START, WINDOW_END + 1):
            print(f"\n[Run: {run} | Window: {window}]")
            # Load datasets
            train = pd.read_csv(f'dataset/{window}_train.csv')
            valid = pd.read_csv(f'dataset/{window}_valid.csv')
            test = pd.read_csv(f'dataset/{window}_test.csv')

            # Extract 'cik' with 'status' from test dataset
            cik_status = test[['cik', 'status']].copy()
            num_samples = len(cik_status) // window * window
            cik_status_df = (cik_status.iloc[0:num_samples:window])
            # [0:num_samples:window] = 0, 0 + window, 0 + 2 * window, ..., 0 + (n_samples//window - 1) * window

            # Undersample datasets
            if BAL_TRAIN: train = undersample_dataset(train, window)
            if BAL_VALID: valid = undersample_dataset(valid, window)
            if BAL_TEST: test = undersample_dataset(test, window)

            # Processing (remove unnecessary columns & separate x, y)
            x_train, y_train = processing(train)
            x_valid, y_valid = processing(valid)
            x_test, y_test = processing(test)

            # Extract variables
            variables = x_train.columns.tolist()

            # Scale the data
            x_train_scaled = SCALER.fit_transform(x_train.values)
            x_valid_scaled = SCALER.transform(x_valid.values)
            x_test_scaled = SCALER.transform(x_test.values)

            # Convert to DataFrame
            x_train_df = pd.DataFrame(x_train_scaled, columns=x_train.columns)
            x_valid_df = pd.DataFrame(x_valid_scaled, columns=x_valid.columns)
            x_test_df = pd.DataFrame(x_test_scaled, columns=x_test.columns)

            # Extract variables for training
            x_train_list: List = []
            x_valid_list: List = []
            for v in variables:
                x_train_list.append(extract_variable_train(x_train_df, v, window))
                x_valid_list.append(extract_variable_train(x_valid_df, v, window))
            x_test_list: List = [extract_variable_train(x_test_df, var, window) for var in variables]

            # Convert to tensors
            train_dataset = to_tensor_dataset(x_train_list, y_train, window)
            valid_dataset = to_tensor_dataset(x_valid_list, y_valid, window)
            test_dataset = to_tensor_dataset(x_test_list, y_test, window)

            # Create DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # Initialize device and model
            device = torch.device(DEVICE)
            model = MODEL(window_size=window,
                          num_variables=len(variables),
                          hidden_size=HIDDEN_SIZE,
                          num_classes=NUM_CLASSES,
                          cell_type=TYPE).to(device)
            
            # Print number of parameters
            print(f"Cell parameters: {get_parameters(model.cell)}",
                  f"Classifier parameters: {get_parameters(model.fc)}",
                  f"Total parameters: {get_parameters(model)}")

            # CSV path for saving results
            csv_path = f"result/{MODEL.__name__}_{TYPE.upper()}_{SCALER.__class__.__name__}_{THRESHOLD}"

            # Train the model
            train_time = fit(model=model,
                             train_loader=train_loader,
                             valid_loader=valid_loader,
                             device=device,
                             lr=LR,
                             epochs=EPOCHS,
                             patience=PATIENCE,
                             threshold=THRESHOLD,
                             run=run,
                             window=window,
                             csv_path=csv_path + "_valid.csv")

            # Evaluate the model
            checkpoint = torch.load('result/best_model.pth', map_location=device)
            model.load_state_dict(checkpoint['model'])

            metrics = eval(model=model,
                           data_loader=test_loader,
                           device=device,
                           threshold=THRESHOLD,
                           cik_status=cik_status_df.values.tolist())
            
            print(f"[Window {window}] "
                  f"Test Accuracy = {metrics[0]:.4f}, "
                  f"AUC = {metrics[1]:.4f}, "
                  f"BAC = {metrics[2]:.4f}")

            # Save results
            if not os.path.exists('result'):
                os.makedirs('result', exist_ok=True)

            save_results(run=run,
                         window=window,
                         metrics=metrics,
                         train_time=train_time,
                         csv_path=csv_path + "_test.csv")

