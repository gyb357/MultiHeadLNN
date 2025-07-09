# Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Model Comparison](#model-comparison)
4. [Getting Started](#getting-started)
   * [Check the Running Environment](#1-check-the-running-environment)
   * [Installation and Dependencies](#2-installation-and-dependencies)
   * [Download Datasets](#3-download-datasets)
   * [Configuration](#4-configuration)
   * [Training the Model](#5-training-the-model)
5. [Project Structure](#project-structure)

---

# 📑Introduction

## Multi-Head Neural Networks for Financial Time Series Classification

This repository implements multiple multi-head neural network architectures for financial time series classification, specifically focused on bankruptcy prediction. The project leverages various recurrent neural network architectures including Liquid Time-Constant Networks (LTC), Closed-form Continuous-time Networks (CfC), LSTM, and GRU, all designed with a multi-head approach to handle multiple financial variables simultaneously.

The implementation supports different data preprocessing techniques, undersampling strategies for imbalanced datasets, and comprehensive evaluation metrics for financial classification tasks.

---

# 🔍Architecture Overview

## Multi-Head Architecture

The core innovation of this project is the multi-head architecture that processes each financial variable independently through separate neural network cells before combining the outputs for final classification. 

### 1. Liquid Time-Constant Networks (LTC)
- **Purpose**: Continuous-time neural networks with liquid time constants
- **Implementation**: Uses the `ncps` library for LTC implementation
- **Paper**: https://arxiv.org/pdf/2006.04439

### 2. Closed-form Continuous-time Networks (CfC)
- **Purpose**: Efficient continuous-time neural networks with closed-form solutions
- **Implementation**: Uses the `ncps` library with tanh activation
- **Paper**: https://arxiv.org/pdf/2106.13898

### 3. Long Short-Term Memory (LSTM)
- **Purpose**: Traditional recurrent neural network with memory cells
- **Implementation**: PyTorch's built-in LSTM with custom weight initialization

### 4. Gated Recurrent Unit (GRU)
- **Purpose**: Simplified recurrent neural network with gating mechanisms
- **Implementation**: PyTorch's built-in GRU with custom weight initialization

---

# 📋Model Comparison

## Model Parameters

<div align="center">

| Model Type | Window Size | Hidden Size | Parameters |
|------------|-------------|-------------|------------|
| LTC        | 3, 4, 5     | 64          | 4948, 6766, 8728 |
| CfC        | 3, 4, 5     | 64          | 43162, 55906, 68650 |
| LSTM       | 3, 4, 5     | 64          | 5074, 6946, 8962 |
| GRU        | 3, 4, 5     | 64          | 4750, 6442, 8242 |

</div>

---

# 🔨Getting Started

## 1. Check the Running Environment

Check your PyTorch installation:
```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## 2. Installation and Dependencies

Clone the repository and install dependencies:

```bash
git clone https://github.com/gyb357/MultiHeadLNN
cd MultiHeadLNN
pip install -r requirements.txt
```

### Required Dependencies

torch, pandas, matplotlib, scikit-learn, ncps, rich, pyyaml

## 3. Download Datasets

You can download the dataset and related papers here:

```bash
https://github.com/sowide/multi-head_LSTM_for_bankruptcy-prediction
https://github.com/sowide/bankruptcy_dataset
```

 > [!Note]
 > We've referenced code from that GitHub.
 > The existing tensorflow code has been replaced with torch and supports GRU and liquid neural network models as well as LSTM.

 > [!IMPORTANT]
 > The dataset is under a CC-BY-4.0 license.
 > Please refer to the readme.md on the corresponding GitHub.


## 4. Configuration

Modify the `./config/configs.yaml` file to customize your experiment:

```yaml
# Basic run and window settings
runs: 100                   # Number of experimental runs
window_start: 3             # Minimum window size
window_end: 5               # Maximum window size

# Undersampling settings
bal_train: true             # Balance training data
bal_valid: false            # Balance validation data
bal_test: false             # Balance test data

# Scaler settings (standard, robust, minmax)
scaler: 'standard'

# DataLoader settings
batch_size: 32
device: 'cpu'               # (cpu, cuda, mps)

# Model settings (ltc, cfc, lstm, gru)
model: 'cfc'
hidden_size: 64
num_classes: 2

# Training settings
lr: 0.0001
epochs: 1000
patience: 100
threshold: 0.5
```

## 5. Training the Model

### Data Preparation

Ensure your dataset follows the expected structure:
- Place training data in `dataset/{window}_train.csv`
- Place validation data in `dataset/{window}_valid.csv`
- Place test data in `dataset/{window}_test.csv`

### Running Training

To train the model, run:

```bash
python main.py
```

### Results

Results are automatically saved to:
- `result/{ModelName}_results_{ScalerName}.csv`
- `result/best_model.pth` (best model checkpoint)

---

# 📁Project Structure

```
MultiHeadLNN/
├── config/                 # Configuration files
│   └── configs.yaml        # Main configuration file
├── dataset/                # Dataset directory
│   ├── {window}_train.csv  # Training data for each window
│   ├── {window}_valid.csv  # Validation data for each window
│   └── {window}_test.csv   # Test data for each window
├── model/                  # Model architectures and utilities
│   ├── classifier.py       # Classification head implementation
│   ├── forward.py          # Forward pass utilities
│   ├── lnn.py              # Liquid neural network implementations
│   ├── rnn.py              # Traditional RNN implementations
│   └── utils.py            # Model utilities
├── result/                 # Results and model checkpoints
│   ├── best_model.pth      # Best model checkpoint
│   └── *_results_*.csv     # Experimental results
├── dataset.py              # Data processing utilities
├── main.py                 # Main training script
├── plot.py                 # Visualization utilities
├── train.py                # Training and evaluation functions
├── requirements.txt        # Python dependencies
└── README.md               # This file
```
