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


*****


# 📑Introduction

## Multi-Head Neural Networks for Financial Time Series Classification

This repository implements multi-head architectures for financial time series classification for bankruptcy prediction. It employs recurrent models—LTC, CfC, LSTM, and GRU—in a multi-head architecture to process multiple financial indicators concurrently. The code supports various preprocessing techniques, undersampling methods against class imbalance, and comprehensive evaluation metrics.


*****


# 🔍Architecture Overview

## Multi-Head Architecture

<img src="assets/architecture.png">

The core innovation is a multi-head design that processes each financial variable in its own network branch and then aggregates their outputs for final classification.

 > [!Note]
 > images and the papers are [here](https://www.mdpi.com/1999-5903/16/3/79)


### 1. Liquid Time-Constant Networks (LTC)
Continuous-time recurrent neural network with liquid time constants. Uses the [ncps](https://github.com/mlech26l/ncps) library for LTC implementation
- **Paper**: https://arxiv.org/pdf/2006.04439

### 2. Closed-form Continuous-time Networks (CfC)
Efficient continuous-time neural networks with closed-form solutions. Uses the [ncps](https://github.com/mlech26l/ncps) library with tanh activation
- **Paper**: https://arxiv.org/pdf/2106.13898

### 3. Long Short-Term Memory (LSTM)
Traditional recurrent neural network with memory cells

### 4. Gated Recurrent Unit (GRU)
Simplified recurrent neural network with gating mechanisms


*****


# 📋Model Comparison

## Model Parameters

<div align="center">

| Model Type | Window Size | Hidden Size (Classifier) | Parameters |
|------------|-------------|--------------------------|------------|
| LTC        | 3, 4, 5     | 64                       | 4948, 6766, 8728 |
| CfC        | 3, 4, 5     | 64                       | 43162, 55906, 68650 |
| LSTM       | 3, 4, 5     | 64                       | 5074, 6946, 8962 |
| GRU        | 3, 4, 5     | 64                       | 4750, 6442, 8242 |

</div>


*****


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
 > The dataset is under a CC-BY-4.0 license. Please refer to the readme.md on the corresponding GitHub.

## 4. Configuration

Modify the `./config/configs.yaml` file to customize your experiment.

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

```bash
python plot.py
```


*****


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

