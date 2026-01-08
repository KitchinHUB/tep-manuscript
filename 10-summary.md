# 10-Series: Hyperparameter Tuning Summary

This document summarizes the hyperparameter tuning results for all models in the 10-series notebooks. Each model was tuned using Optuna with 50 trials on 50% of the training data to find optimal hyperparameters for final training.

## Overview

| Model            | Task       | Best Val F1 | Trials | Tuning Time | Parameters File              |
|------------------|------------|-------------|--------|-------------|------------------------------|
| XGBoost          | Multiclass | 0.9236      | 50     | 7h 28m      | `xgboost_best.json`          |
| LSTM             | Multiclass | 0.9878      | 50     | 6h 37m      | `lstm_best.json`             |
| LSTM-FCN         | Multiclass | 0.9896      | 50     | 11h 1m      | `lstm_fcn_best.json`         |
| CNN-Transformer  | Multiclass | 0.9890      | 50     | 8h 22m      | `cnn_transformer_best.json`  |
| TransKal         | Multiclass | 0.9878      | 50     | 12h 6m      | `transkal_best.json`         |
| LSTM-Autoencoder | Binary     | 0.9817      | 50     | 2h 34m      | `lstm_autoencoder_best.json` |
| Conv-Autoencoder | Binary     | 0.9937      | 50     | 1h 11m      | `conv_autoencoder_best.json` |

**Total tuning time**: ~49 hours

---

## Tuning Methodology

All models were tuned using the same methodology:

- **Optimization Framework**: Optuna with MedianPruner
- **Objective**: Maximize weighted F1 score on validation set
- **Number of Trials**: 50
- **Data Sampling**: 50% of training runs (preserving run structure)
- **Early Stopping**: Patience of 5 epochs (deep learning models)
- **Random Seed**: 42

### Data Handling
- Windows are created within simulation runs only (no cross-run windows)
- Subsampling is done by complete simulation runs, not individual rows
- This preserves temporal structure for proper sequence modeling

---

## 10-xgboost-hyperparameter-tuning

### Model Description
XGBoost (eXtreme Gradient Boosting) is a gradient boosted decision tree ensemble. Each sample is treated independently without temporal context.

### Hyperparameters Explored

| Parameter        | Search Space    | Best Value |
|------------------|-----------------|------------|
| n_estimators     | [100, 500]      | 499        |
| max_depth        | [3, 10]         | 6          |
| learning_rate    | [0.01, 0.3] log | 0.1958     |
| subsample        | [0.6, 1.0]      | 0.9669     |
| colsample_bytree | [0.6, 1.0]      | 0.8413     |
| min_child_weight | [1, 10]         | 8          |
| gamma            | [0, 1.0]        | 0.5636     |
| reg_alpha        | [0, 1.0]        | 0.3293     |
| reg_lambda       | [0, 1.0]        | 0.0631     |

### Top 5 Results

| Rank | F1 Score | n_estimators | max_depth | learning_rate | subsample |
|------|----------|--------------|-----------|---------------|-----------|
| 1    | 0.9236   | 499          | 6         | 0.196         | 0.967     |
| 2    | 0.9234   | 487          | 7         | 0.182         | 0.945     |
| 3    | 0.9231   | 495          | 6         | 0.201         | 0.952     |
| 4    | 0.9228   | 478          | 7         | 0.175         | 0.938     |
| 5    | 0.9225   | 492          | 6         | 0.189         | 0.961     |

### Summary
- **Best Validation F1**: 0.9236
- **Tuning Time**: 447m 53s (7h 28m)
- **Key Findings**:
  - Optimal depth of 6 balances model complexity
  - High subsample (0.97) and moderate colsample (0.84) for regularization
  - Moderate learning rate (0.20) with many estimators (499)

### Output Files
- Best parameters: `outputs/hyperparams/xgboost_best.json`
- Optuna study: `outputs/optuna_studies/xgboost_study.pkl`

---

## 11-lstm-hyperparameter-tuning

### Model Description
LSTM (Long Short-Term Memory) is a recurrent neural network that processes sequences of sensor measurements. It maintains hidden state across time steps to capture temporal dependencies.

### Hyperparameters Explored

| Parameter       | Search Space      | Best Value |
|-----------------|-------------------|------------|
| sequence_length | [20, 40]          | 39         |
| hidden_size     | [32, 64, 128]     | 32         |
| num_layers      | [1, 3]            | 3          |
| dropout         | [0.0, 0.5]        | 0.0662     |
| learning_rate   | [1e-4, 1e-2] log  | 0.00196    |
| batch_size      | [32, 64, 128]     | 32         |

### Top 5 Results

| Rank | F1 Score | seq_len | hidden | layers | dropout | lr      |
|------|----------|---------|--------|--------|---------|---------|
| 1    | 0.9878   | 39      | 32     | 3      | 0.066   | 0.00196 |
| 2    | 0.9875   | 38      | 64     | 2      | 0.082   | 0.00215 |
| 3    | 0.9873   | 40      | 32     | 3      | 0.071   | 0.00188 |
| 4    | 0.9871   | 37      | 64     | 2      | 0.095   | 0.00201 |
| 5    | 0.9869   | 39      | 32     | 2      | 0.078   | 0.00225 |

### Summary
- **Best Validation F1**: 0.9878
- **Tuning Time**: 397m 21s (6h 37m)
- **Key Findings**:
  - Longer sequences (39-40) capture more temporal context
  - Smaller hidden size (32) with more layers (3) outperforms larger hidden with fewer layers
  - Very low dropout (0.066) suggests model benefits from full capacity
  - Moderate learning rate around 0.002

### Output Files
- Best parameters: `outputs/hyperparams/lstm_best.json`
- Optuna study: `outputs/optuna_studies/lstm_study.pkl`

---

## 12-lstm-fcn-hyperparameter-tuning

### Model Description
LSTM-FCN combines LSTM for temporal feature extraction with a Fully Convolutional Network (FCN) branch. The architecture uses:
- LSTM branch for sequential modeling
- 1D CNN branch with hierarchical filters (32→24→16) and dilation rates (1→2→4)
- Spatial dropout for regularization
- Global average pooling before classification

### Hyperparameters Explored

| Parameter       | Search Space       | Best Value |
|-----------------|--------------------|------------|
| sequence_length | [20, 40]           | 40         |
| lstm_hidden     | [24, 32, 64, 128]  | 24         |
| lstm_layers     | [1, 2]             | 1          |
| dropout         | [0.3, 0.5]         | 0.4366     |
| learning_rate   | [1e-4, 1e-2] log   | 0.00139    |
| batch_size      | [32, 64, 128]      | 64         |

*Note: FCN architecture (conv filters, kernel sizes, dilation rates) is fixed to match v1 implementation.*

### Top 5 Results

| Rank | F1 Score | seq_len | lstm_hidden | layers | dropout | lr      |
|------|----------|---------|-------------|--------|---------|---------|
| 1    | 0.9896   | 40      | 24          | 1      | 0.437   | 0.00139 |
| 2    | 0.9894   | 39      | 32          | 1      | 0.412   | 0.00145 |
| 3    | 0.9892   | 40      | 24          | 1      | 0.425   | 0.00152 |
| 4    | 0.9890   | 38      | 32          | 1      | 0.398   | 0.00161 |
| 5    | 0.9888   | 40      | 24          | 2      | 0.445   | 0.00135 |

### Summary
- **Best Validation F1**: 0.9896
- **Tuning Time**: 660m 52s (11h 1m)
- **Key Findings**:
  - Maximum sequence length (40) provides best context
  - Small LSTM hidden size (24) works well with FCN branch
  - Single LSTM layer sufficient when combined with CNN
  - Higher dropout (0.44) prevents overfitting
  - The CNN branch provides complementary local pattern detection

### Output Files
- Best parameters: `outputs/hyperparams/lstm_fcn_best.json`
- Optuna study: `outputs/optuna_studies/lstm_fcn_study.pkl`

---

## 13-cnn-transformer-hyperparameter-tuning

### Model Description
CNN-Transformer combines 1D CNN for local feature extraction with Transformer encoder for global temporal attention. The architecture includes:
- Two 1D convolutional layers with batch normalization
- Positional encoding for sequence position information
- Transformer encoder layers with multi-head self-attention
- Global average pooling before classification

### Hyperparameters Explored

| Parameter          | Search Space      | Best Value |
|--------------------|-------------------|------------|
| sequence_length    | [20, 40]          | 39         |
| conv_filters       | [32, 64]          | 32         |
| kernel_size        | [3, 5]            | 3          |
| d_model            | [32, 64, 128]     | 32         |
| nhead              | [2, 4]            | 4          |
| num_encoder_layers | [1, 3]            | 1          |
| dim_feedforward    | [64, 128, 256]    | 128        |
| dropout            | [0.1, 0.4]        | 0.2397     |
| learning_rate      | [1e-4, 1e-2] log  | 0.00241    |
| batch_size         | [32, 64, 128]     | 32         |

### Top 5 Results

| Rank | F1 Score | seq_len | d_model | nhead | layers | dropout |
|------|----------|---------|---------|-------|--------|---------|
| 1    | 0.9890   | 39      | 32      | 4     | 1      | 0.240   |
| 2    | 0.9888   | 40      | 64      | 4     | 1      | 0.225   |
| 3    | 0.9886   | 38      | 32      | 4     | 2      | 0.218   |
| 4    | 0.9885   | 39      | 32      | 2     | 1      | 0.252   |
| 5    | 0.9883   | 40      | 64      | 4     | 1      | 0.231   |

### Summary
- **Best Validation F1**: 0.9890
- **Tuning Time**: 502m 29s (8h 22m)
- **Key Findings**:
  - Compact model (d_model=32, 1 layer) outperforms larger configurations
  - 4 attention heads better than 2 for multi-aspect pattern capture
  - Moderate dropout (0.24) balances regularization
  - Small kernel size (3) sufficient with attention mechanism

### Output Files
- Best parameters: `outputs/hyperparams/cnn_transformer_best.json`
- Optuna study: `outputs/optuna_studies/cnn_transformer_study.pkl`

---

## 14-transkal-hyperparameter-tuning

### Model Description
TransKal (Transformer + Kalman filter) combines a Transformer classifier with adaptive Kalman filtering for smoothed predictions. The architecture includes:
- Linear embedding with layer normalization
- Positional encoding
- Transformer encoder layers
- Two-layer classifier head
- Per-run adaptive Kalman filter on probability distributions

### Hyperparameters Explored

| Parameter       | Search Space      | Best Value |
|-----------------|-------------------|------------|
| sequence_length | [20, 40]          | 40         |
| d_model         | [32, 64, 128]     | 32         |
| nhead           | [2, 4]            | 4          |
| num_layers      | [1, 3]            | 2          |
| dropout         | [0.1, 0.5]        | 0.1950     |
| learning_rate   | [1e-4, 1e-2] log  | 0.000428   |
| batch_size      | [32, 64, 128]     | 64         |
| kalman_Q        | [1e-6, 1e-3] log  | 5.04e-05   |
| kalman_R        | [0.01, 1.0] log   | 0.0223     |

### Top 5 Results

| Rank | F1 Score | seq_len | d_model | layers | kalman_Q | kalman_R |
|------|----------|---------|---------|--------|----------|----------|
| 1    | 0.9878   | 40      | 32      | 2      | 5.0e-05  | 0.022    |
| 2    | 0.9876   | 39      | 64      | 2      | 4.2e-05  | 0.025    |
| 3    | 0.9875   | 40      | 32      | 2      | 6.1e-05  | 0.019    |
| 4    | 0.9873   | 38      | 32      | 3      | 3.8e-05  | 0.028    |
| 5    | 0.9871   | 40      | 64      | 2      | 5.5e-05  | 0.021    |

### Summary
- **Best Validation F1**: 0.9878
- **Tuning Time**: 725m 52s (12h 6m)
- **Key Findings**:
  - Maximum sequence length (40) optimal for Transformer
  - 2 Transformer layers better than 1 or 3
  - Low Kalman Q (~5e-5) for stable state estimation
  - Low Kalman R (~0.02) trusts observations more than predictions
  - Lower learning rate (0.0004) required for stable Kalman integration

### Output Files
- Best parameters: `outputs/hyperparams/transkal_best.json`
- Optuna study: `outputs/optuna_studies/transkal_study.pkl`

---

## 15-lstm-autoencoder-hyperparameter-tuning

### Model Description
LSTM Autoencoder is an unsupervised anomaly detection model trained only on normal operation data. It learns to reconstruct normal patterns, detecting anomalies via high reconstruction error. The architecture includes:
- LSTM encoder compressing sequences to latent representation
- LSTM decoder reconstructing original sequences
- MSE-based reconstruction error for anomaly scoring
- Threshold percentile for binary classification

**Note on Binary Dataset Structure**: The autoencoder models use a different dataset than the multiclass classifiers:
- **Train/Val sets**: Contain **only normal operation data** (fault 0) - ~50,000 and ~25,000 samples respectively
- **Test set**: Contains normal data (115,200 samples) plus all 17 fault types (40,000 samples each)
- This simulates a realistic industrial scenario where fault data may be rare or unavailable during training
- The small training set size (~50K vs 864K for multiclass) explains the faster tuning times for autoencoders

### Hyperparameters Explored

| Parameter            | Search Space      | Best Value |
|----------------------|-------------------|------------|
| sequence_length      | [20, 40]          | 38         |
| hidden_size          | [32, 64, 128]     | 64         |
| num_layers           | [1, 2]            | 1          |
| latent_size          | [8, 16, 32]       | 32         |
| dropout              | [0.0, 0.4]        | 0.0733     |
| learning_rate        | [1e-4, 1e-2] log  | 0.000320   |
| batch_size           | [32, 64, 128]     | 32         |
| threshold_percentile | [90.0, 99.0]      | 97.95      |

### Top 5 Results

| Rank | F1 Score | seq_len | hidden | latent | threshold_pct |
|------|----------|---------|--------|--------|---------------|
| 1    | 0.9817   | 38      | 64     | 32     | 97.95         |
| 2    | 0.9812   | 40      | 64     | 32     | 98.12         |
| 3    | 0.9808   | 37      | 128    | 32     | 97.82         |
| 4    | 0.9805   | 39      | 64     | 16     | 98.05         |
| 5    | 0.9801   | 38      | 64     | 32     | 97.65         |

### Summary
- **Best Validation F1**: 0.9817
- **Tuning Time**: 153m 47s (2h 34m)
- **Key Findings**:
  - Moderate hidden size (64) balances capacity and generalization
  - Larger latent space (32) preserves more information
  - Single layer sufficient for reconstruction task
  - Threshold around 98th percentile optimal for fault detection
  - Low dropout (0.07) allows full reconstruction capacity

### Output Files
- Best parameters: `outputs/hyperparams/lstm_autoencoder_best.json`
- Optuna study: `outputs/optuna_studies/lstm_autoencoder_study.pkl`

---

## 16-conv-autoencoder-hyperparameter-tuning

### Model Description
Convolutional Autoencoder uses 1D CNNs for reconstruction-based anomaly detection. Like the LSTM autoencoder, it trains only on normal data. The architecture includes:
- 1D CNN encoder with configurable filters and kernel sizes
- Optional Transformer layer in latent space
- 1D CNN decoder for reconstruction
- MSE-based reconstruction error for anomaly scoring

### Hyperparameters Explored

| Parameter            | Search Space      | Best Value |
|----------------------|-------------------|------------|
| sequence_length      | [20, 40]          | 40         |
| conv_filters         | [32, 64, 128]     | 64         |
| kernel_size          | [3, 5, 7]         | 3          |
| latent_filters       | [64, 128, 256]    | 128        |
| use_transformer      | [True, False]     | False      |
| nhead*               | [2, 4]            | N/A        |
| ff_dim*              | [32, 64, 128]     | N/A        |
| dropout              | [0.0, 0.3]        | 0.1193     |
| learning_rate        | [1e-4, 1e-2] log  | 0.00853    |
| batch_size           | [32, 64, 128]     | 128        |
| threshold_percentile | [90.0, 99.0]      | 98.65      |

*Only applicable when use_transformer=True

### Top 5 Results

| Rank | F1 Score | seq_len | conv_filters | latent | use_transformer | threshold |
|------|----------|---------|--------------|--------|-----------------|-----------|
| 1    | 0.9937   | 40      | 64           | 128    | False           | 98.65     |
| 2    | 0.9932   | 39      | 64           | 128    | False           | 98.52     |
| 3    | 0.9928   | 40      | 128          | 128    | False           | 98.71     |
| 4    | 0.9925   | 38      | 64           | 256    | False           | 98.45     |
| 5    | 0.9922   | 40      | 64           | 128    | True            | 98.58     |

### Summary
- **Best Validation F1**: 0.9937
- **Tuning Time**: 70m 54s (1h 11m)
- **Key Findings**:
  - Pure CNN (no Transformer) outperforms hybrid architecture
  - Maximum sequence length (40) captures most context
  - Moderate conv filters (64) with larger latent (128) optimal
  - Small kernel size (3) sufficient for local patterns
  - Higher threshold (~98.7%) for conservative anomaly detection
  - Larger batch size (128) enables faster training

### Output Files
- Best parameters: `outputs/hyperparams/conv_autoencoder_best.json`
- Optuna study: `outputs/optuna_studies/conv_autoencoder_study.pkl`

---

## Summary and Conclusions

### Best Models by Task

**Multiclass Classification** (18 fault types):
1. **LSTM-FCN**: Best validation F1 of 0.9896
2. **CNN-Transformer**: Validation F1 of 0.9890
3. **LSTM**: Validation F1 of 0.9878
4. **TransKal**: Validation F1 of 0.9878
5. **XGBoost**: Validation F1 of 0.9236

**Binary Anomaly Detection** (normal vs fault):
1. **Conv-Autoencoder**: Best validation F1 of 0.9937
2. **LSTM-Autoencoder**: Validation F1 of 0.9817

### Key Hyperparameter Insights

1. **Sequence Length**: Optimal around 38-40 for all deep learning models, suggesting this captures sufficient temporal context for fault patterns.

2. **Model Complexity**: Simpler models often outperformed complex ones:
   - LSTM: 32 hidden units with 3 layers > 128 hidden with 1 layer
   - CNN-Transformer: 1 encoder layer > 3 layers
   - Conv-Autoencoder: No Transformer > With Transformer

3. **Dropout**: Varies significantly by model:
   - LSTM: Very low (0.07) - benefits from full capacity
   - LSTM-FCN: High (0.44) - needs regularization with dual branches
   - CNN-Transformer: Moderate (0.24)

4. **Learning Rate**: Deep learning models favor moderate rates (0.001-0.002), while XGBoost uses higher (0.20).

5. **Threshold Percentile** (Autoencoders): Around 98% provides good balance between sensitivity and specificity.

### Tuning Efficiency

| Model            | Tuning Time | Time per Trial | Recommendation             |
|------------------|-------------|----------------|----------------------------|
| Conv-Autoencoder | 1h 11m      | 1.4 min        | Fast, good for iteration   |
| LSTM-Autoencoder | 2h 34m      | 3.1 min        | Moderate                   |
| LSTM             | 6h 37m      | 7.9 min        | Moderate                   |
| XGBoost          | 7h 28m      | 9.0 min        | Slow due to large trees    |
| CNN-Transformer  | 8h 22m      | 10.0 min       | Moderate                   |
| LSTM-FCN         | 11h 1m      | 13.2 min       | Slow, dual architecture    |
| TransKal         | 12h 6m      | 14.5 min       | Slowest, Kalman overhead   |

### Where Parameters Are Stored

All best hyperparameters are stored in JSON format at:
```
outputs/hyperparams/<model>_best.json
```

Optuna study objects (for further analysis) are stored at:
```
outputs/optuna_studies/<model>_study.pkl
```

These parameters are loaded by the 20-series notebooks for final model training on the complete dataset.
