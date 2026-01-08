# 20-Series: Final Model Training Summary

This document summarizes the final training results for all models in the 20-series notebooks. Each model was trained using the best hyperparameters discovered during the 10-series hyperparameter tuning phase.

## Overview

| Model            | Task       | Test Accuracy | Test F1 (weighted) | Training Time |
|------------------|------------|---------------|-------------------|---------------|
| XGBoost          | Multiclass | 93.91%        | 0.9416            | 40m 0s        |
| LSTM             | Multiclass | 99.14%        | 0.9914            | 20m 5s        |
| LSTM-FCN         | Multiclass | 99.37%        | 0.9937            | 26m 13s       |
| CNN-Transformer  | Multiclass | 99.20%        | 0.9920            | 67m 33s       |
| TransKal         | Multiclass | 99.09%        | 0.9909            | 31m 54s       |
| LSTM-Autoencoder | Binary     | 96.96%        | 0.9705            | 11m 47s       |
| Conv-Autoencoder | Binary     | 99.39%        | 0.9939            | 0m 56s        |

---

## 20-xgboost-final-training

### Model Description
XGBoost (eXtreme Gradient Boosting) is a gradient boosted decision tree ensemble method. It uses a sequential ensemble approach where each tree corrects the errors of previous trees. XGBoost is known for its speed and performance on tabular data.

### Training Data
- **Source**: `data/multiclass_train.csv`, `data/multiclass_val.csv`, `data/multiclass_test.csv`
- **Train samples**: 864,000
- **Validation samples**: 432,000
- **Test samples**: 2,880,000
- **Features**: 52 (41 XMEAS measurements + 11 XMV manipulated variables)
- **Classes**: 18 fault types (0=normal, 1, 2, 4-8, 10-14, 16-20)

### Hyperparameters

| Parameter        | Value  |
|------------------|--------|
| n_estimators     | 499    |
| max_depth        | 6      |
| learning_rate    | 0.1958 |
| subsample        | 0.9669 |
| colsample_bytree | 0.8413 |
| min_child_weight | 8      |
| gamma            | 0.5636 |
| reg_alpha        | 0.3293 |
| reg_lambda       | 0.0631 |

### Results
- **Test Accuracy**: 93.91%
- **Balanced Accuracy**: 93.91%
- **F1 Score (weighted)**: 0.9416
- **F1 Score (macro)**: 0.9416
- **Precision (weighted)**: 0.9484
- **Recall (weighted)**: 0.9391
- **Best Iteration**: 498 (early stopping)
- **Training Time**: 39m 59s

### Per-Class F1 Scores

| Class      | F1 Score | Class | F1 Score |
|------------|----------|-------|----------|
| 0 (Normal) | 0.7355   | 11    | 0.8949   |
| 1          | 0.9939   | 12    | 0.9378   |
| 2          | 0.9926   | 13    | 0.9494   |
| 4          | 0.9672   | 14    | 0.9878   |
| 5          | 0.9915   | 16    | 0.8945   |
| 6          | 1.0000   | 17    | 0.9575   |
| 7          | 1.0000   | 18    | 0.9471   |
| 8          | 0.9572   | 19    | 0.9422   |
| 10         | 0.8680   | 20    | 0.9322   |

### Output Files
- Model: `outputs/models/xgboost_final.pkl`
- Metrics: `outputs/metrics/xgboost_metrics.json`
- Confusion Matrix: `outputs/metrics/xgboost_confusion_matrix.csv`

---

## 21-lstm-final-training

### Model Description
LSTM (Long Short-Term Memory) is a recurrent neural network architecture designed to capture long-term dependencies in sequential data. The model processes time windows of sensor measurements and outputs fault class predictions. The architecture includes multiple LSTM layers followed by a fully connected classification head.

### Training Data
- **Source**: `data/multiclass_train.csv`, `data/multiclass_val.csv`, `data/multiclass_test.csv`
- **Train samples**: 795,600 (windows created within simulation runs)
- **Validation samples**: 397,800
- **Test samples**: 2,743,200
- **Features**: 52
- **Classes**: 18

### Hyperparameters

| Parameter       | Value   |
|-----------------|---------|
| sequence_length | 39      |
| hidden_size     | 32      |
| num_layers      | 3       |
| dropout         | 0.0662  |
| learning_rate   | 0.00196 |
| batch_size      | 32      |

### Results
- **Test Accuracy**: 99.14%
- **Balanced Accuracy**: 99.14%
- **F1 Score (weighted)**: 0.9914
- **F1 Score (macro)**: 0.9914
- **Precision (weighted)**: 0.9917
- **Recall (weighted)**: 0.9914
- **Best Epoch**: 2 (early stopping at epoch 12)
- **Training Time**: 20m 5s

### Per-Class F1 Scores

| Class      | F1 Score | Class | F1 Score |
|------------|----------|-------|----------|
| 0 (Normal) | 0.9689   | 11    | 0.9987   |
| 1          | 0.9996   | 12    | 0.9650   |
| 2          | 0.9998   | 13    | 0.9850   |
| 4          | 0.9996   | 14    | 1.0000   |
| 5          | 0.9992   | 16    | 0.9972   |
| 6          | 0.9983   | 17    | 0.9963   |
| 7          | 0.9994   | 18    | 0.9582   |
| 8          | 0.9949   | 19    | 0.9989   |
| 10         | 0.9925   | 20    | 0.9938   |

### Output Files
- Model: `outputs/models/lstm_final.pkl`
- Metrics: `outputs/metrics/lstm_metrics.json`
- Confusion Matrix: `outputs/metrics/lstm_confusion_matrix.csv`

---

## 22-lstm-fcn-final-training

### Model Description
LSTM-FCN combines LSTM for temporal feature extraction with a Fully Convolutional Network (FCN) branch for local pattern detection. The architecture uses:
- LSTM branch for sequential modeling
- 1D CNN branch with hierarchical filters (32->24->16) and dilation rates (1->2->4)
- Spatial dropout for regularization
- Global average pooling before the classifier head

### Training Data
- **Source**: `data/multiclass_train.csv`, `data/multiclass_val.csv`, `data/multiclass_test.csv`
- **Train samples**: 793,800 (windows created within simulation runs)
- **Validation samples**: 396,900
- **Test samples**: 2,739,600
- **Features**: 52
- **Classes**: 18

### Hyperparameters

| Parameter       | Value   |
|-----------------|---------|
| sequence_length | 40      |
| lstm_hidden     | 24      |
| lstm_layers     | 1       |
| dropout         | 0.4366  |
| learning_rate   | 0.00139 |
| batch_size      | 64      |

### Results
- **Test Accuracy**: 99.37%
- **Balanced Accuracy**: 99.37%
- **F1 Score (weighted)**: 0.9937
- **F1 Score (macro)**: 0.9937
- **Precision (weighted)**: 0.9938
- **Recall (weighted)**: 0.9937
- **Best Epoch**: 12 (early stopping at epoch 22)
- **Training Time**: 26m 14s

### Per-Class F1 Scores

| Class      | F1 Score | Class | F1 Score |
|------------|----------|-------|----------|
| 0 (Normal) | 0.9730   | 11    | 0.9998   |
| 1          | 0.9992   | 12    | 0.9765   |
| 2          | 0.9996   | 13    | 0.9909   |
| 4          | 0.9998   | 14    | 1.0000   |
| 5          | 0.9993   | 16    | 0.9986   |
| 6          | 1.0000   | 17    | 0.9964   |
| 7          | 1.0000   | 18    | 0.9707   |
| 8          | 0.9960   | 19    | 0.9991   |
| 10         | 0.9939   | 20    | 0.9940   |

### Output Files
- Model: `outputs/models/lstm_fcn_final.pkl`
- Metrics: `outputs/metrics/lstm_fcn_metrics.json`
- Confusion Matrix: `outputs/metrics/lstm_fcn_confusion_matrix.csv`

---

## 23-cnn-transformer-final-training

### Model Description
CNN-Transformer combines 1D CNN layers for local feature extraction with a Transformer encoder for global temporal attention. The architecture includes:
- Two 1D convolutional layers with batch normalization
- Positional encoding for sequence position information
- Transformer encoder layers with multi-head self-attention
- Global average pooling and dropout before classification

### Training Data
- **Source**: `data/multiclass_train.csv`, `data/multiclass_val.csv`, `data/multiclass_test.csv`
- **Train samples**: 795,600 (windows created within simulation runs)
- **Validation samples**: 397,800
- **Test samples**: 2,743,200
- **Features**: 52
- **Classes**: 18

### Hyperparameters

| Parameter          | Value   |
|--------------------|---------|
| sequence_length    | 39      |
| conv_filters       | 32      |
| kernel_size        | 3       |
| d_model            | 32      |
| nhead              | 4       |
| num_encoder_layers | 1       |
| dim_feedforward    | 128     |
| dropout            | 0.2397  |
| learning_rate      | 0.00241 |
| batch_size         | 32      |

### Results
- **Test Accuracy**: 99.20%
- **Balanced Accuracy**: 99.20%
- **F1 Score (weighted)**: 0.9920
- **F1 Score (macro)**: 0.9920
- **Precision (weighted)**: 0.9921
- **Recall (weighted)**: 0.9920
- **Best Epoch**: 16 (early stopping at epoch 26)
- **Training Time**: 67m 33s

### Per-Class F1 Scores

| Class      | F1 Score | Class | F1 Score |
|------------|----------|-------|----------|
| 0 (Normal) | 0.9710   | 11    | 0.9997   |
| 1          | 0.9997   | 12    | 0.9689   |
| 2          | 0.9999   | 13    | 0.9886   |
| 4          | 0.9999   | 14    | 1.0000   |
| 5          | 0.9923   | 16    | 0.9988   |
| 6          | 1.0000   | 17    | 0.9952   |
| 7          | 0.9999   | 18    | 0.9572   |
| 8          | 0.9973   | 19    | 0.9997   |
| 10         | 0.9937   | 20    | 0.9939   |

### Output Files
- Model: `outputs/models/cnn_transformer_final.pkl`
- Metrics: `outputs/metrics/cnn_transformer_metrics.json`
- Confusion Matrix: `outputs/metrics/cnn_transformer_confusion_matrix.csv`

---

## 24-transkal-final-training

### Model Description
TransKal (Transformer + Kalman filter) combines a Transformer classifier with adaptive Kalman filtering for smoothed predictions. The architecture includes:
- Linear embedding layer with layer normalization
- Positional encoding
- Transformer encoder layers
- Two-layer classifier head
- Per-run adaptive Kalman filter for temporal smoothing of probability distributions

The Kalman filter operates on the softmax probabilities (not class indices) and includes:
- Adaptive Q/R parameters based on transition detection
- Confidence-based filtering
- Sliding window voting for additional smoothing

### Training Data
- **Source**: `data/multiclass_train.csv`, `data/multiclass_val.csv`, `data/multiclass_test.csv`
- **Train samples**: 793,800 (windows created within simulation runs)
- **Validation samples**: 396,900
- **Test samples**: 2,739,600
- **Features**: 52
- **Classes**: 18

### Hyperparameters

| Parameter       | Value    |
|-----------------|----------|
| sequence_length | 40       |
| d_model         | 32       |
| nhead           | 4        |
| num_layers      | 2        |
| dropout         | 0.1950   |
| learning_rate   | 0.000428 |
| batch_size      | 64       |
| kalman_Q        | 5.04e-05 |
| kalman_R        | 0.0223   |

### Results
- **Test Accuracy**: 99.09%
- **Balanced Accuracy**: 99.09%
- **F1 Score (weighted)**: 0.9909
- **F1 Score (macro)**: 0.9909
- **Precision (weighted)**: 0.9910
- **Recall (weighted)**: 0.9909
- **Raw Accuracy (without Kalman)**: 99.08%
- **Kalman Improvement**: +0.006% accuracy
- **Best Epoch**: 9 (early stopping at epoch 19)
- **Training Time**: 31m 54s

### Per-Class F1 Scores

| Class      | F1 Score | Class | F1 Score |
|------------|----------|-------|----------|
| 0 (Normal) | 0.9707   | 11    | 0.9997   |
| 1          | 0.9987   | 12    | 0.9593   |
| 2          | 0.9990   | 13    | 0.9844   |
| 4          | 1.0000   | 14    | 1.0000   |
| 5          | 0.9994   | 16    | 0.9991   |
| 6          | 0.9991   | 17    | 0.9950   |
| 7          | 0.9990   | 18    | 0.9505   |
| 8          | 0.9958   | 19    | 0.9996   |
| 10         | 0.9924   | 20    | 0.9941   |

### Output Files
- Model: `outputs/models/transkal_final.pkl`
- Metrics: `outputs/metrics/transkal_metrics.json`
- Confusion Matrix: `outputs/metrics/transkal_confusion_matrix.csv`

---

## 25-lstm-autoencoder-final-training

### Model Description
LSTM Autoencoder is an unsupervised anomaly detection model trained only on normal operation data. It learns to reconstruct normal patterns, and anomalies are detected when reconstruction error exceeds a learned threshold. The architecture includes:
- LSTM encoder that compresses sequences to a latent representation
- LSTM decoder that reconstructs the original sequence
- MSE-based reconstruction error for anomaly scoring

### Training Data
- **Source**: `data/binary_train.csv` (normal data only), `data/binary_test.csv` (evaluation)
- **Train samples**: 46,300 (normal operation only)
- **Validation samples**: 23,150
- **Test samples**: 759,310 (includes both normal and fault data)
- **Features**: 52
- **Task**: Binary classification (normal vs. fault)

**Note on Binary Dataset Structure**: The binary dataset is intentionally structured for unsupervised anomaly detection:
- Train/Val sets contain **only normal operation data** (fault 0) - approximately 50,000 and 25,000 samples respectively
- Test set contains normal data (115,200 samples) plus all 17 fault types (40,000 samples each = 680,000 fault samples)
- This simulates a realistic industrial scenario where fault data may be rare or unavailable during training
- The model learns "normal" patterns and flags deviations as anomalies
- This explains the small training set size compared to multiclass models (50K vs 864K samples)

### Hyperparameters

| Parameter            | Value    |
|----------------------|----------|
| sequence_length      | 38       |
| hidden_size          | 64       |
| num_layers           | 1        |
| latent_size          | 32       |
| dropout              | 0.0733   |
| learning_rate        | 0.000320 |
| batch_size           | 32       |
| threshold_percentile | 97.95    |

### Results
- **Test Accuracy**: 96.96%
- **Balanced Accuracy**: 97.16%
- **F1 Score (weighted)**: 0.9705
- **F1 Score (binary/fault)**: 0.9820
- **Precision**: 0.9955
- **Recall**: 0.9688
- **ROC-AUC**: 0.9925
- **PR-AUC**: 0.9988
- **Best Epoch**: 47 (early stopping at epoch 57)
- **Training Time**: 11m 47s

### Per-Class Metrics

| Class  | Precision | Recall | F1 Score |
|--------|-----------|--------|----------|
| Normal | 0.8420    | 0.9743 | 0.9034   |
| Fault  | 0.9955    | 0.9688 | 0.9820   |

### Output Files
- Model: `outputs/models/lstm_autoencoder_final.pkl`
- Metrics: `outputs/metrics/lstm_autoencoder_metrics.json`
- Confusion Matrix: `outputs/metrics/lstm_autoencoder_confusion_matrix.csv`

---

## 26-conv-autoencoder-final-training

### Model Description
Convolutional Autoencoder uses 1D CNNs for reconstruction-based anomaly detection. Like the LSTM autoencoder, it is trained only on normal data and detects anomalies via reconstruction error. The architecture includes:
- 1D CNN encoder with configurable filters and kernel sizes
- Optional Transformer layer in the latent space
- 1D CNN decoder for reconstruction
- MSE-based reconstruction error for anomaly scoring

### Training Data
- **Source**: `data/binary_train.csv` (normal data only), `data/binary_test.csv` (evaluation)
- **Train samples**: 46,100 (normal operation only)
- **Validation samples**: 23,050
- **Test samples**: 757,370 (includes both normal and fault data)
- **Features**: 52
- **Task**: Binary classification (normal vs. fault)

*See note on Binary Dataset Structure in LSTM-Autoencoder section above.*

### Hyperparameters

| Parameter            | Value   |
|----------------------|---------|
| sequence_length      | 40      |
| conv_filters         | 64      |
| kernel_size          | 3       |
| latent_filters       | 128     |
| use_transformer      | false   |
| dropout              | 0.1193  |
| learning_rate        | 0.00853 |
| batch_size           | 128     |
| threshold_percentile | 98.65   |

### Results
- **Test Accuracy**: 99.39%
- **Balanced Accuracy**: 98.63%
- **F1 Score (weighted)**: 0.9939
- **F1 Score (binary/fault)**: 0.9964
- **Precision**: 0.9958
- **Recall**: 0.9970
- **ROC-AUC**: 0.9988
- **PR-AUC**: 0.9998
- **Best Epoch**: 11 (early stopping at epoch 21)
- **Training Time**: 0m 56s

### Per-Class Metrics

| Class  | Precision | Recall | F1 Score |
|--------|-----------|--------|----------|
| Normal | 0.9825    | 0.9757 | 0.9791   |
| Fault  | 0.9958    | 0.9970 | 0.9964   |

### Output Files
- Model: `outputs/models/conv_autoencoder_final.pkl`
- Metrics: `outputs/metrics/conv_autoencoder_metrics.json`
- Confusion Matrix: `outputs/metrics/conv_autoencoder_confusion_matrix.csv`

---

## Summary and Conclusions

### Multiclass Classification Performance
For the 18-class fault classification task, the deep learning models significantly outperform XGBoost:

1. **LSTM-FCN** achieved the best multiclass performance with **99.37% accuracy** and **0.9937 F1 score**
2. **CNN-Transformer** ranked second with **99.20% accuracy**
3. **LSTM** achieved **99.14% accuracy** with the fastest training time among deep models
4. **TransKal** achieved **99.09% accuracy** with minimal Kalman filter improvement
5. **XGBoost** achieved **93.91% accuracy**, substantially lower than the deep learning models

### Binary Anomaly Detection Performance
For unsupervised anomaly detection:

1. **Conv-Autoencoder** achieved the best binary performance with **99.39% accuracy** and remarkable training speed (56 seconds)
2. **LSTM-Autoencoder** achieved **96.96% accuracy** with higher ROC-AUC but lower overall accuracy

**Why Conv-Autoencoder trains so fast (56 seconds)**:
- Small training dataset: Only ~46,000 normal samples (vs 864,000 for multiclass)
- Simple architecture: Pure CNN with no Transformer (`use_transformer: false`)
- Fast convergence: Early stopped at epoch 11 with large batch size (128)
- Efficient operations: 1D convolutions are highly parallelizable on GPU
- No classification training: Just learns reconstruction; threshold-based detection is post-hoc

### Key Observations

1. **Sequence-based models excel**: All models that leverage temporal sequences (LSTM, LSTM-FCN, CNN-Transformer, TransKal) significantly outperform XGBoost, which treats each sample independently.

2. **Normal class is challenging**: Across all multiclass models, fault class 0 (normal operation) and classes 12, 18 consistently have lower F1 scores, indicating these are harder to distinguish.

3. **Perfect classification on some faults**: Classes 6, 7, and 14 achieve near-perfect classification (F1 ~ 1.0) across all deep learning models.

4. **Conv-Autoencoder efficiency**: The convolutional autoencoder provides excellent anomaly detection with minimal training time, making it suitable for rapid deployment.

5. **Kalman filtering benefit is marginal**: The TransKal model shows only 0.006% improvement from Kalman filtering, suggesting the base Transformer already produces stable predictions.

### Recommended Models
- **For multiclass fault diagnosis**: LSTM-FCN (best accuracy with reasonable training time)
- **For binary anomaly detection**: Conv-Autoencoder (best accuracy with fastest training)
- **For interpretability**: XGBoost (feature importance analysis available)
