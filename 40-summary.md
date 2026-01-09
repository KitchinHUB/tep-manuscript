# 40-Series: New TEP Dataset Evaluation Summary

This document summarizes the results of evaluating all trained models on the newly generated independent TEP dataset. This tests model generalization on completely unseen data generated from `tep-sim` with different random seeds than the training data.

## New TEP Dataset Generation

The new evaluation dataset was generated using `tep-sim` (notebook 03) with:
- **Independent random seeds**: Different from training/validation/test data
- **Same fault scenarios**: All 18 fault classes (0, 1, 2, 4-8, 10-14, 16-20)
- **Same simulation parameters**: 48-hour runs, fault injection at 8 hours

### Dataset Statistics

| Dataset | Samples | Purpose |
|---------|---------|---------|
| new_multiclass_eval.csv | 1,978,215 | Multiclass fault classification |
| new_binary_eval.csv | 1,157,585 | Binary anomaly detection |

---

## Comparison: 20-Series (Original Test) vs 40-Series (New Eval)

### Multiclass Models

| Model           | Orig Accuracy | Orig F1 | New Accuracy | New F1 | Acc Delta | Error Reduction |
|-----------------|---------------|---------|--------------|--------|-----------|-----------------|
| CNN-Transformer | 99.20%        | 0.9920  | **99.57%**   | 0.9958 | **+0.38%**| **+47.0%**      |
| LSTM-FCN        | 99.37%        | 0.9937  | 98.93%       | 0.9893 | -0.44%    | -70.0%          |
| LSTM            | 99.14%        | 0.9914  | 98.91%       | 0.9890 | -0.23%    | -27.0%          |
| XGBoost         | 93.91%        | 0.9416  | 89.40%       | 0.8973 | -4.51%    | -74.1%          |
| TransKal        | 99.09%        | 0.9909  | 87.57%       | 0.8484 | **-11.52%**| -1266.5%       |

### Binary Models (Anomaly Detection)

| Model            | Orig Accuracy | Orig F1 | New Accuracy | New F1 | Acc Delta | Error Reduction |
|------------------|---------------|---------|--------------|--------|-----------|-----------------|
| LSTM-Autoencoder | 96.96%        | 0.9820  | 93.93%       | 0.9607 | -3.02%    | -99.5%          |
| Conv-Autoencoder | 99.39%        | 0.9964  | 76.04%       | 0.8639 | **-23.35%**| -3825.8%       |

### Key Findings

1. **CNN-Transformer is the only model that improved** on the new dataset (+0.38% accuracy), demonstrating excellent generalization.

2. **LSTM and LSTM-FCN show robust generalization** with small accuracy drops (<0.5%).

3. **XGBoost degrades moderately** (-4.51%), consistent with its lack of temporal modeling.

4. **TransKal shows severe overfitting** (-11.52%), despite having built-in Kalman filtering.

5. **Conv-Autoencoder shows catastrophic failure** (-23.35%), suggesting its threshold is highly dataset-specific.

---

## 40-xgboost-new-data-evaluation

### Model Description
Evaluates XGBoost on the new TEP dataset. XGBoost treats each sample independently, making it sensitive to distribution shifts.

### Results

| Metric            | Original Test | New Eval | Delta |
|-------------------|---------------|----------|-------|
| Accuracy          | 93.91%        | 89.40%   | -4.51%|
| Balanced Accuracy | 93.91%        | 89.60%   | -4.31%|
| F1 (weighted)     | 0.9416        | 0.8973   | -0.0443|
| F1 (macro)        | 0.9416        | 0.9007   | -0.0409|
| Precision         | 0.9484        | 0.9045   | -0.0439|
| Recall            | 0.9391        | 0.8940   | -0.0451|

### Per-Class F1 Scores

| Class | Original F1 | New F1 | Delta   | Status   |
|-------|-------------|--------|---------|----------|
| 0     | 0.7355      | 0.6227 | -0.1128 | Degraded |
| 1     | 0.9939      | 0.9930 | -0.0009 | Same     |
| 2     | 0.9926      | 0.9919 | -0.0007 | Same     |
| 4     | 0.9672      | 0.9535 | -0.0137 | Degraded |
| 5     | 0.9915      | 0.7806 | -0.2109 | Degraded |
| 6     | 1.0000      | 0.9955 | -0.0045 | Same     |
| 7     | 1.0000      | 0.9993 | -0.0007 | Same     |
| 8     | 0.9572      | 0.9492 | -0.0080 | Degraded |
| 10    | 0.8680      | 0.8314 | -0.0366 | Degraded |
| 11    | 0.8949      | 0.8782 | -0.0167 | Degraded |
| 12    | 0.9378      | 0.9079 | -0.0299 | Degraded |
| 13    | 0.9494      | 0.9417 | -0.0077 | Degraded |
| 14    | 0.9878      | 0.9759 | -0.0119 | Degraded |
| 16    | 0.8945      | 0.8829 | -0.0116 | Degraded |
| 17    | 0.9575      | 0.9418 | -0.0157 | Degraded |
| 18    | 0.9471      | 0.8547 | -0.0924 | Degraded |
| 19    | 0.9422      | 0.9181 | -0.0241 | Degraded |
| 20    | 0.9322      | 0.7941 | -0.1381 | Degraded |

**Summary**: All 18 classes degraded, with Class 5 (-0.21) and Class 20 (-0.14) showing the largest drops.

### Output Files
- Metrics: `outputs/metrics/xgboost_new_eval_metrics.json`
- Figures: `outputs/figures/xgboost_new_eval_confusion_matrix.png`

---

## 41-lstm-new-data-evaluation

### Model Description
Evaluates LSTM on the new TEP dataset. LSTM's recurrent architecture captures temporal dependencies, leading to robust generalization.

### Results

| Metric            | Original Test | New Eval | Delta |
|-------------------|---------------|----------|-------|
| Accuracy          | 99.14%        | 98.91%   | -0.23%|
| Balanced Accuracy | 99.14%        | 98.79%   | -0.35%|
| F1 (weighted)     | 0.9914        | 0.9890   | -0.0024|
| F1 (macro)        | 0.9914        | 0.9888   | -0.0026|

### Per-Class Highlights

| Class | Original F1 | New F1 | Delta   | Status   |
|-------|-------------|--------|---------|----------|
| 0     | ~0.99       | 0.9241 | -0.066  | Degraded |
| 7     | 1.0000      | 1.0000 | +0.0000 | Perfect  |
| 14    | 1.0000      | 1.0000 | +0.0000 | Perfect  |
| 10    | ~0.99       | 0.9686 | -0.021  | Degraded |
| 18    | ~0.99       | 0.9679 | -0.022  | Degraded |

**Summary**: LSTM shows robust generalization with only -0.23% accuracy drop. Most classes maintain >99% F1.

### Output Files
- Metrics: `outputs/metrics/lstm_new_eval_metrics.json`

---

## 42-lstm-fcn-new-data-evaluation

### Model Description
Evaluates LSTM-FCN on the new TEP dataset. Combines recurrent (LSTM) and convolutional (FCN) processing.

### Results

| Metric            | Original Test | New Eval | Delta |
|-------------------|---------------|----------|-------|
| Accuracy          | 99.37%        | 98.93%   | -0.44%|
| Balanced Accuracy | 99.37%        | 98.86%   | -0.51%|
| F1 (weighted)     | 0.9937        | 0.9893   | -0.0044|
| F1 (macro)        | 0.9937        | 0.9887   | -0.0050|

### Per-Class Highlights

| Class | Original F1 | New F1 | Delta   | Status   |
|-------|-------------|--------|---------|----------|
| 0     | ~0.99       | 0.9253 | -0.065  | Degraded |
| 6     | 1.0000      | 1.0000 | +0.0000 | Perfect  |
| 14    | 1.0000      | 1.0000 | +0.0000 | Perfect  |
| 10    | ~0.99       | 0.9498 | -0.040  | Degraded |
| 18    | ~0.99       | 0.9556 | -0.034  | Degraded |

**Summary**: LSTM-FCN shows good generalization with -0.44% accuracy drop. Classes 0, 10, and 18 show the largest degradation.

### Output Files
- Metrics: `outputs/metrics/lstm_fcn_new_eval_metrics.json`

---

## 43-cnn-transformer-new-data-evaluation

### Model Description
Evaluates CNN-Transformer on the new TEP dataset. The Transformer's self-attention mechanism provides global temporal context and excellent generalization.

### Results

| Metric            | Original Test | New Eval | Delta |
|-------------------|---------------|----------|-------|
| Accuracy          | 99.20%        | **99.57%**| **+0.38%**|
| Balanced Accuracy | 99.20%        | 99.44%   | +0.24%|
| F1 (weighted)     | 0.9920        | 0.9958   | +0.0038|
| F1 (macro)        | 0.9920        | 0.9953   | +0.0033|

### Per-Class Highlights

| Class | Original F1 | New F1 | Delta   | Status   |
|-------|-------------|--------|---------|----------|
| 0     | ~0.97       | 0.9688 | ~0.00   | Same     |
| 6     | 1.0000      | 1.0000 | +0.0000 | Perfect  |
| 7     | 1.0000      | 1.0000 | +0.0000 | Perfect  |
| 14    | 1.0000      | 1.0000 | +0.0000 | Perfect  |
| 10    | ~0.99       | 0.9953 | ~0.00   | Same     |

**Summary**: CNN-Transformer is the **only model that improved** on the new dataset. This demonstrates the Transformer architecture's superior generalization capability.

### Output Files
- Metrics: `outputs/metrics/cnn_transformer_new_eval_metrics.json`

---

## 44-transkal-new-data-evaluation

### Model Description
Evaluates TransKal on the new TEP dataset. TransKal combines Transformer with Kalman filtering for temporal smoothing.

### Results

| Metric            | Original Test | New Eval | Delta |
|-------------------|---------------|----------|-------|
| Accuracy          | 99.09%        | 87.57%   | **-11.52%**|
| Balanced Accuracy | 99.09%        | 88.43%   | -10.66%|
| F1 (weighted)     | 0.9909        | 0.8484   | -0.1425|
| F1 (macro)        | 0.9909        | 0.8603   | -0.1306|

### Per-Class F1 Scores (Catastrophic Failures)

| Class | Original F1 | New F1 | Delta   | Status     |
|-------|-------------|--------|---------|------------|
| 0     | ~0.99       | **0.0000** | -0.99 | **Failed** |
| 4     | ~0.99       | **0.2447** | -0.75 | **Failed** |
| 10    | ~0.99       | **0.5510** | -0.44 | **Failed** |
| 6     | 1.0000      | 1.0000 | +0.0000 | Perfect    |
| 7     | 1.0000      | 1.0000 | +0.0000 | Perfect    |
| 14    | 1.0000      | 1.0000 | +0.0000 | Perfect    |

**Summary**: TransKal shows **severe overfitting**. The Kalman filter parameters appear to be highly tuned to the training data distribution. Classes 0, 4, and 10 show catastrophic failure (F1 < 0.6).

### Kalman Filter Analysis
- Raw accuracy (before Kalman): 87.54%
- Filtered accuracy (after Kalman): 87.57%
- Kalman improvement: +0.03% (negligible)

The Kalman filter provides minimal benefit on this new data, suggesting the learned transition matrices don't generalize well.

### Output Files
- Metrics: `outputs/metrics/transkal_new_eval_metrics.json`

---

## 45-lstm-autoencoder-new-data-evaluation

### Model Description
Evaluates LSTM-Autoencoder for binary anomaly detection on the new TEP dataset.

### Results

| Metric    | Original Test | New Eval | Delta |
|-----------|---------------|----------|-------|
| Accuracy  | 96.96%        | 93.93%   | -3.02%|
| Precision | 0.9696        | 0.9466   | -0.0230|
| Recall    | 0.9996        | 0.9753   | -0.0243|
| F1        | 0.9820        | 0.9607   | -0.0212|
| AUC-ROC   | ~0.99         | 0.9821   | -0.008 |

**Summary**: LSTM-Autoencoder shows moderate degradation (-3.02% accuracy). The learned threshold may need recalibration for new data.

### Output Files
- Metrics: `outputs/metrics/lstm_autoencoder_new_eval_metrics.json`

---

## 46-conv-autoencoder-new-data-evaluation

### Model Description
Evaluates Convolutional Autoencoder for binary anomaly detection on the new TEP dataset.

### Results

| Metric    | Original Test | New Eval | Delta |
|-----------|---------------|----------|-------|
| Accuracy  | 99.39%        | 76.04%   | **-23.35%**|
| Precision | 0.9939        | 0.7604   | -0.2335|
| Recall    | 1.0000        | 1.0000   | +0.0000|
| F1        | 0.9964        | 0.8639   | -0.1325|
| AUC-ROC   | ~0.99         | 0.9810   | -0.01  |

**Summary**: Conv-Autoencoder shows **catastrophic failure** on the new dataset. While it maintains 100% recall (catches all faults), precision drops dramatically, indicating many false positives. The threshold (0.023) is far too low for the new data distribution.

### Analysis
- The model's reconstruction error distribution shifted significantly on new data
- High AUC-ROC (0.98) suggests the model still separates classes, but the threshold is miscalibrated
- Threshold recalibration on new data could potentially recover performance

### Output Files
- Metrics: `outputs/metrics/conv_autoencoder_new_eval_metrics.json`

---

## Conclusions

### Generalization Performance Ranking

| Rank | Model           | New Eval Accuracy | Delta from Original | Assessment |
|------|-----------------|-------------------|---------------------|------------|
| 1    | CNN-Transformer | 99.57%            | **+0.38%**          | Excellent  |
| 2    | LSTM-FCN        | 98.93%            | -0.44%              | Good       |
| 3    | LSTM            | 98.91%            | -0.23%              | Good       |
| 4    | LSTM-AE (binary)| 93.93%            | -3.02%              | Moderate   |
| 5    | XGBoost         | 89.40%            | -4.51%              | Poor       |
| 6    | TransKal        | 87.57%            | -11.52%             | Severe     |
| 7    | Conv-AE (binary)| 76.04%            | -23.35%             | Failed     |

### Key Insights

1. **Transformer attention generalizes best**: The CNN-Transformer's self-attention mechanism learns robust temporal patterns that transfer to unseen data.

2. **Recurrent architectures are robust**: LSTM and LSTM-FCN maintain >98.9% accuracy, showing good generalization.

3. **XGBoost suffers without temporal context**: The -4.51% drop confirms that independent sample classification struggles with distribution shift.

4. **Kalman filtering can overfit**: TransKal's catastrophic failure (-11.52%) suggests learned Kalman parameters are dataset-specific.

5. **Autoencoder thresholds are fragile**: Both autoencoders degrade significantly, indicating reconstruction error distributions shift with new data.

### Practical Recommendations

1. **For multiclass fault classification**: Use **CNN-Transformer** for best generalization, or LSTM/LSTM-FCN for good robustness.

2. **Avoid TransKal** for deployment unless Kalman parameters can be adapted online.

3. **Recalibrate autoencoder thresholds** when deploying to new process conditions.

4. **XGBoost with HMM filtering** may improve generalization (see 30-series), but deep learning models are preferred.

### Final Performance Summary

| Task | Best Model | New Eval Performance |
|------|------------|---------------------|
| Multiclass Classification | CNN-Transformer | 99.57% accuracy, 0.9958 F1 |
| Binary Anomaly Detection | LSTM-Autoencoder | 93.93% accuracy, 0.9607 F1 |
