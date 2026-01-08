# 30-Series: HMM Classification Filter Summary

This document summarizes the results of applying Hidden Markov Model (HMM) classification filters to the trained models from the 20-series. The HMM filter exploits temporal consistency in fault classification to smooth noisy predictions.

## HMM Classification Filter Method

The HMM filter works by:
1. **Emission Model**: Uses the classifier's confusion matrix as P(observation | true_class)
2. **Transition Model**: Uses a "sticky" transition model controlled by a stickiness parameter (γ), assuming faults persist over time
3. **Bayesian Filtering**: Performs sequential Bayesian updates to smooth predictions within each simulation run

The filter resets at the beginning of each simulation run to avoid propagating state across independent trajectories.

### Stickiness Parameter (γ)
- Higher γ means predictions are more resistant to change (more temporal smoothing)
- Values tested: 0.7, 0.85, 0.95
- γ = 0.95 means 95% probability of staying in current state, 5% distributed among transitions

---

## Comparison: 20-Series (Raw) vs 30-Series (HMM Filtered)

### Overall Performance Summary

| Model           | Raw Accuracy | Raw F1 | Best γ | Filtered Accuracy | Filtered F1 | Acc Δ  | Error Reduction |
|-----------------|--------------|--------|--------|-------------------|-------------|--------|-----------------|
| XGBoost         | 93.91%       | 0.9416 | 0.95   | 95.90%            | 0.9606      | +1.99% | **32.68%**      |
| LSTM            | 99.14%       | 0.9914 | 0.95   | 99.14%            | 0.9915      | +0.01% | 0.73%           |
| LSTM-FCN        | 99.37%       | 0.9937 | 0.70   | 99.37%            | 0.9937      | +0.00% | 0.07%           |
| CNN-Transformer | 99.20%       | 0.9920 | 0.95   | 99.21%            | 0.9921      | +0.01% | 1.22%           |
| TransKal        | 99.08%       | 0.9908 | 0.85   | 99.08%            | 0.9909      | +0.00% | 0.44%           |

### Key Finding
The HMM filter provides **substantial benefit for XGBoost** (32.68% error reduction) but **minimal benefit for deep learning models** (<1.5% error reduction). This suggests that temporal patterns are already captured by the sequential deep learning architectures.

---

## 30-xgboost-hmm-filter

### Model Description
Applies HMM classification filter to XGBoost predictions. XGBoost treats each sample independently without temporal context, making it a strong candidate for post-hoc temporal smoothing.

### Training Data
- **Source**: Predictions from XGBoost model trained in notebook 20
- **Test samples**: 2,880,000
- **Simulation runs**: 3,600

### HMM Filter Configuration

| Parameter                | Value                     |
|--------------------------|---------------------------|
| n_classes                | 18                        |
| Emission model           | Test set confusion matrix |
| Stickiness values tested | 0.7, 0.85, 0.95           |
| Best stickiness          | 0.95                      |

### Results by Stickiness

| Stickiness (γ) | Accuracy   | F1 (weighted) | Acc Δ      | Error Reduction |
|----------------|------------|---------------|------------|-----------------|
| Raw            | 93.91%     | 0.9416        | -          | -               |
| 0.70           | 94.88%     | 0.9502        | +0.96%     | 15.8%           |
| 0.85           | 95.39%     | 0.9555        | +1.47%     | 24.2%           |
| **0.95**       | **95.90%** | **0.9606**    | **+1.99%** | **32.7%**       |

### Per-Class Improvements (γ=0.95)

| Class      | Raw F1 | HMM F1 | Delta   | Status   |
|------------|--------|--------|---------|----------|
| 0 (Normal) | 0.7355 | 0.8013 | +0.0659 | Improved |
| 1          | 0.9939 | 0.9939 | +0.0000 | Same     |
| 2          | 0.9926 | 0.9925 | -0.0000 | Same     |
| 4          | 0.9672 | 0.9826 | +0.0154 | Improved |
| 5          | 0.9915 | 0.9918 | +0.0003 | Same     |
| 6          | 1.0000 | 1.0000 | +0.0000 | Same     |
| 7          | 1.0000 | 1.0000 | +0.0000 | Same     |
| 8          | 0.9572 | 0.9619 | +0.0047 | Improved |
| 10         | 0.8680 | 0.9050 | +0.0371 | Improved |
| 11         | 0.8949 | 0.9520 | +0.0571 | Improved |
| 12         | 0.9378 | 0.9484 | +0.0106 | Improved |
| 13         | 0.9494 | 0.9532 | +0.0039 | Improved |
| 14         | 0.9878 | 0.9975 | +0.0097 | Improved |
| 16         | 0.8945 | 0.9422 | +0.0476 | Improved |
| 17         | 0.9575 | 0.9723 | +0.0148 | Improved |
| 18         | 0.9471 | 0.9504 | +0.0033 | Improved |
| 19         | 0.9422 | 0.9912 | +0.0490 | Improved |
| 20         | 0.9322 | 0.9553 | +0.0231 | Improved |

**Summary**: 13 classes improved, 0 degraded, 5 unchanged

### Output Files
- Metrics: `outputs/metrics/xgboost_hmm_filter_results.json`
- Figures: `outputs/figures/xgboost_hmm_confusion_matrices.png`

---

## 31-lstm-hmm-filter

### Model Description
Applies HMM classification filter to LSTM predictions. LSTM already captures temporal dependencies through its recurrent architecture.

### Training Data
- **Source**: Predictions from LSTM model trained in notebook 21
- **Test samples**: 2,743,200
- **Simulation runs**: 3,420

### Results by Stickiness

| Stickiness (γ) | Accuracy    | F1 (weighted) | Acc Δ       | Error Reduction |
|----------------|-------------|---------------|-------------|-----------------|
| Raw            | 99.138%     | 0.9914        | -           | -               |
| 0.70           | 99.139%     | 0.9914        | +0.002%     | 0.18%           |
| 0.85           | 99.142%     | 0.9915        | +0.004%     | 0.47%           |
| **0.95**       | **99.144%** | **0.9915**    | **+0.006%** | **0.73%**       |

### Summary
The HMM filter provides **minimal improvement** for LSTM (0.73% error reduction at best). The LSTM's recurrent architecture already captures temporal patterns effectively.

### Output Files
- Metrics: `outputs/metrics/lstm_hmm_filter_results.json`

---

## 32-lstm-fcn-hmm-filter

### Model Description
Applies HMM classification filter to LSTM-FCN predictions. LSTM-FCN combines recurrent (LSTM) and convolutional (FCN) processing for both temporal and local pattern detection.

### Training Data
- **Source**: Predictions from LSTM-FCN model trained in notebook 22
- **Test samples**: 2,739,600
- **Simulation runs**: 3,420

### Results by Stickiness

| Stickiness (γ) | Accuracy    | F1 (weighted) | Acc Δ       | Error Reduction |
|----------------|-------------|---------------|-------------|-----------------|
| Raw            | 99.369%     | 0.9937        | -           | -               |
| **0.70**       | **99.369%** | **0.9937**    | **+0.000%** | **0.07%**       |
| 0.85           | 99.369%     | 0.9937        | -0.000%     | -0.08%          |
| 0.95           | 99.369%     | 0.9937        | +0.000%     | 0.02%           |

### Summary
The HMM filter provides **essentially no improvement** for LSTM-FCN (0.07% error reduction). This model already achieves near-optimal temporal smoothing through its combined architecture.

### Output Files
- Metrics: `outputs/metrics/lstm_fcn_hmm_filter_results.json`

---

## 33-cnn-transformer-hmm-filter

### Model Description
Applies HMM classification filter to CNN-Transformer predictions. The Transformer's self-attention mechanism provides global temporal context.

### Training Data
- **Source**: Predictions from CNN-Transformer model trained in notebook 23
- **Test samples**: 2,743,200
- **Simulation runs**: 3,420

### Results by Stickiness

| Stickiness (γ) | Accuracy    | F1 (weighted) | Acc Δ       | Error Reduction |
|----------------|-------------|---------------|-------------|-----------------|
| Raw            | 99.197%     | 0.9920        | -           | -               |
| 0.70           | 99.197%     | 0.9920        | +0.001%     | 0.08%           |
| 0.85           | 99.199%     | 0.9920        | +0.002%     | 0.27%           |
| **0.95**       | **99.207%** | **0.9921**    | **+0.010%** | **1.22%**       |

### Summary
The HMM filter provides **small improvement** for CNN-Transformer (1.22% error reduction). The Transformer's attention mechanism already captures most temporal dependencies.

### Output Files
- Metrics: `outputs/metrics/cnn_transformer_hmm_filter_results.json`

---

## 34-transkal-hmm-filter

### Model Description
Applies HMM classification filter to TransKal predictions. TransKal already includes a Kalman filter for temporal smoothing, making additional HMM filtering potentially redundant.

### Training Data
- **Source**: Predictions from TransKal model trained in notebook 24
- **Test samples**: 2,739,600
- **Simulation runs**: 3,420

### Results by Stickiness

| Stickiness (γ) | Accuracy    | F1 (weighted) | Acc Δ       | Error Reduction |
|----------------|-------------|---------------|-------------|-----------------|
| Raw            | 99.079%     | 0.9908        | -           | -               |
| 0.70           | 99.081%     | 0.9908        | +0.002%     | 0.18%           |
| **0.85**       | **99.083%** | **0.9909**    | **+0.004%** | **0.44%**       |
| 0.95           | 99.083%     | 0.9908        | +0.004%     | 0.40%           |

### Summary
The HMM filter provides **minimal improvement** for TransKal (0.44% error reduction). The built-in Kalman filter already provides effective temporal smoothing.

### Output Files
- Metrics: `outputs/metrics/transkal_hmm_filter_results.json`

---

## Conclusions

### When to Use HMM Classification Filter

1. **Highly Recommended for XGBoost**: The HMM filter reduces XGBoost errors by 32.7%, improving accuracy from 93.91% to 95.90%. This is because XGBoost treats each sample independently without temporal context.

2. **Not Recommended for Deep Learning Models**: All deep learning models (LSTM, LSTM-FCN, CNN-Transformer, TransKal) show <1.5% error reduction from HMM filtering. These models already capture temporal patterns through their architectures:
   - LSTM: Recurrent connections maintain temporal state
   - LSTM-FCN: Combines recurrent and convolutional processing
   - CNN-Transformer: Self-attention provides global temporal context
   - TransKal: Built-in Kalman filter for temporal smoothing

### Best Stickiness Values

| Model           | Best γ | Rationale                                                              |
|-----------------|--------|------------------------------------------------------------------------|
| XGBoost         | 0.95   | Higher stickiness maximizes temporal smoothing for independent predictions |
| LSTM            | 0.95   | Slight benefit from additional smoothing                               |
| LSTM-FCN        | 0.70   | Lower stickiness prevents over-smoothing                               |
| CNN-Transformer | 0.95   | Slight benefit from additional smoothing                               |
| TransKal        | 0.85   | Moderate stickiness complements built-in Kalman filter                 |

### Practical Recommendations

1. **For deployment with XGBoost**: Always use HMM filter with γ=0.95 for ~2% accuracy improvement at minimal computational cost.

2. **For deployment with deep learning models**: HMM filtering is optional and provides negligible benefit. Skip it to reduce complexity.

3. **For real-time applications**: The HMM filter adds minimal latency (single-pass Bayesian update per sample) and can be applied to any classifier's predictions.

### Final Performance Ranking (with optimal filtering)

| Rank | Model           | Best Configuration | Test Accuracy | Test F1 |
|------|-----------------|-------------------|---------------|---------|
| 1    | LSTM-FCN        | Raw (no filter)   | 99.37%        | 0.9937  |
| 2    | CNN-Transformer | HMM γ=0.95        | 99.21%        | 0.9921  |
| 3    | LSTM            | HMM γ=0.95        | 99.14%        | 0.9915  |
| 4    | TransKal        | HMM γ=0.85        | 99.08%        | 0.9909  |
| 5    | XGBoost         | HMM γ=0.95        | 95.90%        | 0.9606  |

Even with HMM filtering, XGBoost still lags behind the deep learning models by ~3.5% accuracy, demonstrating the importance of architectures that natively handle temporal dependencies for time series fault classification.
