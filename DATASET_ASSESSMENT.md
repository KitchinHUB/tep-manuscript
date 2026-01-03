# TEP Dataset Assessment for Manuscript

**Date**: 2026-01-03
**Dataset Version**: Medium-sized, Balanced
**Total Size**: 4.2M samples (1.9 GB)

---

## Executive Summary

The current medium-sized, perfectly balanced TEP dataset is **suitable for benchmarking machine learning methods** in a controlled experimental setting. However, it has **significant limitations** for claims about real-world industrial deployment. This assessment provides both strengths and critical flaws for transparent reporting in the manuscript.

---

## Why This Dataset IS Suitable for the Manuscript

### 1. **Perfect Experimental Control** ✓

**Strength**: Zero confounding variables from class imbalance
- All 18 classes have exactly 48,000/24,000/160,000 samples (train/val/test)
- Standard deviation = 0.00 across all classes
- Eliminates the need for class weighting, resampling, or specialized loss functions
- **Impact**: Pure algorithm comparison - differences in performance reflect model capabilities, not dataset artifacts

**Justification**:
- Enables fair comparison across diverse model families (tree-based, neural networks, SVMs)
- Published benchmarks (MNIST, CIFAR-10, ImageNet) use balanced datasets for the same reason
- Balanced accuracy, F1-score, and standard accuracy all align - simplifies reporting

### 2. **Strong Statistical Power** ✓

**Strength**: Can detect small performance differences reliably
- Test set: 2.88M samples total (160,000 per class)
- 95% confidence interval width: **±0.5%** at 80% accuracy (vs ±2% for previous version)
- Can reliably detect performance differences as small as **1-2 percentage points**

**Justification**:
- Adequate for publication-quality statistical comparisons
- Meets or exceeds typical ML benchmark test set sizes:
  - MNIST: 10K test samples (10 classes)
  - CIFAR-10: 10K test samples (10 classes)
  - Our dataset: 2.88M test samples (18 classes) - **287× larger than MNIST**
- Per-class metrics (precision, recall, F1) will be stable and reproducible

### 3. **Sufficient Training Data for Modern Methods** ✓

**Strength**: 5× increase enables deep learning
- Training: 864K samples total (48K per class)
- Validation: 432K samples (24K per class)
- Deep neural networks typically need 10K-100K samples per class - we're in that range

**Justification**:
- Previous 9.6K samples/class was too sparse for deep learning (would underfit)
- Current 48K samples/class is adequate for:
  - Multi-layer perceptrons (MLPs): 3-5 layers, 100-500 neurons
  - 1D CNNs: 3-5 conv layers
  - LSTMs/GRUs: 2-3 layers, 64-128 units
  - Small Transformers: 2-4 attention heads
- Still in "sample-efficient" regime - tests models' ability to learn from limited data

### 4. **Zero Data Leakage** ✓

**Strength**: Rigorous train/test separation
- Training/validation from `ftr` source (faulty training runs)
- Test from `fte` source (faulty testing runs - completely independent)
- Zero trajectory overlap verified (train ∩ test = ∅)
- Different fault introduction times (sample 21 vs 161) - prevents temporal memorization

**Justification**:
- Ensures test performance reflects true generalization, not memorization
- Meets best practices for ML evaluation
- Independent test source mimics real-world scenario (deployed model sees new runs)

### 5. **Realistic Process Dynamics** ✓

**Strength**: Captures real industrial process behavior
- Based on Tennessee Eastman Process (TEP) - widely-used chemical process benchmark
- 52 features: 41 measurements + 11 manipulated variables (realistic sensor suite)
- Temporal dynamics: 480-800 samples per trajectory (8-13 hours at 3-min intervals)
- Fault types: Step changes, drifts, sensor biases (not just simple noise)

**Justification**:
- TEP has been used in 500+ publications since 1993
- Represents real industrial complexity (material balance, heat transfer, reaction kinetics)
- Fault library covers common failure modes: valve sticking, sensor drift, feed composition changes

### 6. **Manageable Computational Requirements** ✓

**Strength**: Fast iteration for algorithm development
- Total size: 1.9 GB (fits easily in memory)
- Training time: Expected 3-5× slower than minimal, but still <1 hour for most models
- Can train on laptop/workstation (doesn't require HPC cluster)

**Justification**:
- Enables rapid prototyping and hyperparameter tuning
- Multiple research groups can reproduce results
- Reasonable for classroom/educational use

---

## Critical Flaws and Limitations

### **CRITICAL FLAW #1: Unrealistic Class Balance** ⚠️

**Problem**: Perfect 1:1 balance does not reflect real-world operations

**Reality**:
- Industrial processes: Normal operation ≈ 95-99%, Faults ≈ 1-5%
- Our dataset: Normal = 5.56% (1/18), Each fault = 5.56%
- **Imbalance ratio**: Real-world ~50:1, Our dataset 1:1

**Impact**:
- Models trained on balanced data will have poor calibration in deployment
- Predicted probabilities will be unreliable (overestimate fault likelihood)
- High false alarm rate in production (predicts faults too often)
- Precision-recall tradeoff not representative

**Manuscript Implications**:
- ❌ **Cannot claim** results apply to real industrial deployment
- ❌ **Cannot claim** model is "production-ready"
- ✓ **Can claim** fair comparison of algorithm performance in controlled setting
- **Required**: Add note that real-world deployment requires recalibration and imbalanced evaluation

**Mitigation**:
- Create separate imbalanced test set (90% normal, 10% faults) for realistic evaluation
- Report both balanced and imbalanced performance
- Discuss calibration techniques (Platt scaling, isotonic regression)

### **CRITICAL FLAW #2: Small Training Set for Deep Learning** ⚠️

**Problem**: 48K samples/class is at the low end for deep learning

**Reality**:
- ImageNet: 1.28M training images (1,000 classes) ≈ 1,280 per class (but much larger models)
- Modern vision transformers: Trained on 300M+ images
- Our dataset: 48K samples/class for 18 classes = 864K total

**Impact**:
- Deep neural networks may still underfit compared to their potential
- Cannot fairly compare to state-of-the-art deep learning (would need >1M samples/class)
- Simple models (Random Forest, XGBoost) may outperform deep learning **due to data size, not model quality**

**Manuscript Implications**:
- ❌ **Cannot claim** deep learning is inferior if it underperforms
- ❌ **Cannot claim** comprehensive deep learning evaluation
- ✓ **Can claim** sample-efficient learning comparison
- **Required**: Label this as "limited data regime" evaluation

**Justification for Current Size**:
- We have 12.4M samples available but chose 4.2M for training speed
- Could increase to 4.3M training samples (500 runs × 480 samples × 18 classes) if needed
- Current size balances statistical power vs. iteration speed

### **CRITICAL FLAW #3: Temporal Downsampling Artifacts** ⚠️

**Problem**: Arbitrary truncation of normal class trajectories

**Details**:
- Normal training runs: 500 samples → **downsampled to 480** (lost 4% of data)
- Normal test runs: 960 samples → **downsampled to 800** (lost 17% of data)
- Fault runs: Use post-fault samples only (no downsampling within fault region)

**Impact**:
- Normal class loses information from samples 481-500 (training) and 801-960 (test)
- Models may not see the full range of normal operation dynamics
- Temporal patterns at end of normal runs are censored
- **Different effective trajectory lengths** (480 vs 800) between train and test

**Why This Happened**:
- To balance sample counts: normal (500/run) vs. fault (480/run post-fault)
- Simpler than finding runs with exactly 480 normal samples

**Manuscript Implications**:
- ⚠️ Must document this downsampling in methods section
- ⚠️ Potential criticism: Why not use full trajectories with weighted sampling?
- Alternative: Use class weights instead of downsampling

### **CRITICAL FLAW #4: Limited Fault Diversity** ⚠️

**Problem**: Only 17 fault types, each with single introduction point

**Reality**:
- Real processes: Hundreds of possible failure modes
- Faults can be: gradual vs. abrupt, intermittent vs. persistent, single vs. compound
- Our dataset: 17 faults, each introduced at fixed time (sample 21 or 161)

**Impact**:
- Models may overfit to these specific 17 fault signatures
- No generalization test to **novel fault types** (the most important real-world scenario)
- No multi-fault scenarios (e.g., valve sticking + sensor drift simultaneously)
- No partial faults or degradation trajectories

**Manuscript Implications**:
- ❌ **Cannot claim** fault detection system generalizes to novel faults
- ✓ **Can claim** classification performance on 17 known fault types
- **Required**: Discuss generalization limitations, need for anomaly detection methods

**Data Available**:
- Original TEP has 20 faults (we excluded 3, 9, 15 as "too subtle")
- Excluding subtle faults may create optimistic performance (easier classification task)

### **CRITICAL FLAW #5: No Temporal Validation Strategy** ⚠️

**Problem**: IID sampling doesn't test time-series capabilities

**Details**:
- Current split: Random trajectories to train/val/test (i.i.d. assumption)
- Real deployment: Must detect faults in continuous operation (temporal dependency)
- No evaluation of: Detection delay, false alarm rate over time, early detection capability

**Impact**:
- Sample-level accuracy doesn't measure fault detection speed
- Model could have 90% accuracy but detect faults too late to prevent damage
- No measure of how many samples after fault onset the model flags it

**Manuscript Implications**:
- ❌ **Cannot claim** real-time fault detection capability
- ❌ **Cannot claim** early warning system performance
- ✓ **Can claim** classification accuracy on post-fault samples
- **Recommended**: Add time-series metrics:
  - Detection delay (samples from fault onset to first detection)
  - False alarm rate (alarms per normal trajectory)
  - ROC-AUC over detection threshold

### **FLAW #6: Binary Dataset Is Unbalanced** ⚠️

**Problem**: Binary dataset has different class balance than multiclass

**Details**:
- Binary training: 50K normal (100 runs), 0 faulty (anomaly detection setup)
- Binary test: 115K normal (120 runs), 680K faulty (17 faults × 50 runs × 800 samples)
- Imbalance ratio: 1:5.9 (faulty:normal) - **opposite of real world**

**Impact**:
- Binary classification favors high recall (catches faults) at cost of precision (false alarms)
- Not comparable to multiclass results (different class balance)
- Models can achieve high accuracy by predicting "faulty" for most samples

**Manuscript Implications**:
- ⚠️ Must report precision-recall curves, not just accuracy
- ⚠️ Must use balanced accuracy or F1-score
- Cannot directly compare multiclass vs. binary performance

### **FLAW #7: Feature Engineering Not Explored** ⚠️

**Problem**: Using raw sensor values may not be optimal

**Details**:
- 52 raw features (xmeas, xmv) used directly
- No domain knowledge features:
  - Material balance residuals
  - Heat transfer violations
  - Reaction kinetics deviations
  - Statistical process control (SPC) features (moving average, EWMA)
  - Derivative/rate-of-change features

**Impact**:
- May underestimate performance with proper feature engineering
- Favors deep learning (learns features) vs. classical ML (needs good features)
- Not representative of how industrial fault detection is actually done

**Manuscript Implications**:
- ✓ Good for "fair comparison" (all methods use same raw features)
- ⚠️ Should acknowledge this is a limitation
- ⚠️ Real systems would use engineered features

---

## Recommendations for Manuscript

### **What You CAN Claim** ✓

1. "Fair comparison of ML algorithms on balanced multiclass classification"
2. "Sufficient statistical power to detect 1-2% performance differences"
3. "Zero data leakage, rigorous train/test separation"
4. "Benchmark dataset for sample-efficient learning (48K samples/class)"
5. "Controlled experimental setting with known ground truth"

### **What You CANNOT Claim** ❌

1. ❌ "Model ready for industrial deployment"
2. ❌ "Performance representative of real-world imbalanced scenarios"
3. ❌ "Generalizes to novel fault types not in training set"
4. ❌ "Real-time fault detection capability"
5. ❌ "Comprehensive deep learning evaluation" (data size still limited)

### **Required Disclosures**

**In Methods Section**:
- Document normal class downsampling (480/800 samples)
- Explain class balance choice and limitations
- Note excluded faults (3, 9, 15)
- Describe temporal segmentation (sample 21 vs 161)

**In Results Section**:
- Report both balanced and per-class metrics
- Include confusion matrices
- Discuss failure modes (which faults are hard to detect)

**In Discussion Section**:
- Acknowledge perfect balance is unrealistic
- Recommend imbalanced evaluation for real-world validation
- Discuss calibration needs for deployment
- Note generalization limitations to novel faults

### **Suggested Additional Experiments**

1. **Imbalanced Test Set**: Create 90% normal / 10% faulty test set
2. **Time-Series Metrics**: Measure detection delay, false alarm rate
3. **Ablation Study**: Test on excluded faults (3, 9, 15) to measure generalization
4. **Feature Engineering**: Compare raw vs. engineered features
5. **Calibration Analysis**: Plot reliability diagrams

---

## Overall Assessment

**Verdict**: **SUITABLE for algorithm benchmarking, UNSUITABLE for deployment claims**

### Strengths:
- ✓ Perfect class balance enables fair comparison
- ✓ Strong statistical power (2.88M test samples)
- ✓ Zero data leakage
- ✓ 5× more training data than previous version
- ✓ Manageable size for reproducibility

### Critical Weaknesses:
- ⚠️ **Unrealistic class balance** (biggest limitation)
- ⚠️ Still limited training data for deep learning
- ⚠️ No temporal evaluation metrics
- ⚠️ Limited fault diversity (17 types only)
- ⚠️ No generalization to novel faults

### Bottom Line:
This dataset is **excellent for a controlled ML benchmark study** comparing algorithm performance under identical conditions. It is **not suitable** for making claims about real-world industrial fault detection without additional imbalanced evaluation and calibration studies.

**Recommended Manuscript Positioning**:
> "This study provides a controlled benchmark for comparing machine learning algorithms on multiclass fault classification using perfectly balanced data. While this enables fair algorithmic comparison, real-world deployment would require evaluation on imbalanced data, calibration for production use, and validation on novel fault types."

---

## Action Items for Publication-Quality Work

### High Priority (Required):
1. [ ] Add imbalanced test set evaluation (90% normal)
2. [ ] Report precision-recall curves for all methods
3. [ ] Include confusion matrices in supplementary materials
4. [ ] Document downsampling methodology clearly
5. [ ] Add calibration plots (reliability diagrams)

### Medium Priority (Recommended):
6. [ ] Compute time-series detection metrics (delay, FAR)
7. [ ] Test generalization on excluded faults (3, 9, 15)
8. [ ] Compare to feature-engineered baselines
9. [ ] Analyze failure modes (which faults confuse models)

### Low Priority (Nice to have):
10. [ ] Create "large" dataset version (500 runs/class) for deep learning
11. [ ] Multi-fault scenarios (synthetic combinations)
12. [ ] Transfer learning experiments (pre-train on large, fine-tune on small)

---

**Document Version**: 1.0
**Last Updated**: 2026-01-03
**Author**: Dataset Assessment for TEP Manuscript
