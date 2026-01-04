# Dataset Description for Supporting Information

## Tennessee Eastman Process Fault Detection Datasets

**Version**: Medium-sized, Balanced
**Date Created**: 2026-01-03
**Total Size**: 4.2 million samples (1.9 GB)
**Source**: Tennessee Eastman Process Simulation (Downs & Vogel, 1993)

---

## Overview

This document describes two complementary datasets derived from the Tennessee Eastman Process (TEP) simulation for fault detection research: a **multiclass classification dataset** for identifying specific fault types, and a **binary classification dataset** for anomaly detection. Both datasets are designed to provide rigorous benchmarks for machine learning algorithm evaluation in industrial process monitoring scenarios.

---

## Dataset Specifications

### Multiclass Classification Dataset

The multiclass dataset supports 18-way classification: normal operation (class 0) plus 17 fault types (classes 1, 2, 4-8, 10-14, 16-20).

**Split Sizes:**
- **Training Set**: 864,000 samples (327 MB)
  - 18 classes × 48,000 samples per class
  - 100 runs per class × 480 samples per run

- **Validation Set**: 432,000 samples (164 MB)
  - 18 classes × 24,000 samples per class
  - 50 runs per class × 480 samples per run

- **Test Set**: 2,880,000 samples (1.1 GB)
  - 18 classes × 160,000 samples per class
  - 200 runs per class × 800 samples per run

**Class Balance**: Perfectly balanced with exactly equal sample counts across all 18 classes (standard deviation = 0.00).

### Binary Classification Dataset

The binary dataset supports anomaly detection, with training on normal operation only and testing on both normal and faulty conditions.

**Split Sizes:**
- **Training Set**: 50,000 samples (19 MB)
  - Class 0 (normal): 100 runs × 500 samples per run
  - Fault classes: None (unsupervised training)

- **Validation Set**: 25,000 samples (9.5 MB)
  - Class 0 (normal): 50 runs × 500 samples per run
  - Fault classes: None

- **Test Set**: 795,200 samples (300 MB)
  - Class 0 (normal): 115,200 samples (120 runs × 960 samples)
  - Fault classes: 680,000 samples (17 faults × 50 runs × 800 samples)

**Class Balance**: Intentionally imbalanced (1:5.9 normal:faulty ratio) to evaluate anomaly detection performance.

---

## Data Sources and Partitioning

### Source Files

Data originates from two independent sets of TEP simulation runs:

1. **Faulty Training Runs (ftr)**: Used for training and validation splits
   - 500 simulation runs per fault type
   - Faults introduced at sample 21 (after 20 samples of normal operation)

2. **Faulty Testing Runs (fte)**: Used exclusively for test splits
   - 500 simulation runs per fault type
   - Faults introduced at sample 161 (after 160 samples of normal operation)

This strict separation ensures **zero data leakage** between training and testing phases, with completely independent simulation trajectories and different fault introduction timings.

### Run Allocation Strategy

From the 500 available runs per fault type, runs were allocated to maximize test set size while ensuring adequate training data:

- **Multiclass Dataset**:
  - Training: 100 runs (20% of available)
  - Validation: 50 runs (10% of available)
  - Test: 200 runs (40% of available, from independent source)
  - Remaining: 150 runs per fault reserved for future use

- **Binary Dataset**:
  - Training: 100 normal runs (normal operation only)
  - Validation: 50 normal runs (normal operation only)
  - Test: 120 normal runs + 50 runs per fault (17 faults)

### Trajectory Identification

Each sample is tagged with a unique trajectory key (`traj_key`) in the format:
- Training/Validation: `ftr_f{fault}_r{run}` (e.g., `ftr_f1_r042`)
- Test: `fte_f{fault}_r{run}` (e.g., `fte_f1_r142`)

This enables trajectory-level analysis and verification of zero overlap between splits.

---

## Class Balance Methodology

### Multiclass Dataset: Perfect Balance

To achieve perfect class balance, we addressed the inherent asymmetry between normal and fault trajectories:

**Challenge**: Normal operation runs contain 500 samples (full trajectory), while fault runs contain only 480 samples post-fault (samples 21-500). This creates a 500:480 imbalance.

**Solution**: Downsample normal class trajectories to match fault class sample counts:

1. **Training/Validation Sets**:
   - Normal runs: Truncated from 500 to **480 samples** (samples 1-480)
   - Fault runs: Used **480 post-fault samples** (samples 21-500)
   - Result: 48,000 samples per class (train), 24,000 samples per class (val)

2. **Test Set**:
   - Normal runs: Truncated from 960 to **800 samples** (samples 1-800)
   - Fault runs: Used **800 post-fault samples** (samples 161-960)
   - Result: 160,000 samples per class (test)

**Verification**: Class distribution statistics confirm standard deviation = 0.00 across all 18 classes in each split.

### Binary Dataset: Intentional Imbalance

The binary dataset maintains the natural imbalance between normal and faulty operation to simulate anomaly detection scenarios:

- Training uses only normal operation (unsupervised/semi-supervised setup)
- Test set has 1:5.9 normal:faulty ratio
- This imbalance tests the model's ability to detect rare anomalous events

**Note**: This differs from real-world industrial settings where normal operation comprises 95-99% of data. The 1:5.9 ratio provides sufficient faulty samples for robust evaluation while maintaining anomaly detection characteristics.

---

## Features

### Feature Set

Each sample contains **52 continuous process variables**:

- **41 Process Measurements** (`xmeas_1` through `xmeas_41`):
  - Reactor pressure, level, temperature
  - Separator levels, pressures, temperatures
  - Stripper levels, pressures, temperatures
  - Compressor work
  - Flow rates (feeds, products, purge)
  - Component concentrations (A, B, C, D, E, F, G, H)

- **11 Manipulated Variables** (`xmv_1` through `xmv_11`):
  - Valve positions (D feed, E feed, A feed, A+C feed)
  - Compressor recycle valve
  - Purge valve
  - Separator pot liquid flow
  - Stripper liquid product flow
  - Stripper steam valve
  - Reactor cooling water flow
  - Condenser cooling water flow

### Additional Metadata

Each sample includes:
- `faultNumber`: Class label (0-20, excluding 3, 9, 15)
- `simulationRun`: Run identifier within fault type
- `sample`: Time index within trajectory
- `traj_key`: Unique trajectory identifier (origin_f{fault}_r{run})

### Data Quality

- **Missing Values**: Zero missing values across all features and splits
- **Normalization**: Raw sensor values (not normalized) to preserve physical interpretability
- **Temporal Resolution**: 3-minute sampling intervals (0.05 hours)
- **Trajectory Length**: 480-960 samples (24-48 hours of operation)

---

## Fault Types

### Included Faults (17 total)

The dataset includes faults 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20 from the original TEP fault library:

| Fault | Description | Type |
|-------|-------------|------|
| 1 | A/C feed ratio, B composition constant (stream 4) | Step |
| 2 | B composition, A/C ratio constant (stream 4) | Step |
| 4 | Reactor cooling water inlet temperature | Step |
| 5 | Condenser cooling water inlet temperature | Step |
| 6 | A feed loss (stream 1) | Step |
| 7 | C header pressure loss - reduced availability (stream 4) | Step |
| 8 | A, B, C feed composition (stream 4) | Random variation |
| 10 | C feed temperature (stream 4) | Random variation |
| 11 | Reactor cooling water inlet temperature | Random variation |
| 12 | Condenser cooling water inlet temperature | Random variation |
| 13 | Reaction kinetics | Slow drift |
| 14 | Reactor cooling water valve | Sticking |
| 16 | Unknown | Unknown |
| 17 | Unknown | Unknown |
| 18 | Unknown | Unknown |
| 19 | Unknown | Unknown |
| 20 | Unknown | Unknown |

### Excluded Faults

Faults **3, 9, and 15** were excluded from the dataset as they are documented in the TEP literature as having minimal observable effects on process measurements, making them unsuitable for fault detection benchmarking.

### Fault Introduction Timing

- **Training/Validation**: Faults introduced at sample 21 (1 hour into operation)
- **Test**: Faults introduced at sample 161 (8 hours into operation)

This temporal difference prevents models from memorizing the fault onset time and ensures realistic evaluation where fault timing is unknown.

---

## Data Leakage Prevention

### Zero Trajectory Overlap

**Verification Method**: Trajectory key intersection analysis confirms zero overlap:
- Multiclass: `len(train_traj ∩ test_traj) = 0` ✓
- Binary: `len(train_traj ∩ test_traj) = 0` ✓

### Independent Sources

- Training/Validation: Faulty Training Runs (ftr) source
- Test: Faulty Testing Runs (fte) source
- No simulation run appears in both sources

### Different Fault Timing

- Training: Fault onset at sample 21
- Test: Fault onset at sample 161
- Prevents temporal pattern memorization

---

## Statistical Properties

### Sample Size Justification

**Test Set Statistical Power**:
- Test samples per class: 160,000
- At 80% accuracy, 95% confidence interval: **±0.5%**
- Can reliably detect performance differences of **1-2 percentage points**

This exceeds typical ML benchmark test set sizes:
- MNIST: 10,000 test samples (10 classes) = 1,000 per class
- CIFAR-10: 10,000 test samples (10 classes) = 1,000 per class
- Our dataset: 2,880,000 test samples (18 classes) = 160,000 per class (**160× larger per class**)

**Training Set Size**:
- Training samples per class: 48,000
- Adequate for sample-efficient deep learning (MLPs, 1D CNNs, LSTMs)
- Sufficient for tree-based methods (Random Forest, XGBoost)
- Validation set (24,000/class) enables robust hyperparameter tuning

### Class Distribution Statistics

**Multiclass Dataset**:
- Mean samples per class (train): 48,000
- Standard deviation: 0.00
- Coefficient of variation: 0.00
- Perfect balance achieved

**Binary Dataset**:
- Test set normal: 115,200 samples (14.5%)
- Test set faulty: 680,000 samples (85.5%)
- Imbalance ratio: 1:5.9

---

## Comparison to Previous Versions

This medium-sized, balanced dataset represents a significant improvement over earlier minimal versions:

| Metric | Previous (Minimal) | Current (Medium) | Improvement |
|--------|-------------------|------------------|-------------|
| Train samples/class | 9,600 | 48,000 | **5×** |
| Test samples/class | 12,000 | 160,000 | **13×** |
| Test 95% CI width | ±2.0% | ±0.5% | **4× tighter** |
| Class balance (std) | 0.00 | 0.00 | Maintained |
| Total size | 971K samples | 4.2M samples | **4.3×** |

---

## Use Cases

### Multiclass Classification

The multiclass dataset is designed for:
1. **Fault Diagnosis**: Identifying which specific fault has occurred
2. **Algorithm Benchmarking**: Fair comparison of classification methods
3. **Feature Selection**: Identifying informative process variables
4. **Model Architecture Studies**: Comparing deep learning vs. classical ML

**Evaluation Metrics**: Balanced accuracy, per-class precision/recall/F1, confusion matrix, macro/micro-averaged metrics

### Binary Classification (Anomaly Detection)

The binary dataset is designed for:
1. **Fault Detection**: Distinguishing normal from abnormal operation
2. **Unsupervised Learning**: Training on normal data only
3. **One-Class Classification**: Anomaly detection methods
4. **Threshold Optimization**: ROC curve analysis, precision-recall tradeoffs

**Evaluation Metrics**: ROC-AUC, precision-recall curves, false alarm rate, detection rate

---

## Limitations and Considerations

### Known Limitations

1. **Perfect Class Balance is Unrealistic**:
   - Real industrial processes: normal ≈ 95-99%, faults ≈ 1-5%
   - Our multiclass dataset: all classes equal at 5.56%
   - **Impact**: Models trained on balanced data will overestimate fault probabilities in deployment
   - **Mitigation**: Results represent controlled benchmarking, not production-ready systems

2. **Temporal Downsampling**:
   - Normal class trajectories truncated to match fault class length
   - Training: 500 → 480 samples (lost 4% of normal operation data)
   - Test: 960 → 800 samples (lost 17% of normal operation data)
   - **Impact**: End-of-trajectory dynamics censored for normal class
   - **Alternative**: Could use class weights instead of downsampling

3. **Limited Fault Diversity**:
   - Only 17 fault types (excluded 3, 9, 15 as too subtle)
   - Single fault introduction point per fault type
   - No multi-fault scenarios or gradual degradation
   - **Impact**: Cannot evaluate generalization to novel fault types

4. **Sample-Level Classification**:
   - Data organized as independent samples, not time-series sequences
   - No temporal evaluation metrics (detection delay, false alarm rate)
   - **Impact**: Results measure classification accuracy, not early detection capability

### Recommended Additional Analyses

For comprehensive fault detection research, we recommend supplementing this dataset with:
1. **Imbalanced Test Set**: Create 90% normal / 10% faulty split for realistic evaluation
2. **Time-Series Metrics**: Measure detection delay (samples from fault onset to detection)
3. **Calibration Studies**: Reliability diagrams, Platt scaling for deployment
4. **Generalization Tests**: Evaluate on excluded faults (3, 9, 15)

---

## Data Access

### File Locations

All dataset files are stored in CSV format in the `data/` directory:

**Multiclass Dataset**:
- `data/multiclass_train.csv` (327 MB)
- `data/multiclass_val.csv` (164 MB)
- `data/multiclass_test.csv` (1.1 GB)

**Binary Dataset**:
- `data/binary_train.csv` (19 MB)
- `data/binary_val.csv` (9.5 MB)
- `data/binary_test.csv` (300 MB)

### Loading Instructions

```python
import pandas as pd

# Load multiclass dataset
mc_train = pd.read_csv('data/multiclass_train.csv')
mc_val = pd.read_csv('data/multiclass_val.csv')
mc_test = pd.read_csv('data/multiclass_test.csv')

# Load binary dataset
bin_train = pd.read_csv('data/binary_train.csv')
bin_val = pd.read_csv('data/binary_val.csv')
bin_test = pd.read_csv('data/binary_test.csv')
```

### Column Structure

- Features: `xmeas_1` through `xmeas_41`, `xmv_1` through `xmv_11`
- Label: `faultNumber` (0-20, excluding 3, 9, 15)
- Metadata: `simulationRun`, `sample`, `traj_key`

---

## Reproducibility

### Dataset Generation

The complete dataset generation pipeline is available in:
- Notebook: `notebooks/01-create-datasets.ipynb`
- Automation: `make datasets` (via Makefile)

### Verification

Exploratory data analysis confirming dataset quality is available in:
- Notebook: `notebooks/02-exploratory-data-analysis.ipynb`
- Summary: `outputs/eda_summary.txt`

### Version Control

All dataset creation scripts, analysis notebooks, and documentation are version-controlled in the project repository with full commit history.

---

## Citation

If using these datasets, please cite:

> [Your manuscript citation]

And the original Tennessee Eastman Process:

> Downs, J. J., & Vogel, E. F. (1993). A plant-wide industrial process control problem. *Computers & Chemical Engineering*, 17(3), 245-255.

---

## Contact

For questions about dataset construction, known issues, or requests for alternative configurations (e.g., larger training sets, imbalanced variants), please contact [contact information].

---

**Document Version**: 1.0
**Last Updated**: 2026-01-03
**Dataset Version**: Medium-sized, Balanced
