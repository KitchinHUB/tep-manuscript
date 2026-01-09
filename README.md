# Benchmarking Machine Learning Anomaly Detection Methods on the Tennessee Eastman Process Dataset

This repository contains the code and analysis for benchmarking machine learning and deep learning methods for fault detection and diagnosis (FDD) on the Tennessee Eastman Process (TEP) dataset.

## Project Structure

```
tep-manuscript/
├── notebooks/          # Jupyter notebooks for data processing and analysis
│   ├── 00-system.ipynb                    # System specifications
│   ├── 01-create-datasets.ipynb           # Dataset creation
│   ├── 02-exploratory-data-analysis.ipynb # Data validation
│   ├── 1x-*.ipynb                         # Hyperparameter tuning notebooks
│   ├── 2x-*.ipynb                         # Final model training notebooks
│   ├── 3x-*.ipynb                         # HMM filter post-processing notebooks
│   ├── 4x-*.ipynb                         # New TEP-Sim data evaluation notebooks
│   └── src/                               # Shared Python modules
├── data/              # Generated datasets (not tracked in git)
├── outputs/           # Analysis outputs (not tracked in git)
│   ├── models/        # Trained model checkpoints
│   ├── metrics/       # JSON metrics files
│   ├── figures/       # Generated plots
│   └── hyperparams/   # Best hyperparameters from tuning
├── figures/           # Generated figures (not tracked in git)
├── manuscript/        # LaTeX manuscript and figures
├── Dataset/           # Raw data files (.RData from Harvard Dataverse)
├── v1/                # Original model notebooks (reference implementations)
├── archive/           # Deprecated notebooks (50-series detector experiments)
├── future-work/       # Research proposals for future papers
│   ├── class-wise-autoencoder-ensemble.md
│   └── hierarchical-moe-open-set-recognition.md
├── pyproject.toml     # Python dependencies
├── Makefile           # Build automation
└── README.md          # This file
```

## Quick Start

### 1. Install Dependencies

```bash
make install
```

Or manually:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn pyreadr tqdm jupyter ipykernel nbformat psutil
```

### 2. Run All Analysis Notebooks

```bash
make all
```

This will execute:
- `00-system.ipynb` - Document system specs
- `01-create-datasets.ipynb` - Create reproducible datasets
- `02-exploratory-data-analysis.ipynb` - Analyze and validate data

### 3. Check Status

```bash
make status
```

## Makefile Targets

Run `make help` to see all available targets:

```bash
make help              # Show all available commands
make install           # Install Python dependencies
make all               # Run all notebooks in sequence
make 00-series         # Run all 00-series notebooks
make system            # Run system documentation notebook
make data              # Run dataset creation notebook
make eda               # Run exploratory data analysis
make clean             # Remove generated outputs
make clean-data        # Remove generated datasets
make status            # Show status of generated files
make check             # Verify all outputs exist
```

## Dataset Information

### Data Source

The raw data comes from Rieth et al. (2017):
> Rieth, C. A., Amsel, B. D., Tran, R., & Cook, M. B. (2017). Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation. Harvard Dataverse.

Located in `Dataset/`:
- `TEP_FaultFree_Training.RData.zip` - 500 fault-free training runs
- `TEP_Faulty_Training.RData.zip` - 500 faulty training runs
- `TEP_FaultFree_Testing.RData.zip` - 500 fault-free test runs
- `TEP_Faulty_Testing.RData.zip` - 500 faulty test runs

### Generated Datasets

The `01-create-datasets.ipynb` notebook generates balanced train/val/test splits:

**Supervised Learning (Multiclass):**
- `supervised_train.csv` - 320 normal + 340 faulty runs (20 per fault)
- `supervised_val.csv` - 160 normal + 170 faulty runs (10 per fault)
- `supervised_test.csv` - 240 normal + 255 faulty runs (15 per fault)

**Semi-Supervised Learning (Binary):**
- `semisupervised_train.csv` - 320 normal runs only
- `semisupervised_val.csv` - 160 normal runs only
- `semisupervised_test.csv` - 120 normal + 170 faulty runs (10 per fault)

### Features

- 52 total features:
  - 41 measured variables (`xmeas_1` to `xmeas_41`)
  - 11 manipulated variables (`xmv_1` to `xmv_11`)

### Fault Classes

- 18 classes total: 1 normal (fault 0) + 17 fault types
- Excluded faults: 3, 9, 15 (too subtle to detect reliably)

## Models

### Supervised Multi-class Classifiers

| Model | Test Accuracy | F1 (weighted) | Key Strength |
|-------|---------------|---------------|--------------|
| **XGBoost** | 93.91% | 0.9416 | Fast, interpretable |
| **LSTM** | 99.14% | 0.9914 | Temporal patterns |
| **LSTM-FCN** | 99.37% | 0.9937 | Best overall ⭐ |
| **CNN-Transformer** | 99.20% | 0.9920 | Attention mechanism |
| **TransKal** | 99.09% | 0.9909 | Kalman smoothing |

### Semi-supervised Anomaly Detection

| Model | Test Accuracy | Notes |
|-------|---------------|-------|
| **LSTM Autoencoder** | Binary only | Reconstruction-based |
| **Conv Autoencoder** | Binary only | Reconstruction-based |

### Post-processing (HMM Filter)

The 30-series notebooks apply Hidden Markov Model filtering to smooth predictions:
- XGBoost: 93.91% → 95.90% (+2.0% with HMM)
- Neural networks: <0.1% improvement (already smooth predictions)

## Future Work

Research proposals for follow-up papers are documented in `future-work/`:

### 1. Class-wise Autoencoder Ensemble
**File**: `future-work/class-wise-autoencoder-ensemble.md`

Train one autoencoder per fault class (18 total). At inference, the class whose AE achieves the lowest reconstruction error is predicted. Key benefits:
- Softmax over reconstruction errors provides uncertainty quantification
- May generalize better to distribution shift (learns dynamics, not boundaries)
- Interpretable: which faults have similar reconstruction patterns?

### 2. Hierarchical Mixture of Experts with Open-Set Recognition
**File**: `future-work/hierarchical-moe-open-set-recognition.md`

Combine multiple pre-trained models (XGBoost, LSTM, CNN-Transformer) with:
- Learned gating network to weight experts per input
- Novelty detection for unknown fault classes (3, 9, 15)
- Hierarchical structure for coarse-to-fine classification
- Uncertainty quantification via entropy and expert disagreement

**Key Discovery**: Faults 3, 9, 15 ARE available in the original Harvard Dataverse data. They were explicitly excluded in dataset creation but can be included for open-set evaluation.

## Development Workflow

### Run Individual Notebooks

```bash
make system    # Document system specifications
make data      # Create datasets
make eda       # Analyze datasets
```

### Open Jupyter Notebook Server

```bash
make run-notebook
```

### Clean Generated Files

```bash
make clean       # Remove outputs and figures
make clean-data  # Remove generated datasets
make clean-all   # Remove everything
```

## Git Workflow

The `.gitignore` is configured to:
- ✓ Track: Code, notebooks, configs, manuscript
- ✗ Ignore: Generated data, outputs, figures, model checkpoints

### Adding Changes

```bash
git add notebooks/  # Add new/modified notebooks
git add Makefile    # Add changes to build system
git commit -m "Description of changes"
```

Or use the Makefile:

```bash
make git-add-code
make git-commit MSG="Your commit message"
```

## Authors

- Ethan M. Sunshine
- Naixin Lyu
- Suraj Botcha
- Eesha Kulkarni
- Shreya Pagaria
- Victor Alves
- John R. Kitchin*

*Corresponding author: jkitchin@andrew.cmu.edu

Department of Chemical Engineering, Carnegie Mellon University, Pittsburgh, PA

## License

[Add license information]

## Citation

If you use this code or data, please cite:

```bibtex
@article{sunshine2024benchmarking,
  title={Benchmarking Machine Learning Anomaly Detection Methods on the Tennessee Eastman Process Dataset},
  author={Sunshine, Ethan M. and Lyu, Naixin and Botcha, Suraj and Kulkarni, Eesha and Pagaria, Shreya and Alves, Victor and Kitchin, John R.},
  year={2024}
}
```

## Acknowledgments

This work uses the enriched Tennessee Eastman Process dataset provided by Rieth et al. (2017) from the Harvard Dataverse.
