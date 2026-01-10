# Claude Code Instructions for TEP Manuscript Project

## Project Overview

This project benchmarks machine learning and deep learning methods for fault detection and diagnosis on the Tennessee Eastman Process (TEP) dataset. It includes hyperparameter tuning, model training, evaluation, and manuscript preparation.

## Directory Structure

```
├── notebooks/           # Jupyter notebooks organized by series
│   ├── 0X-*.ipynb      # Data creation and EDA
│   ├── 1X-*.ipynb      # Hyperparameter tuning (10-series)
│   ├── 2X-*.ipynb      # Final model training (20-series)
│   ├── 3X-*.ipynb      # HMM filter evaluation (30-series)
│   ├── 4X-*.ipynb      # New data evaluation (40-series)
│   └── 5X-*.ipynb      # Pluggable detector evaluation (50-series)
├── data/                # Datasets (train/val/test splits)
├── outputs/
│   ├── hyperparams/    # Optimized hyperparameters (JSON)
│   ├── models/         # Trained model weights
│   ├── metrics/        # Evaluation metrics (JSON/CSV)
│   └── optuna_studies/ # Optuna study objects
├── manuscript/          # LaTeX manuscript
│   ├── main.tex        # Main document
│   ├── body.tex        # Document body
│   ├── references.bib  # Bibliography
│   └── figures/        # Manuscript figures
├── Makefile            # Project automation
└── *-summary.md        # Summary documents for each notebook series
```

## Key Make Targets

From root directory:
- `make help` - Show all available targets
- `make data` - Create datasets
- `make hyperparams` - Run all hyperparameter tuning
- `make final-training` - Train all final models
- `make hmm-filters` - Run HMM filter evaluation
- `make new-eval` - Evaluate on independent dataset

From `manuscript/` directory:
- `make pdf` - Build manuscript PDF
- `make clean` - Remove build artifacts

## Models

Seven models are evaluated:

**Multiclass Classification (18 classes):**
1. XGBoost - Gradient boosted trees baseline
2. LSTM - Recurrent neural network
3. LSTM-FCN - Hybrid LSTM + Fully Convolutional Network
4. CNN-Transformer - 1D CNN + Transformer encoder
5. TransKal - Transformer + Kalman filter

**Binary Anomaly Detection:**
6. LSTM Autoencoder - Reconstruction-based
7. Convolutional Autoencoder - 1D CNN reconstruction

## Important Notes

- Use `QUICK_MODE=True` for faster testing (e.g., `make new-tep-data QUICK_MODE=True`)
- Hyperparameters are stored in `outputs/hyperparams/<model>_best.json`
- All metrics use weighted F1 as primary evaluation metric
- Dataset is balanced: 100 runs per class for training
- Faults 3, 9, 15 are excluded (too subtle to detect)

## Data Sources

- Original TEP data: Rieth et al. (2017) Harvard Dataverse
- Independent evaluation data: Generated via `tep-sim` simulator
- Data repository: https://github.com/jkitchin/tennessee-eastman-profbraatz

## Conventions

- Notebooks are numbered by series (0X, 1X, 2X, etc.)
- Each series has a summary markdown file (e.g., `10-summary.md`)
- Model outputs follow pattern: `<model>_<type><mode_suffix>.<ext>`
- Mode suffix is `_quick` when QUICK_MODE=True, empty otherwise
