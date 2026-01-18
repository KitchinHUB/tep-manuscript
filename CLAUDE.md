# Claude Code Instructions for TEP Manuscript Project

## Project Overview

This project benchmarks machine learning and deep learning methods for fault detection and diagnosis on the Tennessee Eastman Process (TEP) dataset. It includes hyperparameter tuning, model training, evaluation, and manuscript preparation.

## Requirements

- Python >= 3.10
- Key dependencies: PyTorch, TensorFlow, XGBoost, Optuna, scikit-learn
- Package manager: uv (recommended) or pip

## Installation

```bash
# Using uv (recommended)
make bootstrap       # Full setup: install uv, configure environment, install dependencies

# Or manually
make install         # Install dependencies with uv
```

## Directory Structure

```
├── notebooks/           # Jupyter notebooks organized by series
│   ├── 0X-*.ipynb      # Setup: system specs, dataset creation, EDA, TEP-Sim generation
│   ├── 1X-*.ipynb      # Hyperparameter tuning (10-series)
│   ├── 2X-*.ipynb      # Final model training (20-series)
│   ├── 3X-*.ipynb      # HMM filter evaluation (30-series)
│   └── 4X-*.ipynb      # New data evaluation (40-series)
├── data/                # Datasets (train/val/test splits, not in git)
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
├── v1/                  # Original model notebooks (reference implementations)
├── Makefile            # Project automation (~800 lines, 50+ targets)
├── pyproject.toml      # Python dependencies
├── _config.yml         # Jupyter Book configuration
├── _toc.yml            # Jupyter Book table of contents
├── using-llms.md       # Documentation on Claude Code usage in this project
└── *-summary.md        # Summary documents for each notebook series
```

## Key Make Targets

Run `make help` to see all available targets. Key targets:

**Setup:**
- `make bootstrap` - Full setup (install uv, configure bashrc, install deps)
- `make install` - Install Python dependencies using uv

**Data & Analysis:**
- `make data` - Create datasets from raw TEP data
- `make eda` - Run exploratory data analysis
- `make new-tep-data` - Generate independent evaluation data via TEP-Sim

**Model Training:**
- `make hyperparams` - Run all hyperparameter tuning (10-series)
- `make final-training` - Train all final models (20-series)
- `make hmm-filters` - Run HMM filter post-processing (30-series)
- `make new-eval` - Evaluate on independent dataset (40-series)

**Jupyter Book:**
- `make book` - Build HTML documentation
- `make book-serve` - Build and serve locally

**Manuscript (from `manuscript/` directory):**
- `make pdf` - Build manuscript PDF
- `make clean` - Remove build artifacts

**Utilities:**
- `make status` - Show status of generated files
- `make run-notebook` - Open Jupyter notebook server

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
- Raw data (~1.3 GB) must be downloaded from Harvard Dataverse and placed in `Dataset/`

## Data Sources

- Original TEP data: Rieth et al. (2017) Harvard Dataverse
- Independent evaluation data: Generated via `tep-sim` simulator
- Data repository: https://github.com/jkitchin/tennessee-eastman-profbraatz

## Conventions

- Notebooks are numbered by series (0X, 1X, 2X, etc.)
- Each series has a summary markdown file (e.g., `10-summary.md`)
- Model outputs follow pattern: `<model>_<type><mode_suffix>.<ext>`
- Mode suffix is `_quick` when QUICK_MODE=True, empty otherwise

## About This Project

This repository was developed collaboratively with Claude Code. See `using-llms.md` for details on the development process, including workflow patterns and lessons learned. The `v1/` folder contains the original manual implementations for reference.
