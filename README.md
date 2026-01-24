# Benchmarking Machine Learning Anomaly Detection Methods on the Tennessee Eastman Process Dataset

This repository contains the code and analysis for benchmarking machine learning and deep learning methods for fault detection and diagnosis (FDD) on the Tennessee Eastman Process (TEP) dataset.

## Project Structure

```
tep-manuscript/
├── notebooks/          # Jupyter notebooks for data processing and analysis
│   ├── 0x-*.ipynb     # Setup: system specs, dataset creation, EDA, TEP-Sim generation
│   ├── 1x-*.ipynb     # Hyperparameter tuning notebooks
│   ├── 2x-*.ipynb     # Final model training notebooks
│   ├── 3x-*.ipynb     # HMM filter post-processing notebooks
│   └── 4x-*.ipynb     # New TEP-Sim data evaluation notebooks
├── data/              # Generated datasets (not tracked in git)
├── outputs/           # Analysis outputs (not tracked in git)
│   ├── models/        # Trained model checkpoints
│   ├── metrics/       # JSON metrics files
│   ├── figures/       # Generated plots
│   └── hyperparams/   # Best hyperparameters from tuning
├── figures/           # Generated figures (not tracked in git)
├── manuscript/        # LaTeX manuscript and figures
├── Dataset/           # Raw data files (not tracked in git, download from Harvard Dataverse)
├── v1/                # Original model notebooks (reference implementations)
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

**Note:** Raw data files are not included in this repository due to their size (~1.3 GB). Download them from [Harvard Dataverse](https://doi.org/10.7910/DVN/6C3JR1) and place in the `Dataset/` directory:
- `TEP_FaultFree_Training.RData.zip` - 500 fault-free training runs
- `TEP_Faulty_Training.RData.zip` - 500 faulty training runs
- `TEP_FaultFree_Testing.RData.zip` - 500 fault-free test runs
- `TEP_Faulty_Testing.RData.zip` - 500 faulty test runs

### Generated Datasets

The `01-create-datasets.ipynb` notebook generates balanced train/val/test splits in `data/`:

**Supervised Learning (Multiclass):**
- `multiclass_train.csv` - 320 normal + 340 faulty runs (20 per fault)
- `multiclass_val.csv` - 160 normal + 170 faulty runs (10 per fault)
- `multiclass_test.csv` - 240 normal + 255 faulty runs (15 per fault)

**Semi-Supervised Learning (Binary):**
- `binary_train.csv` - 320 normal runs only
- `binary_val.csv` - 160 normal runs only
- `binary_test.csv` - 120 normal + 170 faulty runs (10 per fault)

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

## Contributing New Algorithms

To add a new algorithm to this benchmark suite, follow this workflow:

### Required: Create Two Notebooks

**1. Hyperparameter Tuning (1X-series)**

Create `notebooks/1X-<model>-hyperparameter-tuning.ipynb`:

```python
# Use QUICK_MODE pattern for fast iteration
QUICK_MODE = os.environ.get('QUICK_MODE', 'False') == 'True'
N_TRIALS = 5 if QUICK_MODE else 50

def objective(trial):
    # Define hyperparameter search space using trial.suggest_*
    # Train model, return validation F1 score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_TRIALS)
```

Save outputs to:
- `outputs/hyperparams/<model>_best.json` - Best hyperparameters
- `outputs/optuna_studies/<model>_study.pkl` - Optuna study object

**2. Final Training (2X-series)**

Create `notebooks/2X-<model>-final-training.ipynb`:
- Load best hyperparameters from JSON
- Train on full training set
- Evaluate on validation and test sets
- Save model to `outputs/models/<model>_final.pt` (or `.h5`)
- Save metrics to `outputs/metrics/<model>_results.json`

### Optional: Additional Notebooks

- **3X-series**: HMM post-processing for temporal smoothing
- **4X-series**: Evaluation on independent TEP-Sim data

### Key Conventions

1. **Data handling**: Create windows within simulation runs only (no cross-run leakage)
2. **QUICK_MODE**: Support `QUICK_MODE=True` for rapid iteration
3. **File naming**: `<model>_<type><mode_suffix>.<ext>` (add `_quick` suffix in QUICK_MODE)
4. **Progress bars**: Use tqdm for CLI visibility
5. **Random seeds**: Set for reproducibility

### Add Makefile Targets

```makefile
HYPERPARAM_NEWMODEL := outputs/hyperparams/newmodel_best.json
hyperparam-newmodel: $(HYPERPARAM_NEWMODEL)
$(HYPERPARAM_NEWMODEL): notebooks/1X-newmodel-hyperparameter-tuning.ipynb $(DATA_FILES)
	$(RUN_NOTEBOOK) $<
```

### Reference Notebooks

Use existing notebooks as templates:
- `notebooks/10-xgboost-hyperparameter-tuning.ipynb`
- `notebooks/20-xgboost-final-training.ipynb`

Most boilerplate (data loading, metrics, file I/O) can be copied directly.

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code or data, please cite:

```bibtex
@article{sunshine2025benchmarking,
  title={Benchmarking Machine Learning Anomaly Detection Methods on the Tennessee Eastman Process Dataset},
  author={Sunshine, Ethan M. and Lyu, Naixin and Botcha, Suraj and Kulkarni, Eesha and Pagaria, Shreya and Alves, Victor and Kitchin, John R.},
  year={2025}
}
```

## Acknowledgments

This work uses the enriched Tennessee Eastman Process dataset provided by Rieth et al. (2017) from the Harvard Dataverse.
