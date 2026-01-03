# Benchmarking Machine Learning Anomaly Detection Methods on the Tennessee Eastman Process Dataset

This repository contains the code and analysis for benchmarking machine learning and deep learning methods for fault detection and diagnosis (FDD) on the Tennessee Eastman Process (TEP) dataset.

## Project Structure

```
tep-manuscript/
├── notebooks/          # Jupyter notebooks for data processing and analysis
│   ├── 00-system.ipynb                    # System specifications
│   ├── 01-create-datasets.ipynb           # Dataset creation
│   └── 02-exploratory-data-analysis.ipynb # Data validation
├── data/              # Generated datasets (not tracked in git)
├── outputs/           # Analysis outputs (not tracked in git)
├── figures/           # Generated figures (not tracked in git)
├── manuscript/        # LaTeX manuscript and figures
├── Dataset/           # Raw data files (.RData)
├── v1/                # Original model notebooks
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

See `v1/` directory for original model implementations:

1. **XGBoost** - Classical ML baseline (94.40% accuracy)
2. **LSTM** - Recurrent neural network (98.74% accuracy)
3. **LSTM-FCN** - Hybrid LSTM + CNN (99.14% accuracy)
4. **CNN + Transformer** - Hybrid architecture (99.11% accuracy)
5. **TransKal** - Transformer + Kalman filter (99.37% accuracy) ⭐ Best
6. **LSTM Autoencoder** - Semi-supervised (97.65% accuracy)
7. **Convolutional Autoencoder** - Semi-supervised (98.27% accuracy)

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
