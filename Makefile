.PHONY: help clean all 00-series data eda system install new-tep-data tep-sim-validation new-eval book book-clean book-serve book-pdf

# Default target
.DEFAULT_GOAL := help

# Directories
NOTEBOOKS_DIR := notebooks
DATA_DIR := data
OUTPUTS_DIR := outputs
FIGURES_DIR := figures
METRICS_DIR := $(OUTPUTS_DIR)/metrics

# Virtual environment
VENV := .venv
VENV_BIN := $(VENV)/bin

# Python executable (use venv if available, fall back to system python)
PYTHON := $(shell [ -f $(VENV_BIN)/python ] && echo $(VENV_BIN)/python || echo python3)

# Jupyter executable (use venv if available, fall back to system jupyter)
JUPYTER := $(shell [ -f $(VENV_BIN)/jupyter ] && echo $(VENV_BIN)/jupyter || echo jupyter)

# Notebook files
SYSTEM_NB := $(NOTEBOOKS_DIR)/00-system.ipynb
DATASET_NB := $(NOTEBOOKS_DIR)/01-create-datasets.ipynb
EDA_NB := $(NOTEBOOKS_DIR)/02-exploratory-data-analysis.ipynb
NEW_TEP_DATA_NB := $(NOTEBOOKS_DIR)/03-generate-new-tep-dataset.ipynb
TEP_SIM_VALIDATION_NB := $(NOTEBOOKS_DIR)/04-tep-sim-data-validation.ipynb

# Output artifacts
SYSTEM_INFO := $(OUTPUTS_DIR)/system_info.txt
REQUIREMENTS := $(OUTPUTS_DIR)/requirements.txt

# Data artifacts
MULTICLASS_TRAIN := $(DATA_DIR)/multiclass_train.csv
MULTICLASS_VAL := $(DATA_DIR)/multiclass_val.csv
MULTICLASS_TEST := $(DATA_DIR)/multiclass_test.csv
BINARY_TRAIN := $(DATA_DIR)/binary_train.csv
BINARY_VAL := $(DATA_DIR)/binary_val.csv
BINARY_TEST := $(DATA_DIR)/binary_test.csv

# QUICK_MODE suffix - must be defined before variables that use it
ifeq ($(QUICK_MODE),True)
    MODE_SUFFIX := _quick
else
    MODE_SUFFIX :=
endif

# New TEP data artifacts (from tep-sim)
NEW_MULTICLASS_EVAL := $(DATA_DIR)/new_multiclass_eval$(MODE_SUFFIX).csv
NEW_BINARY_EVAL := $(DATA_DIR)/new_binary_eval$(MODE_SUFFIX).csv

# EDA artifacts
EDA_SUMMARY := $(OUTPUTS_DIR)/eda_summary.txt
FAULT_DIST_FIG := $(FIGURES_DIR)/fault_distribution.png
FEATURE_CORR_FIG := $(FIGURES_DIR)/feature_correlations.png
FAULT_SIG_FIG := $(FIGURES_DIR)/fault_signatures.png

# Hyperparameter tuning notebooks (10-series)
HYPERPARAM_XGBOOST_NB := $(NOTEBOOKS_DIR)/10-xgboost-hyperparameter-tuning.ipynb
HYPERPARAM_LSTM_NB := $(NOTEBOOKS_DIR)/11-lstm-hyperparameter-tuning.ipynb
HYPERPARAM_LSTMFCN_NB := $(NOTEBOOKS_DIR)/12-lstm-fcn-hyperparameter-tuning.ipynb
HYPERPARAM_CNNTRANS_NB := $(NOTEBOOKS_DIR)/13-cnn-transformer-hyperparameter-tuning.ipynb
HYPERPARAM_TRANSKAL_NB := $(NOTEBOOKS_DIR)/14-transkal-hyperparameter-tuning.ipynb
HYPERPARAM_LSTMAE_NB := $(NOTEBOOKS_DIR)/15-lstm-autoencoder-hyperparameter-tuning.ipynb
HYPERPARAM_CONVAE_NB := $(NOTEBOOKS_DIR)/16-conv-autoencoder-hyperparameter-tuning.ipynb
HYPERPARAM_SUMMARY_NB := $(NOTEBOOKS_DIR)/17-hyperparameter-summary.ipynb

# Hyperparameter output artifacts
HYPERPARAMS_DIR := $(OUTPUTS_DIR)/hyperparams
OPTUNA_STUDIES_DIR := $(OUTPUTS_DIR)/optuna_studies

HYPERPARAM_XGBOOST := $(HYPERPARAMS_DIR)/xgboost_best$(MODE_SUFFIX).json
HYPERPARAM_LSTM := $(HYPERPARAMS_DIR)/lstm_best$(MODE_SUFFIX).json
HYPERPARAM_LSTMFCN := $(HYPERPARAMS_DIR)/lstm_fcn_best$(MODE_SUFFIX).json
HYPERPARAM_CNNTRANS := $(HYPERPARAMS_DIR)/cnn_transformer_best$(MODE_SUFFIX).json
HYPERPARAM_TRANSKAL := $(HYPERPARAMS_DIR)/transkal_best$(MODE_SUFFIX).json
HYPERPARAM_LSTMAE := $(HYPERPARAMS_DIR)/lstm_autoencoder_best$(MODE_SUFFIX).json
HYPERPARAM_CONVAE := $(HYPERPARAMS_DIR)/conv_autoencoder_best$(MODE_SUFFIX).json
HYPERPARAM_SUMMARY := $(HYPERPARAMS_DIR)/summary$(MODE_SUFFIX).csv

##@ Help
help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\\nUsage:\\n  make \\033[36m<target>\\033[0m\\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \\033[36m%-20s\\033[0m %s\\n", $$1, $$2 } /^##@/ { printf "\\n\\033[1m%s\\033[0m\\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup
install-uv: ## Install uv package manager
	@echo "Installing uv..."
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo "✓ uv installed"
	@echo "Run 'source ~/.local/bin/env' to add uv to PATH"

setup-bashrc: ## Configure bashrc to use venv
	@echo "Configuring bashrc..."
	@grep -q "Deactivate conda and activate venv" ~/.bashrc || \
		echo '\n# Deactivate conda and activate venv on login (interactive shells only)\nif [[ $$- == *i* ]]; then\n    # Deactivate conda if active\n    if [ -n "$$CONDA_DEFAULT_ENV" ] && [ "$$CONDA_DEFAULT_ENV" != "base" ]; then\n        conda deactivate 2>/dev/null\n    fi\n\n    # If still in conda base, deactivate it\n    if [ "$$CONDA_DEFAULT_ENV" = "base" ]; then\n        conda deactivate 2>/dev/null\n    fi\n\n    # Activate project venv if it exists and not already active\n    if [ -z "$$VIRTUAL_ENV" ] && [ -f /home/jovyan/work/jkitchin/tep-manuscript/.venv/bin/activate ]; then\n        source /home/jovyan/work/jkitchin/tep-manuscript/.venv/bin/activate\n        # Change to project directory\n        cd /home/jovyan/work/jkitchin/tep-manuscript 2>/dev/null || true\n    fi\nfi' >> ~/.bashrc
	@echo "✓ bashrc configured"

install: ## Install Python dependencies from pyproject.toml using uv
	@echo "Installing dependencies with uv from pyproject.toml..."
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Error: uv not found. Run 'make install-uv' first"; \
		exit 1; \
	fi
	@. $(HOME)/.local/bin/env && uv pip install -e .
	@echo "✓ Dependencies installed from pyproject.toml"

bootstrap: install-uv setup-bashrc install ## Full setup: install uv, configure bashrc, and install dependencies

##@ Build Targets
all: 00-series ## Run all notebooks in sequence
	@echo "✓ All notebooks executed successfully"

00-series: system data eda ## Run all 00-series notebooks (system, data creation, EDA)
	@echo "✓ 00-series complete"

##@ Individual Notebooks
system: $(SYSTEM_INFO) $(REQUIREMENTS) ## Run 00-system.ipynb - Document system specifications

$(SYSTEM_INFO) $(REQUIREMENTS): $(SYSTEM_NB)
	@echo "Running 00-system.ipynb..."
	@mkdir -p $(OUTPUTS_DIR)
	@$(JUPYTER) nbconvert --to notebook --execute \
		--output $(SYSTEM_NB) \
		$(SYSTEM_NB)
	@echo "✓ System information documented"

data: $(MULTICLASS_TRAIN) ## Run 01-create-datasets.ipynb - Create reproducible datasets

$(MULTICLASS_TRAIN) $(MULTICLASS_VAL) $(MULTICLASS_TEST) $(BINARY_TRAIN) $(BINARY_VAL) $(BINARY_TEST): $(DATASET_NB)
	@echo "Running 01-create-datasets.ipynb..."
	@echo "This may take several minutes..."
	@mkdir -p $(DATA_DIR)
	@$(JUPYTER) nbconvert --to notebook --execute \
		--output 01-create-datasets.ipynb \
		$(DATASET_NB)
	@echo "✓ Datasets created"

eda: $(EDA_SUMMARY) ## Run 02-exploratory-data-analysis.ipynb - Analyze datasets

$(EDA_SUMMARY) $(FAULT_DIST_FIG) $(FEATURE_CORR_FIG) $(FAULT_SIG_FIG): $(EDA_NB) $(MULTICLASS_TRAIN)
	@echo "Running 02-exploratory-data-analysis.ipynb..."
	@mkdir -p $(OUTPUTS_DIR) $(FIGURES_DIR)
	@$(JUPYTER) nbconvert --to notebook --execute \
		--output 02-exploratory-data-analysis.ipynb \
		$(EDA_NB)
	@echo "✓ Exploratory data analysis complete"

new-tep-data: $(NEW_MULTICLASS_EVAL) ## Run 03-generate-new-tep-dataset.ipynb - Generate new TEP evaluation data

$(NEW_MULTICLASS_EVAL) $(NEW_BINARY_EVAL): $(NEW_TEP_DATA_NB)
	@echo "Running 03-generate-new-tep-dataset.ipynb (QUICK_MODE=$(if $(QUICK_MODE),$(QUICK_MODE),False))..."
	@if [ "$(QUICK_MODE)" = "True" ]; then \
		echo "⚡ Quick mode: generating minimal test dataset (~1 min)"; \
	else \
		echo "Full mode: generating complete evaluation dataset (~4 hrs)"; \
	fi
	@mkdir -p $(DATA_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=-1 \
		--log-level=INFO \
		$(NEW_TEP_DATA_NB)
	@if [ ! -f "$(NEW_MULTICLASS_EVAL)" ]; then \
		echo "Error: Expected output file $(NEW_MULTICLASS_EVAL) not created"; \
		exit 1; \
	fi
	@echo "✓ New TEP evaluation datasets generated"

tep-sim-validation: ## Run 05-tep-sim-data-validation.ipynb - Validate TEP-Sim data independence and distribution
	@if [ ! -f "$(NEW_MULTICLASS_EVAL)" ]; then \
		echo "Error: $(NEW_MULTICLASS_EVAL) not found. Run 'make new-tep-data QUICK_MODE=$(QUICK_MODE)' first."; \
		exit 1; \
	fi
	@echo "Running TEP-Sim data validation (QUICK_MODE=$(if $(QUICK_MODE),$(QUICK_MODE),False))..."
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=-1 --log-level=INFO \
		$(TEP_SIM_VALIDATION_NB)
	@echo "✓ TEP-Sim data validation complete"

##@ HMM Filters (30-series)
# Output metrics for 30-series
HMM_XGBOOST := $(METRICS_DIR)/xgboost_hmm_filter_results.json
HMM_LSTM := $(METRICS_DIR)/lstm_hmm_filter_results.json
HMM_LSTMFCN := $(METRICS_DIR)/lstm_fcn_hmm_filter_results.json
HMM_CNNTRANS := $(METRICS_DIR)/cnn_transformer_hmm_filter_results.json
HMM_TRANSKAL := $(METRICS_DIR)/transkal_hmm_filter_results.json
HMM_SUMMARY := $(METRICS_DIR)/hmm_filter_comparison.csv

.PHONY: hmm-filters

hmm-filters: $(HMM_XGBOOST) $(HMM_LSTM) $(HMM_LSTMFCN) $(HMM_CNNTRANS) $(HMM_TRANSKAL) $(HMM_SUMMARY) ## Run all 30-series HMM filter notebooks
	@echo "✓ All HMM filter evaluations complete"

hmm-xgboost: $(HMM_XGBOOST) ## Run 30-xgboost-hmm-filter.ipynb

$(HMM_XGBOOST): $(NOTEBOOKS_DIR)/30-xgboost-hmm-filter.ipynb $(FINAL_XGBOOST_MODEL) $(MULTICLASS_TEST)
	@echo "Running XGBoost HMM filter evaluation..."
	@mkdir -p $(METRICS_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=-1 --log-level=INFO \
		$(NOTEBOOKS_DIR)/30-xgboost-hmm-filter.ipynb
	@touch $@
	@echo "✓ XGBoost HMM filter complete"

hmm-lstm: $(HMM_LSTM) ## Run 31-lstm-hmm-filter.ipynb

$(HMM_LSTM): $(NOTEBOOKS_DIR)/31-lstm-hmm-filter.ipynb $(FINAL_LSTM_MODEL) $(MULTICLASS_TEST)
	@echo "Running LSTM HMM filter evaluation..."
	@mkdir -p $(METRICS_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=-1 --log-level=INFO \
		$(NOTEBOOKS_DIR)/31-lstm-hmm-filter.ipynb
	@touch $@
	@echo "✓ LSTM HMM filter complete"

hmm-lstm-fcn: $(HMM_LSTMFCN) ## Run 32-lstm-fcn-hmm-filter.ipynb

$(HMM_LSTMFCN): $(NOTEBOOKS_DIR)/32-lstm-fcn-hmm-filter.ipynb $(FINAL_LSTMFCN_MODEL) $(MULTICLASS_TEST)
	@echo "Running LSTM-FCN HMM filter evaluation..."
	@mkdir -p $(METRICS_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=-1 --log-level=INFO \
		$(NOTEBOOKS_DIR)/32-lstm-fcn-hmm-filter.ipynb
	@touch $@
	@echo "✓ LSTM-FCN HMM filter complete"

hmm-cnn-transformer: $(HMM_CNNTRANS) ## Run 33-cnn-transformer-hmm-filter.ipynb

$(HMM_CNNTRANS): $(NOTEBOOKS_DIR)/33-cnn-transformer-hmm-filter.ipynb $(FINAL_CNNTRANS_MODEL) $(MULTICLASS_TEST)
	@echo "Running CNN-Transformer HMM filter evaluation..."
	@mkdir -p $(METRICS_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=-1 --log-level=INFO \
		$(NOTEBOOKS_DIR)/33-cnn-transformer-hmm-filter.ipynb
	@touch $@
	@echo "✓ CNN-Transformer HMM filter complete"

hmm-transkal: $(HMM_TRANSKAL) ## Run 34-transkal-hmm-filter.ipynb

$(HMM_TRANSKAL): $(NOTEBOOKS_DIR)/34-transkal-hmm-filter.ipynb $(FINAL_TRANSKAL_MODEL) $(MULTICLASS_TEST)
	@echo "Running TransKal HMM filter evaluation..."
	@mkdir -p $(METRICS_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=-1 --log-level=INFO \
		$(NOTEBOOKS_DIR)/34-transkal-hmm-filter.ipynb
	@touch $@
	@echo "✓ TransKal HMM filter complete"

hmm-summary: $(HMM_SUMMARY) ## Run 37-hmm-filter-comparison-summary.ipynb

$(HMM_SUMMARY): $(NOTEBOOKS_DIR)/37-hmm-filter-comparison-summary.ipynb $(HMM_XGBOOST) $(HMM_LSTM) $(HMM_LSTMFCN) $(HMM_CNNTRANS) $(HMM_TRANSKAL)
	@echo "Running HMM filter comparison summary..."
	@mkdir -p $(METRICS_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=-1 --log-level=INFO \
		$(NOTEBOOKS_DIR)/37-hmm-filter-comparison-summary.ipynb
	@touch $@
	@echo "✓ HMM filter comparison summary complete"

##@ New Data Evaluation (40-series)
# Output metrics for 40-series
NEW_EVAL_XGBOOST := $(METRICS_DIR)/xgboost_new_eval_metrics$(MODE_SUFFIX).json
NEW_EVAL_LSTM := $(METRICS_DIR)/lstm_new_eval_metrics$(MODE_SUFFIX).json
NEW_EVAL_LSTMFCN := $(METRICS_DIR)/lstm_fcn_new_eval_metrics$(MODE_SUFFIX).json
NEW_EVAL_CNNTRANS := $(METRICS_DIR)/cnn_transformer_new_eval_metrics$(MODE_SUFFIX).json
NEW_EVAL_TRANSKAL := $(METRICS_DIR)/transkal_new_eval_metrics$(MODE_SUFFIX).json
NEW_EVAL_LSTMAE := $(METRICS_DIR)/lstm_autoencoder_new_eval_metrics$(MODE_SUFFIX).json
NEW_EVAL_CONVAE := $(METRICS_DIR)/conv_autoencoder_new_eval_metrics$(MODE_SUFFIX).json
NEW_EVAL_SUMMARY := $(METRICS_DIR)/multiclass_comparison_new_eval$(MODE_SUFFIX).csv

.PHONY: new-eval

new-eval: $(NEW_EVAL_XGBOOST) $(NEW_EVAL_LSTM) $(NEW_EVAL_LSTMFCN) $(NEW_EVAL_CNNTRANS) $(NEW_EVAL_TRANSKAL) $(NEW_EVAL_LSTMAE) $(NEW_EVAL_CONVAE) $(NEW_EVAL_SUMMARY) ## Run all 40-series new data evaluation notebooks
	@echo "✓ All new data evaluations complete"

new-eval-xgboost: $(NEW_EVAL_XGBOOST) ## Run 40-xgboost-new-data-evaluation.ipynb

$(NEW_EVAL_XGBOOST): $(NOTEBOOKS_DIR)/40-xgboost-new-data-evaluation.ipynb $(NEW_MULTICLASS_EVAL) $(FINAL_XGBOOST_MODEL)
	@echo "Running XGBoost new data evaluation..."
	@mkdir -p $(METRICS_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=-1 --log-level=INFO \
		$(NOTEBOOKS_DIR)/40-xgboost-new-data-evaluation.ipynb
	@touch $@
	@echo "✓ XGBoost new data evaluation complete"

new-eval-lstm: $(NEW_EVAL_LSTM) ## Run 41-lstm-new-data-evaluation.ipynb

$(NEW_EVAL_LSTM): $(NOTEBOOKS_DIR)/41-lstm-new-data-evaluation.ipynb $(NEW_MULTICLASS_EVAL) $(FINAL_LSTM_MODEL)
	@echo "Running LSTM new data evaluation..."
	@mkdir -p $(METRICS_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=-1 --log-level=INFO \
		$(NOTEBOOKS_DIR)/41-lstm-new-data-evaluation.ipynb
	@touch $@
	@echo "✓ LSTM new data evaluation complete"

new-eval-lstm-fcn: $(NEW_EVAL_LSTMFCN) ## Run 42-lstm-fcn-new-data-evaluation.ipynb

$(NEW_EVAL_LSTMFCN): $(NOTEBOOKS_DIR)/42-lstm-fcn-new-data-evaluation.ipynb $(NEW_MULTICLASS_EVAL) $(FINAL_LSTMFCN_MODEL)
	@echo "Running LSTM-FCN new data evaluation..."
	@mkdir -p $(METRICS_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=-1 --log-level=INFO \
		$(NOTEBOOKS_DIR)/42-lstm-fcn-new-data-evaluation.ipynb
	@touch $@
	@echo "✓ LSTM-FCN new data evaluation complete"

new-eval-cnn-transformer: $(NEW_EVAL_CNNTRANS) ## Run 43-cnn-transformer-new-data-evaluation.ipynb

$(NEW_EVAL_CNNTRANS): $(NOTEBOOKS_DIR)/43-cnn-transformer-new-data-evaluation.ipynb $(NEW_MULTICLASS_EVAL) $(FINAL_CNNTRANS_MODEL)
	@echo "Running CNN-Transformer new data evaluation..."
	@mkdir -p $(METRICS_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=-1 --log-level=INFO \
		$(NOTEBOOKS_DIR)/43-cnn-transformer-new-data-evaluation.ipynb
	@touch $@
	@echo "✓ CNN-Transformer new data evaluation complete"

new-eval-transkal: $(NEW_EVAL_TRANSKAL) ## Run 44-transkal-new-data-evaluation.ipynb

$(NEW_EVAL_TRANSKAL): $(NOTEBOOKS_DIR)/44-transkal-new-data-evaluation.ipynb $(NEW_MULTICLASS_EVAL) $(FINAL_TRANSKAL_MODEL)
	@echo "Running TransKal new data evaluation..."
	@mkdir -p $(METRICS_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=-1 --log-level=INFO \
		$(NOTEBOOKS_DIR)/44-transkal-new-data-evaluation.ipynb
	@touch $@
	@echo "✓ TransKal new data evaluation complete"

new-eval-lstm-ae: $(NEW_EVAL_LSTMAE) ## Run 45-lstm-autoencoder-new-data-evaluation.ipynb

$(NEW_EVAL_LSTMAE): $(NOTEBOOKS_DIR)/45-lstm-autoencoder-new-data-evaluation.ipynb $(NEW_BINARY_EVAL) $(FINAL_LSTMAE_MODEL)
	@echo "Running LSTM-Autoencoder new data evaluation..."
	@mkdir -p $(METRICS_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=-1 --log-level=INFO \
		$(NOTEBOOKS_DIR)/45-lstm-autoencoder-new-data-evaluation.ipynb
	@touch $@
	@echo "✓ LSTM-Autoencoder new data evaluation complete"

new-eval-conv-ae: $(NEW_EVAL_CONVAE) ## Run 46-conv-autoencoder-new-data-evaluation.ipynb

$(NEW_EVAL_CONVAE): $(NOTEBOOKS_DIR)/46-conv-autoencoder-new-data-evaluation.ipynb $(NEW_BINARY_EVAL) $(FINAL_CONVAE_MODEL)
	@echo "Running Conv-Autoencoder new data evaluation..."
	@mkdir -p $(METRICS_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=-1 --log-level=INFO \
		$(NOTEBOOKS_DIR)/46-conv-autoencoder-new-data-evaluation.ipynb
	@touch $@
	@echo "✓ Conv-Autoencoder new data evaluation complete"

new-eval-summary: $(NEW_EVAL_SUMMARY) ## Run 47-model-comparison-new-data.ipynb

$(NEW_EVAL_SUMMARY): $(NOTEBOOKS_DIR)/47-model-comparison-new-data.ipynb $(NEW_EVAL_XGBOOST) $(NEW_EVAL_LSTM) $(NEW_EVAL_LSTMFCN) $(NEW_EVAL_CNNTRANS) $(NEW_EVAL_TRANSKAL) $(NEW_EVAL_LSTMAE) $(NEW_EVAL_CONVAE)
	@echo "Running new data model comparison summary..."
	@mkdir -p $(METRICS_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=-1 --log-level=INFO \
		$(NOTEBOOKS_DIR)/47-model-comparison-new-data.ipynb
	@touch $@
	@echo "✓ New data model comparison summary complete"

##@ Hyperparameter Tuning (10-series)
hyperparams: hyperparam-xgboost hyperparam-lstm hyperparam-lstm-fcn hyperparam-cnn-transformer hyperparam-transkal hyperparam-lstm-ae hyperparam-conv-ae hyperparam-summary ## Run all hyperparameter tuning notebooks
	@echo "✓ All hyperparameter tuning complete"

hyperparam-xgboost: $(HYPERPARAM_XGBOOST) ## Tune XGBoost hyperparameters

$(HYPERPARAM_XGBOOST): $(HYPERPARAM_XGBOOST_NB) $(MULTICLASS_TRAIN)
	@echo "Tuning XGBoost hyperparameters (QUICK_MODE=$(if $(QUICK_MODE),$(QUICK_MODE),False))..."
	@mkdir -p $(HYPERPARAMS_DIR) $(OPTUNA_STUDIES_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--log-level=INFO \
		$(HYPERPARAM_XGBOOST_NB)
	@if [ ! -f "$(HYPERPARAM_XGBOOST)" ]; then \
		echo "Error: Expected output file $(HYPERPARAM_XGBOOST) not created"; \
		exit 1; \
	fi
	@touch $(HYPERPARAM_XGBOOST)
	@echo "✓ XGBoost hyperparameters optimized"

hyperparam-lstm: $(HYPERPARAM_LSTM) ## Tune LSTM hyperparameters

$(HYPERPARAM_LSTM): $(HYPERPARAM_LSTM_NB) $(MULTICLASS_TRAIN)
	@echo "Tuning LSTM hyperparameters (QUICK_MODE=$(if $(QUICK_MODE),$(QUICK_MODE),False))..."
	@mkdir -p $(HYPERPARAMS_DIR) $(OPTUNA_STUDIES_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--log-level=INFO \
		$(HYPERPARAM_LSTM_NB)
	@if [ ! -f "$(HYPERPARAM_LSTM)" ]; then \
		echo "Error: Expected output file $(HYPERPARAM_LSTM) not created"; \
		exit 1; \
	fi
	@touch $(HYPERPARAM_LSTM)
	@echo "✓ LSTM hyperparameters optimized"

hyperparam-lstm-fcn: $(HYPERPARAM_LSTMFCN) ## Tune LSTM-FCN hyperparameters

$(HYPERPARAM_LSTMFCN): $(HYPERPARAM_LSTMFCN_NB) $(MULTICLASS_TRAIN)
	@echo "Tuning LSTM-FCN hyperparameters (QUICK_MODE=$(if $(QUICK_MODE),$(QUICK_MODE),False))..."
	@mkdir -p $(HYPERPARAMS_DIR) $(OPTUNA_STUDIES_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--log-level=INFO \
		$(HYPERPARAM_LSTMFCN_NB)
	@if [ ! -f "$(HYPERPARAM_LSTMFCN)" ]; then \
		echo "Error: Expected output file $(HYPERPARAM_LSTMFCN) not created"; \
		exit 1; \
	fi
	@touch $(HYPERPARAM_LSTMFCN)
	@echo "✓ LSTM-FCN hyperparameters optimized"

hyperparam-cnn-transformer: $(HYPERPARAM_CNNTRANS) ## Tune CNN+Transformer hyperparameters

$(HYPERPARAM_CNNTRANS): $(HYPERPARAM_CNNTRANS_NB) $(MULTICLASS_TRAIN)
	@echo "Tuning CNN+Transformer hyperparameters (QUICK_MODE=$(if $(QUICK_MODE),$(QUICK_MODE),False))..."
	@mkdir -p $(HYPERPARAMS_DIR) $(OPTUNA_STUDIES_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--log-level=INFO \
		$(HYPERPARAM_CNNTRANS_NB)
	@if [ ! -f "$(HYPERPARAM_CNNTRANS)" ]; then \
		echo "Error: Expected output file $(HYPERPARAM_CNNTRANS) not created"; \
		exit 1; \
	fi
	@touch $(HYPERPARAM_CNNTRANS)
	@echo "✓ CNN+Transformer hyperparameters optimized"

hyperparam-transkal: $(HYPERPARAM_TRANSKAL) ## Tune TransKal hyperparameters

$(HYPERPARAM_TRANSKAL): $(HYPERPARAM_TRANSKAL_NB) $(MULTICLASS_TRAIN)
	@echo "Tuning TransKal hyperparameters (QUICK_MODE=$(if $(QUICK_MODE),$(QUICK_MODE),False))..."
	@mkdir -p $(HYPERPARAMS_DIR) $(OPTUNA_STUDIES_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--log-level=INFO \
		$(HYPERPARAM_TRANSKAL_NB)
	@if [ ! -f "$(HYPERPARAM_TRANSKAL)" ]; then \
		echo "Error: Expected output file $(HYPERPARAM_TRANSKAL) not created"; \
		exit 1; \
	fi
	@touch $(HYPERPARAM_TRANSKAL)
	@echo "✓ TransKal hyperparameters optimized"

hyperparam-lstm-ae: $(HYPERPARAM_LSTMAE) ## Tune LSTM Autoencoder hyperparameters

$(HYPERPARAM_LSTMAE): $(HYPERPARAM_LSTMAE_NB) $(BINARY_TRAIN)
	@echo "Tuning LSTM Autoencoder hyperparameters (QUICK_MODE=$(if $(QUICK_MODE),$(QUICK_MODE),False))..."
	@mkdir -p $(HYPERPARAMS_DIR) $(OPTUNA_STUDIES_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--log-level=INFO \
		$(HYPERPARAM_LSTMAE_NB)
	@if [ ! -f "$(HYPERPARAM_LSTMAE)" ]; then \
		echo "Error: Expected output file $(HYPERPARAM_LSTMAE) not created"; \
		exit 1; \
	fi
	@touch $(HYPERPARAM_LSTMAE)
	@echo "✓ LSTM Autoencoder hyperparameters optimized"

hyperparam-conv-ae: $(HYPERPARAM_CONVAE) ## Tune Convolutional Autoencoder hyperparameters

$(HYPERPARAM_CONVAE): $(HYPERPARAM_CONVAE_NB) $(BINARY_TRAIN)
	@echo "Tuning Convolutional Autoencoder hyperparameters (QUICK_MODE=$(if $(QUICK_MODE),$(QUICK_MODE),False))..."
	@mkdir -p $(HYPERPARAMS_DIR) $(OPTUNA_STUDIES_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--log-level=INFO \
		$(HYPERPARAM_CONVAE_NB)
	@if [ ! -f "$(HYPERPARAM_CONVAE)" ]; then \
		echo "Error: Expected output file $(HYPERPARAM_CONVAE) not created"; \
		exit 1; \
	fi
	@touch $(HYPERPARAM_CONVAE)
	@echo "✓ Convolutional Autoencoder hyperparameters optimized"

hyperparam-summary: $(HYPERPARAM_SUMMARY) ## Summarize all hyperparameter tuning results

$(HYPERPARAM_SUMMARY): $(HYPERPARAM_SUMMARY_NB) $(HYPERPARAM_XGBOOST) $(HYPERPARAM_LSTM) $(HYPERPARAM_LSTMFCN) $(HYPERPARAM_CNNTRANS) $(HYPERPARAM_TRANSKAL) $(HYPERPARAM_LSTMAE) $(HYPERPARAM_CONVAE)
	@echo "Generating hyperparameter summary..."
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--log-level=INFO \
		$(HYPERPARAM_SUMMARY_NB)
	@if [ ! -f "$(HYPERPARAM_SUMMARY)" ]; then \
		echo "Error: Expected output file $(HYPERPARAM_SUMMARY) not created"; \
		exit 1; \
	fi
	@touch $(HYPERPARAM_SUMMARY)
	@echo "✓ Hyperparameter summary generated"

##@ Final Training (20-series)
# Final training notebooks
FINAL_XGBOOST_NB := $(NOTEBOOKS_DIR)/20-xgboost-final-training.ipynb
FINAL_LSTM_NB := $(NOTEBOOKS_DIR)/21-lstm-final-training.ipynb
FINAL_LSTMFCN_NB := $(NOTEBOOKS_DIR)/22-lstm-fcn-final-training.ipynb
FINAL_CNNTRANS_NB := $(NOTEBOOKS_DIR)/23-cnn-transformer-final-training.ipynb
FINAL_TRANSKAL_NB := $(NOTEBOOKS_DIR)/24-transkal-final-training.ipynb
FINAL_LSTMAE_NB := $(NOTEBOOKS_DIR)/25-lstm-autoencoder-final-training.ipynb
FINAL_CONVAE_NB := $(NOTEBOOKS_DIR)/26-conv-autoencoder-final-training.ipynb
FINAL_COMPARISON_NB := $(NOTEBOOKS_DIR)/27-model-comparison-summary.ipynb

# Final model output artifacts (match notebook output paths)
MODELS_DIR := $(OUTPUTS_DIR)/models
METRICS_DIR := $(OUTPUTS_DIR)/metrics
FINAL_XGBOOST_MODEL := $(MODELS_DIR)/xgboost_final$(MODE_SUFFIX).pkl
FINAL_LSTM_MODEL := $(MODELS_DIR)/lstm_final$(MODE_SUFFIX).pt
FINAL_LSTMFCN_MODEL := $(MODELS_DIR)/lstm_fcn_final$(MODE_SUFFIX).pt
FINAL_CNNTRANS_MODEL := $(MODELS_DIR)/cnn_transformer_final$(MODE_SUFFIX).pt
FINAL_TRANSKAL_MODEL := $(MODELS_DIR)/transkal_final$(MODE_SUFFIX).pt
FINAL_LSTMAE_MODEL := $(MODELS_DIR)/lstm_autoencoder_final$(MODE_SUFFIX).pt
FINAL_CONVAE_MODEL := $(MODELS_DIR)/conv_autoencoder_final$(MODE_SUFFIX).pt
FINAL_COMPARISON := $(METRICS_DIR)/model_comparison_combined$(MODE_SUFFIX).csv

final-training: final-xgboost final-lstm final-lstm-fcn final-cnn-transformer final-transkal final-lstm-ae final-conv-ae final-comparison ## Run all final training notebooks (20-series)
	@echo "✓ All final training complete"

final-training-parallel: ## Run all 7 model training notebooks in parallel (use with: make -j7 final-training-parallel)
	@echo "Training all models in parallel (QUICK_MODE=$(if $(QUICK_MODE),$(QUICK_MODE),False))..."
	$(MAKE) -j7 final-xgboost final-lstm final-lstm-fcn final-cnn-transformer final-transkal final-lstm-ae final-conv-ae
	@echo "✓ All models trained in parallel"
	$(MAKE) final-comparison
	@echo "✓ All final training complete"

final-xgboost: $(FINAL_XGBOOST_MODEL) ## Train final XGBoost model

$(FINAL_XGBOOST_MODEL): $(FINAL_XGBOOST_NB) $(HYPERPARAM_XGBOOST) $(MULTICLASS_TRAIN)
	@echo "Training final XGBoost model (QUICK_MODE=$(if $(QUICK_MODE),$(QUICK_MODE),False))..."
	@mkdir -p $(MODELS_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--log-level=INFO \
		$(FINAL_XGBOOST_NB)
	@touch $@
	@echo "✓ XGBoost final model trained"

final-lstm: $(FINAL_LSTM_MODEL) ## Train final LSTM model

$(FINAL_LSTM_MODEL): $(FINAL_LSTM_NB) $(HYPERPARAM_LSTM) $(MULTICLASS_TRAIN)
	@echo "Training final LSTM model (QUICK_MODE=$(if $(QUICK_MODE),$(QUICK_MODE),False))..."
	@mkdir -p $(MODELS_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--log-level=INFO \
		$(FINAL_LSTM_NB)
	@touch $@
	@echo "✓ LSTM final model trained"

final-lstm-fcn: $(FINAL_LSTMFCN_MODEL) ## Train final LSTM-FCN model

$(FINAL_LSTMFCN_MODEL): $(FINAL_LSTMFCN_NB) $(HYPERPARAM_LSTMFCN) $(MULTICLASS_TRAIN)
	@echo "Training final LSTM-FCN model (QUICK_MODE=$(if $(QUICK_MODE),$(QUICK_MODE),False))..."
	@mkdir -p $(MODELS_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--log-level=INFO \
		$(FINAL_LSTMFCN_NB)
	@touch $@
	@echo "✓ LSTM-FCN final model trained"

final-cnn-transformer: $(FINAL_CNNTRANS_MODEL) ## Train final CNN+Transformer model

$(FINAL_CNNTRANS_MODEL): $(FINAL_CNNTRANS_NB) $(HYPERPARAM_CNNTRANS) $(MULTICLASS_TRAIN)
	@echo "Training final CNN+Transformer model (QUICK_MODE=$(if $(QUICK_MODE),$(QUICK_MODE),False))..."
	@mkdir -p $(MODELS_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--log-level=INFO \
		$(FINAL_CNNTRANS_NB)
	@touch $@
	@echo "✓ CNN+Transformer final model trained"

final-transkal: $(FINAL_TRANSKAL_MODEL) ## Train final TransKal model

$(FINAL_TRANSKAL_MODEL): $(FINAL_TRANSKAL_NB) $(HYPERPARAM_TRANSKAL) $(MULTICLASS_TRAIN)
	@echo "Training final TransKal model (QUICK_MODE=$(if $(QUICK_MODE),$(QUICK_MODE),False))..."
	@mkdir -p $(MODELS_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--log-level=INFO \
		$(FINAL_TRANSKAL_NB)
	@touch $@
	@echo "✓ TransKal final model trained"

final-lstm-ae: $(FINAL_LSTMAE_MODEL) ## Train final LSTM Autoencoder model

$(FINAL_LSTMAE_MODEL): $(FINAL_LSTMAE_NB) $(HYPERPARAM_LSTMAE) $(BINARY_TRAIN)
	@echo "Training final LSTM Autoencoder model (QUICK_MODE=$(if $(QUICK_MODE),$(QUICK_MODE),False))..."
	@mkdir -p $(MODELS_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--log-level=INFO \
		$(FINAL_LSTMAE_NB)
	@touch $@
	@echo "✓ LSTM Autoencoder final model trained"

final-conv-ae: $(FINAL_CONVAE_MODEL) ## Train final Convolutional Autoencoder model

$(FINAL_CONVAE_MODEL): $(FINAL_CONVAE_NB) $(HYPERPARAM_CONVAE) $(BINARY_TRAIN)
	@echo "Training final Convolutional Autoencoder model (QUICK_MODE=$(if $(QUICK_MODE),$(QUICK_MODE),False))..."
	@mkdir -p $(MODELS_DIR)
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--log-level=INFO \
		$(FINAL_CONVAE_NB)
	@touch $@
	@echo "✓ Convolutional Autoencoder final model trained"

final-comparison: $(FINAL_COMPARISON) ## Generate model comparison summary

$(FINAL_COMPARISON): $(FINAL_COMPARISON_NB) $(FINAL_XGBOOST_MODEL) $(FINAL_LSTM_MODEL) $(FINAL_LSTMFCN_MODEL) $(FINAL_CNNTRANS_MODEL) $(FINAL_TRANSKAL_MODEL) $(FINAL_LSTMAE_MODEL) $(FINAL_CONVAE_MODEL)
	@echo "Generating model comparison summary..."
	QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
		--log-level=INFO \
		$(FINAL_COMPARISON_NB)
	@touch $@
	@echo "✓ Model comparison summary generated"

##@ Jupyter Book
.PHONY: book book-clean book-serve

BOOK_DIR := _build

book: ## Build Jupyter Book HTML
	@echo "Building Jupyter Book..."
	@jupyter-book build .
	@echo "Book built at $(BOOK_DIR)/html/index.html"

book-clean: ## Clean Jupyter Book build artifacts
	@echo "Cleaning Jupyter Book build..."
	@jupyter-book clean .
	@rm -rf $(BOOK_DIR)
	@echo "Jupyter Book build cleaned"

book-serve: book ## Build and serve Jupyter Book locally
	@echo "Serving Jupyter Book at http://localhost:8000..."
	@cd $(BOOK_DIR)/html && $(PYTHON) -m http.server 8000

book-pdf: ## Build Jupyter Book PDF (requires LaTeX)
	@echo "Building Jupyter Book PDF..."
	@jupyter-book build . --builder pdflatex
	@echo "PDF built at $(BOOK_DIR)/latex/book.pdf"

##@ Utilities
clean: ## Remove generated outputs (keeps source data)
	@echo "Cleaning generated files..."
	@rm -rf $(OUTPUTS_DIR)
	@rm -rf $(FIGURES_DIR)
	@echo "✓ Cleaned outputs and figures"

clean-data: ## Remove generated data files
	@echo "WARNING: This will delete all generated datasets!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf $(DATA_DIR); \
		echo "✓ Cleaned data files"; \
	else \
		echo "Cancelled"; \
	fi

clean-all: clean clean-data ## Remove all generated files and data
	@echo "✓ All generated files removed"

check: ## Verify all expected outputs exist
	@echo "Checking for expected outputs..."
	@missing=0; \
	for file in $(SYSTEM_INFO) $(REQUIREMENTS) $(MULTICLASS_TRAIN) $(MULTICLASS_VAL) $(MULTICLASS_TEST) $(EDA_SUMMARY); do \
		if [ -f $$file ]; then \
			echo "  ✓ $$file"; \
		else \
			echo "  ✗ $$file (missing)"; \
			missing=$$((missing+1)); \
		fi; \
	done; \
	if [ $$missing -eq 0 ]; then \
		echo "✓ All expected outputs present"; \
	else \
		echo "✗ $$missing file(s) missing"; \
		exit 1; \
	fi

status: ## Show status of generated files
	@echo "Project Status:"
	@echo "==============="
	@echo ""
	@echo "System Documentation:"
	@if [ -f $(SYSTEM_INFO) ]; then echo "  ✓ system_info.txt"; else echo "  ✗ system_info.txt"; fi
	@if [ -f $(REQUIREMENTS) ]; then echo "  ✓ requirements.txt"; else echo "  ✗ requirements.txt"; fi
	@echo ""
	@echo "Datasets:"
	@if [ -f $(MULTICLASS_TRAIN) ]; then echo "  ✓ multiclass_train.csv"; else echo "  ✗ multiclass_train.csv"; fi
	@if [ -f $(MULTICLASS_VAL) ]; then echo "  ✓ multiclass_val.csv"; else echo "  ✗ multiclass_val.csv"; fi
	@if [ -f $(MULTICLASS_TEST) ]; then echo "  ✓ multiclass_test.csv"; else echo "  ✗ multiclass_test.csv"; fi
	@if [ -f $(BINARY_TRAIN) ]; then echo "  ✓ binary_train.csv"; else echo "  ✗ binary_train.csv"; fi
	@if [ -f $(BINARY_VAL) ]; then echo "  ✓ binary_val.csv"; else echo "  ✗ binary_val.csv"; fi
	@if [ -f $(BINARY_TEST) ]; then echo "  ✓ binary_test.csv"; else echo "  ✗ binary_test.csv"; fi
	@echo ""
	@echo "EDA Outputs:"
	@if [ -f $(EDA_SUMMARY) ]; then echo "  ✓ eda_summary.txt"; else echo "  ✗ eda_summary.txt"; fi
	@if [ -f $(FAULT_DIST_FIG) ]; then echo "  ✓ fault_distribution.png"; else echo "  ✗ fault_distribution.png"; fi
	@if [ -f $(FEATURE_CORR_FIG) ]; then echo "  ✓ feature_correlations.png"; else echo "  ✗ feature_correlations.png"; fi
	@if [ -f $(FAULT_SIG_FIG) ]; then echo "  ✓ fault_signatures.png"; else echo "  ✗ fault_signatures.png"; fi

##@ Git Operations
git-status: ## Show git status
	@git status

git-add-code: ## Add code files to git (notebooks, configs)
	@echo "Adding code files to git..."
	@git add pyproject.toml .gitignore Makefile README.md
	@git add notebooks/*.ipynb
	@git add manuscript/
	@echo "✓ Code files staged"

git-commit: ## Create a git commit (requires message: make git-commit MSG="your message")
	@if [ -z "$(MSG)" ]; then \
		echo "Error: MSG variable required. Usage: make git-commit MSG=\"your message\""; \
		exit 1; \
	fi
	@git commit -m "$(MSG)"
	@echo "✓ Committed: $(MSG)"

##@ Development
run-notebook: ## Open Jupyter notebook server
	@echo "Starting Jupyter notebook server..."
	@$(JUPYTER) notebook --notebook-dir=$(NOTEBOOKS_DIR)

list-notebooks: ## List all notebooks
	@echo "Available notebooks:"
	@ls -1 $(NOTEBOOKS_DIR)/*.ipynb | sed 's/^/  /'
