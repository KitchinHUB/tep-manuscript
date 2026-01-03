.PHONY: help clean all 00-series data eda system install

# Default target
.DEFAULT_GOAL := help

# Directories
NOTEBOOKS_DIR := notebooks
DATA_DIR := data
OUTPUTS_DIR := outputs
FIGURES_DIR := figures

# Python executable
PYTHON := python3

# Jupyter executable
JUPYTER := jupyter

# Notebook files
SYSTEM_NB := $(NOTEBOOKS_DIR)/00-system.ipynb
DATASET_NB := $(NOTEBOOKS_DIR)/01-create-datasets.ipynb
EDA_NB := $(NOTEBOOKS_DIR)/02-exploratory-data-analysis.ipynb

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

# EDA artifacts
EDA_SUMMARY := $(OUTPUTS_DIR)/eda_summary.txt
FAULT_DIST_FIG := $(FIGURES_DIR)/fault_distribution.png
FEATURE_CORR_FIG := $(FIGURES_DIR)/feature_correlations.png
FAULT_SIG_FIG := $(FIGURES_DIR)/fault_signatures.png

##@ Help
help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\\nUsage:\\n  make \\033[36m<target>\\033[0m\\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \\033[36m%-20s\\033[0m %s\\n", $$1, $$2 } /^##@/ { printf "\\n\\033[1m%s\\033[0m\\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup
install: ## Install Python dependencies
	@echo "Installing dependencies..."
	pip install -q numpy pandas scikit-learn matplotlib seaborn pyreadr tqdm jupyter ipykernel nbformat psutil
	@echo "✓ Dependencies installed"

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
		--ExecutePreprocessor.timeout=600 \
		$(SYSTEM_NB)
	@echo "✓ System information documented"

data: $(MULTICLASS_TRAIN) ## Run 01-create-datasets.ipynb - Create reproducible datasets

$(MULTICLASS_TRAIN) $(MULTICLASS_VAL) $(MULTICLASS_TEST) $(BINARY_TRAIN) $(BINARY_VAL) $(BINARY_TEST): $(DATASET_NB)
	@echo "Running 01-create-datasets.ipynb..."
	@echo "This may take several minutes..."
	@mkdir -p $(DATA_DIR)
	@$(JUPYTER) nbconvert --to notebook --execute \
		--output 01-create-datasets.ipynb \
		--ExecutePreprocessor.timeout=1800 \
		$(DATASET_NB)
	@echo "✓ Datasets created"

eda: $(EDA_SUMMARY) ## Run 02-exploratory-data-analysis.ipynb - Analyze datasets

$(EDA_SUMMARY) $(FAULT_DIST_FIG) $(FEATURE_CORR_FIG) $(FAULT_SIG_FIG): $(EDA_NB) $(MULTICLASS_TRAIN)
	@echo "Running 02-exploratory-data-analysis.ipynb..."
	@mkdir -p $(OUTPUTS_DIR) $(FIGURES_DIR)
	@$(JUPYTER) nbconvert --to notebook --execute \
		--output 02-exploratory-data-analysis.ipynb \
		--ExecutePreprocessor.timeout=600 \
		$(EDA_NB)
	@echo "✓ Exploratory data analysis complete"

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
