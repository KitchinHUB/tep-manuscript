# Using LLMs in Scientific Research: A Case Study with Claude Code

This document describes how Claude Code (Anthropic's AI coding assistant) was used throughout the development of this TEP fault detection benchmark project. We share what worked well, workflow patterns that emerged, and lessons learned.

## Project Overview

This project benchmarks 7 machine learning models for fault detection and diagnosis on the Tennessee Eastman Process (TEP) dataset. The work involved:

- Creating balanced datasets from raw TEP simulation data
- Hyperparameter tuning with Optuna (50 trials per model, ~49 hours total)
- Final model training and evaluation
- HMM post-processing for temporal smoothing
- Validation on independently generated simulation data
- Manuscript preparation in LaTeX

### Background: A Year of Prior Work

This repository represents a **redo** of work that originally took four people approximately one year to complete manually. The original notebooks, model implementations, and analysis are preserved in the `v1/` folder for reference.

After completing the initial work and drafting a manuscript, we identified several concerns:

1. **Dataset imbalance** - The original datasets had a 16:1 class imbalance (normal vs. fault classes), which could bias model training and evaluation
2. **Consistency issues** - With multiple contributors working on different models over many months, notebook structures and evaluation procedures varied
3. **Reproducibility gaps** - Ad-hoc development made it difficult to reproduce results or systematically compare models

Rather than patch the existing work, we decided to redo the entire analysis from scratch using Claude Code. This time we had the benefit of **20/20 hindsight**: we knew exactly what models to implement, what the dataset issues were, and what evaluation pipeline we needed. Claude Code allowed us to execute this clear plan rapidly and consistently.

## Scope of Collaboration

**Timeline:** This entire repository was created in **7 days** (January 3-10, 2026), rebuilding what originally took four people approximately one year.

**By the numbers:**
- 39 total commits in the repository
- 33 commits (85%) co-authored with Claude
- Used Claude Sonnet 4.5 and Claude Opus 4.5
- 35+ Jupyter notebooks across 5 series
- ~800 lines of Makefile automation

Claude was involved in nearly every aspect of the project except running the long computational jobs (hyperparameter tuning took 49 hours, model training took several hours each).

## What Worked Well

### 1. Project Infrastructure & Boilerplate

Claude excelled at setting up project infrastructure that would be tedious to create manually:

**Makefile automation** - The project Makefile grew to include 50+ targets with proper dependencies:

```makefile
# Example: Final training depends on hyperparameters and data
$(FINAL_LSTM_MODEL): $(FINAL_LSTM_NB) $(HYPERPARAM_LSTM) $(MULTICLASS_TRAIN)
    @echo "Training final LSTM model..."
    QUICK_MODE=$(QUICK_MODE) $(JUPYTER) nbconvert --to notebook --execute --inplace \
        --log-level=INFO $(FINAL_LSTM_NB)
```

**QUICK_MODE pattern** - Claude suggested and implemented a `QUICK_MODE` environment variable that reduced dataset sizes and trial counts for rapid iteration:

```python
QUICK_MODE = os.environ.get('QUICK_MODE', 'False') == 'True'
N_TRIALS = 5 if QUICK_MODE else 50
SUBSAMPLE_FRAC = 0.01 if QUICK_MODE else 0.5
```

This pattern proved invaluable for testing the full pipeline before committing to multi-hour runs.

### 2. Repetitive Code Generation

With 7 models and 5 notebook series, the project required 35+ notebooks with similar structure but model-specific details. Claude maintained remarkable consistency across all notebooks while adapting:

- Model architectures and hyperparameter spaces
- Training loops and early stopping logic
- Evaluation metrics and visualization code
- File naming conventions

Each notebook series has the same structure:
- Import and setup cells
- Data loading with proper windowing
- Model definition
- Training loop with progress tracking
- Evaluation and metrics saving

### 3. Documentation Generation

After completing computational work, Claude was effective at synthesizing results into documentation:

**Summary files** - Each notebook series has an accompanying markdown summary (e.g., `10-summary.md`, `20-summary.md`) that aggregates results across all models with tables, key findings, and recommendations.

**Commit messages** - Claude generated detailed commit messages that serve as project history:

```
Fix TEPSimulator random_seed bug and transpose 30/40 series notebooks

- Fix critical bug in notebook 03: pass random_seed to TEPSimulator()
  constructor so each simulation run gets different random state
- Transpose 30-series (now HMM filters) and 40-series (now new data eval)
- Update notebook 04 to compare TEP-Sim against test data (fte) instead
  of training data, since TEP-Sim matches test structure

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

### 4. Debugging & Bug Fixes

Claude helped identify and fix several critical bugs:

**TEPSimulator random_seed bug** (`c9e629b`) - Simulations weren't getting different random seeds, causing supposedly independent runs to be identical. Claude identified the issue and fixed the constructor call.

**TransKal Kalman filter** (`75466e2`) - The Kalman filter wasn't being applied per-run, causing state leakage between simulation trajectories.

**Autoencoder evaluation split** (`2206f79`) - Autoencoders were being evaluated on validation data (which had only normal samples) instead of test data (which included faults).

### 5. Dataset Engineering

The dataset went through several iterations to achieve proper balance:

1. Initial: 16:1 class imbalance (normal vs fault classes)
2. Iteration 1: Undersampled normal class (`4695de1`)
3. Iteration 2: Renamed for clarity (supervised → multiclass) (`eff869a`)
4. Iteration 3: Increased test set size for statistical power (`b4950de`)
5. Final: 100 runs per class, perfectly balanced (`1854c97`)

Each iteration was a collaborative discussion about tradeoffs between dataset size, balance, and computational feasibility.

## Workflow Patterns

### Pattern 1: CLAUDE.md for Project Context

We created `CLAUDE.md` with project-specific instructions:

```markdown
## Directory Structure
├── notebooks/           # Jupyter notebooks organized by series
│   ├── 0X-*.ipynb      # Data creation and EDA
│   ├── 1X-*.ipynb      # Hyperparameter tuning (10-series)
...

## Important Notes
- Use QUICK_MODE=True for faster testing
- Hyperparameters are stored in outputs/hyperparams/<model>_best.json
```

This helped Claude understand naming conventions, file locations, and project-specific patterns across sessions.

### Pattern 2: Iterative Refinement

Rather than trying to get everything right the first time:

1. Start with QUICK_MODE to test the pipeline
2. Run full computations (hours/days)
3. Review results and iterate

Claude handled the code changes; humans reviewed scientific validity of results.

### Pattern 3: Summary Generation After Computation

The workflow for each notebook series:

1. Human runs long computational notebooks
2. Ask Claude to read outputs and generate summary markdown
3. Human reviews summary for accuracy
4. Commit both notebooks and summary together

### Pattern 4: Co-Author Attribution

All Claude-assisted commits include:

```
Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

This provides transparency about AI involvement and creates a traceable record.

## Limitations & Challenges

### 1. Long-Running Computations

Claude cannot execute code that takes hours to run. The workflow requires:
- Claude writes/modifies code
- Human executes long jobs
- Claude analyzes results

### 2. Scientific Judgment

While Claude can implement models and generate documentation, scientific decisions required human judgment:
- Which faults to exclude (3, 9, 15 were too subtle)
- Whether results indicate overfitting vs. distribution shift
- Interpretation of generalization failures

### 3. Context Across Sessions

Claude doesn't retain memory across sessions. The `CLAUDE.md` file partially addresses this, but complex project history still requires re-explaining.

### 4. Verification of Claims

Claude's documentation needed verification against actual results. Occasionally, summary statistics didn't match notebook outputs and required correction.

## Recommendations for Using LLMs in Research

1. **Create a CLAUDE.md (or similar) file** - Document project conventions, file locations, and important constraints upfront.

2. **Implement QUICK_MODE patterns** - Design workflows that can run quickly for iteration, then scale up for final results.

3. **Use detailed commit messages** - They serve as project documentation and help reconstruct reasoning later.

4. **Generate summaries after computation** - Let the LLM synthesize results from multiple sources into coherent documentation.

5. **Maintain human oversight for science** - LLMs excel at code and documentation but scientific interpretation requires domain expertise.

6. **Use co-author attribution** - Be transparent about AI involvement for reproducibility and ethics.

7. **Leverage LLMs for repetitive but precise tasks** - Creating 35 similar notebooks with consistent structure is tedious for humans but straightforward for LLMs.

## Conclusion

Claude Code proved valuable as a research assistant for this machine learning benchmark project. It handled infrastructure, boilerplate, documentation, and debugging effectively while humans focused on experimental design, running computations, and scientific interpretation. The collaboration produced a well-documented, reproducible research project with clear attribution of AI involvement.

The key insight is that LLMs work best as collaborators in a human-led research process, not as autonomous agents. They amplify researcher productivity on implementation details while humans retain responsibility for scientific judgment.
