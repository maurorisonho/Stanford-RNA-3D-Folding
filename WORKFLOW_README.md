# Project Workflow Guide

**Author**: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>  
**License**: MIT License

This document outlines a suggested workflow for experimenting with the Stanford
RNA 3D Folding project. Adapt the steps to match your own exploration style.

## 1. Environment Preparation

1. Ensure Python 3.13.5 is installed on your machine.
2. Create the virtual environment and install dependencies:
   ```bash
   python3.13 01_create_env.py
   source .venv/bin/activate
   ```
3. Verify the installation by running:
   ```bash
   python -m pip check
   ```

## 2. Project Scaffolding

1. Create directories, notebook stubs, and fetch data (if credentials allow):
   ```bash
   python 02_setup_project.py
   ```
2. Inspect `stanford_rna3d/data/raw/` to confirm that the Kaggle files are
   available. If the download was skipped, retrieve them manually (see
   `DATA_DOWNLOAD.md`).

## 3. Exploratory Analysis

1. Launch Jupyter Lab from the repository root:
   ```bash
   jupyter lab
   ```
2. Work through `notebooks/01_eda.ipynb` to understand the dataset and produce
   summary visualisations.

## 4. Baseline Modelling

1. Implement data loaders and baseline models in `stanford_rna3d/src`.
2. Document experiments in `notebooks/02_baseline.ipynb`, saving artefacts to
   `stanford_rna3d/checkpoints/`.
3. Use `tests/` for lightweight checks that guard against regressions.

## 5. Advanced Experiments

1. Iterate on richer models within `notebooks/03_advanced.ipynb` leveraging
   additional techniques (transformers, graph models, physic-informed loss, etc.).
2. Track hyper-parameter configurations using Optuna or W&B if desired.
3. Record findings in `stanford_rna3d/SOLUTION_WRITEUP.md`.

## 6. Submission

1. Export predictions to `stanford_rna3d/submissions/<timestamp>.csv`.
2. Submit to Kaggle:
   ```bash
   python 03_submit_late.py stanford_rna3d/submissions/best.csv \
       --message "Submission generated on $(date)"
   ```
3. Store the corresponding score and notes in `stanford_rna3d/EXECUTION_PIPELINE.md`.

## 7. Maintenance

- Commit frequently and open pull requests using the template in `PR_TEMPLATE.md`.
- Keep documentation aligned with implementation changes.
- Re-run `01_create_env.py --recreate` when dependencies change substantially.
