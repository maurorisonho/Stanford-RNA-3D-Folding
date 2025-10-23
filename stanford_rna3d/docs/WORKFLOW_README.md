# Project Workflow Guide

**Author**: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>  
**Created**: October 18, 2025 at 14:30:00  
**License**: MIT License  
**Kaggle Competition**: https://www.kaggle.com/competitions/stanford-rna-3d-folding  

---

**MIT License**

Copyright (c) 2025 Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---



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
