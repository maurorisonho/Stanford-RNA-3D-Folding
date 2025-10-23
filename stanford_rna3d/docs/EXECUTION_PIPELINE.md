# Execution Pipeline – Stanford RNA 3D Folding

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



This pipeline describes the recommended order of operations for training and
evaluating models in the repository.

## 1. Environment

1. Create or update the project virtual environment:
   ```bash
   python3.13 01_create_env.py
   source .venv/bin/activate
   ```
2. Validate that dependencies install correctly:
   ```bash
   python -m pip check
   ```

## 2. Data Preparation

1. Fetch competition data using:
   ```bash
   python 02_setup_project.py
   ```
   or download manually and place the files under `stanford_rna3d/data/raw/`.
2. Run any preprocessing scripts or notebooks to generate `processed/`
   artefacts (for example, feature engineering stored in parquet files).

## 3. Experimentation

1. `notebooks/01_eda.ipynb` – inspect dataset characteristics.
2. `notebooks/02_baseline.ipynb` – train baseline LSTM models.
3. `notebooks/03_advanced.ipynb` – run transformer experiments, hyper-parameter
   sweeps, and ensembling logic.

Intermediate outputs should be saved in `stanford_rna3d/checkpoints/` with
timestamped filenames and accompanying configuration metadata.

## 4. Evaluation

1. Use validation splits defined in the notebooks or via utility functions in
   `stanford_rna3d/src/data_processing.py`.
2. Monitor RMSD on validation folds and log results to the experiment tracker
   of choice (e.g. W&B).
3. Store evaluation summaries in `stanford_rna3d/SOLUTION_WRITEUP.md`.

## 5. Submission

1. Create predictions on the test set and export them to CSV in
   `stanford_rna3d/submissions/`.
2. Submit to Kaggle:
   ```bash
   python 03_submit_late.py stanford_rna3d/submissions/<file>.csv \
       --message "Model description"
   ```
3. Record the submission score, Kaggle link, and any observations back in this
   document.

## 6. Maintenance

- Keep the documentation up to date when pipelines change.
- Run `system_specs_checker.py --output system_specs.json` when benchmarking
  hardware for reproducibility notes.
- Use the PR template (`PR_TEMPLATE.md`) for any collaborative work.
