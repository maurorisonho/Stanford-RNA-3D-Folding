# Environment Setup Report

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


**Date**: October 2025  
**Python Version**: 3.13.5  

## Environment Configuration Complete

### Python Version
- **Requested**: Python 3.13.x
- **Configured**: Python 3.13.5 (virtual environment `.venv`)
- **Environment Type**: Virtual Environment (venv)

### Core Libraries (Verified Compatible Versions)

#### Data Science & Analysis
- **pandas**: 2.1.4
- **numpy**: 1.26.4
- **scipy**: 1.11.4
- **matplotlib**: 3.8.2
- **seaborn**: 0.13.2
- **plotly**: 5.18.0
- **scikit-learn**: 1.3.2

#### Deep Learning & AI
- **torch**: 2.1.0+cpu
- **torchvision**: 0.16.0+cpu
- **transformers**: 4.35.2
- **pytorch-lightning**: 2.1.3
- **torchmetrics**: 1.2.0

#### Optimisation & Experiment Tracking
- **optuna**: 3.4.0
- **wandb**: 0.15.12

#### Bioinformatics
- **biopython**: 1.81

#### Development & Jupyter
- **jupyter**: 1.0.0
- **ipykernel**: 6.25.2
- **jupyterlab**: 4.0.7

### Status Summary
- Python environment configured with virtual environment
- All required libraries installed with compatible versions
- No import errors detected
- Notebooks configured with correct kernel
- First notebook executed successfully

### Next Steps
1. Notebooks set to use the `.venv` kernel (`Python 3.13.5`)
2. Environment validated for RNA 3D folding analysis
3. Compatible ML/AI libraries installed and importable

### Notes
- Ensure VS Code references `${workspaceFolder}/.venv/bin/python`.
- If `pip install -r requirements.txt` reports missing wheels, upgrade `pip`, `setuptools`, and `wheel` first.
- GPU users should install CUDA-enabled wheels for PyTorch when necessary.
