# Stanford RNA 3D Folding — Portfolio Project

**Author**: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>  
**Email**: mauro.risonho@gmail.com  
**Created**: October 18, 2025 at 14:30:00  
**License**: MIT License  
**Kaggle Competition**: https://www.kaggle.com/competitions/stanford-rna-3d-folding  
**Python Version**: 3.13.5 (Latest Stable)

---

This repository demonstrates cutting-edge machine learning techniques for RNA 3D structure prediction using the latest compatible libraries and frameworks.

## MIT License

Copyright (c) 2025 Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Project Structure
- `notebooks/` - Jupyter notebooks with complete analysis pipeline
- `src/` - Python source code modules
- `data/` - Competition datasets and processed data
- `configs/` - Model configuration files
- `SOLUTION_WRITEUP.md` - Complete solution documentation
- `TECHNICAL_DETAILS.md` - Detailed technical implementation

## Quick Start
1. Create & activate a Python 3.13.5 virtual environment: `python3.13 -m venv .venv && source .venv/bin/activate`
2. Install requirements: `pip install -r requirements.txt`
3. Start Jupyter: `jupyter lab`
4. Execute notebooks in sequence: `01_eda.ipynb` → `02_baseline.ipynb` → `03_advanced.ipynb` → `04_submission.ipynb`

## Solution Overview
Our approach combines multiple advanced architectures:
- **Baseline**: LSTM with attention mechanism
- **Advanced**: Transformer-based architecture with physics constraints
- **Ensemble**: Multi-model ensemble with uncertainty quantification
- **Innovation**: Physics-informed neural networks for biological validity

## Documentation
- **[Solution Write-up](SOLUTION_WRITEUP.md)**: Complete competition solution documentation
- **[Technical Details](TECHNICAL_DETAILS.md)**: Detailed implementation specifications
- **[Environment Setup](ENVIRONMENT_SETUP.md)**: Development environment configuration
- LSTM and Transformer architectures
- 3D coordinate prediction
