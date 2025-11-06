# Stanford RNA 3D Folding — Kaggle Competition

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3135/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 11.8](https://img.shields.io/badge/CUDA-11.8-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

**Predicting 3D RNA structure from sequence using deep learning**

[Competition](https://www.kaggle.com/competitions/stanford-rna-3d-folding) •
[Documentation](stanford_rna3d/docs/) •
[Notebooks](stanford_rna3d/notebooks/)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Environment Setup](#-environment-setup)
- [Usage](#-usage)
- [GPU Support](#-gpu-support)
- [Kaggle Submission](#-kaggle-submission)
- [Notebooks](#-notebooks)
- [Documentation](#-documentation)
- [Performance](#-performance)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

This repository contains a complete solution pipeline for the **Stanford RNA 3D Folding** Kaggle competition, which challenges participants to predict 3D atomic coordinates of RNA molecules from their nucleotide sequences.

**Competition Goal**: Predict the 3D spatial coordinates (x, y, z) for each nucleotide in RNA sequences to advance computational biology and drug discovery.

**Approach**: Deep learning models (LSTM/Transformer-based) trained on RNA sequences with multiple sequence alignments (MSA) features.

---

## ✨ Key Features

- 🧬 **End-to-end RNA 3D structure prediction pipeline**
- 🚀 **GPU-accelerated training** (CUDA 11.8 support for sm_61+ architectures)
- 📊 **Interactive Jupyter notebooks** for EDA, training, and submission
- 🔄 **Automated preprocessing** with MSA feature extraction
- 📤 **One-click Kaggle submission** with secure credential handling
- 🧪 **Production-ready model deployment** with validation and post-processing
- 📝 **Comprehensive documentation** and code examples
- 🛠️ **Modular architecture** for easy experimentation

---

## 📁 Project Structure

```
Stanford-RNA-3D-Folding/
├── stanford_rna3d/              # Main project directory
│   ├── data/                    # Data storage
│   │   ├── raw/                 # Competition data (sequences, labels, MSA)
│   │   ├── processed/           # Preprocessed features
│   │   ├── interim/             # Intermediate processing files
│   │   └── external/            # External datasets
│   ├── notebooks/               # Jupyter notebooks
│   │   ├── 00_competition_overview.ipynb
│   │   ├── 01_eda.ipynb         # Exploratory data analysis
│   │   ├── 02_baseline.ipynb   # Baseline model training
│   │   ├── 03_advanced.ipynb   # Advanced architectures
│   │   └── 04_submission.ipynb # Submission generation + Kaggle upload
│   ├── src/                     # Source code modules
│   │   ├── __init__.py
│   │   ├── models.py            # Neural network architectures
│   │   └── data_processing.py  # Data preprocessing utilities
│   ├── scripts/                 # Utility scripts
│   │   ├── 00_environment_manager.py
│   │   ├── 01_create_env.py
│   │   ├── 02_setup_project.py
│   │   ├── 03_submit_late.py   # CLI submission tool
│   │   ├── pii_scanner.py
│   │   └── system_specs_checker.py
│   ├── checkpoints/             # Saved model weights
│   ├── submissions/             # Generated submission files
│   ├── configs/                 # Configuration files
│   ├── docs/                    # Detailed documentation
│   ├── tests/                   # Unit tests
│   ├── requirements.txt         # Python dependencies
│   ├── Makefile                 # Build automation
│   └── .venv/                   # Virtual environment
├── scripts/                     # Root-level scripts
│   └── setup_dev_env.py
├── LICENSE                      # MIT License
├── README.md                    # This file
├── .gitignore
└── mypy.ini                     # Type checking config
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.13.5+**
- **CUDA 11.8** (for GPU support, optional)
- **Git**
- **8GB+ RAM** (16GB+ recommended)
- **NVIDIA GPU with sm_61+ compute capability** (e.g., GTX 1060 or better)

### Installation

```bash
# Clone the repository
git clone https://github.com/maurorisonho/Stanford-RNA-3D-Folding.git
cd Stanford-RNA-3D-Folding/stanford_rna3d

# Create and activate virtual environment
python3.13 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies with GPU support
pip install --upgrade pip
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Verify GPU detection
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### Download Competition Data

```bash
# Option 1: Manual download from Kaggle
# Visit https://www.kaggle.com/competitions/stanford-rna-3d-folding/data
# Download and extract to stanford_rna3d/data/raw/

# Option 2: Using Kaggle API (requires kaggle.json in ~/.kaggle/)
kaggle competitions download -c stanford-rna-3d-folding -p data/raw/
unzip data/raw/stanford-rna-3d-folding.zip -d data/raw/
```

---

## 🔧 Environment Setup

### Manual Setup

```bash
cd stanford_rna3d
python3.13 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA 11.8
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

### Automated Setup

```bash
cd stanford_rna3d
python scripts/01_create_env.py     # Creates virtual environment
python scripts/02_setup_project.py  # Sets up project structure
```

### Verify Installation

```bash
python scripts/system_specs_checker.py
```

Expected output:
```
✓ Python 3.13.5
✓ PyTorch 2.7.1+cu118
✓ CUDA 11.8
✓ GPU: NVIDIA GeForce GTX 1060
```

---

## 💻 Usage

### 1. Exploratory Data Analysis

```bash
jupyter lab notebooks/01_eda.ipynb
```

Explore RNA sequences, labels, MSA features, and data distributions.

### 2. Train Baseline Model

```bash
jupyter lab notebooks/02_baseline.ipynb
```

Train a simple LSTM-based model for RNA coordinate prediction.

### 3. Train Advanced Model

```bash
jupyter lab notebooks/03_advanced.ipynb
```

Experiment with Transformer architectures and graph neural networks.

### 4. Generate Submission

```bash
jupyter lab notebooks/04_submission.ipynb
```

- Load best model from `checkpoints/`
- Generate predictions for test set
- Apply post-processing (smoothing, normalization)
- Create submission CSV
- Upload to Kaggle (interactive button)

---

## 🎮 GPU Support

### Enable CUDA

The project automatically detects and uses available GPUs. PyTorch is configured with CUDA 11.8 for compatibility with compute capability 6.1+ GPUs (e.g., GTX 1060, RTX series).

**Verification:**

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

### Disable CUDA (CPU-only mode)

If you need to force CPU execution:

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

---

## 📤 Kaggle Submission

### Method 1: Interactive Notebook Upload (Recommended)

1. Open `notebooks/04_submission.ipynb`
2. Run all cells to generate submission
3. Click **"🚀 Upload to Kaggle"** button in the last cell
4. Enter credentials when prompted (input is hidden for security)

### Method 2: CLI Upload

```bash
python scripts/03_submit_late.py submissions/submission_20251103_215343.csv -m "First submission"
```

### Method 3: Manual Upload

1. Generate submission: Run `04_submission.ipynb`
2. Download CSV from `submissions/`
3. Upload at: https://www.kaggle.com/competitions/stanford-rna-3d-folding/submit

### Kaggle API Setup

**Option A: Environment Variables** (Recommended for security)

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

**Option B: Kaggle JSON file**

```bash
mkdir -p ~/.kaggle
# Create ~/.kaggle/kaggle.json with:
# {"username": "your_username", "key": "your_api_key"}
chmod 600 ~/.kaggle/kaggle.json
```

⚠️ **Security Note**: Never commit `kaggle.json` or API keys to the repository!

---

## 📓 Notebooks

| Notebook | Description | Key Features |
|----------|-------------|--------------|
| **00_competition_overview.ipynb** | Competition introduction and objectives | Problem statement, evaluation metrics |
| **01_eda.ipynb** | Exploratory data analysis | Sequence statistics, label distributions, visualizations |
| **02_baseline.ipynb** | Baseline model training | Simple LSTM, data preprocessing, training loop |
| **03_advanced.ipynb** | Advanced architectures | Transformers, attention mechanisms, GNNs |
| **04_submission.ipynb** | Submission generation | Model loading, inference, post-processing, Kaggle upload |

---

## 📚 Documentation

Comprehensive documentation is available in [`stanford_rna3d/docs/`](stanford_rna3d/docs/):

- **[ENVIRONMENT_SETUP.md](stanford_rna3d/docs/ENVIRONMENT_SETUP.md)** — Environment configuration guide
- **[DATA_DOWNLOAD.md](stanford_rna3d/docs/DATA_DOWNLOAD.md)** — Data acquisition instructions
- **[EXECUTION_PIPELINE.md](stanford_rna3d/docs/EXECUTION_PIPELINE.md)** — End-to-end workflow
- **[TECHNICAL_DETAILS.md](stanford_rna3d/docs/TECHNICAL_DETAILS.md)** — Model architectures and algorithms
- **[SOLUTION_WRITEUP.md](stanford_rna3d/docs/SOLUTION_WRITEUP.md)** — Complete solution documentation
- **[WORKFLOW_README.md](stanford_rna3d/docs/WORKFLOW_README.md)** — Development workflow
- **[pii_scanner_README.md](stanford_rna3d/docs/pii_scanner_README.md)** — Security scanning tools

---

## 📊 Performance

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU-only (slow training)
- **Recommended**: 16GB RAM, NVIDIA GPU with 4GB+ VRAM
- **Optimal**: 32GB RAM, NVIDIA RTX GPU with 8GB+ VRAM

### Training Times (Approximate)

| Configuration | Baseline Model | Advanced Model |
|--------------|----------------|----------------|
| CPU-only | ~4 hours | ~12 hours |
| GTX 1060 (6GB) | ~45 minutes | ~2 hours |
| RTX 3080 (10GB) | ~20 minutes | ~1 hour |

### Evaluation Metric

The competition uses **Mean Absolute Error (MAE)** between predicted and true atomic coordinates:

$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
$$

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest mypy flake8 black

# Run tests
pytest tests/

# Type checking
mypy src/

# Code formatting
black src/ notebooks/
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Mauro Risonho de Paula Assumpção

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
```

---

## 📧 Contact

**Author**: Mauro Risonho de Paula Assumpção

- **Email**: mauro.risonho@gmail.com
- **GitHub**: [@maurorisonho](https://github.com/maurorisonho)
- **Kaggle**: [Stanford RNA 3D Folding Competition](https://www.kaggle.com/competitions/stanford-rna-3d-folding)

---

## 🙏 Acknowledgments

- **Stanford University** for hosting the competition
- **Kaggle** for providing the platform
- **PyTorch Team** for the deep learning framework
- **BioPython** for bioinformatics utilities
- The RNA structure prediction research community

---

## 🔗 Related Resources

- [Kaggle Competition](https://www.kaggle.com/competitions/stanford-rna-3d-folding)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [BioPython Tutorial](https://biopython.org/wiki/Documentation)
- [RNA Structure Prediction Review](https://www.nature.com/articles/s41586-021-03819-2)

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

Made with ❤️ for the RNA structure prediction community

</div>
