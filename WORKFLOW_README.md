# Stanford RNA 3D Folding - Workflow Scripts

**Author**: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>  
**Email**: mauro.risonho@gmail.com  
**Created**: October 18, 2025  
**License**: MIT License  
**Competition**: [Stanford RNA 3D Folding](https://www.kaggle.com/competitions/stanford-rna-3d-folding)

---

## Overview

This set of scripts provides a complete workflow to configure, develop, and submit a solution for the Stanford RNA 3D Folding competition. The scripts are executed sequentially for maximum ease and reproducibility.

## Sequential Workflow

### **01_create_env.py** - Environment Setup

**Purpose**: Creates and configures a Python virtual environment with all necessary dependencies.

**Features:**
- Python version verification (minimum 3.9, recommended 3.13.5)
- System requirements verification (pip, git, CUDA)
- Isolated virtual environment creation
- Automatic ML/Data Science dependencies installation
- Installation validation with import tests
- Basic project structure creation
- Environment activation helper scripts

**Installed Dependencies:**
```
# Core ML and Deep Learning
torch>=2.9.0, pytorch-lightning>=2.5.0, transformers>=4.57.0

# Data Science
pandas>=2.3.0, numpy>=2.3.0, matplotlib>=3.10.0, seaborn>=0.13.0

# Bioinformatics  
biopython>=1.85

# Optimization and Tracking
optuna>=4.5.0, wandb>=0.22.0

# Jupyter and Development
jupyter>=1.1.0, jupyterlab>=4.4.0
```

**Usage:**
```bash
# Basic configuration
python 01_create_env.py

# Advanced configuration
python 01_create_env.py --verbose --log-level DEBUG

# Force recreation
python 01_create_env.py --force
```

### **02_setup_project.py** - Project Setup

**Purpose**: Sets up the complete project structure, downloads competition data and creates ready-to-use notebooks.

**Features:**
- Complete project directory structure
- Automatic competition data download (via Kaggle API)
- Structured Jupyter notebooks creation:
  - `01_eda.ipynb` - Exploratory data analysis
  - `02_baseline.ipynb` - LSTM baseline model
  - `03_advanced.ipynb` - Advanced models (Transformer, GNN, Ensemble)
  - `04_submission.ipynb` - Submission preparation
- Ready-to-use Python modules (`data_processing.py`, `models.py`)
- Configuration files (Makefile, .gitignore, requirements.txt)

**Created Structure:**
```
stanford_rna3d/
├── notebooks/          # Jupyter notebooks
├── src/               # Python code
├── data/              # Competition data
│   ├── raw/              # Original data
│   ├── processed/        # Processed data
│   └── external/         # External data
├── configs/           # Configurations
├── checkpoints/       # Trained models
├── submissions/       # Generated submissions
└── tests/            # Automated tests
```

**Usage:**
```bash
# Basic setup (without data download)
python 02_setup_project.py --dest ./stanford_rna3d --lang en

# Complete setup with download
python 02_setup_project.py --dest ./stanford_rna3d --lang en --download-data

# Portuguese version
python 02_setup_project.py --dest ./stanford_rna3d --lang pt
```

### **03_submit_late.py** - Submission and Portfolio

**Purpose**: Manages late submission for the competition and creates complete portfolio documentation.

**Features:**
- Automatic submission via Kaggle API (if competition active)
- Complete professional portfolio documentation
- Submission file validation
- Portfolio archive creation
- Detailed results and methodology report

**Usage:**
```bash
# Normal submission
python 03_submit_late.py --project ./stanford_rna3d --submission submission.csv

# Portfolio only (no submission)
python 03_submit_late.py --project ./stanford_rna3d --portfolio-only

# With compressed archive
python 03_submit_late.py --project ./stanford_rna3d --portfolio-only --archive
```

## Complete Usage Guide

### Step 1: Environment Setup
```bash
# Execute environment setup script
python 01_create_env.py --dev-packages

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Or use helper script
./activate_env.sh          # Linux/Mac  
activate_env.bat          # Windows
```

### Step 2: Project Setup
```bash
# Configure Kaggle API (optional, for automatic download)
# 1. Go to https://www.kaggle.com/account
# 2. Create API token and download kaggle.json
# 3. Place in ~/.kaggle/kaggle.json
# 4. chmod 600 ~/.kaggle/kaggle.json

# Create complete project
python 02_setup_project.py --dest ./stanford_rna3d --lang en --download-data

# Enter project
cd stanford_rna3d
```

### Step 3: Development
```bash
# Start Jupyter Lab
jupyter lab

# Execute notebooks in sequence:
# 1. notebooks/01_eda.ipynb - Exploratory analysis
# 2. notebooks/02_baseline.ipynb - Baseline model  
# 3. notebooks/03_advanced.ipynb - Advanced models
# 4. notebooks/04_submission.ipynb - Final submission
```

### Step 4: Submission/Portfolio
```bash
# Return to main directory
cd ..

# Submit (or create portfolio if competition closed)
python 03_submit_late.py --project ./stanford_rna3d --submission submission.csv --archive
```

## Technical Features

### Python Environment
- **Version**: 3.9+ (recommended 3.13.5)
- **Environment**: Isolated virtual environment
- **Dependencies**: Specific tested and compatible versions
- **GPU**: Automatic CUDA support if available

### Code Structure
- **Modular**: Clear separation between analysis, models and data
- **Documented**: Complete docstrings and comments
- **Testable**: Structure prepared for automated testing
- **Reproducible**: Fixed and versioned configurations

### Jupyter Notebooks
- **Sequential**: Logical development workflow
- **Interactive**: Ready-to-explore cells
- **Visualizations**: Integrated graphics for insights
- **Exportable**: Automatic conversion to reports

## Troubleshooting

### Error: Python too old
```bash
# Install newer Python
# Ubuntu/Debian:
sudo apt update && sudo apt install python3.13

# macOS (with Homebrew):
brew install python@3.13

# Windows: Download from python.org
```

### Error: Kaggle API not configured
```bash
# Configure Kaggle API
pip install kaggle
# Follow instructions: https://github.com/Kaggle/kaggle-api#api-credentials
```

### Error: Packages not installing
```bash
# Clear cache and reinstall
pip cache purge
python 01_create_env.py --force
```

### Error: CUDA not available
```bash
# Check NVIDIA drivers
nvidia-smi

# Install PyTorch with CUDA
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

## Generated Files

After executing the complete workflow, you will have:

1. **Configured environment** with all dependencies
2. **Structured project** with ready code and notebooks
3. **Downloaded data** from competition (if API configured)
4. **Complete documentation** for portfolio
5. **Generated and validated submission**

## Useful Links

- **Competition**: https://www.kaggle.com/competitions/stanford-rna-3d-folding
- **Kaggle API**: https://github.com/Kaggle/kaggle-api
- **PyTorch**: https://pytorch.org/get-started/locally/
- **Jupyter**: https://jupyter.org/install

---

**Next Steps**: After executing the three scripts, you will have a complete development environment for the Stanford RNA 3D Folding competition, ready for analysis, modeling and submission!