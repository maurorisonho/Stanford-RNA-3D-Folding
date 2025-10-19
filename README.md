# Stanford RNA 3D Folding - Machine Learning Competition

**Author**: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>  
**Created**: October 18, 2025  
**License**: MIT License  
**Competition**: [Stanford RNA 3D Folding](https://www.kaggle.com/competitions/stanford-rna-3d-folding)

---

## Overview

This repository contains a comprehensive solution for the **Stanford RNA 3D Folding** Kaggle competition, implementing advanced machine learning techniques for predicting RNA 3D structures from sequence data. The project demonstrates expertise in bioinformatics, deep learning, and scientific computing.

## Problem Statement

The challenge involves predicting the 3D coordinates of RNA molecules from their nucleotide sequences. This is a critical problem in computational biology with applications in:

- Drug discovery and development
- Understanding RNA function and regulation
- Protein-RNA interaction prediction
- Therapeutic RNA design

## Solution Architecture

### Machine Learning Pipeline

1. **Data Processing**: Advanced sequence encoding and 3D coordinate normalization
2. **Baseline Models**: LSTM-based sequence-to-structure prediction
3. **Advanced Models**: 
   - Transformer architectures with positional encoding
   - Graph Neural Networks for molecular relationships
   - Physics-Informed Neural Networks with domain constraints
   - Ensemble methods for improved accuracy

### Technical Stack

- **Deep Learning**: PyTorch 2.9.0, PyTorch Lightning 2.5.0
- **Transformers**: Hugging Face Transformers 4.57.1
- **Scientific Computing**: NumPy, SciPy, BioPython
- **Data Analysis**: Pandas, Matplotlib, Seaborn, Plotly
- **Optimization**: Optuna for hyperparameter tuning
- **Experiment Tracking**: Weights & Biases

## Project Structure

```
Stanford-RNA-3D-Folding/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── .gitignore                  # Git ignore rules
├── DATA_DOWNLOAD.md            # Data acquisition instructions
├── WORKFLOW_README.md          # Detailed workflow documentation
├── README_SCRIPTS.md           # Script usage guide
│
├── 01_create_env.py            # Environment setup automation
├── 02_setup_project.py         # Project structure creation
├── 03_submit_late.py           # Submission and portfolio generation
│
└── stanford_rna3d/             # Main project directory
    ├── README.md               # Project-specific documentation
    ├── SOLUTION_WRITEUP.md     # Comprehensive solution description
    ├── TECHNICAL_DETAILS.md    # Technical implementation details
    ├── EXECUTION_PIPELINE.md   # Step-by-step execution guide
    ├── ENVIRONMENT_SETUP.md    # Environment configuration
    │
    ├── notebooks/              # Jupyter notebooks
    │   ├── 01_eda.ipynb       # Exploratory data analysis
    │   ├── 02_baseline.ipynb  # Baseline LSTM model
    │   ├── 03_advanced.ipynb  # Advanced models (Transformer, GNN)
    │   └── 04_submission.ipynb # Submission preparation
    │
    ├── src/                    # Source code modules
    │   ├── __init__.py
    │   ├── data_processing.py  # Data preprocessing utilities
    │   └── models.py          # Model architectures
    │
    ├── data/                   # Data directory structure
    │   ├── raw/               # Original competition data
    │   ├── processed/         # Processed datasets
    │   └── external/          # External datasets
    │
    ├── configs/               # Configuration files
    ├── checkpoints/           # Model checkpoints
    ├── submissions/           # Generated submissions
    └── tests/                 # Unit tests
```

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/maurorisonho/Stanford-RNA-3D-Folding.git
cd Stanford-RNA-3D-Folding

# Create and configure environment
python 01_create_env.py

# Activate environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### 2. Project Creation

```bash
# Create complete project structure with notebooks and documentation
python 02_setup_project.py --dest ./stanford_rna3d --lang en

# Optional: Download competition data (requires Kaggle API)
python 02_setup_project.py --dest ./stanford_rna3d --lang en --download-data
```

### 3. Development Workflow

```bash
# Enter project directory
cd stanford_rna3d

# Start Jupyter Lab for development
jupyter lab

# Execute notebooks in sequence:
# 1. notebooks/01_eda.ipynb - Comprehensive data exploration
# 2. notebooks/02_baseline.ipynb - LSTM baseline implementation
# 3. notebooks/03_advanced.ipynb - Advanced model development
# 4. notebooks/04_submission.ipynb - Final submission preparation
```

### 4. Submission Generation

```bash
# Return to root directory
cd ..

# Generate submission and portfolio documentation
python 03_submit_late.py --project ./stanford_rna3d --portfolio-only --archive
```

## Key Features

### Advanced Machine Learning

- **Multi-Architecture Approach**: LSTM, Transformer, and GNN models
- **Physics-Informed Learning**: Domain knowledge integration
- **Ensemble Methods**: Optimized model combination
- **Hyperparameter Optimization**: Automated tuning with Optuna

### Bioinformatics Expertise

- **Sequence Analysis**: Advanced RNA sequence processing
- **Structural Biology**: 3D coordinate prediction and validation
- **Domain Knowledge**: Physics-based constraints and validation
- **Evaluation Metrics**: RMSD, GDT-TS, and custom scoring

### Software Engineering

- **Modular Architecture**: Clean, maintainable code structure
- **Comprehensive Documentation**: Detailed explanations and guides
- **Automated Workflows**: Scripts for setup, development, and submission
- **Testing Framework**: Unit tests and validation pipelines
- **Version Control**: Git-based development workflow

## Performance Metrics

The solution implements multiple evaluation metrics relevant to structural biology:

- **RMSD (Root Mean Square Deviation)**: Primary evaluation metric
- **GDT-TS Score**: Global Distance Test - Total Score
- **Physics Validation**: Energy-based structure validation
- **Custom Metrics**: Domain-specific performance measures

## Documentation

Comprehensive documentation is provided for all aspects of the project:

- **[Solution Writeup](stanford_rna3d/SOLUTION_WRITEUP.md)**: Complete solution description
- **[Technical Details](stanford_rna3d/TECHNICAL_DETAILS.md)**: Implementation specifics
- **[Execution Pipeline](stanford_rna3d/EXECUTION_PIPELINE.md)**: Step-by-step guide
- **[Environment Setup](stanford_rna3d/ENVIRONMENT_SETUP.md)**: Configuration details
- **[Workflow Guide](WORKFLOW_README.md)**: Automated workflow documentation
- **[Data Instructions](DATA_DOWNLOAD.md)**: Data acquisition guide

## Requirements

### System Requirements

- Python 3.9+ (recommended 3.13.5)
- 8GB+ RAM (16GB+ recommended)
- GPU with CUDA support (optional but recommended)
- 10GB+ disk space

### Python Dependencies

All dependencies are automatically installed by the setup scripts:

```
# Core ML and Deep Learning
torch>=2.9.0
pytorch-lightning>=2.5.0
transformers>=4.57.1

# Scientific Computing
numpy>=2.3.0
scipy>=1.14.0
biopython>=1.85

# Data Analysis
pandas>=2.3.0
matplotlib>=3.10.0
seaborn>=0.13.0
plotly>=5.0.0

# Optimization and Tracking
optuna>=4.5.0
wandb>=0.22.0

# Development
jupyter>=1.1.0
jupyterlab>=4.4.0
```

## Competition Context

This project was developed for the Stanford RNA 3D Folding competition on Kaggle, which challenges participants to predict 3D molecular structures from RNA sequences. The competition represents a significant challenge in computational biology and has real-world applications in:

- **Drug Discovery**: Understanding RNA targets for therapeutic intervention
- **Synthetic Biology**: Designing RNA molecules with specific functions
- **Fundamental Research**: Advancing our understanding of RNA structure-function relationships

## Results and Impact

The solution demonstrates:

- Advanced machine learning techniques applied to structural biology
- Integration of domain knowledge with deep learning
- Comprehensive software engineering practices
- Professional documentation and reproducibility standards

## Contributing

This is a competition project and portfolio demonstration. For questions or collaboration opportunities, please contact:

**Mauro Risonho de Paula Assumpção**  
Email: mauro.risonho@gmail.com  
LinkedIn: [Connect with me](https://linkedin.com/in/maurorisonho)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Stanford University for organizing the competition
- Kaggle for providing the platform and data
- The computational biology community for foundational research
- Open source contributors for the underlying tools and libraries

---

**Built with precision, documented with care, and designed for impact.**