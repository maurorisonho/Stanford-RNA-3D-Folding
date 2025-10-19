# Stanford RNA 3D Folding - Portfolio Scripts

This project contains 2 Python scripts to work with the Stanford RNA 3D Folding Kaggle competition as a portfolio project.

## Available Scripts

### 1. `02_setup_project.py` - Project Setup
Creates complete project structure with notebooks, source code and documentation.

**Features:**
- Professional project structure
- Pre-configured Jupyter notebooks
- Modular Python source code
- Documentation in English or Portuguese
- Automatic data download (optional)
- Configurations and Makefile

**Usage:**
```bash
# Create project in English
python 02_setup_project.py --dest ./stanford_rna3d --lang en

# Create project in Portuguese
python 02_setup_project.py --dest ./stanford_rna3d --lang pt

# Include automatic data download
python 02_setup_project.py --dest ./stanford_rna3d --download-data
```

### 2. `03_submit_late.py` - Late Submission
Manages late submission to Kaggle with portfolio documentation.

**Features:**
- Kaggle submission attempt
- Graceful fallback if competition is closed
- Portfolio documentation generation
- Submission file and report
- ZIP archive creation for portfolio

**Usage:**
```bash
# Try normal submission
python 03_submit_late.py --project ./stanford_rna3d

# Portfolio documentation only
python 03_submit_late.py --project ./stanford_rna3d --portfolio-only

# Create portfolio archive
python 03_submit_late.py --project ./stanford_rna3d --archive
```

## Complete Workflow

### Step 1: Prepare Environment
```bash
# 1. Create virtual environment (example using venv)
python3 -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\activate  # Windows PowerShell

# 3. Install main dependencies (include extras as needed)
pip install --upgrade pip
pip install -r stanford_rna3d/requirements.txt
pip install kaggle
```

### Step 2: Configure Project
```bash
# 4. Create project structure
python 02_setup_project.py --dest ./stanford_rna3d --lang en

# 5. Enter project
cd stanford_rna3d
# (optional) Reinstall specific dependencies
pip install -r requirements.txt
```

### Step 3: Development
```bash
# 6. (Optional) Register current environment kernel
python -m ipykernel install --user --name stanford_rna3d --display-name "Python (stanford_rna3d)"

# 7. Start Jupyter Lab
jupyter lab

# 8. Work on notebooks (4 complete notebooks created):
# - notebooks/01_eda.ipynb (Detailed exploratory analysis)
# - notebooks/02_baseline.ipynb (LSTM baseline model)
# - notebooks/03_advanced.ipynb (Transformer, GNN, Ensemble models)
# - notebooks/04_submission.ipynb (Final submission preparation)
```

### Step 4: Submission and Portfolio
```bash
# 9. Generate submission and documentation
cd ..
python 03_submit_late.py --project ./stanford_rna3d --archive
```

## Created Project Structure

```
stanford_rna3d/
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── Makefile                   # Automated commands
├── .gitignore                # Files to ignore in Git
├── data/
│   ├── raw/                  # Original competition data
│   ├── processed/            # Processed data
│   └── external/             # External data
├── notebooks/
│   ├── 01_eda.ipynb         # Complete exploratory analysis
│   ├── 02_baseline.ipynb    # LSTM baseline model
│   ├── 03_advanced.ipynb    # Advanced models (Transformer, GNN)
│   └── 04_submission.ipynb  # Submission preparation
├── src/
│   ├── __init__.py
│   ├── data_processing.py    # Data processing
│   ├── models.py            # Model architectures
│   ├── training.py          # Training
│   └── evaluation.py        # Evaluation
├── configs/
│   └── model_config.yaml    # Configurations
├── checkpoints/             # Saved models
├── submissions/             # Submission files
└── tests/                   # Unit tests
```

## Created Notebooks Details

### 1. `01_eda.ipynb` - Detailed Exploratory Analysis
**Structured content:**
- Scientific libraries import (pandas, numpy, matplotlib, seaborn, plotly)
- Competition data loading and structure
- RNA sequences analysis (length, nucleotide composition, patterns)
- 3D coordinates exploration (spatial distributions, molecular geometry)
- Data quality verification (missing values, outliers, consistency)
- Interactive visualizations and key insights

### 2. `02_baseline.ipynb` - LSTM Baseline Model
**Complete implementation:**
- Custom dataset for RNA sequences and 3D coordinates
- Bidirectional LSTM model with nucleotide embeddings
- Training pipeline with validation and early stopping
- Specialized metrics (RMSD, GDT-TS score)
- Performance evaluation and analysis
- Solid foundation for comparison with advanced models

### 3. `03_advanced.ipynb` - Sophisticated Models
**Advanced architectures:**
- **Specialized Transformer** for RNA with positional encoding
- **Graph Neural Networks** for molecular spatial relationships
- **Ensemble methods** with optimized weights
- **Physics-Informed Neural Networks** with physical constraints
- **Hyperparameter optimization** with Optuna
- **Systematic comparison** of all architectures
- 3D visualization of predictions

### 4. `04_submission.ipynb` - Professional Preparation
**Submission workflow:**
- Loading of best trained model
- Consistent preprocessing of test data
- Prediction generation with physical validation
- Post-processing and quality verification
- Formatting in competition required format
- Complete submission metadata
- Final validation and documentation

## Portfolio Value

This project demonstrates:

### Advanced Technical Skills
- **Deep Learning**: PyTorch with LSTM, Transformer and GNN architectures
- **Bioinformatics**: Advanced processing of RNA sequences and 3D structures
- **MLOps**: Complete pipeline with versioning, reproducibility and automation
- **Data Science**: Detailed EDA, interactive visualizations, specialized metrics
- **Physics-Informed ML**: Models with physical constraints and domain knowledge

### Software Engineering Skills
- **Architecture**: Modular, object-oriented, well-documented code
- **Documentation**: Detailed READMEs, docstrings, submission metadata
- **Automation**: Setup scripts, progress bars, automatic validation
- **Quality**: Pre/post-processing, data validation, testing
- **Versioning**: Git workflow, checkpoints, reproducibility

### Scientific Domain Skills
- **Molecular Biology**: 3D RNA structures, protein folding
- **Computational Chemistry**: Molecular properties, energy constraints
- **Physics**: Force modeling, molecular geometry, dynamics
- **Mathematics**: Optimization, spatial metrics (RMSD, GDT-TS), linear algebra

### Specialized Machine Learning Skills
- **Sequence Modeling**: Transformers for biological data
- **Graph Neural Networks**: Molecular relationship modeling
- **Ensemble Methods**: Intelligent model combination
- **Hyperparameter Optimization**: Optuna for automatic tuning
- **Custom Loss Functions**: Physics-informed training

## Next Steps

1. **Execute complete workflow** following the steps above
2. **Develop models** in Jupyter notebooks
3. **Document insights** and thought process
4. **Create visualizations** of RNA structures
5. **Generate submission** even if late
6. **Add to GitHub** as portfolio project
7. **Include in resume** as ML/Bioinformatics project

## Useful Links

- **Competition**: https://www.kaggle.com/competitions/stanford-rna-3d-folding
- **RNA Documentation**: https://www.rcsb.org/
- **PyTorch**: https://pytorch.org/
- **BioPython**: https://biopython.org/

---

**Created to demonstrate expertise in Machine Learning applied to Computational Biology**
