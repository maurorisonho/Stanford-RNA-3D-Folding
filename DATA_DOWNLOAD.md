# Data Download Instructions

**Note**: Large data files are not included in this repository due to GitHub size limits.

## Download Competition Data

### Option 1: Kaggle API (Recommended)
```bash
# Install Kaggle API
pip install kaggle

# Configure API credentials (get from https://www.kaggle.com/account)
# Place kaggle.json in ~/.kaggle/

# Download data
cd stanford_rna3d/data/raw/
kaggle competitions download -c stanford-rna-3d-folding
unzip stanford-rna-3d-folding.zip
```

### Option 2: Manual Download
1. Go to https://www.kaggle.com/competitions/stanford-rna-3d-folding/data
2. Download all files manually
3. Place in `stanford_rna3d/data/raw/`

## Required Files
- `train_sequences.csv` (3.0MB)
- `train_labels.csv` (9.3MB)  
- `train_sequences.v2.csv` (54MB)
- `train_labels.v2.csv` (256MB) - Large file
- `validation_sequences.csv` (12KB)
- `validation_labels.csv` (2.4MB)
- `test_sequences.csv` (12KB)
- `sample_submission.csv` (188KB)

## File Sizes
**Total**: ~325MB (too large for GitHub)

## Alternative: Use Automated Script
```bash
# Use the project setup script
python 02_setup_project.py --dest ./stanford_rna3d --download-data
```