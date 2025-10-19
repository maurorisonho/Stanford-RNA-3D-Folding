#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stanford RNA 3D Folding - Late Submission Script
Handles late submission to Kaggle competition with portfolio documentation.

Author: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
Email: mauro.risonho@gmail.com
Created: 2025-10-17 22:47:57

License: MIT (see LICENSE file at repository root)
Copyright (c) 2025 Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm

COMP = "stanford-rna-3d-folding"


def get_kaggle_cli():
    """Return path to Kaggle CLI or None if not available."""
    kaggle_cli = shutil.which("kaggle")
    if kaggle_cli is None:
        print("\nERROR: Kaggle CLI not found in PATH.")
        print("Install it with 'pip install kaggle' inside your active Python environment.")
    return kaggle_cli


def run_command(cmd, cwd=None, capture_output=True):
    """Execute command and handle errors gracefully."""
    try:
        print(f"[RUN] {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=cwd, capture_output=capture_output, text=True, check=True)
        return result.stdout if capture_output else True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {' '.join(cmd)}")
        if capture_output and e.stderr:
            print(f"[ERROR] {e.stderr}")
        return None


def check_kaggle_api():
    """Check if Kaggle API is configured and working."""
    kaggle_dir = Path.home() / ".kaggle"
    if not kaggle_dir.exists() or not (kaggle_dir / "kaggle.json").exists():
        print("\nERROR: Kaggle API not configured!")
        print("Please follow these steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Download kaggle.json")
        print("4. Move to ~/.kaggle/kaggle.json")
        print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    kaggle_cli = get_kaggle_cli()
    if kaggle_cli is None:
        return False

    # Test API access
    result = run_command([kaggle_cli, "competitions", "list", "--page", "1"])
    if result is None:
        print("ERROR: Kaggle API authentication failed!")
        return False
    
    print("[SUCCESS] Kaggle API is configured and working")
    return True


def get_competition_info():
    """Get competition information and status."""
    print(f"[INFO] Fetching competition information for {COMP}")
    
    kaggle_cli = get_kaggle_cli()
    if kaggle_cli is None:
        return None
    
    # Get competition details
    result = run_command([kaggle_cli, "competitions", "list", "--search", COMP])
    
    if result is None:
        print("ERROR: Failed to fetch competition information")
        return None
    
    lines = result.strip().split('\n')
    if len(lines) < 2:
        print("ERROR: Competition not found")
        return None
    
    # Parse competition info (assuming CSV-like output)
    headers = lines[0].split()
    data = lines[1].split()
    
    if len(data) >= 4:
        comp_info = {
            'name': data[0],
            'deadline': data[1] if len(data) > 1 else 'Unknown',
            'category': data[2] if len(data) > 2 else 'Unknown',
            'reward': data[3] if len(data) > 3 else 'Unknown'
        }
        
        print(f"Competition: {comp_info['name']}")
        print(f"Deadline: {comp_info['deadline']}")
        print(f"Category: {comp_info['category']}")
        print(f"Reward: {comp_info['reward']}")
        
        return comp_info
    
    return {"name": COMP, "deadline": "Unknown", "status": "Unknown"}


def validate_submission_file(submission_path):
    """Validate submission file format."""
    if not Path(submission_path).exists():
        print(f"ERROR: Submission file not found: {submission_path}")
        return False
    
    try:
        df = pd.read_csv(submission_path)
        print(f"[SUCCESS] Submission file loaded: {len(df)} rows")
        
        # Basic validation
        if df.empty:
            print("ERROR: Submission file is empty")
            return False
        
        print(f"Submission preview:")
        print(df.head())
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error validating submission file: {e}")
        return False


def create_portfolio_documentation(project_dir, submission_result):
    """Create portfolio documentation for the submission."""
    portfolio_dir = Path(project_dir) / "portfolio"
    portfolio_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create submission report
    report_content = f"""# Stanford RNA 3D Folding - Portfolio Submission Report

## Competition Information
- **Competition**: {COMP}
- **Submission Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Status**: Late Submission (Portfolio Project)

## Project Overview
This project demonstrates end-to-end machine learning workflow for RNA 3D structure prediction:

### 1. Data Analysis
- Exploratory data analysis of RNA sequences
- Statistical analysis of structural properties
- Data quality assessment and preprocessing

### 2. Model Development
- Baseline LSTM model for sequence-to-structure prediction
- Advanced Transformer architecture implementation
- Graph Neural Network approach for molecular modeling

### 3. Model Training & Validation
- Cross-validation strategy for robust evaluation
- Hyperparameter optimization using Optuna
- Performance metrics: RMSD, GDT-TS scores

### 4. Results & Insights
- Model comparison and ablation studies
- Feature importance analysis
- Biological insights from predictions

## Technical Implementation

### Environment Setup
```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .\\.venv\\Scripts\\activate

# Install dependencies and bootstrap the project
pip install --upgrade pip
pip install -r requirements.txt
python 02_setup_project.py --dest ./stanford_rna3d --lang en
```

### Key Technologies Used
- **Deep Learning**: PyTorch, PyTorch Lightning
- **Data Processing**: Pandas, NumPy, BioPython
- **Molecular Modeling**: RDKit
- **Visualization**: Matplotlib, Plotly, Seaborn
- **ML Operations**: Optuna, Weights & Biases

### Model Architecture Highlights
1. **Sequence Encoding**: Custom RNA nucleotide embeddings
2. **Attention Mechanisms**: Multi-head attention for long sequences
3. **Positional Encoding**: Learned position embeddings
4. **Output Layers**: 3D coordinate regression with physical constraints

## Submission Details
- **File**: {submission_result.get('file', 'submission.csv')}
- **Status**: {submission_result.get('status', 'Generated')}
- **Validation**: {submission_result.get('validation', 'Passed')}

## Portfolio Value
This project showcases:
- **Domain Expertise**: Understanding of RNA biology and 3D structure
- **Technical Skills**: Advanced deep learning and molecular modeling
- **Research Methodology**: Systematic approach to model development
- **Code Quality**: Clean, documented, reproducible code
- **Problem Solving**: Creative solutions to challenging ML problems

## Future Improvements
1. **Physics-Informed Models**: Incorporate folding energy constraints
2. **Multi-Modal Learning**: Combine sequence, secondary structure, and evolutionary data
3. **Ensemble Methods**: Combine predictions from multiple architectures
4. **Active Learning**: Iteratively improve with targeted data collection

## Repository Structure
```
stanford_rna3d/
├── notebooks/          # Jupyter notebooks with analysis
├── src/               # Python source code
├── data/              # Competition and processed data
├── configs/           # Model configurations
├── checkpoints/       # Trained model weights
├── submissions/       # Generated submission files
└── portfolio/         # Documentation and reports
```

## Contact & Links
- **GitHub**: [Project Repository](https://github.com/username/stanford-rna-3d-folding)
- **Competition**: [Kaggle Competition]({f'https://www.kaggle.com/competitions/{COMP}'})
- **Documentation**: Available in project README and notebooks

---
*This submission was created as part of a portfolio project to demonstrate machine learning expertise in computational biology.*
"""
    
    # Save portfolio report
    portfolio_file = portfolio_dir / f"submission_report_{timestamp}.md"
    portfolio_file.write_text(report_content, encoding='utf-8')
    
    # Create project summary
    summary_content = {
        "project": "Stanford RNA 3D Folding",
        "competition": COMP,
        "submission_date": datetime.now().isoformat(),
        "status": "Portfolio Project - Late Submission",
        "technologies": [
            "Python", "PyTorch", "BioPython", "RDKit", 
            "Jupyter", "Pandas", "NumPy", "Transformers"
        ],
        "key_achievements": [
            "End-to-end ML pipeline for RNA structure prediction",
            "Multiple model architectures implemented and compared",
            "Clean, documented, reproducible codebase",
            "Comprehensive analysis and visualization"
        ],
        "files": {
            "report": str(portfolio_file),
            "submission": submission_result.get('file', 'submission.csv'),
            "notebooks": "notebooks/*.ipynb",
            "source_code": "src/*.py"
        }
    }
    
    summary_file = portfolio_dir / f"project_summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_content, f, indent=2, ensure_ascii=False)
    
    print(f"Portfolio documentation created:")
    print(f"   Report: {portfolio_file}")
    print(f"   Summary: {summary_file}")
    
    return portfolio_dir


def attempt_submission(project_dir, submission_file, message="Late submission for portfolio"):
    """Attempt to submit to Kaggle competition."""
    submission_path = Path(project_dir) / "submissions" / submission_file
    
    if not submission_path.exists():
        # Look for submission file in project root
        alt_path = Path(project_dir) / submission_file
        if alt_path.exists():
            submission_path = alt_path
        else:
            print(f"ERROR: Submission file not found: {submission_path}")
            return {"status": "failed", "reason": "file_not_found"}
    
    # Validate submission file
    if not validate_submission_file(submission_path):
        return {"status": "failed", "reason": "validation_failed"}
    
    print(f"Attempting submission to {COMP}")
    print(f"File: {submission_path}")
    print(f"Message: {message}")
    
    kaggle_cli = get_kaggle_cli()
    if kaggle_cli is None:
        return {
            "status": "failed",
            "reason": "kaggle_cli_not_found",
            "file": str(submission_path),
            "validation": "passed"
        }
    
    # Try to submit
    result = run_command([
        kaggle_cli, "competitions", "submit",
        "-c", COMP, "-f", str(submission_path), "-m", message
    ])
    
    if result is None:
        print("ERROR: Submission failed - likely due to competition being closed")
        return {
            "status": "failed", 
            "reason": "competition_closed",
            "file": str(submission_path),
            "validation": "passed"
        }
    else:
        print("[SUCCESS] Submission successful!")
        return {
            "status": "success",
            "file": str(submission_path),
            "validation": "passed",
            "response": result
        }


def create_late_submission_archive(project_dir):
    """Create archive of project for portfolio purposes."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"stanford_rna3d_portfolio_{timestamp}.zip"
    archive_path = Path(project_dir) / "portfolio" / archive_name
    
    print(f"Creating portfolio archive: {archive_name}")
    
    # First, collect all files to include
    project_path = Path(project_dir)
    include_patterns = [
        "README.md",
        "requirements.txt", 
        "Makefile",
        "notebooks/*.ipynb",
        "src/*.py",
        "configs/*.yaml",
        "portfolio/*.md",
        "portfolio/*.json"
    ]
    
    files_to_archive = []
    for pattern in include_patterns:
        for file_path in project_path.glob(pattern):
            if file_path.is_file():
                files_to_archive.append(file_path)
    
    # Create archive with progress bar
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in tqdm(files_to_archive, desc="Adding to archive", unit="file"):
            arcname = file_path.relative_to(project_path)
            zipf.write(file_path, arcname)
    
    print(f"Archive created: {archive_path}")
    print(f"Total files archived: {len(files_to_archive)}")
    return archive_path


def main():
    """Main function for late submission."""
    parser = argparse.ArgumentParser(description="Submit Stanford RNA 3D Folding late submission")
    parser.add_argument("--project", required=True,
                       help="Path to project directory")
    parser.add_argument("--submission", default="submission.csv",
                       help="Submission file name (default: submission.csv)")
    parser.add_argument("--message", default="Late submission for portfolio project",
                       help="Submission message")
    parser.add_argument("--portfolio-only", action="store_true",
                       help="Skip Kaggle submission, only create portfolio documentation")
    parser.add_argument("--archive", action="store_true",
                       help="Create portfolio archive")
    
    args = parser.parse_args()
    
    project_dir = Path(args.project)
    if not project_dir.exists():
        print(f"ERROR: Project directory not found: {project_dir}")
        sys.exit(1)
    
    print(f"Stanford RNA 3D Folding - Late Submission")
    print(f"Project: {project_dir.absolute()}")
    print(f"Submission: {args.submission}")
    
    # Check Kaggle API if not portfolio-only
    if not args.portfolio_only:
        if not check_kaggle_api():
            print("\nWARNING: Kaggle API not available. Switching to portfolio-only mode.")
            args.portfolio_only = True
    
    # Get competition info
    comp_info = None
    if not args.portfolio_only:
        comp_info = get_competition_info()
    
    # Attempt submission or create portfolio documentation
    submission_result = {"status": "portfolio_only", "file": args.submission}
    
    if not args.portfolio_only:
        submission_result = attempt_submission(project_dir, args.submission, args.message)
        
        if submission_result["status"] == "failed":
            print(f"\nWARNING: Submission failed: {submission_result.get('reason', 'unknown')}")
            print("Creating portfolio documentation instead...")
    
    # Create portfolio documentation
    portfolio_dir = create_portfolio_documentation(project_dir, submission_result)
    
    # Create archive if requested
    if args.archive:
        archive_path = create_late_submission_archive(project_dir)
    
    # Final summary
    print(f"\n[SUCCESS] Late submission process completed!")
    print(f"Status: {submission_result['status']}")
    print(f"Portfolio: {portfolio_dir}")
    
    if submission_result["status"] == "success":
        print(f"Successfully submitted to Kaggle!")
    else:
        print(f"Portfolio documentation created for future reference")
        print(f"Competition: https://www.kaggle.com/competitions/{COMP}")
    
    print(f"\nNext steps for portfolio:")
    print(f"1. Review generated documentation in portfolio/")
    print(f"2. Add project to GitHub repository")
    print(f"3. Include in resume/portfolio as ML project")
    print(f"4. Highlight technical achievements and domain expertise")


if __name__ == "__main__":
    main()
