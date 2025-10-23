# Environment Manager - Stanford RNA 3D Folding

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


## [RNA] Description

Complete Python script to manage the virtual environment and dependencies for the Stanford RNA 3D Folding project. This script automates all tasks related to setup and maintenance of the development environment.

## Key Capabilities

- **Complete Environment Management**: Create, recreate, and validate virtual environments
- **Dependency Verification**: Check all required libraries are properly installed
- **Cross-Platform Support**: Works on Windows, macOS, and Linux
- **Detailed Status Reports**: Comprehensive information about your environment
- **Import Testing**: Verify that critical libraries can be imported successfully
- **Automatic Requirements Management**: Handle requirements.txt files efficiently
- **User-Friendly Interface**: Clear commands and helpful error messages
- **Safety First**: Backup existing environments before making changes

## Prerequisites

## [INFO] Prerequisites

- Python 3.8+ (recommended: 3.13.5)
- Operating system: Linux, macOS or Windows
- Access to terminal/command prompt

## 🛠️ Usage

### Basic Commands

```bash
# Check complete environment status
python environment_manager.py check

# Create virtual environment
python environment_manager.py create

# Recreate virtual environment (force recreation)
python environment_manager.py recreate

# Install all dependencies
python environment_manager.py install

# List installed packages
python environment_manager.py list

# List packages with details
python environment_manager.py list --detailed

# Test critical imports
python environment_manager.py test

# Generate requirements freeze file
python environment_manager.py freeze

# Generate freeze to specific file
python environment_manager.py freeze --output my_requirements.txt

# Clean virtual environment
python environment_manager.py clean

# Show project information
python environment_manager.py info
```

### Complete Setup Workflow

```bash
# 1. Check initial status
python environment_manager.py check

# 2. Create virtual environment
python environment_manager.py create

# 3. Install dependencies
python environment_manager.py install

# 4. Test if everything works
python environment_manager.py test

# 5. Check final status
python environment_manager.py check
```

### To Recreate Environment from Scratch

```bash
# Clean existing environment
python environment_manager.py clean

# Create new environment
python environment_manager.py create

# Install dependencies
python environment_manager.py install

# Test functionality
python environment_manager.py test
```

## Output Examples

### `check` Command
```
================================================================================
STANFORD RNA 3D FOLDING - ENVIRONMENT STATUS
================================================================================
Python: 3.13.5 [OK]
System: Linux x86_64
Project: /home/user/Stanford-RNA-3D-Folding
Virtual Environment: [OK] Exists
   Location: /home/user/Stanford-RNA-3D-Folding/stanford_rna3d/.venv
   Python: /home/user/Stanford-RNA-3D-Folding/stanford_rna3d/.venv/bin/python
   Pip: /home/user/Stanford-RNA-3D-Folding/stanford_rna3d/.venv/bin/pip
Requirements.txt: [OK] Exists
   Location: /home/user/Stanford-RNA-3D-Folding/stanford_rna3d/requirements.txt
Installed Packages: 118 packages
Dependencies: [OK]
================================================================================
```

### `test` Command
```
Testing critical imports...
  [OK] pandas
  [OK] numpy
  [OK] matplotlib
  [OK] seaborn
  [OK] plotly
  [OK] sklearn
  [OK] scipy
  [OK] torch
  [OK] torchvision
  [OK] transformers
  [OK] pytorch_lightning
  [OK] torchmetrics
  [OK] optuna
  [OK] wandb
  [OK] Bio
  [OK] jupyter
  [OK] tqdm
  [OK] yaml
  [OK] requests

Result: 19/19 libraries OK
```

### `list` Command
```
[PACKAGE] INSTALLED PACKAGES (118 total)
------------------------------------------------------------
aiohappyeyeballs   aiohttp            aiosignal          alembic            
annotated-types    anyio              argon2-cffi        argon2-cffi-bindings
arrow              asttokens          async-lru          attrs              
babel              beautifulsoup4     biopython          bleach             
...
------------------------------------------------------------
```

## Critical Libraries Tested

The script automatically tests the import of the following essential libraries:

- **Data Analysis**: pandas, numpy, matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn (sklearn), scipy
- **Deep Learning**: torch, torchvision, transformers, pytorch_lightning, torchmetrics
- **Optimization**: optuna, wandb
- **Bioinformatics**: biopython (Bio)
- **Development**: jupyter, tqdm, yaml, requests

## File Structure

```
Stanford-RNA-3D-Folding/
├── environment_manager.py          # Main script
├── ENVIRONMENT_MANAGER_README.md   # This file
├── stanford_rna3d/
│   ├── .venv/                     # Virtual environment (created automatically)
│   ├── requirements.txt           # Project dependencies
│   └── ...                        # Other project files
└── requirements_freeze.txt         # Generated freeze file (optional)
```

## Troubleshooting

### Error: "Virtual environment does not exist"
```bash
python environment_manager.py create
```

### Error: "Requirements.txt not found"
Make sure the `stanford_rna3d/requirements.txt` file exists.

### Error: "Broken dependencies"
```bash
python environment_manager.py recreate
python environment_manager.py install
```

### Error: "Command not found"
Make sure Python 3.8+ is installed and in PATH.

### Imports failing
```bash
# Check which libraries are failing
python environment_manager.py test

# Reinstall dependencies
python environment_manager.py install
```

## Performance Tips

- Use `recreate` only when necessary (it's slower)
- Use `check` for quick diagnosis
- Use `test` to validate installations
- Use `freeze` for dependency backup

## Logs and Debugging

The script shows detailed information during execution:
- [OK] Successful operations
- [ERROR] Errors and failures
- [WARNING] Important warnings
- [EXEC] Commands being executed
- [TIME] Execution time

## Contributing

To improve this script:
1. Fork the repository
2. Create a branch for your development
3. Implement improvements
4. Test with different scenarios
5. Open a Pull Request

## License

This script is part of the Stanford RNA 3D Folding project and follows the same license as the main project.

---

**Author**: Auto-generated for Stanford RNA 3D Folding project  
**Date**: October 22, 2025  
**Version**: 1.0.0