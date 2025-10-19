# Stanford RNA 3D Folding

**Author**: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>  
**License**: MIT License  

Utilities, notebooks, and helper scripts for the Kaggle competition
**Stanford RNA 3D Folding**. The repository provides a lightweight baseline
implementation together with environment management helpers and extensive
documentation to streamline experimentation.

## Repository Layout

- `stanford_rna3d/` – primary project package, notebooks, and supporting docs  
- `01_create_env.py` – bootstrap a Python 3.13 virtual environment  
- `02_setup_project.py` – create folders, notebook stubs, and optionally download data  
- `03_submit_late.py` – submit predictions to Kaggle  
- `system_specs_checker.py` – inspect hardware capability for ML workloads

## Quick Start

1. Create the virtual environment:
   ```bash
   python3.13 01_create_env.py
   source .venv/bin/activate
   ```
2. Initialise the project structure and (optionally) download data:
   ```bash
   python 02_setup_project.py
   ```
3. Launch Jupyter Lab and open the notebooks in `stanford_rna3d/notebooks/`.
4. Generate submissions in `stanford_rna3d/submissions/` and upload with
   `python 03_submit_late.py submissions/my_predictions.csv -m "First attempt"`.

## Requirements

All Python dependencies are listed in `stanford_rna3d/requirements.txt`. The
project expects Python **3.13.5** and was tested on Linux with the libraries
specified in `stanford_rna3d/ENVIRONMENT_SETUP.md`.

## License

The contents of this repository are released under the MIT License. See
`LICENSE` for details.
