# Stanford RNA 3D Folding

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
