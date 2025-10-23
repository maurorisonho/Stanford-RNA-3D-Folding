# Project Scripts Reference

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



This document summarises the helper scripts shipped with the Stanford RNA 3D
Folding toolkit and how to use them.

## 01_create_env.py — Environment Bootstrap

Creates a dedicated Python 3.13 virtual environment and installs the project
dependencies.

```bash
# create a new environment
python3.13 01_create_env.py

# recreate the environment from scratch
python 01_create_env.py --recreate
```

Optional flags:
- `--venv PATH` – custom virtual environment location (default: `.venv`)
- `--python /path/to/python` – interpreter used to create the environment
- `--extra PACKAGE ...` – additional packages to install after requirements

## 02_setup_project.py — Project Scaffolding

Initialises the data folders, notebooks, and optionally downloads the Kaggle
competition files.

```bash
# create folders and attempt to fetch competition data
python 02_setup_project.py

# skip the Kaggle download
python 02_setup_project.py --skip-download
```

The script creates notebook stubs if they are missing. Pass `--force` to
overwrite existing notebooks with the default template.

## 03_submit_late.py — Kaggle Submission Helper

Uploads a prepared CSV file to the competition.

```bash
python 03_submit_late.py submissions/prediction.csv \
    --message "Validation tuned model"
```

Ensure the Kaggle CLI is installed inside the virtual environment and that the
API token (`~/.kaggle/kaggle.json`) is available before running the script.

## system_specs_checker.py — Hardware Profiler

Provides insight into the host machine. The module exposes a `SystemSpecsChecker`
class and example usage in `system_checker_examples.py`.

```bash
python system_specs_checker.py --output system_specs.json
python system_checker_examples.py
```
