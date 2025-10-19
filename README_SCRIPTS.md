# Project Scripts Reference

**Author**: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>  
**License**: MIT License

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
