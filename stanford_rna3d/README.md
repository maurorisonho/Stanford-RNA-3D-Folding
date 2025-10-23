# Stanford RNA 3D Folding – Project Package

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



The `stanford_rna3d` package contains source code, notebooks, and supporting
documentation used throughout the Kaggle competition project.

## Structure

| Path | Description |
| ---- | ----------- |
| `configs/` | Optional configuration files for experiments |
| `data/` | Raw, interim, processed, and external datasets |
| `notebooks/` | Jupyter notebooks for EDA, baselines, and advanced modelling |
| `src/` | Python modules (`data_processing.py`, `models.py`, etc.) |
| `checkpoints/` | Saved model weights and artefacts |
| `submissions/` | Kaggle submission files |
| `tests/` | Placeholder for automated test suites |

## Getting Started

1. Ensure the repository root virtual environment has been created:
   ```bash
   python3.13 01_create_env.py
   source .venv/bin/activate
   ```
2. Run the setup script to populate the directory structure and optional data:
   ```bash
   python 02_setup_project.py
   ```
3. Launch `jupyter lab` and open the notebooks inside `notebooks/`.

## Development Notes

- Source code under `src/` is designed to be importable as a package:
  ```python
  from stanford_rna3d.src.data_processing import RNADataProcessor
  ```
- Use `make` targets (`make data`, `make train`, `make predict`) for common
  operations once the missing pieces in `src/` are implemented.
- Keep documentation such as `SOLUTION_WRITEUP.md`, `TECHNICAL_DETAILS.md`, and
  `EXECUTION_PIPELINE.md` updated as the project evolves.

## License

Distributed under the MIT License. See the repository root `LICENSE` file.
