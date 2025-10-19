# Stanford RNA 3D Folding – Project Package

**Author**: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>  
**License**: MIT License  

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
