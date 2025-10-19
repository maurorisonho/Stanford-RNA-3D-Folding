# Data Download Instructions

**Author**: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>  
**License**: MIT License

The Stanford RNA 3D Folding competition is hosted on Kaggle. Follow the steps
below to obtain the datasets required by the notebooks.

## Option A — Kaggle CLI (recommended)

1. Install the Kaggle command line interface inside the project virtual
   environment:
   ```bash
   pip install kaggle
   ```
2. Generate an API token by visiting
   https://www.kaggle.com/settings/account and selecting **Create New API Token**.
   The downloaded `kaggle.json` file must be placed in `~/.kaggle/kaggle.json`
   with permissions `600`.
3. Run the project setup script which will fetch and extract the files:
   ```bash
   python 02_setup_project.py
   ```
   The raw data will be stored under `stanford_rna3d/data/raw/`.

## Option B — Manual Download

1. Visit the competition page:
   https://www.kaggle.com/competitions/stanford-rna-3d-folding
2. Download the desired archives (`train.csv`, `test.csv`, `sample_submission.csv`,
   and any metadata files).
3. Extract the archives and copy the contents to
   `stanford_rna3d/data/raw/`. Keep the default filenames expected by the
   notebooks.

## Additional Resources

- `stanford_rna3d/ENVIRONMENT_SETUP.md` documents the verified Python packages.
- `stanford_rna3d/EXECUTION_PIPELINE.md` explains the end-to-end workflow.

For troubleshooting Kaggle CLI authentication, consult the official guide:
https://github.com/Kaggle/kaggle-api#api-credentials.
