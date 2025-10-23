# Data Download Instructions

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
