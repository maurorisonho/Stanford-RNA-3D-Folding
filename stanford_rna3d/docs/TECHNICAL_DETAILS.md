# Technical Details – Stanford RNA 3D Folding

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



## Data Pipeline

1. **Ingestion:** Raw CSV files (`train.csv`, `test.csv`, `sample_submission.csv`)
   are loaded via `pandas`. Metadata is cached under `data/interim/`.
2. **Pre-processing:**
   - Encode nucleotides (`A, U, G, C`) to integers.
   - Generate k-mer statistics and predicted secondary structure features.
   - Normalise coordinate inputs by centring at the origin.
3. **Data loaders:** Implemented in `stanford_rna3d/src/data_processing.py` with
   PyTorch `Dataset` and `DataLoader` wrappers for both training and evaluation.

## Model Components

### LSTM Baseline

- Embedding size: 128  
- Hidden size: 256  
- Layers: 2 (bidirectional)  
- Output: fully connected projection to 3D coordinates with padding mask support.

### Transformer Encoder

- Depth: 6 encoder blocks  
- Heads: 8 attention heads, key/query dimension 64  
- Feed-forward dimension: 512  
- Positional encoding: sinusoidal  
- Output head: linear layer predicting xyz offsets which accumulate to absolute
  positions.

### Physics-informed Loss

- Bond length penalty anchored at 1.5 Å between consecutive atoms.
- Angle penalty computed from triplets of coordinates to discourage impossible
  bends.
- Total loss: `L_total = L_mse + λ_bond * L_bond + λ_angle * L_angle`
  with default coefficients `λ_bond = 0.1`, `λ_angle = 0.05`.

## Optimisation Settings

- Optimiser: AdamW (β1=0.9, β2=0.95, weight decay=1e-4)
- Learning rate: 3e-4 with cosine decay and warmup over first 2 epochs
- Batch size: 32 sequences
- Gradient clipping at 1.0
- Early stopping monitored on validation RMSD with patience of 10 epochs

## Infrastructure

- Python 3.13.5
- PyTorch 2.1 (CPU builds included by default)
- Optional GPU acceleration via CUDA 12 (install `torch` with CUDA support)
- Logging and experiment tracking via Weights & Biases (optional)

## File Structure

| Path | Description |
| ---- | ----------- |
| `stanford_rna3d/src/data_processing.py` | Data loading utilities |
| `stanford_rna3d/src/models.py` | Model definitions and baselines |
| `stanford_rna3d/notebooks/` | Jupyter notebooks for EDA, baselines, advanced experiments |
| `stanford_rna3d/checkpoints/` | Directory for model artefacts |
| `stanford_rna3d/submissions/` | Generated Kaggle submission files |

## Reproducibility Notes

- Seed values set via `torch.manual_seed`, `numpy.random.seed`, and Python's
  built-in `random.seed`.
- Deterministic algorithms toggled with `torch.use_deterministic_algorithms(True)`
  when performance impact is acceptable.
- Model checkpoints stored alongside a JSON metadata file containing hyper-
  parameters, dataset hash, and code commit reference.
