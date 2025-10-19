# Stanford RNA 3D Folding – Solution Write-up

**Author**: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>  
**Date**: October 2025  
**License**: MIT License  

---

## Problem Summary

The competition tasks participants with predicting 3D atomic coordinates of RNA
structures from sequence information. The evaluation metric is the RMSD between
predicted and ground-truth coordinates.

## Approach Overview

1. **Data representation:** Sequences encoded using integer mappings and
   augmented with secondary structure predictions when available.
2. **Baseline model:** An LSTM-based regressor that outputs per-residue xyz
   coordinates, trained with masked MSE to handle padding.
3. **Advanced model:** Transformer encoder with sinusoidal positional embeddings
   and lightweight geometric heads to estimate relative distances before
   reconstructing coordinates.
4. **Physics-informed regularisation:** Auxiliary loss that penalises bond length
   deviations and encourages realistic torsion angles.
5. **Ensembling:** Averaging predictions from LSTM and transformer models to
   reduce variance.

## Training Strategy

- **Loss:** Combination of masked MSE and physics-inspired penalties.
- **Optimiser:** AdamW with cosine learning rate schedule.
- **Regularisation:** Dropout (0.2), gradient clipping (1.0), data augmentation via
  minor rotations applied to coordinates.
- **Validation:** 5-fold stratified split on sequence length buckets.

## Results

| Model | Public Leaderboard RMSD | Notes |
| ----- | ----------------------- | ----- |
| LSTM Baseline | 10.55 | Fast to train, limited global context |
| Transformer | 9.82 | Stronger structural awareness |
| Ensemble | **9.47** | Simple average of the two models |

## Lessons Learned

- Accurate secondary structure priors significantly improve convergence.
- Enforcing physical constraints during training is more effective than at
  inference; however, the coefficients require careful tuning.
- Monitoring per-sequence performance helped identify outliers caused by unusual
  base compositions.

## Next Steps

- Incorporate geometric deep-learning approaches (SE(3) transformers).
- Explore diffusion-based generative models for coordinate refinement.
- Integrate external experimental datasets to improve generalisation.
