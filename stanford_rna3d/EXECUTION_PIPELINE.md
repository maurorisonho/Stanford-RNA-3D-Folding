# Execution Pipeline Configuration - Stanford RNA 3D Folding

**Author**: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>  
**Email**: mauro.risonho@gmail.com  
**Created**: October 18, 2025 at 14:30:00  
**License**: MIT License  
**Competition**: [Stanford RNA 3D Folding](https://www.kaggle.com/competitions/stanford-rna-3d-folding)

---

## Complete Execution Pipeline

This document provides step-by-step instructions for executing the complete Stanford RNA 3D Folding solution pipeline.

### Prerequisites

**System Requirements:**
- Python 3.13.5 (latest stable version)
- CUDA-compatible GPU (recommended: 16GB+ VRAM)
- 32GB+ RAM (minimum 16GB)
- 50GB+ available storage space

**Environment Setup:**
```bash
# Navigate to project directory
cd stanford_rna3d/

# Create virtual environment
python3.13 -m venv .venv

# Activate environment (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Data Preparation

**Download Competition Data:**
```bash
# Create data directories
mkdir -p data/raw data/processed data/external

# Download from Kaggle (requires Kaggle API setup)
kaggle competitions download -c stanford-rna-3d-folding -p data/raw/
cd data/raw && unzip stanford-rna-3d-folding.zip && cd ../..
```

**Verify Data Structure:**
```bash
# Check data files
ls -la data/raw/
# Expected files: train.csv, test.csv, sample_submission.csv
```

### Execution Sequence

#### Phase 1: Exploratory Data Analysis

**Execute EDA Notebook:**
```bash
# Launch Jupyter Lab
jupyter lab

# Or execute notebook directly
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --inplace
```

**EDA Outputs:**
- Data quality assessment
- Sequence length distributions
- Coordinate statistics
- Visualization of structural patterns

#### Phase 2: Baseline Model Implementation

**Execute Baseline Notebook:**
```bash
jupyter nbconvert --to notebook --execute notebooks/02_baseline.ipynb --inplace
```

**Baseline Model Training:**
```python
# Alternative: Direct Python execution
python -c "
from src.models import SimpleRNAPredictor
from src.data_processing import RNADataProcessor

# Initialize components
processor = RNADataProcessor()
model = SimpleRNAPredictor()

# Load and process data
train_data = processor.load_data('data/raw/train.csv')
processed_data = processor.preprocess(train_data)

# Train model
model.fit(processed_data, epochs=50)
model.save('checkpoints/baseline_model.pt')
"
```

#### Phase 3: Advanced Model Development

**Execute Advanced Models Notebook:**
```bash
jupyter nbconvert --to notebook --execute notebooks/03_advanced.ipynb --inplace
```

**Advanced Training Configuration:**
```python
# Hyperparameter optimization
python -c "
import optuna
from src.models import RNATransformer, PhysicsInformedRNA

def objective(trial):
    # Define hyperparameter space
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical('hidden_dim', [256, 512, 1024])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    # Train model with suggested hyperparameters
    model = RNATransformer(hidden_dim=hidden_dim, dropout=dropout)
    val_loss = model.train_and_validate(lr=lr)
    
    return val_loss

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
print(f'Best parameters: {study.best_params}')
"
```

#### Phase 4: Ensemble Model Training

**Train Complete Ensemble:**
```python
# Ensemble training script
python -c "
from src.models import RNAEnsemble
from src.data_processing import RNADataProcessor

# Initialize ensemble
ensemble = RNAEnsemble([
    'checkpoints/baseline_model.pt',
    'checkpoints/transformer_model.pt',
    'checkpoints/physics_model.pt'
])

# Load test data
processor = RNADataProcessor()
test_data = processor.load_data('data/raw/test.csv')

# Generate ensemble predictions
predictions = ensemble.predict(test_data)

# Save submission
processor.save_submission(predictions, 'submissions/ensemble_submission.csv')
"
```

#### Phase 5: Final Submission Preparation

**Execute Submission Notebook:**
```bash
jupyter nbconvert --to notebook --execute notebooks/04_submission.ipynb --inplace
```

**Generate Final Submission:**
```bash
# Validate submission format
python -c "
import pandas as pd

# Load submission
submission = pd.read_csv('submissions/ensemble_submission.csv')

# Validate format
required_columns = ['id', 'x', 'y', 'z']
assert all(col in submission.columns for col in required_columns)
assert len(submission) > 0
assert not submission.isnull().any().any()

print('Submission validation passed!')
print(f'Submission shape: {submission.shape}')
"
```

### Performance Monitoring

**Track Training Progress:**
```bash
# Launch TensorBoard (if logging enabled)
tensorboard --logdir logs/

# Or use Weights & Biases
wandb login
# Then execute training with wandb.init() in code
```

**Evaluate Model Performance:**
```python
# Model evaluation script
python -c "
from src.models import RNAEnsemble
from src.data_processing import RNADataProcessor
import numpy as np

# Load trained ensemble
ensemble = RNAEnsemble.load('checkpoints/ensemble_model.pt')

# Load validation data
processor = RNADataProcessor()
val_data = processor.load_validation_data()

# Generate predictions
predictions = ensemble.predict(val_data['sequences'])
true_coords = val_data['coordinates']

# Calculate metrics
rmsd = np.sqrt(np.mean((predictions - true_coords) ** 2))
mae = np.mean(np.abs(predictions - true_coords))

print(f'Validation RMSD: {rmsd:.3f} Å')
print(f'Validation MAE: {mae:.3f} Å')
"
```

### Debugging and Troubleshooting

**Common Issues and Solutions:**

1. **CUDA Out of Memory:**
```bash
# Reduce batch size
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Or use gradient checkpointing
python -c "
model.gradient_checkpointing_enable()
"
```

2. **Slow Training:**
```bash
# Enable mixed precision training
python -c "
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()
# Use with autocast() context in training loop
"
```

3. **Memory Issues:**
```bash
# Monitor memory usage
python -c "
import psutil
import GPUtil

print(f'CPU Memory: {psutil.virtual_memory().percent}%')
gpus = GPUtil.getGPUs()
for gpu in gpus:
    print(f'GPU {gpu.id}: {gpu.memoryUtil*100:.1f}%')
"
```

### Validation and Testing

**Cross-Validation Setup:**
```python
# 5-fold cross-validation
python -c "
from sklearn.model_selection import KFold
from src.models import RNATransformer
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
    print(f'Training fold {fold+1}/5')
    
    model = RNATransformer()
    train_data = data[train_idx]
    val_data = data[val_idx]
    
    model.fit(train_data)
    val_score = model.evaluate(val_data)
    cv_scores.append(val_score)
    
    print(f'Fold {fold+1} RMSD: {val_score:.3f}')

print(f'Mean CV RMSD: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}')
"
```

### Deployment and Submission

**Final Model Export:**
```bash
# Export for deployment
python -c "
from src.models import RNAEnsemble
import torch

# Load best ensemble
ensemble = RNAEnsemble.load('checkpoints/best_ensemble.pt')

# Export to ONNX for deployment
dummy_input = torch.randint(0, 4, (1, 100))
torch.onnx.export(ensemble, dummy_input, 'models/rna_predictor.onnx')

print('Model exported successfully')
"
```

**Kaggle Submission:**
```bash
# Submit to competition
kaggle competitions submit -c stanford-rna-3d-folding -f submissions/ensemble_submission.csv -m "Ensemble model with physics constraints"
```

### Performance Benchmarks

**Expected Execution Times:**
- **EDA Notebook**: 5-10 minutes
- **Baseline Training**: 2-3 hours (GPU), 8-12 hours (CPU)
- **Advanced Training**: 8-12 hours (GPU), 24-48 hours (CPU)
- **Ensemble Training**: 15-20 hours (GPU), 48-72 hours (CPU)
- **Hyperparameter Optimization**: 24-48 hours (depending on trials)

**Resource Requirements by Phase:**
| Phase | GPU Memory | RAM | Storage |
|-------|------------|-----|---------|
| EDA | 0GB | 8GB | 5GB |
| Baseline | 4-8GB | 16GB | 10GB |
| Advanced | 12-16GB | 24GB | 25GB |
| Ensemble | 16-24GB | 32GB | 40GB |

---

## Automated Execution Script

For convenience, here's a complete automation script:

```bash
#!/bin/bash
# complete_pipeline.sh - Full execution pipeline

set -e  # Exit on any error

echo "Starting Stanford RNA 3D Folding Pipeline"
echo "========================================"

# Environment setup
echo "Setting up environment..."
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Data preparation
echo "Preparing data..."
mkdir -p data/raw data/processed data/external checkpoints submissions logs
kaggle competitions download -c stanford-rna-3d-folding -p data/raw/
cd data/raw && unzip -o stanford-rna-3d-folding.zip && cd ../..

# Execute pipeline
echo "Executing EDA..."
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --inplace

echo "Training baseline model..."
jupyter nbconvert --to notebook --execute notebooks/02_baseline.ipynb --inplace

echo "Training advanced models..."
jupyter nbconvert --to notebook --execute notebooks/03_advanced.ipynb --inplace

echo "Preparing final submission..."
jupyter nbconvert --to notebook --execute notebooks/04_submission.ipynb --inplace

echo "Pipeline execution completed successfully!"
echo "Check submissions/ directory for final results."
```

**Execute with:**
```bash
chmod +x complete_pipeline.sh
./complete_pipeline.sh
```

This comprehensive pipeline ensures reproducible execution of the complete Stanford RNA 3D Folding solution.