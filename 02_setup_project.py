#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stanford RNA 3D Folding - Project Setup Script
Downloads competition data, creates project structure, and sets up notebooks.

Author: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
Email: mauro.risonho@gmail.com
Created: 2025-10-17 22:47:57

License: MIT (see LICENSE file at repository root)
Copyright (c) 2025 Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from tqdm import tqdm

COMP = "stanford-rna-3d-folding"


def run_command(cmd, cwd=None):
    """Execute command and handle errors gracefully."""
    try:
        print(f"[RUN] {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {' '.join(cmd)}")
        print(f"[ERROR] {e.stderr}")
        return None


def check_kaggle_api():
    """Check if Kaggle API is configured."""
    kaggle_dir = Path.home() / ".kaggle"
    if not kaggle_dir.exists() or not (kaggle_dir / "kaggle.json").exists():
        print("\n[WARNING] Kaggle API not configured!")
        print("Please follow these steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Download kaggle.json")
        print("4. Move to ~/.kaggle/kaggle.json")
        print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    return True


def download_competition_data(project_dir):
    """Download competition data using Kaggle API."""
    if not check_kaggle_api():
        print("[INFO] Skipping data download. You can download manually later.")
        return False

    kaggle_cli = shutil.which("kaggle")
    if kaggle_cli is None:
        print("\n[WARNING] Kaggle CLI não encontrado no PATH.")
        print("Instale com 'pip install kaggle' dentro do seu ambiente virtual.")
        print("Pulando o download automático dos dados.")
        return False
    
    data_dir = project_dir / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Downloading competition data to {data_dir}")
    
    # Download competition files with progress indication
    cmd = [kaggle_cli, "competitions", "download", "-c", COMP, "-p", str(data_dir)]
    
    print("[INFO] Downloading files...")
    with tqdm(total=1, desc="Kaggle Download", unit="file") as pbar:
        result = run_command(cmd)
        pbar.update(1)
    
    if result is None:
        print("[ERROR] Failed to download competition data")
        return False
    
    # Extract zip files with progress
    zip_files = list(data_dir.glob("*.zip"))
    if zip_files:
        print(f"[INFO] Extracting {len(zip_files)} archive(s)...")
        
        for zip_file in tqdm(zip_files, desc="Extracting", unit="archive"):
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    # Get list of files in archive for nested progress
                    file_list = zip_ref.namelist()
                    
                    # Extract with progress for large archives
                    for member in tqdm(file_list, desc=f"Files from {zip_file.name}", 
                                     unit="file", leave=False):
                        zip_ref.extract(member, data_dir)
                
                zip_file.unlink()  # Remove zip file after extraction
                
            except zipfile.BadZipFile:
                print(f"[WARNING] Skipping corrupted archive: {zip_file.name}")
                continue
    
    return True


def create_notebooks(project_dir):
    """Create Jupyter notebooks for the project."""
    notebooks_dir = project_dir / "notebooks"
    notebooks_dir.mkdir(exist_ok=True)
    
    # 1. EDA Notebook
    eda_notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Stanford RNA 3D Folding - Exploratory Data Analysis\n\nThis notebook conducts a comprehensive exploratory data analysis of the Stanford RNA 3D Folding competition dataset, providing strategic insights for model development and feature engineering."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# Import essential libraries for data analysis\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport plotly.express as px\nimport plotly.graph_objects as go\nfrom pathlib import Path\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Configure visualization settings\nplt.style.use('seaborn-v0_8')\nsns.set_palette('husl')\n\nprint('Libraries successfully imported!')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 1. Data Loading and Structure\n\nLet's start by loading the data and understanding its basic structure."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# Define data paths\ndata_dir = Path('../data/raw')\nprocessed_dir = Path('../data/processed')\nprocessed_dir.mkdir(exist_ok=True)\n\n# List available files\nprint('Available files:')\nfor file in data_dir.glob('*'):\n    print(f'- {file.name} ({file.stat().st_size / 1024 / 1024:.2f} MB)')"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# TODO: Load main competition data\n# Expected structure example:\n# df_train = pd.read_csv(data_dir / 'train.csv')\n# df_test = pd.read_csv(data_dir / 'test.csv')\n# df_sample = pd.read_csv(data_dir / 'sample_submission.csv')\n\nprint('Data loaded! Structure to be implemented.')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 2. RNA Sequence Analysis\n\nAnalysis of RNA sequence properties, including length, nucleotide composition and patterns."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# TODO: RNA sequence analysis\n# - Length distribution\n# - Nucleotide composition (A, U, G, C)\n# - Sequence patterns\n# - Known secondary structures\n\nprint('Sequence analysis will be implemented here.')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 3. 3D Coordinate Analysis\n\nExploration of target 3D coordinates, including spatial distributions and geometric properties."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# TODO: 3D coordinate analysis\n# - Coordinate distributions x, y, z\n# - Inter-atom distances\n# - Molecular angles and geometry\n# - 3D visualizations\n\nprint('3D coordinate analysis will be implemented.')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 4. Data Quality\n\nData quality verification, missing values, outliers and inconsistencies."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# TODO: Quality analysis\n# - Missing values\n# - Coordinate outliers\n# - Consistency between sequence and coordinates\n# - Duplicate data\n\nprint('Quality analysis will be implemented.')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 5. Insights and Conclusions\n\nSummary of key insights obtained from exploratory analysis."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# TODO: Summarize key insights\nprint('Key insights:')\nprint('1. [To be filled based on analysis]')\nprint('2. [To be filled based on analysis]')\nprint('3. [To be filled based on analysis]')"]}
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 4
    }
    
    # 2. Baseline Notebook
    baseline_notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Stanford RNA 3D Folding - Baseline Model\n\nImplementation of a simple baseline model for RNA 3D structure prediction."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# Import modeling libraries\nimport pandas as pd\nimport numpy as np\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import Dataset, DataLoader\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.metrics import mean_squared_error\nimport matplotlib.pyplot as plt\nfrom pathlib import Path\nimport pickle\n\nprint('Modeling libraries imported!')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 1. Data Preparation\n\nLoading and preparing data for baseline model training."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# Load preprocessed data from EDA\ndata_dir = Path('../data/processed')\n\n# TODO: Implement data loading\n# df_processed = pd.read_pickle(data_dir / 'processed_data.pkl')\n\nprint('Data prepared for modeling.')"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# RNA dataset class\nclass RNADataset(Dataset):\n    \"\"\"Dataset for RNA sequences and 3D coordinates.\"\"\"\n    \n    def __init__(self, sequences, coordinates):\n        self.sequences = sequences\n        self.coordinates = coordinates\n        \n        # Nucleotide mapping\n        self.nucleotide_to_idx = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'PAD': 4}\n        \n    def __len__(self):\n        return len(self.sequences)\n    \n    def __getitem__(self, idx):\n        sequence = self.encode_sequence(self.sequences[idx])\n        coords = torch.tensor(self.coordinates[idx], dtype=torch.float32)\n        return sequence, coords\n    \n    def encode_sequence(self, sequence):\n        \"\"\"Encode nucleotide sequence into tensor.\"\"\"\n        encoded = [self.nucleotide_to_idx.get(nuc, 4) for nuc in sequence]\n        return torch.tensor(encoded, dtype=torch.long)\n\nprint('RNADataset class defined.')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 2. Baseline Model: Simple LSTM\n\nImplementation of a basic LSTM model for 3D coordinate prediction."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["class RNABaselineModel(nn.Module):\n    \"\"\"Baseline LSTM model for RNA 3D structure prediction.\"\"\"\n    \n    def __init__(self, vocab_size=5, embed_dim=64, hidden_dim=128, num_layers=2, dropout=0.1):\n        super().__init__()\n        \n        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=4)\n        self.lstm = nn.LSTM(\n            embed_dim, hidden_dim, num_layers,\n            batch_first=True, dropout=dropout, bidirectional=True\n        )\n        self.dropout = nn.Dropout(dropout)\n        self.fc = nn.Sequential(\n            nn.Linear(hidden_dim * 2, hidden_dim),\n            nn.ReLU(),\n            nn.Dropout(dropout),\n            nn.Linear(hidden_dim, 3)  # x, y, z coordinates\n        )\n        \n    def forward(self, x):\n        # x shape: (batch_size, seq_len)\n        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)\n        \n        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim*2)\n        lstm_out = self.dropout(lstm_out)\n        \n        coords = self.fc(lstm_out)  # (batch_size, seq_len, 3)\n        return coords\n\n# Instantiate model\nmodel = RNABaselineModel()\nprint(f'Baseline model created with {sum(p.numel() for p in model.parameters())} parameters.')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 3. Model Training\n\nBaseline model training with validation."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# Training configuration\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nprint(f'Using device: {device}')\n\n# Hyperparameters\nlearning_rate = 1e-3\nnum_epochs = 50\nbatch_size = 32\n\n# TODO: Create dataloaders\n# train_dataset = RNADataset(train_sequences, train_coords)\n# val_dataset = RNADataset(val_sequences, val_coords)\n# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\n# val_loader = DataLoader(val_dataset, batch_size=batch_size)\n\nprint('Training configuration defined.')"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# Training function\ndef train_model(model, train_loader, val_loader, num_epochs):\n    model.to(device)\n    criterion = nn.MSELoss()\n    optimizer = optim.Adam(model.parameters(), lr=learning_rate)\n    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)\n    \n    train_losses = []\n    val_losses = []\n    \n    for epoch in range(num_epochs):\n        # TODO: Implement training loop\n        # Training\n        model.train()\n        train_loss = 0.0\n        \n        # Validation\n        model.eval()\n        val_loss = 0.0\n        \n        train_losses.append(train_loss)\n        val_losses.append(val_loss)\n        \n        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')\n    \n    return train_losses, val_losses\n\nprint('Training function defined.')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 4. Model Evaluation\n\nBaseline model evaluation using appropriate metrics."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# Evaluation metrics for 3D structures\ndef calculate_rmsd(pred_coords, true_coords):\n    \"\"\"Calculate Root Mean Square Deviation.\"\"\"\n    diff = pred_coords - true_coords\n    return np.sqrt(np.mean(np.sum(diff**2, axis=-1)))\n\ndef calculate_gdt_ts(pred_coords, true_coords, thresholds=[1.0, 2.0, 4.0, 8.0]):\n    \"\"\"Calculate GDT-TS score.\"\"\"\n    distances = np.sqrt(np.sum((pred_coords - true_coords)**2, axis=-1))\n    scores = [np.mean(distances <= t) * 100 for t in thresholds]\n    return np.mean(scores)\n\nprint('Evaluation metrics defined.')"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# TODO: Evaluate model on test set\n# model.eval()\n# predictions = []\n# targets = []\n\n# with torch.no_grad():\n#     for sequences, coords in test_loader:\n#         pred = model(sequences)\n#         predictions.append(pred.cpu().numpy())\n#         targets.append(coords.cpu().numpy())\n\nprint('Model evaluation will be implemented.')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 5. Results and Next Steps\n\nBaseline model results analysis and improvement directions."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# TODO: Summarize results\nprint('Baseline Model Results:')\nprint('- Average RMSD: [to be calculated]')\nprint('- GDT-TS score: [to be calculated]')\nprint('\\nNext steps:')\nprint('1. Implement more sophisticated architectures (Transformer)')\nprint('2. Add chemical and physical features')\nprint('3. Use ensemble methods')\nprint('4. Incorporate domain knowledge')"]}
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 4
    }
    
    # 3. Advanced Models Notebook
    advanced_notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Stanford RNA 3D Folding - Modelos Avançados\n\nImplementação de arquiteturas sofisticadas para predição de estruturas 3D de RNA."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# Importar bibliotecas avançadas\nimport pandas as pd\nimport numpy as np\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch.utils.data import DataLoader\nfrom transformers import AutoModel, AutoTokenizer\nimport pytorch_lightning as pl\nfrom pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping\nimport optuna\nimport wandb\nfrom pathlib import Path\n\nprint('Bibliotecas avançadas importadas!')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 1. Transformer para RNA\n\nImplementação de uma arquitetura Transformer especializada para sequências de RNA."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["class RNATransformer(pl.LightningModule):\n    \"\"\"Transformer model para predição de estruturas 3D de RNA.\"\"\"\n    \n    def __init__(self, vocab_size=5, d_model=512, nhead=8, num_layers=6, \n                 dropout=0.1, max_seq_len=1000, learning_rate=1e-4):\n        super().__init__()\n        self.save_hyperparameters()\n        \n        # Embedding layers\n        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=4)\n        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, d_model))\n        \n        # Transformer encoder\n        encoder_layer = nn.TransformerEncoderLayer(\n            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True\n        )\n        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)\n        \n        # Output layers\n        self.norm = nn.LayerNorm(d_model)\n        self.dropout = nn.Dropout(dropout)\n        self.fc_out = nn.Sequential(\n            nn.Linear(d_model, d_model // 2),\n            nn.GELU(),\n            nn.Dropout(dropout),\n            nn.Linear(d_model // 2, 3)\n        )\n        \n    def forward(self, x, attention_mask=None):\n        batch_size, seq_len = x.shape\n        \n        # Embeddings + positional encoding\n        embedded = self.embedding(x) + self.pos_encoding[:seq_len].unsqueeze(0)\n        \n        # Create attention mask for padding\n        if attention_mask is None:\n            attention_mask = (x == 4)  # padding token\n        \n        # Transformer\n        transformer_out = self.transformer(embedded, src_key_padding_mask=attention_mask)\n        transformer_out = self.norm(transformer_out)\n        transformer_out = self.dropout(transformer_out)\n        \n        # Output coordinates\n        coords = self.fc_out(transformer_out)\n        return coords\n    \n    def training_step(self, batch, batch_idx):\n        sequences, target_coords = batch\n        pred_coords = self(sequences)\n        \n        # Mask for non-padding positions\n        mask = (sequences != 4).unsqueeze(-1).float()\n        \n        # Masked MSE loss\n        loss = F.mse_loss(pred_coords * mask, target_coords * mask, reduction='sum')\n        loss = loss / mask.sum()\n        \n        self.log('train_loss', loss)\n        return loss\n    \n    def validation_step(self, batch, batch_idx):\n        sequences, target_coords = batch\n        pred_coords = self(sequences)\n        \n        mask = (sequences != 4).unsqueeze(-1).float()\n        loss = F.mse_loss(pred_coords * mask, target_coords * mask, reduction='sum')\n        loss = loss / mask.sum()\n        \n        self.log('val_loss', loss)\n        return loss\n    \n    def configure_optimizers(self):\n        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)\n        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)\n        return {'optimizer': optimizer, 'lr_scheduler': scheduler}\n\nprint('Modelo Transformer definido.')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 2. Graph Neural Network\n\nImplementação de GNN para capturar relações espaciais em moléculas de RNA."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# TODO: Implement GNN for RNA\n# Requires libraries like PyTorch Geometric\n# class RNAGraphNet(pl.LightningModule):\n#     def __init__(self):\n#         super().__init__()\n#         # Implement GNN layers\n#         pass\n\nprint('GNN will be implemented with PyTorch Geometric.')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 3. Ensemble Model\n\nCombination of multiple models to improve performance."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["class RNAEnsemble(nn.Module):\n    \"\"\"Ensemble of multiple RNA models.\"\"\"\n    \n    def __init__(self, models, weights=None):\n        super().__init__()\n        self.models = nn.ModuleList(models)\n        \n        if weights is None:\n            self.weights = nn.Parameter(torch.ones(len(models)) / len(models))\n        else:\n            self.register_buffer('weights', torch.tensor(weights))\n    \n    def forward(self, x):\n        predictions = []\n        for model in self.models:\n            with torch.no_grad():\n                pred = model(x)\n            predictions.append(pred)\n        \n        # Weighted average\n        weights = F.softmax(self.weights, dim=0)\n        ensemble_pred = sum(w * pred for w, pred in zip(weights, predictions))\n        \n        return ensemble_pred\n\nprint('Ensemble class defined.')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 4. Hyperparameter Optimization\n\nUsing Optuna for automatic hyperparameter optimization."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["def objective(trial):\n    \"\"\"Objective function for Optuna optimization.\"\"\"\n    \n    # Suggest hyperparameters\n    d_model = trial.suggest_categorical('d_model', [256, 512, 768])\n    nhead = trial.suggest_categorical('nhead', [4, 8, 12])\n    num_layers = trial.suggest_int('num_layers', 3, 8)\n    dropout = trial.suggest_uniform('dropout', 0.1, 0.3)\n    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)\n    \n    # Create and train model\n    model = RNATransformer(\n        d_model=d_model,\n        nhead=nhead,\n        num_layers=num_layers,\n        dropout=dropout,\n        learning_rate=learning_rate\n    )\n    \n    # TODO: Train model and return validation metric\n    # trainer = pl.Trainer(max_epochs=10, ...)\n    # trainer.fit(model, train_loader, val_loader)\n    # return best_val_loss\n    \n    return 0.5  # Placeholder\n\nprint('Optimization function defined.')"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# Execute hyperparameter optimization\n# study = optuna.create_study(direction='minimize')\n# study.optimize(objective, n_trials=50)\n\n# print('Best hyperparameters:')\n# print(study.best_params)\n\nprint('Optimization will be executed when data is ready.')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 5. Physics-Informed Neural Networks\n\nIncorporação de restrições físicas no treinamento."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["def physics_loss(pred_coords, sequences):\n    \"\"\"Calculate loss based on physical constraints.\"\"\"\n    \n    # Distance constraints between consecutive atoms\n    bond_distances = torch.norm(pred_coords[:, 1:] - pred_coords[:, :-1], dim=-1)\n    bond_loss = F.mse_loss(bond_distances, torch.ones_like(bond_distances) * 1.5)\n    \n    # Angle constraints\n    # TODO: Implement bond angle constraints\n    angle_loss = torch.tensor(0.0)\n    \n    # Energy constraints\n    # TODO: Implement molecular energy calculation\n    energy_loss = torch.tensor(0.0)\n    \n    return bond_loss + angle_loss + energy_loss\n\nclass PhysicsInformedRNA(RNATransformer):\n    \"\"\"Model with physical constraints.\"\"\"\n    \n    def __init__(self, physics_weight=0.1, **kwargs):\n        super().__init__(**kwargs)\n        self.physics_weight = physics_weight\n    \n    def training_step(self, batch, batch_idx):\n        sequences, target_coords = batch\n        pred_coords = self(sequences)\n        \n        # Standard loss\n        mask = (sequences != 4).unsqueeze(-1).float()\n        mse_loss = F.mse_loss(pred_coords * mask, target_coords * mask, reduction='sum')\n        mse_loss = mse_loss / mask.sum()\n        \n        # Physics loss\n        phys_loss = physics_loss(pred_coords, sequences)\n        \n        total_loss = mse_loss + self.physics_weight * phys_loss\n        \n        self.log('train_loss', total_loss)\n        self.log('mse_loss', mse_loss)\n        self.log('physics_loss', phys_loss)\n        \n        return total_loss\n\nprint('Physics-Informed model defined.')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 6. Model Comparison\n\nSystematic comparison of different architectures."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# TODO: Implement model comparison\nmodel_results = {\n    'LSTM Baseline': {'RMSD': 0.0, 'GDT-TS': 0.0},\n    'Transformer': {'RMSD': 0.0, 'GDT-TS': 0.0},\n    'GNN': {'RMSD': 0.0, 'GDT-TS': 0.0},\n    'Ensemble': {'RMSD': 0.0, 'GDT-TS': 0.0},\n    'Physics-Informed': {'RMSD': 0.0, 'GDT-TS': 0.0}\n}\n\nprint('Model comparison will be implemented after training.')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 7. Results Visualization\n\nVisualization of predictions and error analysis."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# TODO: Implement 3D visualizations\n# import plotly.graph_objects as go\n\n# def plot_rna_structure(coords, title='RNA Structure'):\n#     fig = go.Figure()\n#     fig.add_trace(go.Scatter3d(\n#         x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],\n#         mode='markers+lines',\n#         marker=dict(size=5),\n#         name=title\n#     ))\n#     fig.show()\n\nprint('3D visualizations will be implemented.')"]}
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 4
    }
    
    # 4. Submission Notebook
    submission_notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Stanford RNA 3D Folding - Submission Preparation\n\nFinal model preparation and submission file generation."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# Import required libraries\nimport pandas as pd\nimport numpy as np\nimport torch\nimport torch.nn as nn\nfrom pathlib import Path\nimport pickle\nimport json\nfrom datetime import datetime\n\nprint('Libraries imported for submission!')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 1. Best Model Loading\n\nLoading the model with best validation performance."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# Load the best trained model\ncheckpoints_dir = Path('../checkpoints')\nmodel_path = checkpoints_dir / 'best_model.pth'\n\n# TODO: Load specific model based on results\n# model = torch.load(model_path)\n# model.eval()\n\nprint('Best model loaded (placeholder).')"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# Load test data\ndata_dir = Path('../data/raw')\n\n# TODO: Load competition test data\n# test_df = pd.read_csv(data_dir / 'test.csv')\n# sample_submission = pd.read_csv(data_dir / 'sample_submission.csv')\n\nprint('Test data loaded (placeholder).')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 2. Test Data Preprocessing\n\nApplying the same transformations used in training."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# Load preprocessor\nprocessed_dir = Path('../data/processed')\n\n# TODO: Load and apply preprocessing\n# with open(processed_dir / 'preprocessor.pkl', 'rb') as f:\n#     preprocessor = pickle.load(f)\n\n# test_processed = preprocessor.transform(test_df)\n\nprint('Preprocessing applied to test data.')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 3. Prediction Generation\n\nGenerating predictions for the test set."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# Function to generate predictions\ndef generate_predictions(model, test_data, batch_size=32):\n    \"\"\"Generate predictions for test data.\"\"\"\n    \n    model.eval()\n    predictions = []\n    \n    # TODO: Implement prediction generation\n    # with torch.no_grad():\n    #     for batch in test_loader:\n    #         pred = model(batch)\n    #         predictions.append(pred.cpu().numpy())\n    \n    # return np.concatenate(predictions)\n    \n    return np.random.randn(100, 3)  # Placeholder\n\n# Generate predictions\n# predictions = generate_predictions(model, test_processed)\nprint('Predictions generated (placeholder).')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 4. Postprocessing and Validation\n\nApplying postprocessing and validation to predictions."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["def postprocess_predictions(predictions):\n    \"\"\"Apply postprocessing to predictions.\"\"\"\n    \n    # Clip extreme values\n    predictions = np.clip(predictions, -50, 50)\n    \n    # Trajectory smoothing\n    # TODO: Implement physics-based smoothing\n    \n    # Normalization\n    # TODO: Apply normalization if necessary\n    \n    return predictions\n\ndef validate_predictions(predictions, sequences):\n    \"\"\"Validate predictions against known constraints.\"\"\"\n    \n    issues = []\n    \n    # Check bond distances\n    for i, pred in enumerate(predictions):\n        distances = np.linalg.norm(pred[1:] - pred[:-1], axis=1)\n        if np.any(distances < 0.5) or np.any(distances > 3.0):\n            issues.append(f'Sequence {i}: suspicious bond distances')\n    \n    # Check valid coordinates\n    if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):\n        issues.append('Invalid coordinates found')\n    \n    return issues\n\n# Apply postprocessing\n# predictions_processed = postprocess_predictions(predictions)\n# validation_issues = validate_predictions(predictions_processed, test_sequences)\n\nprint('Postprocessing and validation implemented.')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 5. Submission File Formatting\n\nFormatting predictions in the format required by the competition."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["def format_submission(predictions, sample_submission):\n    \"\"\"Format predictions for submission.\"\"\"\n    \n    submission = sample_submission.copy()\n    \n    # TODO: Map predictions to competition format\n    # This depends on the specific format required\n    \n    # Generic example:\n    # submission['prediction'] = predictions.flatten()\n    \n    return submission\n\n# Create submission file\n# submission = format_submission(predictions_processed, sample_submission)\n\nprint('Submission format prepared.')"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# Save submission file\nsubmissions_dir = Path('../submissions')\nsubmissions_dir.mkdir(exist_ok=True)\n\ntimestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\nsubmission_filename = f'submission_{timestamp}.csv'\n\n# submission.to_csv(submissions_dir / submission_filename, index=False)\n\nprint(f'Submission file saved: {submission_filename}')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 6. Final Validation and Metadata\n\nFinal file validation and submission metadata creation."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# Final file validation\ndef final_validation(submission_path):\n    \"\"\"Final submission file validation.\"\"\"\n    \n    # Load file\n    submission = pd.read_csv(submission_path)\n    \n    checks = {\n        'correct_format': True,  # Check required columns\n        'no_null_values': not submission.isnull().any().any(),\n        'correct_size': len(submission) > 0,\n        'numeric_values': submission.select_dtypes(include=[np.number]).shape[1] > 0\n    }\n    \n    return checks\n\n# validation_results = final_validation(submissions_dir / submission_filename)\nprint('Final validation implemented.')"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["# Create submission metadata\nsubmission_metadata = {\n    'timestamp': datetime.now().isoformat(),\n    'model_type': 'Ensemble (LSTM + Transformer)',\n    'preprocessing': 'StandardScaler + Sequence encoding',\n    'postprocessing': 'Clipping + Physics constraints',\n    'validation_score': 0.0,  # TODO: Validation score\n    'training_epochs': 100,\n    'notes': 'Final submission with best ensemble model',\n    'files': {\n        'submission': submission_filename,\n        'model': 'best_model.pth',\n        'preprocessor': 'preprocessor.pkl'\n    }\n}\n\n# Save metadata\nmetadata_filename = f'submission_metadata_{timestamp}.json'\nwith open(submissions_dir / metadata_filename, 'w') as f:\n    json.dump(submission_metadata, f, indent=2)\n\nprint(f'Metadata saved: {metadata_filename}')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 7. Submission Summary\n\nFinal submission summary and next steps."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], 
             "source": ["print('=== SUBMISSION SUMMARY ===')\nprint(f'File: {submission_filename}')\nprint(f'Model: Ensemble (LSTM + Transformer)')\nprint(f'Validation score: [to be filled]')\nprint(f'Timestamp: {timestamp}')\nprint()\nprint('=== NEXT STEPS ===')\nprint('1. Verify submission file')\nprint('2. Upload to Kaggle')\nprint('3. Document results')\nprint('4. Prepare final report')\nprint()\nprint('=== GENERATED FILES ===')\nprint(f'- {submission_filename}')\nprint(f'- {metadata_filename}')\nprint('- Training logs in ../checkpoints/')"]}
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 4
    }
    
    # Save all notebooks
    notebooks = {
        "01_eda.ipynb": eda_notebook,
        "02_baseline.ipynb": baseline_notebook,
        "03_advanced.ipynb": advanced_notebook,
        "04_submission.ipynb": submission_notebook
    }
    
    for filename, notebook_content in notebooks.items():
        with open(notebooks_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(notebook_content, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Created {len(notebooks)} notebooks in {notebooks_dir}")


def create_source_files(project_dir):
    """Create Python source files for the project."""
    src_dir = project_dir / "src"
    src_dir.mkdir(exist_ok=True)
    
    # Create __init__.py
    init_code = '''#!/usr/bin/env python3
"""
Stanford RNA 3D Folding - Source Package

Author: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
Created: October 18, 2025 at 14:30:00
License: MIT License
Kaggle Competition: https://www.kaggle.com/competitions/stanford-rna-3d-folding

MIT License

Copyright (c) 2025 Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""'''
    (src_dir / "__init__.py").write_text(init_code)
    
    # Create data_processing.py
    data_processing_code = '''#!/usr/bin/env python3
"""
Data processing utilities for Stanford RNA 3D Folding.

Author: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
Created: October 18, 2025 at 14:30:00
License: MIT License
Kaggle Competition: https://www.kaggle.com/competitions/stanford-rna-3d-folding

MIT License

Copyright (c) 2025 Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class RNADataProcessor:
    """Processes RNA sequence and structure data."""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.nucleotide_mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
        
    def load_raw_data(self):
        """Load raw competition data."""
        print("Loading raw competition data...")
        # TODO: Implement actual data loading
        return pd.DataFrame(), pd.DataFrame()
    
    def encode_sequence(self, sequence):
        """Encode RNA sequence to numerical format."""
        return [self.nucleotide_mapping.get(nuc, 0) for nuc in sequence.upper()]


if __name__ == "__main__":
    processor = RNADataProcessor("../data/raw")
    processor.load_raw_data()
'''
    (src_dir / "data_processing.py").write_text(data_processing_code)
    
    # Create models.py
    models_code = '''#!/usr/bin/env python3
"""
Model architectures for Stanford RNA 3D Folding.

Author: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
Created: October 18, 2025 at 14:30:00
License: MIT License
Kaggle Competition: https://www.kaggle.com/competitions/stanford-rna-3d-folding

MIT License

Copyright (c) 2025 Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn


class SimpleRNAPredictor(nn.Module):
    """Simple LSTM-based RNA 3D structure predictor."""
    
    def __init__(self, vocab_size=4, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3)  # x, y, z coordinates
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        coords = self.fc(lstm_out)
        return coords


if __name__ == "__main__":
    model = SimpleRNAPredictor()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
'''
    (src_dir / "models.py").write_text(models_code)
    
    print(f"[INFO] Created source files in {src_dir}")


def create_project_files(project_dir, language="en"):
    """Create project configuration files."""
    
    # Create README
    readme_en = """# Stanford RNA 3D Folding — Portfolio Project

**Author**: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>  
**Created**: October 18, 2025 at 14:30:00  
**License**: MIT License  
**Kaggle Competition**: https://www.kaggle.com/competitions/stanford-rna-3d-folding  

---

This repository demonstrates machine learning techniques for RNA 3D structure prediction.

## MIT License

Copyright (c) 2025 Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Project Structure
- `notebooks/` - Jupyter notebooks with analysis
- `src/` - Python source code
- `data/` - Competition data
- `configs/` - Configuration files

## Quick Start
1. Create & activate a Python virtual environment (e.g. `python3 -m venv .venv && source .venv/bin/activate`)
2. Install requirements: `pip install -r requirements.txt`
3. Start Jupyter: `jupyter lab`
4. Open `notebooks/01_eda.ipynb`

## Modeling Approach
- Sequence-based deep learning models
- LSTM and Transformer architectures
- 3D coordinate prediction
"""
    
    readme_pt = """# Stanford RNA 3D Folding — Projeto de Portfólio

**Autor**: test  
**Criado**: October 18, 2025 at 14:30:00  
**Licença**: MIT License  
**Competição Kaggle**: https://www.kaggle.com/competitions/stanford-rna-3d-folding  

---

Este repositório demonstra técnicas de machine learning para predição de estruturas 3D de RNA.

## Licença MIT

Copyright (c) 2025 Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Estrutura do Projeto
- `notebooks/` - Notebooks Jupyter com análises
- `src/` - Código fonte Python
- `data/` - Dados da competição
- `configs/` - Arquivos de configuração

## Início Rápido
1. Criar e ativar um ambiente virtual Python (ex.: `python3 -m venv .venv && source .venv/bin/activate`)
2. Instalar dependências: `pip install -r requirements.txt`
3. Iniciar Jupyter: `jupyter lab`
4. Abrir `notebooks/01_eda.ipynb`

## Abordagem de Modelagem
- Modelos de deep learning baseados em sequência
- Arquiteturas LSTM e Transformer
- Predição de coordenadas 3D
"""
    
    readme_content = readme_pt if language == "pt" else readme_en
    (project_dir / "README.md").write_text(readme_content, encoding='utf-8')
    
    # Create requirements.txt
    requirements = """# Stanford RNA 3D Folding - Requirements
# Author: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
# Created: October 18, 2025 at 14:30:00
# License: MIT License
# Kaggle Competition: https://www.kaggle.com/competitions/stanford-rna-3d-folding

# Additional requirements for Stanford RNA 3D Folding
plotly>=5.0.0
seaborn>=0.11.0
biopython>=1.79
optuna>=3.0.0
wandb>=0.15.0
tqdm>=4.64.0
"""
    (project_dir / "requirements.txt").write_text(requirements)
    
    # Create Makefile
    makefile = """# Stanford RNA 3D Folding - Makefile
# Author: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
# Created: October 18, 2025 at 14:30:00
# License: MIT License
# Kaggle Competition: https://www.kaggle.com/competitions/stanford-rna-3d-folding

.PHONY: help data train predict clean

PYTHON ?= python

help:
\t@echo "Available commands:"
\t@echo "  data     - Download and preprocess data"
\t@echo "  train    - Train baseline model"
\t@echo "  predict  - Generate predictions"
\t@echo "  clean    - Clean temporary files"

data:
\t$(PYTHON) src/data_processing.py

train:
\t$(PYTHON) src/training.py

predict:
\t$(PYTHON) src/models.py

clean:
\tfind . -type f -name "*.pyc" -delete
\tfind . -type d -name "__pycache__" -delete
"""
    (project_dir / "Makefile").write_text(makefile)
    
    # Create .gitignore
    gitignore = """# Stanford RNA 3D Folding - GitIgnore
# Author: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
# Created: October 18, 2025 at 14:30:00
# License: MIT License
# Kaggle Competition: https://www.kaggle.com/competitions/stanford-rna-3d-folding

# Python
__pycache__/
*.py[cod]
*.so
.Python
build/
dist/
*.egg-info/
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/

# Data files
data/raw/*
!data/raw/.gitkeep
checkpoints/
submissions/*.csv

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Model files
*.pth
*.pt
*.pkl
"""
    (project_dir / ".gitignore").write_text(gitignore)


def create_project_structure(project_dir, language="en"):
    """Create complete project structure."""
    project_dir = Path(project_dir)
    project_dir.mkdir(exist_ok=True)
    
    # Create directory structure with progress
    directories = [
        "data/raw", "data/processed", "data/external",
        "notebooks", "src", "configs", "checkpoints", 
        "submissions", "tests"
    ]
    
    print("[INFO] Creating directory structure...")
    for dir_path in tqdm(directories, desc="Creating directories", unit="dir"):
        (project_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create project files with progress indication
    setup_tasks = [
        ("Project files", lambda: create_project_files(project_dir, language)),
        ("Notebooks", lambda: create_notebooks(project_dir)),
        ("Source files", lambda: create_source_files(project_dir)),
    ]
    
    print("[INFO] Setting up project components...")
    for task_name, task_func in tqdm(setup_tasks, desc="Setup components", unit="task"):
        task_func()
    
    # Create empty .gitkeep files
    gitkeep_dirs = ["data/raw", "data/processed", "data/external", "checkpoints", "tests"]
    for subdir in tqdm(gitkeep_dirs, desc="Creating .gitkeep files", unit="file"):
        (project_dir / subdir / ".gitkeep").touch()


def main():
    """Main function to setup the project."""
    parser = argparse.ArgumentParser(description="Setup Stanford RNA 3D Folding project")
    parser.add_argument("--dest", default="./stanford_rna3d", 
                       help="Destination directory for the project")
    parser.add_argument("--lang", choices=["en", "pt"], default="en",
                       help="Language for documentation (en=English, pt=Portuguese)")
    parser.add_argument("--download-data", action="store_true",
                       help="Download competition data (requires Kaggle API setup)")
    
    args = parser.parse_args()
    
    project_dir = Path(args.dest)
    
    print(f"[INFO] Setting up Stanford RNA 3D Folding project in: {project_dir.absolute()}")
    
    # Create project structure
    create_project_structure(project_dir, args.lang)
    
    # Download data if requested
    if args.download_data:
        download_competition_data(project_dir)
    
    print(f"\n[SUCCESS] Project setup completed!")
    print(f"Project directory: {project_dir.absolute()}")
    print(f"Documentation language: {'Portuguese' if args.lang == 'pt' else 'English'}")
    
    print("\nNext steps:")
    print(f"1. cd {project_dir}")
    print("2. Activate your Python virtual environment")
    print("3. pip install -r requirements.txt")
    print("4. jupyter lab")
    print("5. Open notebooks/01_eda.ipynb to start exploring")
    
    if not args.download_data:
        print(f"\nNote: Competition data not downloaded.")
        print(f"   Run with --download-data flag after setting up Kaggle API")


if __name__ == "__main__":
    main()
