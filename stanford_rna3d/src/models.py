#!/usr/bin/env python3
"""
Baseline neural-network models for Stanford RNA 3D Folding.

Author: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
License: MIT License
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Configuration parameters for baseline models."""

    vocab_size: int = 5  # includes padding/unknown token
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2


class SimpleRNAPredictor(nn.Module):
    """Bidirectional LSTM regressor that predicts 3D coordinates per residue."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        self.embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.encoder = nn.LSTM(
            input_size=self.config.embedding_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 3),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return xyz coordinates for each token in the sequence."""
        # tokens: (batch, seq_len)
        embedded = self.embedding(tokens)
        encoded, _ = self.encoder(embedded)
        coords = self.head(encoded)
        return coords


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute MSE while ignoring padded positions indicated by mask."""
    # mask: 1 for valid positions, 0 for padding
    diff = (prediction - target) ** 2
    diff = diff.sum(dim=-1)
    masked = diff * mask
    return masked.sum() / mask.sum().clamp_min(1.0)


if __name__ == "__main__":
    model = SimpleRNAPredictor()
    dummy = torch.randint(0, model.config.vocab_size, (2, 10))
    output = model(dummy)
    print(f"Output shape: {output.shape}")
