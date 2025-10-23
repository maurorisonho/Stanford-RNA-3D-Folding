# Author: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
# Created: October 18, 2025 at 14:30:00
# License: MIT License
# Kaggle Competition: https://www.kaggle.com/competitions/stanford-rna-3d-folding
#
# ---
#
# MIT License
#
# Copyright (c) 2025 Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ---

#!/usr/bin/env python3

"""
Data processing utilities for the Stanford RNA 3D Folding competition.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

RAW_FILENAMES = {
    "train": "train.csv",
    "test": "test.csv",
    "submission": "sample_submission.csv",
}

NUCLEOTIDE_TO_ID = {"A": 0, "U": 1, "G": 2, "C": 3, "N": 4}
ID_TO_NUCLEOTIDE = {idx: nuc for nuc, idx in NUCLEOTIDE_TO_ID.items()}


@dataclass
class RNASequence:
    """Container for an encoded RNA sequence and optional coordinates."""

    sequence_id: str
    tokens: np.ndarray
    coordinates: np.ndarray | None = None


class RNADataProcessor:
    """Process RNA sequence and structure data stored in CSV files."""

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def _csv_path(self, name: str) -> Path:
        if name not in RAW_FILENAMES:
            raise KeyError(f"Unknown dataset: {name}")
        return self.data_dir / RAW_FILENAMES[name]

    def load_dataframe(self, name: str) -> pd.DataFrame:
        """Load a raw CSV file into a DataFrame."""
        path = self._csv_path(name)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset '{name}' not found at {path}. Download the Kaggle files first."
            )
        return pd.read_csv(path)

    def load_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convenience wrapper returning both train and test DataFrames."""
        return self.load_dataframe("train"), self.load_dataframe("test")

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def encode_sequence(self, sequence: str) -> np.ndarray:
        """Convert a nucleotide string into integer identifiers."""
        tokens = [NUCLEOTIDE_TO_ID.get(char.upper(), NUCLEOTIDE_TO_ID["N"]) for char in sequence]
        return np.asarray(tokens, dtype=np.int64)

    def decode_sequence(self, tokens: Iterable[int]) -> str:
        """Convert integer identifiers back into characters."""
        return "".join(ID_TO_NUCLEOTIDE.get(int(token), "N") for token in tokens)

    # ------------------------------------------------------------------
    # Dataset assembly
    # ------------------------------------------------------------------
    def build_sequences(self, frame: pd.DataFrame) -> List[RNASequence]:
        """Create a list of RNASequence objects from a DataFrame."""
        sequences: List[RNASequence] = []
        for row in frame.itertuples(index=False):
            sequence_id = getattr(row, "id", getattr(row, "sequence_id", None))
            if sequence_id is None:
                raise KeyError("Expected an 'id' or 'sequence_id' column in the dataset")

            sequence = getattr(row, "sequence")
            coords = getattr(row, "coordinates", None)

            encoded = self.encode_sequence(sequence)
            coord_array = None
            if isinstance(coords, str):
                coord_array = self._parse_coordinate_string(coords)

            sequences.append(RNASequence(sequence_id, encoded, coord_array))
        return sequences

    @staticmethod
    def _parse_coordinate_string(coords: str) -> np.ndarray:
        """Convert a semicolon-delimited coordinate string to a numpy array."""
        triples = []
        for item in coords.split(";"):
            if not item.strip():
                continue
            parts = [float(value) for value in item.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Malformed coordinate triple: '{item}'")
            triples.append(parts)
        return np.asarray(triples, dtype=np.float32)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def kmer_counts(self, sequence: str, k: int = 3) -> Dict[str, int]:
        """Compute simple k-mer counts for exploratory analysis."""
        sequence = sequence.upper()
        counts: Dict[str, int] = {}
        for i in range(max(0, len(sequence) - k + 1)):
            kmer = sequence[i:i + k]
            counts[kmer] = counts.get(kmer, 0) + 1
        return counts


if __name__ == "__main__":
    processor = RNADataProcessor(Path(__file__).resolve().parents[1] / "data" / "raw")
    try:
        train_df, test_df = processor.load_train_test()
        print(f"train rows: {len(train_df)}, test rows: {len(test_df)}")
    except FileNotFoundError as exc:
        print(f"[warn] {exc}")
