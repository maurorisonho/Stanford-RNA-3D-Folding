# Stanford RNA 3D Folding - Makefile
# Author: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
# Created: October 18, 2025 at 14:30:00
# License: MIT License
# Kaggle Competition: https://www.kaggle.com/competitions/stanford-rna-3d-folding

.PHONY: help data train predict clean

PYTHON ?= python

help:
	@echo "Available commands:"
	@echo "  data     - Download and preprocess data"
	@echo "  train    - Train baseline model"
	@echo "  predict  - Generate predictions"
	@echo "  clean    - Clean temporary files"

data:
	$(PYTHON) src/data_processing.py

train:
	$(PYTHON) src/training.py

predict:
	$(PYTHON) src/models.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
