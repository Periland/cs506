# Makefile for 2-Hour Trip Prediction Model
# This makefile sets up the environment and runs the prediction model

PYTHON = python3
VENV = venv
VENV_PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
REQUIREMENTS = requirements.txt
SCRIPT = final_enhanced_predict2h.py

.PHONY: all setup install run clean help run-default run-ensemble run-rmsprop run-l1 run-huber run-smape interactive

all: help

help:
	@echo "Usage:"
	@echo "  make setup     - Create Python virtual environment"
	@echo "  make install   - Install all dependencies"
	@echo "  make run       - Run the model with default parameters"
	@echo "  make run-ensemble - Run the model with ensemble approach"
	@echo "  make run-rmsprop - Run the model with RMSprop optimizer"
	@echo "  make run-l1    - Run the model with L1 regularization"
	@echo "  make run-huber - Run the model with Huber loss"
	@echo "  make run-smape - Run the model with SMAPE loss"
	@echo "  make interactive - Run the model with interactive parameter selection"
	@echo "  make clean     - Remove virtual environment and cached files"

# Create virtual environment
$(VENV):
	$(PYTHON) -m venv $(VENV)

# Create requirements file
$(REQUIREMENTS):
	@echo "pandas>=1.3.0" > $(REQUIREMENTS)
	@echo "numpy>=1.19.5" >> $(REQUIREMENTS)
	@echo "matplotlib>=3.4.0" >> $(REQUIREMENTS)
	@echo "seaborn>=0.11.0" >> $(REQUIREMENTS)
	@echo "holidays>=0.14" >> $(REQUIREMENTS)
	@echo "requests>=2.25.0" >> $(REQUIREMENTS)
	@echo "scikit-learn>=1.0.0" >> $(REQUIREMENTS)
	@echo "tensorflow>=2.8.0" >> $(REQUIREMENTS)

# Setup virtual environment
setup: $(VENV)

# Install dependencies
install: setup $(REQUIREMENTS)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS)
	@echo "All dependencies have been installed."

# Run model with default parameters
run: install
	$(VENV_PYTHON) $(SCRIPT) --mode default

# Run model with ensemble approach
run-ensemble: install
	$(VENV_PYTHON) $(SCRIPT) --mode ensemble

# Run model with RMSprop optimizer
run-rmsprop: install
	$(VENV_PYTHON) $(SCRIPT) --mode rmsprop

# Run model with L1 regularization
run-l1: install
	$(VENV_PYTHON) $(SCRIPT) --mode l1

# Run model with Huber loss
run-huber: install
	$(VENV_PYTHON) $(SCRIPT) --mode huber

# Run model with SMAPE loss
run-smape: install
	$(VENV_PYTHON) $(SCRIPT) --mode smape

# Run model with interactive parameter selection
interactive: install
	$(VENV_PYTHON) $(SCRIPT) --mode interactive

# Clean up the environment
clean:
	rm -rf $(VENV)
	rm -f $(REQUIREMENTS)
	rm -f *.png
	rm -f *.keras
	rm -f *.h5
	rm -f *.pkl
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete