# Configuration
PYTHON = python3
PIP = pip3
PROJECT_DIR = .
DATA_DIR = ./data
MODELS_DIR = ./models
RESULTS_DIR = ./results
DOCS_DIR = ./docs

# Default target
all: install data train

# Installation targets
install:
	@echo "Installing Python dependencies..."
	$(PIP) install -r requirements.txt

install-dev:
	@echo "Installing development dependencies..."
	$(PIP) install -r requirements-dev.txt

# Data targets
data: download-data prepare-data

download-data:
	@echo "Downloading MNIST dataset..."
	$(PYTHON) -c "from torchvision import datasets; datasets.MNIST('$(DATA_DIR)', train=True, download=True)"

prepare-data:
	@echo "Preparing data directories..."
	mkdir -p $(DATA_DIR) $(MODELS_DIR) $(RESULTS_DIR)/{metrics,predictions,plots}

# Training targets
train: train-teacher train-students

train-teacher:
	@echo "Training teacher model..."
	$(PYTHON) $(PROJECT_DIR)/main.py --train-teacher

train-students:
	@echo "Training student models..."
	$(PYTHON) $(PROJECT_DIR)/main.py --train-students

distill:
	@echo "Running knowledge distillation..."
	$(PYTHON) $(PROJECT_DIR)/main.py --distill

# Evaluation targets
evaluate:
	@echo "Evaluating models..."
	$(PYTHON) $(PROJECT_DIR)/main.py --evaluate

benchmark:
	@echo "Running comprehensive benchmarks..."
	$(PYTHON) $(PROJECT_DIR)/model_benchmark.py

# Visualization targets
visualize:
	@echo "Generating visualizations..."
	$(PYTHON) $(PROJECT_DIR)/visualization.py

# Documentation targets
docs:
	@echo "Building documentation..."
	cd $(DOCS_DIR) && make html

view-docs:
	@echo "Opening documentation in browser..."
	xdg-open $(DOCS_DIR)/_build/html/index.html || open $(DOCS_DIR)/_build/html/index.html

# Cleanup targets
clean:
	@echo "Cleaning up..."
	rm -rf $(DATA_DIR)/processed
	rm -f $(MODELS_DIR)/*.pth
	rm -rf $(RESULTS_DIR)/*

clean-models:
	@echo "Cleaning model files..."
	rm -f $(MODELS_DIR)/*.pth

clean-results:
	@echo "Cleaning result files..."
	rm -rf $(RESULTS_DIR)/*

clean-docs:
	@echo "Cleaning documentation..."
	cd $(DOCS_DIR) && make clean

# Utility targets
lint:
	@echo "Running linter..."
	flake8 $(PROJECT_DIR)

test:
	@echo "Running tests..."
	pytest tests/

format:
	@echo "Formatting code..."
	black $(PROJECT_DIR)
	isort $(PROJECT_DIR)

# Help target
help:
	@echo "Knowledge Distillation Project Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all             : Install dependencies and run full pipeline (default)"
	@echo "  install         : Install production dependencies"
	@echo "  install-dev     : Install development dependencies"
	@echo "  data            : Download and prepare data"
	@echo "  train           : Train both teacher and student models"
	@echo "  train-teacher   : Train only the teacher model"
	@echo "  train-students  : Train only the student models"
	@echo "  distill         : Run knowledge distillation"
	@echo "  evaluate        : Evaluate model performance"
	@echo "  benchmark       : Run comprehensive benchmarks"
	@echo "  visualize       : Generate visualizations"
	@echo "  docs            : Build documentation"
	@echo "  view-docs       : View documentation in browser"
	@echo "  clean           : Remove all generated files"
	@echo "  clean-models    : Remove trained models"
	@echo "  clean-results   : Remove result files"
	@echo "  clean-docs      : Clean documentation builds"
	@echo "  lint            : Run code linter"
	@echo "  test            : Run tests"
	@echo "  format          : Format code"
	@echo "  help            : Show this help message"

.PHONY: all install install-dev data download-data prepare-data \
        train train-teacher train-students distill evaluate benchmark \
        visualize docs view-docs clean clean-models clean-results clean-docs \
        lint test format help