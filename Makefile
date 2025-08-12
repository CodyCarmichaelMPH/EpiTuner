# EpiTuner Makefile
# Automate common tasks for LoRA training and evaluation

.PHONY: help install setup clean train infer eval run-app check-deps

# Default target
help:
	@echo "EpiTuner - LoRA Fine-tuning for Medical Data"
	@echo ""
	@echo "Available commands:"
	@echo "  install       Install Python dependencies"
	@echo "  setup         Create necessary directories"
	@echo "  clean         Clean up generated files"
	@echo "  run-app       Start Streamlit application"
	@echo "  train         Train LoRA model (requires DATA, MODEL, TOPIC, OUTPUT)"
	@echo "  infer         Run inference (requires MODEL, CONFIG, DATA, TOPIC, OUTPUT)"
	@echo "  eval          Evaluate model (requires PREDICTIONS, GROUND_TRUTH, OUTPUT_DIR)"
	@echo "  check-deps    Check system dependencies"
	@echo "  check-ollama  Check Ollama installation and models"
	@echo ""
	@echo "Examples:"
	@echo "  make install"
	@echo "  make run-app"
	@echo "  make train DATA=sample_data/medical_sample.csv MODEL=microsoft/phi-2 TOPIC='motor vehicle collisions' OUTPUT=outputs/test_model"
	@echo "  make infer MODEL=outputs/test_model CONFIG=configs/config_base.yaml DATA=sample_data/medical_sample.csv TOPIC='motor vehicle collisions' OUTPUT=outputs/predictions.json"

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

# Setup directories
setup:
	@echo "Creating directory structure..."
	@mkdir -p configs data outputs adapters chat_templates scripts workflows sample_data
	@echo "Directory structure created!"

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	@rm -rf outputs/training_*
	@rm -rf outputs/predictions_*
	@rm -rf outputs/evaluation_*
	@rm -rf __pycache__
	@rm -rf .streamlit
	@find . -name "*.pyc" -delete
	@find . -name "*.pyo" -delete
	@echo "Cleanup complete!"

# Run Streamlit app
run-app:
	@echo "Starting EpiTuner application..."
	@echo "Open your browser to: http://localhost:8501"
	streamlit run app.py

# Check system dependencies
check-deps:
	@echo "Checking system dependencies..."
	@python --version || (echo "Python not found!" && exit 1)
	@pip --version || (echo "pip not found!" && exit 1)
	@echo "Checking Python packages..."
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')" || (echo "PyTorch not installed!" && exit 1)
	@python -c "import transformers; print(f'Transformers: {transformers.__version__}')" || (echo "Transformers not installed!" && exit 1)
	@python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')" || (echo "Streamlit not installed!" && exit 1)
	@echo "All dependencies OK!"

# Check Ollama installation
check-ollama:
	@echo "Checking Ollama installation..."
	@ollama --version || (echo "Ollama not found! Install from https://ollama.ai" && exit 1)
	@echo "Available Ollama models:"
	@ollama list || echo "No models found. Try: ollama pull llama2"

# Train model (requires parameters)
train:
	@if [ -z "$(DATA)" ] || [ -z "$(MODEL)" ] || [ -z "$(TOPIC)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make train DATA=<csv_file> MODEL=<model_name> TOPIC=<classification_topic> OUTPUT=<output_dir>"; \
		echo "Example: make train DATA=sample_data/medical_sample.csv MODEL=microsoft/phi-2 TOPIC='motor vehicle collisions' OUTPUT=outputs/mvc_model"; \
		exit 1; \
	fi
	@echo "Training LoRA model..."
	@mkdir -p $(OUTPUT)
	python scripts/train.py \
		--config configs/config_base.yaml \
		--data $(DATA) \
		--model $(MODEL) \
		--topic "$(TOPIC)" \
		--output $(OUTPUT)

# Run inference (requires parameters)  
infer:
	@if [ -z "$(MODEL)" ] || [ -z "$(CONFIG)" ] || [ -z "$(DATA)" ] || [ -z "$(TOPIC)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make infer MODEL=<model_path> CONFIG=<config_path> DATA=<csv_file> TOPIC=<topic> OUTPUT=<output_json>"; \
		echo "Example: make infer MODEL=outputs/mvc_model CONFIG=configs/config_base.yaml DATA=sample_data/medical_sample.csv TOPIC='motor vehicle collisions' OUTPUT=outputs/predictions.json"; \
		exit 1; \
	fi
	@echo "Running inference..."
	python scripts/inference.py \
		--model $(MODEL) \
		--config $(CONFIG) \
		--data $(DATA) \
		--topic "$(TOPIC)" \
		--output $(OUTPUT)

# Evaluate model (requires parameters)
eval:
	@if [ -z "$(PREDICTIONS)" ] || [ -z "$(GROUND_TRUTH)" ] || [ -z "$(OUTPUT_DIR)" ]; then \
		echo "Usage: make eval PREDICTIONS=<predictions_json> GROUND_TRUTH=<ground_truth_csv> OUTPUT_DIR=<output_directory>"; \
		echo "Example: make eval PREDICTIONS=outputs/predictions.json GROUND_TRUTH=sample_data/medical_sample.csv OUTPUT_DIR=outputs/evaluation"; \
		exit 1; \
	fi
	@echo "Evaluating model..."
	@mkdir -p $(OUTPUT_DIR)
	python scripts/evaluate.py \
		--predictions $(PREDICTIONS) \
		--ground_truth $(GROUND_TRUTH) \
		--output_dir $(OUTPUT_DIR)

# Quick test pipeline
test-pipeline:
	@echo "Running test pipeline..."
	@make setup
	@make train DATA=sample_data/medical_sample.csv MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 TOPIC="motor vehicle collisions" OUTPUT=outputs/test_model
	@make infer MODEL=outputs/test_model CONFIG=configs/config_base.yaml DATA=sample_data/medical_sample.csv TOPIC="motor vehicle collisions" OUTPUT=outputs/test_predictions.json
	@make eval PREDICTIONS=outputs/test_predictions.json GROUND_TRUTH=sample_data/medical_sample.csv OUTPUT_DIR=outputs/test_evaluation
	@echo "Test pipeline complete! Check outputs/ directory for results."

# Development commands
dev-install:
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	pip install black flake8 pytest

format:
	@echo "Formatting code with black..."
	black .

lint:
	@echo "Linting code with flake8..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Windows-specific commands
windows-check:
	@echo "Checking Windows environment..."
	@powershell -Command "Get-WmiObject -Class Win32_VideoController | Select-Object Name, AdapterRAM"
	@echo "PowerShell available: OK"

windows-setup:
	@echo "Setting up EpiTuner for Windows..."
	@powershell -Command "if (!(Test-Path 'configs')) { New-Item -ItemType Directory -Force -Path configs }"
	@powershell -Command "if (!(Test-Path 'data')) { New-Item -ItemType Directory -Force -Path data }"
	@powershell -Command "if (!(Test-Path 'outputs')) { New-Item -ItemType Directory -Force -Path outputs }"
	@echo "Windows setup complete!"

# Help for specific topics
help-train:
	@echo "Training Help:"
	@echo "============="
	@echo "The train command fine-tunes a language model using LoRA on medical data."
	@echo ""
	@echo "Required parameters:"
	@echo "  DATA    - Path to CSV file with medical records"
	@echo "  MODEL   - HuggingFace model name (e.g., microsoft/phi-2)"
	@echo "  TOPIC   - Description of classification task"
	@echo "  OUTPUT  - Directory to save trained model"
	@echo ""
	@echo "Example:"
	@echo "  make train DATA=sample_data/medical_sample.csv \\"
	@echo "             MODEL=microsoft/phi-2 \\"
	@echo "             TOPIC='motor vehicle collision detection' \\"
	@echo "             OUTPUT=outputs/mvc_model"

help-infer:
	@echo "Inference Help:"
	@echo "=============="
	@echo "The infer command runs predictions on new data using a trained model."
	@echo ""
	@echo "Required parameters:"
	@echo "  MODEL   - Path to trained model directory"
	@echo "  CONFIG  - Path to configuration file"
	@echo "  DATA    - Path to CSV file with records to classify"
	@echo "  TOPIC   - Same classification topic used in training"
	@echo "  OUTPUT  - Path for output JSON file"
	@echo ""
	@echo "Example:"
	@echo "  make infer MODEL=outputs/mvc_model \\"
	@echo "             CONFIG=configs/config_base.yaml \\"
	@echo "             DATA=new_data.csv \\"
	@echo "             TOPIC='motor vehicle collision detection' \\"
	@echo "             OUTPUT=outputs/new_predictions.json"

help-eval:
	@echo "Evaluation Help:"
	@echo "==============="
	@echo "The eval command evaluates model predictions against expert annotations."
	@echo ""
	@echo "Required parameters:"
	@echo "  PREDICTIONS   - Path to predictions JSON file from inference"
	@echo "  GROUND_TRUTH  - Path to CSV file with expert annotations"
	@echo "  OUTPUT_DIR    - Directory to save evaluation results"
	@echo ""
	@echo "Example:"
	@echo "  make eval PREDICTIONS=outputs/predictions.json \\"
	@echo "            GROUND_TRUTH=sample_data/medical_sample.csv \\"
	@echo "            OUTPUT_DIR=outputs/evaluation"
