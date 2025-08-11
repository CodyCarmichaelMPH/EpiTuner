# EpiTuner

**LoRA Fine-tuning Tool for Medical Data Classification**

EpiTuner is a user-friendly application for fine-tuning language models on medical data using LoRA (Low-Rank Adaptation). It provides a complete workflow from data upload to model deployment, with a focus on PHI-safe local processing and expert-in-the-loop validation.

## Features

- **Completely Local Processing** - All data stays on your machine (PHI-safe)
- **Expert-in-the-Loop** - Review and correct model predictions
- **Confidence Scoring** - Models provide confidence levels for predictions
- **Easy Ollama Integration** - Deploy trained models with Ollama
- **Memory Efficient** - QLoRA training on consumer GPUs
- **Comprehensive Evaluation** - Detailed metrics and visualizations
- **User-Friendly GUI** - Streamlit-based interface

## Quick Start

### 1. Installation

**Windows:**
```powershell
# Clone the repository
git clone https://github.com/CodyCarmichaelMPH/EpiTuner.git
cd EpiTuner

# Install dependencies and setup
.\setup.ps1 install
.\setup.ps1 setup

# Check dependencies
.\setup.ps1 check-deps
```

**Linux/macOS:**
```bash
# Clone the repository
git clone https://github.com/CodyCarmichaelMPH/EpiTuner.git
cd EpiTuner

# Install dependencies
make install

# Create directory structure
make setup

# Check dependencies
make check-deps
```

### 2. Run the Application

**Windows:**
```powershell
.\setup.ps1 run-app
```

**Linux/macOS:**
```bash
# Start the Streamlit GUI
make run-app
```

Open your browser to `http://localhost:8501` to access the EpiTuner interface.

### 3. Using the GUI

1. **Data Upload** - Upload your CSV file with medical records
2. **Model Selection** - Choose from your local Ollama models
3. **Configuration** - Set training parameters and confidence thresholds
4. **Training** - Fine-tune your LoRA model
5. **Expert Review** - Review and correct model predictions
6. **Export** - Download your trained model and results

## Data Format

Your CSV file should include these columns:

### Required Columns
- `C_Biosense_ID` - Unique identifier for each record
- `ChiefComplaintOrig` - Chief complaints (separated by ;)
- `DischargeDiagnosis` - Discharge diagnosis
- `Expert Rating` - "Match", "Not a Match", or "Unknown/Not able to determine"
- `Rationale_of_Rating` - Expert's reasoning for the rating

### Optional but Recommended
- `Sex` - Patient sex
- `Age` - Patient age
- `c_ethnicity` - Ethnicity
- `c_race` - Race
- `Admit_Reason_Combo` - Admission reasons
- `Diagnosis_Combo` - Diagnosis codes with descriptors
- `CCDD Category` - CCDD categories (separated by ;)
- `TriageNotes` or `TriageNotesOrig` - Triage notes

### Sample Data

A sample dataset is provided in `sample_data/medical_sample.csv` for testing purposes.

## Model Support

EpiTuner works with models available in your local Ollama installation:

### Recommended Models
- **llama2** - Good general performance
- **mistral** - Efficient and accurate
- **phi** - Fast training, good for testing
- **qwen** - Strong medical domain performance

### Install Ollama Models

```bash
# Install Ollama (visit https://ollama.ai)
ollama pull llama2
ollama pull mistral
ollama pull phi
```

## Command Line Usage

For advanced users, EpiTuner provides command-line tools:

### Training

```bash
make train DATA=sample_data/medical_sample.csv \
           MODEL=microsoft/phi-2 \
           TOPIC="motor vehicle collision detection" \
           OUTPUT=outputs/mvc_model
```

### Inference

```bash
make infer MODEL=outputs/mvc_model \
           CONFIG=configs/config_base.yaml \
           DATA=new_data.csv \
           TOPIC="motor vehicle collision detection" \
           OUTPUT=outputs/predictions.json
```

### Evaluation

```bash
make eval PREDICTIONS=outputs/predictions.json \
          GROUND_TRUTH=sample_data/medical_sample.csv \
          OUTPUT_DIR=outputs/evaluation
```

## Project Structure

```
EpiTuner/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── Makefile                   # Automation commands
├── configs/
│   └── config_base.yaml       # Training configuration
├── scripts/
│   ├── train.py               # Training script
│   ├── inference.py           # Inference script
│   └── evaluate.py            # Evaluation script
├── chat_templates/
│   └── medical_classification.jinja  # Chat template
├── sample_data/
│   └── medical_sample.csv     # Sample dataset
├── outputs/                   # Training outputs
├── data/                      # User data
└── adapters/                  # LoRA adapters
```

## Configuration

### Training Parameters

Key configuration options in `configs/config_base.yaml`:

```yaml
model:
  max_seq_len: 512             # Maximum sequence length

train:
  learning_rate: 2e-4          # Learning rate
  num_epochs: 3                # Number of training epochs
  batch_size: "auto"           # Automatic batch sizing

tuning:
  mode: "qlora"                # LoRA mode (qlora/lora/full)
  lora_r: 16                   # LoRA rank
  lora_alpha: 32               # LoRA alpha
  lora_dropout: 0.1            # LoRA dropout

confidence:
  threshold: 0.7               # Auto-approval threshold
```

### Confidence Levels

The system provides five confidence levels:
- **Very Confident** - Model is highly certain
- **Confident** - Model is reasonably certain  
- **Somewhat Confident** - Model has moderate certainty
- **Not Very Confident** - Model has low certainty
- **Not at all Confident** - Model is very uncertain

## Ollama Integration

Once training is complete, integrate your model with Ollama:

### 1. Save LoRA Adapter

Download the LoRA adapter from the GUI or copy from `outputs/`.

### 2. Create Modelfile

```dockerfile
FROM llama2
ADAPTER ./adapter_model.safetensors

TEMPLATE """{{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }}"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
```

### 3. Create Ollama Model

```bash
ollama create epituner-medical -f ./Modelfile
```

### 4. Use Your Model

```bash
# Interactive
ollama run epituner-medical "Classify this medical record..."

# API
curl http://localhost:11434/api/generate \
  -d '{"model": "epituner-medical", "prompt": "Your medical record here..."}'
```

## Evaluation Metrics

EpiTuner provides comprehensive evaluation metrics:

### Classification Metrics
- **Accuracy** - Overall correctness
- **Precision** - True positives / (True positives + False positives)
- **Recall** - True positives / (True positives + False negatives)
- **F1-Score** - Harmonic mean of precision and recall

### Confidence Analysis
- **Calibration** - How well confidence scores match actual accuracy
- **Distribution** - Breakdown of predictions by confidence level
- **Agreement Rate** - Model-expert agreement by confidence

### Visualizations
- Confusion matrices
- Confidence calibration plots
- Performance trends

## Troubleshooting

### Common Issues

#### CUDA Out of Memory (Common on Consumer GPUs)
```bash
# Step 1: Use consumer GPU config
CONFIG=configs/config_consumer_gpu.yaml

# Step 2: Reduce LoRA rank
lora_r: 4  # Instead of 8 or 16

# Step 3: Use smaller model
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Step 4: Close other applications
# - Close Chrome/browsers
# - Close games or GPU-heavy apps
# - Check Task Manager GPU usage
```

#### Ollama Models Not Found
```bash
# Check Ollama installation
make check-ollama

# Install models
ollama pull llama2
```

#### Training Slow (Windows Consumer GPU)
```bash
# Use consumer GPU optimized config
CONFIG=configs/config_consumer_gpu.yaml

# Use smaller dataset for testing
head -20 your_data.csv > small_data.csv

# Optimize for consumer GPU
num_epochs: 1
lora_r: 4
max_seq_len: 128

# Windows-specific optimization
dataloader_num_workers: 0  # Avoid multiprocessing issues
```

### System Requirements

**Windows 10/11 with Consumer GPU Setup:**
- **GPU**: NVIDIA GTX 1660 Ti / RTX 3060 / RTX 4060 or better
  - 4GB VRAM minimum (TinyLlama only)
  - 6-8GB VRAM recommended (Phi-2, small models)
  - 12GB+ VRAM ideal (larger models)
- **RAM**: 16GB+ system RAM
- **Storage**: 10GB+ free space for models and outputs
- **Python**: 3.8+ (3.11 recommended)
- **CUDA**: NVIDIA drivers 11.8+ (download from nvidia.com)

## Advanced Usage

### Custom Chat Templates

Modify `chat_templates/medical_classification.jinja` to customize how data is formatted for training:

```jinja
Medical Record Classification

Task: {{ classification_topic }}

Record: {{ chief_complaint }}
Diagnosis: {{ discharge_diagnosis }}

Classification: {{ classification }}
Confidence: {{ confidence }}
Rationale: {{ rationale }}
```

### Custom Configurations

Create new configuration files in `configs/` for different use cases:

```yaml
# configs/rapid_training.yaml
train:
  num_epochs: 1
  learning_rate: 5e-4

tuning:
  lora_r: 8
  lora_alpha: 16
```

### Batch Processing

Process multiple datasets programmatically:

```python
from scripts.train import LoRATrainer
from scripts.inference import MedicalClassificationInference

# Train multiple models
datasets = ['mvc_data.csv', 'fall_data.csv', 'cardiac_data.csv']
for dataset in datasets:
    # Training logic here
    pass
```

## Contributing

We welcome contributions! Please see our contribution guidelines for details.

### Development Setup (Windows)

```bash
# Install development dependencies
make dev-install

# Check Windows environment
make windows-check

# Setup Windows directories
make windows-setup

# Format code
make format

# Run linting
make lint
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:

1. Check the troubleshooting section above
2. Review the sample data and configurations
3. Open an issue on GitHub
4. Contact the development team

## Roadmap

### Version 0.2
- [ ] Support for additional model architectures
- [ ] Automated hyperparameter tuning
- [ ] Multi-label classification support
- [ ] Integration with more deployment platforms

### Version 0.3
- [ ] Federated learning capabilities
- [ ] Advanced active learning strategies
- [ ] Real-time model monitoring
- [ ] Advanced data preprocessing tools

---

**EpiTuner** - Making medical AI accessible, reliable, and privacy-preserving.

*Built for the medical AI community*
