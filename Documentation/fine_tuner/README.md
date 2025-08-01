# Fine Tuner Module

## Overview

The Fine Tuner module handles fine-tuning of Ollama-compatible models or creation of model-set configurations for meta-prompting when fine-tuning is not supported. This module provides a flexible approach to adapt models to dataset-specific rating and rationale patterns.

## Key Features

- **Fine-tuning Support**: Attempts to fine-tune Ollama models when supported
- **Fallback Mode**: Creates model-set configurations for meta-prompting when fine-tuning is unavailable
- **Dataset Validation**: Checks dataset size and quality for effective fine-tuning
- **Training Data Preparation**: Converts formatted prompts to JSONL format for training
- **Configuration Management**: Generates and saves training configurations
- **Model Validation**: Validates fine-tuned models after training
- **Comprehensive Logging**: Detailed logging for troubleshooting

## Architecture

The Fine Tuner module integrates with the existing pipeline:

```
formatted_dataset.csv → FineTuner → fine_tuned_model.gguf OR model_set_config.json
```

### Dependencies

- **Input**: Formatted dataset from `formatter_promptbuilder`
- **Output**: Fine-tuned model or model-set configuration
- **External**: Ollama CLI for model operations

## Usage

### Basic Usage

```python
from scripts.fine_tuner import FineTuner

# Initialize fine tuner
fine_tuner = FineTuner(debug_mode=True, fallback_mode=True)

# Process dataset
success, metadata = fine_tuner.process_dataset(
    df, 
    model_name="llama2",
    rating_mapping=rating_mapping,
    target_topics="respiratory infections",
    output_dir="outputs"
)
```

### Command Line Usage

```bash
python scripts/fine_tuner.py --input outputs/formatted_dataset.csv \
                            --model llama2 \
                            --topics "respiratory infections" \
                            --output-dir outputs \
                            --epochs 3 \
                            --learning-rate 0.0001 \
                            --batch-size 4
```

## Configuration

### Fine-tuning Parameters

- **epochs**: Number of training epochs (default: 3)
- **learning_rate**: Learning rate for training (default: 0.0001)
- **batch_size**: Batch size for training (default: 4)
- **fallback_mode**: Enable fallback to model-set configuration (default: True)

### Dataset Requirements

- **Minimum Size**: 50 rows recommended for fine-tuning
- **Format**: Must include formatted prompts and standardized ratings
- **Quality**: Clean, validated data from previous pipeline stages

## Output Files

### Fine-tuning Approach

- `training_data.jsonl`: Training examples in JSONL format
- `training_config.yaml`: Training configuration file
- `fine_tuning_metadata.json`: Processing metadata and results

### Model-set Configuration Approach

- `model_set_config.json`: Configuration for meta-prompting
- `fine_tuning_metadata.json`: Processing metadata and results

## Error Handling

The module handles various error conditions:

- **FineTuningNotSupportedError**: When base model cannot be fine-tuned
- **DatasetTooSmallWarning**: When dataset is below recommended size
- **TrainingFailedError**: When fine-tuning process fails
- **ModelSetConfigError**: When model-set configuration creation fails

## Integration

The Fine Tuner module is designed to work seamlessly with the existing pipeline:

1. **Input**: Uses formatted dataset from `formatter_promptbuilder`
2. **Processing**: Adapts model to dataset-specific patterns
3. **Output**: Provides fine-tuned model or configuration for `inference_runner`

## Testing

Run the test suite:

```bash
pytest tests/test_fine_tuner.py -v
```

## Demo

Run the demo script:

```bash
python demo_fine_tuner.py
```

This will demonstrate the fine-tuning process using the existing formatted dataset. 