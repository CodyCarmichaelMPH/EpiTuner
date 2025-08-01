# Fine Tuner Quick Reference

## Class: FineTuner

### Initialization
```python
fine_tuner = FineTuner(debug_mode=False, fallback_mode=True)
```

### Key Methods

#### `check_fine_tuning_capability(model_name: str) -> bool`
Check if Ollama model supports fine-tuning.

#### `prepare_training_data(df: pd.DataFrame, rating_mapping: Dict) -> List[Dict]`
Convert formatted dataset to JSONL training format.

#### `process_dataset(df, model_name, rating_mapping, target_topics, **kwargs) -> Tuple[bool, Dict]`
Main method to process dataset for fine-tuning or model set configuration.

#### `create_model_set_config(model_name, df, rating_mapping, target_topics) -> Dict`
Create model set configuration for meta-prompting approach.

## Configuration Parameters

### Fine-tuning Parameters
- `epochs`: Training epochs (default: 3)
- `learning_rate`: Learning rate (default: 0.0001)
- `batch_size`: Batch size (default: 4)

### Dataset Requirements
- **Minimum size**: 50 rows recommended
- **Required columns**: `formatted_prompt` or `Prompt`, `Standardized_Rating`
- **Optional columns**: `Rationale of Rating`

## Output Files

### Fine-tuning Approach
```
outputs/
├── training_data.jsonl          # Training examples
├── training_config.yaml         # Training configuration
└── fine_tuning_metadata.json    # Processing metadata
```

### Model Set Configuration Approach
```
outputs/
├── model_set_config.json        # Meta-prompting configuration
└── fine_tuning_metadata.json    # Processing metadata
```

## Error Types

- `FineTuningNotSupportedError`: Model cannot be fine-tuned
- `DatasetTooSmallWarning`: Dataset below recommended size
- `TrainingFailedError`: Fine-tuning process failed
- `ModelSetConfigError`: Configuration creation failed

## Usage Examples

### Basic Fine-tuning
```python
from scripts.fine_tuner import FineTuner

fine_tuner = FineTuner(debug_mode=True)
success, metadata = fine_tuner.process_dataset(
    df, "llama2", rating_mapping, "respiratory infections"
)
```

### Model Set Configuration Only
```python
fine_tuner = FineTuner(fallback_mode=True)
config = fine_tuner.create_model_set_config(
    "llama2", df, rating_mapping, "cardiac conditions"
)
```

### Command Line
```bash
python scripts/fine_tuner.py \
    --input outputs/formatted_dataset.csv \
    --model llama2 \
    --topics "respiratory infections" \
    --epochs 5 \
    --learning-rate 0.001
```

## Integration

### Input Dependencies
- Formatted dataset from `formatter_promptbuilder`
- Rating mapping from `schema_mapper`

### Output Usage
- Fine-tuned model for `inference_runner`
- Model set config for `contextualizer`

## Testing

```bash
# Run all tests
pytest tests/test_fine_tuner.py -v

# Run specific test
pytest tests/test_fine_tuner.py::TestFineTuner::test_process_dataset_fine_tuning_approach -v
```

## Demo

```bash
python demo_fine_tuner.py
```

## Logging

### Log File
- `fine_tuner.log`

### Log Levels
- **DEBUG**: Training progress, data preparation details
- **INFO**: Process milestones, file operations
- **WARNING**: Dataset size issues, fallback usage
- **ERROR**: Training failures, configuration errors

## Performance Notes

### Fine-tuning
- **Memory**: High usage during training
- **Time**: Scales with dataset size and epochs
- **Storage**: Requires space for model files

### Model Set Configuration
- **Memory**: Low usage
- **Time**: Fast generation
- **Storage**: Minimal requirements 