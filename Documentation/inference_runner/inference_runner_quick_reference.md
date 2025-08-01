# Inference Runner - Quick Reference

## Key Methods

| Method | Purpose | Parameters | Returns |
|--------|---------|------------|---------|
| `__init__()` | Initialize runner | `debug_mode`, `batch_size`, `timeout`, `max_retries` | None |
| `check_model_availability()` | Check if model exists | `model_name` | bool |
| `get_model_metadata()` | Get model info | `model_name` | Dict |
| `run_inference()` | Single prompt inference | `prompt`, `model_name` | Dict |
| `run_batch_inference()` | Batch inference | `prompts`, `model_name`, `row_ids` | List[Dict] |
| `process_dataset()` | Full dataset processing | `df`, `model_name`, `prompt_column` | DataFrame |
| `save_results()` | Save to file | `df`, `output_path`, `format` | None |
| `create_inference_metadata()` | Create metadata | `model_name`, `df`, `processing_time` | Dict |
| `save_metadata()` | Save metadata | `metadata`, `output_path` | None |

## Configuration Options

### Initialization Parameters
```python
runner = InferenceRunner(
    debug_mode=False,      # Enable verbose logging
    batch_size=5,          # Prompts per batch
    timeout=30,            # Seconds per call
    max_retries=3          # Retry attempts
)
```

### Output Formats
- `'csv'` - Comma-separated values
- `'json'` - JSON records format

## Model Availability Checking

```python
# Check if model is available
if runner.check_model_availability('llama2'):
    print("Model ready")
else:
    print("Model not found")

# Get model metadata
metadata = runner.get_model_metadata('llama2')
print(f"Size: {metadata['size']}")
print(f"Parameters: {metadata['parameters']}")
```

## Response Structure

### Expected Response Format
```python
{
    'prediction': 1,                    # Integer rating
    'rationale': 'Explanation text',    # String explanation
    'confidence': 0.85                  # Float 0.0-1.0
}
```

### Parsing Strategy
1. **JSON parsing** - Try to parse as structured JSON
2. **Regex parsing** - Extract fields using patterns
3. **Fallback** - Use full response as rationale

## Common Usage Patterns

### Basic Dataset Processing
```python
# Initialize and process
runner = InferenceRunner(debug_mode=True)
results_df = runner.process_dataset(df, 'llama2')

# Save results
runner.save_results(results_df, 'outputs/results.csv')
```

### Batch Processing with Custom Handling
```python
# Process in batches
prompts = df['formatted_prompt'].tolist()
row_ids = df['C_BioSense_ID'].tolist()
results = runner.run_batch_inference(prompts, 'llama2', row_ids)

# Convert to DataFrame
results_df = pd.DataFrame(results)
```

### Metadata Collection
```python
# Create metadata
processing_time = time.time() - start_time
metadata = runner.create_inference_metadata('llama2', results_df, processing_time)

# Save metadata
runner.save_metadata(metadata, 'outputs/metadata.json')
```

## Error Handling Reference

### Exception Types
- `ModelNotFoundError` - Model not available
- `ResponseParsingError` - Cannot parse response
- `InferenceError` - General inference failure

### Error Handling Pattern
```python
try:
    results_df = runner.process_dataset(df, 'llama2')
except ModelNotFoundError as e:
    print(f"Model not found: {e}")
except ResponseParsingError as e:
    print(f"Parse error: {e}")
except InferenceError as e:
    print(f"Inference failed: {e}")
```

### Retry Logic
- Automatic retry on failures
- Configurable retry count (`max_retries`)
- Exponential backoff between retries
- Graceful degradation for partial failures

## Performance Tips

### Batch Size Optimization
- **Small datasets** (< 50 rows): batch_size = 5-10
- **Medium datasets** (50-500 rows): batch_size = 10-20
- **Large datasets** (> 500 rows): batch_size = 20-50

### Timeout Settings
- **Fast models** (smaller): timeout = 15-30 seconds
- **Slow models** (larger): timeout = 60-120 seconds
- **Network issues**: increase timeout and retries

### Memory Management
- Batch processing reduces memory usage
- Results are processed incrementally
- Large datasets processed in chunks

## Integration Points

### Input Requirements
- DataFrame with `formatted_prompt` column
- Valid Ollama model name
- Optional `C_BioSense_ID` column for tracking

### Output Structure
- Original DataFrame + inference columns
- `prediction`, `rationale`, `confidence` columns
- Maintains row order consistency

### File Outputs
- Results CSV/JSON file
- Metadata JSON file
- Log file (`inference_runner.log`)

## Command Line Usage

```bash
# Basic usage
python scripts/inference_runner.py input.csv llama2

# With options
python scripts/inference_runner.py input.csv llama2 \
    --output-dir outputs \
    --batch-size 10 \
    --debug
```

## Testing

### Unit Tests
```bash
# Run all tests
pytest tests/test_inference_runner.py

# Run specific test
pytest tests/test_inference_runner.py::TestInferenceRunner::test_run_inference_success
```

### Demo Script
```bash
# Run demo
python demo_inference_runner.py
```

## Troubleshooting

### Common Issues
1. **Model not found** - Check Ollama installation and model availability
2. **Timeout errors** - Increase timeout or check model size
3. **Parse errors** - Check model response format
4. **Memory issues** - Reduce batch size

### Debug Mode
```python
runner = InferenceRunner(debug_mode=True)
# Enables verbose logging to console and file
``` 