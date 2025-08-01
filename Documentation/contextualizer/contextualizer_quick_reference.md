# Contextualizer Quick Reference

## Quick Start

```python
from scripts.contextualizer import Contextualizer
from scripts.data_loader import DataLoader
from scripts.schema_mapper import SchemaMapper

# Initialize
loader = DataLoader(debug_mode=True)
mapper = SchemaMapper(debug_mode=True)
contextualizer = Contextualizer(debug_mode=True, max_rows_context=10)

# Load and process data
df, unique_ratings, _ = loader.process_dataset("data/sample_dataset.csv")
rating_mapping = mapper.suggest_mapping(unique_ratings)
df_mapped, _ = mapper.process_mapping(df, rating_mapping)

# Evaluate single row
result = contextualizer.evaluate_single_row(
    df_mapped, 0, "respiratory infections", rating_mapping, "llama2"
)

# Process entire dataset
results_df, metadata = contextualizer.process_dataset(
    df_mapped, "respiratory infections", rating_mapping, "llama2", "outputs"
)
```

## Key Methods

### Core Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `evaluate_single_row(df, row_index, topics, rating_mapping, model_name)` | Evaluate one row | Dict with prediction, rationale |
| `process_dataset(df, topics, rating_mapping, model_name, output_dir)` | Process entire dataset | (DataFrame, Dict) |
| `sample_few_shot_examples(df, rating_mapping, topics)` | Select examples for context | DataFrame |
| `construct_meta_prompt(few_shot_df, query_row, topics, rating_mapping)` | Build prompt | String |

### Utility Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `check_model_availability(model_name)` | Check if model exists | Boolean |
| `build_schema_description(rating_mapping)` | Create schema text | String |
| `format_example_row(row, example_num)` | Format example for prompt | String |

## Configuration

### Settings File (`config/settings.json`)

```json
{
  "contextualizer": {
    "max_rows_context": 10,
    "timeout": 30,
    "max_retries": 3,
    "default_prompt_template": null
  }
}
```

### Command Line

```bash
# Run contextualizer directly
python scripts/contextualizer.py --input data.csv --topics "respiratory infections" --model phi3:mini

# Run demo
python demo_contextualizer.py
```

## Output Format

### Single Row Result

```json
{
  "C_BioSense_ID": "patient_001",
  "prediction": 1,
  "confidence": null,
  "rationale": "Clear respiratory symptoms present in diagnosis"
}
```

### Batch Results

- **CSV File**: `outputs/contextual_evaluation_results.csv`
- **Metadata**: `outputs/contextual_evaluation_metadata.json`

### Metadata Structure

```json
{
  "evaluation_type": "contextual",
  "model_name": "phi3:mini",
  "topics": "respiratory infections",
  "success_rate": 0.95,
  "processing_time_seconds": 45.2,
  "prediction_distribution": {"0": 10, "1": 15, "2": 5}
}
```

## Common Patterns

### Basic Evaluation

```python
# Evaluate one row
result = contextualizer.evaluate_single_row(
    df_mapped, 0, topics, rating_mapping, "phi3:mini"
)
print(f"Prediction: {result['prediction']}")
print(f"Rationale: {result['rationale']}")
```

### Batch Processing

```python
# Process entire dataset
results_df, metadata = contextualizer.process_dataset(
    df_mapped, topics, rating_mapping, "phi3:mini", "outputs"
)

# Check results
print(f"Success rate: {metadata['success_rate']:.2%}")
print(f"Processing time: {metadata['processing_time_seconds']:.2f}s")
```

### Custom Configuration

```python
# Initialize with custom settings
contextualizer = Contextualizer(
    debug_mode=True,
    max_rows_context=15,  # More examples
    timeout=60,           # Longer timeout
    max_retries=5         # More retries
)
```

### Error Handling

```python
try:
    result = contextualizer.evaluate_single_row(
        df_mapped, 0, topics, rating_mapping, "llama2"
    )
except ContextualizerError as e:
    print(f"Contextualizer error: {e}")
except ModelResponseError as e:
    print(f"Model response error: {e}")
```

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Model not found | Ollama model not installed | `ollama pull model_name` |
| Timeout errors | Model too slow | Increase timeout in config |
| Parsing errors | Malformed response | Check model output format |
| Low success rate | Poor examples | Improve data quality |

### Debug Mode

```python
contextualizer = Contextualizer(debug_mode=True)
# Provides detailed logging for troubleshooting
```

### Model Availability Check

```python
if contextualizer.check_model_availability("phi3:mini"):
    print("Model available")
else:
    print("Model not found - install with: ollama pull phi3:mini")
```

## Performance Tips

### Optimization

- **Small datasets** (< 1000 rows): Use contextualizer directly
- **Large datasets**: Consider fine-tuning instead
- **Context size**: Start with 5-10 examples, adjust based on performance
- **Model selection**: llama2 for speed, mistral for accuracy

### Memory Management

```python
# Process in batches for large datasets
batch_size = 100
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    results = contextualizer.process_dataset(batch, topics, rating_mapping, "llama2")
```

## Integration Examples

### With Existing Pipeline

```python
# Complete pipeline with fallback
from scripts.data_loader import DataLoader
from scripts.schema_mapper import SchemaMapper
from scripts.formatter_promptbuilder import PromptBuilder
from scripts.contextualizer import Contextualizer

# Process data
loader = DataLoader(debug_mode=True)
mapper = SchemaMapper(debug_mode=True)
df, unique_ratings, _ = loader.process_dataset("data.csv")
rating_mapping = mapper.suggest_mapping(unique_ratings)
df_mapped, _ = mapper.process_mapping(df, rating_mapping)

# Choose approach based on dataset size
if len(df_mapped) < 1000:
    # Use contextualizer for small datasets
    contextualizer = Contextualizer(debug_mode=True)
    results_df, metadata = contextualizer.process_dataset(
        df_mapped, topics, rating_mapping, "llama2", "outputs"
    )
else:
    # Use fine-tuning for large datasets
    formatter = PromptBuilder(debug_mode=True)
    formatted_df, _ = formatter.process_dataset(df_mapped, topics, rating_mapping, "outputs")
```

### Custom Prompt Templates

```python
# Use custom prompt template
custom_template = """
You are a medical expert evaluating patient records.
Focus on: {topics}

Rating schema:
{rating_schema}

Examples:
{examples}

Evaluate this record: {query_record}

Provide rating and rationale.
"""

# Apply custom template in contextualizer
```

## Testing

### Run Tests

```bash
# Run contextualizer tests
python -m pytest tests/test_contextualizer.py -v

# Run specific test
python -m pytest tests/test_contextualizer.py::TestContextualizer::test_sample_few_shot_examples -v
```

### Test Data

```python
# Create test data
test_data = pd.DataFrame({
    'C_BioSense_ID': ['test_001', 'test_002'],
    'ChiefComplaintOrig': ['Chest pain', 'Headache'],
    'Discharge Diagnosis': ['Angina', 'Migraine'],
    'Expert Rating': [1, 0],
    'Rationale of Rating': ['Cardiac symptoms', 'Neurological']
})
```

## Best Practices

### Data Quality

- Ensure expert ratings are consistent
- Provide clear rationales for examples
- Balance examples across rating categories
- Clean and validate input data

### Model Selection

- **phi3:mini**: Good balance of speed and accuracy
- **llama2**: Good balance of speed and accuracy
- **mistral**: Better reasoning, slower inference
- **codellama**: Good for structured data
- **qwen2**: Strong on medical text

### Prompt Design

- Keep topics specific and concise
- Use clear, unambiguous rating schema
- Include diverse examples
- Monitor success rates and adjust

### Error Handling

- Always check model availability
- Implement retry logic for failures
- Log errors for debugging
- Provide fallback options 