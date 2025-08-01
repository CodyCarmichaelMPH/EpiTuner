# Contextualizer Module

## Overview

The Contextualizer module provides a fallback approach for scenarios where fine-tuning is not available, underperforms, or is not cost-effective. It builds structured meta-prompts using dataset rows, expert ratings, rationales, and schema mapping to guide the base Ollama model to make improved predictions.

## Key Features

- **No Fine-tuning Required**: Works with any base Ollama model without training
- **Few-shot Learning**: Uses examples from your dataset to guide predictions
- **Interpretable Results**: Provides rationales for each prediction
- **Cost-effective**: Ideal for small to medium datasets
- **Fallback Option**: Can be used when fine-tuning fails or is too expensive
- **Balanced Sampling**: Automatically selects diverse examples across rating categories

## Architecture

```
Dataset → Few-shot Sampling → Meta-prompt Construction → Ollama Inference → Structured Results
```

### Components

1. **Data Sampling**: Selects representative examples from the dataset
2. **Context Construction**: Builds meta-prompts with schema, examples, and query
3. **Execution**: Sends prompts to Ollama model via CLI
4. **Output**: Returns structured predictions with rationales

## Usage

### Basic Usage

```python
from scripts.contextualizer import Contextualizer
from scripts.data_loader import DataLoader
from scripts.schema_mapper import SchemaMapper

# Initialize components
loader = DataLoader(debug_mode=True)
mapper = SchemaMapper(debug_mode=True)
contextualizer = Contextualizer(debug_mode=True, max_rows_context=10)

# Load and process data
df, unique_ratings, _ = loader.process_dataset("data/sample_dataset.csv")
rating_mapping = mapper.suggest_mapping(unique_ratings)
df_mapped, _ = mapper.process_mapping(df, rating_mapping)

# Define topics
topics = "respiratory infections, cardiac conditions, and neurological disorders"

# Evaluate a single row
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

print(f"Success rate: {metadata['success_rate']:.2%}")
print(f"Processing time: {metadata['processing_time_seconds']:.2f} seconds")
```

## Configuration

The contextualizer can be configured through the `config/settings.json` file:

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

### Configuration Options

- **max_rows_context**: Maximum number of example rows to include in context (default: 10)
- **timeout**: Timeout for API calls in seconds (default: 30)
- **max_retries**: Maximum number of retry attempts for failed calls (default: 3)
- **default_prompt_template**: Optional custom prompt template (default: null)

## Meta-prompt Structure

The contextualizer constructs prompts with the following structure:

```
You are evaluating patient record alignment with topics: [topics].
Use this rating schema:
- Rating 0: [meaning]
- Rating 1: [meaning]
- Rating 2: [meaning]

Examples:
1. Record: [context] → Rating: [rating] | Rationale: [rationale]
2. Record: [context] → Rating: [rating] | Rationale: [rationale]
...

New record to evaluate:
[query_record_context]

Respond with:
- Numeric rating (use the schema above)
- Brief rationale explaining your decision

Format your response as:
Rating: [number]
Rationale: [explanation]
```

## Output Format

The contextualizer returns structured results in the following format:

```json
{
  "C_BioSense_ID": "patient_001",
  "prediction": 1,
  "confidence": null,
  "rationale": "Clear respiratory symptoms present in diagnosis"
}
```

### Output Fields

- **C_BioSense_ID**: Patient identifier from the dataset
- **prediction**: Numeric rating based on the schema
- **confidence**: Always null (contextualizer doesn't provide confidence scores)
- **rationale**: Explanation for the prediction

## Integration with Pipeline

The contextualizer integrates seamlessly with the existing EpiTuner pipeline:

```python
# Complete pipeline with contextualizer fallback
from scripts.data_loader import DataLoader
from scripts.schema_mapper import SchemaMapper
from scripts.formatter_promptbuilder import PromptBuilder
from scripts.contextualizer import Contextualizer

# Initialize all components
loader = DataLoader(debug_mode=True)
mapper = SchemaMapper(debug_mode=True)
formatter = PromptBuilder(debug_mode=True)
contextualizer = Contextualizer(debug_mode=True)

# Process data through pipeline
df, unique_ratings, _ = loader.process_dataset("data/sample_dataset.csv")
rating_mapping = mapper.suggest_mapping(unique_ratings)
df_mapped, _ = mapper.process_mapping(df, rating_mapping)

# Option 1: Use formatter for fine-tuning
formatted_df, _ = formatter.process_dataset(df_mapped, topics, rating_mapping, "outputs")

# Option 2: Use contextualizer as fallback
results_df, metadata = contextualizer.process_dataset(
    df_mapped, topics, rating_mapping, "llama2", "outputs"
)
```

## Error Handling

The contextualizer includes comprehensive error handling:

- **Model Availability**: Checks if the specified Ollama model is available
- **Response Parsing**: Handles malformed model responses with retry logic
- **Missing Data**: Gracefully handles rows without expert ratings or rationales
- **Timeout Handling**: Manages long-running inference requests
- **Logging**: Comprehensive logging for debugging and monitoring

## Performance Considerations

### Context Size Management

- Automatically limits context size to avoid token limits
- Balances examples across rating categories
- Prioritizes rows with clear expert ratings and rationales

### Processing Speed

- Sequential processing to avoid overwhelming the model
- Configurable timeouts and retry limits
- Progress logging for long-running operations

### Memory Usage

- Processes one row at a time to minimize memory footprint
- Efficient DataFrame operations
- Cleanup of temporary data structures

## Best Practices

### When to Use Contextualizer

- **Small datasets** (< 1000 rows)
- **Quick prototyping** and hypothesis testing
- **Fine-tuning alternatives** when cost is a concern
- **Fallback option** when fine-tuning fails
- **Interpretability requirements** where rationales are needed

### Model Selection

- **phi3:mini**: Good balance of performance and speed
- **llama2**: Good balance of performance and speed
- **mistral**: Better reasoning capabilities
- **codellama**: Specialized for code-like structured data
- **qwen2**: Good performance on medical text

### Prompt Optimization

- Keep topics concise and specific
- Ensure rating schema is clear and unambiguous
- Use diverse examples across rating categories
- Monitor success rates and adjust context size

## Troubleshooting

### Common Issues

1. **Model Not Found**
   - Ensure Ollama is installed and running
   - Check model name spelling
   - Pull the model: `ollama pull model_name`

2. **Timeout Errors**
   - Increase timeout in configuration
   - Reduce max_rows_context
   - Check system resources

3. **Parsing Errors**
   - Check model response format
   - Verify rating schema consistency
   - Review example quality

4. **Low Success Rate**
   - Improve example quality
   - Adjust rating schema clarity
   - Try different model
   - Increase context size

### Debug Mode

Enable debug mode for detailed logging:

```python
contextualizer = Contextualizer(debug_mode=True)
```

This provides:
- Detailed prompt construction logs
- Model response debugging
- Performance metrics
- Error tracebacks

## Examples

See the `demo_contextualizer.py` script for complete working examples.

## Testing

Run the test suite:

```bash
python -m pytest tests/test_contextualizer.py -v
```

## Dependencies

- pandas
- requests
- subprocess
- logging
- pathlib
- typing

## License

Part of the EpiTuner suite - see main README for license information. 