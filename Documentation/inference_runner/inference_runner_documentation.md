# Inference Runner Module - Complete Documentation

## Overview

The Inference Runner module (`scripts/inference_runner.py`) is responsible for executing inference on formatted dataset prompts using Ollama models. It provides a robust interface for running batch inference, parsing model responses, and managing the complete inference pipeline.

## Class Structure

### InferenceRunner

The main class that handles all inference operations.

**Location**: `scripts/inference_runner.py`

**Dependencies**:
- `pandas` for data manipulation
- `subprocess` for Ollama CLI integration
- `json` for response parsing
- `re` for regex-based parsing
- `logging` for structured logging

## Custom Exception Classes

### ModelNotFoundError
Raised when the specified Ollama model is not found or not available.

### ResponseParsingError
Raised when model response cannot be parsed into required fields (prediction, rationale, confidence).

### InferenceError
Raised when inference operations fail due to system errors, timeouts, or other issues.

## Public Methods

### `__init__(debug_mode=False, batch_size=5, timeout=30, max_retries=3)`

**Purpose**: Initialize the InferenceRunner instance.

**Parameters**:
- `debug_mode` (bool): Enable verbose logging for debugging
- `batch_size` (int): Number of prompts to process in each batch
- `timeout` (int): Timeout for API calls in seconds
- `max_retries` (int): Maximum number of retry attempts for failed calls

**Returns**: None

**Example**:
```python
runner = InferenceRunner(debug_mode=True, batch_size=10, timeout=60)
```

### `check_model_availability(model_name)`

**Purpose**: Check if the specified Ollama model is available.

**Parameters**:
- `model_name` (str): Name of the Ollama model to check

**Returns**: bool - True if model is available, False otherwise

**Behavior**:
- Executes `ollama list` command
- Searches for model name in the output
- Handles various error conditions gracefully

**Example**:
```python
if runner.check_model_availability('llama2'):
    print("Model is available")
else:
    print("Model not found")
```

### `get_model_metadata(model_name)`

**Purpose**: Get metadata for the specified model.

**Parameters**:
- `model_name` (str): Name of the Ollama model

**Returns**: Dict[str, Any] - Dictionary containing model metadata

**Raises**: ModelNotFoundError if model is not found

**Metadata Structure**:
```python
{
    'name': 'llama2',
    'available': True,
    'size': '3.8GB',
    'parameters': '7B'
}
```

**Example**:
```python
metadata = runner.get_model_metadata('llama2')
print(f"Model size: {metadata['size']}")
```

### `run_inference(prompt, model_name)`

**Purpose**: Run inference on a single prompt using the specified model.

**Parameters**:
- `prompt` (str): Formatted prompt string
- `model_name` (str): Name of the Ollama model to use

**Returns**: Dict[str, Any] - Dictionary containing prediction, rationale, and confidence

**Raises**: 
- InferenceError if inference fails
- ResponseParsingError if response cannot be parsed

**Response Structure**:
```python
{
    'prediction': 1,
    'rationale': 'Symptoms align strongly with target condition.',
    'confidence': 0.85
}
```

**Example**:
```python
result = runner.run_inference('Analyze this medical case...', 'llama2')
print(f"Prediction: {result['prediction']}")
```

### `run_batch_inference(prompts, model_name, row_ids=None)`

**Purpose**: Run inference on a batch of prompts.

**Parameters**:
- `prompts` (List[str]): List of formatted prompt strings
- `model_name` (str): Name of the Ollama model to use
- `row_ids` (Optional[List[str]]): Optional list of row IDs to include in results

**Returns**: List[Dict[str, Any]] - List of result dictionaries

**Behavior**:
- Processes prompts in batches
- Handles errors gracefully for individual prompts
- Maintains order consistency with input data

**Example**:
```python
prompts = ['prompt1', 'prompt2', 'prompt3']
row_ids = ['row1', 'row2', 'row3']
results = runner.run_batch_inference(prompts, 'llama2', row_ids)
```

### `process_dataset(df, model_name, prompt_column='formatted_prompt')`

**Purpose**: Process entire dataset through inference.

**Parameters**:
- `df` (pd.DataFrame): DataFrame with formatted prompts
- `model_name` (str): Name of the Ollama model to use
- `prompt_column` (str): Name of column containing formatted prompts

**Returns**: pd.DataFrame - DataFrame with inference results added

**Raises**:
- ModelNotFoundError if model is not available
- ValueError if prompt column is missing

**Behavior**:
- Validates model availability
- Processes data in batches
- Merges results with original DataFrame
- Maintains row order consistency

**Example**:
```python
results_df = runner.process_dataset(df, 'llama2', 'formatted_prompt')
```

### `save_results(df, output_path, format='csv')`

**Purpose**: Save inference results to file.

**Parameters**:
- `df` (pd.DataFrame): DataFrame with inference results
- `output_path` (str): Path to save results
- `format` (str): Output format ('csv' or 'json')

**Returns**: None

**Raises**: ValueError if format is not supported

**Example**:
```python
runner.save_results(results_df, 'outputs/results.csv', 'csv')
runner.save_results(results_df, 'outputs/results.json', 'json')
```

### `create_inference_metadata(model_name, df, processing_time)`

**Purpose**: Create metadata for the inference run.

**Parameters**:
- `model_name` (str): Name of the model used
- `df` (pd.DataFrame): DataFrame with results
- `processing_time` (float): Total processing time in seconds

**Returns**: Dict[str, Any] - Dictionary containing inference metadata

**Metadata Structure**:
```python
{
    'model_name': 'llama2',
    'model_metadata': {...},
    'total_rows': 100,
    'processing_time_seconds': 45.2,
    'batch_size': 5,
    'timestamp': '2024-01-15T10:30:00',
    'columns_in_output': ['C_BioSense_ID', 'prediction', ...],
    'prediction_stats': {...}
}
```

**Example**:
```python
metadata = runner.create_inference_metadata('llama2', results_df, 45.2)
```

### `save_metadata(metadata, output_path)`

**Purpose**: Save inference metadata to file.

**Parameters**:
- `metadata` (Dict[str, Any]): Metadata dictionary
- `output_path` (str): Path to save metadata

**Returns**: None

**Example**:
```python
runner.save_metadata(metadata, 'outputs/metadata.json')
```

## Private Methods

### `_setup_logging()`

**Purpose**: Setup logging configuration.

**Behavior**:
- Configures logging level based on debug_mode
- Sets up both console and file handlers
- Creates logger instance

### `_extract_model_size(show_output)`

**Purpose**: Extract model size from ollama show output.

**Parameters**:
- `show_output` (str): Output from `ollama show` command

**Returns**: Optional[str] - Model size or None if not found

### `_extract_parameters(show_output)`

**Purpose**: Extract parameter count from ollama show output.

**Parameters**:
- `show_output` (str): Output from `ollama show` command

**Returns**: Optional[str] - Parameter count or None if not found

### `_parse_response(response)`

**Purpose**: Parse model response to extract prediction, rationale, and confidence.

**Parameters**:
- `response` (str): Raw model response string

**Returns**: Dict[str, Any] - Dictionary with parsed fields

**Raises**: ResponseParsingError if response cannot be parsed

**Parsing Strategy**:
1. Try to parse as JSON first
2. Fall back to regex-based parsing
3. Extract prediction, rationale, and confidence separately

### `_extract_prediction(response)`

**Purpose**: Extract numeric prediction from response.

**Parameters**:
- `response` (str): Model response string

**Returns**: Optional[int] - Numeric prediction or None

**Strategy**: Uses regex to find numeric values that could be predictions.

### `_extract_rationale(response)`

**Purpose**: Extract rationale text from response.

**Parameters**:
- `response` (str): Model response string

**Returns**: Optional[str] - Rationale text or None

**Strategy**: Looks for text after common rationale indicators or returns full response.

### `_extract_confidence(response)`

**Purpose**: Extract confidence score from response.

**Parameters**:
- `response` (str): Model response string

**Returns**: Optional[float] - Confidence score or None

**Strategy**: Looks for confidence values between 0 and 1, clamps to valid range.

### `_calculate_prediction_stats(df)`

**Purpose**: Calculate statistics on predictions.

**Parameters**:
- `df` (pd.DataFrame): DataFrame with prediction column

**Returns**: Dict[str, Any] - Statistics dictionary

**Statistics Include**:
- Total predictions count
- Unique prediction values
- Prediction value counts
- Mean confidence score

## Usage Examples

### Basic Usage

```python
from scripts.inference_runner import InferenceRunner

# Initialize runner
runner = InferenceRunner(debug_mode=True, batch_size=5)

# Check model availability
if runner.check_model_availability('llama2'):
    # Process dataset
    results_df = runner.process_dataset(df, 'llama2')
    
    # Save results
    runner.save_results(results_df, 'outputs/results.csv')
```

### Advanced Usage with Custom Configuration

```python
# Custom configuration
runner = InferenceRunner(
    debug_mode=True,
    batch_size=10,
    timeout=60,
    max_retries=5
)

# Get model metadata
metadata = runner.get_model_metadata('llama2')

# Process in batches with custom handling
prompts = df['formatted_prompt'].tolist()
results = runner.run_batch_inference(prompts, 'llama2')

# Create and save metadata
processing_time = 45.2
metadata = runner.create_inference_metadata('llama2', results_df, processing_time)
runner.save_metadata(metadata, 'outputs/metadata.json')
```

### Error Handling

```python
try:
    results_df = runner.process_dataset(df, 'nonexistent_model')
except ModelNotFoundError as e:
    print(f"Model not found: {e}")
except ResponseParsingError as e:
    print(f"Failed to parse response: {e}")
except InferenceError as e:
    print(f"Inference failed: {e}")
```

## Performance Characteristics

### Time Complexity
- **Model checking**: O(1) - Single command execution
- **Single inference**: O(1) - Single model call
- **Batch inference**: O(n) where n is number of prompts
- **Dataset processing**: O(n) where n is number of rows

### Space Complexity
- **Memory usage**: O(batch_size) for batch processing
- **Storage**: O(n) for results where n is number of rows

### Performance Considerations
- Batch size affects memory usage and processing speed
- Timeout settings impact reliability vs. speed trade-offs
- Retry logic increases reliability but may slow processing
- Model size and complexity affect inference speed

## Integration Points

### Input Dependencies
- Formatted prompts from `formatter_promptbuilder` module
- Model availability from Ollama installation
- Dataset structure with required columns

### Output Dependencies
- Results for evaluation and scoring
- Metadata for tracking and analysis
- Structured output for GUI consumption

### Error Handling Integration
- Logging integration with debugging logger
- Error propagation to calling modules
- Graceful degradation for partial failures 