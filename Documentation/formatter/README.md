# Formatter Module

## Overview

The Formatter module converts cleaned and mapped dataset rows into structured, context-rich prompts that LLMs can use for both fine-tuning and inference. It provides a flexible and efficient way to create standardized prompts from medical datasets.

## Purpose

- Convert dataset rows into structured prompts for LLM training
- Create inference prompts for model prediction
- Support mixed datasets with both training and inference data
- Provide customizable prompt templates
- Enable batch processing for large datasets
- Validate input data and generated prompts
- Support file operations for prompt persistence

## Key Features

### Training Prompt Creation
- Convert expert-rated data into few-shot examples
- Include expert ratings and rationales in prompts
- Support for various rating scales and formats
- Automatic handling of missing or invalid data

### Inference Prompt Creation
- Create prompts for model prediction without expert guidance
- Structured output format for model responses
- Support for various medical topics and conditions

### Mixed Processing
- Handle datasets containing both training and inference data
- Automatic separation and processing of different data types
- Consistent prompt formatting across all data types

### Custom Templates
- Configurable prompt templates for different use cases
- Support for custom training and inference templates
- Template validation and error handling

### Batch Processing
- Efficient processing of large datasets
- Memory-conscious operations
- Progress tracking and error handling

### Validation
- Comprehensive input data validation
- Prompt format validation
- Error reporting with actionable suggestions

## Installation

The formatter is included in the `scripts` directory and requires no additional installation.

```bash
# No additional installation required
# Module is available at: scripts/formatter.py
```

## Quick Start

### Basic Usage

```python
from scripts.formatter import Formatter

# Initialize formatter
formatter = Formatter(debug_mode=True)

# Create training prompts
training_prompts = formatter.create_training_prompts(
    df, 
    topic="Respiratory Issues", 
    include_rationale=True
)

# Create inference prompts
inference_prompts = formatter.create_inference_prompts(
    df, 
    topic="Respiratory Issues"
)
```

### Mixed Processing

```python
# Handle datasets with both training and inference data
mixed_prompts = formatter.create_mixed_prompts(
    df, 
    topic="Respiratory Issues",
    include_rationale=True
)

training_prompts = mixed_prompts['training']
inference_prompts = mixed_prompts['inference']
```

## API Reference

### Formatter Class

#### Constructor

```python
Formatter(debug_mode: bool = False)
```

**Parameters:**
- `debug_mode`: Enable verbose logging for debugging

#### Methods

##### Training Prompt Creation

```python
def create_training_prompts(
    self, 
    df: pd.DataFrame,
    topic: str,
    include_rationale: bool = True,
    custom_template: Optional[str] = None
) -> List[Dict[str, Any]]
```

Creates training prompts with expert ratings and rationales.

**Parameters:**
- `df`: DataFrame with training data (must include 'Standardized_Rating' column)
- `topic`: Target topic for alignment analysis
- `include_rationale`: Whether to include expert rationale in prompts
- `custom_template`: Custom prompt template (optional)

**Returns:**
- List of training prompt dictionaries

##### Inference Prompt Creation

```python
def create_inference_prompts(
    self, 
    df: pd.DataFrame,
    topic: str,
    custom_template: Optional[str] = None
) -> List[Dict[str, Any]]
```

Creates inference prompts for model prediction.

**Parameters:**
- `df`: DataFrame with inference data
- `topic`: Target topic for alignment analysis
- `custom_template`: Custom prompt template (optional)

**Returns:**
- List of inference prompt dictionaries

##### Mixed Prompt Creation

```python
def create_mixed_prompts(
    self, 
    df: pd.DataFrame,
    topic: str,
    include_rationale: bool = True,
    custom_templates: Optional[Dict[str, str]] = None
) -> Dict[str, List[Dict[str, Any]]]
```

Creates both training and inference prompts from the same dataset.

**Parameters:**
- `df`: DataFrame with mixed data
- `topic`: Target topic for alignment analysis
- `include_rationale`: Whether to include expert rationale in training prompts
- `custom_templates`: Custom templates for training and inference

**Returns:**
- Dictionary with 'training' and 'inference' prompt lists

##### Batch Processing

```python
def batch_process(
    self, 
    df: pd.DataFrame,
    topic: str,
    batch_size: int = 100,
    include_rationale: bool = True
) -> List[Dict[str, Any]]
```

Process dataset in batches for memory efficiency.

**Parameters:**
- `df`: Input DataFrame
- `topic`: Target topic
- `batch_size`: Number of rows to process per batch
- `include_rationale`: Whether to include rationale in training prompts

**Returns:**
- List of all prompts

##### File Operations

```python
def save_prompts_to_jsonl(
    self, 
    prompts: List[Dict[str, Any]], 
    output_path: str
) -> None
```

Save prompts to JSONL format file.

```python
def load_prompts_from_jsonl(
    self, 
    input_path: str
) -> List[Dict[str, Any]]
```

Load prompts from JSONL format file.

##### Validation

```python
def validate_input_data(
    self, 
    df: pd.DataFrame
) -> Tuple[bool, List[str]]
```

Validate input DataFrame for formatting.

```python
def validate_prompt_format(
    self, 
    prompt: Dict[str, Any]
) -> Tuple[bool, List[str]]
```

Validate a single prompt format.

##### Statistics

```python
def get_prompt_statistics(
    self, 
    prompts: List[Dict[str, Any]]
) -> Dict[str, Any]
```

Get statistics about the prompts.

##### Template Management

```python
def set_custom_templates(
    self, 
    training_template: Optional[str] = None,
    inference_template: Optional[str] = None
) -> None
```

Set custom prompt templates.

### Default Templates

#### Training Template

```python
def _get_default_training_template(self) -> str:
    return """INPUT:
You are evaluating whether this case aligns with topic '{topic}'.

Patient Data:
{context_block}

OUTPUT (Expert Provided):
Rating: {rating}
Rationale: {rationale}"""
```

#### Inference Template

```python
def _get_default_inference_template(self) -> str:
    return """INPUT:
You are evaluating whether this case aligns with topic '{topic}'.

Patient Data:
{context_block}

OUTPUT (Model Prediction):
Rating: <predicted integer>
Rationale: <model explanation>"""
```

## Usage Examples

### Basic Training Prompt Creation

```python
from scripts.formatter import Formatter
import pandas as pd

# Sample training data
training_data = pd.DataFrame({
    'C_BioSense_ID': ['P001', 'P002'],
    'Context_Block': [
        'Chief Complaint: Fever, cough\nDischarge Diagnosis: Viral pneumonia',
        'Chief Complaint: Chest pain\nDischarge Diagnosis: Angina'
    ],
    'Standardized_Rating': [1, 2],
    'Rationale of Rating': [
        'Clear respiratory infection',
        'Cardiac symptoms present'
    ]
})

# Create formatter
formatter = Formatter(debug_mode=True)

# Create training prompts
prompts = formatter.create_training_prompts(
    training_data, 
    topic="Respiratory Issues", 
    include_rationale=True
)

print(f"Created {len(prompts)} training prompts")
for prompt in prompts:
    print(f"ID: {prompt['id']}, Rating: {prompt['rating']}")
```

### Custom Template Usage

```python
# Custom training template
custom_template = """MEDICAL CASE ANALYSIS:
Topic: {topic}

Patient Information:
{context_block}

Expert Assessment:
Rating: {rating}
Reasoning: {rationale}

Please analyze this case for respiratory conditions."""

# Create prompts with custom template
prompts = formatter.create_training_prompts(
    training_data, 
    topic="Respiratory Issues",
    custom_template=custom_template
)
```

### Mixed Dataset Processing

```python
# Dataset with both training and inference data
mixed_data = pd.DataFrame({
    'C_BioSense_ID': ['P001', 'P002', 'P003', 'P004'],
    'Context_Block': ['Context 1', 'Context 2', 'Context 3', 'Context 4'],
    'Standardized_Rating': [1, 2, None, None],  # Some with ratings, some without
    'Rationale of Rating': ['Reason 1', 'Reason 2', None, None]
})

# Process mixed dataset
mixed_prompts = formatter.create_mixed_prompts(
    mixed_data, 
    topic="Respiratory Issues"
)

print(f"Training prompts: {len(mixed_prompts['training'])}")
print(f"Inference prompts: {len(mixed_prompts['inference'])}")
```

### Batch Processing

```python
# Large dataset
large_data = pd.DataFrame({
    'C_BioSense_ID': [f'P{i:03d}' for i in range(1, 1001)],  # 1000 rows
    'Context_Block': [f'Context for patient {i}' for i in range(1, 1001)],
    'Standardized_Rating': [i % 3 for i in range(1, 1001)],
    'Rationale of Rating': [f'Rationale for patient {i}' for i in range(1, 1001)]
})

# Process in batches
prompts = formatter.batch_process(
    large_data, 
    topic="Respiratory Issues",
    batch_size=100,
    include_rationale=True
)

print(f"Processed {len(prompts)} prompts in batches")
```

### File Operations

```python
# Save prompts to file
formatter.save_prompts_to_jsonl(prompts, "output.jsonl")

# Load prompts from file
loaded_prompts = formatter.load_prompts_from_jsonl("output.jsonl")

print(f"Loaded {len(loaded_prompts)} prompts from file")
```

### Validation

```python
# Validate input data
is_valid, errors = formatter.validate_input_data(training_data)
if not is_valid:
    print(f"Validation errors: {errors}")
else:
    print("Data validation passed")

# Validate individual prompts
for prompt in prompts:
    is_valid, errors = formatter.validate_prompt_format(prompt)
    if not is_valid:
        print(f"Prompt validation errors: {errors}")
```

### Statistics

```python
# Get prompt statistics
stats = formatter.get_prompt_statistics(prompts)

print(f"Total prompts: {stats['total_prompts']}")
print(f"Average length: {stats['avg_prompt_length']} characters")
print(f"Rating distribution: {stats['rating_distribution']}")
print(f"Topics: {stats['topics']}")
```

## Input Data Requirements

### Required Columns

For all operations:
- `C_BioSense_ID`: Unique identifier for each row
- `Context_Block`: Combined patient data for analysis

For training prompts:
- `Standardized_Rating`: Expert rating (integer)
- `Rationale of Rating`: Expert rationale (optional, if include_rationale=True)

### Data Format

```python
# Example input DataFrame
df = pd.DataFrame({
    'C_BioSense_ID': ['P001', 'P002'],
    'Context_Block': [
        'Chief Complaint: Fever, cough\nDischarge Diagnosis: Viral pneumonia\nTriage Notes: Persistent cough',
        'Chief Complaint: Chest pain\nDischarge Diagnosis: Angina\nTriage Notes: Severe chest pain'
    ],
    'Standardized_Rating': [1, 2],
    'Rationale of Rating': [
        'Clear respiratory infection matching the topic',
        'Cardiac symptoms present, partial match to respiratory issues'
    ]
})
```

## Output Format

### Training Prompt Structure

```python
{
    'id': 'P001',
    'prompt': 'INPUT:\nYou are evaluating whether this case aligns with topic \'Respiratory Issues\'.\n\nPatient Data:\nChief Complaint: Fever, cough\nDischarge Diagnosis: Viral pneumonia\nTriage Notes: Persistent cough\n\nOUTPUT (Expert Provided):\nRating: 1\nRationale: Clear respiratory infection matching the topic',
    'rating': 1,
    'rationale': 'Clear respiratory infection matching the topic',
    'topic': 'Respiratory Issues',
    'row_index': 0
}
```

### Inference Prompt Structure

```python
{
    'id': 'P001',
    'prompt': 'INPUT:\nYou are evaluating whether this case aligns with topic \'Respiratory Issues\'.\n\nPatient Data:\nChief Complaint: Fever, cough\nDischarge Diagnosis: Viral pneumonia\nTriage Notes: Persistent cough\n\nOUTPUT (Model Prediction):\nRating: <predicted integer>\nRationale: <model explanation>',
    'topic': 'Respiratory Issues',
    'row_index': 0
}
```

## Template Customization

### Template Placeholders

- `{topic}`: The analysis topic
- `{context_block}`: Patient data context
- `{rating}`: Expert rating (training only)
- `{rationale}`: Expert rationale (training only)

### Custom Template Examples

#### Medical Analysis Template

```python
custom_template = """MEDICAL CASE EVALUATION:
Topic: {topic}

Patient Information:
{context_block}

Expert Evaluation:
Rating: {rating}
Clinical Reasoning: {rationale}

Please evaluate this case for the specified condition."""
```

#### Research Template

```python
custom_template = """RESEARCH CASE ANALYSIS:
Research Topic: {topic}

Case Data:
{context_block}

Expert Classification:
Score: {rating}
Justification: {rationale}

Analyze this case for research purposes."""
```

#### Simple Template

```python
custom_template = """Topic: {topic}
Data: {context_block}
Rating: {rating}
Reason: {rationale}"""
```

## Error Handling

### Common Errors

#### ValidationError

Raised when input validation fails.

```python
try:
    prompts = formatter.create_training_prompts(invalid_df, "Topic")
except ValidationError as e:
    print(f"Validation error: {e}")
```

#### FormatterError

Raised when formatting operations fail.

```python
try:
    formatter.save_prompts_to_jsonl(prompts, "/invalid/path/file.jsonl")
except FormatterError as e:
    print(f"Formatter error: {e}")
```

### Error Recovery

```python
# Handle missing ratings gracefully
def safe_create_prompts(df, topic):
    try:
        return formatter.create_training_prompts(df, topic)
    except ValidationError as e:
        print(f"Validation failed: {e}")
        # Try inference-only mode
        return formatter.create_inference_prompts(df, topic)
```

## Performance Considerations

### Memory Usage

- Use batch processing for large datasets
- Clear prompt lists when no longer needed
- Monitor memory usage during processing

### Processing Speed

- Batch processing improves performance for large datasets
- Custom templates have minimal performance impact
- File I/O operations are optimized for JSONL format

### Optimization Tips

```python
# Process large datasets in batches
prompts = formatter.batch_process(large_df, topic, batch_size=50)

# Use custom templates for efficiency
formatter.set_custom_templates(training_template=my_template)

# Validate data before processing
is_valid, errors = formatter.validate_input_data(df)
if not is_valid:
    # Fix validation issues before processing
    pass
```

## Integration with Other Modules

### With Data Loader

```python
from scripts.data_loader import DataLoader
from scripts.formatter import Formatter

# Load and process data
data_loader = DataLoader(debug_mode=True)
df, unique_ratings, metadata = data_loader.process_dataset("data.csv")

# Create prompts
formatter = Formatter(debug_mode=True)
prompts = formatter.create_training_prompts(df, "Respiratory Issues")
```

### With Schema Mapper

```python
from scripts.schema_mapper import SchemaMapper
from scripts.formatter import Formatter

# Map ratings
schema_mapper = SchemaMapper(debug_mode=True)
rating_mapping = {1: 1, 2: 2, 0: 0}
mapped_df = schema_mapper.apply_rating_mapping(df, rating_mapping)

# Create prompts
formatter = Formatter(debug_mode=True)
prompts = formatter.create_training_prompts(mapped_df, "Respiratory Issues")
```

### With Fine Tuner

```python
from scripts.formatter import Formatter
from scripts.fine_tuner import FineTuner

# Create prompts
formatter = Formatter(debug_mode=True)
prompts = formatter.create_training_prompts(df, "Respiratory Issues")

# Fine-tune model
fine_tuner = FineTuner(debug_mode=True)
fine_tuner.fine_tune_model(prompts, "llama3.2:3b")
```

### With Inference Runner

```python
from scripts.formatter import Formatter
from scripts.inference_runner import InferenceRunner

# Create inference prompts
formatter = Formatter(debug_mode=True)
prompts = formatter.create_inference_prompts(df, "Respiratory Issues")

# Run inference
inference_runner = InferenceRunner(debug_mode=True)
results = inference_runner.run_batch_inference(prompts, "llama3.2:3b")
```

## Best Practices

### 1. Data Preparation

```python
# Ensure data is properly cleaned before formatting
df = df.dropna(subset=['C_BioSense_ID', 'Context_Block'])

# Validate data before processing
is_valid, errors = formatter.validate_input_data(df)
if not is_valid:
    print(f"Fix validation errors: {errors}")
    return
```

### 2. Template Design

```python
# Design clear, consistent templates
template = """CLEAR TEMPLATE:
Topic: {topic}
Data: {context_block}
Rating: {rating}
Reason: {rationale}"""

# Test templates with sample data
test_prompts = formatter.create_training_prompts(
    sample_df, "Test Topic", custom_template=template
)
```

### 3. Error Handling

```python
# Always handle potential errors
try:
    prompts = formatter.create_training_prompts(df, topic)
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
    # Handle validation error
except FormatterError as e:
    logger.error(f"Formatting failed: {e}")
    # Handle formatting error
```

### 4. Performance Optimization

```python
# Use batch processing for large datasets
if len(df) > 1000:
    prompts = formatter.batch_process(df, topic, batch_size=100)
else:
    prompts = formatter.create_training_prompts(df, topic)

# Save prompts for reuse
formatter.save_prompts_to_jsonl(prompts, "cached_prompts.jsonl")
```

### 5. Quality Assurance

```python
# Validate generated prompts
for prompt in prompts:
    is_valid, errors = formatter.validate_prompt_format(prompt)
    if not is_valid:
        print(f"Invalid prompt: {errors}")

# Check prompt statistics
stats = formatter.get_prompt_statistics(prompts)
print(f"Prompt quality metrics: {stats}")
```

## Troubleshooting

### Common Issues

#### Missing Required Columns

**Problem**: ValidationError for missing columns
**Solution**: Ensure DataFrame has required columns

```python
required_columns = ['C_BioSense_ID', 'Context_Block']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Missing columns: {missing_columns}")
```

#### Empty Context Blocks

**Problem**: Empty or None context blocks
**Solution**: Clean data before formatting

```python
df = df.dropna(subset=['Context_Block'])
df = df[df['Context_Block'].str.strip() != '']
```

#### Invalid Ratings

**Problem**: Non-integer ratings in training data
**Solution**: Convert ratings to integers

```python
df['Standardized_Rating'] = pd.to_numeric(df['Standardized_Rating'], errors='coerce')
df = df.dropna(subset=['Standardized_Rating'])
```

#### Template Errors

**Problem**: Template formatting errors
**Solution**: Check template placeholders

```python
# Ensure all placeholders are present
required_placeholders = ['{topic}', '{context_block}']
for placeholder in required_placeholders:
    if placeholder not in template:
        print(f"Missing placeholder: {placeholder}")
```

### Performance Issues

#### Memory Usage

**Problem**: High memory usage with large datasets
**Solution**: Use batch processing

```python
prompts = formatter.batch_process(df, topic, batch_size=50)
```

#### Slow Processing

**Problem**: Slow prompt generation
**Solution**: Optimize data preparation

```python
# Pre-filter data
df = df[df['Context_Block'].notna()]
df = df[df['Context_Block'].str.len() > 0]
```

## Testing

### Running Tests

```bash
# Run formatter tests
pytest tests/test_formatter.py -v

# Run with coverage
pytest tests/test_formatter.py --cov=scripts.formatter
```

### Test Coverage

The test suite covers:
- Training prompt creation
- Inference prompt creation
- Mixed processing
- Custom templates
- Batch processing
- File operations
- Validation
- Edge cases

### Demo Script

```bash
# Run the formatter demo
python demo_formatter.py
```

## Related Modules

- **Data Loader**: Provides cleaned data for formatting
- **Schema Mapper**: Maps ratings before formatting
- **Fine Tuner**: Uses formatted prompts for training
- **Inference Runner**: Uses formatted prompts for inference
- **Debugging Logger**: Provides logging for formatter operations

## Version History

- **v1.0.0**: Initial implementation with basic prompt creation
- **v1.1.0**: Added custom templates and validation
- **v1.2.0**: Added batch processing and file operations
- **v1.3.0**: Enhanced error handling and statistics 