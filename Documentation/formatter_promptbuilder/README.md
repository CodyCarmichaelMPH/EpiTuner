# Formatter PromptBuilder Module

## Overview

The Formatter PromptBuilder module transforms structured dataset rows into optimized, consistent prompts for LLM inference. It ensures that each row of data is presented with the necessary context, formatted predictably for reliable inference.

## Purpose

This module serves as a bridge between the cleaned dataset (from `data_loader` and `schema_mapper`) and the LLM inference system. It creates well-structured prompts that include:

- Patient context information
- Medical complaint and diagnosis data  
- Task instructions with rating schema
- Expected response format

## Key Features

- **Data Validation**: Ensures each row has required context fields
- **Text Preprocessing**: Cleans and normalizes text fields
- **Prompt Construction**: Combines context into structured prompts
- **Template Customization**: Supports custom prompt templates
- **Multiple Output Formats**: CSV, JSONL, and metadata files
- **Error Handling**: Graceful handling of missing or invalid data

## Architecture

```
Input: Cleaned DataFrame + Rating Mapping + Target Topics
       ↓
   [Data Validation]
       ↓
   [Text Preprocessing] 
       ↓
   [Prompt Construction]
       ↓
   [Output Generation]
       ↓
Output: Formatted Prompts (CSV/JSONL) + Metadata
```

## Usage

### Basic Usage

```python
from scripts.formatter_promptbuilder import PromptBuilder

# Initialize formatter
formatter = PromptBuilder(debug_mode=True)

# Define parameters
target_topics = "respiratory infections, cardiac conditions"
rating_mapping = {
    'Match': 1,
    'Does Not Match': 0, 
    'Unknown': -1
}

# Process dataset
formatted_df, metadata = formatter.process_dataset(
    df, target_topics, rating_mapping, output_dir="outputs"
)
```

### Custom Prompt Template

```python
# Set custom template
custom_template = """
Patient: {age} year old {sex}
Complaint: {chief_complaint}
Diagnosis: {discharge_diagnosis}
Topics: {target_topics}
Schema: {schema_description}
"""

formatter.set_prompt_template(custom_template)
```

## Input Requirements

### Required Fields
- `C_BioSense_ID`: Patient identifier
- `ChiefComplaintOrig`: Original chief complaint
- `Discharge Diagnosis`: Final diagnosis
- `Sex`: Patient sex
- `Age`: Patient age
- `Admit_Reason_Combo`: Admission reason
- `Chief_Complaint_Combo`: Combined complaints
- `Diagnosis_Combo`: Combined diagnoses
- `CCDD`: Chief complaint discharge diagnosis
- `CCDDCategory`: CCDD category
- `TriageNotes`: Triage notes

### Optional Fields
- `Expert Rating`: Original expert rating
- `Standardized_Rating`: Mapped rating value
- `Rationale of Rating`: Expert rationale

## Output Formats

### 1. Formatted Dataset (CSV)
Contains original data plus:
- `Prompt`: Complete formatted prompt
- `Row_ID`: Row identifier

### 2. Prompts JSONL
JSON Lines format for LLM training/inference:
```json
{
  "C_BioSense_ID": "P001",
  "prompt": "Context:\n- Patient Info: Age 25, Sex M\n...",
  "target_rating": 1,
  "original_rating": 1
}
```

### 3. Metadata (JSON)
Processing information and data quality metrics:
```json
{
  "formatting_info": {
    "total_rows": 10,
    "target_topics": "respiratory infections",
    "rating_mapping": {...},
    "prompt_template_used": "..."
  },
  "data_quality": {
    "rows_with_missing_fields": 0,
    "rows_with_na_values": 0
  }
}
```

## Default Prompt Template

```
Context:
- Patient Info: Age {age}, Sex {sex}
- Chief Complaint: {chief_complaint}
- Discharge Diagnosis: {discharge_diagnosis}
- Admit Reason: {admit_reason}
- Combined Complaints: {chief_complaint_combo}
- Diagnosis Combo: {diagnosis_combo}
- CCDD: {ccdd}, Category: {ccdd_category}
- Triage Notes: {triage_notes}

Task:
Based on the above information, evaluate whether this record aligns with the topic(s): {target_topics}.
Use the following rating schema:
{schema_description}

Respond with:
- Numeric rating (from schema).
- Brief rationale (1–3 sentences).
```

## Error Handling

### Missing Fields
- Logs warnings for missing required fields
- Continues processing with available data
- Uses "N/A" placeholder for missing values

### Empty Rows
- Skips completely empty rows
- Logs errors for debugging

### Data Type Issues
- Handles None/NaN values gracefully
- Converts all text to strings
- Strips whitespace and normalizes

## Integration with Other Modules

### Dependencies
- **data_loader**: Provides cleaned dataset
- **schema_mapper**: Provides rating mapping

### Dependents
- **inference_runner**: Uses formatted prompts for LLM calls
- **fine_tuner**: Uses prompts for model training
- **contextualizer**: Uses prompts for few-shot learning

## Testing

Run the test suite:
```bash
python -m pytest tests/test_formatter_promptbuilder.py -v
```

### Test Coverage
- Initialization and configuration
- Data validation
- Text preprocessing
- Prompt building
- File I/O operations
- Error handling
- Custom templates

## Demo

Run the demo script:
```bash
python demo_formatter_promptbuilder.py
```

This will:
1. Load the processed dataset
2. Format prompts with sample topics
3. Display sample outputs
4. Save results to `outputs/` directory

## Configuration

### Debug Mode
Enable verbose logging:
```python
formatter = PromptBuilder(debug_mode=True)
```

### Log Files
- `formatter_promptbuilder.log`: Processing logs
- `formatting_metadata.json`: Processing metadata

## Performance Considerations

- Processes datasets row by row for memory efficiency
- Supports both small (5-10 rows) and large (500+ rows) datasets
- Deterministic output for same input data
- No random rewording or variations

## Future Enhancements

- Support for multiple prompt templates
- Batch processing optimization
- Integration with prompt versioning
- Support for different output formats
- Template validation and testing tools 