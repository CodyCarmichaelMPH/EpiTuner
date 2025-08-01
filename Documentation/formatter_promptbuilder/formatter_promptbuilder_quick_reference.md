# Formatter PromptBuilder Quick Reference

## Quick Start

```python
from scripts.formatter_promptbuilder import PromptBuilder

# Initialize
formatter = PromptBuilder(debug_mode=True)

# Process dataset
formatted_df, metadata = formatter.process_dataset(
    df, "respiratory infections", rating_mapping, "outputs"
)
```

## Common Usage Patterns

### 1. Basic Prompt Formatting
```python
# Format single row
prompt = formatter.build_prompt(row, "cardiac conditions", rating_mapping)

# Format entire dataset
formatted_df = formatter.format_dataset(df, "neurological disorders", rating_mapping)
```

### 2. Custom Template
```python
custom_template = """
Patient: {age} year old {sex}
Complaint: {chief_complaint}
Diagnosis: {discharge_diagnosis}
Topics: {target_topics}
Schema: {schema_description}
"""
formatter.set_prompt_template(custom_template)
```

### 3. Save Specific Formats
```python
# Save as CSV
formatter.save_formatted_dataset(formatted_df, "my_prompts.csv")

# Save as JSONL
formatter.save_prompts_jsonl(formatted_df, "my_prompts.jsonl")

# Save metadata
formatter.save_metadata(metadata, "my_metadata.json")
```

## Input Parameters

### Required Fields
```python
REQUIRED_FIELDS = [
    'C_BioSense_ID',      # Patient identifier
    'ChiefComplaintOrig', # Original complaint
    'Discharge Diagnosis', # Final diagnosis
    'Sex',                # Patient sex
    'Age',                # Patient age
    'Admit_Reason_Combo', # Admission reason
    'Chief_Complaint_Combo', # Combined complaints
    'Diagnosis_Combo',    # Combined diagnoses
    'CCDD',               # Chief complaint discharge diagnosis
    'CCDDCategory',       # CCDD category
    'TriageNotes'         # Triage notes
]
```

### Rating Mapping
```python
rating_mapping = {
    'Match': 1,
    'Does Not Match': 0,
    'Unknown': -1,
    'Partial Match': 2
}
```

## Output Files

### 1. formatted_dataset.csv
- Original data + `Prompt` column + `Row_ID` column

### 2. prompts.jsonl
```json
{"C_BioSense_ID": "P001", "prompt": "...", "target_rating": 1, "original_rating": 1}
```

### 3. formatting_metadata.json
```json
{
  "formatting_info": {
    "total_rows": 10,
    "target_topics": "respiratory infections",
    "rating_mapping": {...}
  },
  "data_quality": {
    "rows_with_missing_fields": 0,
    "rows_with_na_values": 0
  }
}
```

## Error Handling

### Missing Fields
```python
# Automatically handled - uses "N/A" placeholder
# Logs warnings for missing required fields
```

### Invalid Data
```python
# None/NaN values → "N/A"
# Empty strings → "N/A"
# Whitespace-only → "N/A"
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

## Integration Examples

### With data_loader
```python
from scripts.data_loader import DataLoader
from scripts.formatter_promptbuilder import PromptBuilder

# Load and process data
loader = DataLoader()
df, unique_ratings, metadata = loader.process_dataset("data.csv")

# Format prompts
formatter = PromptBuilder()
formatted_df, format_metadata = formatter.process_dataset(
    df, "respiratory infections", rating_mapping
)
```

### With schema_mapper
```python
from scripts.schema_mapper import SchemaMapper
from scripts.formatter_promptbuilder import PromptBuilder

# Map ratings
mapper = SchemaMapper()
mapped_df, mapping_metadata = mapper.process_mapping(df, rating_mapping)

# Format prompts
formatter = PromptBuilder()
formatted_df, format_metadata = formatter.process_dataset(
    mapped_df, "cardiac conditions", rating_mapping
)
```

## Testing

### Run Tests
```bash
python -m pytest tests/test_formatter_promptbuilder.py -v
```

### Run Demo
```bash
python demo_formatter_promptbuilder.py
```

## Logging

### Debug Mode
```python
formatter = PromptBuilder(debug_mode=True)  # Verbose logging
formatter = PromptBuilder(debug_mode=False) # Standard logging
```

### Log Files
- `formatter_promptbuilder.log`: Processing logs

## Performance Tips

- Use `debug_mode=False` for production
- Process large datasets in batches if needed
- Monitor log files for warnings/errors
- Check metadata for data quality issues

## Common Issues

### Missing Required Fields
**Symptom**: Warnings in logs
**Solution**: Ensure all required fields are present in DataFrame

### Template Errors
**Symptom**: KeyError during formatting
**Solution**: Check template placeholders match available fields

### File Permission Errors
**Symptom**: FormattingError on save
**Solution**: Ensure write permissions to output directory 